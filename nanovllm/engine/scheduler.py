from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # 等待首次或继续 prefill 的序列
        self.running: deque[Sequence] = deque()  # 已完成 prefill、等待逐 token decode 的序列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选择本轮交给模型的序列，并标记执行阶段。

        调度器优先处理 waiting 队列中的 prefill，再处理 running 队列中的 decode；
        prefill 受序列数和 token 预算限制，长请求可以通过 chunked prefill 分段推进。
        decode 每个序列每轮推进一个 token，KV Cache 不足时通过抢占其他序列释放物理块。
        返回 (本轮序列列表, is_prefill)，供 ModelRunner 选择对应的前向路径。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:  # 依次尝试把等待请求装入本轮 prefill 批次
            seq = self.waiting[0]  # 始终先看队首，保持 FIFO，并优先完成被拆分的请求
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:  # 本轮 prefill 的 token 预算已用完
                break
            if not seq.block_table:  # 新请求或被抢占的请求还没有块表，先检查前缀缓存和 KV Cache 容量
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:  # 空闲 KV Cache 块不足，队首请求暂时无法调度
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # 只有本轮首个请求允许 chunked prefill，后续请求放不下就等下轮
                break
            if not seq.block_table:  # 正式分配物理块，并复用已经命中的前缀缓存块
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)  # 装得下就全调度，否则只调度剩余预算对应的一段
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:  # 本轮覆盖全部待 prefill token，可移入 running 队列
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:  # 本轮有 prefill 就优先执行，不再混入 decode
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:  # 没有 prefill 时，尽量让所有运行序列各生成一个 token
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):  # 新 token 需要新块但显存不足时，持续抢占直到腾出空间
                if self.running:  # 还有其他运行请求时，抢占队尾请求来释放 KV Cache 块
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:  # while 正常结束表示空间已足够，当前序列可以参加本轮 decode
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs  # 简化实现假定抢占后本轮仍有可调度的 decode 序列
        self.running.extendleft(reversed(scheduled_seqs))  # 恢复刚才从队首弹出后的原有顺序
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True  # 块释放后需要重新走 prefill 调度，但仍可能命中前缀缓存
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)  # 被抢占的请求放到等待队列最前面，之后优先恢复

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):  # 每个调度序列对应采样器返回的一个 token
            self.block_manager.hash_blocks(seq)  # 将本轮新算完整的 KV 块登记到前缀缓存索引
            seq.num_cached_tokens += seq.num_scheduled_tokens  # 标记本轮 token 的 KV Cache 已计算完成
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue  # chunked prefill 尚未到 prompt 末尾，本轮采样结果没有意义，不追加
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 请求结束后立即归还物理块供其他请求使用
                self.running.remove(seq)
