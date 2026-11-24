from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """在请求队列、token 预算和 KV Cache 容量之间选择本轮工作。"""

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
        """选择本轮要执行的序列，并返回执行阶段。

        调度遵循以下原则：
        1. waiting 队列中的 prefill 优先于 running 队列中的 decode。
        2. prefill 同时受最大序列数和 token 预算限制；必要时只对本轮首个请求做
           chunked prefill，保证长 prompt 也能持续推进。
        3. 没有 prefill 时，每个 running 序列每轮 decode 一个 token；若 KV Cache
           不足，则先抢占其他序列释放物理块。

        Returns:
            tuple[list[Sequence], bool]: 本轮序列列表和是否为 prefill 阶段。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # Prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:  # 新请求或被抢占请求先检查缓存和容量
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:  # 空闲 KV Cache 块不足，队首请求暂时无法调度
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # 只有本轮首个请求允许 chunked prefill
                break
            if not seq.block_table:  # 正式建立块表并复用命中的前缀缓存
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):  # 新 token 需要新块但显存不足
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:  # while 正常结束表示空间已足够，当前序列可以参加本轮 decode
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))  # 恢复从队首弹出前的顺序
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """暂停一个序列并释放其 KV Cache，让其他序列获得显存。

        Args:
            seq: 要暂停的运行序列；其 token 历史会继续保留。

        Notes:
            重新调度时会走 prefill，但完整前缀仍可能命中缓存。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """消费采样结果并推进序列状态。

        Args:
            seqs: 本轮执行过的序列。
            token_ids: 与 seqs 一一对应的采样结果。
            is_prefill: 本轮是否执行 prefill。

        Notes:
            对未完成的 chunked prefill 只更新 KV Cache 进度，不追加中间位置的采样 token；
            prefill 完成或 decode 得到 token 后，再检查 EOS 和 max_tokens。
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
