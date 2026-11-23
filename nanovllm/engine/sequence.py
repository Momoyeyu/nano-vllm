from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()  # 等待 prefill，或 decode 时因显存不足被抢占
    RUNNING = auto()  # 已完成 prefill，可以逐 token decode
    FINISHED = auto()  # 已生成 EOS 或达到 max_tokens


class Sequence:
    block_size = 256  # 由 Config 在引擎启动时统一覆盖
    counter = count()  # 为每个请求分配单调递增的 ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)  # 同时保存 prompt 和之后生成的 token
        self.last_token = token_ids[-1]  # decode 时每轮只把最后一个 token 送入模型
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0  # 已有可用 KV Cache 的 token 数，包含复用和本轮计算完成的部分
        self.num_scheduled_tokens = 0  # 当前这一轮准备交给模型计算的 token 数
        self.is_prefill = True  # 新请求从 prefill 开始，抢占后也会重新 prefill
        self.block_table = []  # 逻辑 token 块到物理 KV Cache 块 ID 的映射
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size  # 向上取整得到所需逻辑块数

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]  # 取出第 i 个逻辑块包含的 token

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids  # 跨进程序列化时，prefill 需要完整 token，decode 只需最后 token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)  # 只发送其他 GPU 进程恢复调度状态所需的数据

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):  # prefill 进程收到完整 token 列表
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:  # decode 进程只需要最后一个输入 token，不必复制整个历史
            self.token_ids = []
            self.last_token = last_state
