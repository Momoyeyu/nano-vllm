from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False  # 当前批次是可分段的多 token prefill，还是单 token decode
    cu_seqlens_q: torch.Tensor | None = None  # varlen Attention 中各序列 Query 的累计边界
    cu_seqlens_k: torch.Tensor | None = None  # 各序列 Key/Value 可见范围的累计边界
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None  # 每个输入 token 应写入 KV Cache 的扁平物理槽位
    context_lens: torch.Tensor | None = None  # decode 时每个序列当前可见的 KV 长度
    block_tables: torch.Tensor | None = None  # 每个序列的逻辑块到物理 KV Cache 块映射

_CONTEXT = Context()

def get_context():
    return _CONTEXT  # Attention 层从这里读取 ModelRunner 准备好的批次元数据

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)  # 每轮用新对象替换，避免逐字段更新留下旧值

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()  # 模型执行结束后清空，防止下一轮意外复用旧批次元数据
