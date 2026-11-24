from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """在一次模型前向期间向各 Attention 层传递的批次元数据。"""

    is_prefill: bool = False                  # 可分段的多 token prefill，或单 token decode
    cu_seqlens_q: torch.Tensor | None = None  # varlen Attention 的 Query 累计边界
    cu_seqlens_k: torch.Tensor | None = None  # Key/Value 可见范围的累计边界
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None  # token 到 KV Cache 扁平物理槽位的映射
    context_lens: torch.Tensor | None = None  # decode 时每个序列可见的 KV 长度
    block_tables: torch.Tensor | None = None  # 逻辑块到物理 KV Cache 块的映射

_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """替换当前前向步骤使用的 Attention 元数据。"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """清空当前步骤的元数据，避免下一批次读取旧状态。"""
    global _CONTEXT
    _CONTEXT = Context()
