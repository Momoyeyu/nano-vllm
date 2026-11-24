import torch
from torch import nn


class Sampler(nn.Module):
    """根据 logits 和 temperature 为每个序列采样下一个 token。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """执行 temperature 调整和 categorical sampling。

        用 argmax(prob / Exp(1)) 实现采样，避免显式构造 multinomial 分布。
        """
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
