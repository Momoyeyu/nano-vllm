import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)  # 合并投影的前一半是 gate，后一半是 up
        return F.silu(x) * y  # Qwen 的 SwiGLU：silu(gate) * up
