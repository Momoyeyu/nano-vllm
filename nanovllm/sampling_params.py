from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0  # 控制采样随机性，越小越偏向高概率 token
    max_tokens: int = 64      # 每个请求最多生成的 token 数
    ignore_eos: bool = False  # True 时遇到 EOS 也继续生成，常用于压测

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
