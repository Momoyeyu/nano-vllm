import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str                                  # 本地模型目录，而不是 Hugging Face 模型名
    max_num_batched_tokens: int = 16384         # 单次 prefill 最多调度的 token 总数
    max_num_seqs: int = 512                     # 单次模型前向最多容纳的序列数
    max_model_len: int = 4096                   # prompt 与生成 token 的总长度上限
    gpu_memory_utilization: float = 0.9         # 模型加载后允许使用的显存比例
    tensor_parallel_size: int = 1               # 使用多少张 GPU 做张量并行
    enforce_eager: bool = False                 # True 时禁用 CUDA Graph，便于调试
    hf_config: AutoConfig | None = None         # 从模型目录读取的 Hugging Face 配置
    eos: int = -1                               # tokenizer 初始化后填入真正的结束 token ID
    kvcache_block_size: int = 256               # 每个 KV Cache 物理块容纳的 token 数
    num_kvcache_blocks: int = -1                # 模型预热后根据剩余显存自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)        # 当前加载器只支持已下载到本地的模型
        assert self.kvcache_block_size % 256 == 0  # KV Cache 块大小必须按 256 对齐
        assert 1 <= self.tensor_parallel_size <= 8  # 当前实现最多支持 8 路张量并行
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 读取层数、头数、词表大小等模型结构
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # 不能超过模型支持的位置范围
