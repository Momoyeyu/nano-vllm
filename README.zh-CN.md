[English](README.md) | **简体中文**

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

从零开始实现的轻量级 vLLM。

## 核心特性

* 🚀 **极速离线推理** - 推理速度可与 vLLM 媲美
* 📖 **代码清晰易读** - 仅用约 1,200 行 Python 代码实现
* ⚡ **优化套件** - 前缀缓存（Prefix Caching）、张量并行（Tensor Parallelism）、Torch 编译、CUDA Graph 等

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 模型下载

如需手动下载模型权重，请使用以下命令（需要 `huggingface_hub` 1.x，该版本提供 `hf` 命令）：
```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```
下载默认支持断点续传。如果你使用的是 `huggingface_hub` 0.x，请改用旧版命令：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

用法参见 `example.py`。API 与 vLLM 的接口基本一致，仅在 `LLM.generate` 方法上略有差异：
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 基准测试

基准测试参见 `bench.py`。

**测试配置：**
- 硬件：RTX 4070 Laptop（8GB 显存）
- 模型：Qwen3-0.6B
- 请求总数：256 个序列
- 输入长度：在 100–1024 个 token 之间随机采样
- 输出长度：在 100–1024 个 token 之间随机采样

**性能结果：**
| 推理引擎 | 输出 Token 数 | 耗时（秒） | 吞吐量（tokens/s） |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
