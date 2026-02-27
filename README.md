# simple-vllm

## 项目简介

simple-vllm 是一个简化的[vLLM](https://github.com/vllm-project/vllm)实现，主要用于学习和理解大语言模型推理的核心原理。本项目基于官方 [vLLM](https://github.com/vllm-project/vllm) 代码进行了简化和修改，去除了部分复杂的实现细节，保留了核心功能，便于学习者理解和研究。

## 项目特点

- **依赖和CUDA代码简化**：移除了第三方库和自定义 CUDA 内核代码的依赖，使用 PyTorch 内置功能替代，降低了项目复杂度