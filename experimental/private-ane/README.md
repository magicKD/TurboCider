# Private ANE 实验说明

更新时间：2026-09-08

## 定位

Apple private ANE API 只用于研究，不属于 TurboCider 正式后端。实验实现来自工作区 `gpu_ane/ANE` 与 H3 研究快照；当前固定的外部证据版本为：

- `gpu_ane/ANE`：`d91c9845c0784dec7753048954fc6d0e8411fe29`
- `gpu_ane/mac_transformer`：`9f322e10e0165e59d498e6a0daf17ad750a9bb12`

TurboCider 仓库内的 private bridge 快照位于 `experimental/video/h3/vendor/h3_ane_bridge.m`，相关 MLP/QKV 原型也只保留在同一 vendor 目录。正式 `native/` 已删除调用 `_ANE*` 类的实现文件，并由 `native/models/h3_runtime/h3_ane_disabled.c` 提供不可用 stub。

## 为什么不进入正式软件

private 路径依赖未公开的类、selector、MIL 接口和 IOSurface 绑定约定，存在以下不可接受的发行风险：

1. macOS 更新可直接改变类名、selector、编译器行为或错误码；
2. 不同芯片代际对 MIL 版本、blob 布局和 channel shape 的接受范围不同；
3. public Core ML 的模型校验、compute-unit 选择和兼容承诺不适用于 private runtime；
4. private cache、program capacity 和 load/unload 行为没有稳定 ABI；
5. 局部 dispatch 更快不代表串行 Transformer 的完整端到端更快。

因此 private API 不会由正式 request、profile、App 开关或环境变量启用，也不会进入 `libturbocider.dylib`。正式可发行加速只使用 public Core ML 和 Metal/MLX。

## 已有实测结论

Apple M4 Max 64 GB 的研究结果显示：

| 项目 | public Core ML | private ANE | 结论 |
|---|---:|---:|---|
| `1024x512` ANE→Metal handoff | 0.317 ms | 0.253 ms | private 约 1.255× |
| `2048x512` ANE→Metal handoff | 0.424 ms | 0.303 ms | private 约 1.398× |
| SmolLM2 `up_proj [2048,576,1536]` p50 | 0.796 ms | 0.611 ms | private 约 1.302× |
| H3 matched BF16、全 block MLP+QKV E2E | 319.34 s GPU 对照 | 306.85 s | 只有约 1.041× |
| H3 sequence-row split | block micro 约 1.282× | all-50 warm forward 约 1.009× | 不采用 |

private runtime 确实能减少 wrapper、同步和部分 handoff 成本，但没有改变 attention→MLP 串行依赖、program rotation、共享内存竞争和 VAE 等 Amdahl 上限。它是研究证据，不是绕过 public Core ML 生命周期问题的产品捷径。

## 显式研究入口

只有 `tools/experimental/build_h3.sh` 会编译 `experimental/video/h3/vendor`。它输出到实验 build 目录，不会被 `make build` 或 `make package` 调用。运行者必须自行承担系统兼容、稳定性和不可发行风险。

产品边界回归由 `tests/repository/test_layout.py` 检查；正式构建后还应确认：

```sh
strings build/native/libturbocider.dylib | grep -E '_ANE(InMemoryModel|Request|IOSurfaceObject)'
```

正确结果应为空。
