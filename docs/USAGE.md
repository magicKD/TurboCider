# TurboCider Native

C++/Objective-C++ 推理库、SwiftUI App、C/Swift SDK、CLI 与 Unix socket 服务。FLUX.2-klein-4B 支持文生图、图生图、1–8 张参考图编辑；H3/LTX 为不可执行的接入契约。运行无 Python 子进程，无模型自动下载。

## 构建与发行目录

```sh
export MLX_ROOT=/path/to/site-packages/mlx
tools/native/build.sh
tools/native/package.sh
```

固定 MLX C++ 0.32.0，完整 Xcode，Apple Silicon arm64。通过 `DEVELOPER_DIR`/`SDKROOT` 选已有编译器。部署目标 macOS 15.0；当前真实测试系统 macOS 26.6，其他系统尚待认证。构建不下载依赖。输出 `dist/TurboCider.app` 和 `dist/cli/`，包含 MLX dylib/metallib；本机 ad-hoc 签名不等于 Developer ID 公证。模型保持在用户选择的原目录。

## 请求与 CLI

```sh
build/native/turbocider doctor
build/native/turbocider models
build/native/turbocider plan request.json
build/native/turbocider generate /path/to/FLUX.2-klein-4B request.json
build/native/turbocider batch /path/to/FLUX.2-klein-4B first.json second.json
```

推荐 schema 2（schema 1 仍兼容）：

```json
{
  "schema_version": 2,
  "model": "flux2-klein-4b",
  "operation": "image.edit",
  "inputs": [
    {"kind": "text", "role": "prompt", "text": "把这只狐狸放在雪地里，保持毛色与外观"},
    {"kind": "image", "role": "reference", "path": "/absolute/reference.png"}
  ],
  "outputs": [{"kind": "image", "path": "/absolute/result.png", "width": 512, "height": 512}],
  "sampling": {"steps": 4, "seed": 42},
  "execution": {"policy": "gpu", "residency": "resident"},
  "parameters": {"dynamic_text": true}
}
```

文生图使用 `image.generate`，仅保留 prompt 输入。图生图使用 `image.transform`，一张图片角色 `init_image`，可附 `strength: 0.5`。strength 表示保留原图程度；正值从 `max(1, floor(steps*strength))` 的采样阶段开始，1 不执行 DiT，0 执行全部步骤。编辑参考图的 strength 不用于加噪控制；顺序影响 reference 位置编码。

stdout 输出最终结果 JSON，stderr 输出事件 JSON。Ctrl-C 在安全边界取消，返回码 2；导出开始前可取消，文件原子提交后返回成功。使用唯一输出路径以保留历史。`batch` 复用同一会话，要求模型一致。动态文本最大 512 tokens；尺寸 64–2048 且 16 倍数，实际受内存预算限制；种子 0–2147483647，步数 1–50。

## 设备配置与编译缓存

默认 GPU，`auto` 也选择 GPU。请求 `execution.profile` 可指向本地 JSON。`profiles/apple-m4-pro-48gb.example.json` 默认关闭，复制后填写匹配的设备身份、artifact 路径并显式启用。配置覆盖请求的 policy/residency，计划含配置内容 hash。可控制 allocator cache、预算和 Core ML warmup 次数。

混合 `gpu_ane` 必须 `allow_approximation=true`，使用本地 schema 2 `ane_manifest`。当前支持20个 single block MLP、K=N=3072、单固定桶。量化 MLP 改变算法精度，结果明确标注；公开 `cpuAndNeuralEngine` 不保证子图全部实际驻留 ANE。旧 artifact provenance 只有源路径/大小，故仍为实验。超过 bucket 明确失败，不裁剪输入、不静默改 GPU。

```sh
build/native/turbocider compile-coreml /path/to/block.mlpackage /path/to/cache
```

按源文件 SHA、OS、GPU/架构身份缓存编译结果，带互斥和原子提交。仅编译已有 Core ML 模型，不自动转换/切分任意网络。驻留模式速度优先；`component_staged` 释放阶段权重，FLUX 不支持 block streamed offload。

## 常驻后台服务

```sh
build/native/turbocider serve /tmp/turbocider.sock /absolute/job-state
build/native/turbocider rpc /tmp/turbocider.sock rpc.json
```

RPC 为每连接一个换行终止 JSON，响应 `{ "ok": true, "result": ... }` 或错误。客户端可直接用 Unix socket。支持：

| action | 其他字段 |
|---|---|
| submit | `model_path` 与完整 `request` 对象，返回 job id |
| status / cancel | `id` |
| jobs | 可选 `offset`、`limit`（默认20，最大100） |
| plan | `request` |
| models / doctor | 无 |

单 worker、最多32排队、持久化状态、取消、分页、模型/conditioning 复用。重启将中断任务标记 interrupted，不自动重做。socket 限当前用户；崩溃后可回收同用户 stale socket。App 当前使用嵌入模式，共享队列需要通过服务 RPC。跨进程 GPU 锁使独立嵌入会话冲突返回 busy，不会隐式等待。

## SDK 与 App

`bindings/c/include/turbocider/turbocider.h` 提供 ABI 1。`tc_engine_create_model` 选择注册模块，`tc_engine_generate` 同步执行；返回字符串必须 `tc_string_free`。回调同步发生在生成线程，不能阻塞或重入生成；Engine 只能在所有调用完成后释放。`cancel` 可跨线程调用。

`bindings/swift/TurboCiderNative.swift` 提供 async generate、cancel、plan、system/models，主线程不执行推理。取消使用 `engine.cancel()`；Swift Task.cancel 尚未自动映射。App 的 JobStore 共用此接口，持久化历史并恢复 interrupted 状态。App 支持素材选择、图像模式、参数/配置、生成/取消、预览及历史参数复用。

## 开发验收

```sh
python3 tests/native/test_contract.py
build/native/turbocider self-test
build/native/turbocider-lifecycle-test /path/to/model /tmp/new-lifecycle-directory
python3 tests/native/test_service.py --model /path/to/model --output /tmp/new-service-directory
```

真实推理需要 Metal 权限。生命周期目录应为空。oracle 工具需要已有 mflux/MLX/transformers 的开发 Python 环境并强制离线：

```sh
/path/to/dev-python tools/native/flux_reference.py --model MODEL --request REQUEST --output REFERENCE
/path/to/dev-python tools/native/compare_tensors.py REFERENCE CANDIDATE --require-exact --report parity.json
/path/to/dev-python tools/native/benchmark_comparison.py --model MODEL --engine ORIGINAL_ENGINE --mflux MFLUX --manifest MANIFEST --bridge BRIDGE --output BENCHMARK
```

原生请求 `dump_tensors` 指定候选张量目录。性能测试不使用 dump；比较器任何缺失、shape/finite/数值差异均返回失败。最新证据与限制见 [重构状态](design/rewrite-implementation-status.md)、[性能对比](design/flux-performance-comparison.md)、[视频模型验收](design/video-model-acceptance.md)。
