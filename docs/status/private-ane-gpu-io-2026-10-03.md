# Private ANE 接续：GPU IOSurface I/O（2026-10-03）

接续 [基础执行器](private-ane-foundation-2026-10-03.md)。原任务的 W8A8、
GPU GGUF/Q4→W8、Hadamard、固定两套 staging slots、四格 ≥1.2× 和
完整 LoRA/画质资格均保留，不以本次 FP16 I/O 改造替代。

## 实现与保持的边界

`ane_runtime.hpp` 增加不依赖 ObjC/MLX 的 `DeviceMatrixView` 和
`DeviceAdapterInput`。opaque Metal buffer、真实 extent/offset/stride、dtype
和 allocation owner 一起传递；input 必须已经完成上游生产，immutable 到
finish。Public 默认 `supports_device_io=false`，原 host route 不变。

`HybridFfn` 从已 eval 的 MLX array 提取实际 buffer/offset，保留 array owner；
输出使用独立 MLX allocator buffer。Private worker 直接 GPU 读取输入并写
拥有所有权的 MLX 输出，不调用 host dataPointer、CPU transpose/restore 或
typed-pointer tensor constructor copy。源和目的重叠、不同 Metal device、
短 extent、错误 dtype/几何都会拒绝。输出仍只在整个多 chunk 成功后发布；
任何失败、取消都 drain，并使用原有整段 GPU 重算。

`private/ane_transfer_kernels.hpp` 的两个 32×32 tiled 内核带尾块、pitch 和
offset：upload 做 FP16/BF16/F32→FP16 转置，restore 做 FP16→FP16/BF16
转置及 headroom 恢复。FP16 转换采用明确整数 RNE，避免 Metal half
subnormal flushing 改变 CPU 合同；FP32 scale 禁用 fast math。GPU 检查
source nonfinite、输出 nonfinite、恢复 dtype overflow，拒绝而不 clamp。

`Device::prepare_transfer` 和 `Transfer`：

1. 构造/校验 bindings 并分配每次调用的两个小状态 buffer。
2. ANE request 等待未到达的 ready 值；input CB clear GPU validation flag、
   pack 输入并 signal ready，先 commit。
3. 独立 output CB wait ANE done，再读 IOSurface/restore 到 MLX 输出。
4. callback/timeout 用另一个 **sticky CPU failure flag**，先写 flag 再释放
   done。output kernel 在任何 IOSurface 读取之前检查它，避免失败释放事件
   后读仍在被 driver 写的输出。GPU 从不清除此 CPU flag。
5. ANE 与 GPU completion 都收齐后检查 flags；headroom 重试只缩放 up 和
   up-LoRA，下一次 chunk/层保留 factor。状态/Surface/MLX allocation owners
   被 CB 和 ticket 保留；缺回调/超时不释放仍可能在用的资源。

LoRA 原生 MIL 现在把 `[down; corrected_hidden]` 打包成一个物理 y 输出，
consumer 再用零拷贝 surface row views 分开处理。这样 signal symbol0 覆盖
全部输出，不以第一个输出的事件猜测第二个输出就绪。Private program 的
输入 ABI 不变、source hash 改变；Public 仍保留原 y/h 两输出 ABI。consumer
view 明确不能作为 ANE driver 的完整 IOSurface binding。

**仍有 host 边界**：上游 MLX attention/LoRA 的 `eval` 和完成状态检查仍在
host；权重仍是单层 CPU→FP16 staging。这里只有 device-side pack/restore
和真实事件顺序，不是完整 GPU command graph、W8A8、双缓冲或重叠追踪证明。
新 input/output timers 是 host submission/wait spans，不是 GPU kernel 时间，
不能与旧 CPU copy timer 直接相减当作加速。

## 显式实验选路

只在 private-enabled 构建及显式 runtime route 使用：

```sh
TURBOCIDER_ANE_BACKEND=private TURBOCIDER_ALLOW_PRIVATE_ANE=1 \
TURBOCIDER_PRIVATE_ANE_GPU_IO=1 <cli> batch <model-root> <runtime-request>

.venv/bin/python tools/validation/runtime_ane_model_screen.py \
  --cli <private-enabled-cli> --model <model-root> --model-id <model-id> \
  --runtime-manifest <manifest> --runtime-backend private --private-gpu-io \
  --routes gpu,runtime --steps <steps> --size <size> --output <new-output>
```

不默认开启。receipt 的 `io_path=gpu_iosurface` 与
`device_io_calls_session_total` 必须和每次 prediction 对上；工具拒绝只写
环境变量但走 host/Public 的冒名结果。对照清除继承的 private I/O 环境。

## 已执行验证

- standalone GPU oracle：全部有限 FP16 encodings（含正负 subnormal/零）、
  BF16/F32 RNE/ties、恢复 scale64、padding/offset/tails；与独立 CPU 转换
  逐位对照。真实 GPU→ANE→GPU identity、多 chunk、missing producer 的
  timeout 后 caller output poison 不变，均通过。
- GPU Executor：SwiGLU base/A/base、非零 gate/up corrections、BF16 corrected
  hidden、headroom overflow/retry/跨层保留、alias/Inf 拒绝与恢复通过。
- 7 项 private host/hardware tests 和 46 项 screen 合同通过。
- `637466ab5ab3d423957210cceaa0ce1cbdaee568962efaca297afd94964168b1`
  隔离库构建与 CLI 编译成功，既有 12 项 Public Core ML/MLX 集成通过；
  本库不替换 `build/native` 或 App bundle。后续异常守护改动须另重建，
  不重标上述库的计时成绩。

## 当前完整请求初筛（尚非正式性能资格）

同一 6374 库、狐狸/seed42、每 arm 一冷一热、GPU→Private、chunks=auto。
只测 request_wall，含 VAE/PNG，不包含 cold；无正反序或多热样本：

| 工作负载 | GPU hot | GPU I/O Private hot | GPU/Private | session device calls |
| --- | ---: | ---: | ---: | ---: |
| Z 512² / 8 steps | 6.993325 s | 7.010890 s | 0.997× | 79 |
| Z 1024² / 8 steps | 31.253114 s | 30.543168 s | 1.023× | 415 |
| Qwen 512² / 40 steps，匹配 Q/K 融合 | 41.700037 s | 41.949513 s | 0.994× | 198 |
| Qwen 1024² / 40 steps，匹配 Q/K 融合 | 183.036338 s | 150.541790 s | 1.216× | 2432 |
| Qwen Viggle r256 LoRA 512² / 6 steps | 8.149879 s | 8.492745 s | 0.960× | 263 |

五组 complete，device calls 与 runtime calls 一致，零失败回退。调用是
**session 累计**，不能当作热请求覆盖率。新 GPU 输出与 GPU head 共用
设备资源，IO 局部改进不保证完整 block 更快。
Qwen 1024 的热请求有1248次device predictions，是初筛中唯一超过1.2×
的 geometry；它是完整 request_wall 样本，不是只拿 denoise 或单算子。
但只有一个 hot、一个 prompt/seed、无正反序/独立进程树压力认证，仍不
算正式性能/画质交付，不外推其他三格或所有 LoRA，不升级默认。
原始数据位于 `outputs/private-ane-gpu-io-<model><size>-first-20261003/`。
原始回执/PNG SHA、实际 hot predictions、库身份和负结果见
[便携初筛回执](../design/validation/private-ane-gpu-io-pilot-20261003.json)。
Qwen1024 的 GPU/Private 图片已目视：主体、姿态、构图接近，仍有细节
差异；这不是 latent/LPIPS/广泛 holdout 资格。Qwen LoRA 使用 inference_time，
263次预测均为device I/O，corrected hidden 流程未报错；只覆盖这一个adapter。

最终异常守护源重建成功，计时仍绑定6374旧库，不重标为最终库的成绩：

- Private：`291eff04fe30c9125e3584846c42884f879e4673ac0c18ec2db25912a3808da8`。
- Public-only：`5c71b16ce380479ec8097ea80f7095cb17e5da2889a8480b2d25d931501f6e7a`。

Public-only 完整库不含七类 private ANE class strings 或 PrivateFrameworks
链接；最终 public 库的12项Core ML/MLX集成通过，7 host/7 private/46 screen
检查通过；未执行完整make test。没有仍在运行的benchmark，不删除已有
证据或模型，不替换发行库/App，也不stage/commit。保留fastMathEnabled的
deprecated警告以维持当前已测兼容参数，未把warning当成测试失败。

不认证所有训练 LoRA、media holdout、low-memory envelope、多设备或 native
INT8 arithmetic。完成当前 I/O 验证后必须继续实现 W8A8/Hadamard 和 GPU
权重 stager/双缓冲，再按完整原目标复测。
