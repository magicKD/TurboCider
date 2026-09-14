# LTX 2.5 Video VAE 生命周期优化与代码整理

**日期：2026-09-13**  
**范围：TurboCider 的 LTX 2.5，Sol attention 与 ANE 暂不纳入默认路径**

## 1. 结论

TurboCider 的 LTX 2.5 Transformer 已经采用 C/Metal GPU 主路径，当前实测 denoise 约
52.94 s，快于同条件 `ltx-mac` 的约 56.21 s。现在的主要差距来自 Video VAE，而不是
VAE 卷积数学实现：当 VAE 在 Transformer 父进程存活期间启动时，Metal、MPSGraph、MLX
allocator 和统一内存页仍然被旧阶段占用，GPU decode 会从干净进程的约 3 s 退化到约
8--10 s。

因此默认产品路径确定为：

1. Transformer、conditioning、upsample 使用 C/Metal GPU；
2. `allow_approximation=false`，Sol Stage 1/2 和 Stage-2 text pruning 默认关闭；
3. `execution=gpu`，ANE profile 不自动选择；
4. `component_staged` 在 denoise 完成后释放 Transformer 对象，并通过 `exec` 将当前进程
   替换为干净的 MLX C++ Video/Audio finalizer；
5. finalizer 完成 Video VAE、Audio VAE/vocoder/BWE、RGB 转换和 AAC mux 后退出。

这套方案是生命周期优化，不改变 latent 或 VAE 算法，因此质量应与原始路径保持一致。

## 2. 根因与证据

### 2.1 为什么普通子进程仍可能很慢

`posix_spawn` 出来的 VAE helper 虽然拥有新的用户态地址空间，但父进程仍然占用统一
内存中的 Metal/MPSGraph/MLX 资源和 GPU 页。子进程创建 VAE graph 时，系统仍可能受到
内存压力、回收和 GPU 队列竞争影响；仅仅把 VAE 调用放入子进程并不能保证与干净启动相同。

`exec` 则会用 finalizer 映像替换当前进程，旧的 C/Metal、MPSGraph、Core ML 和 MLX
allocator 生命周期全部结束。Transformer latent 通过临时 BF16 文件交给新映像，文件传输
只占毫秒级，主要收益来自释放框架级缓存，而不是减少拷贝。

### 2.2 当前同条件测量

工作负载为 704×448、97 帧、24 FPS、11 steps、seed 42。数字应理解为本机 M4 Max 的
warm/近 warm 结果，正式发布仍需用 ABBA harness 重测。

| 路径 | Video VAE decode | finalizer/阶段 wall | 结论 |
|---|---:|---:|---|
| `ltx-mac` clean finalizer | 约 2.77--2.95 s | 约 2.95 s | 参考基线 |
| TurboCider 普通 spawned helper（父进程仍存活） | 约 8.21 s | 约 8.46 s | 诊断/回退，不作为默认 |
| TurboCider clean exec，video-only | 约 2.98--3.21 s | 约 3.4--5.4 s | 默认目标路径 |
| TurboCider clean exec，video+audio | 约 3.05 s | 约 3.88 s | 音频路径已接通 |

带音频的 clean finalizer 另外测得 Audio VAE/vocoder/BWE 约 0.62 s，AAC mux 约 0.04 s。
Video-only 与 clean exec 的同一 latent 输出通过 `framemd5` 比较完全一致，说明进程切换
没有引入画质变化。

## 3. 运行时流程

```text
request
  |
  +-- C/Metal Transformer (Stage 1 -> upsample -> Stage 2)
  |       |
  |       +-- video latent BF16
  |       +-- audio latent BF16 (audio=true)
  |       +-- release denoiser / MPSGraph / mmap / media handles
  |
  +-- component_staged + TURBOCIDER_LTX_EXEC_FINALIZER=1
  |       |
  |       +-- exec ltx-video-finalizer
  |               +-- Video VAE decode (MLX C++)
  |               +-- BF16 planar -> RGB24 -> temporary MP4
  |               +-- optional Audio VAE -> base vocoder -> BWE
  |               +-- optional AAC mux -> requested output
  |               +-- remove temporary latent/video files
  |
  +-- fallback: spawned helper / in-process VAE
```

`video.generate` 和 `video.image` 都可以进入 clean finalizer；I2V 的首帧 clean-prefix 在
Transformer 阶段完成，finalizer 只处理最终 latent，不会改变 conditioning。

服务端的 `turbociderd` 对所有 `component_staged` LTX 请求使用 disposable worker。worker
本身负责 Transformer，结束时再由 session `exec` 到 finalizer；这样 daemon 不会把长期存活
的 allocator 带入 VAE 阶段。`resident` 仍保留为性能候选和诊断路径，不应绕过 component
staged 的默认生命周期合同。

## 4. 回退策略与开关

| 层级 | 条件 | 行为 | 适用场景 |
|---|---|---|---|
| Clean exec finalizer | helper 存在且请求为 component-staged | `exec` 替换进程 | 默认、性能验收 |
| Spawned helper | 无法 exec 或显式诊断 | 新进程调用 Video VAE | 故障回退、生命周期对照 |
| In-process | 没有 helper | 直接调用 MLX VAE | 最低依赖的最后回退 |

`TURBOCIDER_LTX_VAE_TIMING=1` 会输出 helper 的 input read、weight load、decode、output
write 和 child wall 分项。`TURBOCIDER_LTX_EXEC_FINALIZER` 只能由 CLI 的 component-staged
计划或 disposable service worker 设置，不能由普通 resident 请求静默开启。

Sol/ANE 相关选项继续保持显式 opt-in：

- `allow_approximation=false` 时不启用 Sol 或 text pruning；
- `execution=gpu` 是默认，`gpu_ane` 必须显式提供已验证 manifest；
- `ltx_video_attention_batch` 默认关闭；
- `ltx_fast_av` 是数学等价的 C/Metal 调度优化，可保留默认开启，但不得与 Sol/ANE 的
  结果混为同一个质量基线。

## 5. 代码职责整理

### 5.1 Session 层：`native/platform/apple/ltx_session.mm`

- 负责请求校验、checkpoint/conditioning、C/Metal denoise 和阶段计时；
- 在 Stage 2 后获取最终 latent 并释放 denoiser；
- `decode_ltx_video_isolated` 只保留为 spawned-helper 回退；
- `exec_ltx_video_finalizer` 只负责创建临时目录、写入 latent、构造 argv/env 和执行替换；
- 不在 session 内重复实现 Video/Audio VAE 的数学逻辑。

### 5.2 工具层

- `tools/native/ltx_video_vae_decode.c`：最小 Video VAE helper 与可选分项计时；
- `tools/native/ltx_video_finalizer.mm`：clean process 中的 Video/Audio VAE、vocoder、BWE、
  RGB 转换、MP4/AAC mux 和结果 JSON；
- `tools/native/build.sh`：finalizer 链接 `audio.o`、`video.o`、AudioToolbox，并保证 MLX
  runtime 路径可重定位。

### 5.3 入口层

- `apps/cli/main.mm`：对 `video.generate` 与 `video.image` 的 component-staged 请求设置
  exec finalizer 和 conditioning cache；
- `services/turbociderd/service.mm`：把 component-staged LTX 请求路由到 disposable worker，
  保留取消、SIGTERM/SIGKILL 和 stderr 捕获；
- `tests/native/test_contract.py`：静态检查 audio finalizer、worker 路由、argv 合同和默认
  关闭的近似路径。

### 5.4 代码整理原则

1. 媒体后端只由 finalizer/session 的明确边界持有，不把 VAE graph 放回 Transformer block；
2. 每个结果 JSON 同时写出 `video_vae_isolation`、阶段计时、audio metadata 和实际 backend；
3. 临时文件使用随机目录和精确清理，不删除模型目录、共享 cache 根或工作区；
4. 近似算法只能通过 Request/profile 显式进入，不能被环境变量偷偷改变；
5. 不把一次失败的 Core ML/ANE 编译或普通 helper 的慢结果误标为算法退化。

## 6. 验收方案

### 6.1 功能门

- T2V：704×448×97、768×448×121、1280×704×121（720p 使用 tiled VAE）；
- I2V：同 shape，至少 strength 0.65，首帧必须影响输出；
- 音频：T2V 和 I2V 各一组，输出包含 48 kHz、双声道 AAC，时长与视频对齐；
- 取消：Transformer 阶段和 finalizer 阶段都能中止，并回收临时文件；
- 服务：CLI、C ABI 和 daemon 均报告同一 isolation/backend metadata。

### 6.2 性能门

使用相同 checkpoint、prompt、seed、shape、steps 和输出口径，先 warmup，再交错运行
A-B-B-A 与 B-A-A-B，每种至少 3 个 measured runs。分别记录：model load、conditioning、
Stage 1、upsample、Stage 2、Video VAE、audio decode、mux、request wall、client wall、
peak RSS 和 swap。

建议门槛（同一台机器、同一 MLX 版本）：

- Transformer denoise：TurboCider / `ltx-mac` ≤ 1.10；当前 52.94 / 56.21，已满足；
- clean Video VAE decode：≤ `ltx-mac` warm median 的 1.15 倍，目标约 ≤ 3.4 s；
- video-only finalizer wall：≤ 4.5 s（不把 Transformer 时间重复计入）；
- video+audio finalizer wall：≤ 5.0 s；
- 普通 spawned helper 只需可用，不作为性能发布门；若 clean exec 失败，必须记录回退原因。

### 6.3 质量门

- clean exec 与 in-process/spawned 采用同一 latent 时，Video VAE BF16 和 `framemd5` 必须
  完全一致；
- T2V/I2V 跨后端质量用固定 prompt×seed 套件检查主体、动作、构图、颜色、闪烁、黑帧和
  播放完整性；
- 音频检查采样率、声道、可解码性、时长、首尾裁剪和 mux 后视频帧数；
- Sol/ANE 的质量报告必须单独标记 `algorithm_approximations`，不能与 quality 基线合并。

### 6.4 失败定位顺序

1. `video_vae_decode` 高而 `input_read`/`output_write` 低：检查父进程是否仍存活、是否走
   exec、GPU 内存压力和 swap；
2. `weight_load` 高：检查 checkpoint 路径、mmap、MLX 版本和模型缓存；
3. RGB/export 高：检查输出分辨率、临时磁盘和 ffmpeg/VideoToolbox；
4. audio decode/mux 高：分别检查 audio checkpoint、vocoder/BWE 和 AAC 编码；
5. 输出不一致：先比较 latent，再比较 VAE BF16，最后比较 RGB/MP4，避免把编码器差异误判
   为 Transformer 数值问题。

## 7. 当前状态与后续

已完成 clean finalizer 的 Video-only 和 Video+Audio 路径、CLI/daemon 路由、分项计时、同
latent `framemd5` 一致性检查和静态契约测试。下一步只需要在目标设备完成同口径 ABBA、多
prompt/seed 和 720p tiled VAE 复验；在此之前不打开 Sol attention 或 ANE 默认开关，也不把
C++/MLX Transformer 替换 C/Metal 主路径。

