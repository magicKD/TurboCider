# Studio 首版实现与验证

日期：2026-09-05。此文记录已实现内容；此前的 App 方案保留为设计背景。项目目前正式路径为 `apps/macos`、`bindings/swift` 和 `native`，旧方案中 `Sources/`、`native/swift/` 路径已过时。

## 已实现

- 原生 SwiftUI 工作室：导航（创作/素材库/任务/模型）、中央媒体预览、历史缩略条、底部输入与生成状态、可收起的右侧参数。
- FLUX 4B/9B 文生图、单图修改、有序参考编辑；H3/LTX/FastMetal 通过同一 descriptor 驱动的模型中心进入视频请求。每次一个主结果、App 一个活动任务。后台不自动连发。
- 视频模型会从 descriptor 读取操作、尺寸、帧数、FPS、音频和输出类型；MP4 结果使用视频预览，图片结果使用降采样预览。
- 文件选择、拖放、图片粘贴共用受管素材导入；PNG/TIFF 剪贴板 representations 去重。提示词内纯文本正常粘贴；图片粘贴进入素材区。提供“粘贴图片”按钮及 Cmd+Shift+V。
- 素材有独立 UUID；排序、移除、撤销、选择原图、模式切换保留未使用素材。超 8 张时整批拒绝并解释，不截断。读取中途失败清理本次新副本，已提交任务输入不受草稿删除影响。
- 默认固定种子 42，可输入整数、随机一次固定、每次提交随机。随机仅在构建本次请求时解析；历史保存实际 seed。
- 常用输出尺寸、原图保留强度；高级区支持 1–50 步、动态文本、常驻/分阶段释放、本机加速 profile。支持独立 `.safetensors` LoRA 文件、角色和强度；运行时只传文件路径和身份，不复制或合并基础 checkpoint。
- 草稿自动保存、任务请求快照、历史恢复、复用参数、图像预览/原图切换、另存为与 Finder 定位。
- 进度保留用户阶段；内部 transformer_block 不反复覆盖采样阶段。真实完成步计算最近最多 5 步的秒/步，至少 2 个样本才展示；图生图显示实际采样步数，排除被跳过的 schedule 步。只估计当前采样阶段剩余时间，不伪造全程 ETA。
- 缩略图与媒体预览在后台解码、按路径/尺寸变化加载，不因遥测刷新反复读取原图。内部块回调只按最多 4 Hz 更新时间；主要阶段发布原子快照，避免 UI 看到新阶段搭配旧总步数。

## Load / Unload 的准确语义

以下是首版 C ABI load 的语义。App 现已改用包含当前文本和加速分区的 prepare；大内存机不再无条件释放预加载权重。最新实现与条件见 [模型准备与性能](model-preparation-and-performance.md)。

新增 C ABI `tc_engine_load`、`tc_engine_unload`，Swift 提供 async `load/unload`。均与生成共享引擎/进程准入锁；load 也遵守跨进程 GPU 作业锁。App 在忙碌时拒绝模型切换/卸载，失败保留准确会话状态。

Load 先建立本地 FLUX 会话，再读取并 materialize transformer 与 VAE 权重。不是仅解析目录、登记 lazy tensor 或伪造进度。约 7.92 GB 是本机该 BF16 模型图像权重的测量值，并非所有模型承诺。文本编码器仍在生成时按需加载；加载不等于 prompt/shape 预热，也不保证首次出图会更快。为避免内存峰值，首次/改变提示词时引擎会先释放图像权重再编码文本，然后重新装入图像权重。

Unload 在安全空闲边界同步，释放图像权重、conditioning、混合会话及 allocator cache，再移除 App 会话引用。不删除磁盘模型和图片。报告内存作用域是 MLX allocator，不能声称系统页缓存/全部进程内存归零。

## 代码职责

| 文件 | 职责 |
|---|---|
| `apps/macos/App.swift` | 窗口/导航/页面、文件对话框、提交与退出提示 |
| `apps/macos/StudioState.swift` | descriptor 驱动草稿、seed/视频参数、LoRA 文件、输入映射、序号/撤销、统一导入/粘贴 |
| `apps/macos/MediaViews.swift` | 后台降采样预览、纯文本/图片粘贴分流 |
| `apps/macos/JobStore.swift` | 独立任务/模型生命周期、持久化、真实阶段遥测 |
| `bindings/swift/TurboCiderNative.swift` | 异步 session 创建、生成/load/unload、C 字符串所有权 |
| `native/api/c_api.mm` | C ABI 加载/释放与资源互斥 |
| `native/models/flux2/`、`native/backends/mlx.cpp` | C++ FLUX executor、图像权重 materialize 和运行时 LoRA；推理数学保持原样 |

## 复现

```sh
# 一次完整原生构建；依赖已存在本地，不自动下载。
MLX_ROOT=/path/to/mlx make build
# 仅修改 Swift UI 后可使用快速构建。
make build-app
make test
make test-app
make test-model MODEL=/path/to/FLUX.2-klein-4B OUTPUT=/path/to/new-validation
MLX_ROOT=/path/to/mlx tools/native/package.sh
```

`test-app` 使用独立命名剪贴板，不改用户当前剪贴板；需要普通 macOS pasteboard 服务访问。真实模型测试需要 Metal GPU 与已有权重。在限制执行环境中，缺少 pasteboard/GPU/socket 访问不表示 App 本身失败，需在普通本机权限下运行。

## 验证范围与证据

- 仓库边界 3 项、原生请求/ABI 契约 10 项。
- Studio 行为测试：种子、输入角色/顺序/撤销、超限拒绝、粘贴去重、失败导入回滚、草稿恢复与真实步速计算，并覆盖 FLUX 9B/H3/LTX/FastMetal descriptor、视频参数和独立 LoRA 转发。
- Studio 真机测试：显式载入真实权重、拒绝 busy unload/第二次提交、FLUX 三操作、实际图生图采样计数、素材和任务恢复、释放内存、卸载后再生成。
- 256×256 的三操作请求分别通过 App store 和直接引擎执行，PNG 逐字节比较；此检查证明 App 层未改变请求/输出，不作为全部模型质量范围认证。
- 实际打包 App 手动 UI 检查：文件插入 → 单图修改（512×512）→ 参考编辑（512×512），中文“将茶壶改成红色”成功输出红色茶壶。提示词、参数、主按钮在本机约 980×750 窗口同时可见。
- 继承的原生 self-test、生命周期（中途/导出取消、同 seed 暖输出一致、prompt 缓存失效、持久化）及 Unix socket 服务测试用于回归公共引擎边界。

最终机器可读摘要见 [Studio 验证](validation/app-studio-validation.json)。原始图像、报告及运行数据在本机 `outputs/studio-validation/`，不提交大图/权重。

## 保留的边界

H3、LTX、FastMetal 的模型中心请求映射和 descriptor 契约已接入；真实视频端到端仍受各模型自己的 provenance/profile/音频门禁约束，App 不会绕过这些 fail-closed 检查。App 仍使用嵌入会话，退出运行中的 App 会提示中断，不宣称后台续跑。素材删除只移除绑定，持久副本保留以保护历史；自动垃圾回收、完整资产项目管理、视频编辑、批次和任务重排留后续。

首版导入单帧图片上限 8000 万像素，是防止异常解码的 App 限制；动画/多页图片需先导出单帧。保留源编码与方向元信息，实际缩放/色彩预处理仍由引擎拥有。App 当前本机 ad-hoc 签名，未做 Developer ID 公证或 App Sandbox 发行验收。
