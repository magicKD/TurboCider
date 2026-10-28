# ML Drift：CLT-only 隔离构建、失败与依赖身份

<!-- knowledge
id: technique-gpu-ane-case-studies-mldrift-isolated-build-and-dependencies
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [完整 FC 实测](mldrift-complete-int4-fc.md) · 构建 receipt · [来源范围](../sources.md#mldrift-int4-fc)

2026-10-16，固定源码 ML Drift `1b18771f44c0be3d972847b31a80f6dc3d6964b5`，实际构建原始 `common/kernels:fully_connected`，再链接自有 Metal host。没有修改参考 checkout、上游源码、上游 BUILD、系统 xcode-select 或生产后端。构建不是设备推理，最后 GPU 是否真的运行另由证明。

## 工具与复制边界

上游 `.bazelversion` 指定 8.0.0。本轮只下载[官方 Darwin arm64 二进制](https://github.com/bazelbuild/bazel/releases/download/8.0.0/bazel-8.0.0-darwin-arm64)及[官方 SHA](https://github.com/bazelbuild/bazel/releases/download/8.0.0/bazel-8.0.0-darwin-arm64.sha256)到 ignored outputs，实际校验值：

```text
ea529c3c03d6981e13e25af126b63b41435c9a42bf28435bbabbe3b11542ddcf
```

没有全局安装 Bazel。系统提供真实 `/Library/Developer/CommandLineTools`，没有完整 Xcode.app；使用真实 CLT 路径，不捏造 Xcode 版本。

从 tracked-clean 固定参考复制已 materialize 的 `ml_drift/`、`third_party/` 及 `MODULE.bazel`、根 `BUILD`、`LICENSE`、`requirements.txt` 到隔离 `workspace/`。556 个复制 tracked 文件逐一按 Git blob（含 symlink 语义）核验，全部匹配原 commit；manifest 及其 SHA 留在原始 artifacts/receipt。它不是完整仓库快照，也不是 556 文件人工源码阅读，本轮明确阅读清单为 22 文件。

仅额外加入 `local_probe/mldrift_full_fc.mm` 与 `local_probe/BUILD`，对应可分发的原创 host和原创 BUILD。上游 `metal/testing/test_util.h` 引用 `testing/base/public/gunit.h`；本轮选择直接使用原始 operation/task，不使用这个测试封装。没有构建该测试 target 来复现其失败，不能宣称发现了上游 test build bug。

## 五次尝试：失败不可从历史中消失

| 记录 | exit | 观察与处理 |
| --- | --- | --- |
| common-build-attempt-1 | 37 | Bazel internal crash，DottedVersion 收到 `None`；未显式传 target/host CLT env |
| common-build-attempt-2 | 37 | 仅 target `--action_env` 仍失败；host action 环境未覆盖 |
| common-build-attempt-3 | 0 | target 与 host 均显式传真实 CLT 路径，原始 FC target 构建成功 |
| metal-build-attempt-1 | 1 | 自有 `cc_binary` 直接收 `.mm` 不被所选规则接受；是本地 wrapper BUILD 错误 |
| metal-build-attempt-2 | 0 | 改用 `objc_library` 编译 `.mm`、`cc_binary` 链接，成功生成完整 host |

全部由已有 owned-process supervisor运行，超时分别有界，全部确认 terminal；完整 stdout/stderr/command/receipt 与第一次错误 BUILD 保留。前两次不是 GPU 编译/数值错误，第四次不是上游 kernel 缺陷；只有最后成功的 host 用于后续 GPU 准入。构建日志耗时、supervisor elapsed 或 admission elapsed 都不能当 inference latency。

## 为什么必须同时传 target 和 host 环境

精确读取 [Bazel 8.0.0 `XcodeLocalEnvProvider` 54–140](https://github.com/bazelbuild/bazel/blob/8.0.0/src/main/java/com/google/devtools/build/lib/exec/local/XcodeLocalEnvProvider.java#L54)：当 action env 含 Xcode version、却不含 `DEVELOPER_DIR` 时，它会解析版本并查找 developer dir。本机失败日志中的版本值是 `None`，不是合法 dotted version。

观察到仅覆盖 target action 后仍失败，而同时覆盖 host action 后构建成功；据源码和日志，本轮将其归因为 host/target action 环境差异下的 Bazel CLT-only 路径问题。这不是对所有 Bazel/macOS/Xcode 组合的通用修复承诺。没有修改 Bazel Java 源码、伪造版本、安装 Xcode 或改系统选择。

核心成功配置如下；在自己的**新隔离目录**中执行，`probe_root` 必须是实际 artifact 根。工具下载、源复制和自有 target 放置是显式准备步骤，curator 不会代为执行：

```sh
probe_root="$PWD/outputs/gpu-ane-mldrift-fullop-replay"
cd "$probe_root/workspace"
"$probe_root/tools/bazel-8.0.0-darwin-arm64" \
  --batch --output_user_root="$probe_root/bazel-user-root" \
  --host_jvm_args=-Xmx2g --ignore_all_rc_files \
  build //local_probe:full_fc --jobs=4 --cxxopt=-std=c++17 \
  --enable_bzlmod --lockfile_mode=update \
  --action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  --host_action_env=DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  --repository_cache="$probe_root/repository-cache" \
  --disk_cache="$probe_root/disk-cache" \
  --color=no --curses=no --verbose_failures
```

common target 使用相同配置，仅将 target 换为 `//ml_drift/common/kernels:fully_connected`。实际运行通过 supervisor，分别限制 240/300 秒；上面只展示 child build 命令，不自动占用 GPU。忽略所有 rc 是为了保留实际可核查的 jobs=4 等配置；不修改上游 jobs 配置或用户全局 rc。

## 固定源码不等于完全固定依赖闭包

根 MODULE 声明版本与 Bzlmod 实际解析并非一回事。真实构建日志的 warning 包含：

| 根声明 | 日志观察到的解析结果 |
| --- | --- |
| rules_python 0.40.0 | 1.7.0 |
| rules_cc 0.2.0 | 0.2.22 |
| googletest 1.17.0 | 1.18.0 |
| protobuf 29.3 | 35.0 |

完整观察列表在receipt。`MODULE.bazel.lock` 的 lockFileVersion 为 16，SHA 为 `64eeb6309f237822453195f689375aaa1aaa743015a1bd14e6a083dff77ef3e6`。lock 中出现某个 registry URL 只说明记录了候选/访问，不能全部列为选中版本；本轮列表来自实际解析 warning，不声称枚举完整传递依赖闭包。

构建使用 `--lockfile_mode=update`，不是预先冻结 lock 的离线 hermetic 构建。记录原源码、官方工具 checksum、生成 lock、解析差异、原始日志和最终 executable SHA，能解释“本机究竟构建了什么”，但不能保证未来 fresh resolve、其它 SDK/OS/系统库会字节一致。最终 host SHA 为 `4db46d4b43ef735996c783c0b726f53f2267c3ea9b8fc97addc28d9688149278`，不应从它推出所有动态系统依赖已冻结。

## 只读重核

从仓库根目录，已有原始 artifacts 可重新做 Git blob、工具和日志绑定检查：

```sh
python3 techniques/gpu_ane/mldrift_full_fc_probe.py \
  --reference ../references/ml-drift-research-20261016 \
  --artifact-root outputs/gpu-ane-mldrift-fullop-round12-20261016
python3 techniques/gpu_ane/capture_mldrift_fc_sources.py \
  --reference-root ../references \
  --artifact-root outputs/gpu-ane-mldrift-fullop-round12-20261016
```

这里不会 fetch、安装、build、import 上游库或启动 GPU；source manifest 已存在时要求字节一致，未存在时仅在显式 artifact 根内创建它。可分发报告不携带上游源码、工具、lock 原文或编译物。没有清系统缓存、改产品路由或删除构建 artifacts。
