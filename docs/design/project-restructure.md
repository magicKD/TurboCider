# TurboCider 项目级重构与旧系统退役

2026-09-05。本文描述已执行的仓库结构调整，解决“native 是新实现，但根目录仍有另一套系统”的问题。运行能力以 [实现状态](rewrite-implementation-status.md) 为准；更长远的设计不因目录迁移而自动完成。

## 唯一正式代码树

```text
TurboCider/
  Makefile                         统一 build/package/test/test-model/doctor
  README.md                        唯一产品入口
  native/                          推理引擎，不包含 App、服务或研究 vendor
    core/                          请求、ABI实现、设备配置、执行准入
    backends/                      MLX/Metal、Core ML、编译缓存
    models/                        Registry、FLUX数学、H3/LTX/FastMetal Session 与正式 runtime
    media/                         正式图像输入与PNG导出
  apps/
    macos/                         SwiftUI App、客户端历史/草稿
    cli/                           原生 CLI
  services/turbociderd/             本地持久任务服务，由同一CLI serve命令启动
  bindings/
    c/include/                     公共C ABI与Clang module map
    swift/                         async Swift SDK
  profiles/                        默认关闭的设备加速配置
  examples/requests/               文生图/图生图/参考编辑可修改示例
  tests/
    repository/                    单一入口及发行依赖边界门禁
    native/                        请求/服务测试
    integration/                   Swift生命周期/App数据流程
  tools/
    native/                        正式构建、打包、FLUX oracle和性能对比
    experimental/                  不进入默认目标的视频迁移工具
  experimental/video/              已冻结的早期 H3/LTX源码草稿、vendor、合成测试
  docs/
    USAGE.md                       运行/SDK/服务说明
    status/                        当前完成度、性能和版本提交准备度
    design/                        架构、性能、验收与JSON证据
    archive/control-plane/         已退役Python控制层的历史文档
  build/ dist/ outputs/             忽略版本控制的构建、发行及实验结果
```

没有用空目录/占位类伪装完整通用框架。`native/core` 内部目前按实际职责组织少量文件；将来出现独立生命周期或多实现时，再拆 planning/runtime/memory 子模块。性能热路径保留具体张量函数，避免仅为抽象新增逐算子虚调用。

## 已移除的旧入口

| 旧位置 | 原职责 | 替代 |
|---|---|---|
| `src/turbocider` / `pyproject.toml` | Python runner、adapters、daemon、model management | native库 + 原生本地服务；模型下载/通用转换未纳入本期 |
| `Sources` / 旧 `Package.swift` | 连接Python HTTP服务的Swift App/SDK/CLI | apps/macos + bindings/swift + apps/cli |
| `scripts` / `packaging` / `requirements-flux2.txt` | Python环境bootstrap和旧App打包 | Makefile + tools/native，运行不依赖Python |
| `engines` | 三个旧子进程引擎的副本 | FLUX/H3/LTX正式 runtime 已进入 `native/models`；历史草稿仍隔离到 `experimental/video` |
| `model-packs` / `device-profiles` / `schemas` | 旧CLI路径/env与控制协议 | ModelModule能力、版本化Request、profiles；不保留冲突schema |
| 旧 `tests` / `benchmarks` | 旧控制层与engine命令验收 | 新tests、examples与可重复AB/BA性能工具 |

删除前逐文件比较 HEAD 内容，确认以上旧文件没有本轮外的未提交修改或未跟踪文件。旧系统完整记录位于 Git 提交 `0035171b0906d608f570072f62bcea8481fdac3d`；可用 `git show COMMIT:path` 阅读，或在另一个 checkout/worktree 恢复。没有重写Git历史，也没有修改外部 h3.c-fork、ltx-mac、flux2-engine 参考仓库。

旧 M4 Max/64GB 配置的“validated”声明仅作为历史证据保存在 `validation/legacy-m4-max-policy.json`；它不是新引擎可启用的认证策略。新M4Pro配置默认关闭，启用时验证本机身份。

## 构建与依赖决定

正式入口 `make build` / `make package`，底层脚本明确列出每个发行源文件，不使用 `native/models/*.mm` 自动收集，防止实验目录意外链接。H3 与 LTX 正式 runtime 已进入动态库目标；package 同时携带两套 Metal shader 和 LTX clean-exec Video VAE helper。`experimental/video` 仍不进入构建。

当前环境未安装CMake，系统xcrun存在本机工具链问题；已验证的构建直接使用完整Xcode编译器与SDK，MLX_ROOT明确指定本地MLX C++ 0.32.0。不新增一个无法在本机验证的占位CMake/SwiftPM配置，也不保留会构建旧App的Package.swift。后续引入CMake/SwiftPM时，必须调用同一库目标并通过相同验收，而不是再次产生另一套产品。

正式产物只有 `dist/TurboCider.app` 和 `dist/cli/turbocider`；不再生成名为Native的平行产品。动态库名 `libturbocider.dylib` 保持ABI兼容。Swift内部类型名和旧历史存储目录可以保留，避免仅改名字破坏用户已有历史。

## 分层约束与后续实现

- App只经Swift SDK发起任务；CLI与服务只经C ABI，不嵌入模型数学。
- 公共C头只包含基础C类型，MLX/Objective-C对象不跨ABI。
- 模型Module管理能力、语义与Session工厂。H3、FastMetal 与 LTX video-only 已有显式 executor；LTX I2V/音频仍需各自 parity/provenance gate，不能由于 vendor 能编译就开放未验收 operation。
- `experimental/video`不属于库依赖；后续每迁一个真实子图，先通过测试再移入native，并在构建源清单显式加入。
- 原生服务目前通过CLI的serve子命令启动；是否最终拆出独立helper可后续决定，服务实现已与CLI参数层分离。
- 默认App嵌入式、服务共享队列、通用AssetStore/自动分区等剩余设计边界见实现状态。目录重构不掩盖这些差距。

## 本次迁移验收

1. `make test` 检查旧入口不存在、发行源清单不含experimental/vendor、核心产品入口齐全，并执行请求契约。
2. 重新从新目录完整构建并打包，不依赖被删除目录。
3. 发行CLI真实FLUX推理、自检、Swift生命周期与服务恢复；App启动与参考图生成/预览。
4. 数学实现不因目录移动改变；最终PNG导出单独对照原始引擎。性能报告继续适用相同推理库，产品层路径变动不引入运行时适配器。
