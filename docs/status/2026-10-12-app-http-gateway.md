# 2026-10-12 App 内置 HTTP 网关

本记录补充同日 ComfyUI 接入验收，更新了“App 需要单独启动 HTTP 适配器”的旧说明。

## 实现

- App「本地 API」启动原生服务后自动启动原生 Swift / Network HTTP 网关；不启动 Python，不增加模型或解释器下载。
- 默认 `127.0.0.1:8765`，页面可修改端口，成功启动后保存设置。HTTP 协议与独立 Python 网关及现有 ComfyUI 节点兼容。
- 端口冲突会回滚本次原生服务；停止、退出 App 或原生子进程退出时关闭网关，保留任务历史和输出。
- 支持 health、catalog、assets、plan、jobs、status、cancel、artifact。参数与执行能力仍由原生服务校验；模型与 LoRA 来自登记库。
- 页面展示 HTTP 地址、九类路由、ComfyUI / Desktop 连接区别和四组可复制 curl / Python 示例；错误显示在启动控件附近。
- 中文教程、英文接口文档及配套发行目录已同步。App 网关只供可信本机客户端，无公网监听或 Bearer Token 设置；独立 Python 网关仍支持可选 Token。

## 验证

- Swift App 构建、发行打包及 ad-hoc 签名校验通过。
- 原生 HTTP 集成测试通过：分片请求、非法长度、重复头/JSON 字段、Origin/Host、上传、LoRA 解析、plan/submit/status/cancel、结果下载、丢失结果、端口冲突与重启读取历史。
- App 生命周期回归通过：自动启动 HTTP、端口占用回滚、不影响已占用端口的服务、停止释放端口、原生子进程退出关闭网关、旧异步响应与新服务隔离。
- `make test-api`、`make test-app`、仓库 layout 8 项、文档链接/代码块及 `git diff --check` 通过。
- 在实际 App 页面点击验证端口冲突提示和启动成功状态，展开并目视检查 HTTP 示例布局。
- 真实 ComfyUI → App 网关完成 11 个工作流：FLUX 批量、编辑、transform；Z-Image 基础、runtime LoRA、内存合并 LoRA、恢复基础；Qwen 基础、Viggle 六步、单参考编辑、双参考编辑，均 success。全部复用本机已有权重。
- 用户要求快速收尾后停止扩展测试调度，当时已提交的 Qwen 普通 LoRA 任务也完成并返回 success，共计 12 个成功工作流；没有继续执行剩余全量组合。此前独立网关的 14 项验收记录仍保留，但不据此宣称全部组合在 App 网关重新验证。
- `make test` 再次停在原有 `test_cli_ane_override.py` 的 GGUF runtime LoRA 预期差异：测试预期拒绝，当前原生 plan 已接受。此项未改动，不将仓库全量测试报告为通过。

## 本机证据

`outputs/comfyui-validation/` 下保留 `app-gateway-*.log`、`app-gateway-e2e/` 的请求、回执和输出，以及最后已提交任务的 `app-gateway-current-history.json`（完成后保存）。生成物不入 Git。

最终 `dist/TurboCider.app` 已实际打开并启动网关；health、catalog 及重启后的历史 PNG 下载快速检查通过。最终使用 `dist/TurboCider.app`。更新后的[中文教程](../public/COMFYUI_TUTORIAL.zh-CN.md)说明 App 启动方式；同版本 `dist/comfyui/` 提供节点和工作流工具。
