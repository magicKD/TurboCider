# H3 / LTX 早期迁移快照

这里保留 2026-09-05 之前的研究代码、vendor 和合成测试，仅用于阅读历史，不进入默认发行库。当前正式实现已经迁移到 `native/models/h3_runtime`、`native/models/ltx_runtime`、`h3_session.mm` 和 `ltx_session.mm`；不要在此目录继续修复或同步模型数学。

当前完成度见 [实现状态](../../docs/status/implementation-status-2026-09-06.md)，接入顺序、变体约束、分区/流式内存与真实模型门禁见 [视频验收](../../docs/design/video-model-acceptance.md)。原控制层和 benchmark main 可在 Git 提交 `0035171b0906d608f570072f62bcea8481fdac3d` 读取。`tools/experimental` 仅构建这份历史快照，不能作为当前发行验证入口。
