# H3 / LTX 后续迁移草稿

这里保留已研究/部分迁移的模型计算代码、vendor和合成测试，不进入默认发行库。两模型注册为executor=false，不能用这些草稿声称完整生成或数值正确。

接入顺序、变体约束、分区/流式内存与真实模型验收见 [视频验收](../../docs/design/video-model-acceptance.md)。原控制层和benchmark main可在Git提交0035171b0906d608f570072f62bcea8481fdac3d读取。未下载模型。辅助构建脚本位于tools/experimental，生成archive不等于生成系统已可用。
