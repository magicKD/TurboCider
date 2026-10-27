# ANE 配置样例

`export-z-image.json` / `compile-z-image.json` 是 Public frozen Core ML 的准备请求。

`qwen21-runtime-shape-r*.json` 只是 **Private runtime shape descriptor**，不含
模型权重或已编译程序。它们需要 Private-enabled native、显式 Private 授权、
原始模型根和匹配的 W8A8 环境；公共发布库不能将其当作 frozen Core ML manifest。
FFN 全宽仍 H4096/F12288；ANE/GPU 的通道划分由运行配置选择，不修改 descriptor。

完整研究配方见[最快配置索引](../../docs/benchmark/CURRENT_FASTEST_CONFIGS.md)：
Base 512² 完整 40 步使用 R1024；Turbo 作者 8 步在 512²、768×512、1024² 的已测
最佳行数分别是 R1024、R1536、R2048。在创建 session 前选定一份 descriptor，
不要在热 owner 里替换它后宣称无新 load/recompile。R1536/R2048 不是 Base QKV、
任意参考数或 LoRA 的通用资格。

这些配置可作为 profiling 脚本的 `--manifest`，不用依赖已忽略的历史 `outputs/`
文件。模型、编译缓存、参考图片和采集结果仍需自行准备，不能从 shape 推导画质或
物理 INT8 MAC、硬件重叠、所有尺寸的收益。
