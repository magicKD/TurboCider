# 轻量检索与文档维护工具

<!-- knowledge
id: technique-scripts-readme
type: navigation
-->

[知识库](../README.md) · [使用指南](../guides/knowledge-base.md) · [维护规则](../MAINTENANCE.md)

仅依赖 Python 标准库，直接读取 Markdown 标题、短元数据与链接。没有 JSON catalog、证据仓库、版本 hash 账本或设备依赖，不执行被检索的代码。

## 按阅读目的查询

```sh
python3 -B techniques/scripts/query.py DeepCache --intent learn --limit 3 --compact
python3 -B techniques/scripts/query.py small-M --intent implement --limit 3 --compact
python3 -B techniques/scripts/query.py MLX --intent sources --limit 3 --compact
python3 -B techniques/scripts/query.py '前馈网络' --stage dit --backend metal --limit 5
```

`learn`优先相关方法，`implement`优先实现/案例，`sources`优先源码阅读材料；`all`只按关键词排序。可按 `--page-type`、`--section`、`--type technique|source`及主题标签过滤。同一维度及跨维度均为 AND；关键词别名组按 OR 评分，不因同义词重复加分。

默认 stdout 为结构化内容，`--compact`仅显示 ID、类型、标题和路径。不会写 JSON 文件。标签描述页面主题，不承诺适用性、兼容性或验证状态。

## 有界阅读

```sh
python3 -B techniques/scripts/get_page.py algorithms/methods/deepcache.md --follow-sources --max-bytes 16000
python3 -B techniques/scripts/get_page.py designs/hybrid-ffn.md --include-code --max-bytes 24000
```

接受已登记的 ID、旧 ID 别名或相对 Markdown 路径。正文、来源页和小型教学代码共用字节预算；返回裁剪、省略项和下一字节 offset，UTF-8 不从字符中间截断。仅展开知识库内来源页与 examples 代码；产品链接和互联网链接不自动打开。

旧 `--intent evidence`兼容为来源阅读；`--follow-evidence`仅返回“不分发实验包”的说明，不冒充数据可用。

## 维护检查

```sh
python3 -B techniques/scripts/check_wiki.py
python3 -B techniques/scripts/generate_query_index.py --check
```

`generate_query_index.py`默认将 Markdown 索引打印到 stdout，`--check`检查当前索引是否一致；不运行实验。文档检查只核对标题、链接、元数据与无 JSON 规则，不能证明历史数字、模型正确性或性能。
