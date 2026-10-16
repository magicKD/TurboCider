# KernelWiki检索与版本复查：借合同，不搬硬件别名

[总目录](../../README.md) · [源码索引](../README.md) · [本地查询合同](../../data/README.md) · [既有组织导读](../../references/kernelwiki-sana.md)

2026-10-16只读复核`mit-han-lab/KernelWiki`固定commit `b6b4301f15e8ce6955a56776690643ce5db369e6`，不是当前远端版本声明。六个文件的完整SHA/阅读范围登记在[同一来源清单](../../references/sources.json)，未fetch/reset或修改参考checkout，也未运行上游脚本/安装依赖。许可为MIT，Copyright (c) 2026 hanlab；本地代码独立实现，没有复制其函数体、YAML数据库、skill或CUDA资产。

## 实际读到的机制

- `scripts/query.py`的`load_alias_expansions`将别名归一到canonical，`score_keyword_match`对同一keyword各variant取最大值，避免同义词重复加分；过滤和正文评分分开。该版本对alias文件解析异常会回退空表，本地受控词表改为显式报错，不静默失去约束。
- `references/schema.md`区分source/wiki页面、confidence与reproducibility，并将controlled vocabulary指向单一`data/tags.yaml`，没有在说明页再维护值列表。源码/official组合不等于本机Mac性能资格，本地不继承其verified/snippet语义。
- `data/aliases.yaml`含CUDA instruction和GPU型号的特定归一规则；本地不移植这些硬件映射。NAX≠ANE、FP8≠Q8、GGUF容器≠量化格式，M4/Pro/Max不互相授予性能profile。
- `data/version-claims.yaml`将结论、版本、复核日期、source IDs与适用页关联；`scripts/check_version_freshness.py`只读committed输入，分开检查日期窗口、release receipt和过期候选。它明确不能发现receipt日期之后的新release。本地只借离线复查合同，不把“日期新”当自动正确，也不伪造版本区间。

上游入口：[query](https://github.com/mit-han-lab/KernelWiki/blob/b6b4301f15e8ce6955a56776690643ce5db369e6/scripts/query.py)、[schema](https://github.com/mit-han-lab/KernelWiki/blob/b6b4301f15e8ce6955a56776690643ce5db369e6/references/schema.md)、[freshness checker](https://github.com/mit-han-lab/KernelWiki/blob/b6b4301f15e8ce6955a56776690643ce5db369e6/scripts/check_version_freshness.py)。链接供出处核查，离线核心不需要上游checkout。

## 迁移到Apple Silicon知识库

继续使用`labs/catalog.json`和`references/sources.json`作为页面登记，新增`data/vocabulary.json`只存受控词表/单跳别名，`data/version-claims.json`只存绑定已有ID的复查声明。由同一登记生成`queries/README.md`，不另造第二张章节清单。

查询标签只描述内容主题；空列表为未分类，不能作“所有设备通用”。原始evidence保留自身来源/失败状态，不从关联章节继承已验证标签。exact context匹配仅显示recorded-context-only；缺上下文、超龄、源码/证据hash变化或环境不同都可见，永不授予runtime/质量/性能资格。完整实现与反例见[查询合同](../../data/README.md)。
