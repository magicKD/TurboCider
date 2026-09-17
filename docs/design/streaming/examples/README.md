# 配置与验收示例（模型执行尚未放行）

这些文件放在docs而非production profiles目录。当前已支持schema2 request的`execution.streaming` plan-only解析及profile v2合并。
通过parser不代表相应checkpoint/layout已获执行授权，实际实现范围见 [13](../13-implementation-progress.md)。

| 文件 | 用途 |
|---|---|
| [manual-ltx.json](manual-ltx.json) | request execution 片段：精确三槽、不设全进程预算 |
| [bounded-ltx.json](bounded-ltx.json) | 完全相同布局，加 Y=20 GiB、buffer=15% 的 guard |
| [resident-ltx.json](resident-ltx.json) | 框架显式 resident，非 legacy 自动回退 |
| [profile-v2.json](profile-v2.json) | 未来 profile 的完整示例，同一 StreamingConfig 结构 |
| [compiler-golden.json](compiler-golden.json) | 合成编译器测试输入及手算期望值；不是请求配置 |
| [performance-policy.json](performance-policy.json) | P0/P1/P2验收阈值与执行前必填项；不是可执行campaign |
| [ltx-p1-same-layout-policy.json](ltx-p1-same-layout-policy.json) | LTX 冻结 private same-layout P1 campaign |
| [z-image-p1-same-layout-policy.json](z-image-p1-same-layout-policy.json) | Z-Image 冻结 private same-layout P1 campaign |
| [h3-p1-same-layout-policy.json](h3-p1-same-layout-policy.json) | H3 Turbo 原始 BF16 P0/G1/K2/D1/Q1 campaign |
| [flux9-k2-q2-request.json](flux9-k2-q2-request.json) | Flux.2 Klein 9B private candidate 的 retained 双 class K2/G1/D1/Q2 两步请求；不是P1 policy |
| [flux9-p1-same-layout-policy.json](flux9-p1-same-layout-policy.json) | Flux.2 Klein 9B 同布局 direct/generic P1 campaign；正式20-pair已PASS，逐请求校验implementation身份 |

request 片段需与模型、输入、输出、sampling 合并才是完整请求。示例 P=1 用于清楚表达参数，不是性能推荐。
profile 名称/机器匹配值是演示，不允许通过更改字符串给别的机器授予认证。
字段含义、优先级及冲突见 [配置规范](../02-configuration.md)。

Flux 示例只能通过 private candidate constructor 执行，public route 仍 fail-closed。单请求文件用于重现功能、
质量、内存和audit证据；P1 policy的baseline使用仅供benchmark的同布局direct replay，candidate使用通用
`StageExecutor`。两侧必须保持完全相同的P0/G1/K2/D1/Q2、双pool retention、reader同步边界和pager；仍不能
拿resident与streaming的耗时差直接签成framework P1 overhead。
