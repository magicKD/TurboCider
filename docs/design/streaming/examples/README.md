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

request 片段需与模型、输入、输出、sampling 合并才是完整请求。示例 P=1 用于清楚表达参数，不是性能推荐。
profile 名称/机器匹配值是演示，不允许通过更改字符串给别的机器授予认证。
字段含义、优先级及冲突见 [配置规范](../02-configuration.md)。
