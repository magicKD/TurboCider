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

## Test-only public catalog campaign

带 `TURBOCIDER_BUILD_TEST_HOOKS=1` 构建的 dylib 可以在 native campaign variant 中显式安装一个
engine-scoped immutable catalog：

```json
{
  "backend": "native",
  "library": "/absolute/path/to/test-hooks/libturbocider.dylib",
  "model_id": "z-image-turbo",
  "model_path": "/absolute/path/to/model",
  "constructor": "public",
  "test_streaming_catalog": "/absolute/path/to/reviewed-test-catalog.json"
}
```

catalog 外层合同为：

```json
{
  "schema": "turbocider-streaming-test-catalog-v1",
  "revision": "与每条 record.catalog_revision 完全一致",
  "records": ["完整 canonical StreamingPresetRecord"]
}
```

runner 会把 catalog 的路径、SHA-256 和大小写入 `build-identity.json`，并在每次 engine 创建后、生成请求前
调用 test-only installer。该字段只允许 `constructor=public`；release dylib 没有 installer symbol，因此无法
通过配置或环境变量获得测试 authority。此机制只用于 public-semantics calibration/evidence，不会修改
production catalog，也不能替代 catalog builder、review、P0/P1/P2/P3 或 release gate。

test catalog 必须由真实 public probe 和模型 adapter 生成，不允许手写 source/runtime/layout digest。示例命令：

```sh
python3 -B tools/native/build_test_streaming_catalog.py \
  --library /absolute/path/to/test-hooks/libturbocider.dylib \
  --model-id z-image-turbo \
  --model-path /absolute/path/to/Comfy-Org-z_image_turbo \
  --request /absolute/path/to/zimage-10g-request.json \
  --plan /absolute/path/to/zimage-p7-k2-plan.json \
  --target-gib 10 \
  --catalog-revision tc-zimage-10g-public-calibration-r1 \
  --output /absolute/path/to/zimage-10g-test-catalog.json
```

`request` 必须是 schema-v2 `selection=memory_tier` public selector；`plan` 只包含
`canonical_config`、`pass_transition`、`multi_pool_policy`。输出中的 calibration/performance/review 是明确的
test template，只负责使同一真实 adapter 能走完整 public resolve/generate transaction，不具备 production
发布资格，也不能跳过 evidence builder。

Z-Image 10 GiB 的完整请求与 P7/K2 计划样例分别见
[`zimage-10g-memory-tier-request.json`](zimage-10g-memory-tier-request.json) 和
[`zimage-p7-k2-plan.json`](zimage-p7-k2-plan.json)。示例中的 `${OUTPUT}` 只用于 campaign 模板替换，不能
直接作为 App 的最终输出路径。

public-semantics audit 需要同时启用 audit counters 与 test hooks 的独立 dylib，并安装完全相同的 catalog：

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=/private/tmp/tc-public-audit \
TURBOCIDER_BUILD_AUDIT_COUNTERS=1 \
TURBOCIDER_BUILD_TEST_HOOKS=1 \
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh

python3 -B tools/native/run_streaming_audit.py \
  --library /private/tmp/tc-public-audit/libturbocider.dylib \
  --model-id z-image-turbo \
  --model /absolute/path/to/Comfy-Org-z_image_turbo \
  --request /absolute/path/to/resolved-public-request.json \
  --constructor public \
  --test-streaming-catalog /absolute/path/to/zimage-10g-test-catalog.json \
  --expect framework-active \
  --output /absolute/path/to/zimage-10g-public-audit.json
```
