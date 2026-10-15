# 首版知识库验证记录（历史）

[总目录](../README.md) · [覆盖审计](coverage.md)

日期：2026-10-15。基于TurboCider `ad8100f` 开始；仅新增知识库/脚本/测试与README导航，没有改变模型、runtime、产品默认、用户服务或第三方checkout。

以下42篇/33项是首版的历史验收范围，不重标成后续扩展成绩；新增实现实验与当前测试见 [扩展验证](labs-validation.md)。

## 最终结果

| 检查 | 本轮实际终态 |
| --- | --- |
| wiki结构/来源检查 | 42篇Markdown、472个本地链接、67个来源文件、7个外部关键文件SHA256；0errors/0warnings |
| 新工具CPU回归 | 33tests全部通过、0skip；Python3.9.6与项目Python3.11.16分别运行同一suite，不重复计为66项 |
| FFN代数示例 | channel拆分maxabs约1.39e-17、row拆分0；包含完整LoRA与不均匀cuts |
| 统计示例 | 实际输出ABBA、2个syntheticwarm pairs、描述性倍率；`qualification_passed=false`保留 |
| phase预算示例 | backingalias去重，decode声明峰值4,462,739,456B；`synthetic=true/hard_memory_bound=false`保留 |
| 既有ANEhost回归 | 选定5tests通过、0skip：rowscheduler、耦合costmodel、calibrationtiming/error-drain、selection/weaksourcecache、graphgeometry |
| 既有boundedpread原生fixture | 现源码临时编译运行PASS：两槽/alias/range/owner/errorpoison/cancel/destructiondrain；非GPU实验 |
| 既有SeedVR2phase回归 | 3tests通过、0skip：完整trace、非法history/peak拒绝、inclusive不重复相加 |
| 历史索引smoke | 对既有日期子目录重新只读扫描3个匹配session，0readfailures，未导出正文 |
| 清理/变更范围 | Git仅2个根README导航变更和51个techniques新文件；无binary/权重/媒体/rawsessions |
| 空白检查 | trackedREADME的`git diff --check`通过；新目录文本trailing-whitespace扫描无匹配 |

所有上述进程/编译测试均已正常终止，没有留下本轮后台模型任务。CPU临时fixture由测试自身回收，没有删除既有用户产物。完整覆盖审计已按原始要求逐项走查；桌面新模型演练只证明手册可指导设计，不虚称已加速一个新模型。

## 可复现命令

从仓库根目录执行；以下均无ML/GPU依赖：

```sh
python3 -B techniques/scripts/check_wiki.py
python3 -B -m unittest discover -s techniques/tests -v
.venv/bin/python -B -m unittest discover -s techniques/tests -v
python3 -B techniques/examples/split_ffn_reference.py
python3 -B techniques/scripts/compare_runs.py techniques/examples/paired-runs.json
python3 -B techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
```

以下是本轮实际选定的现有源码回归，不等于完整`make test`：

```sh
python3 -B tests/native/test_ane_runtime_host.py \
  RuntimeHostTests.test_row_scheduler_alignment_disable_reprobe_and_isolation \
  RuntimeHostTests.test_channel_bandwidth_fit_and_memory_constrained_selection \
  RuntimeHostTests.test_shared_gpu_calibration_order_samples_and_exception_drain \
  RuntimeHostTests.test_native_channel_identity_trial_gate_and_weak_source_cache \
  RuntimeHostTests.test_projection_layout_and_validation
.venv/bin/python -B tests/native/test_read_ahead_worker.py
.venv/bin/python -B tests/native/test_seedvr2_vae_phase_summary.py
git diff --check
```

其中host/nativefixture需要本机clang/SDK，输出置于测试私有临时目录；没有CoreML实际prediction、GPUdispatch或模型fixture。historysmoke的sessions路径仅由调用者本地提供，不保存个人路径到手册。

反向测试明确覆盖坏链接、孤页、optional来源缺失/必需来源缺失、snapshot漂移、源码符号缺失、缺配对/重复/混build、失败或unknownquality、NaN/极端单位、aliasbytes不一致和reviewer上下文过滤。链接/案例没有用一次绿色正常样本替代这些拒绝路径。

## 结果解释

本轮脚本是知识复用/教学与离线审计工具。代数demo不模拟BF16/W8A8；内存planner只算声明的serializedphase；统计脚本不验证原报告的质量/hash/cache事实；链接检查不验证语义/HTTP/anchors。绿色结果不能替代完整模型、真实ANEplacement、速度/画质/低RAM或App资格。

本轮未运行大型模型、未重做历史GPU/ANEbenchmark、未下载/转换/删除权重或媒体、未改系统cache/wired/swap/电源参数、未终止其他应用、未替换App、未提交或推送Git。来源和历史结论见 [证据地图](../references/evidence-map.md)。
