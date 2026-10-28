# DiCache：在线浅层probe、恢复执行与轨迹对齐

<!-- knowledge
id: technique-algorithms-methods-dicache
type: method
stage: dit
operator: sampling, attention
backend: cpu-reference, mlx
device: cpu, gpu
symptom: stale-cache, precision-loss, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [FBCache/DBCache](block-caches.md) · [预测状态合同](../case-studies/forecast-state-audit.md) · [实验](../../examples/README.md)

2026-10-16第四轮。固定官方仓库[Bujiazi/DiCache](https://github.com/Bujiazi/DiCache/tree/fdbe20b669c9174bbed5ec994de073fd881c8010)，阅读FLUX、HunyuanVideo、Wan三个adapter的控制段，逐文件hash和范围见[S38](../sources.md#s38)。论文入口：[DiCache 2508.17356](https://arxiv.org/abs/2508.17356)。本页机制以所读实现为依据，不拿论文速度表推导Apple收益。

## 与TeaCache/FBCache相比，省算和信号不同

当前step仍做embedding、time/text准备、前m个transformer blocks的probe，以及最终norm/projection。probe后预测是否需要剩余blocks；如果不命中，直接从probe的image/text状态继续，避免把前m层再算一遍。若命中，则回到本步原始embedded image输入，加一个旧/预测的**整段residual**，不是在probe输出上再加整个residual。

因此“命中”仍付浅层attention/FFN的代价；相比timestep调制信号，probe更接近真实网络，但不是免费的error oracle。若m接近总层数，剩余可省计算很少。

令x为本步embedding、p为probe输出，代码计算相邻观测的相对L1变化δx、δp；FLUX选择累计δp，或累计|δp−δx|。低于阈值才skip，refresh时归零。δp反映的是浅层信号，不是被省深层残差的严格误差上界。

## 多历史轨迹对齐不是通用时间导数

保留full时的整段残差R和probe残差r。历史不足两份时用上一R；否则用当前probe残差相对前两份r的L1距离比得到gamma，再用两个历史R线性组合。gamma是幅度比，经过clip；不是带方向的最小二乘系数，也未显式除以full-anchor时间间隔。

probe方向反转、浅深轨迹解耦、历史间隔变化或分母接近0，都需要独立对照。clip限制horizon不等于finite保护：NaN经过clip仍可保持NaN。FLUX代码只取最近两份计算，但list会append所有full历史直至结束；Wan/Hunyuan的list先到3份，随后滚动最近两份，不能把三者描述成同一个内存合同。

## 三个adapter必须分别阅读

| 合同 | FLUX | HunyuanVideo | Wan |
| --- | --- | --- | --- |
| 计数单位 | 每次forward | 每次forward | 每次forward；配置为采样步数×2 |
| 状态分流 | 单套 | 单套；外部CFG调用需另核对 | cnt%2分离两套，依赖调用交替顺序 |
| 前段保护 | cnt≤floor(ret_ratio×steps) | cnt<floor(...) | cnt<floor(...) |
| 末次保护 | 明确强制full | 所读控制段无相同强制 | 所读控制段无相同强制 |
| gate边界 | 累计error<threshold | 累计δp≤threshold | 累计δp<threshold |
| gamma范围 | [1,1.5] | [1,1.5] | [1,2] |

配置里的ret_ratio、num_steps和阈值不能跨adapter照搬。Wan的奇偶状态不是通用branch identity；CFG换序、合并batch、重入、取消和新请求仍需明确reset。Hunyuan若把ret_ratio设0，首步就可能进入需要旧状态的probe段；本轮仅静态指出前提，没有执行该adapter或其CFG pipeline。

## 真实FLUX forward body的34行CPU接缝

dicache_flux_probe.py先核对完整源文件digest，再只提取`dicache_forward`，不执行顶层模型下载、class monkeypatch或CUDA。Torch2.14.1、显式CPU/FP64。替身为四个out-of-place加法image blocks、恒等text/embedding/norm/projection；真实attention、ControlNet网络、PEFT、checkpoint与scheduler均未执行。

逐步完整oracle由独立的四层加法公式和disabled-cache原forward双重检查。完整trace在，包含异常数值的结构化记录，未把NaN写成非法JSON或改成成功。

脚本return0表示观察完成，不表示所有oracle通过；首次非finite以`actual:null`和`nonfinite_kind:"nan"`保留，随后停止该分支，不能把null解读为缺测或0。

| 控制 | 实际观察 | 解释范围 |
| --- | --- | --- |
| 不带ControlNet，阈值0，8步 | 全部匹配完整oracle；每步四层各一次 | probe后refresh恢复没有重复执行浅层 |
| 带四份不同ControlNet加法sample | warmup/末步匹配；中间resume比oracle少40 | 此微型输入可到达接缝不等价，不代表真实图像回归已测 |
| 恒定probe残差，两份以上full history | step3首次hit产生NaN；只执行probe层 | gamma的0/0未被clip修复；首次非finite后停止，不继续传播 |
| 只有一份history的恒定轨迹 | 四个hit都finite并匹配完整oracle | 单历史replay控制，不能因此保护多历史分支 |
| probe随步线性增加，gamma被clip | hit误差0.5/1.5/2.5/3.5，末步full归零 | 展示近似/阻尼取舍，不要求预测天然精确 |

ControlNet接缝有两个独立原因：probe循环没有注入相同ControlNet residual；resume对剩余block重新从0编号，后续sample索引未加probe_depth。控制里原输入5、block总加10、ControlNet总加100，完整结果115，resume得到75。这里所有四层都实际执行了，不能把这个偏差归因为“近似命中理应有误差”。

## Apple迁移的可用机制与成本

probe/gate/residual组合没有CUDA专属数学依赖，可研究在MLX中拆prelude/probe/rest/tail；full路径复用probe状态，缓存只提交成功的owned history。不能直接导入上游CUDA launcher当成MPS支持。

主要成本是每step前m层、reduction/gate的host同步、两套CFG历史、克隆以及full-residual读写。应同时比较：无hook的优化dense、启用但零命中的DiCache、单历史replay、多历史alignment。记录actual blocks、probe/resume次数和history bytes，不能只报cache hit率。

移植时建议另开A/B：近零分母/非finite时保守full或明确replay；使用全局block编号与完整ControlNet依赖；把request/CFG/shape/step坐标纳入状态身份，并将取消/异常设为不提交。这里没有替上游实施这些修改，不重写原负例为已修复，也没有新Apple性能/媒体质量资格。

默认离线tests只校验trace、hash和公式，不运行上游代码/GPU。未来需要实际模型、多prompt/seed/CFG/ControlNet与完整请求验证；源码存在与数学可迁移不是部署结论。

## 第四轮交付核查

算法专项49项通过，其中13项为第四轮新增；独立标准库整理器重算k4逐行统计与replay统计，并拒绝改标NaN、ControlNet偏差、source identity、head/witness缺失和score-on-hit计数。当前来源39条/112个文件digest，其中本轮新增DiCache四文件直接取固定Git blob核验，旧108个digest保留而非声称全部重新下载。
