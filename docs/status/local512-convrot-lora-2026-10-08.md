# ConvRot512²：真实 LoRA、共享旋转和 MPP/F32 混合接续

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。上一轮
[MPP base盈利](local512-convrot-mpp-partial-2026-10-08.md)已改变真实模型
状态与下一步决策，本轮继续完整目标，不缩成单纯兼容接口。只读本地
Z/Qwen、adapter和参考工程，没有下载、模型改写或dense sidecar。
完整两模型base/LoRA、1–2参考图、encoder、双后端、≥1.2×及严格/广泛
质量目标仍未完成；本轮带LoRA只有小幅诊断盈利，默认继续关闭。

## 完整 ConvRot LoRA 路线，不是只开门/漏掉前缀

新的 `TURBOCIDER_Z_RUNTIME_CONVROT_LORA=1` 默认0。共用计划/运行
predicate要求resident512²、明确runtime GPU+ANE、approximation、
inference_time LoRA、Private固定非零channels、Comfy W8A8/F32 join，
原legacy BF16 scales与hidden ABI；software BF16-value graph、FP32
scale override、Public/auto/non512/merge/guard/streaming拒绝。普通GPU
的有效flag忽略，非法值仍拒绝。原base无需新flag，默认路径不改变。

Private Comfy执行器已经具有F32 base输出与独立BF16/FP16 LoRA hidden
接口。本轮将已有**完整**模型回调接通该显式source route：GPU产生
pre-SiLU gate/up LoRA correction，ANE只读base codes/scales；GPU
complement带对应gate/up adapter，恢复两边corrected hidden后，再对
完整joined hidden执行一次down-LoRA。attention/QKV及其它adapter投影
继续GPU。没有B@A merge、把LoRA写进W bank、删除alpha/负adapter，
或以旧`lora_suffix`不完整路线当加速成绩。

原MPP selector另要求这个LoRA opt-in，选定profile仍snapshot到当前
Weights；base-only与LoRA source bank保持独立。stage失败或晚chunk失败
仍完整GPU FFN重算，不发布半成品。新marker描述实际选择，不当作
physical INT8 MAC/overlap或low-rank kernel计数。

## GPU helper：LoRA 也共享 frozen-base 的 H256

此前`Weights::project_many`遇见任何runtime adapter便退回独立projection。
现在复用一个只处理adapter的共同实现：冻结ConvRot base共享同一XH；
每个LoRA仍使用**原未旋转X**，依原顺序做FP32/既有可选FP16 ranks、
每adapter output narrowing和最终bias。不是把LoRA旋转或合入base。
`project`的数学顺序不变；mixed/nonConvRot列表仍用原独立projection。
Z的既有显式 `TURBOCIDER_Z_CONVROT_SHARED_GATE_UP=1` 才选择shared helper。

72个实际MLX/Metal用例：三dtype、raw/packed g32/g64、dense/Metal H256、
两rank精度、M1/67、stacked正/负强度及BF16 alpha、bias、partial fused
QKV和完整FFN。single/shared输出按bytes一致，masters id/bytes不改。
随后追加独立已知A/B、alpha和per-adapter舍入oracle，防止两个路径
共用同一错误helper却互相“通过”；仍通过，不放宽等价门槛。

## 真实八步 LoRA：MPP 原混合约12s →9.1s，较GPU小幅快

同Private v1 library，本地原ConvRot及原151MiB distill-patch rank32
adapter，strength1，fox/seed42，512²/八步。每请求238个真实投影，
base保持packed，不生成derived adapter。Fa4096/Fg6144、fixed-async/
F32 join，stage-specialize/launch-fence1，prefetch/lookahead/W-code cache0。
敏感ordinal2用既有完整GPU block policy，两混合臂相同；它不是fallback。

GPU是完整source/LoRA、shared-rotation优化控制，不是base无LoRA分母，
不设compiled-dense GPU recipe（既有草稿该recipe仅base可用）。两个
hybrid arm也使用同shared flag，MPP唯一算子变化仍是base-down F32核。
各独立进程一冷两同prompt热请求，conditioning false/true/true。
100ms process-tree memory采样，无strict-load资格、无dump。

| order | complete GPU warm s | original F32 hybrid s | MPP hybrid s | GPU/MPP |
| --- | ---: | ---: | ---: | ---: |
| GPU→original→MPP | 9.520813 | 11.995366 | 9.151589 | 1.04035× |
| MPP→original→GPU | 9.507782 | 11.989949 | 9.112708 | 1.04335× |

MPP比旧F32路径减少约23.7–24.0% wall，比完整GPU只减少约3.9–4.2%。
是两方向有限小样本诊断，不是1.2×、统计/多scene或默认选择资格。
cold GPU/original/MPP：正向10.622371/13.381750/10.587852s；反向
10.537796/13.237226/10.303101s。全部样本保留，不合并有利分母。

每请求248 actual Private calls、248 successful channel blocks和248个
narrow LoRA correction callbacks；累计248/496/744。完整GPU policy
blocks累计8/16/24，full correction callbacks0。两臂headroom1、
failure/fallback/retry0，不借用上一轮base的cold-retry-cap2。六个memory
reports complete、system swap-in/out0；MPP/original peak约19.29/19.28GB，
GPU约19.46GB。scope为load/cold/warm/exit进程树，不含外部服务/driver
归因，未放宽4GiB reserve/2GiB optional tier、load/memory或数值gates。

目录 `outputs/local512-convrot-lora-v1-{forward,reverse}-diagnostic-20261008/`。

## 画面与同图状态切换

两方向每组三张original/MPP均bytes exact，两方向也同hash。MPP为
`d259e962d5c80ef480cf464a4de1e768752c1e2b478df96e5d9f135d71ff4062`；
完整GPU为
`e50a74b9e8e829f699d87e834feecf27816f9d01ed2411715b90b41ab10ca542`。
已目视GPU/MPP及original/MPP两组whole和各三同坐标裁剪：狐姿态、
耳/脸、毛色、尾巴、雪和构图很接近，眼周表情、毛发/雪细纹略变，
未见新增断裂/色块。第一组original sheet左侧固定GPU标题实际是
Private control，未误称GPU。仅此scene/seed有限观察；visual manifests
保持pending，原source/ANE N1失败不改写，不当作用户批准。

新增同一进程base→**原**LoRA1→**同**LoRA.5→base，没有生成/替换adapter。
248/496/744/992 actual calls，load_seconds始终0.022637792、同一个
admitted executor，headroom1及zero failure/retry/fallback。correction
累计0/248/496/496，binding0/238/238/0。首尾base PNG exact、两个strength
输出不同，恢复证明不是只看lora selection label。

base hash `de07fcac207a52fdb8a213108990d2ddaef9f6e6b35ad86d0d4ee9dc8edd1887`；
strength.5 hash
`e149ac8b3a50fe5b51906d8fb1cedfdbca9c40a64b7e82be42b406c8f39b2f0c`。
state memory report complete、peak约19.28GB，scope仍不含外部服务/driver。
状态切换不当作speed/第二个训练adapter或用户画质资格。
目录 `outputs/local512-convrot-lora-v1-state-switch-diagnostic-20261008/`。

## 回归、构建、保留

Private16、Public29项选定native/default/LoRA/encoder/receipt回归通过，
无skip；另最终独立oracle＋host state/forged-counter7项通过。实际
Private direct Comfy driver raw/packed × serial/lookahead四格通过，含
F32+LoRA hidden bytes、拒绝widened-hidden、正/负/零down scales、future
basis、晚期失败及健康refill。首次probe误用系统symlink `/tmp`被原
cache安全检查拒绝；改canonical `/private/tmp`后通过，未放宽安全检查。
不是全仓、Public实模型或更多LoRA/多prompt验收。

Private实验、Public普通native-only builds exit0；Public actual发行
class/flags/link guard通过。两库各493source input hashes匹配验证时
工作树，包含原ConvRot草稿；selective commit不夹带原草稿，不冒称
clean staged-only或稳定App发布。

```text
Private 8eedfe6b13d6dce4f6682aabb1293fa8beb1c16e03967d7d62628ce840d09182
Public  0cddcef7dbb505971c6b9f93bab447ed24e34a27dd3a6b96ef405fa50c952a32
```

完成全部owned jobs后清理425个`.o`、33,127,816 logical bytes、两个空
module-cache，以及自建temporary driver cache的12个文件/1,083,000
logical bytes，均可重建。保留库/CLI/probe、logs、manifest、PNG、原
models/adapters，未清理外部服务或用户缓存。原negative logs亦保留。
相对路径机器汇总见
[本轮证据](../design/validation/local512-convrot-lora-20261008.json)。

接续仍需LoRA请求local compiled FFN/共享rank/handoff进一步真实盈利，
BF16 Z/Qwen base与真实LoRA同库复测、Qwen 1–2图/encoder、Public实模型
与不可变按operation backend选择、GGUF真实decode/retention审计、更多
scene/seed与有效strict窗口。已有GGUF R8负结果及Qwen未盈利结论仍有效；
不能将本轮ConvRot LoRA的1.04×套给Qwen或全部格式。
