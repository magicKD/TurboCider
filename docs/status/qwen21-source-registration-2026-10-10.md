# Qwen：首次 SHA 注册，推理复用

2026-10-10。按用户要求，完整模型 SHA 不再是每个新进程的固定推理
成本。Qwen GGUF encoder/DiT、verified tokenizer、实验 LoRA 和原 BF16
streaming source 默认使用首次注册：第一次完整读取 SHA，之后跨进程
复用 native 创建的本地记录。已有 session 内的 packed weights/tokenizer
复用不变；没有修改量化表示、kernel、精度或误差门槛。

## 运行合同

- 每次打开原模型并持有 fd；检查 normalized named path、canonical target、
  device、inode、size、mtime、ctime，推理/导入后的 fd/path 复验仍保留。
  原地同大小修改（即使恢复 mtime）、替换、改大小、软链接换目标都会
  使旧注册记录失效，重新校验；请求中源变更仍直接拒绝，不发布候选 bank。
- 记录只有首次 native 全 SHA 和所有源/expected digest/取消检查成功后
  才可发布。输入的 `SourceFileIdentity.content_digest` 仍只是待验证的
  expectation；普通 metadata/replay 不会升级为内容 proof。
- 注册记录带版本和完整文件 generation 绑定，并由随机本地 key 做
  HMAC-SHA256。目录必须为本用户 0700，key/lock/record 为 0600，拒绝
  缓存目录或记录软链接、其他 owner 和不安全权限。文件锁保护短事务，
  临时文件 + fsync + atomic rename 发布，没有在完整模型 SHA 期间持锁。
- 信任边界是本机本用户的 native 注册缓存，不是对恶意同 uid 程序/root
  或可伪造 stat 的文件系统提供防篡改认证；不等价于第三方模型签名。
- 缺失、截断、伪造、错误 key、失效 generation 会重新 SHA；不可写或
  不可信的缓存安全退回原 process-local 校验，不接受任意外部 digest。
  清除缓存不会破坏模型，只会使下次再次完整校验。

默认缓存位于本用户 `Library/Caches/TurboCider/source-registrations-v1`。
可用 `TURBOCIDER_SOURCE_REGISTRATION_CACHE` 指向指定的绝对路径私有目录。
`TURBOCIDER_QWEN21_SOURCE_REGISTRATION=0` 保留每个 fresh process 完整
SHA 的对照路径；默认 1。旧性能 runner 显式设 0，避免把过去的 full-SHA
合同悄悄改成注册命中；新 runner 区分 `process / first / reuse`。

`verification_bytes` 是真实本次 source verification 读取量，
`registration_cache_hits` 是持久化记录命中，
`registration_records_written` 是实际新记录。记录首次注册耗时、轻量
检查耗时和完整 import/request 耗时，首次请求总耗时仍包含首次注册，
不会通过减掉 hash span 制造 cold 加速。重启后的模型权重 import/read
仍然存在，零 SHA bytes 不是零模型 I/O；此项不是 GPU/ANE kernel 加速。

## 回归与保留失败

24 项独立 native process 测试包括：首次/重启复用、Python 独立 HMAC
验真、mtime 恢复后的同大小修改、替换、大小变化、软链接换目标、
缺失/损坏/伪造记录、不同 path/generation 的签名记录不串用、错误
expected digest、多源失败不发布部分校验结果、取消、pending metadata
不升级、held-fd 复验、缓存权限/key/软链接负例与四轮四进程首次注册。
最终同源码 ASan/UBSan 与 TSan 各 24 项通过；旧 SourceLease fixture
同样通过。所有原始失败日志保留。

首次并发压力暴露 Darwin 上缺失 lock 文件的 `O_CREAT|O_NOFOLLOW`
并发 open 可暂时返回 ENOENT，安全 fallback 后 warm 又发布记录，违反
测试的“不额外重写”约束。修为 explicit exclusive-create / existing-open，
验证 regular fd 后移除 NONBLOCK，正常持锁；同时允许认证已经被 atomic
replace 移除名字但仍由 fd 持有的合法旧 receipt。未降低 HMAC/源校验。
不是数值质量失败，未把原失败重写为成功。

## 真实原 GGUF：4 步预检

同一新库，512² / seed29 / 三个 fresh fox-light prompts，GPU 与 Private
W8A8 ANE5120/GPU7168 两路；prefill1、retention1、workers8、CPU verified
import schedule1。先 process 对照，再首次注册，再独立新进程复用。

初次仍完整读取 encoder 5,027,785,568 bytes 和 DiT 4,199,565,024 bytes，
每项实际新记录1，encoder full verify 约1.751–1.755s、DiT约1.464–1.466s。
这些 span 与 CPU import 重叠，不能简单从总请求扣除其相加值。

新进程两路 encoder/DiT `verification_bytes=0`、persistent hit1、record
write0、first registration seconds0。验证 worker 的真实轻量检查约
.136–.150ms；processor 同样0 SHA bytes、persistent hit1，GPU检查约
.124ms。权重 bank import 仍约 .588–.722s，不能说模型不再读取。

| route | 每进程全 SHA cold text / request s | 首次注册 text / request s | 已注册新进程 text / request s |
| --- | ---: | ---: | ---: |
| GPU | 2.224362 / 9.069972 | 2.084929 / 8.625019 | .976066 / 6.613268 |
| ANE5120/GPU7168 | 2.101773 / 8.206781 | 2.077775 / 8.063303 | 1.043916 / 6.160472 |

全 SHA/首次注册的差异包含首轮系统初始化噪声，不把首次注册宣称更快。
48 对完整 text/initial/latents/pixels safetensors 与 12 对完整 PNG 逐字节
相同，delta0。此轮不提供跨 engine 或 BF16 数值等价、更多场景编辑质量
或 public/hard-RAM 资格。512项 build source inputs 与测试库全部匹配。

完整 40 步计时在首次注册已经完成之后另外比较全 SHA 对照与新进程复用，
不重复首次注册，不从请求时间人为扣除 SHA。

## 同库 40 步实测

同模型 / 三 fresh prompts / seed29 / 512²，无 dump，GPU 和混合各一个
新进程，每进程一 cold、两 warm。首次注册由前述 4 步窗口完成，复用
记录与 steps 无关，不在 40 步中再次注册。全部 native/source/phase/
1280成功 Private calls 每请求的 receipts 与完整进程树验真。

| route | cold text s | cold request s | warm text s | warm request s | tree peak physical bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU 每新进程全 SHA | 2.079334 | 47.908525 | .090550 | 44.441606 | 15,211,175,448 |
| GPU 已注册新进程 | 1.010263 | 45.983923 | .090434 | 44.437596 | 15,513,837,104 |
| ANE5120/GPU7168 每新进程全 SHA | 2.084692 | 40.651904 | .093555 | 36.858041 | 15,491,194,584 |
| ANE5120/GPU7168 已注册新进程 | 1.013958 | 38.650411 | .093247 | 36.854592 | 15,416,434,368 |

冷 text 两路少约51.41% / 51.36%，完整 cold request 少约4.02% / 4.92%，
实际少1.925 / 2.001s。warm 没有新增 kernel 加速，差异仅约4ms。新进程
encoder/DiT/processor SHA bytes 全0，持久化命中各1、不重写记录；模型
验证 worker 约 .138–.181ms，processor约 .127–.137ms。依然实际 import
原权重，encoder约 .671–.678s、DiT约 .598–.610s。

40 步 6 对完整 PNG 精确相同。物理峰值变化+1.99% / -.48%，均在对照
±10%内，不是 hard RAM cap。四个进程树完整；系统 compression/swap-in/
swap-out 全0，decompression依次16,384 / 32,768 / 16,384 / 0 bytes，是
系统计数，不归因单个模型或 kernel。没有质量或 public qualification 升级。

此前已完成的原 Unsloth pinned `sd-server`（明确 GPU conditioner，原 flags，
40步、三相同 fresh prompts）记录保留，不重复重跑：cold condition1.18s，
warm condition .13s，warm sampling107.27s，native generation125.325s，
HTTP125.360592s。当前已注册的 native 新进程 cold text约1.010–1.014s，
此边界下更快；**首次未注册** native text仍约2.08s，不声称胜过1.18s。
native warm text约.090–.093s、DiT约43.736 / 36.151s、全请求44.438 /
36.855s，显著更快，但跨 engine 原 RNG/seed 不等于相同 noise 或图片，
不把性能结果称为质量 parity。此项也不是同内存 BF16 streaming 新窗口；
旧四路结果仍只按其原库/原窗口解释，完整优化目标保持 active。

可复现配置：`tools/validation/qwen21_gguf_hybrid_screen.py`，匹配 CLI
`build/qwen21-source-registration-v1-private/turbocider`，显式
`--channels 5120 --decode-workers 8 --encoder-prefill --verified-import-overlap`
及 `--source-registration process|first|reuse --registration-cache <private-root>`。
先4步含 dump，再40步无 dump；每臂有独立 cache 子目录。
库 native-only / private-enabled / experimental-probes0，512源 inputs全匹配。

[机器记录](../design/validation/qwen21-source-registration-20261010.json)汇总
源/库 SHA、首次注册/跨进程命中、完整 raw receipts、memory、精确输出、
回归、原失败日志以及历史参考计时。缓存 key/record 不进入 Git。
