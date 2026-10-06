# 真链跨域（PACS）· 预注册 v1 —— 隔离"协议/键"因子

日期：2026-10-05。动机：之前 PACS/FMoW/Office-Home 的"键不迁移"结论，是用**简化替身**
（ridge + 朴素 pair-key）测的，从没跑过主线真链（ratio2 键）。此实验把真链原样搬到 PACS，
其余因子全冻结，唯一变量 = 协议/键。

## 因子控制
- 特征：复用 `xdom_pacs/featsA.npy`（R50-2048）、`featsB_{域}.npy`（源域监督 R18-512）——与替身实验同源；
- LOPO：4 域，与替身同；预算：20%，与替身同；
- **唯一变量**：键 = 真链 ratio2（shared_cov → class_mass → project_class_evidence → robust_scores），替替身 pair-key。

## K=7 泛化（忠实照搬冻结配方，仅 K 参数化）
- 决策特征 = Helmert(7) 对比 [base|expert] logits → 12 维（冻结为 helmert(8)→14 维）；
- fit_source：12 维、7 类、OAS 共享协方差；shared_cov predict：K/d 自适配（冻结函数本身已泛化）；
- class_mass / robust_scores：2^7 支持集枚举（冻结 2^8）。

## 臂与判据（每域）
base / expert / rank（真链 robust_scores 排序，20% 预算）/ rand（随机）/ thr（源侧分位）。
- P1 = expert ≥ base；P2 = rank ≥ rand；P3 = rank > thr（**在支配域上**算，与 PACS 口径一致）；
- 报 macro BA 与加权 ACC 两列。
- **判读**：若 rank 跨域显著 > rand（P2 在支配域 ≥ 1/2 且差值 >0），说明真链键跨域成立、替身伪影；
  若 rank ≈ rand（P2 低），则键跨域失效成立，边界为真。

## 边界
- 不用目标标签（rank 臂用 64 标签 + 源侧似然 + 目标未标注行的 EM 投影）；
- 若某域 expert < base（无支配），该域不计入 P2/P3（无交接空间）。
