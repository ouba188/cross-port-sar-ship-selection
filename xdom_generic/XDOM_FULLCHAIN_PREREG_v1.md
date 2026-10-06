# 真链跨域复验（FMoW / Office-Home，K=8 前 8 高频类）· 预注册 v1

日期：2026-10-05。前置：`FULLCHAIN_PACS_PREREG_v1.md` —— 真链（ratio2 键）在 PACS 跨域成立（P2 2/2）。
此复验把真链搬到 FMoW（跨 5 区域）与 Office-Home（跨 4 域），验证其跨域性是否普遍。

## 为何 K=8（想清楚再动手）
- FMoW K=62 / OH K=65 下，robust_scores 为 2^K 枚举不可行，且 64 标签对 62/65 类 ≈ 每类 1 个、
  shared_cov 类均值被 TAU=8 先验淹没。故**不搬全 K**；
- 取每数据集**前 8 高频类**（确定性、非挑拣），K=8，与主线/PACS 同 K ⇒ 干净复刻；
- 唯一变量 = 域（区域/域）+ 键（真链 ratio2）。其余全冻结（同 feats、同 LOPO、64 标签、20% 预算）。

## 臂与判据（每折）
base / expert / rank（真链 robust_scores 排序，20%）/ rand / thr（源侧 80 分位）。
- P1 = expert ≥ base；P2 = 支配折上 rank ≥ rand；P3 = 支配折上 rank > thr；
- macro BA 与加权 ACC 两列；主读数 = P2（真链键是否跨域排序优于随机）。

## 边界
- 64 标签 + 源侧似然 + 目标未标注行 EM 投影，不用目标标签；
- FMoW 视图 B = 采集元数据（6 维）；OH 视图 B = 源域监督 R18（PCA-64）。
