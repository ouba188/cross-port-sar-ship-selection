# FMoW + CLIP 视图 B · 预注册 v1（第二视图须真互补才能检验键迁移）

日期：2026-10-05。动机：FMoW 元数据视图太弱（expert−base ≤1.25pp）、Office-Home 的 R18 全负，
都"无头寸"退化；要干净检验"真链键跨域成立"的边界，须配一个**真正互补**的第二视图。
CLIP（图文对齐、互联网语料）与 ImageNet-R50 训练信号不同 → 互补头寸应明显大于同家族 backbone。

## 设计（其余全冻结，与 XDOM_FULLCHAIN_PREREG 同）
- 视图 A = ImageNet R50（`featsA_fmow_sub.npy`，已缓存）；
- 视图 B = CLIP ViT-B/32 图像编码 512 维（对同一 27,557 子样本提取）；
- K=8 前 8 高频类、5 区 LOPO、主线真链（ratio2 键）、64 标签、20% 预算。

## 判据
- **先看头寸**：P1 = expert ≥ base 的区数（若 CLIP 也无头寸 ⇒ 记"无头寸"负/退化，不冒充键失败）；
- **键迁移**：P2 = 支配区上 rank ≥ rand、P3 = rank > thr；主读数 = P2 且 rank−rand 的差值。

## 边界
- CLIP 权重经 openai CDN 下载；若 CDN 不通，回退 torchvision ViT-21k（pytorch CDN，已知直连），
  但回退会减弱"真互补"前提，须在结果里如实标注。
