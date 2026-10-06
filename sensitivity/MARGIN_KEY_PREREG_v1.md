# expert 置信度是否键的信息来源 · 预注册 v1（margin 当键）

日期：2026-10-05。前置：`CAPTURE_DRIVER_FINDINGS_v1.md`（AUROC_conf ↔ AUROC_s2 0.526）。
标准温度缩放是空操作（单调→argmax 不变→capture 不变），故改测"margin 直接当键"。

## 问题
expert 置信度（margin）是不是键（ratio2 的 s2）判别力的**因果信息来源**？

## 方法（每港，诊断用 oracle）
- 键 A（现有）：s2 = robust_scores(ratio2, base, expert, observed)；
- 键 B（新）：expert margin = 分歧行上 expert logits 的 top1−top2；
- 各键排序分歧行取 top-k（5%），算 capture；
- 另算 AUROC_conf、AUROC_s2（同 CAPTURE_DRIVER）。

## 判据
1. capture_B 与 capture_A 高相关（Spearman>0.7）⇒ expert 置信度是键的信息来源（假设A，非混杂）；
2. capture_B ≥ capture_A（均值）⇒ margin 键更简单且不劣，方法贡献；
3. 若 capture_B 与 capture_A 不相关 ⇒ 0.526 是混杂假象，假设B成立。
