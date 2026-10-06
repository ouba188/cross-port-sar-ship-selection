# capture 驱动挖掘 · 预注册 v1（键定位效率为什么逐港 0.11~0.80 甚至负）

日期：2026-10-05。前置：`SAR_CAUSAL_V2_FINDINGS_v1.md`（capture 是紧预算下唯一变量、不可预测）。

## 要回答
1. **机制定位**：capture 是否 ≈ 键分数 s2 对"win/loss"的判别力（AUROC）？若是，问题归约为 s2 的判别力。
2. **驱动**：s2 的判别力由什么决定？候选：expert 置信度校准（margin vs 对错）、base 准确率、win_rate、分歧数、s2 离散度、观测类覆盖。

## 每港测（分歧行上，oracle=真标签、诊断用）
- g = +1(expert赢)/0/-1(expert输)；
- AUROC_s2 = roc_auc(win vs loss, s2)（s2 的 win/loss 判别力）；
- capture（同 SAR_CAUSAL_V2）；win_rate、n_dis、base_acc、s2_std；
- AUROC_conf = roc_auc(expert 对错, expert margin)（expert 置信度校准度）。

## 判读
- capture 与 AUROC_s2 近单调 ⇒ 机制定位成立（capture=键判别力）；
- AUROC_s2 与某个驱动强相关 ⇒ 找到 capture 的驱动；
- 都不相关 ⇒ capture 是更深机制（键的类证据结构与域错位），如实记，换机制。
