# capture 驱动挖掘 · 结果 v1（键定位效率 = 专家置信度校准）

日期：2026-10-05。预注册 `CAPTURE_DRIVER_PREREG_v1.md`。23 港 × 主线真链，诊断用 oracle。产物 `capture_driver.json`。

## 结果
1. **capture vs AUROC_s2（键的 win/loss 判别力）：Spearman 0.807，p=3.2e-6** —— capture 本质就是 s2 的判别力；
2. **AUROC_s2 的驱动**：唯一显著 = **expert 置信度校准 AUROC_conf（margin↔对错），Spearman 0.526，p=0.010**；
   其余候选全不显著：win_rate 0.242、base 准确率 0.271、分歧数 −0.197、s2 离散度 0.150、meanH 0.050。

## 机制链（完整）
expert 置信度校准（margin 能否预测 expert 对错）→ 类证据 s2 的 win/loss 判别力 → capture → 紧预算下实际收益。

## 结论
- 键定位准不准 = expert 置信度是否校准好；校准差的港 capture 低甚至负（键帮倒忙）；
- 这解释了 capture 0.11~0.80 的逐港差异，也指向可操作修复：**校准 expert 置信度（温度缩放/保序回归，源侧）**。

## 边界
- 相关性（0.526）非因果；方向与机制逻辑一致，但需后续干预实验坐实；
- AUROC_conf 本身逐港 0.54~0.72，expert 校准度是可变属性。
