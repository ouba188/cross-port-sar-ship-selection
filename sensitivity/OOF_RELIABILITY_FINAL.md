# 端口外可靠性模型（真实 SAR OOF，重训 PCA/Ridge）· 最终结论

日期 2026-10-07。23 港 × 5 seed。协议：从原始 embedding 重训每折 PCA128/64 + balanced Ridge(alpha=1)，
4D 特征（expert/base margin+entropy），Δ=1(e=y)−1(b=y)，源按港×类平衡，observed[expert] 守卫统一。

## 结果
| 方法 | BA | vs simple | p |
|---|---|---|---|
| base / fusion | 0.4327 / 0.4709 | — | — |
| simple / s2 | 0.4790 / 0.4791 | — | — |
| oof_lr0 | 0.4804 | +0.0014 | 0.71 |
| oof_utility0 | 0.4799 | +0.0008 | 0.75 |
| oof_lr64 | 0.4771 | −0.0020 | 0.16 |
| oof_utility64 | 0.4746 | −0.0044 | 0.056 |
| insample_lr0（泄漏对照） | 0.4818 | — | — |
| oracle | 0.5244 | — | — |

## 裁决
1. **4.54pp oracle 空间不可用可部署信息识别**：端口外 LR/相对收益 相对 simple 仅 +0.08~0.14pp（p=0.71/0.75），无可靠增益。
2. **64 标签适配有害**：utility64 −0.44pp（p=0.056）、lr64 −0.20pp（p=0.16）。
3. **训练内泄漏坐实**：insample_lr0(0.4818) > oof_lr0(0.4804)，oof_utility0 − insample_utility0 = −0.11pp（p=0.041）——旧"训练内预测"高估可靠性信号。
4. s2 ≈ simple（+0.00003, p=0.211）、simple > fusion 边际（+0.0081, p=0.119）均确认。

## 论文含义（诚实定论）
- 模型互补空间在"冻结 embedding + 少标签"下，从可部署信息角度**不可识别**（真实 OOF 证据）。
- 当前四条负证据合流：s2≈simple（EM/OT 无增量）、simple>fusion 边际（p=0.119）、共享适配（消融）、可靠性模型（OOF 打不平 simple）。
- 需重建核心创新。候选方向："类别支持未知"的文献差异 + 理论推导（缺类 vs 漏标），理论先行、不堆实现。
