# 深度对齐基线（DANN/IRM/MixStyle）· 预注册 v1

日期 2026-10-06。闭环 BASELINE_COMPARE 缺失的 3 个深度 DA 基线。

## 协议（与 BASELINE_COMPARE_PREREG_v1 一致）
- LOPO；source = 类分层子样本(seed 1000)；target = query；评估集 query\ann；macro BA + 加权 ACC 双列。
- 输入 = 冻结 [VV-128|VH-64]（pred_cache2 的 Xs/Z，与 CORAL/MMD 同源）。
- 分类器容量对齐我们的 ridge（线性），对齐机制用非线性 MLP。

## 三方法（torch，冻结特征上）
| 方法 | 架构 | 目标 |
|---|---|---|
| DANN | G=MLP(192→128 ReLU→128)、C=Linear(128→8)、D=MLP(128→2)，梯度反转 λ=1 | 对抗域对齐 |
| IRMv1 | w=Linear(192→8)，环境=源港，惩罚 ∇_{c=1} R_e 的梯度范数，λ=1 | 不变风险最小化 |
| MixStyle | Linear(192→8) + 跨源港实例统计(mean/std)混合，α~Beta(0.1,0.1) p=0.5 | 风格混合 |

## 判据
- 报三方法 macro BA / ACC，与 fusion(0.4696/0.4889)、CORAL(0.405)、MMD(0.314)、双向s2(0.4774/0.5010) 同表。
- 预期（不据以调参）：深度对齐在冻结 PCA 特征上难超 fusion；如实报。
