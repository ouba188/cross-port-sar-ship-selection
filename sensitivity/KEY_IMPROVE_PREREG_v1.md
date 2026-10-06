# 双向键提升 · 预注册 v1

日期 2026-10-06。目标：把双向选模型从 oracle 头寸 49% 往上提。

## 变量
- 分歧样本 D = {base≠expert}，双向路由：键符号决定用 base 还是 expert。
- 上限 oracle = 按真实 g=ec−bc 挑对（BA 0.5232）。
- 现状 s2 = robust_scores（min-over-supports 类证据）→ BA 0.4774，头寸 49%。

## 候选键（全部在分歧样本上取符号）
| # | 键 | 定义 |
|---|---|---|
| 1 | s2 | robust_scores(ratio2)（现基线） |
| 2 | logratio | log(ratio2[expert] / ratio2[base]) |
| 3 | postdiff | posterior[expert] − posterior[base]（class_mass） |
| 4 | margin_diff | expert_margin − base_margin |
| 5 | g_lr_conf | 源侧 logistic([e_margin,b_margin,e_entropy,b_entropy])→P(专家胜) |
| 6 | g_lr_feat | 源侧 logistic(4 置信 + VV-128 + VH-64)→P(专家胜) |

## 判据（写死）
- 主指标 = oracle 头寸捕获率 capture = (key_BA − base_BA)/(oracle_BA − base_BA)。
- 有效 = capture > s2 的 0.49；且 BA 与 ACC 双列都报。
- g 预测器在**源侧分歧样本**训练、**目标侧分歧样本**推理（无标签泄漏）。

## 说明
- g_lr 用源标签训「专家是否胜过基座」，正是模型池失败说的「模型可靠性」维度——但那是 4 模型池；这里 2 模型 base-vs-expert，且含原始特征，值得一试。若仍不迁移，则坐实该墙在 2 模型也成立。
