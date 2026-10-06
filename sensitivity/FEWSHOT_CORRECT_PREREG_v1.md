# 路由 + 少样本修正（M5）· 预注册 v1

日期 2026-10-06。定位：把 64 标签从「只路由」升级为「路由 + 修正」，回应 64-ridge 的 ACC 软肋。

## 问题与变量
- base = ridge(VV-128) 源训；expert = ridge([VV|VH]) 源训；ann=64 标签；预算 5%。
- 现状（OTH）：64 标签只用于**路由键** s2（类证据），选中样本用**源训 expert**。
- 升级（M5）：选中样本改用 **64 标签训的分类器**（fewshot）或**源 expert 与 fewshot 的融合**，把域移下的源 expert 换成目标适应的分类器。

## 臂（5）
| # | 臂 | 选中样本(5%)用 | 未选中(95%)用 |
|---|---|---|---|
| A | OTH（参照，冻结） | 源 expert | base |
| B | M5-fewshot | 64 标签 ridge([VV\|VH]) | base |
| C | M5-blend | argmax(softmax(expert)+softmax(fewshot)) | base |
| D | fewshot-full（参照） | 64-ridge 全样本（100% VH） | — |
| E | fusion（参照） | 源 expert 全样本 | — |

## 协议
- 与 BASELINE_COMPARE_PREREG_v1 同：LOPO、评估集 query\ann、单次抽取、natural 5%、类分层源子样本 seed 1000、ann seed 0。
- fewshot = ridge(one-hot(ay), [VV|VH] on ann, alpha=1, balanced sample_weight)。

## 判据（写死）
- **主指标 macro BA**：B、C 的 23 港均值须 **> OTH(A)**，才可声称「少样本修正提升路由」。
- **次指标加权 ACC**：B、C 须**收窄**与 fewshot-full(D) 的差距（D 的 ACC 是上界参照）。
- 两列都报、逐港 + 均值 + 正港数。负结果如实报，不改判据。
