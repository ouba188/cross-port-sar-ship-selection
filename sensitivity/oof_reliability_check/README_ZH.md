# 下一轮：端口外相对可靠性强基线核验

针对仓库 c82933b9137135d93f3bf03ed42c467d9996c9f7。目的：纠正旧 source gate 的训练内预测、收益标签及评估口径，然后检查模型互补空间是否能用少量目标标签兑现。本包不是已成立的 TPAMI 创新。

## 已完成与尚未完成

- 已实现端口外预测生成、统一门控与配对统计。
- 已用现有真实预测缓存复现 23 港×5 seeds 的 simple/s2/fusion/oracle 参照；结果在 verified_reference_replay/。
- 已验证：改变非支持集目标标签不改变预测；BA增益=挽回贡献−误切贡献；partial/错父缓存不能进入主评估。
- builder 的合成数据隔离核验已通过。
- **尚未运行真实 SAR 端口外重训。** 这里缺原始 VV/VH embedding，需要在现有本地环境运行。合成数据通过不代表真实实验完成。

## 文件

- build_oof_predictions.py：从原始 encoder embedding 重拟合各折 PCA、标准化和两个 Ridge 分类头。
- run_oof_reliability.py：同一64个支持标签、同一query\support、5 seeds 的统一对照。
- verify_pipeline.py：针对隔离、奖励与BA核算的检查。
- frozen_modules/：c82933b 的冻结似然、协方差、权重和OT参照函数。
- outer_recipe_reference.py：原仓库缓存生成配方的来源副本，供核对，不要直接运行其旧硬编码路径。

依赖：numpy、scipy、scikit-learn、threadpoolctl、pandas、POT。使用原 ClearSAR 环境即可。

## 固定协议

1. 外层保持原 pred_cache 的 qpos/spos，保留同 MMSI/product 排除。
2. 每个内层源港的全部标签都从分类头训练中排除；同船和同product关联也排除。PCA128/64、标准化、Ridge(alpha=1, balanced)在该折剩余源数据上重新拟合。
3. 小源港仍提供端口外预测，包括 Melbourne；不用64标签后空query来调参。
4. 与原外层缓存核对：base/expert 的 argmax 必须零差异；logit 误差须在预定容差内。失败时检查特征、行顺序和sklearn版本，不跳过验证。
5. 每个目标港随机抽取64个不同样本（seed0–4），全部方法共享支持集，评估排除支持集。
6. 源域权重：每港等权、港内每类等权，类别计数在筛选分歧样本之前计算。目标适配权重只取64支持集频率。
7. 相对收益 Δ=I(expert=y)−I(base=y)，值为−1/0/+1。两模型都错的分歧样本收益为0。Ridge保留0；正确的binary LR只训练Δ≠0行，取其正负选择概率的符号。
8. 所有学得门控使用相同 observed[expert] 限制，集中核验已观测类中的选择错误。
9. 本轮固定四维特征：两个模型的margin和entropy。source Ridge平均加权损失的L2=0.01，LR C=1，目标64标签适配向源参数收缩的强度=8。不调参，不在一次OOF缓存上冒称nested CV。
10. 目标LR适配用条件二元交叉熵，目标Ridge用Δ平方损失。不同损失和正则尺度不完全匹配；差值不能单独归因于三值目标。

## 上游 encoder 的范围

脚本重训分类头和PCA，**不重训encoder**。它证明的是给定冻结embedding条件下的端口外监督head预测。

运行前由Codex核对 b2_ssl_feats_38091.npz / e206 的来源并记录依据：独立预训练可声明 independent；若使用SAR目标的无标签数据，须所有对照采用一致的 transductive-unlabeled 规则并披露。若encoder用过留出港监督标签，需要另做按港隔离的encoder重训，本脚本不支持用声明代替重训。身份排除也只能检查manifest已有的非空标识，不能证明缺失标识和整个上游数据流程无泄漏。

## 运行顺序（PowerShell）

将整个目录放入现有仓库的 sensitivity/oof_reliability_check。下面默认使用已知本地路径；路径不存在时从现有脚本核对，不下载或生成替代数据。

```powershell
$py = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
$task = 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check'
$raw = 'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916'
$ws = 'E:/临时会话/visual_reliable_baseline'
$pred = 'E:/Hermes/paper_tracking/sensitivity/pred_cache'
$oof = 'E:/Hermes/paper_tracking/sensitivity/oof_predictions_v1'
$access = '请根据上游来源填写 independent 或 transductive-unlabeled'

& $py "$task/build_oof_predictions.py" --self-test
& $py "$task/verify_pipeline.py" --pred-cache $pred

# 一港一折兼容性检查。PARTIAL是预期状态，不能拿它评主结果。
& $py "$task/build_oof_predictions.py" --manifest "$raw/manifest.csv" --vv-features "$raw/pooled_features.npy" --vh-features "$ws/b2_ssl_feats_38091.npz" --pred-cache $pred --out $oof --encoder-access $access --ports 'Port Said' --inner-ports Melbourne --threads 1

# 兼容性通过后完整生成。复用上面同一路径会验证指纹并继续已有折。
& $py "$task/build_oof_predictions.py" --manifest "$raw/manifest.csv" --vv-features "$raw/pooled_features.npy" --vh-features "$ws/b2_ssl_feats_38091.npz" --pred-cache $pred --out $oof --encoder-access $access --threads 1

# 评估输出目录必须全新或为空，避免旧summary被当成本轮结果。
& $py "$task/run_oof_reliability.py" --pred-cache $pred --oof-cache $oof --manifest "$raw/manifest.csv" --out 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_results_v1' --threads 1
```

每一步检查退出码；失败不继续后面的命令。大约23个outer×23个inner，另23次外层复现；从smoke的实测每折秒数估总时长，不以脚本exit0单独认定主实验完成。builder的exit0可能只生成PARTIAL，必须完整coverage后才能运行主评估。

## 主表与回报

主参照：base、fusion、simple、s2。新增：insample_lr0、insample_utility0、oof_lr0、oof_utility0、oof_lr64、oof_utility64。oracle仅作诊断上界。

回传：summary.json、逐港JSON、builder日志（包含外层复现误差、coverage、每折耗时）、encoder来源说明。主要看：

- OOF相对训练内门控有没有变化；
- OOF＋64支持适配相对simple能否稳定降低BA误切，同时保留挽回；
- 优势是否依赖少数港，以及支持集中实际有多少分歧/有效正负样本。

统计单位为每港5seed均值，用港口配对bootstrap CI与Wilcoxon。5次支持抽样高度重叠，不当作115个独立港。全部现有港已经用于探索；该轮为发展阶段核验，不能称全新盲测。

四维线性门控的失败不能否定所有类别条件可靠性信号，也不能证明剩余oracle空间不可学。正结果先归为正确强基线；只有进一步提出其不能解释、且能在新留出数据验证的机制，才有资格立TPAMI方法贡献。
