"""在全部训练和重放完成后，保存优化效果、论文对照及结论边界。"""
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path('results_scene_v5_deterministic_search')
    s = json.loads((root/'summary.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    assert s['status']=='complete' and verification['status']=='PASS'
    base = json.loads(Path('results_scene_v4_5fold_80epochs/summary.json').read_text())
    paper = {'LRAP': (.878,.014), 'Micro_F1': (.748,.016), 'Macro_F1': (.758,.014),
             'Hamming': (.086,.007), 'Ranking': (.070,.010), 'Coverage': (.434,.058)}
    names = {'LRAP':'AP / LRAP ↑', 'Micro_F1':'Micro-F1 ↑', 'Macro_F1':'Macro-F1 ↑',
             'Hamming':'Hamming loss ↓', 'Ranking':'Ranking loss ↓', 'Coverage':'Coverage ↓'}
    better = {k: (s['mean'][k]>=v[0] if k in ['LRAP','Micro_F1','Macro_F1'] else s['mean'][k]<=v[0]) for k,v in paper.items()}
    comparison = {'paper_sha256':hashlib.sha256(Path('../paper.pdf').read_bytes()).hexdigest(),
                  'paper_table':1, 'paper_metrics':{k:{'mean':v[0],'std':v[1]} for k,v in paper.items()},
                  'difference_from_v4':{k:s['mean'][k]-base['mean'][k] for k in paper},
                  'difference_from_paper':{k:s['mean'][k]-v[0] for k,v in paper.items()},
                  'paper_mean_targets_met':better,
                  'decision':'all_six_paper_mean_targets_met' if all(better.values()) else 'partial_or_no_paper_mean_targets_met',
                  'strict_reproduction':'not_established',
                  'independent_confirmation':'not_established_previously_evaluated_dataset'}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    with (root/'fold_test_metrics.csv').open(encoding='utf-8-sig') as f: tests=list(csv.DictReader(f))
    with (root/'candidate_metrics.csv').open(encoding='utf-8-sig') as f: trials=list(csv.DictReader(f))
    for src,dst in [('formula_alignment_v5/training_repeatability.json','training_repeatability.json'),
                    ('formula_alignment_v5/experiment_plan.json','experiment_plan.json'),
                    ('data/scene_source/provenance.json','data_provenance.json')]: shutil.copy2(src,root/dst)
    lp=Path('diagnosis_paper_run/label_order_results_scene_v5_deterministic_search.json')
    permutation=json.loads(lp.read_text())
    assert all(r['invariance_at_1e_5'] for r in permutation['label_permutations'])
    shutil.copy2(lp,root/'label_order_verification.json')
    report=['# Scene v5：确定性修复与优化结果','',
            f'五折全部完成，每折用内部验证集比较 SGD 和 Adam，最多 120 轮；共训练 {s["total_epochs"]} 轮，用时 {s["seconds"]/60:.1f} 分钟。模型结构和五折划分沿用 v4，选模及所有预测通过重放核验。', '',
            '| 指标 | 原 v4 五折 | 本次 v5 五折 | 论文 Table 1 | v5 减论文 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for k,(pm,ps) in paper.items():
        report.append(f'| {names[k]} | {base["mean"][k]:.4f} ± {base["std"][k]:.4f} | {s["mean"][k]:.4f} ± {s["std"][k]:.4f} | {pm:.3f} ± {ps:.3f} | {s["mean"][k]-pm:+.4f} |')
    reached=[names[k] for k,v in better.items() if v]
    report += ['', '达到或超过论文均值的指标：'+('、'.join(reached) if reached else '无')+'。',
               '相对 v4，LRAP 均值仅增加 0.09 个百分点，Micro-F1 增加 0.58 个百分点，Macro-F1 增加 0.48 个百分点；未检验显著性，不能称为明确的性能突破。距论文均值仍分别低约 3.63、4.68、4.84 个百分点。',
               '标准差为五折样本标准差（ddof=1），不根据均值接近或标准差重叠声称显著性、统计等效或严格复现成功。', '',
               '## 每折内部选模与最终成绩','',
               '| 折 | SGD 最佳验证 LRAP | Adam 最佳验证 LRAP | 选中 | 选中轮次 | 测试 LRAP | Micro-F1 | Macro-F1 |',
               '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in tests:
        fold=int(row['fold']);c={r['candidate']:r for r in trials if int(r['fold'])==fold}
        report.append(f'| {fold} | {float(c["sgd"]["dev_LRAP"]):.4f} | {float(c["adam"]["dev_LRAP"]):.4f} | {row["selected_candidate"]} | {row["selected_epoch"]} | {float(row["LRAP"]):.4f} | {float(row["Micro_F1"]):.4f} | {float(row["Macro_F1"]):.4f} |')
    report += ['', '两种优化器每折从完全相同的初始权重开始，按该折内部验证 LRAP 独立选择优化器及轮次；测试不参与选择。每候选至少 80 轮、最多 120 轮，达到 80 轮后连续 20 轮没有更高验证 LRAP 就结束。学习率 0.001、gamma=0.933、分类阈值 0.5、C=3、k=5、β=1、批量 64、隐藏维度 128、两项额外正则为零；除优化器选择和训练预算外，训练设置与模型沿用 v4。', '',
               '## 修复与验证','',
               '首次 CUDA 运算前设置 cuBLAS 工作区，并开启严格确定性算法。两个独立进程各从头训练真实 Scene 第一折三轮，最终权重、每批损失和验证预测完全一致，最大概率差为 0；没有重复完整 120 轮，因此不将三轮检查夸大为完整训练重复性证明。', '',
               f'重放核对源码/数据哈希、全部分折、训练归一化与标签频率、候选与轮次选择、早停和学习率记录，以及保存的验证/测试预测。2407 行恰好各进入一次外层测试。重放最大概率误差 {verification["max_prediction_abs_error"]:.3g}，测试阈值分类及排序相同。公式、任务梯度、批次隔离和标签重排检查通过。', '',
               '## 结论边界','',
               '本次是固定预算的训练优化尝试，优化器等细节未在论文中充分给出，不能据 Adam 成绩认定作者使用同一配置。Scene 294 维按作者代码拆为 98/196 两视图；论文未说明精确分组、β、MLP、KNN 刷新及内部验证等细节，仍不能证明与作者训练过程完全一致。', '',
               '五折全部选择 SGD，当前 Adam（学习率 0.001、gamma=0.933）在内部验证上较弱。这个结论只针对本次固定配置，不代表 Adam 的其他学习率或训练策略也一定更差。', '',
               '此前已经查看过该数据集的 v4 五折结果；本轮虽然没有用新测试成绩选择候选，但属于在已评估数据集上的探索后评估，不能包装成全新盲测或独立确认。原 v4 与当前 v5 同时改变了确定性计算、候选选择和训练预算，整体提升不能单独归因于一个因素。', '',
               '论文 Table 1 的 Scene Coverage 标准差为 0.058，Table 3 为 0.008；主对照仍采用 Table 1。论文报告异常与我们取得的分数是不同证据，本次不据成绩差距推断造假。', '',
               '本轮按预先固定的两个候选完成，不根据测试结果扩展网格、种子或阈值搜索。检查点、候选曲线、验证和测试预测、源码、数据来源与核验文件均保留。']
    (root/'结果报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
