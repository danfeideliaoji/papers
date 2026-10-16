"""完成核验后生成最终重拟合报告；不根据结果追加搜索。"""
import csv
import hashlib
import json
import shutil
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    root = Path('results_scene_v5_refit_outer_train')
    manifest = load(root/'manifest.json'); s = load(root/'summary.json'); v = load(root/'verification.json')
    assert manifest['status'] == s['status'] == 'complete' and v['status'] == 'PASS'
    parent = Path(manifest['parent_run']); previous = load(parent/'summary.json')
    paper_hash = hashlib.sha256(Path('../paper.pdf').read_bytes()).hexdigest()
    assert paper_hash == load(parent/'comparison.json')['paper_sha256']
    base = load(Path('results_scene_v4_5fold_80epochs/summary.json'))
    paper = {'LRAP':(.878,.014),'Micro_F1':(.748,.016),'Macro_F1':(.758,.014),
             'Hamming':(.086,.007),'Ranking':(.070,.010),'Coverage':(.434,.058)}
    names = {'LRAP':'AP / LRAP ↑','Micro_F1':'Micro-F1 ↑','Macro_F1':'Macro-F1 ↑',
             'Hamming':'Hamming loss ↓','Ranking':'Ranking loss ↓','Coverage':'Coverage ↓'}
    higher = {'LRAP','Micro_F1','Macro_F1'}
    met = {k:s['mean'][k]>=m if k in higher else s['mean'][k]<=m for k,(m,sd) in paper.items()}
    permutation_path = Path('diagnosis_paper_run/label_order_'+root.name+'.json')
    permutation = load(permutation_path)
    assert permutation['query_role'] == 'training_subset_former_inner_dev'
    assert all(r['invariance_at_1e_5'] for r in permutation['label_permutations'])
    comparison = {'paper_sha256':paper_hash,
                  'paper_table':1,'paper_metrics':{k:{'mean':m,'std':sd} for k,(m,sd) in paper.items()},
                  'difference_from_v4':{k:s['mean'][k]-base['mean'][k] for k in paper},
                  'difference_from_inner_selection_model':{k:s['mean'][k]-previous['mean'][k] for k in paper},
                  'difference_from_paper':{k:s['mean'][k]-m for k,(m,sd) in paper.items()},
                  'paper_mean_targets_met':met,'strict_reproduction':'not_established',
                  'independent_confirmation':'not_established_previously_evaluated_dataset'}
    save(root/'comparison.json',comparison)
    for source,destination in [(parent/'training_repeatability.json','training_repeatability.json'),
                               (parent/'experiment_plan.json','parent_experiment_plan.json'),
                               (parent/'data_provenance.json','data_provenance.json'),
                               (Path('formula_alignment_v5/完整训练折拟合说明.md'),'refit_protocol.md'),
                               (permutation_path,'label_order_verification.json')]:
        shutil.copy2(source,root/destination)
    with (root/'fold_test_metrics.csv').open(encoding='utf-8-sig') as f: tests = list(csv.DictReader(f))
    diagnostics = load(root/'prediction_diagnostics.json')
    report = ['# Scene v5：完整训练折最终拟合结果','',
              f'五折完整训练及重放核验已完成。内部验证选定配置后，使用每折全部外层训练样本重新拟合，固定训练轮数，共 {s["total_epochs"]} 轮，用时 {s["seconds"]/60:.1f} 分钟。', '',
              '| 指标 | v4 五折 | v5 内部选模模型 | 本次完整训练折模型 | 论文 Table 1 | 本次减论文 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k,(m,sd) in paper.items():
        report.append(f'| {names[k]} | {base["mean"][k]:.4f} ± {base["std"][k]:.4f} | {previous["mean"][k]:.4f} ± {previous["std"][k]:.4f} | {s["mean"][k]:.4f} ± {s["std"][k]:.4f} | {m:.3f} ± {sd:.3f} | {s["mean"][k]-m:+.4f} |')
    reached = [names[k] for k,ok in met.items() if ok]
    report += ['', '达到或超过论文均值的指标：'+('、'.join(reached) if reached else '无')+'。',
               f'相对上一阶段内部选模模型，LRAP 变化 {(s["mean"]["LRAP"]-previous["mean"]["LRAP"])*100:+.2f} 个百分点，Micro-F1 变化 {(s["mean"]["Micro_F1"]-previous["mean"]["Micro_F1"])*100:+.2f} 个百分点，Macro-F1 变化 {(s["mean"]["Macro_F1"]-previous["mean"]["Macro_F1"])*100:+.2f} 个百分点。',
               '标准差为五折样本标准差（ddof=1）。未检验显著性或统计等效；均值接近、单折达到或标准差重叠均不能证明严格复现成功。', '',
               '## 固定选择与每折结果','',
               '| 折 | 优化器 | 固定轮数 | 训练样本 | 测试 LRAP | Micro-F1 | Macro-F1 |',
               '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in tests:
        report.append(f'| {row["fold"]} | {row["candidate"]} | {row["fixed_epochs"]} | {row["train_rows"]} | {float(row["LRAP"]):.4f} | {float(row["Micro_F1"]):.4f} | {float(row["Macro_F1"]):.4f} |')
    report += ['', '## 预测下降现象','',
               '| 折 | 旧模型预测正标签 / 行 | 本次预测正标签 / 行 | 实际正标签 / 行 | 本次无正标签预测的行比例 |',
               '| --- | ---: | ---: | ---: | ---: |']
    for row in diagnostics['folds']:
        old = row['inner_selected_model']; new = row['outer_train_refit']
        report.append(f'| {row["fold"]} | {old["predicted_positive_labels_per_row"]:.3f} | {new["predicted_positive_labels_per_row"]:.3f} | {new["true_positive_labels_per_row"]:.3f} | {new["rows_without_predicted_labels_fraction"]:.1%} |')
    report += ['', '第 4、5 折正标签预测大幅减少，明显漏报是 F1 下降的直接表现。LRAP 同时下降还说明排序质量变差，不能靠改变阈值解决全部差距。这些是固定阈值 0.5 下保存测试预测的描述，没有搜索阈值或重新选模型；该检查不能确定训练退化的单一根因。']
    report += ['', '## 固定拟合方案','', '优化器和轮数来自上一阶段各折内部验证 LRAP：101、91、103、82、114 轮，全部 SGD。重拟合期间不再验证或早停，不选择新的轮数，每折只保存固定末轮检查点并进行最终测试。selection_dev_LRAP 是旧内部选模模型的验证分数，不是本次模型的独立验证表现。', '',
               '学习率 0.001、momentum=0.9、gamma=0.933、分类阈值 0.5、C=3、k=5、β=1、批量 64、隐藏维度 128，两项额外正则为零。训练初始权重与上一阶段该折完全相同。', '',
               '## 修复及证据','',
               '开启严格 CUDA 确定性计算，两个独立进程从头训练真实 Scene 第一折三轮，权重、批次损失和验证预测完全一致；未额外重复完整训练，不扩大三轮检验的证明范围。模型结构沿用已完成公式、梯度及批次隔离检查的 v4。', '',
               '归一化、标签频率和 KNN 参考库均用完整外层训练折构建。原内部训练与验证行的并集为 1925／1926 行，与该折测试行完全分离。2407 个样本恰好各进入一次外层测试。', '',
               f'核验源码、数据、父实验候选文件哈希，重新计算候选选择、固定轮数、分折、预处理、频率和学习率记录，重放五折全部测试预测。最大概率误差 {v["max_prediction_abs_error"]:.3g}，分类、排序和六项指标与保存结果一致。标签编号重排回归通过，检查使用旧验证行构成的训练子集，不读取测试标签。', '',
               '## 结论及剩余限制','',
               '本次完成了确定性修复、内部选模和完整外层训练折最终拟合。它同时改变了训练样本、拟合的归一化与频率、KNN 参考库以及每轮更新次数，不能将效果变化单独归因于训练样本数。', '',
               '模型虽已按论文公式修正，论文仍未充分说明 β、具体 MLP、KNN 空间与刷新、内部验证及优化器等细节。Scene 按作者代码将 294 维分成 98/196 两视图，但论文没有明确这项分组；因此仍不能证明训练过程与作者完全一致。', '',
               '此前已查看同一 Scene 数据集的测试结果，这属于探索后的协议完善与评估，不能包装为全新盲测或独立确认。原内部选模及所有历史结果保留，不能只挑表现最好的阶段汇报。', '',
               '论文 Table 1 的 Scene Coverage 标准差为 0.058，Table 3 为 0.008，本报告使用 Table 1。成绩差距和论文表格疑点不足以认定数据造假。', '',
               '本阶段按预先冻结的五折配置结束，不根据这次测试结果追加超参数、种子、阈值或数据集搜索。检查点、结构化曲线、预测、源码快照、数据来源与核验文件全部保留。']
    (root/'结果报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    scripts = root/'verification_source'; scripts.mkdir(exist_ok=True)
    hashes = {}
    for name in ['verify_outer_train_refit.py','verify_label_order.py','report_outer_train_refit.py','diagnose_refit_predictions.py']:
        shutil.copy2(name,scripts/name); hashes[name] = hashlib.sha256((scripts/name).read_bytes()).hexdigest()
    save(scripts/'source_manifest.json',hashes)
    save(Path('formula_alignment_v5/last_full_run.json'),
         {'run':root.name,'parent_run':parent.name,'status':'complete_verified','folds':5,
          'epochs':s['total_epochs'],'test_rows':v['test_rows'],'paper_mean_targets_met':met,
          'strict_reproduction':'not_established'})
    print(json.dumps(comparison,indent=2))


if __name__ == '__main__': main()
