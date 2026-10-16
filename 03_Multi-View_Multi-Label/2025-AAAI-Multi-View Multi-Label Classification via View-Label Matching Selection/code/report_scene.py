"""核对 Scene 五折汇总，保存论文对照和中文报告；不训练或调整参数。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    root = Path('results_scene_v4_5fold_80epochs')
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    verified = json.loads((root / 'verification.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'complete' and verified['status'] == 'PASS'
    assert verified['folds'] == 5 and verified['epochs'] == 400 and verified['test_rows'] == 2407
    with (root / 'fold_test_metrics.csv').open(encoding='utf-8-sig') as f:
        folds = list(csv.DictReader(f))
    paper = {'LRAP': (.878, .014), 'Micro_F1': (.748, .016),
             'Macro_F1': (.758, .014), 'Hamming': (.086, .007),
             'Ranking': (.070, .010), 'Coverage': (.434, .058)}
    for metric in paper:
        values = [float(row[metric]) for row in folds]
        assert abs(np.mean(values) - summary['mean'][metric]) < 1e-12
        assert abs(np.std(values, ddof=1) - summary['std'][metric]) < 1e-12
    ids = np.concatenate([np.load(root / f'fold_{i}_predictions.npz')['sample_ids'] for i in range(1, 6)])
    np.testing.assert_array_equal(np.sort(ids), np.arange(2407))
    comparison = {'dataset': 'Scene', 'paper_page': 5, 'paper_table': 1,
                  'paper_sha256': hashlib.sha256(Path('../paper.pdf').read_bytes()).hexdigest(),
                  'paper_metrics': {k: {'mean': v[0], 'std': v[1]} for k, v in paper.items()},
                  'mean_difference': {k: summary['mean'][k]-v[0] for k, v in paper.items()},
                  'paper_inconsistency': 'Scene Coverage std: Table 1 = 0.058; Table 3 = 0.008. Main comparison uses Table 1.',
                  'decision': 'fixed_protocol_complete_no_claim_of_exact_reproduction'}
    (root / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    shutil.copy2('data/scene_source/provenance.json', root / 'data_provenance.json')
    order_report = Path('diagnosis_paper_run/label_order_results_scene_v4_5fold_80epochs.json')
    order = json.loads(order_report.read_text(encoding='utf-8'))
    assert all(r['invariance_at_1e_5'] for r in order['label_permutations'])
    shutil.copy2(order_report, root / 'label_order_verification.json')
    names = {'LRAP': 'AP / LRAP ↑', 'Micro_F1': 'Micro-F1 ↑', 'Macro_F1': 'Macro-F1 ↑',
             'Hamming': 'Hamming loss ↓', 'Ranking': 'Ranking loss ↓', 'Coverage': 'Coverage ↓'}
    lines = ['# Scene：固定 v4 完整五折实验', '',
             '五折、每折 80 轮全部完成，共 400 轮。2407 个样本恰好各进入一次外层测试，源码、数据、分折、验证选模和保存预测重放核验通过。参数沿用 Emotions 实验，没有根据 Scene 测试结果调参。', '',
             '| 指标 | 本次五折 | 论文 Table 1 | 本次减论文 |',
             '| --- | ---: | ---: | ---: |']
    for k, (mean, std) in paper.items():
        lines.append(f'| {names[k]} | {summary["mean"][k]:.4f} ± {summary["std"][k]:.4f} | {mean:.3f} ± {std:.3f} | {comparison["mean_difference"][k]:+.4f} |')
    lines += ['', '本次六项均值均弱于论文：LRAP 低 3.72 个百分点，Micro-F1 低 5.26 个百分点，Macro-F1 低 5.32 个百分点。完成了可核验的固定配置实验，仍未达到论文报告效果。第一、第五折选中第 80 轮，部分验证曲线接近预算末尾仍在提升，因此不能断言模型已充分收敛；本次不自动扩展预算或调参。', '',
              '标准差采用五折样本标准差（ddof=1）。论文未提供相同分割和原始折成绩，以上只是描述性对照，不作配对显著性判断。论文 Table 1 的 Scene Coverage 标准差为 0.058，Table 3 为 0.008，主对照采用 Table 1。', '',
              '## 各折结果', '', '| 折 | 验证选择轮次 | 验证 LRAP | 测试 LRAP | Micro-F1 | Macro-F1 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in folds:
        lines.append(f'| {row["fold"]} | {row["selected_epoch"]} | {float(row["dev_LRAP"]):.4f} | {float(row["LRAP"]):.4f} | {float(row["Micro_F1"]):.4f} | {float(row["Macro_F1"]):.4f} |')
    lines += ['', '## 数据、协议和限制', '',
              '数据来自 Mulan 官方 Scene 归档；原训练 1211 行、测试 1196 行，合并后与官方 scene.arff 完全一致。合并后重新五折划分，与官方固定训练/测试划分不同。294 个特征按作者 demo.py 的 Scene 规则分为前 98 和后 196 两个视图；论文没有明确这项分组，无法确认与作者论文实验数据表示完全一致。转换阶段不做归一化。', '',
              '五折种子 2020，模型种子 2020 至 2024。每折外层训练部分留出 20% 验证集：内层训练 1540 行，验证 385/386 行，测试 481/482 行。归一化、标签频率、KNN 参考库仅用内层训练数据；用验证 LRAP 选择轮次，结束训练后做最终测试。', '',
              '固定 C=3、k=5、β=1、学习率 0.001、SGD 动量 0.9、指数衰减 0.933、阈值 0.5、批量 64、隐藏维度 128，权重衰减和额外范数正则为 0。dropout 参数仍无对应 Dropout 层，批次顺序固定，行为保持与上一实验相同。', '',
              'β、精确 MLP、KNN 空间与刷新策略、内部验证等细节没有在论文中充分说明。训练标签频率估计与论文 whole data set 的文字不同。当前结果检验固定 v4 实现在新数据集上的表现，不证明实现及评估与作者完全一致。', '',
              '## 核验', '',
              f'全部五折模型重放通过；最大预测绝对误差 {verified["max_prediction_abs_error"]:.3g}，容差 atol=5e-6、rtol=1e-6。阈值分类和标签排序完全一致，五折均值/标准差重新计算一致。第一折验证集标签重排回归检查通过，不使用测试标签调参。', '',
              f'训练及保存耗时 {summary["seconds"]:.2f} 秒，约 {summary["seconds"]/60:.1f} 分钟；不含数据准备、解释器启动和重放核验。', '',
              '文件：summary.json、comparison.json、verification.json、label_order_verification.json、data_provenance.json、manifest.json、splits.json、epoch_metrics.csv、fold_test_metrics.csv、checkpoints/、fold_*_predictions.npz 和 source/。']
    (root / '结果报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'mean': summary['mean'], 'std': summary['std'], 'seconds': summary['seconds'], 'comparison': comparison}, indent=2))


if __name__ == '__main__':
    main()
