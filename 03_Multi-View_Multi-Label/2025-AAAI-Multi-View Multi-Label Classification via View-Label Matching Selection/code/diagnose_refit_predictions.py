"""描述保存预测的退化现象；固定阈值 0.5，不调参、不重新选模型。"""
import json
from pathlib import Path
import numpy as np


def describe(path):
    with np.load(path) as data:
        y = data['labels'].astype(bool); score = data['predictions']; pred = score >= .5
        return {'rows':len(y),'true_positive_labels_per_row':float(y.sum(1).mean()),
                'predicted_positive_labels_per_row':float(pred.sum(1).mean()),
                'rows_without_predicted_labels_fraction':float((pred.sum(1)==0).mean()),
                'true_positive_count':int((pred&y).sum()),'false_positive_count':int((pred&~y).sum()),
                'false_negative_count':int((~pred&y).sum()),
                'label_probability_mean':score.mean(0).tolist(),
                'label_probability_std':score.std(0).tolist(),
                'mean_within_label_probability_std':float(score.std(0).mean())}


def main():
    root = Path('results_scene_v5_refit_outer_train')
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert json.loads((root/'verification.json').read_text())['status'] == 'PASS'
    parent = Path(manifest['parent_run'])
    report = {'threshold':.5,'descriptive_only':True,'configuration_or_model_selection':False,
              'causal_explanation_established':False,'folds':[]}
    for fold in range(1,6):
        row = {'fold':fold,'inner_selected_model':describe(parent/f'fold_{fold}_predictions.npz'),
               'outer_train_refit':describe(root/f'fold_{fold}_predictions.npz')}
        report['folds'].append(row)
        print(f'Fold {fold}: predicted positive labels/row '
              f'{row["inner_selected_model"]["predicted_positive_labels_per_row"]:.3f} -> '
              f'{row["outer_train_refit"]["predicted_positive_labels_per_row"]:.3f}; '
              f'true {row["outer_train_refit"]["true_positive_labels_per_row"]:.3f}')
    (root/'prediction_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
