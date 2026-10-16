# 核验已保存的 GPU 六折 × 80 轮结果，无须训练日志，也不追加训练。
# 检查逐轮指标、六个检查点和第 1 折预测重放；工作目录应为当前 code 目录。

"""Audit a completed 6-fold/80-epoch run and replay fold 1 without training."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'results' / 'source'))
import demo
from MLP import merge_views
from GNBlock_model import _Model
from util import one_hot2, shuffle, get_k_fold_data


# 读取 results/ 的结构化指标，核验模型最佳轮，再重建特征与标签初始化做预测。
def main():
    root = Path('results')
    previous_summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    with (root / 'epoch_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key: int(value) if key in ('fold', 'epoch') else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    # 要求六折各 80 轮，且后续检查每折轮号连续、所有指标有限。
    assert len(rows) == 480, len(rows)
    assert all(np.isfinite(v) for row in rows for v in row.values())
    for f in range(1, 7):
        assert [row['epoch'] for row in rows if row['fold'] == f] == list(range(1, 81))
    fields = list(rows[0])
    with (root / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 逐个读取模型参数，检查参数有限性及保存分数与对应轮记录一致。
    checkpoint_metrics = []
    for f in range(1, 7):
        path = Path(f'Model_Best_GPU/{f}_fold_emotions_model_best.pth')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
        matching = [row for row in rows if row['fold'] == f]
        assert abs(float(checkpoint['best_score']) - max(row['Average_Precision'] for row in matching)) < 1e-6
        selected = matching[checkpoint['epoch']]
        assert abs(selected['Average_Precision'] - float(checkpoint['best_score'])) < 1e-6
        checkpoint_metrics.append(selected)

    # Recreate the frozen random input projections and label features from the original seed.
    args = SimpleNamespace(dataset='emotions.mat', batch_size=64, k=5, k_fold=6,
                           GPUS=1, nhidden=128, target_dim=128, dropout=0.4, seed=2020)
    demo.args = args
    # 随机输入 MLP 和标签初始化未单独保存，必须按原运行种子与调用顺序重建。
    demo.setup_seed(args.seed)
    views, target = demo.data_loader('./data/emotions.mat')
    target = torch.tensor(target).permute(1, 0)
    views, target = demo.data_clean(views, target)
    views = merge_views(views, args)
    label_features = one_hot2(target, args).unsqueeze(0)
    views, target = shuffle(views, target, args.seed)
    test_views = [get_k_fold_data(6, 0, view, target)[2] for view in views]
    test_labels = get_k_fold_data(6, 0, views[0], target)[3]
    neighbors = torch.stack([
        torch.from_numpy(NearestNeighbors(n_neighbors=6, metric='cosine')
                         .fit(view.numpy()).kneighbors(view.numpy(), return_distance=False)[:, 1:])
        for view in test_views
    ]).unsqueeze(0)
    full_test = torch.stack(test_views).transpose(0, 1)
    # 重新建立同结构图模型，加载第 1 折权重，仅执行 eval() 和 no_grad() 推理。
    model = _Model().to('cuda')
    checkpoint = torch.load('Model_Best_GPU/1_fold_emotions_model_best.pth', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    prediction = []
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, len(test_labels), 64)):
            samples = full_test[start:start + 64]
            batch_labels = test_labels[start:start + 64].unsqueeze(1).repeat(1, 2, 1)
            edges, counts = model(samples, label_features, batch_labels, args,
                                  neighbors, full_test.unsqueeze(0), batch_number)
            scores = edges.reshape(len(samples), 2, 6).mean(1)
            assert scores.device.type == 'cuda' and torch.isfinite(scores).all()
            prediction.append(scores.cpu())
    prediction = torch.cat(prediction).numpy()
    # 重算 LRAP 并与检查点记录比较；这是模型保存结果与实际预测的一致性检查。
    replay_lrap = label_ranking_average_precision_score(test_labels.numpy(), prediction)
    assert abs(replay_lrap - float(checkpoint['best_score'])) < 1e-6, (replay_lrap, checkpoint['best_score'])
    np.savez_compressed(root / 'fold1_replay_predictions.npz', labels=test_labels.numpy(), predictions=prediction)
    with (root / 'fold_best_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checkpoint_metrics)
    metrics = fields[2:]
    summary = {
        'status': 'complete_and_verified', 'dataset': 'emotions.mat',
        'folds': 6, 'epochs_per_fold': 80, 'verified_epochs': len(rows),
        'device': torch.cuda.get_device_name(0),
        'training_and_export_seconds': previous_summary['training_and_export_seconds'],
        'mean_of_fold_best_metrics': {key: float(np.mean([row[key] for row in checkpoint_metrics])) for key in metrics},
        'sample_std_of_fold_best_metrics': {key: float(np.std([row[key] for row in checkpoint_metrics], ddof=1)) for key in metrics},
        'fold1_replayed_lrap': replay_lrap,
        'data_sha256': hashlib.sha256(Path('data/emotions.mat').read_bytes()).hexdigest(),
        'limitations': ['Original global preprocessing and test-best epoch selection retained.',
                        'Not a strict paper reproduction or independent held-out estimate.'],
    }
    # 更新结构化汇总，不生成日志文件，也不修改模型参数。
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
