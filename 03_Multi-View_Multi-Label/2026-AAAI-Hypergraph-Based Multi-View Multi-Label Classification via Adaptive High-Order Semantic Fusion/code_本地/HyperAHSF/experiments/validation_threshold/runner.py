"""用同一训练权重比较五个验证图余弦阈值。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data_preprocessor import MyDataset_k_fold
from experiments.hgnn_ablation.reporting import (
    prepare_output_dir,
    write_csv,
    write_json,
)
from experiments.hgnn_ablation.runner import _environment, _train_epoch, fix_seed
from metrics import all_metrics
from model import HGMVMLC_Model
from utils import PrecomputedContrastiveLoss

from .configuration import SweepConfig
from .evaluation import evaluate_threshold
from .graph_stats import summarize_local_hypergraph
from .reporting import (
    metric_row,
    plot_learning_curves,
    summarize_graph_statistics,
    summarize_thresholds,
    write_report,
)


def _metric_pairs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_bar: float,
) -> dict[str, float]:
    """从已计算的 logits 生成原项目固定的八项指标。"""

    predictions = (torch.sigmoid(logits) > confidence_bar).to(torch.int64)
    return dict(
        all_metrics(
            logits.detach().cpu().numpy(),
            predictions.cpu().numpy(),
            labels.detach().cpu().numpy(),
        )
    )


def _serialized_config(
    config: SweepConfig,
    params: dict[str, object],
    environment: dict[str, object],
) -> dict[str, object]:
    """记录所有固定条件和作者原始阈值。"""

    return {
        "dataset": "emotions",
        "folds": list(config.folds),
        "epochs": config.epochs,
        "seed": config.seed,
        "device": config.device,
        "project_root": str(Path(config.project_root).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "source_test_thr": params["test_thr"],
        "thresholds": [
            {"name": threshold.name, "value": threshold.value}
            for threshold in config.thresholds
        ],
        "shared_weight_evaluation": True,
        "cache_reset_before_each_threshold": True,
        "params": params,
        "environment": environment,
    }


def run_sweep(config: SweepConfig) -> dict[str, object]:
    """每折训练一个模型，并在每个 epoch 用同一权重评估全部阈值。"""

    project_root = Path(config.project_root).resolve()
    output_dir = prepare_output_dir(Path(config.output_dir).resolve())
    with (project_root / "config.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["emotions"]
    if float(params["test_thr"]) != 99.0:
        raise ValueError("当前源配置的 test_thr 不是作者原值 99，拒绝混合实验。")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 实验，但当前 CUDA 不可用。")
    environment = _environment(device)
    metrics_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    fold_timings: list[dict[str, object]] = []
    epoch_times: list[float] = []
    overall_start = time.perf_counter()

    for fold in config.folds:
        fix_seed(config.seed)
        train_dataset = MyDataset_k_fold(
            int(params["num_views"]),
            str(params["name"]),
            fold,
            "train",
            device=device,
            scalers=None,
        )
        val_dataset = MyDataset_k_fold(
            int(params["num_views"]),
            str(params["name"]),
            fold,
            "val",
            device=device,
            scalers=train_dataset.scalers,
        )
        model = HGMVMLC_Model(dataset=train_dataset, params=params, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        contrastive_loss = PrecomputedContrastiveLoss(
            train_dataset.labels,
            float(params["w_intra"]),
        )
        fold_start = time.perf_counter()

        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            loss = _train_epoch(
                train_dataset,
                model,
                optimizer,
                contrastive_loss,
                config.seed,
            )
            shared_parameter_ids = tuple(id(parameter) for parameter in model.parameters())

            for threshold in config.thresholds:
                evaluation_start = time.perf_counter()
                evaluated = evaluate_threshold(
                    model,
                    val_dataset.features,
                    threshold.value,
                )
                if shared_parameter_ids != tuple(
                    id(parameter) for parameter in model.parameters()
                ):
                    raise RuntimeError("不同阈值没有共享同一组模型参数对象。")
                evaluation_seconds = time.perf_counter() - evaluation_start
                metrics = _metric_pairs(
                    evaluated.logits,
                    val_dataset.labels,
                    float(params["confidence_bar"]),
                )
                metrics_rows.append(
                    metric_row(
                        variant=threshold.name,
                        fold=fold,
                        epoch=epoch,
                        threshold=threshold.value,
                        loss=loss,
                        metrics=metrics,
                        epoch_seconds=evaluation_seconds,
                    )
                )
                for view, hyperedge_index in enumerate(evaluated.local_hyperedges):
                    graph_rows.append(
                        {
                            "variant": threshold.name,
                            "fold": fold,
                            "epoch": epoch,
                            "threshold": threshold.value,
                            "view": view,
                            **summarize_local_hypergraph(
                                hyperedge_index,
                                int(val_dataset.features[view].shape[0]),
                            ),
                        }
                    )

            epoch_seconds = time.perf_counter() - epoch_start
            epoch_times.append(epoch_seconds)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "fold": fold,
                            "epoch": epoch + 1,
                            "epochs": config.epochs,
                            "loss": loss,
                            "epoch_seconds": epoch_seconds,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        fold_timings.append(
            {
                "fold": fold,
                "seconds": time.perf_counter() - fold_start,
            }
        )
        del model, optimizer, contrastive_loss, train_dataset, val_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - overall_start
    threshold_summaries = summarize_thresholds(metrics_rows)
    graph_summaries = summarize_graph_statistics(graph_rows)
    timing: dict[str, object] = {
        "total_seconds": total_seconds,
        "average_epoch_seconds": float(np.mean(epoch_times)),
        "folds": fold_timings,
    }
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "graph_statistics.csv", graph_rows)
    write_csv(output_dir / "aggregate_results.csv", threshold_summaries)
    write_json(
        output_dir / "config.json",
        _serialized_config(config, params, environment),
    )
    plot_learning_curves(output_dir / "learning_curves.svg", metrics_rows)
    write_report(
        output_dir / "验证阈值诊断报告.md",
        summaries=threshold_summaries,
        graph_summaries=graph_summaries,
        timing=timing,
    )
    return {
        "output_dir": str(output_dir),
        "trained_models": len(config.folds),
        "metrics_rows": len(metrics_rows),
        "graph_rows": len(graph_rows),
        "total_seconds": total_seconds,
    }


def main() -> None:
    """解析命令行参数并运行阈值诊断。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "research_outputs" / "validation_threshold_sweep",
    )
    args = parser.parse_args()
    result = run_sweep(
        SweepConfig(
            project_root=project_root,
            output_dir=args.output_dir,
            folds=tuple(args.folds),
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
