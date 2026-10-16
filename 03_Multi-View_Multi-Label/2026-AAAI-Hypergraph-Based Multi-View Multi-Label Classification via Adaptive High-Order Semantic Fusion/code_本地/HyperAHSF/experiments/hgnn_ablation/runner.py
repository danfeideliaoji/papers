"""Emotions 公共基线与单机制消融的运行入口。"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from data_preprocessor import MyDataset_k_fold
from metrics import test_metrics
from model import HGMVMLC_Model
from utils import PrecomputedContrastiveLoss

from .configuration import AblationFlags, validate_ablation_flags
from .reporting import (
    metric_row,
    plot_learning_curves,
    prepare_output_dir,
    summarize_folds,
    write_csv,
    write_json,
    write_variant_report,
)


VARIANTS = (
    "baseline",
    "residual_only",
    "hyperedge_weight_only",
    "hypergraph_attention_only",
)
SUPPORTED_VARIANTS: FrozenSet[str] = frozenset({"baseline"})


@dataclass(frozen=True)
class ExperimentConfig:
    """一次可复现实验的不可变公共配置。"""

    project_root: Path
    output_root: Path
    variant: str = "baseline"
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 125
    seed: int = 0
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant 必须属于 {VARIANTS}")
        if not self.folds or any(fold not in range(5) for fold in self.folds):
            raise ValueError("fold 必须是 0..4 中至少一个编号。")
        if self.epochs <= 0:
            raise ValueError("epochs 必须为正整数。")


def flags_for_variant(variant: str) -> AblationFlags:
    """把实验名称映射为恰好零个或一个机制开关。"""

    mapping = {
        "baseline": AblationFlags(),
        "residual_only": AblationFlags(use_residual=True),
        "hyperedge_weight_only": AblationFlags(learnable_hyperedge_weight=True),
        "hypergraph_attention_only": AblationFlags(use_attention=True),
    }
    try:
        flags = mapping[variant]
    except KeyError as exc:
        raise ValueError(f"未知实验名称: {variant}") from exc
    validate_ablation_flags(flags)
    return flags


def validate_branch_support(
    variant: str,
    *,
    supported_variants: FrozenSet[str] = SUPPORTED_VARIANTS,
) -> None:
    """避免在不含相应实现的分支静默运行错误变体。"""

    if variant not in supported_variants:
        raise ValueError(f"当前分支不支持实验 {variant}；支持项为 {sorted(supported_variants)}。")


def fix_seed(seed: int) -> None:
    """复现原训练入口的 Python、NumPy、Torch 与 cuDNN 随机设置。"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.Generator().manual_seed(seed)


def _environment(device: torch.device) -> dict[str, object]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device) if cuda else None,
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory) if cuda else 0
        ),
    }


def _serialize_config(
    config: ExperimentConfig,
    params: dict[str, object],
    flags: AblationFlags,
    environment: dict[str, object],
) -> dict[str, object]:
    return {
        "variant": config.variant,
        "dataset": "emotions",
        "folds": list(config.folds),
        "epochs": config.epochs,
        "seed": config.seed,
        "device": config.device,
        "project_root": str(config.project_root),
        "output_root": str(config.output_root),
        "flags": {
            "use_residual": flags.use_residual,
            "learnable_hyperedge_weight": flags.learnable_hyperedge_weight,
            "use_attention": flags.use_attention,
        },
        "params": params,
        "environment": environment,
    }


def _validate_gradients(model: nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("训练后没有任何参数梯度。")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise FloatingPointError("检测到 NaN 或 Inf 梯度。")


def _train_epoch(
    dataset: MyDataset_k_fold,
    model: HGMVMLC_Model,
    optimizer: torch.optim.Optimizer,
    contrastive_loss: PrecomputedContrastiveLoss,
    seed: int,
) -> float:
    """完全复用原始分类损失、标签对比损失与每 epoch 固定种子语义。"""

    fix_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, _, specific_representations = model(dataset.features)
    fix_seed(seed)
    classification_loss = nn.BCELoss()(
        torch.sigmoid(output), dataset.labels.to(torch.float32)
    )
    contrastive = contrastive_loss(specific_representations)
    loss = classification_loss + contrastive
    if not torch.isfinite(loss).item():
        raise FloatingPointError("训练损失出现 NaN 或 Inf。")
    loss.backward()
    _validate_gradients(model)
    optimizer.step()
    return float(loss.detach().cpu())


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """运行一个受当前分支支持的独立变体并保存完整公共产物。"""

    validate_branch_support(config.variant)
    flags = flags_for_variant(config.variant)
    project_root = Path(config.project_root).resolve()
    output_dir = prepare_output_dir(Path(config.output_root).resolve() / config.variant)
    with (project_root / "config.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["emotions"]

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 实验，但当前 CUDA 不可用。")
    device = requested_device
    environment = _environment(device)
    metrics_rows: list[dict[str, object]] = []
    fold_timings: list[dict[str, object]] = []
    trainable_parameters = 0
    total_parameters = 0
    overall_start = time.perf_counter()
    peak_gpu_memory_bytes = 0

    for fold in config.folds:
        fix_seed(config.seed)
        train_dataset = MyDataset_k_fold(
            params["num_views"], params["name"], fold, "train", device=device, scalers=None
        )
        val_dataset = MyDataset_k_fold(
            params["num_views"],
            params["name"],
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
        contrastive = PrecomputedContrastiveLoss(
            train_dataset.labels, float(params["w_intra"])
        )
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fold_start = time.perf_counter()
        epoch_times: list[float] = []
        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            loss = _train_epoch(
                train_dataset, model, optimizer, contrastive, config.seed
            )
            metric_pairs = test_metrics(
                model=model,
                test_features=val_dataset.features,
                labels=val_dataset.labels,
                params=params,
                mode="val",
            )
            epoch_seconds = time.perf_counter() - epoch_start
            epoch_times.append(epoch_seconds)
            metrics_rows.append(
                metric_row(
                    variant=config.variant,
                    fold=fold,
                    epoch=epoch,
                    loss=loss,
                    metrics=dict(metric_pairs),
                    epoch_seconds=epoch_seconds,
                )
            )
        fold_seconds = time.perf_counter() - fold_start
        if device.type == "cuda":
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
        fold_timings.append(
            {
                "fold": fold,
                "seconds": fold_seconds,
                "average_epoch_seconds": float(np.mean(epoch_times)),
            }
        )
        del model, optimizer, contrastive, train_dataset, val_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - overall_start
    fold_summary = summarize_folds(metrics_rows)
    timing: dict[str, object] = {
        "variant": config.variant,
        "fold_count": len(config.folds),
        "epochs_per_fold": config.epochs,
        "total_seconds": total_seconds,
        "average_epoch_seconds": float(
            np.mean([float(row["epoch_seconds"]) for row in metrics_rows])
        ),
        "folds": fold_timings,
        "gpu_model": environment["gpu_model"],
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
    parameter_data = {
        "variant": config.variant,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    write_json(
        output_dir / "config.json",
        _serialize_config(config, params, flags, environment),
    )
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "fold_summary.csv", fold_summary)
    write_json(output_dir / "timing.json", timing)
    write_json(output_dir / "parameter_count.json", parameter_data)
    plot_learning_curves(output_dir / "learning_curves.svg", metrics_rows)
    write_variant_report(
        output_dir / "单机制实验报告.md",
        variant=config.variant,
        summaries=fold_summary,
        timing=timing,
        trainable_parameters=trainable_parameters,
    )
    return {
        "variant": config.variant,
        "output_dir": str(output_dir),
        "metrics_rows": len(metrics_rows),
        "folds": len(fold_summary),
    }


def main() -> None:
    """解析公共实验参数并打印机器可读运行摘要。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "research_outputs" / "hgnn_ablation",
    )
    args = parser.parse_args()
    result = run_experiment(
        ExperimentConfig(
            project_root=project_root,
            output_root=args.output_root,
            variant=args.variant,
            folds=tuple(args.folds),
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
