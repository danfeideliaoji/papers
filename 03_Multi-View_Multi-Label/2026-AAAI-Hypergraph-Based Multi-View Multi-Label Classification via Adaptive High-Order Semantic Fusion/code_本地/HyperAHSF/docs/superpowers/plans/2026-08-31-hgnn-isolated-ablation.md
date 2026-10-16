# HyperAHSF Isolated HGNN Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a common behavior-preserving Emotions experiment harness and three independently branched HGNN ablations—node residual, dynamic hyperedge weight, and grouped hypergraph attention—then compare all three only against the same rerun baseline.

**Architecture:** The common branch adds a runner/reporting package and disabled-by-default ablation configuration without changing the active model path. Each mechanism branch starts from the exact common `BASE_COMMIT`, implements only its named model path, records mechanism-specific statistics, runs smoke and full experiments, and commits its own report. The summary branch also starts from `BASE_COMMIT` and receives only result artifacts and reports, never the three model implementations.

**Tech Stack:** Python 3.12, PyTorch 2.12.1 CUDA 12.6, PyTorch Geometric, NumPy, scikit-learn, PyYAML, pytest, Matplotlib/SVG, Git.

**Spec:** `C:\Users\A\.codex\attachments\85e8fd89-76aa-4eef-b502-e2001011c791\pasted-text-1.txt`

## Global Constraints

- Project root is exactly `C:\Users\A\Desktop\科研\HyperAHSF\HyperAHSF代码\HyperAHSF`.
- Dataset is Emotions with the original five folds, 125 epochs per fold, seed 0, and all optimizer/model/graph/loss settings from `config.yaml`.
- Smoke gate is fold 0 for 20 epochs; a finite successful smoke run is followed by the full run regardless of short-run metric direction.
- Do not alter hypergraph construction, contrastive loss, classifier, encoders, train/validation splits, or use validation labels for graph construction/tuning.
- `use_residual`, `learnable_hyperedge_weight`, and `use_attention` are disabled by default and satisfy `int(a)+int(b)+int(c) <= 1`; violations raise a Chinese `ValueError`.
- Never use destructive Git commands, broad staging, dataset/checkpoint/cache/log tracking, or overwrite existing results.
- All user-facing outputs live under `research_outputs/hgnn_ablation/<variant>/` and are written through a collision guard unless explicitly resuming the same incomplete run.

---

### Task 1: Preserve the original project and establish the runnable environment

**Files:**
- Modify: `.gitignore`
- Preserve exactly: `model.py`, `train.py`, `config.yaml`, `metrics.py`, `utils.py`, `data_preprocessor.py`, `README.md`
- Preserve exactly: `experiments/candidate1/**`, `docs/superpowers/plans/2026-08-30-*.md`

**Interfaces:**
- Consumes: existing project files and `E:\project\cuda\.venv\Scripts\python.exe`.
- Produces: an initial preservation commit, an ignored `.deps/` runtime directory if required, and a verified import command.

- [ ] **Step 1: Stage only source/config/tests/scripts/docs and the ignore file**

Run explicit `git add` paths for `.gitignore`, root Python/config/README files, `docs/`, and `experiments/`; do not stage `research_outputs/`, datasets, caches, or logs.

- [ ] **Step 2: Audit the staged file list**

Run: `git diff --cached --name-only`

Expected: only the explicitly allowed source/config/test/script/doc files.

- [ ] **Step 3: Commit the preservation snapshot**

Run: `git commit -m "chore: preserve original HyperAHSF sources"`

- [ ] **Step 4: Create the common branch and project-local dependency target**

Run: `git switch -c experiment/hgnn-ablation-base`

If imports remain missing, install only `torch-geometric`, `scikit-learn`, `PyYAML`, `matplotlib`, `tqdm`, and `psutil` to ignored `.deps/`; run all commands with `.deps` prepended to `PYTHONPATH` and the existing CUDA PyTorch interpreter.

- [ ] **Step 5: Verify imports and device metadata**

Expected imports: `torch`, `torch_geometric`, `numpy`, `sklearn`, `yaml`, `matplotlib`, `pytest`; record Torch/PyG versions, CUDA availability, GPU name, and total VRAM.

### Task 2: Add common ablation configuration and fail-first tests

**Files:**
- Create: `experiments/hgnn_ablation/__init__.py`
- Create: `experiments/hgnn_ablation/configuration.py`
- Create: `experiments/hgnn_ablation/tests/test_configuration.py`
- Modify: `model.py`

**Interfaces:**
- Produces: `AblationFlags(use_residual=False, learnable_hyperedge_weight=False, use_attention=False)`, `validate_ablation_flags(flags)`, and disabled model constructor arguments with no active-path change.
- Consumes: all later runners and branch implementations.

- [ ] **Step 1: Write failing mutual-exclusion/default tests**

```python
def test_default_flags_disable_every_mechanism():
    flags = AblationFlags()
    assert flags.enabled_count == 0

@pytest.mark.parametrize("flags", [
    AblationFlags(use_residual=True, use_attention=True),
    AblationFlags(use_residual=True, learnable_hyperedge_weight=True),
    AblationFlags(use_attention=True, learnable_hyperedge_weight=True),
])
def test_multiple_mechanisms_raise_chinese_error(flags):
    with pytest.raises(ValueError, match="三种机制只能单独启用"):
        validate_ablation_flags(flags)
```

- [ ] **Step 2: Run tests and verify failure because the interfaces do not exist**

Run: `python -m pytest experiments/hgnn_ablation/tests/test_configuration.py -q`

- [ ] **Step 3: Implement immutable flags and validation**

```python
@dataclass(frozen=True)
class AblationFlags:
    use_residual: bool = False
    learnable_hyperedge_weight: bool = False
    use_attention: bool = False

    @property
    def enabled_count(self) -> int:
        return sum(map(int, (self.use_residual, self.learnable_hyperedge_weight, self.use_attention)))

def validate_ablation_flags(flags: AblationFlags) -> None:
    if flags.enabled_count > 1:
        raise ValueError("三种机制只能单独启用，不能同时启用两个或三个机制。")
```

- [ ] **Step 4: Thread the three False defaults through model constructors without activating code**

All-false construction must create the same active modules, parameter initialization order, and output path as the preservation snapshot.

- [ ] **Step 5: Run configuration and baseline-consistency tests**

Use a fixed synthetic hypergraph and identical saved state dict; compare all-false outputs with `torch.testing.assert_close(..., rtol=0, atol=0)`.

### Task 3: Build the common experiment runner and artifact writers

**Files:**
- Create: `experiments/hgnn_ablation/runner.py`
- Create: `experiments/hgnn_ablation/reporting.py`
- Create: `experiments/hgnn_ablation/summarize.py`
- Create: `experiments/hgnn_ablation/tests/test_runner.py`
- Create: `experiments/hgnn_ablation/tests/test_reporting.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `ExperimentConfig`, `run_variant(config)`, `write_metrics`, `summarize_folds`, `write_timing`, `write_parameter_count`, `plot_learning_curves`, and a CLI.
- CSV metric columns: `variant`, `fold`, `epoch`, `loss`, `Hamming Loss`, `Average Precision`, `One Error`, `Ranking Loss`, `Subset Accuracy`, `Coverage`, `Macro-F1`, `Micro-F1`.
- Output files: `config.json`, `metrics.csv`, `fold_summary.csv`, `timing.json`, `parameter_count.json`, `learning_curves.svg`, `单机制实验报告.md`.

- [ ] **Step 1: Write fail-first tests for schemas, collision protection, fold selection, finite-loss rejection, and exact config serialization**

```python
def test_metrics_schema_contains_required_columns():
    assert REQUIRED_METRIC_COLUMNS <= set(metric_row(...))

def test_existing_nonempty_output_is_not_overwritten(tmp_path):
    target = tmp_path / "baseline" / "metrics.csv"
    target.parent.mkdir()
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_dir(target.parent, allow_resume=False)
```

- [ ] **Step 2: Run runner/reporting tests and verify they fail**

- [ ] **Step 3: Implement the runner using the unchanged full-fold training/evaluation path**

For each fold: reset seed 0, load train/val from existing pickle files, construct one model, AdamW, and unchanged `PrecomputedContrastiveLoss`; record every epoch's loss, metrics, wall time, CUDA peak memory, and final mechanism statistics.

- [ ] **Step 4: Implement fold summaries and plotting**

For each metric/fold store best value/epoch and final value; aggregate mean/sample standard deviation without cherry-picking. Generate SVG curves from all folds.

- [ ] **Step 5: Verify CPU and optional CUDA smoke paths with one synthetic epoch**

- [ ] **Step 6: Run all original Candidate1 tests plus new harness tests**

- [ ] **Step 7: Commit the common harness**

Run: `git commit -m "experiment: add isolated HGNN ablation harness"`; record this hash as `BASE_COMMIT`.

### Task 4: Rerun the common baseline

**Files:**
- Create at runtime: `research_outputs/hgnn_ablation/baseline/*`

**Interfaces:**
- Produces: the sole baseline used by all branch reports and the final summary.

- [ ] **Step 1: Run syntax and all tests on `experiment/hgnn-ablation-base`**

- [ ] **Step 2: Run `baseline`, folds 0–4, 125 epochs, seed 0**

- [ ] **Step 3: Validate 625 metric rows, finite losses, complete files, timing/VRAM/parameter data, and five fold summaries**

- [ ] **Step 4: Commit only the required baseline CSV/JSON/SVG/report artifacts on the common branch if the result policy requires them in `BASE_COMMIT`; otherwise preserve them for the summary branch without changing `BASE_COMMIT`**

The three mechanism branches must still point directly to the already-recorded harness `BASE_COMMIT`.

### Task 5: Residual-only branch

**Files:**
- Modify: `model.py`
- Modify: `experiments/hgnn_ablation/runner.py` only to declare branch support/stat reporting, not to add another mechanism
- Create: `experiments/hgnn_ablation/tests/test_residual.py`
- Create at runtime: `research_outputs/hgnn_ablation/residual_only/*`

**Interfaces:**
- Produces: `residual_projection`, an `Identity` when `in_dim == hid_dim` or bias-free `Linear(in_dim, hid_dim)` otherwise.

- [ ] **Step 1: Switch directly from `BASE_COMMIT` to new branch `experiment/residual-only`**

- [ ] **Step 2: Write failing tests for projection type, `act(n_agg + projection(x))`, fixed coefficient 1, gradients, shape, all-false exact equality, and other mechanisms disabled**

- [ ] **Step 3: Implement only node residual propagation**

```python
residual = self.residual_projection(x)
n = self.act(n_agg + residual) if self.use_residual else self.act(n_agg)
```

Do not add residuals to `e_agg`, gates, normalization, attention, weighting, or fusion.

- [ ] **Step 4: Run syntax, original/new tests, CPU, optional CUDA, gradient, shape, exact off-equivalence, and exclusivity checks**

- [ ] **Step 5: Commit `experiment: add isolated node residual propagation`**

- [ ] **Step 6: Run fold0×20 smoke, then folds0–4×125 full experiment; write `residual_only` artifacts and report**

### Task 6: Dynamic-hyperedge-weight-only branch

**Files:**
- Modify: `model.py`
- Modify: `experiments/hgnn_ablation/runner.py` only for branch support/stat collection
- Create: `experiments/hgnn_ablation/tests/test_hyperedge_weight.py`
- Create at runtime: `research_outputs/hgnn_ablation/hyperedge_weight_only/*`

**Interfaces:**
- Produces: one shared scorer per HGNN layer, positive mean-one edge weights, `last_hyperedge_weights`, and distribution/statistics output.

- [ ] **Step 1: Switch directly from `BASE_COMMIT` to `experiment/hyperedge-weight-only`**

- [ ] **Step 2: Write failing tests for graph-size independence, positivity, mean≈1, nonzero scorer gradient, W-position scaling in edge→node messages, returned-edge semantics, off-equivalence, and disabled residual/attention**

- [ ] **Step 3: Implement dynamic weights from current aggregated edge context**

```python
raw_weight = self.hyperedge_weight_scorer(e_agg).squeeze(-1)
weight = F.softplus(raw_weight) + self.weight_eps
weight = weight / weight.mean().clamp_min(self.weight_eps)
```

For `e2n`, each membership message is `norm * weight[edge_id] * e_agg[edge_id]`. Apply the same scalar diagonal `W` to the legacy returned hyperedge representation and document that the code's returned mean-normalized raw-edge representation is an implementation proxy for paper `Q`, not a literal unnormalized `H^T Z`.

- [ ] **Step 4: Run all required tests and commit `experiment: add isolated dynamic hyperedge weighting`**

- [ ] **Step 5: Run smoke and full experiments; save mean/std/min/max and a weight-distribution SVG/CSV plus the standard files/report**

### Task 7: Grouped-hypergraph-attention-only branch

**Files:**
- Modify: `model.py`
- Modify: `experiments/hgnn_ablation/runner.py` only for branch support/stat collection
- Create: `experiments/hgnn_ablation/tests/test_attention.py`
- Create at runtime: `research_outputs/hgnn_ablation/hypergraph_attention_only/*`

**Interfaces:**
- Produces: `grouped_softmax(scores, target_index, num_targets)`, four-head n2e/e2n attention, stored direction statistics, and singleton self-loop validation.

- [ ] **Step 1: Switch directly from `BASE_COMMIT` to `experiment/hypergraph-attention-only`**

- [ ] **Step 2: Write failing tests for group sums, correct grouping IDs, head independence/concatenation, singleton=1, empty edges, isolated nodes, gradients, shape, off-equivalence, and disabled residual/weighting**

```python
alpha_n2e = grouped_softmax(score_n2e, edge_id, num_edges)
alpha_e2n = grouped_softmax(score_e2n, node_id, num_nodes)
torch.testing.assert_close(group_sum(alpha_n2e, edge_id), torch.ones(...))
torch.testing.assert_close(group_sum(alpha_e2n, node_id), torch.ones(...))
```

- [ ] **Step 3: Implement stable target-group softmax per head**

Subtract each target/head maximum, exponentiate, divide by the target/head sum, and return an empty tensor unchanged. Attention replaces degree averaging in the enabled path so coefficients themselves sum to one per target/head; concatenate four 32-dimensional head messages to 128 dimensions.

- [ ] **Step 4: Store entropy/min/max per direction and assert every one-member self-loop edge has n2e attention exactly one within tolerance**

- [ ] **Step 5: Run all required tests and commit `experiment: add isolated grouped hypergraph attention`**

- [ ] **Step 6: Run smoke and full experiments; save attention statistics and standard files/report**

### Task 8: Create the results-only summary branch

**Files:**
- Create: `research_outputs/hgnn_ablation/aggregate_results.csv`
- Create: `research_outputs/hgnn_ablation/metric_comparison.svg`
- Create: `research_outputs/hgnn_ablation/runtime_comparison.svg`
- Create: `research_outputs/hgnn_ablation/最终对比报告.md`
- Create/modify: reproduction documentation under `research_outputs/hgnn_ablation/`

**Interfaces:**
- Consumes: baseline plus three independent result directories.
- Produces: required comparison table, absolute/relative deltas, folds-won counts, best/final epochs, stability/overfitting analysis, mechanism answers, branch/commit/status inventory, and reproduction commands.

- [ ] **Step 1: Switch directly from `BASE_COMMIT` to `experiment/hgnn-ablation-summary`**

- [ ] **Step 2: Copy/cherry-pick only common harness and required CSV/JSON/SVG/Markdown artifacts—not branch model implementations**

- [ ] **Step 3: Write fail-first aggregate tests for four rows, required columns, improvement direction, percentage calculations, and five-fold counts**

- [ ] **Step 4: Generate aggregate files and the Chinese final report using only the saved complete five-fold results**

- [ ] **Step 5: Verify the summary branch does not contain residual, scorer, or attention implementation diffs relative to `BASE_COMMIT`**

- [ ] **Step 6: Commit the summary and record its hash**

### Task 9: Final audit

**Files:**
- Verify all required outputs and Git metadata; no new implementation files.

**Interfaces:**
- Produces: final user handoff with `BASE_COMMIT`, four branch names, four implementation/report hashes, every branch's status, exact reproduction commands, and evidence-backed conclusions.

- [ ] **Step 1: Run syntax/tests on every branch and validate all output schemas/counts**

- [ ] **Step 2: Audit tracked files for `*.pt`, `*.pth`, `*.pickle`, `*.mat`, caches, logs, datasets, and unrelated user files; expect none**

- [ ] **Step 3: Verify all three experiment branch merge-bases equal `BASE_COMMIT` and no mechanism branch descends from another mechanism branch**

- [ ] **Step 4: Capture clean/expected `git status --short` for every branch**

- [ ] **Step 5: Stop after reporting; do not implement or run any combined mechanism**
