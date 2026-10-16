
# CP-General-Missing: Reproducibility Instructions

This repository accompanies the AISTATS 2026 submission *"Mask-Conditional Conformal Prediction: Valid Uncertainty For All  Missing Data Mechanisms"*.

It supports reproducibility: from generating synthetic and real datasets, to running experiments and visualizing results.

## Requirements
We recommend Python 3.11.
All dependencies are listed in `requirements.txt`.  
To install:

    pip install -r requirements.txt

We recommend running experiments in a virtual environment (e.g., `venv`, `conda`).

## Project Structure

cp_general_missing/
├── data/                  # Auto-generated folder for training/calibration/test data
├── results/               # Output folder for experiment results
├── *.py                   # Experiment scripts and model implementations
├── *.sh                   # Shell scripts for running full experiments
├── *.ipynb                # Jupyter notebooks to visualize experiment results
└── requirements.txt       # Python dependencies

## Running Experiments

To generate datasets and run full experiments, simply run one of the shell scripts:

    bash run_MCAR_concrete.sh            # Concrete dataset, MCAR
    bash run_MAR_concrete.sh             # Concrete dataset, MAR
    bash run_MCAR_synthetic.sh           # Synthetic data, MCAR
    bash run_MARMNAR_synthetic.sh        # Synthetic data, MAR & MNAR

Each script will:

- Automatically generate data in `./data/`
- Save experimental results to `./results/`

## Visualizing Results

After running experiments, open the corresponding notebook to generate plots and summaries:

- `concrete_results.ipynb` — Visualizes concrete results
- `ablation_impute_without_correction.ipynb` — Shows ablation study without correction
- `synthetic_results_all.ipynb` — Summarizes all synthetic experiments
- `Quality_density_ratio_Discussion.ipynb` — Analyzes estimator quality vs. coverage

## Notes

- All synthetic datasets are auto-generated during script execution.

## License

This code is released under the MIT License. 

