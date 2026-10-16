#!/bin/bash
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export XGBOOST_NUM_THREADS=16

MAX_JOBS=3

commands=(
"OMP_NUM_THREADS=16 python main.py --prob_missing 0.2 --accept_reject true --reweight false --model RF --imputation iterative_ridge --n_rep 50 --exact true --dataset_name meps_19 --method CQR_MDA --pre_imputation MICE --mechanism MCAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=16 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 50 --exact true --dataset_name meps_19 --method CQR_MDA --pre_imputation MICE --mechanism MCAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=16 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 50 --exact false --dataset_name meps_19 --method CQR_MDA --pre_imputation MICE --mechanism MCAR --test_size_per_pattern 100 --cal_size 100"
)


current_jobs=0

for cmd in "${commands[@]}"; do
    eval "$cmd" &
    sleep 5

    ((current_jobs++))

    if ((current_jobs >= MAX_JOBS )); then
        wait -n
        ((current_jobs--))
    fi
done

wait

