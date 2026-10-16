#!/bin/bash
export PYTHONUNBUFFERED=1

MAX_JOBS=8

commands=(
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact false --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject true --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight true --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MAR --test_size_per_pattern 100 --cal_size 100"

"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MNAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact false --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MNAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject true --reweight false --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MNAR --test_size_per_pattern 100 --cal_size 100"
"OMP_NUM_THREADS=8 python main.py --prob_missing 0.2 --accept_reject false --reweight true --model RF --imputation iterative_ridge --n_rep 100 --exact true --dataset_name synthetic --method CQR_MDA --pre_imputation MICE --var_missing 0,0,0,0,0,1,1,1,1,1 --mechanism MNAR --test_size_per_pattern 100 --cal_size 100"
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