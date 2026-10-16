from dataloader import Synthetic_linear_data, Real_data, Real_data_MCAR, Real_data_MNAR
from experiments import Missing_experiments_MCAR, Missing_experiments
import numpy as np
import wandb
import time
import argparse
from utils import str2bool, positive_int, get_run_name
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Run models with different configurations.")
    parser.add_argument("--var_missing", type=lambda x: np.array(list(map(int, x.split(',')))),required=False,default=np.full(10, 1),
                        help="A list of 10 integers")
    parser.add_argument('--model', type=str, choices=['RF', 'histgbdt', 'XGBoost',], required=True, help='Model type')
    parser.add_argument('--mechanism', type=str, choices=['MCAR', 'MAR', 'MNAR'], default='MCAR', required=False, help='Missing mechanism')
    parser.add_argument('--imputation', type=str, choices=['mean', 'constant', 'MICE', 'iterative_ridge', 'None','missforest'],
                        required=True, help='Imputation method')
    parser.add_argument('--n_rep', type=positive_int, required=True, help='Number of repetitions, must be a positive integer')
    parser.add_argument('--method', type=str, choices=['CQR_MDA', 'QR_TrainCal','CQR','Oracle','Oracle_mean'], required=True, help='CQR_MDA')
    parser.add_argument('--exact', type=str2bool, required=True, help='Boolean indicating whether exact')
    parser.add_argument('--dataset_name', type=str,  required=True, help='dataset')
    parser.add_argument('--accept_reject', type=str2bool, required=False, default=False,help='If use accept_rejection method')
    parser.add_argument('--cal_size', type=positive_int, required=False, default=100,help='calibration dataset')
    parser.add_argument('--test_size_per_pattern', type=positive_int, required=False, default=100,help='test_size_per_pattern dataset')
    parser.add_argument('--reweight', type=str2bool, required=False, default=False,help='whether using reweight method')
    parser.add_argument('--RN_model', type=str, required=False, default='histgbdt',help='specify the likelihood model')
    parser.add_argument('--pre_imputation', type=str, required=False, default='MICE',help='pre_imputation')
    parser.add_argument('--prob_missing', type=float, required=False, default=0.2,help='prob_missing')
    parser.add_argument('--phi', type=float, required=False, default=0.8,help='phi')
    parser.add_argument('--reweight_with_y', type=str2bool, required=False, default=False,help='reweight_with_y')
    parser.add_argument('--ocacle_pre_impute', type=str2bool, required=False, default=False,help='ocacle_pre_impute')
    return parser.parse_args()

def main():
    args = get_args()
    # print(args.imputation, args.model)
    if args.imputation == 'None' and (args.model != 'histgbdt' and args.model != 'NNet_missing'):
        return
    # try:
    config = {"train_size": 500,
              "cal_size": args.cal_size,
              "test_size_iid": 2000,
              "test_size_per_pattern": args.test_size_per_pattern,
              "test_size_per_num_pattern": 100,
              "imputation": args.imputation,
              "n_rep": args.n_rep,
              "method": args.method,
              "basemodel": args.model,
              "dim": 10,
              "regression": 'Linear',
              "x_mean": 1,
              "phi": args.phi,
              "prob_missing":args.prob_missing,
              "var_missing": args.var_missing,
              "mechanism": args.mechanism,
              "cores": 8,
              "other_model_parameters": {'n_estimators':100,'min_samples_leaf':3,'max_features':'sqrt'},
              "exact": args.exact,
              "alpha": 0.1,
              "dataset_name": args.dataset_name,
              'accept_reject': args.accept_reject,
              'reweight': args.reweight,
              'RN_model': args.RN_model,
              'pre_imputation':args.pre_imputation,
              'ocacle_pre_impute':args.ocacle_pre_impute
              }

    run_name = ''

    if config['method']=='CQR_MDA':
        if config['accept_reject']:
            run_name += 'ARC-CP_'
        elif config['reweight']:
            run_name += 'Weighted-CP_'
        else:
            if config['exact']:
                run_name += 'Exact_'
            else:
                run_name += 'Nested_'
    else:
        run_name += config['method']
    # run_name = get_run_name(config)

    print(config)
    pre_impute = config['accept_reject'] or config['reweight']
    if config['dataset_name'] == 'synthetic':
        data = Synthetic_linear_data(n_rep=config['n_rep'],
                                     train_size=config['train_size'],
                                     cal_size=config['cal_size'],
                                     test_size_iid=config['test_size_iid'],
                                     test_size_per_pattern=config['test_size_per_pattern'],
                                     mechanism=config['mechanism'],
                                     missing_prob=config['prob_missing'],
                                     var_missing=np.array(config['var_missing']), var_dependent=[],
                                     rho=config['phi'],)
    elif config['dataset_name'] == 'simple_synthetic':
        from dataloader import Simple_synthetic
        data = Simple_synthetic(config['n_rep'], config['train_size'], config['cal_size'],
                                     config['test_size_iid'], config['test_size_per_pattern'],
                                     config['mechanism'], config['prob_missing'],
                                     np.array(config['var_missing']), [])

    else:
        if config['mechanism'] == 'MCAR':
            data = Real_data_MCAR(n_rep=config['n_rep'],
                                  prob_missing=config['prob_missing'],
                                  dataset_name=config['dataset_name'],
                                  cal_size=config['cal_size'],)
        elif config['mechanism'] == 'MAR':
            data = Real_data(
                name=config['dataset_name'],
                n_rep=config['n_rep'], mechanism='MAR',
                missing_prob=config['prob_missing'], var_missing=np.array(config['var_missing']),
                var_dependent=[],
                missing_function=None,
                seed=0,
                cal_size=config['cal_size'],
                test_size_per_pattern=config['test_size_per_pattern'],
            )
        else:
            data = Real_data_MNAR(
                name=config['dataset_name'],
                n_rep=config['n_rep'], mechanism=config['mechanism'],
                missing_prob=config['prob_missing'], var_missing=np.array(config['var_missing']),
                var_dependent=[],
                missing_function=None,
                seed=0,
                cal_size=config['cal_size'],
                test_size_per_pattern=config['test_size_per_pattern'],
            )
    data.generate_multiple_data()

    run_name += data.unique_name
    print(run_name)
    print(config['mechanism'])
    if config['mechanism']=='MCAR':
        exp = Missing_experiments_MCAR(data, pre_impute, config['pre_imputation'], config['imputation'],
                                       config['accept_reject'], config['reweight'], config['RN_model'], False,
                                       False, 0,
                                       config['basemodel'], config['alpha'], config['cores'], config['exact'],
                                       {'n_estimators': 100, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
                                       ocacle_pre_impute=config['ocacle_pre_impute'])
    else:
        exp = Missing_experiments(data, pre_impute, config['pre_imputation'], config['imputation'],
                                  config['accept_reject'], config['reweight'], config['RN_model'], False,
                                  False, 0,
                                  config['basemodel'], config['alpha'], config['cores'], config['exact'],
                                  {'n_estimators': 100, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
                                  ocacle_pre_impute=config['ocacle_pre_impute'])

    if config['dataset_name'] == 'simple_synthetic':
        exp.run_simple_synthetic()
    else:
        exp.run_experiments()
    results = exp.calculate_results_pattern()
    import os, pickle
    def save_variable(var, run_name, folder='./results'):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f'{run_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(var, f)
        print(f"Saved to {filepath}")
    save_variable(results, run_name)


if __name__ == '__main__':
    main()