import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from decimal import Decimal, ROUND_HALF_UP

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# this function helps us print numbers in a prettier way
def round_to_sf_sci(number, sf=2):
    # Convert the number to a Decimal with the desired significant figures
    decimal_number = Decimal(str(number))
    
    # Round the Decimal to the specified number of significant figures
    rounded_decimal = round(decimal_number, sf - decimal_number.adjusted())
    
    sci_notation = rounded_decimal
    if rounded_decimal > 1000 or rounded_decimal < -1000:
        # Convert to scientific notation with two decimal places
        sci_notation = f"{rounded_decimal:.2e}"
    
    return sci_notation

def create_coefficient_table(data, filename):

    # learning_methods = set(data['forest'])
    # hard code the order of the learning methods
    learning_methods = ['DECISION TREE', 'RANDOM FOREST', 'XGBOOST', 'SVM', 'KNN']

    
    results_df = pd.DataFrame()
    subset_data = data[data['forest'] == 'DECISION TREE']

    # get the full list of imputation methods for the full table
    # full_table_methods = list(set(subset_data['method']))

    # hard code the order of the imputation methods
    full_table_methods = ['ctree', 'ctree + mask', 'rpart', 'rpart + mask', 'Gaussian', 'Gaussian + mask', 'oor', 'oor + mask',
                            'mean', 'mean + mask', 'MIA']

    full_table_methods.append('const')

    results_df['Imputation Method'] = full_table_methods
    for learning_method in learning_methods:
        subset_data = data[data['forest'] == learning_method]
        
        method_names = list(set(subset_data['method']))
        
        # create categorical dummy variables
        dummied_data = pd.get_dummies(subset_data, columns=['method', 'fold'], drop_first=True)
        # drop the columns of data that we don't want to include in our regression
        dummied_data = dummied_data.drop(columns=['index', 'score', 'forest', 'R2'])

        dummied_data = sm.add_constant(dummied_data)

        # get the target variable
        y = subset_data['R2']

        # Fit the GLM (linear model)
        model = sm.OLS(y, dummied_data)
        results = model.fit()

        # not every learning algorithm has every imputation method, so make
        # full length arrays for coef and pval values and populate it with
        # na values
        coef = [np.nan for _ in range(len(full_table_methods))]
        pval = [np.nan for _ in range(len(full_table_methods))]

        for i in range(len(method_names)+1):
            # get the current method_name that we're working with
            method_name = results.params.index[i].removeprefix('method_')
            try:
                # get the index in the full table for the imputation method we are working
                # with right now
                index = full_table_methods.index(method_name)
                
                coef[index] = round_to_sf_sci(results.params[i], sf=2)
                pval[index] = round_to_sf_sci(results.pvalues[i], sf=2)
            except ValueError:
                pass

        # add the coefficient and p-values to the final table dataframe
        results_df['Coef_'+learning_method] = coef
        results_df['Pval_'+learning_method] = pval

    with PdfPages(filename) as pdf:
        fix, ax = plt.subplots()

        # Hide the axes
        ax = plt.gca()
        ax.axis('off')

        ax.table(cellText=results_df.values, colLabels=results_df.keys(), loc='center')

        # code for setting font size
        # table.auto_set_font_size(False)
        # table.set_fontsize(4)

        pdf.savefig()
        plt.close()