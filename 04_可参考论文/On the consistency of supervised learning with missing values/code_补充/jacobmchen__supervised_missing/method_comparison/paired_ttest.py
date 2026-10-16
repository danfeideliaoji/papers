import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def return_empty_matrix(n):
    pval_matrix = []
    for i in range(n):
        arr = []
        for j in range(n):
            # if the indexes are the same then there is no need to make a statistical test
            if i == j:
                arr.append(np.nan)
            else:
                arr.append(0)
        pval_matrix.append(arr)
    return pval_matrix

def xaxis_label_format(labels):
    formatted = []

    # add a line break for every other label so that the x axis labels
    # don't overlap with each other on the table
    for i in range(len(labels)):
        if i % 2 == 1:
            formatted.append('\n' + labels[i])
        else:
            formatted.append(labels[i])
    return formatted

def paired_ttest(data, filename):
    # find the unique categories in method
    # method_categories = set(data['method'])
    # method_categories = list(method_categories)
    # method_categories.remove('rpart')
    # method_categories.remove('rpart + mask')
    # method_categories.remove('ctree')
    # method_categories.remove('ctree + mask')
    # method_categories.remove('MIA')

    # hard code the method categories and the order of the method categories
    # method_categories = ['Gaussian', 'Gaussian + mask', 'oor', 'oor + mask', 'mean', 'mean + mask']

    # hard code the learning methods
    learning_methods = ['DECISION TREE', 'RANDOM FOREST', 'XGBOOST', 'SVM', 'KNN']

    # Create a PDF file to save the plots
    pdf_filename = filename
    pdf_pages = PdfPages(pdf_filename)

    for learning_method in learning_methods:
        # hard code the method categories depending on which learning method we are using
        if learning_method == 'DECISION TREE':
            method_categories = ['ctree', 'ctree + mask', 'rpart', 'rpart + mask', 'Gaussian', 'Gaussian + mask', 
                                 'oor', 'oor + mask', 'mean', 'mean + mask', 'MIA']
        elif learning_method == 'RANDOM FOREST' or learning_method == 'XGBOOST':
            method_categories = ['Gaussian', 'Gaussian + mask', 
                                 'oor', 'oor + mask', 'mean', 'mean + mask', 'MIA']
        else:
            method_categories = ['Gaussian', 'Gaussian + mask', 'oor', 'oor + mask', 'mean', 'mean + mask']

        # consider each learning method separately
        subset_data = data[data['forest'] == learning_method]
        # print(learning_method)
        # print(subset_data)
        pval_matrix = return_empty_matrix(len(method_categories))

        # execute statistical tests
        for i in range(len(method_categories)):
            for j in range(i+1, len(method_categories)):
                # compare method i and method j by extracting their respective R2 values and
                # using a paired t test to compare them
                method_i = subset_data[subset_data['method'] == method_categories[i]]['R2']
                # print(method_categories[i], np.mean(method_i), np.std(method_i))

                method_j = subset_data[subset_data['method'] == method_categories[j]]['R2']
                # print(method_categories[j], np.mean(method_j), np.std(method_j))

                # calculate the test statistic
                t_statistic, pval = stats.ttest_rel(method_i, method_j)

                pval_matrix[i][j] = pval
                pval_matrix[j][i] = pval
                # print out the test if the p-value is less than alpha = 0.05
                # if pval < 0.05:
                #     print(method_categories[i], 'and', method_categories[j], 'are statistically significant, pval=', pval)
        
        # code below makes plots
        column_labels = method_categories
        row_labels = xaxis_label_format(method_categories)
        matrix_data = np.array(pval_matrix)

        # Apply a color map based on values between 0 and 0.5
        cmap = plt.cm.get_cmap('coolwarm')  # Choose a colormap, for example, 'coolwarm'

        # Set values outside the range to be transparent
        norm = plt.Normalize(vmin=0, vmax=0.5)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Display the matrix with colors
        img = ax.imshow(matrix_data, cmap=cmap, norm=norm)
        
        if learning_method == 'DECISION TREE':
            fontsize = 5
        else:
            fontsize = 7

        # Add labels to columns and rows
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(row_labels, fontsize=fontsize)
        ax.set_yticklabels(column_labels, fontsize=fontsize)

        # ax.set_xticklabels(column_labels, rotation=90)  # Rotate x-labels vertically

        ax.set_title(learning_method)

        # Display numbers inside each cell
        for i in range(len(row_labels)):
            for j in range(len(column_labels)):
                text = ax.text(j, i, f'{matrix_data[i, j]:.3f}', ha='center', va='center', color='black', fontsize=fontsize)

        # Add a colorbar for reference
        cbar = plt.colorbar(img, ax=ax)

        # Save the plot to the PDF file
        pdf_pages.savefig()

        # Close the current figure to release resources
        plt.close()

    # Close the PDF file
    pdf_pages.close()