# On the consistency of supervised learning with missing values

**Authors: Julie Josse (CMAP, Inria), Nicolas Prost (CMAP, Inria), Erwan Scornet (CMAP), Gaël Varoquaux (Inria).**

## Simple example notebook

**Update**: the directory **Notebook** contains a tutorial on key results of the paper.

Access the Notebok via Binder at the following URL 

**Notes**
Without Binder, install the required Python libraries via
```
pip install -r requirements.txt
```

## Code for the simulations in the paper

This repository contains the code for the paper:

Julie Josse, Nicolas Prost, Erwan Scornet, Gaël Varoquaux. On the consistency of supervised learning with missing values. 2019. 〈hal-02024202〉https://arxiv.org/abs/1902.06931

The directory **analysis** contains the code for figures 1 and 2 (section 5).

**boxplots** corresponds to figures 3 and 4 (section 6). There are three separate files: one containing the functions, one containing the script for computation, and two for the visualisation (one of each of the two boxplots).

**consistency** is used for figure 5 (section 6). There are three files as for the boxplot, but in addition, approximate Bayes rates are computed in *bayesrates.R* with oracle multiple imputation, as detailed in the paper. 

The scripts require the following R packages:
```r
rpart
party
ranger
xgboost
MASS
norm
doParallel
doSNOW
gridExtra
viridis
```

To build the figures for R2 values and computation times, run the scripts in the boxplots folder:

```bash
Rscript script_boxplots.R

Rscript post_process_boxplot.R
python3 plot_grid_boxplots.py

Rscript train_test_times_post_process_boxplot.R
python3 train_test_times_plot_grid_boxplots.py
```

You may adjust hyperparameters for the experiments by changing the variables on lines 43-52 in `script_boxplots.R`.

All figure outputs go to the directory **figures** (created when nonexistent). Be careful that some files
may be overwritten when running the code repeatedly.

Nicolas Prost

July 10, 2019

