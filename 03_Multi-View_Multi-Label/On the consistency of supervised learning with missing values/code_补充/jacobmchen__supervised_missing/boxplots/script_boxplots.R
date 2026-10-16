#!/usr/bin/env Rscript
cmd.args = commandArgs(trailingOnly=TRUE)
cmd.args = sapply(cmd.args, as.numeric)

if (length(cmd.args) > 0) {
    n.jobs = cmd.args[1]
} else n.jobs = 1

if (length(cmd.args) > 1) {
    num.threads = cmd.args[2]
} else num.threads = 1

# This runs consistency results (score: MSE) for different parameters,
# in parallel, with n.jobs jobs and num.threads.ranger cores per forest

dir.create(file.path('boxplots/results'), showWarnings=FALSE)

cat("This script runs consistency results for
    * one dataset: Gaussian X, Y=X*2+noise
    * three mechanisms: 1. MCAR, 2. MNAR, 3. PRED
    * two tree models: rpart, ctree, one forest model: ranger, one boosting model: xgboost
    * several methods: surrogates (rpart, ctree) (+ mask), gaussian imputation (+ mask),
    mean imputation (+ mask), MIA, oor imputation (+ mask) 
    ")
cat("Starting R script\n")

set.seed(42)  # so .Random.seed exists

library(norm)
rngseed(123)  # for the EM imputation

# 1 for figure 3, 2 for figure 4, -1 for both
boxplot_choice <- -1

################################################################################
library(doParallel)
library(doSNOW)

cl <- makeCluster(n.jobs, outfile="")
registerDoSNOW(cl)

Parallel <- function(dataset, n_features, num.threads.ranger=num.threads) {
    iter.seed <- 15
    sizes <- c(2000)
    # sizes <- c(10)
    n_rep <- 100
    # n_rep <- 1000
    prob <- 0.2
    # prob <- 0.5
    noise = 0.1
    min_samples_leaf = 30
    rho = 0.5
    results.list <- foreach (param = list(
        "rpart" = list(dataset=dataset, model='rpart', strategy='none', withpattern=FALSE)
        ,
        "rpart + mask" = list(dataset=dataset, model='rpart', strategy='none', withpattern=TRUE)
        ,
        "rpart mean" = list(dataset=dataset, model='rpart', strategy='mean', withpattern=FALSE)
        ,
        "rpart mean + mask" = list(dataset=dataset, model='rpart', strategy='mean', withpattern=TRUE)
        ,
        "rpart oor" = list(dataset=dataset, model='rpart', strategy='oor', withpattern=FALSE)
        ,
        "rpart oor + mask" = list(dataset=dataset, model='rpart', strategy='oor', withpattern=TRUE)
        ,
        "rpart gaussian" = list(dataset=dataset, model='rpart', strategy='gaussian', withpattern=FALSE)
        ,
        "rpart gaussian + mask" = list(dataset=dataset, model='rpart', strategy='gaussian', withpattern=TRUE)
        ,
        "rpart mia" = list(dataset=dataset, model='rpart', strategy='mia', withpattern=FALSE)
        ,
        "ctree" = list(dataset=dataset, model='ctree', strategy='none', withpattern=FALSE)
        ,
        "ctree + mask" = list(dataset=dataset, model='ctree', strategy='none', withpattern=TRUE)
        ,
        "ranger mean" = list(dataset=dataset, model='ranger', strategy='mean', withpattern=FALSE)
        ,
        "ranger mean + mask" = list(dataset=dataset, model='ranger', strategy='mean', withpattern=TRUE)
        ,
        "ranger oor" = list(dataset=dataset, model='ranger', strategy='oor', withpattern=FALSE)
        ,
        "ranger oor + mask" = list(dataset=dataset, model='ranger', strategy='oor', withpattern=TRUE)
        ,
        
        "ranger gaussian" = list(dataset=dataset, model='ranger', strategy='gaussian', withpattern=FALSE)
        ,
        "ranger gaussian + mask" = list(dataset=dataset, model='ranger', strategy='gaussian', withpattern=TRUE)
        ,
        "ranger mia" = list(dataset=dataset, model='ranger', strategy='mia', withpattern=FALSE)
        ,
        "xgboost mean" = list(dataset=dataset, model='xgboost', strategy='mean', withpattern=FALSE)
        ,
        "xgboost mean + mask" = list(dataset=dataset, model='xgboost', strategy='mean', withpattern=TRUE)
        ,
        "xgboost oor" = list(dataset=dataset, model='xgboost', strategy='oor', withpattern=FALSE)
        ,
        "xgboost oor + mask" = list(dataset=dataset, model='xgboost', strategy='oor', withpattern=TRUE)
        ,
        
        "xgboost gaussian" = list(dataset=dataset, model='xgboost', strategy='gaussian', withpattern=FALSE)
        ,
        "xgboost gaussian + mask" = list(dataset=dataset, model='xgboost', strategy='gaussian', withpattern=TRUE)
        ,
        # why does this strategy have to be none?
        "xgboost mia" = list(dataset=dataset, model='xgboost', strategy='none', withpattern=FALSE)
        ,
        "svm mean" = list(dataset=dataset, model='svm', strategy='mean', withpattern=FALSE)
        ,
        "svm mean + mask" = list(dataset=dataset, model='svm', strategy='mean', withpattern=TRUE)
        ,
        "svm oor" = list(dataset=dataset, model='svm', strategy='oor', withpattern=FALSE)
        ,
        "svm oor + mask" = list(dataset=dataset, model='svm', strategy='oor', withpattern=TRUE)
        ,
        "svm gaussian" = list(dataset=dataset, model='svm', strategy='gaussian', withpattern=FALSE)
        ,
        "svm gaussian + mask" = list(dataset=dataset, model='svm', strategy='gaussian', withpattern=TRUE)
        ,
        "knn mean" = list(dataset=dataset, model='knn', strategy='mean', withpattern=FALSE)
        ,
        "knn mean + mask" = list(dataset=dataset, model='knn', strategy='mean', withpattern=TRUE)
        ,
        "knn oor" = list(dataset=dataset, model='knn', strategy='oor', withpattern=FALSE)
        ,
        "knn oor + mask" = list(dataset=dataset, model='knn', strategy='oor', withpattern=TRUE)
        ,
        "knn gaussian" = list(dataset=dataset, model='knn', strategy='gaussian', withpattern=FALSE)
        ,
        "knn gaussian + mask" = list(dataset=dataset, model='knn', strategy='gaussian', withpattern=TRUE)
    )) %dopar% {
        #iter.seed <- iter.seed + 1
        source('functions_boxplots.R')
        run_scores(model=param$model, strategy=param$strategy, withpattern=param$withpattern,
                   dataset=param$dataset,
                   sizes=sizes, n_rep=n_rep, prob=prob, n_features=n_features, noise=noise,
                   min_samples_leaf=min_samples_leaf, rho=rho, seed=iter.seed,
                   num.threads.ranger=num.threads.ranger)
    }
    
    return(results.list)
}

n_features_boxplot1 <- 9
if (boxplot_choice == 1 | boxplot_choice == -1) {
    # if (file.exists("results/boxplot_MCAR.RData")) {
    #     load("results/boxplot_MCAR.RData")
    #     res <- Parallel("make_data1", n_features=n_features_boxplot1)
    #     scores_mcar <- modifyList(scores_mcar, res)
    # } else 
    scores_mcar <- Parallel("make_data1", n_features=n_features_boxplot1)
    # separate the R2 scores and the training/testing times
    result <- list()
    for (i in seq_along(scores_mcar)) {
        result <- append(result, list(scores_mcar[[i]]$result))
    }
    train_times <- list()
    for (i in seq_along(scores_mcar)) {
        train_times <- append(train_times, list(scores_mcar[[i]]$train_times))
    }
    test_times <- list()
    for (i in seq_along(scores_mcar)) {
        test_times <- append(test_times, list(scores_mcar[[i]]$test_times))
    }

    scores_mcar <- result
    train_times_mcar <- train_times
    test_times_mcar <- test_times
    save(scores_mcar, file="results/boxplot_MCAR.RData")
    save(train_times_mcar, file="results/train_times_boxplot_MCAR.RData")
    save(test_times_mcar, file="results/test_times_boxplot_MCAR.RData")

    
    # if (file.exists("results/boxplot_MNAR.RData")) {
    #     load("results/boxplot_MNAR.RData")
    #     scores_mnar <- modifyList(scores_mnar, Parallel("make_data3", n_features=n_features_boxplot1))
    # } else 
    scores_mnar <- Parallel("make_data3", n_features=n_features_boxplot1)
    result <- list()
    for (i in seq_along(scores_mnar)) {
        result <- append(result, list(scores_mnar[[i]]$result))
    }
    train_times <- list()
    for (i in seq_along(scores_mnar)) {
        train_times <- append(train_times, list(scores_mnar[[i]]$train_times))
    }
    test_times <- list()
    for (i in seq_along(scores_mnar)) {
        test_times <- append(test_times, list(scores_mnar[[i]]$test_times))
    }

    scores_mnar <- result
    train_times_mnar <- train_times
    test_times_mnar <- test_times
    save(scores_mnar, file="results/boxplot_MNAR.RData")
    save(train_times_mnar, file="results/train_times_boxplot_MNAR.RData")
    save(test_times_mnar, file="results/test_times_boxplot_MNAR.RData")

    # if (file.exists("results/boxplot_PRED.RData")) {
    #     load("results/boxplot_PRED.RData")
    #     scores_pred <- modifyList(scores_pred, Parallel("make_data3bis", n_features=n_features_boxplot1))
    # } else 
    scores_pred <- Parallel("make_data3bis", n_features=n_features_boxplot1)
    result <- list()
    for (i in seq_along(scores_pred)) {
        result <- append(result, list(scores_pred[[i]]$result))
    }
    train_times <- list()
    for (i in seq_along(scores_pred)) {
        train_times <- append(train_times, list(scores_pred[[i]]$train_times))
    }
    test_times <- list()
    for (i in seq_along(scores_pred)) {
        test_times <- append(test_times, list(scores_pred[[i]]$test_times))
    }

    scores_pred <- result
    train_times_pred <- train_times
    test_times_pred <- test_times
    save(scores_pred, file="results/boxplot_PRED.RData")
    save(train_times_pred, file="results/train_times_boxplot_PRED.RData")
    save(test_times_pred, file="results/test_times_boxplot_PRED.RData")

} 

# n_features_boxplot2 <- 10
# if (boxplot_choice == 2 || boxplot_choice == -1) {
#     if (file.exists("results/boxplot2_1.RData")) {
#         load("results/boxplot2_1.RData")
#         scores_21 <- modifyList(scores_21, Parallel("make_data4", n_features=n_features_boxplot2))
#     } else scores_21 <- Parallel("make_data4", n_features=n_features_boxplot2)
#     save(scores_21, file="results/boxplot2_1.RData")

#     if (file.exists("results/boxplot2_2.RData")) {
#         load("results/boxplot2_2.RData")
#         scores_22 <- modifyList(scores_22, Parallel("make_data5", n_features=n_features_boxplot2))
#     } else scores_22 <- Parallel("make_data5", n_features=n_features_boxplot2)
#     save(scores_22, file="results/boxplot2_2.RData")

#     if (file.exists("results/boxplot2_3.RData")) {
#         load("results/boxplot2_3.RData")
#         scores_23 <- modifyList(scores_23, Parallel("make_data6", n_features=n_features_boxplot2))
#     } else scores_23 <- Parallel("make_data6", n_features=n_features_boxplot2)
#     save(scores_23, file="results/boxplot2_3.RData")

# } else stop("Invalid boxplot_choice")

stopCluster(cl)
