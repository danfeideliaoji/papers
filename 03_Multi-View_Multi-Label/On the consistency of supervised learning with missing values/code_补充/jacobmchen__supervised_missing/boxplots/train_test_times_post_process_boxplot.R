load("results/train_times_boxplot_MCAR.RData")
load("results/test_times_boxplot_MCAR.RData")
load("results/train_times_boxplot_MNAR.RData")
load("results/test_times_boxplot_MNAR.RData")
load("results/train_times_boxplot_PRED.RData")
load("results/test_times_boxplot_PRED.RData")
# these are of lists of matrices of shape n_rep*length(sizes)

names(train_times_mcar) <- c(
  "rpart", "rpart + mask", 
  "mean", "mean + mask", "oor", "oor + mask",
  "Gaussian", "Gaussian + mask", 
  "MIA",
  "ctree", "ctree + mask",
  "mean (forest)", "mean + mask (forest)", "oor (forest)", "oor + mask (forest)",
  "Gaussian (forest)", "Gaussian + mask (forest)", 
  "MIA (forest)",
  "mean (xgboost)", "mean + mask (xgboost)", "oor (xgboost)", "oor + mask (xgboost)",
  "Gaussian (xgboost)", "Gaussian + mask (xgboost)", 
  "MIA (xgboost)",
  "mean (svm)", "mean + mask (svm)", "oor (svm)", "oor + mask (svm)",
  "Gaussian (svm)", "Gaussian + mask (svm)",
  "mean (knn)", "mean + mask (knn)", "oor (knn)", "oor + mask (knn)",
  "Gaussian (knn)", "Gaussian + mask (knn)"
)

names(test_times_mcar) <- names(train_times_mcar)
names(train_times_mnar) <- names(train_times_mcar)
names(test_times_mnar) <- names(train_times_mcar)
names(train_times_pred) <- names(train_times_mcar)
names(test_times_pred) <- names(train_times_mcar)
     
df_for_ggplot <- function(scores, col_name) {
    aa <- cbind.data.frame(unlist(scores), rep(names(scores), each=length(scores[[1]])))
    colnames(aa) <- c(col_name, "method")
    aa$forest <- as.factor(
        ifelse(grepl("xgboost", aa$method), "XGBOOST", 
               ifelse(grepl("forest", aa$method), "RANDOM FOREST", 
                  ifelse(grepl("svm", aa$method), "SVM", 
                    ifelse(grepl("knn", aa$method), "KNN", "DECISION TREE")))))
    aa$method <- as.factor(sub("\\ \\(forest\\)", "", aa$method))
    aa$method <- as.factor(sub("\\ \\(xgboost\\)", "", aa$method))
    aa$method <- as.factor(sub("\\ \\(svm\\)", "", aa$method))
    aa$method <- as.factor(sub("\\ \\(knn\\)", "", aa$method))
    aa$method <- as.factor(sub("\\ \\+\\ mask", "\\ +\\ mask", aa$method))
    return(aa)
}


aa <- df_for_ggplot(train_times_mcar, "train_time")
write.csv(aa,'results/train_times_mcar.csv')
aa <- df_for_ggplot(test_times_mcar, "test_time")
write.csv(aa,'results/test_times_mcar.csv')

aa <- df_for_ggplot(train_times_mnar, "train_time")
write.csv(aa,'results/train_times_mnar.csv')
aa <- df_for_ggplot(test_times_mnar, "test_time")
write.csv(aa,'results/test_times_mnar.csv')

aa <- df_for_ggplot(train_times_pred, "train_time")
write.csv(aa,'results/train_times_pred.csv')
aa <- df_for_ggplot(test_times_pred, "test_time")
write.csv(aa,'results/test_times_pred.csv')
# aa <- df_for_ggplot(scores_21)
# write.csv(aa,'boxplots/results/scores_linearlinear.csv')
# aa <- df_for_ggplot(scores_22)
# write.csv(aa,'boxplots/results/scores_linearnonlinear.csv')
# aa <- df_for_ggplot(scores_23)
# write.csv(aa,'boxplots/results/scores_nonlinearnonlinear.csv')
