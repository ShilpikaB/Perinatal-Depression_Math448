# Libraries
library(ggplot2)
library(scorecard)
library(dplyr)
library(class)
library(leaps)
library(e1071)
library(caret)
library(MASS)
library(boot)
library(bestglm)
library(caret)
library(glmnet)
library(gbm)
library(plotROC)
library(ROCR)
library(pROC)
library(tidyverse)
library(tree)
library(maptree)
library(randomForest)
library(rpart)
library(rpart.plot)

mainFunction = function(){
  # Import data set
  ds = read.csv("C:/Users/sengu/Dropbox/PC/Desktop/SFSU_Sem3/Math448/Math448_TermProject/kaggle_data.csv", 
                header=TRUE, sep=";")
  dim(ds)  # initial dimension of the dataset 
  names(ds)
  
  #ds = find_flag.missingValues(ds)
  #ds = RemoveMissingData(ds)
  
  
  #####################################
  ### QQ plots ####
  # Normality Check for Predictors = "AGE_YEARS", "WEEKS_PREGNANCY"
  par(mfrow=c(1,1))
  nRows = dim(ds)[1]
  marginal_quantiles = vector(length=nRows)
  
  #names(ds)
  age_years.sort = sort(ds[,"AGE_YEARS"])
  weeks_preg.sort = sort(ds[,"WEEKS_PREGNANCY"])
  
  for(j in 1:nRows){
    prob_val = (j-0.5)/nRows
    marginal_quantiles[j] = qnorm(prob_val, mean=0, sd=1)
  }
  
  par(mfrow=c(1,2))
  plot(marginal_quantiles,age_years.sort, main="QQ plot for AGE_YEARS", 
       xlab ="Quantiles", ylab="Age Years", pch=21, bg = "green", col = "blue")
  
  plot(marginal_quantiles,weeks_preg.sort, main="QQ plot for WEEKS_PREGNANCY", 
       xlab ="Quantiles", ylab="Term of Pregnancy in weeks", pch=21, bg = "green", col = "red")
  
  
  ########   FACTORIZING THE RESPONSE VARIABLE    #########
  ds$DEPRESSION_CAT = 0  # Set this new column to 0: No depression
  ds[which(ds$DEPRESSION == 1),"DEPRESSION_CAT"] = 1 
  ds$DEPRESSION_CAT = as.factor(ds$DEPRESSION_CAT)
  contrasts(as.factor(ds$DEPRESSION_CAT))
  
  prop.table(table(ds$DEPRESSION_CAT)) # CHECK # OF SAMPLES IN EACH CLASS 
  
  ds_1 = full_data(ds) # consider all data points
  dim(ds_1)
  
  k = 0.7  # using k% of the total data for training the model
  set.seed(81)
  ds_2 = split_train.test(ds_1, k)
  
  train.ds = ds_2$train
  test.ds = ds_2$test
  
  dim(train.ds)
  dim(test.ds)
  
  dim(train.ds[train.ds$DEPRESSION_CAT == 0])
  dim(train.ds[train.ds$DEPRESSION_CAT == 1])
  
  trainUp = upSampling(train.ds) # up sample training data
  
  # Apply logistic regression to trainUp dataset and predict for test.ds dataset
  entire_log_reg = performLogisticReg(trainUp, aic_flg = 0, test.ds)
  
  log.fit_1 = entire_log_reg[[1]]
  summary(log.fit_1)
  table.logReg = entire_log_reg[2]
  mean_correct_rate.logReg = entire_log_reg[3]
  mean_error_rate.logReg = entire_log_reg[4]
  probs.logReg = entire_log_reg[5]
  
  par(mfrow=c(1,1))
  
  ROCRpred = prediction(probs.logReg, t(test.ds[,"DEPRESSION_CAT"]))
  ROCRperf  = performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, col="magenta")
  auc_log_reg = performance(ROCRpred, 'auc')@y.values
  auc_log_reg
  
  table.logReg
  mean_correct_rate.logReg
  mean_error_rate.logReg
  
  # Using stepwiseAIC subset selection on full model
  entire_log_reg_aic = performLogisticReg(trainUp, aic_flg = 1, test.ds)
  
  log.fit_aic = entire_log_reg_aic[[1]]
  step.aic.logReg = entire_log_reg_aic[6]
  step.aic.logReg
  
  # Using logistic regression on stepwiseAIC chosen model
  trainUp_mod = trainUp[c("HEALTH_PROB_CAT","HEALTH_PROBLEM_CAT","INSTRUCTION_CAT","DEPRESSION_CAT")]
  entire_log_reg_mod = performLogisticReg(trainUp_mod, aic_flg = 0, test.ds)
  
  log.fit_mod = entire_log_reg_mod[[1]]
  summary(log.fit_mod)
  table.logReg = entire_log_reg_mod[2]
  mean_correct_rate.logReg = entire_log_reg_mod[3]
  mean_error_rate.logReg = entire_log_reg_mod[4]
  probs.logReg = entire_log_reg_mod[5]
  
  par(mfrow=c(1,1))
  
  ROCRpred = prediction(probs.logReg, t(test.ds[,"DEPRESSION_CAT"]))
  ROCRperf  = performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, col="magenta")
  auc_log_reg = performance(ROCRpred, 'auc')@y.values
  auc_log_reg
  
  table.logReg
  mean_correct_rate.logReg
  mean_error_rate.logReg
  
  # Using shrinkage methods
  ridge.regression_model = ridge.regression(trainUp, test.ds)
  ridge.regression_model[2]
  ridge_err = ridge.regression_model[3]
  1-as.numeric(ridge_err)
  ridge_err
  
  lasso.regression_model = lasso.regression(trainUp, test.ds)
  lasso.regression_model[2]
  lasso_err = lasso.regression_model[3]
  1-as.numeric(lasso_err)
  lasso_err
  
  ###############################################
  
  lda_model = performLDA(trainUp, test.ds)  # Perform LDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  lda_model[2]
  lda_model[3]
  lda_model[4]
  
  qda_model = performQDA(trainUp, test.ds) # Perform QDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  qda_model[2]
  qda_model[3]
  qda_model[4]
  
  naive.bayes_model = naive_bayesian(trainUp, test.ds)
  naive.bayes_model[2]
  naive.bayes_model[3]
  naive.bayes_model[4]

  #############################################
  
  knn_model = performKNN(trainUp, test.ds) # Perform KNN Returns: K, mean_correct_rate
  knn_model
  
  decision.tree = decision_tree(trainUp, test.ds)
  summary(decision.tree[[1]])
  decision.tree[2]
  decision.tree[3]
  decision.tree[4]
  
  rpart.plot(decision.tree[[1]])
  # Misclassification Error rate for Training data = 14/90 = 15.5%
  
  
  # (FUN=prune.misclass) indicates that classification error rate
  # is used to guide the cross-validation and pruning process
  set.seed(81)
  cv_tree = cv.tree(decision.tree[[1]], FUN=prune.misclass) 
  names(cv_tree)
  cv_tree
  # The tree with 13 terminal nodes results in the lowest cross-validation error
  par (mfrow = c(1, 1))
  plot(cv_tree$size,cv_tree$dev,type="b")
  min_indx = which.min(cv_tree$dev)
  points(cv_tree$size[min_indx], cv_tree$dev[min_indx], col = 'red', cex = 2, pch = 19)
 
 
  pruned_tree_13 = pruned_tree(decision.tree[[1]], test.ds, best = 13)
  pruned_tree_13[2]
  pruned_tree_13[3]
  pruned_tree_13[4]
  draw_tree(pruned_tree_13[[1]])
  
  bag_fit = bagging(trainUp, test.ds)
  importance(bag_fit[[1]])
  varImpPlot(bag_fit[[1]])
  bag_fit[2]
  bag_fit[3]
  bag_fit[4]
  
  random_forest.fit = random_forest(trainUp, test.ds)
  importance(random_forest.fit[[1]]) # importance of each predictor
  varImpPlot(random_forest.fit[[1]])
  random_forest.fit[2]
  random_forest.fit[3]
  random_forest.fit[4]
  
  ####################################################################
  #################       USING ENVIRONMENTAL DATA ONLY      ###################
  ####################################################################
  
  ds_1 = env_data(ds) # consider only environmental predictors 
  dim(ds_1)
  
  k = 0.7  # using k% of the total data for training the model
  ds_2 = split_train.test(ds_1, k)
  
  train.ds = ds_2$train
  test.ds = ds_2$test
  
  dim(train.ds)
  dim(test.ds)
  
  trainUp = upSampling(train.ds) # up sample training data
  
  
  # Apply logistic regression to trainUp dataset and predict for test.ds dataset
  entire_log_reg = performLogisticReg(trainUp, aic_flg = 0, test.ds)
  
  log.fit_1 = entire_log_reg[[1]]
  summary(log.fit_1)
  table.logReg = entire_log_reg[2]
  mean_correct_rate.logReg = entire_log_reg[3]
  mean_error_rate.logReg = entire_log_reg[4]
  probs.logReg = entire_log_reg[5]
  
  par(mfrow=c(1,1))
  
  ROCRpred = prediction(probs.logReg, t(test.ds[,"DEPRESSION_CAT"]))
  ROCRperf  = performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, col="magenta")
  auc_log_reg = performance(ROCRpred, 'auc')@y.values
  auc_log_reg
  
  table.logReg
  mean_correct_rate.logReg
  mean_error_rate.logReg
  
  # Using stepwiseAIC subset selection on full model
  entire_log_reg_aic = performLogisticReg(trainUp, aic_flg = 1, test.ds)
  
  log.fit_aic = entire_log_reg_aic[[1]]
  step.aic.logReg = entire_log_reg_aic[6]
  step.aic.logReg
  
  # Using logistic regression on stepwiseAIC chosen model
  trainUp_mod = trainUp[c("HEALTH_PROB_CAT","HEALTH_PROBLEM_CAT","INSTRUCTION_CAT","DEPRESSION_CAT")]
  entire_log_reg_mod = performLogisticReg(trainUp_mod, aic_flg = 0, test.ds)
  
  log.fit_mod = entire_log_reg_mod[[1]]
  summary(log.fit_mod)
  table.logReg = entire_log_reg_mod[2]
  mean_correct_rate.logReg = entire_log_reg_mod[3]
  mean_error_rate.logReg = entire_log_reg_mod[4]
  probs.logReg = entire_log_reg_mod[5]
  
  par(mfrow=c(1,1))
  
  ROCRpred = prediction(probs.logReg, t(test.ds[,"DEPRESSION_CAT"]))
  ROCRperf  = performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, col="magenta")
  auc_log_reg = performance(ROCRpred, 'auc')@y.values
  auc_log_reg
  
  table.logReg
  mean_correct_rate.logReg
  mean_error_rate.logReg
  
  # Using shrinkage methods
  ridge.regression_model = ridge.regression(trainUp, test.ds)
  ridge.regression_model[2]
  ridge_err = ridge.regression_model[3]
  1-as.numeric(ridge_err)
  ridge_err
  
  lasso.regression_model = lasso.regression(trainUp, test.ds)
  lasso.regression_model[2]
  lasso_err = lasso.regression_model[3]
  1-as.numeric(lasso_err)
  lasso_err
  
  ###############################################
  
  lda_model = performLDA(trainUp, test.ds)  # Perform LDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  lda_model[2]
  lda_model[3]
  lda_model[4]
  
  qda_model = performQDA(trainUp, test.ds) # Perform QDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  qda_model[2]
  qda_model[3]
  qda_model[4]
  
  naive.bayes_model = naive_bayesian(trainUp, test.ds)
  naive.bayes_model[2]
  naive.bayes_model[3]
  naive.bayes_model[4]
  
  #############################################
  
  knn_model = performKNN(trainUp, test.ds) # Perform KNN Returns: K, mean_correct_rate
  knn_model
  
  decision.tree = decision_tree(trainUp, test.ds)
  summary(decision.tree[[1]])
  decision.tree[2]
  decision.tree[3]
  decision.tree[4]
  
  rpart.plot(decision.tree[[1]])
  # Misclassification Error rate for Training data = 14/90 = 15.5%
  
  
  # (FUN=prune.misclass) indicates that classification error rate
  # is used to guide the cross-validation and pruning process
  set.seed(81)
  cv_tree = cv.tree(decision.tree[[1]], FUN=prune.misclass) 
  names(cv_tree)
  cv_tree
  # The tree with 13 terminal nodes results in the lowest cross-validation error
  par (mfrow = c(1, 1))
  plot(cv_tree$size,cv_tree$dev,type="b")
  min_indx = which.min(cv_tree$dev)
  points(cv_tree$size[min_indx], cv_tree$dev[min_indx], col = 'red', cex = 2, pch = 19)
  
  
  pruned_tree_13 = pruned_tree(decision.tree[[1]], test.ds, best = 13)
  pruned_tree_13[2]
  pruned_tree_13[3]
  pruned_tree_13[4]
  draw_tree(pruned_tree_13[[1]])
  
  bag_fit = bagging(trainUp, test.ds)
  importance(bag_fit[[1]])
  varImpPlot(bag_fit[[1]])
  bag_fit[2]
  bag_fit[3]
  bag_fit[4]
  
  random_forest.fit = random_forest(trainUp, test.ds)
  importance(random_forest.fit[[1]]) # importance of each predictor
  varImpPlot(random_forest.fit[[1]])
  random_forest.fit[2]
  random_forest.fit[3]
  random_forest.fit[4]
  
  ####################################################################
  # Apply logistic regression to trainUp dataset and predict for test.ds dataset
  entire_log_reg = performLogisticReg(trainUp, aic_flg = 1, test.ds)
  
  log.fit_1 = entire_log_reg[[1]]
  
  table.logReg = entire_log_reg[2]
  mean_correct_rate.logReg = entire_log_reg[3]
  mean_error_rate.logReg = entire_log_reg[4]
  probs.logReg = entire_log_reg[5]
  step.aic.logReg = entire_log_reg[6]
  
  
  log.fit_1
  step.aic.logReg
  table.logReg
  mean_correct_rate.logReg
  mean_error_rate.logReg
  
  
  ROCRpred = prediction(probs.logReg, t(test.ds[,"DEPRESSION_CAT"]))
  ROCRperf  = performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, col="magenta")
  auc_log_reg = performance(ROCRpred, 'auc')@y.values
  auc_log_reg
  
  
  performLDA(trainUp, test.ds)  # Perform LDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  
  performQDA(trainUp, test.ds) # Perform QDA. Returns: lda.fit, conf_tbl, mean_correct_rate, mean_error_rate
  
  performKNN(trainUp, test.ds) # Perform KNN Returns: K, mean_correct_rate
  
}


################   ALL FUNCTIONS   ########################################

############### CALCULATE MODE  #########################
calc_mode = function(x){
  
  # List the distinct / unique values
  distinct_values = unique()
  
  # Count the occurrence of each distinct value
  distinct_tabulate = tabulate(match(x, distinct_values))
  
  # Return the value with the highest occurrence
  return(distinct_values[which.max(distinct_tabulate)])
}

#####################  FIND AND FLAG MISSING VALUES  ###################
# Some features have values = "", "-".
# Set them to NA
find_flag.missingValues = function(ds){
  ds[ds == ""] = NA
  ds[ds == "-"] = NA
  
  # Add a new column to flag all those rows which have one or more missing data
  ds$FlagRows_MissingValue = FALSE
  dim(ds)[1] # number of columns increases by 1
  
  # capture the row and column indexes where data is NA
  ind_missing_val = which(is.na(ds), arr.ind = TRUE)
  row_ind_missing = unique(sort(ind_missing_val[,1]))
  print(paste("Number of rows missing data: ", length(row_ind_missing)))
  print(paste0(round(((length(row_ind_missing)*100)/dim(ds)[1]), 2), "% of total data have missing values."))
  
  for(i in row_ind_missing){
    ds[i, "FlagRows_MissingValue"] = TRUE
  }
  
  x = row_ind_missing
  y = which(ds$FlagRows_MissingValue == TRUE)
  identical(x, y)
  # Check
  if(identical(x, y) == TRUE){
    print("All rows with missing values successfully flagged.")
  }else{
    print("Error. Something went wrong when flagging rows with missing values.")
  }
  
  return(ds)
}


######### HANDLING MISSING DATA - Remove rows with missing data ############################
RemoveMissingData = function(ds){
  ds_nu = subset(ds, ds$FlagRows_MissingValue != TRUE)
  return(ds_nu)
}


######### USING FULL RANGE OF DATA  ##########

full_data = function(ds){
  keeps <- c("AGE_YEARS","WEEKS_PREG_CAT","HEALTH_PROB_CAT"             
             ,"HEALTH_PROBLEM_CAT","DESIRED_PREG_CAT","EMPLOYED_CAT"         
             ,"INSTRUCTION_CAT","INCOME_CAT","MARITAL_STATUS_CAT"
             ,"MENTAL_CAT","DEPRESSION_CAT")
  return(ds[keeps])
}

env_data = function(ds){
  keeps <- c("DESIRED_PREG_CAT","EMPLOYED_CAT"         
             ,"INSTRUCTION_CAT","INCOME_CAT","MARITAL_STATUS_CAT"
             ,"MENTAL_CAT","DEPRESSION_CAT")
  return(ds[keeps])
}



################    SPLIT INTO TRAINING AND TEST DATA    #####################
split_train.test = function(ds, k){
  set.seed = 81
  dt_list = split_df(ds, ratio = k) # using k% of the total data for training the model
  
  return(dt_list)
}

##############  UPSAMPLE ONLY TRAINING DATA   #########################

upSampling = function(ds){
  trainUp = upSample(ds, y = train.ds$DEPRESSION_CAT)
  trainUp = subset(trainUp, select = -Class)
  print(table(trainup$DEPRESSION_CAT))
  
  return(trainUp)
}

####################     MODEL SELECTION      #########################

# BEST SUBSET SELECTION
regfit.full=regsubsets(DEPRESSION_CAT~., data=trainUp, nvmax=10)
reg.summary=summary(regfit.full)
names(reg.summary)

par(mfrow = c(1, 2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$rsq, xlab = "Number of Variables", ylab = "R2", type = "l")

################################
par(mfrow = c(2, 2))
# Max Adjusted R-squared
max_indx = which.max(reg.summary$adjr2)
max_indx
plot(reg.summary$adjr2, xlab="Number of Variables", ylab="Adjusted R2", type="l")
points(max_indx, reg.summary$adjr2[max_indx], col="red", cex=2, pch=20)

# Min Cp
min_indx = which.min(reg.summary$cp)
min_indx
plot(reg.summary$cp, xlab="Number of Variables", ylab="Cp", type="l")
points(min_indx, reg.summary$cp[min_indx], col="red", cex=2, pch=20)

# Min BIC
min_indx = which.min(reg.summary$bic)
min_indx
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC", type="l")
points(min_indx, reg.summary$bic[min_indx], col="red", cex=2, pch=20)

################################
par(mfrow = c(2, 2))
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")


####################     LOGISTIC REGRESSION      #######################

# LOGISTIC REGRESSION function with NO INTERACTION TERMS

performLogisticReg = function(ds, aic_flg, test.ds){
  set.seed(81)
  log.fit = glm(DEPRESSION_CAT~., data = ds, family = binomial)
  
  # predict the response variable 'DEPRESSION_CAT' for the test data
  log.probs = predict(log.fit, newdata = test.ds, type="response")
  log.pred = vector(length = length(log.probs))
  
  log.pred[log.probs >= 0.5] = 1  # Predicted with Having Depression
  log.pred[log.probs < 0.5] = 0   # Predicted with Having No Depression
  
    # Confusion matrix
  conf_tbl = table(log.pred, t(test.ds[,"DEPRESSION_CAT"]))
  
  mean_correct_rate = mean(log.pred == t(test.ds[,"DEPRESSION_CAT"])) # correct rate in test data
  mean_error_rate = mean(log.pred != t(test.ds[,"DEPRESSION_CAT"])) # error rate in test data
  
  if(aic_flg == 1){ # Run Step-wise AIC
    
    aic = stepAIC(log.fit, direction="both")
    output = list(log.fit, conf_tbl, mean_correct_rate, mean_error_rate, log.probs, aic)
  }
  else{ # SKIP Step-wise AIC
    output = list(log.fit, conf_tbl, mean_correct_rate, mean_error_rate, log.probs)
  }
  return(output)
}

#################   Ridge Regression    ##########################

ridge.regression = function(ds, test.ds){
  set.seed(81)
  xtrain = model.matrix(DEPRESSION_CAT~., ds)[,-1]
  ytrain = ds$DEPRESSION_CAT
  
  xtest = model.matrix(DEPRESSION_CAT~., test.ds)[,-1]
  ytest = test.ds$DEPRESSION_CAT
  
  grid = 10^seq(10, -2, length=100)
  ridge.mod = glmnet(xtrain, ytrain, alpha = 0, lambda = grid, family="binomial")  # alpha =0 for Ridge Regression
  
  cv.glmmod = cv.glmnet(xtrain, ytrain, alpha = 0, family="binomial", type.measure="class")
  plot(cv.glmmod, main="Ridge Regression Lambda")
  best.lambda<- cv.glmmod$lambda.min
  
  ridge_pred = predict(ridge.mod, newx = xtest, s = best.lambda, type="class") 
  ridge.err = mean(ridge_pred != ytest)
  
  output = list(ridge.mod, best.lambda, ridge.err)
  return(output)
}

#################   Lasso Regression    ##########################

lasso.regression = function(ds, test.ds){
  set.seed(81)
  xtrain = model.matrix(DEPRESSION_CAT~., ds)[,-1]
  ytrain = ds$DEPRESSION_CAT
  
  xtest = model.matrix(DEPRESSION_CAT~., test.ds)[,-1]
  ytest = test.ds$DEPRESSION_CAT
  
  grid = 10^seq(10, -2, length=100)
  lasso.mod = glmnet(xtrain, ytrain, alpha = 1, lambda = grid, family="binomial")  # alpha = 1 for Lasso Regression
  
  cv.glmmod = cv.glmnet(xtrain, ytrain, alpha = 1, family="binomial", type.measure="class")
  plot(cv.glmmod, main="Lasso Lambda")
  best.lambda<- cv.glmmod$lambda.min
  
  lasso_pred = predict(lasso.mod, newx = xtest, s = best.lambda, type="class") 
  lasso.err = mean(lasso_pred != ytest)
  
  output = list(lasso.mod, best.lambda, lasso.err)
  return(output)
}

####################   LINEAR DISCRIMINANT ANALYSIS (LDA)   #####################

performLDA = function(ds, test.ds){
  set.seed(81)
  lda.fit = lda(DEPRESSION_CAT~., data = ds) #lda fit 
  #plot(lda.fit)
  
  lda.pred = predict(lda.fit, test.ds)
  lda.pred_1 = lda.pred$class
  
  conf_tbl = table(lda.pred_1, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(lda.pred_1 == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(lda.pred_1 != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(lda.fit, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

#################   QUADRATIC DISCRIMINAT ANALYSIS  (QDA) ################

performQDA = function(ds, test.ds){
  set.seed(81)
  qda.fit = qda(DEPRESSION_CAT~., data = ds)
  
  qda.pred = predict(qda.fit, test.ds)
  qda.pred_1 = qda.pred$class
  
  conf_tbl = table(qda.pred_1, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(qda.pred_1 == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(qda.pred_1 != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(qda.fit, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

#################  Naive Bayesian  ############################

naive_bayesian = function(ds, test.ds){
  set.seed(81)
  nb.fit = naiveBayes(DEPRESSION_CAT~., data =ds)
  
  nb.class = predict(nb.fit, test.ds)
  conf_tbl = table(nb.class, t(test.ds$DEPRESSION_CAT))
  mean_correct_rate = mean(nb.class == t(test.ds$DEPRESSION_CAT))
  mean_error_rate = mean(nb.class != t(test.ds$DEPRESSION_CAT))
  output = list(nb.fit, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

###################    K-Nearest Neighbors    ############################

performKNN = function(ds, test.ds){
  set.seed(81)
  K = seq(1,10,1)
  mean_knn = vector(length = length(K))
  
  for(i in 1:length(K)){
    
    knn.pred1 = knn(ds, test.ds, ds[,"DEPRESSION_CAT"], k = K[i]) 
    
    table(knn.pred1, t(test.ds[, "DEPRESSION_CAT"]))
    mean_knn[i] = mean(knn.pred1 == t(test.ds[, "DEPRESSION_CAT"]))
  }
  
  result_knn = data.frame(K, mean_knn)
  
  return(result_knn)
}

###########################   Decision Tree   ################################### 

decision_tree = function(ds, test.ds){
  set.seed(81)
  decision.tree = rpart(DEPRESSION_CAT~., data=ds, method="class", maxdepth=6)
  
  tree_pred = predict(decision.tree, test.ds, type="class")
  conf_tbl = table(tree_pred, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(tree_pred == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(tree_pred != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(decision.tree, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

###########################   PRUNED Decision Tree   ###################################

pruned_tree = function(ds_tree, test.ds, best_val){
  set.seed(81)
  pruned.tree = prune.misclass(ds_tree, best = best_val)
  
  pruned_tree_pred = predict(pruned.tree, test.ds, type="class")
  conf_tbl = table(pruned_tree_pred, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(pruned_tree_pred == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(pruned_tree_pred != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(pruned.tree, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

#################  Bagging  ######################

bagging = function(ds, test.ds){
  set.seed(81)
  bag_fit = randomForest(DEPRESSION_CAT~., data = ds, mtry = 10, importance = TRUE)
  bag_fit
  
  yhat.bag = predict(bag_fit, newdata=test.ds, type="class")
  conf_tbl = table(yhat.bag, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(yhat.bag == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(yhat.bag != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(bag_fit, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

#################  Random Forest  ######################

random_forest = function(ds, test.ds){
  set.seed(81)
  rnd_forest_fit = randomForest(DEPRESSION_CAT~., data = ds, importance = TRUE)
  
  yhat.rf = predict(rnd_forest_fit, newdata=test.ds, type="class")
  conf_tbl = table(yhat.rf, t(test.ds[, "DEPRESSION_CAT"]))
  mean_correct_rate = mean(yhat.rf == t(test.ds[, "DEPRESSION_CAT"]))
  mean_error_rate = mean(yhat.rf != t(test.ds[, "DEPRESSION_CAT"]))
  output = list(rnd_forest_fit, conf_tbl, mean_correct_rate, mean_error_rate)
  
  return(output)
}

###################    Draw Tree   ############################
draw_tree = function(tree_ds)
{
  par(mfrow=c(1,1))
  draw.tree(tree_ds, cex=0.9)
}

#########################  DATA DISCRETIZATION: BINNING #################################################
# 
# Binning for age
ageBinning = function(ds){
  ds_nu = ds %>% mutate(AGE_YEARS_Bin = ntile(AGE_YEARS, n=10)) #Splitting women into 5 age groups
  return(ds_nu)
}



ds = ageBinning(ds)
#ds
hist(ds$AGE_YEARS_Bin, freq = FALSE)
barplot(ds$AGE_YEARS_Bin)
histPlot = hist(ds$AGE_YEARS_Bin)
#, ylab = "Frequency", xlab = "AgeYears", cex.lab=1.5, cex.axis=1.5, 
#cex.main=1.5, cex.sub=1.5,
#col = 12, main = "Histogram of the AgeYears", ylim = c(0,18), xlim=c(10,46))
x =  seq(min(ds$AGE_YEARS),max(ds$AGE_YEARS))
y =  dnorm(x, mean = mean(ds$AGE_YEARS), sd=sd(ds$AGE_YEARS))
y<- y*diff(histPlot$mids[13:15])*length(ds$AGE_YEARS)
lines(x, y, col=2, lwd=4)


summary(ds_nu$AGE_YEARS)
hist(ds_nu$AGE_YEARS, freq=TRUE)
pairs(ds_nu$DEPRESSION~ds_nu$AGE_YEARS) 
# No direct correlation between age and depression

ds_nu = ds_nu %>% mutate(AGE_YEARS_Bin = ntile(AGE_YEARS, n=5)) #Splitting women into 5 age groups
ds_nu$AGE_YEARS_Bin
hist(ds_nu$AGE_YEARS_Bin, freq=TRUE)

##########################################
#Binning for 'Weeks of Pregnancy'
weeksPregBinning = function(){
  
# First Trimester Week 0-14
# Second Trimester Week 14-27
# Third Trimester Week 27- end
  ds_nu = ds_nu %>% mutate(WEEKS_PREGNANCY_Bin = cut(WEEKS_PREGNANCY, breaks = c(0, 14, 27, 44), labels = c(1,2,3)))
  return(ds_nu)
}

summary(ds_nu$WEEKS_PREGNANCY)
hist(ds_nu$WEEKS_PREGNANCY, freq=TRUE)
pairs(ds_nu$DEPRESSION~ds_nu$WEEKS_PREGNANCY) 
# Shows at all stages of pregnancy are equally vulnerable to depression

ds_nu = ds_nu %>% mutate(WEEKS_PREGNANCY_Bin = cut(WEEKS_PREGNANCY, breaks = c(0, 14, 27, 44), labels = c(1,2,3)))
ds_nu$WEEKS_PREGNANCY_Bin
##########################################

########################## FACTORIZE STRING DATA INTO NUMERIC #####################################
# Convert string data from column "Health_Problem?" to numeric values
ds_nu$HEALTH_PROBLEM_Bin = as.numeric(as.factor(ds_nu$HEALTH_PROBLEM.))
summary(ds_nu$HEALTH_PROBLEM_Bin)
hist(ds_nu$HEALTH_PROBLEM_Bin, freq=TRUE) # Health conditions factorized as 15-20 is more prevalent in the subjects.
pairs(ds_nu$DEPRESSION~ds_nu$HEALTH_PROBLEM_Bin)

# Convert column "DESIRED_PREGNANCY." to numeric values
ds_nu$DESIRED_PREGNANCY_Bin = as.numeric(as.factor(ds_nu$DESIRED_PREGNANCY.))
summary(ds_nu$DESIRED_PREGNANCY_Bin)

# Convert column "DESIRED_PREGNANCY." to numeric values
ds_nu$DESIRED_PREGNANCY_Bin = as.numeric(as.factor(ds_nu$DESIRED_PREGNANCY.))
summary(ds_nu$DESIRED_PREGNANCY_Bin)

# Convert column "CURRENTLY_EMPLOYED" to numeric values
ds_nu$CURRENTLY_EMPLOYED_Bin = as.numeric(as.factor(ds_nu$CURRENTLY_EMPLOYED))
summary(ds_nu$CURRENTLY_EMPLOYED_Bin)

# Convert column "DEGREE_OF_INSTRUCTION" to numeric values
ds_nu$DEGREE_OF_INSTRUCTION_Bin = as.numeric(as.factor(ds_nu$DEGREE_OF_INSTRUCTION))
summary(ds_nu$DEGREE_OF_INSTRUCTION_Bin)

# Convert column "FAMILY_INCOME" to numeric values
ds_nu$FAMILY_INCOME_Bin = as.numeric(as.factor(ds_nu$FAMILY_INCOME))
summary(ds_nu$FAMILY_INCOME_Bin)

# Convert column "MARITAL_STATUS" to numeric values
ds_nu$MARITAL_STATUS_Bin = as.numeric(as.factor(ds_nu$MARITAL_STATUS))
summary(ds_nu$MARITAL_STATUS_Bin)


# Convert column "MENTAL_HEALTH_HISTORY_PROBLEM" to numeric values
ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM_Bin = as.numeric(as.factor(ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM))
summary(ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM_Bin)


############################################

# Using Pairs
plotPairs = function(df, lbl, title_nm){
  pairs(df, labels = lbl, main = title_nm, upper.panel = NULL, cex.labels=2)
}

testpairs_ext_cat = ds[, c("AGE_YEARS", 
                      "WEEKS_PREGNANCY", "WEEKS_PREG_CAT",
                      "HEALTH_PROBLEM_CAT",
                      "DESIRED_PREG_CAT",
                      "EMPLOYED_CAT",
                      "INSTRUCTION_CAT",
                      "INCOME_CAT",
                      "MARITAL_STATUS_CAT",
                      "MENTAL_CAT",
                      "DEPRESSION")]
testpairs_ext_cat

lbl = c("age","weeks_preg", "health_prob","desired_preg", "Employment","Education","Income","Marital_stat","History","Depress")
title_nm = "Pairs matrix (Exisiting categories)"

plotPairs(testpairs_ext_cat, lbl, title_nm)

testpairs_new = ds_nu[, c("AGE_YEARS_Bin",
                          "WEEKS_PREGNANCY_Bin",
                          "HEALTH_PROBLEM_Bin", 
                          "DESIRED_PREGNANCY_Bin", 
                          "CURRENTLY_EMPLOYED_Bin", 
                          "DEGREE_OF_INSTRUCTION_Bin", 
                          "FAMILY_INCOME_Bin", 
                          "MARITAL_STATUS_Bin",
                          "MENTAL_HEALTH_HISTORY_PROBLEM_Bin", 
                          "DEPRESSION")]
lbl = c("age","weeks_preg", "health_prob","desired_preg", "Employment","Education","Income","Marital_stat","History","Depress")
title_nm = "Pairs matrix (New categories)"

plotPairs(testpairs_new, lbl, title_nm)

histPlot = hist(ds$AGE_YEARS, ylab = "Frequency", breaks = 10, xlab = "AgeYears", 
                cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5,
                col = 12, main = "Histogram of the AgeYears", 
                ylim = c(0,18), xlim=c(10,46))
x =  seq(min(ds$AGE_YEARS),max(ds$AGE_YEARS))
y =  dnorm(x, mean = mean(ds$AGE_YEARS), sd=sd(ds$AGE_YEARS))
y<- y*diff(histPlot$mids[13:15])*length(ds$AGE_YEARS)
lines(x, y, col=2, lwd=4)




cor(ds_nu$HEALTH_PROBLEM_Bin, ds_nu$AGE_YEARS_Bin)
cor(ds_nu$HEALTH_PROBLEM_CAT, ds_nu$AGE_YEARS)

cor(ds_nu$HEALTH_PROBLEM_Bin, ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM_Bin)
cor(ds_nu$HEALTH_PROBLEM_CAT, ds_nu$HEALTH_PROB_CAT)

cor(ds_nu$HEALTH_PROBLEM_Bin, ds_nu$MARITAL_STATUS_Bin)
cor(ds_nu$HEALTH_PROBLEM_CAT, ds_nu$MARITAL_STATUS_CAT)

ds_nu$MARITAL_STATUS_Bin
ds_nu$MARITAL_STATUS

ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM.
ds_nu$MENTAL_HEALTH_HISTORY_PROBLEM_Bin

plot((ds$EMPLOYED_CAT)