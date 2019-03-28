if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
library(tidyverse)
library(dslabs)
library(class)
library(knitr)
library(ggplot2)
library(dplyr)
library(randomForest)
library(missForest)
library(doParallel)
library(e1071)
train_data <- read.table("..\train.csv",
                         header = TRUE,
                         sep=",",
                         row.names="Id")

##data organization and manipulation
########################################################################################
#
#
Sale_Price <- train_data$SalePrice #storing sale price
train_data_temp <- train_data
train_data_temp$SalePrice <- NULL
cat_train <- select_if(train_data_temp, is.factor) #saving only categorical varaibles
num_train <- select_if(train_data_temp, is.numeric) #saving only numerical variables


## set seed and cores

registerDoParallel(cores = 3)
set.seed(120)


##use missForest package to impute missing values in data

all_data_mis <-  missForest(xmis = train_data_temp, maxiter = 10, ntree = 100, variablewise = FALSE,
                            decreasing = FALSE, verbose = TRUE, mtry = floor(sqrt(ncol(train_data_temp))), replace = TRUE,
                            classwt = NULL, cutoff = NULL, strata = NULL, sampsize = NULL, nodesize = NULL, 
                            maxnodes = NULL, xtrue = NA, parallelize = "variables")

all_dataM <- all_data_mis$ximp


## one hot encoding the categorical variables

dum <- dummyVars(~., data = all_dataM)
alldata_ohe <- data.frame(predict(dum, newdata = all_dataM))
alldata_ohe <- cbind(alldata_ohe, Sale_Price)


##Splitting dataset into training and validation sets

rt <- sample(1:nrow(alldata_ohe), size = 0.2*nrow(alldata_ohe))
val_set <- alldata_ohe[rt,]
train_set <- alldata_ohe[-rt,]
#
#
############################################################################################


#creating each training each model method
#
#
############################################################################################

## glm method########################################################
glm_method <- train(Sale_Price ~ ., method = "glm", data = train_set)
#storing R^2 and RMSE
results <- data_frame(method = "glm", RMSE = glm_method$results$RMSE , Rsq = glm_method$results$Rsquared)
#saving top 30 variables and removing text after "."
cols_n <- varImp(glm_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
top_list <- cols_n$names[1:50]
top_list <- gsub("\\..*", "", top_list)

## lm method ###################################################################################
lm_method <- train(Sale_Price ~ ., method = "lm", data = train_set)
#storing results
results <- bind_rows(results, 
                     data_frame(method = "lm",
                                RMSE = lm_method$results$RMSE,
                                Rsq = lm_method$results$Rsquared))
#saving top 50 variables and removing text after "."
cols_n <- varImp(lm_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
#only keeping similar variables between list
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))


## knn method with control #######################################################################
control <- trainControl(method = "cv", number = 10, p = 0.9)
knn_method <- train(Sale_Price ~ .,
                       method = "knn",
                       data = train_set,
                       tuneGrid = data.frame(k= seq(1, 49, 2)))
#storing results
results <- bind_rows(results, 
                     data_frame(method = "knn",
                                RMSE = knn_method$results$RMSE,
                                Rsq = knn_method$results$Rsquared))
#saving top 30 important variables
cols_n <- varImp(knn_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
#only keeping the similar variables between lists
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))

######kknn method ################################################################
kknn_method <- train(Sale_Price ~ .,
                    method = "kknn",
                    data = train_set)
results <- bind_rows(results, 
                     data_frame(method = "kknn",
                                RMSE = kknn_method$results$RMSE,
                                Rsq = kknn_method$results$Rsquared))
cols_n <- varImp(kknn_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))


####svmRadial method ################################################################
svmR_method <- train(Sale_Price ~ ., method = "svmRadial", data = train_set)
results <- bind_rows(results, 
                     data_frame(method = "svmRadial",
                                RMSE = svmR_method$results$RMSE,
                                Rsq = svmR_method$results$Rsquared))
cols_n <- varImp(svmR_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
cols_n$names[1:30]
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))
 
#svmRadial squared method ################################################################
svmRS_method <- train(Sale_Price ~ ., method = "svmRadialSigma", data = train_set)
results <- bind_rows(results, 
                     data_frame(method = "svmRadialSigma",
                                RMSE = svmRS_method$results$RMSE,
                                Rsq = svmRS_method$results$Rsquared))
cols_n <- varImp(svmRS_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))

#svmRadial Cost method ################################################################
svmRC_method <- train(Sale_Price ~ ., method = "svmRadialCost" , data = train_set)
results <- bind_rows(results, 
                     data_frame(method = "svmRadialCost",
                                RMSE = svmRC_method$results$RMSE,
                                Rsq = svmRC_method$results$Rsquared))
cols_n <- varImp(svmRC_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))

#svmLinear  method ################################################################
svmL_method <- train(Sale_Price ~ ., method = "svmLinear", data = train_set)
results <- bind_rows(results, 
                     data_frame(method = "svmLinear",
                                RMSE = svmL_method$results$RMSE,
                                Rsq = svmL_method$results$Rsquared))
cols_n <- varImp(svmL_method)$importance %>% mutate(names=row.names(.)) %>% arrange(-Overall)
top_list <- intersect(unique(top_list), unique(gsub("\\..*", "", cols_n$names[1:50])))
#
#
###########################################################################################


##creating the data frame of optimal variables from original data set
optimal_ts <- train_data[ ,which(colnames(train_data) %in% top_list)]

#perform one hot encoding
registerDoParallel(cores = 3)
set.seed(120)
#use missForest package to impute missing values in data
opt_mis <-  missForest(xmis = optimal_ts, maxiter = 10, ntree = 100, variablewise = FALSE,
                            decreasing = FALSE, verbose = TRUE, mtry = floor(sqrt(ncol(optimal_ts))), replace = TRUE,
                            classwt = NULL, cutoff = NULL, strata = NULL, sampsize = NULL, nodesize = NULL, 
                            maxnodes = NULL, xtrue = NA, parallelize = "variables")

optM <- opt_mis$ximp
#one hot encoding the categorical variables
dum <- dummyVars(~., data = optM)
optdata_ohe <- data.frame(predict(dum, newdata = all_dataM))
optdata_ohe <- cbind(optdata_ohe, Sale_Price)





## recreate best methods with optimal data
opt_glm <- train(Sale_Price ~ ., method = "glm", data = optdata_ohe)

## create prediction for validation set
glm_pred <- predict(opt_glm, val_set, type = "raw")

#store desired metrics
opt_res <- data_frame(method = "glm", 
                      RMSE = opt_glm$results$RMSE, 
                      Rsq = opt_glm$results$Rsquared,
                      val_er = mean(abs(glm_pred - val_set$Sale_Price)),
                      Acc = (1 - (val_er/mean(val_set$Sale_Price))))

opt_lm <- train(Sale_Price ~ ., method = "lm", data = optdata_ohe)
lm_pred <- predict(opt_lm, val_set, type = "raw")
opt_res <- bind_rows(opt_res, 
                     data_frame(method = "lm",
                                RMSE = opt_lm$results$RMSE, 
                                Rsq = opt_lm$results$Rsquared,
                                val_er = mean(abs(lm_pred - val_set$Sale_Price)),
                                Acc = (1 - (val_er/mean(val_set$Sale_Price)))))

grid <- expand.grid(C = seq(0, 2, 0.1))
ctrl <- trainControl(method = "cv", number = 10, p = 0.9)
set.seed(789)
opt_svmL <- train(Sale_Price ~ ., method = "svmLinear", data = optdata_ohe,
                  trControl = ctrl,
                  tuneGrid = grid)
svmL_pred <- predict(opt_svmL, val_set, type = "raw")
opt_res <- bind_rows(opt_res, 
                     data_frame(method = "svmLinear",
                                RMSE = opt_svmL$results$RMSE, 
                                Rsq = opt_svmL$results$Rsquared,
                                val_er = mean(abs(svmL_pred - val_set$Sale_Price)),
                                Acc = (1 - (val_er/mean(val_set$Sale_Price)))))
opt_res
opt_svmL$finalModel











