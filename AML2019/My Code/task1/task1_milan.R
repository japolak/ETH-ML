
######################### AML project 1 ##############################
library(randomForest)
library(e1071)
library(gbm)
library(xgboost)

set.seed(122)

rm(list = ls())

x.train <- read.csv("X_train.csv")[,-1]
y.train <- read.csv("y_train.csv")[,-1]
x.test <- read.csv("X_test.csv")[,-1]



# 1.1) Impute the NAs with median value (outlier robust)


n <- dim(x.train)[1]
d <- dim(x.train)[2]

for (i in 1:d) {
  na.ind <- which(is.na(x.train[,i]))
  x.train[na.ind, i] <- rep(median(na.omit(x.train[, i])), length(na.ind))
}


# # 1.2) Outlier detection and imputing with median
# 
# 
# for (i in 1:d) {
#   out.ind <- which(x.train[,i] - quantile(x.train[,i],0.75) > 1.5 * IQR(x.train[,i]) |
#                      quantile(x.train[,i],0.25) - x.train[,i] > 1.5 * IQR(x.train[,i]))
#   x.train[out.ind, i] <- rep(median(x.train[, i]), length(out.ind))
# }


# 1.3) Feature selection based on L1


library(glmnet)

X <- data.matrix(x.train)
y <- y.train

# cv.lasso <- cv.glmnet(X,y,alpha=1)
# cv.lasso$lambda.1se; cv.lasso$lambda.min
# lambda <- (cv.lasso$lambda.1se + cv.lasso$lambda.min)/2
lambda <- 0.6340848
lasso.mod <- glmnet(X, y, alpha=1, lambda = lambda)
vars <- (which(coef(lasso.mod)!= 0) - 1)[-1]


# 1.4) Remove the noise variables


x.train <- read.csv("X_train.csv")[,-1]
y.train <- read.csv("y_train.csv")[,-1]
x.test <- read.csv("X_test.csv")[,-1]

x.train <- x.train[,vars]
x.test <- x.test[,vars]



########################################################################



# 1) Impute the NAs with KNN (different imputation?, different k?)


library(DMwR)
X.join <- rbind(x.train,x.test)
n <- dim(X.join)[1]; d <- dim(X.join)[2]
XnoNA <- X.join

for (i in 1:d) {
  na.ind <- which(is.na(XnoNA[,i]))
  XnoNA[na.ind, i] <- rep(median(na.omit(XnoNA[, i])), length(na.ind))
}

for (i in 1:n) {
  XnoNA[i,] <- X.join[i,]
  XnoNA <- knnImputation(XnoNA, k = 5)
  X.join[i,] <- XnoNA[i,]
}

x.train <- X.join[1:dim(x.train)[1],]
x.test <- X.join[-(1:dim(x.train)[1]),]

data <- data.frame(cbind(y.train, x.train))



# # LM imputation
# X.join <- rbind(x.train,x.test)
# n <- dim(X.join)[1]; d <- dim(X.join)[2]
# XnoNA <- X.join
# 
# for (i in 1:d) {
#   na.ind <- which(is.na(XnoNA[,i]))
#   XnoNA[na.ind, i] <- rep(median(na.omit(XnoNA[, i])), length(na.ind))
# }
# 
# for (i in 1:n) {
#   na.ind <- which(is.na(X.join[i,]))
#   if (length(na.ind > 0)) {
#   for (j in 1:length(na.ind)) {
#     XnoNA[i,na.ind[j]] <- lm(XnoNA[,na.ind[j]] ~ data.matrix(XnoNA[,-na.ind[j]]))$fitted.values[i] 
#   }
#   }
#   X.join[i,] <- XnoNA[i,]
# }
# 
# x.train <- X.join[1:dim(x.train)[1],]
# x.test <- X.join[-(1:dim(x.train)[1]),]
# 
# data <- data.frame(cbind(y.train, x.train))



# # RF imputation
# library(randomForest)
# X.join <- rbind(x.train,x.test)
# n <- dim(X.join)[1]; d <- dim(X.join)[2]
# XnoNA <- X.join
# 
# for (i in 1:d) {
#   na.ind <- which(is.na(XnoNA[,i]))
#   XnoNA[na.ind, i] <- rep(median(na.omit(XnoNA[, i])), length(na.ind))
# }
# 
# for (i in 1:n) {
#   na.ind <- which(is.na(X.join[i,]))
#   if (length(na.ind > 0)) {
#     for (j in 1:length(na.ind)) {
#       XnoNA[i,na.ind[j]] <- predict(randomForest(data.matrix(XnoNA[,-na.ind[j]]),
#                                          XnoNA[,na.ind[j]], ntree=100),
#                                     newdata = data.matrix(XnoNA[i,-na.ind[j]]))
#     }
#   }
#   X.join[i,] <- XnoNA[i,]
# }
# 
# x.train <- X.join[1:dim(x.train)[1],]
# x.test <- X.join[-(1:dim(x.train)[1]),]
# 
# data <- data.frame(cbind(y.train, x.train))
# 
# save(data, "dataRF.RData")





# 2) Outlier detection for multivarite and deletion 


mod <- lm(y.train ~., data)
# out.leverage <- which(hatvalues(mod) > (2*dim(data)[2])/dim(data)[1])
# out.y <-  which(y - quantile(y,0.75) > 1.5 * IQR(y) |
#                   quantile(y,0.25) - y > 1.5 * IQR(y))
out.cook <- which(cooks.distance(mod) > 3.5*mean(cooks.distance(mod)))
data.out <- data[-out.cook,]
plot(cooks.distance(mod))
abline(h = 4*mean(cooks.distance(mod)), col = "red")



###################################################################

# Second round imputation

# library(randomForest)
# library(e1071)
# 
# XnoNA <- XnoNA[-out.cook,]
# 
# x.train1 <- read.csv("X_train.csv")[,-1]
# y.train1 <- read.csv("y_train.csv")[,-1]
# x.test1 <- read.csv("X_test.csv")[,-1]
# 
# x.train1 <- x.train1[,vars]
# x.test1 <- x.test1[,vars]
# 
# 
# X.join1 <- rbind(x.train1,x.test1)
# X.join1 <- X.join1[-out.cook,]
# n <- dim(X.join1)[1]; d <- dim(X.join1)[2]
# 
# for (i in 1:n) {
#   na.ind <- which(is.na(X.join1[i,]))
#   if (length(na.ind > 0)) {
#     for (j in 1:length(na.ind)) {
#       XnoNA[i,na.ind[j]] <- predict(svm(data.matrix(XnoNA[,-na.ind[j]]),
#                                          XnoNA[,na.ind[j]], kernel = "radial"),
#                                     newdata = data.matrix(XnoNA[i,-na.ind[j]]))
#     }
#   }
#   X.join1[i,] <- XnoNA[i,]
# }
# 
# x.train <- X.join1[1:(dim(x.train1)[1] - length(out.cook)),]
# x.test <- X.join1[-(1:dim(x.train)[1]),]
# 
# data <- data.frame(cbind(y.train1[-out.cook], x.train))
# names(data)[1] <- "y.train"

# load("SVRdata.RData")
# load("SVRdatatest.RData")
# data.out <- dataSVR


# # Second round imputation
# 
# mod <- lm(y.train ~., data)
# plot(cooks.distance(mod))
# abline(h = 4*mean(cooks.distance(mod)), col = "red")


###############################################################
        
# 3) Make a model based on data


# #  First try LM useless
rsq <- function(pred, true){
  rsq <- 1 - sum((pred-true)^2)/sum((true - mean(true))^2)
  return(rsq)
}
# lm.mod <- lm(y.train~., data = data.out)
# summary(lm.mod)
# rsq(lm.mod$fitted.values,data.out$y.train)
# rsq(round(lm.mod$fitted.values),data.out$y.train)
# 
# 
# # Try best subsets
# library(leaps)
# regfit.best <- regsubsets(y.train~., data = data.out, nvmax = 30)
# summary(regfit.best)$cp


# # Try elnet
# X <- data.matrix(data[,-1])
# y <- data[,1]
# cv.elnet <- cv.glmnet(X,y,alpha=0.5) 
# lambda <- cv.elnet$lambda.min
# elnet.mod <- glmnet(X, y, alpha=0.5, lambda = lambda)
# rsq(predict(elnet.mod, newx = X),data$y)
# rsq(round(predict(elnet.mod, newx = X)),data$y)


# Try RFR
library(randomForest)
rf.reg <- randomForest(y.train~., data.out, 
                       importance=TRUE,ntree=1000)
rsq(predict(rf.reg),data.out$y.train)
rsq(round(predict(rf.reg)),data.out$y.train)
# importance(rf.reg)
data.out <- data.out[,c(1, (order(importance(rf.reg)[,1]) + 1)[-(1:3)])]
x.test <- x.test[ ,order(importance(rf.reg)[,1])[-(1:3)]]
names(data.out)[2:27] == names(x.test)
# rf.reg.imp <- randomForest(y.train~., data.out,mtry=dim(data.out)[2]/3,
#                        importance=TRUE,ntree=1000)
# rsq(predict(rf.reg.imp),data.out$y.train)
# rsq(round(predict(rf.reg.imp)),data.out$y.train)


# Try SVR

# library(e1071)
# X <- data.matrix(data.out[,-1])
# y <- data.out[,1]
# tune <- tune.svm(y =  y, x = X, kernel = "radial",
#                  cost = 2^c(-5:5), gamma = 2^c(-5:5)) # C = 4, G = 0.03125
# best_C <- tune$best.parameters$cost
# best_G <- tune$best.parameters$gamma
best_C <- 4; best_G <- 0.03125
# model.svm <- svm(y =  y, x = X, kernel = "radial", cost = best_C, gamma = best_G)
# rsq(predict(model.svm),data.out$y)
# rsq(round(predict(model.svm)),data.out$y)


# # 4) Evaluation of the models
# 
# 
# k <- 10
# shuff <- 10
# CVerror <- rep(NA,shuff)
# X <- data.matrix(data.out[,-1])
# y <- data.out[,1]
# n <- length(y)
# 
# for (u in 1:shuff) {
#   rand <- sample(n,replace = F)
#   y_rand <- y[rand]
#   X_rand <- X[rand,]
#   error <- rep(NA,k)
#   for (i in 1:k) {
#     index <- ((i-1)*(n/k)+1):(i*(n/k))
#     train_y <- y_rand[-index]
#     test_y <- y_rand[index]
#     train_X <- X_rand[-index,]
#     test_X <- X_rand[index,]
#     model1 <- svm(y =  train_y, x = train_X, kernel = "radial", 
#                   cost = best_C, gamma = best_G)
# #    model2 <- randomForest(train_X, train_y, mtry = dim(train_X)[2]/3, 
# #                           importance=TRUE,ntree=1000)
#     pred1 <- predict(model1, newdata = test_X)
# #    pred2 <- predict(model2, newdata = test_X)
#     error[i] <- rsq(pred1,test_y)
# #    error[i] <- rsq(((pred1 + pred2)/2),test_y)
#   }
#   CVerror[u] = sum(error)/k
# }
# 
# mean(CVerror)
# 
# 
# 
# # Gradient Boosting
# 
# library(gbm)
# # model.boost <- gbm(y.train ~ . ,data = data.out, distribution = "laplace",
# #                    n.trees = 10000, shrinkage = 0.005, interaction.depth = 5)
# # summary(model.boost)
# # rsq(model.boost$fit, data.out$y.train)
# 
# 
# k <- 10
# shuff <- 1
# CVerror <- rep(NA,shuff)
# X <- data.matrix(data.out[,-1])
# y <- data.out[,1]
# n <- length(y)
# 
# for (u in 1:shuff) {
#   rand <- sample(n,replace = F)
#   y_rand <- y[rand]
#   X_rand <- X[rand,]
#   error <- rep(NA,k)
#   for (i in 1:k) {
#     index <- ((i-1)*(n/k)+1):(i*(n/k))
#     train_y <- y_rand[-index]
#     test_y <- y_rand[index]
#     train_X <- X_rand[-index,]
#     test_X <- X_rand[index,]
#     data.train <- data.frame(cbind(train_y,train_X))
#     data.test <- data.frame(test_X)
#     model1 <- gbm(train_y ~ . ,data = data.train, distribution = "laplace",
#                   n.trees = 10000, shrinkage = 0.01, interaction.depth = 5)
#     #    model2 <- randomForest(train_X, train_y, mtry = dim(train_X)[2]/3, 
#     #                           importance=TRUE,ntree=1000)
#     pred1 <- predict(model1, newdata = data.test, n.trees = 10000)
#     #    pred2 <- predict(model2, newdata = test_X)
#     error[i] <- rsq(pred1,test_y)
#     #    error[i] <- rsq(((pred1 + pred2)/2),test_y)
#   }
#   CVerror[u] = sum(error)/k
# }
# 
# mean(CVerror)
# 
# 
# # XGboost 
# 
# 
# library(xgboost)
# # model.xgboost <- xgboost(data = data.matrix(data.out[,-1]), label = data.out[,1],
# #                          eta = 0.01, max_depth = 15,
# #                   subsample = 0.5, nrounds = 1000)
# # summary(model.xgboost)
# # rsq(predict(model.xgboost, newdata = data.matrix(data.out[,-1])), data.out$y.train)
# 
# k <- 10
# shuff <- 1
# CVerror <- rep(NA,shuff)
# X <- data.matrix(data.out[,-1])
# y <- data.out[,1]
# n <- length(y)
# 
# for (u in 1:shuff) {
#   rand <- sample(n,replace = F)
#   y_rand <- y[rand]
#   X_rand <- X[rand,]
#   error <- rep(NA,k)
#   for (i in 1:k) {
#     index <- ((i-1)*(n/k)+1):(i*(n/k))
#     train_y <- y_rand[-index]
#     test_y <- y_rand[index]
#     train_X <- X_rand[-index,]
#     test_X <- X_rand[index,]
#     model1 <- xgboost(data = train_X, label = train_y, eta = 0.005, max_depth = 15, 
#                       subsample = 0.5, nrounds = 2000)
#     pred1 <- predict(model1, newdata = test_X)
#     error[i] <- rsq(pred1,test_y)
#   }
#   CVerror[u] = sum(error)/k
# }
# 
# mean(CVerror)




# 5) Write the solution

prediction_mat <- matrix(NA, nrow = 10, ncol = 776)

for (i in 1:10) {
set.seed(i)
X <- data.matrix(data.out[,-1])
y <- data.out[,1]
sample <- read.csv("sample.csv")
sol <- sample$id
model_sol1 <-  svm(y = y, x = X, kernel = "radial",
                   cost = best_C, gamma = best_G)
model_sol2 <- randomForest(X, y, mtry = dim(X)[2]/3, 
                           importance=TRUE,ntree=10000)
model_sol3 <- gbm(y.train ~ . ,data = data.out, distribution = "gaussian",
                  n.trees = 10000, shrinkage = 0.01, interaction.depth = 7)
model_sol4 <- xgboost(data = X, label = y, eta = 0.005, max_depth = 15, 
                      subsample = 0.5, nrounds = 2000)
prediction_sol <- (predict(model_sol1, newdata = x.test) +
                     predict(model_sol2, newdata = x.test) +
                     predict(model_sol3, newdata = x.test, n.trees = 10000) +
                     predict(model_sol4, newdata = data.matrix(x.test)))/4
prediction_mat[i,] <- prediction_sol
}

prediction_sol_avg <- colMeans(prediction_mat)
solution <- data.frame(id = sol, y = prediction_sol_avg)
write.table(solution, file = "solution.csv", row.names = FALSE, sep = ",")

# round.ind <- which(abs(prediction_sol - round(prediction_sol)) < 0.15)
# prediction_sol_round <- prediction_sol
# prediction_sol_round[round.ind] <- round(prediction_sol_round[round.ind])
# solution.r <- data.frame(id = sol, y = prediction_sol_round)
# write.table(solution.r, file = "solutionr.csv", row.names = FALSE, sep = ",")


 