path <- "/Users/leandereberhard/Desktop/ETH/AML/task3"
setwd(path)

train <- read.csv("processed_logged.csv")
head(train)

train$y <- as.factor(train$y)

################################# 
# function to calculate micro-averaged F1 score
#################################
f1 <- function(data, lev = NULL, model = NULL){
  # data has columns obs and pred 
  sum.tp <- 0
  sum.fn <- 0
  sum.fp <- 0
  
  n.classes <- length(unique(data$obs))
  
  for (i in unique(data$obs)){
    tp <- sum(data$pred == i & data$obs == i)
    fn <- sum(data$pred != i & data$obs == i)
    fp <- sum(data$pred == i & data$obs != i)
    
    sum.tp <- sum.tp + tp
    sum.fn <- sum.fn + fn
    sum.fp <- sum.fp + fp
  }
  
  recall <- tp / (tp + fn)
  precision <- tp / (tp + fp)
  
  c(F1 = 2 * precision * recall / (precision + recall))
}

accuracy <- function(data, lev = NULL, model = NULL){
  classes <- unique(data$obs)
  correct <- sum(data$pred == data$obs)
  incorrect <- sum(data$pred != data$obs)
  
  c(Accuracy = correct / (correct + incorrect))
}
################################# 
################################# 

# test to see if these functions are the same 
pred <- c(1,1,1,0,1,1,1,1,1)
obs <- c(1,1,1,0,1,1,1,1,0)

data <- data.frame(pred = pred, obs = obs)

f1(data); accuracy(data)



######### Class weights, if needed
n.class0 <- sum(train$y == 0)
n.class1 <- sum(train$y == 1)
n.class2 <- sum(train$y == 2)
n.class3 <- sum(train$y == 3)

total.points <- n.class0 + n.class1 + n.class2

class.weight0 <- (n.class0 / (total.points))^-1
class.weight1 <- (n.class1 / (total.points))^-1
class.weight2 <- (n.class2 / (total.points))^-1
class.weight3 <- (n.class3 / (total.points))^-1



# fit model 
library(caret)
library(doParallel)
library(kernlab)


################################## 
#Fit ALl Classes Together
##################################
n.cores <- 4
nfolds.outer <- 5
nfolds.inner <- 2
grid.fineness <- 1


# CV
train.control <- trainControl(method = "repeatedcv", 
                              number = nfolds.outer, 
                              repeats = nfolds.inner,
                              summaryFunction = f1)

# set the parameter grid to search over
param.grid <- expand.grid(C = 10^seq(-3,3, length.out = grid.fineness),
                          sigma = 10^seq(-5,5, length.out = grid.fineness))

# param.grid <- data.frame(C = 9.482505, sigma = 8.376776e-06)



# set number of cores
clust <- makePSOCKcluster(n.cores)
registerDoParallel(clust)

# perform CV 
model.svm <- train(y ~ ., data = train, 
                    method = "svmRadial", 
                    trControl = train.control, 
                    metric = "F1",
                    tuneGrid = param.grid,
                    type = "C-svc",
                    class.weights = c("0" = class.weight0, "1" = class.weight1, "2" = class.weight2, "3" = class.weight3),
                    preProcess = c("center", "scale"))

stopCluster(clust)

print(model.svm)


# best params:
# C = 2.154435
# sigma = 0.000129155




