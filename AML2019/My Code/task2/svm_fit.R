#################################
# SET THESE PARAMETERS BEFORE RUNNING
#################################
file.path <- "/Users/leandereberhard/Desktop/ETH/AML/task2"

nfolds.inner <- 10
nfolds.outer <- 5

grid.fineness <- 10
#################################
#################################



#################################
# import the data 
#################################
setwd(file.path)

list.files(path = getwd())

X_test <- read.csv("X_test.csv")
X_train <- read.csv("X_train.csv")
y_train <- read.csv("y_train.csv")

# remove the id row
test.id <- X_test[,1]
X_test = X_test[,-1]
X_train = X_train[,-1]
y_train = y_train[,-1]

# count the number of points in each class
sum(y_train == 0) # 600
sum(y_train == 1) # 3600
sum(y_train == 2) # 600
#################################
################################# 


#################################
# SVM to separate class 1 from the rest
#################################
# first assign all points from class 0 and 2 to a new class 3
y_train.two.class <- y_train
y_train.two.class[y_train %in% c(0,2)] <- 3

sum(y_train.two.class == 3) # 1200
sum(y_train.two.class == 1) # 3600

# set up class weights
class.weight1 <- (3600  / (1200 + 3600))^-1
class.weight3 <- (1200 / (1200 + 3600))^-1

# use only the features that are highly correlated with these new responses
correlations.two.class <- cor(cbind(y_train.two.class, X_train))

dim(correlations.two.class)
high.cor.ind.two.class <- abs(correlations.two.class[1,1:1001]) >= .4

sum(high.cor.ind.two.class)

X_train.high.cor.two.class <- X_train[,high.cor.ind.two.class]

data.two.class <- data.frame(y = as.factor(y_train.two.class), X_train.high.cor.two.class)



# get an idea of the structure of the data 
feature.correlations <- cor(X_train.high.cor.two.class)

# find the feature that is most correlated with y
which(feature.correlations[1,-1] == max(abs(feature.correlations[1,-1])))

# find the feature that is least correlated with x562
which(feature.correlations["x983",] == min(abs(feature.correlations["x983",])))

# plot the classes versus these two features
plot(data.two.class[,"x983"], data.two.class[,"x885"], col = y_train.two.class, pch = 20)



################################# 
# function to calculate BMAC
#################################
bmac <- function(data, lev = NULL, model = NULL){
  # data has columns obs and pred 
  sum.recall <- 0
  n.classes <- length(unique(data$obs))
  
  for (i in unique(data$obs)){
    tp <- sum(data$pred == i & data$obs == i)
    fn <- sum(data$pred != i & data$obs == i)
    recall <- tp / (tp + fn)
    
    sum.recall <- sum.recall + recall
  }
  c(BMAC = sum.recall / n.classes)
}
################################# 
################################# 


# for easy CV 
library(caret)
library(doParallel)

train.control <- trainControl(method = "repeatedcv", number = nfolds.inner, repeats = nfolds.outer, summaryFunction = bmac)

# set the parameter grid to search over

param.grid <- expand.grid(C = 10^seq(-7,2, length.out = grid.fineness), 
                          sigma = 10^seq(-7,2, length.out = grid.fineness))

# set number of cores
clust <- makePSOCKcluster(4)
registerDoParallel(clust)

# perform CV 
model.svm1 <- train(y ~ ., data = data.two.class, 
                    method = 'svmRadial', 
                    trControl = train.control, 
                    metric = "BMAC",
                    tuneGrid = param.grid,
                    class.weights = c("1" = class.weight1, "3" = class.weight3),
                    type = "C-svc")

stopCluster(clust)

print(model.svm1)

# save optimal parameters 
C1 <- model.svm1$bestTune$C # 1
sigma1 <- model.svm1$bestTune$sigma # 0.004641589



# fit SVM using selected parameters

svm.two.class.fit <- ksvm(y ~ ., data = data.two.class,
                          C = C1, 
                          kpar = list(sigma = sigma1), # kpar sets kernel hyperparameters
                          kernel = "rbfdot", 
                          class.weights = c("1" = class.weight1, "3" = class.weight3))

# calculate training BMAC 
pred <- predict(svm.two.class.fit)
two.class.pred <- data.frame(pred = pred, obs = y_train.new)

bmac(two.class.pred)
################################## 
##################################


################################## 
#Separate the remaining two classes
##################################

# create training data; only use highly correlated features
min.ind <- y_train %in% c(0,2)
X_train.min <- X_train[min.ind,]
y_train.min <- y_train[min.ind]

correlations <- cor(cbind(y_train.min, X_train.min))
high.cor.ind.min.class <- abs(correlations[1,1:1001]) >= .2

sum(high.cor.ind.min.class)

X_train.high.cor.min <- X_train.min[, high.cor.ind.min.class]

data.min.class <- data.frame(y = as.factor(y_train.min), X_train.high.cor.min)





# CV
train.control <- trainControl(method = "repeatedcv", number = 3, repeats = 2, summaryFunction = bmac)

# set the parameter grid to search over
param.grid <- expand.grid(C = 10^seq(-7,2, length.out = grid.fineness), 
                          sigma = 10^seq(-7,2, length.out = grid.fineness))

# set number of cores
clust <- makePSOCKcluster(4)
registerDoParallel(clust)

# perform CV 
model.svm2 <- train(y ~ ., data = data.min.class, 
                    method = "svmRadial", 
                    trControl = train.control, 
                    metric = "BMAC",
                    tuneGrid = param.grid,
                    type = "C-svc")

stopCluster(clust)

print(model.svm2)

C2 <- model.svm2$bestTune$C # 1
sigma2 <- model.svm2$bestTune$sigma # 0.001


# fit the selected model and calculate training BMAC

svm.min.class.fit <- ksvm(y ~ ., data = data.min.class,
                          C = C2, 
                          kpar = list(sigma = sigma2), # kpar sets kernel hyperparameters
                          kernel = "rbfdot")

# calculate training BMAC 
pred <- predict(svm.min.class.fit)
min.class.pred <- data.frame(pred = pred, obs = y_train.min)

bmac(min.class.pred)
##################################
##################################



##################################
# predict for test data using fitted models
##################################

# process the data in the same way as the training data
# use same indices to select features as in training data

# split into majority class vs minority classes 

X_test.high.cor <- X_test[,high.cor.ind.two.class]

pred.two.class <- predict(svm.two.class.fit, newdata = X_test.high.cor)


# set min.ind as the points which were classified as 3 by the first model
test.min.ind <- pred.two.class == 3

X_test.min <- X_test[test.min.ind,]
X_test.high.cor.min <- X_test.min[, high.cor.ind.min.class]

pred.min.class <- predict(svm.min.class.fit, newdata = X_test.high.cor.min)

test.pred <- as.numeric(as.character(pred.two.class))
test.pred[test.pred == 3] <- as.numeric(as.character(pred.min.class))
test.pred

sum(test.pred == 0)
sum(test.pred == 1)
sum(test.pred == 2)


# export to csv
out <- data.frame(id = test.id, y = test.pred)
write.csv(out, "task2_test_pred2.csv")

cat("Training BMAC Minority vs. Majority", bmac(two.class.pred))
cat("Training BMAC Between Minority Classes", bmac(min.class.pred))
##################################
##################################







