install.packages("randomForest")
library(randomForest)

# READ IN THE DATA

#x_train <- read.csv("/home/leslie/Desktop/AML/Task1/X_train2.csv", header=T)[,-1]
x_train <- read.csv("/home/leslie/Desktop/AML/Task1/X_train2.csv", header=T)[,-1]
y_train <- read.csv("/home/leslie/Desktop/AML/Task1/y_train.csv", header=T)[,-1]
x_test <- read.csv("/home/leslie/Desktop/AML/Task1/X_test1.csv", header=T)[,-1]
data <- cbind(y_train, x_train)


# fit random forest
brain <- randomForest(y_train ~ ., data=data, ntree=500, importance=T)
varImpPlot(brain)
importance(brain)
?randomForest

hist(var_imp[,2][sort.list(var_imp$IncNodePurity)], breaks=50)

extract_variables <- function(rf_object, threshold=100, x_train_data=x_train, x_test_data=x_test) {
  rows_train <- dim(x_train_data)[1]
  rows_test <- dim(x_test_data)[1]
  
  var_imp <- as.data.frame(importance(rf_object))
  
  variables_to_keep <- rownames(var_imp[which(var_imp[,2]>threshold),])
  
  trunc_x_train <- matrix(0, nrow=rows_train, ncol=length(variables_to_keep))
  trunc_x_test <- matrix(0, nrow=rows_test, ncol=length(variables_to_keep))
  
  j = 1
  for (i in variables_to_keep) {
    trunc_x_train[,j] <- x_train_data[,i]
    trunc_x_test[,j] <- x_test_data[,i]
    j = j+1
  }
  colnames(trunc_x_train) <- variables_to_keep
  colnames(trunc_x_test) <- variables_to_keep
  
  # insert id columns
  id <- seq(0,rows_train-1, 1)
  trunc_x_train <- cbind(id, trunc_x_train)
  
  id <- seq(0, rows_test-1, 1)
  trunc_x_test <- cbind(id, trunc_x_test)
  
  # save to csv
  write.csv(trunc_x_train, "/home/leslie/Desktop/AML/Task1/trunc_x_train.csv", row.names = FALSE)
  write.csv(trunc_x_test, "/home/leslie/Desktop/AML/Task1/trunc_x_test.csv", row.names = FALSE)
  
  return(list(trunc_x_train=trunc_x_train, trunc_x_test=trunc_x_test))
}

a <-extract_variables(rf_object=brain, threshold=70)

cv_function <- function(x, y, k=10) {
  
  x_dim <- dim(x)[1]
  print(x_dim)
  
  fold <- sample(rep(1:k, length.out=x_dim), size=x_dim, replace=FALSE)
  print(length(fold))
  
  r2 <- c(length(k))
  print(r2)
  
  for (i in 1:k) {
    
    # create test set
    test_ind <- which(fold==i)
    x_test_set <- x[test_ind,]
    y_test_set <- y[test_ind]
    
    # create train set
    x_train_set <- x[-test_ind,]
    y_train_set <- y[-test_ind]
    com_data <- cbind(y_train_set, x_train_set)
    
    # fit rf and predict
    brain <- randomForest(y_train_set ~ ., data=com_data, ntree=500, importance=T)
    y_pred <- predict(brain, x_test_set)
    
    # calculate mse
    r2[i] <- 1 - ((y_pred-y_test_set)^2)/((mean(y_test_set)-y_test_set)^2)
    
  }
  
  return((r2))
}

a <- cv_function(x=x_train, y=y_train)
