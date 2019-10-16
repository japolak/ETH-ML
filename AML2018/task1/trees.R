library(randomForest)
library(xgboost)

replace_nas <- function(data) {
  for (l in 1:ncol(data)) {
    print(l)
    med1 <- median(data[,l], na.rm=TRUE)
    for (k in 1:nrow(data)) {
      data[k,l] <- ifelse(is.na(data[k,l]), med1, data[k,l])
    }
  }
  colnames(data)[1] <- "id"
  return(data)
}

# READ IN THE ORIGINAL DATA (here already NAs and outliers removed)
x_train <- read.csv("/home/leslie/Desktop/AML/Task1/X_train2.csv", header=T)[,-1]
y_train <- read.csv("/home/leslie/Desktop/AML/Task1/y_train.csv", header=T)[,-1]
x_test <- read.csv("/home/leslie/Desktop/AML/Task1/X_test1.csv", header=T)[,-1]

# remove 0 columns:
x_train <- x_train[,-which(apply(x_train, 2, var)==0)]

# fit random forest to input into extract function
data <- cbind(y_train, x_train)
brain <- randomForest(y_train ~ ., data=data, ntree=500, importance=T)

# function to identify most important variables, returns csv that is a truncated version of variables. 
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



a <- extract_variables(rf_object=brain, threshold=25)
                
                      
# READ IN TRUNCATED VARIABLES TO FIT GRADIENT BOOSTED TREES
x_train_trunc <- read.csv("/home/leslie/Desktop/AML/Task1/trunc_x_train.csv", header=T)[,-1]
y_train_trunc <- read.csv("/home/leslie/Desktop/AML/Task1/y_train.csv", header=T)[,-1]
x_test_trunc <- read.csv("/home/leslie/Desktop/AML/Task1/trunc_x_test.csv", header=T)[,-1]


### CV FUNCTION FOR GRADIENT BOOSTED TREES
gbt_cv_function <- function(x, y, k=2, e.ta = 0.1, n.rounds=100) {

  x_dim <- dim(x)[1]
  fold <- sample(rep(1:k, length.out=x_dim), size=x_dim, replace=FALSE)
  r2 <- c(length(k))
  
  for (i in 1:k) {
    
    # create test set
    test_ind <- which(fold==i)
    x_test_set <- x[test_ind,]
    y_test_set <- y[test_ind]
    y_test_mean <- mean(y_test_set)
    
    # create train set
    x_train_set <- x[-test_ind,]
    y_train_set <- y[-test_ind]
    
    # combine data
    com_data <- as.matrix(cbind(y_train_set, x_train_set))

    xgb.fit <- xgboost::xgboost(data = as.matrix(x_train_set), label = y_train_set,
                                booster = "gbtree",
                                objective = "reg:linear",
                                eta = e.ta,
                                nrounds = n.rounds,
                                max.depth=4)
    
    y_pred <- predict(xgb.fit, as.matrix(x_test_set))
    
    # calculate r2
    r2[i] <- 1 - sum(((y_test_set - y_pred)^2))/sum(((y_test_set - y_test_mean)^2))
                                
  }
  return(mean(r2))
}


a <- gbt_cv_function(x=x_train_trunc, y=y_train_trunc, e.ta=0.01, n.rounds=5000, k=2)
print(a)

####### FIT FULL MODEL, PREDICT AND SAVE TO CSV
xgb.fit <- xgboost::xgboost(data = as.matrix(x_train), label = y_train,
                            booster = "gbtree",
                            objective = "reg:linear",
                            eta = 0.01,
                            nrounds = 10000,
                            max.depth=4)

y_test_full_model <- predict(xgb.fit, as.matrix(x_test))
id <- seq(0, dim(x_test)[1]-1, 1)
y_test_full_model <- cbind(id, y_test_full_model)
colnames(y_test_full_model) <- c("id", "y")
write.csv(y_test_full_model, "/home/leslie/Desktop/AML/Task1/xgboost_task1.csv", row.names = FALSE)


