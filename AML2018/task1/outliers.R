

x_train <- read.csv("/Users/Jakub/Documents/ETH/AML/task1/X_train.csv", header=T)[-1]
x_test <- read.csv("/Users/Jakub/Documents/ETH/AML/task1/X_test.csv", header=T)[-1]



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


remove_outliers <- function(data, a, b) {
  num_columns <- dim(data)[2]
  num_rows <- dim(data)[1]
  
  for (i in 1:num_columns) {
    print(i)
    for (j in 1:num_rows) {
      if (is.na(data[j,i]) == TRUE) {
        data[j,i] <- median(data[,i])
      }
      else if ((is.na(data[j,i]) == FALSE) & (data[j,i] < quantile(data[,i], a, na.rm=T))) {
          # print(((is.na(data[j,i]) == FALSE) & (data[j,i] < quantile(data[,i], 0.05, na.rm=T))))
          data[j,i] <- median(data[,i])
      }
      else if (((is.na(data[j,i]) == FALSE)) & (data[j,i] > quantile(data[,i], b, na.rm=T))) {
          data[j,i] <- median(data[,i])
      }
      else {
          data[j,i] <- data[j,i]
      }
      
    }
  }
  return(data)
}



x_test1 <- replace_nas(x_test)
x_test_01 <- remove_outliers(x_test1, a=0.02, b=0.98)  

x_train1 <- replace_nas(x_train)
x_train_01 <- remove_outliers(x_train1, a=0.02, b=0.98)  

write.csv(x_test_01, "/Users/Jakub/Documents/ETH/AML/task1/X_test1.csv")
write.csv(x_train_01, "/Users/Jakub/Documents/ETH/AML/task1/X_train1.csv")