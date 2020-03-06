
####################################################################

# Task 2: Multiclass classification

install.packages("reticulate")
devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow()

#####################################################################

X.train <- read.csv("X_train.csv")[-1]
y.train <- read.csv("y_train.csv")[-1]
X.test <- read.csv("X_test.csv")[-1]


# Define the accuracy metric

# bmac <- function(true, pred){
#   C <- length(unique(true))
#   error.sum <- 0
#   for (i in 1:C) {
#     class <- unique(true)[i]
#     index <- which(true == class)
#     error.sum <- error.sum + mean(true[index] == pred[index])
#   }
#   return((1/C)*(error.sum))
# }

# Autoencoder dimension reduction

library(keras)
X <- data.matrix(X.train)

model.ae <- keras_model_sequential()
model.ae %>%
  layer_dense(units = 500, activation = "tanh", 
              input_shape = ncol(X), name = "bottleneck") %>%
  layer_dense(units = ncol(X))
summary(model.ae)

model.ae %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

model.ae %>% fit(
  x = X, 
  y = X, 
  epochs = 500,
  batch_size = ncol(X),
  verbose = 2
)

mse.ae <- evaluate(model.ae, X, X)
mse.ae

intermediate_layer_model <- keras_model(inputs = model.ae$input, outputs = get_layer(model.ae, "bottleneck")$output)
X.ae <- predict(intermediate_layer_model, X)

y.train <- as.numeric(y.train[[1]])
y.train <- as.factor(y.train)
data <- data.frame(y.train, X.ae)

