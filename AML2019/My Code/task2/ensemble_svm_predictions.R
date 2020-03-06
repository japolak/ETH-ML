# SET PATH to a folder containing all the results
path <- "/Users/leandereberhard/Desktop/ETH/AML/task2/SVM Predictions"
  

setwd(path)

file.names <- list.files()
n.files <- length(file.names)

id <- read.csv(file.names[1])["id"]

preds <- data.frame(read.csv(file.names[1])["y"])
names(preds) <- c("y1")

for (i in 2:n.files){
  preds[paste("y",i, sep = "")] <- read.csv(file.names[i])["y"]
}


# return a vector with the majority class prediction for each row of preds
majority.vote <- function(row){
  choices <- unique(unlist(row))
  votes <- rep(0, length = length(choices))
  for (i in 1:length(choices)){
    votes[i] <- sum(row == choices[i])
  }
  
  if (length(choices[which(votes == max(votes))]) == 1){
    majority <- choices[which(votes == max(votes))]
  } else {
    majority <- sample(choices[which(votes == max(votes))], size=1)
  }
  
  majority
}

maj.pred <- apply(preds, 1, majority.vote)
out <- data.frame(id = id, y = maj.pred)


# write file out to a csv
write.csv(out, file = "majority_vote_svm.csv", row.names = FALSE, col.names = TRUE)
