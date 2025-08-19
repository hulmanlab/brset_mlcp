library(readr)
library(dplyr)
library(pROC)
library(caret)
library(mcca)
rm(list = ls(all.names = TRUE)) 
gc()

prob_root<-"/home/livieymli/brset_analysis/BRSET/output_predicted_probabilities/mBRSET_EXEVAL"
setwd(prob_root)
files <- list.files(prob_root)
files <- sort(files)

brier_results <- data.frame(
  model = character(),
  mode = character(),
  brier = character(),
  mean_brier = numeric(),
  lower_brier = numeric(),
  upper_brier = numeric()
)

mbrier <- function(outc,preds,k){
  outm=1*(outc==1)
  for (j in 2:k){
    outm = cbind(outm,1*(outc==j))
  }
  return((mean((preds-outm)*(preds-outm))))
}

for (i in seq_along(files)) {
  name <- files[i]
  if (grepl("^y.*\\.csv$", name) && !grepl("reproduced", name) && !grepl("pdi", name) && !grepl("ensemble", name)) {
    print(name)
    parts <- unlist(strsplit(name, "_"))
    model <- parts[2]
    # mode <- if (parts[length(parts) - 3] == "eval") "Head fine-tune" else "Full fine-tuned"
    if (grepl("_fine_tune_", name)) {
      mode<-"Full" # Head, Full
    } else {
      mode<-"Head"
    }
    df<-read.csv(name)
    
    df$y_camera <- NULL
    arr <- as.matrix(df)
    
    # Extract labels and scores
    y_true <- arr[, 1:3]
    y_score <- arr[, 4:ncol(arr)]
    
    # Apply epsilon adjustments
    y_score[y_score == 0] <- 1e-6
    y_score[y_score == 1] <- 0.999999
    
    # Bootstrap settings
    n_iterations <- 1000
    n <- nrow(y_true)
    brier_scores <- numeric(n_iterations)
    
    
    for (itr in 1:n_iterations) {
      idx <- sample(1:n, n, replace = TRUE)
      y_resample <- y_true[idx, ]
      y_score_resample <- y_score[idx, ]
      
      true_labels <- apply(y_resample, 1, which.max)
      
      # Multiclass Brier score
      # https://github.com/benvancalster/OrdinalCalibration/blob/main/ordcalfunctions.R l 230
      brier_cal <- mbrier(outc = true_labels, preds = y_score_resample, k = 3)
      brier_scores[itr] <- brier_cal
      
    }
    
    mean_brier <- mean(brier_scores)
    ci_brier <- quantile(brier_scores, c(0.025, 0.975))
    
    # Append to dataframe
    brier_results <- rbind(brier_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      brier = sprintf("%.2f [%.2f - %.2f]", mean_brier, ci_brier[1], ci_brier[2]),
      mean_brier = mean_brier,
      lower_brier = ci_brier[1],
      upper_brier = ci_brier[2],
      stringsAsFactors = FALSE
    ))
  }
}
print(brier_results)

write.csv(brier_results, "brier_results.csv", row.names = FALSE)

