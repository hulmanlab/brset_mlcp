library(readr)
library(dplyr)
library(pROC)
library(caret)
library(mcca)
library(ggplot2)
rm(list = ls(all.names = TRUE)) 
gc()

library(optparse)

option_list <- list(
  make_option("--prob_root", default = "BRSET_TL_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
# prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/BRSET_TL_b'
setwd(prob_root)
files <- list.files(prob_root)
files <- files[
  grepl("^y.*\\.csv$", files) &
  !grepl("convnextv2|resnet", files)
]
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
    # print(name)
    parts <- unlist(strsplit(name, "_"))
    model <- if (grepl("retfound_d2_s", name)) {
      "RETFound DINOv2 Shanghai"
    } else if (grepl("retfound_d2_m", name)) {
      "RETFound DINOv2 MEH"
    } else if (grepl("retfound", name)) {
      "RETFound"
    } else if (grepl("dinov3_large", name)) {
      "DINOv3 Large"
    } else if (grepl("visionfm", name)) {
      "VisionFM"
    } else if (grepl("dinov2", name)) {
      "DINOv2 Large"
    } else if (grepl("eyeclip", name)) {
      "EyeCLIP"
    } else if (grepl("convnext", name)) {
      "ConvNeXt"
    } else if (grepl("resnet200d", name)) {
      "ResNet200d"
    } else if (grepl("resnet50", name)) {
      "ResNet50"
    } else {
      "Unknown Model"
    }
    if (grepl("_fine_tune_", name)) {
      mode<-"Full fine-tune" # Head, Full
    } else {
      mode<-"Head fine-tune"
    }
    df<-read.csv(name)
    
    df$y_camera <- NULL
    arr <- as.matrix(df)
    
    # Extract labels and scores
    if (ncol(arr) > 3) {
      y_true <- arr[, 1:3]
      y_score <- arr[, 4:ncol(arr)]
    } else {
      y_true <- matrix(
        as.numeric(arr[, 1]),
        ncol = 1
      )
      y_score <- matrix(
        as.numeric(arr[, 2]),
        ncol = 1
      )
    }

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
    
      if (ncol(arr) > 3) {
          true_labels <- apply(y_resample, 1, which.max)
        } else {
          true_labels <- y_resample
        }

      # Multiclass Brier score
      # https://github.com/benvancalster/OrdinalCalibration/blob/main/ordcalfunctions.R l 230
      if (ncol(arr) > 3) {
        brier_cal <- mbrier(outc = true_labels, preds = y_score_resample, k = 3)
        brier_scores[itr] <- brier_cal
      } else {
        brier_cal <- mbrier(outc = true_labels, preds = y_score_resample, k = 2)
        brier_scores[itr] <- brier_cal
      }
    }
    
    mean_brier <- mean(brier_scores)
    ci_brier <- quantile(brier_scores, c(0.025, 0.975))
    
    # Append to dataframe
    brier_results <- rbind(brier_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      brier = sprintf("%.2f [%.2f, %.2f]", mean_brier, ci_brier[1], ci_brier[2]),
      mean_brier = mean_brier,
      lower_brier = ci_brier[1],
      upper_brier = ci_brier[2],
      stringsAsFactors = FALSE
    ))
  }
}
# print(brier_results)
write.csv(brier_results, file.path(prob_root, "summary", "Brier_results.csv"), row.names = FALSE)

gc()