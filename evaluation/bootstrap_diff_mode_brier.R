library(readr)
library(dplyr)
library(pROC)
library(caret)
library(mcca)
rm(list = ls(all.names = TRUE)) 
gc()
library(optparse)

option_list <- list(
  make_option("--prob_root", default = "BRSET_TL")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
# prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/mBRSET_TL_b'
setwd(prob_root)
files <- list.files(prob_root)
files <- files[
  grepl("^y.*\\.csv$", files) &
  !grepl("convnextv2|resnet", files)
]
files <- sort(files)

mbrier <- function(outc,preds,k){
  outm=1*(outc==1)
  for (j in 2:k){
    outm = cbind(outm,1*(outc==j))
  }
  return((mean((preds-outm)*(preds-outm))))
}

brier_diff_results <- data.frame(
  model = character(),
  mode = character(),
  brier = character(),
  mean_diff = numeric(),
  lower_diff = numeric(),
  upper_diff = numeric()
)

# ------------------------------------------------------------
# file selection (mirrors Python)
# ------------------------------------------------------------
pairs <- lapply(
  files[grepl("_eval_(binary|3class)\\.csv$", files)],
  function(f) {
    c(f,
      sub("_eval_(binary|3class)\\.csv$", "_fine_tune_\\1.csv", f))
  }
)



# ------------------------------------------------------------
# metadata (same logic as before)
# ------------------------------------------------------------
get_model <- function(name) {
  if (grepl("retfound_d2_s", name)) "RETFound DINOv2 Shanghai"
  else if (grepl("retfound_d2_m", name)) "RETFound DINOv2 MEH"
  else if (grepl("retfound", name)) "RETFound"
  else if (grepl("dinov3_large", name)) "DINOv3 Large"
  else if (grepl("visionfm", name)) "VisionFM"
  else if (grepl("dinov2", name)) "DINOv2 Large"
  else if (grepl("eyeclip", name)) "EyeCLIP"
  else if (grepl("convnext", name)) "ConvNeXt"
  else if (grepl("resnet200d", name)) "ResNet200d"
  else if (grepl("resnet50", name)) "ResNet50"
  else "Unknown Model"
}

get_mode <- function(name) {
  if (grepl("eval", name)) "Head fine-tune" else "Full fine-tune"
}

for (pair in pairs) {

  name1 <- pair[[1]]
  name2 <- pair[[2]]

  mode <- paste0(get_mode(name1), " vs ", get_mode(name2))
  model <- get_model(name1)

  df1 <- read.csv(file.path(prob_root, name1))
  df2 <- read.csv(file.path(prob_root, name2))

  df1 <- df1 %>% select(-any_of("y_camera"))
  df2 <- df2 %>% select(-any_of("y_camera"))

  arr1 <- as.matrix(df1)
  arr2 <- as.matrix(df2)
  
  if (ncol(arr1) > 3) {
    y_true <- matrix(apply(arr1[, 1:3], 1, which.max))
    y_score1 <- arr1[, 4:ncol(arr1)]
    k <- 3
  } else {
    y_true <- matrix(as.numeric(arr1[, 1]), ncol = 1)
    y_score1 <- matrix(as.numeric(arr1[, 2]), ncol = 1)
    k <- 2
  }

  y_score1[y_score1 == 0] <- 1e-6
  y_score1[y_score1 == 1] <- 0.999999


  # arr2
  if (ncol(arr2) > 3) {
    y_score2 <- arr2[, 4:ncol(arr2)]
  } else {
    y_score2 <- matrix(as.numeric(arr2[, 2]), ncol = 1)
  }

  y_score2[y_score2 == 0] <- 1e-6
  y_score2[y_score2 == 1] <- 0.999999

  n_iterations <- 1000
  n <- nrow(y_true)
  brier_scores_diff <- numeric(n_iterations)

  for (itr in 1:n_iterations) {
    indices <- sample(1:n, n, replace = TRUE)
    y_true_sample <- y_true[indices]
    y_score1_sample <- y_score1[indices]
    y_score2_sample <- y_score2[indices]

    brier_scores1 <- mbrier(y_true_sample, y_score1_sample, k = k)
    brier_scores2 <- mbrier(y_true_sample, y_score2_sample, k = k)

    brier_scores_diff[itr] <- brier_scores1 - brier_scores2
  }

  mean_brier_diff <- mean(brier_scores_diff)
  ci_brier_diff <- quantile(brier_scores_diff, c(0.025, 0.975))

  brier_diff_results <- rbind(brier_diff_results, data.frame(
    model = paste0(model),
    mode = mode,
    brier = sprintf("%.2f [%.2f, %.2f]", mean_brier_diff, ci_brier_diff[1], ci_brier_diff[2]),
    mean_diff = mean_brier_diff,
    lower_diff = ci_brier_diff[1],
    upper_diff = ci_brier_diff[2],
    stringsAsFactors = FALSE
  ))

}  
# print(brier_diff_results)
write.csv(brier_diff_results, file.path(prob_root, "summary", "Brier_diff_mode_results.csv"), row.names = FALSE)
gc()