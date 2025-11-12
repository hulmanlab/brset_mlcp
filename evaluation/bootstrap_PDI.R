library(readr)
library(dplyr)
library(pROC)
library(caret)
library(mcca)
rm(list = ls(all.names = TRUE)) 
gc()

prob_root <- file.path(getwd(), "output", "predicted_probabilities", "mBRSET_EXEVAL")
setwd(prob_root)
files <- list.files(prob_root)
files <- sort(files)

pdi_results <- data.frame(
  model = character(),
  mode = character(),
  pdi = character(),
  mean_pdi = numeric(),
  lower_pdi = numeric(),
  upper_pdi = numeric()
)

pdi_category_results <- data.frame(
  model = character(),
  mode = character(),
  pdi = character(),
  mean_pdi = numeric(),
  lower_pdi = numeric(),
  upper_pdi = numeric(),
  pdi_0 = character(),
  mean_pdi_0 = numeric(),
  lower_pdi_0 = numeric(),
  upper_pdi_0 = numeric(),
  pdi_1 = character(),
  mean_pdi_1 = numeric(),
  lower_pdi_1 = numeric(),
  upper_pdi_1 = numeric(),
  pdi_2 = character(),
  mean_pdi_2 = numeric(),
  lower_pdi_2 = numeric(),
  upper_pdi_2 = numeric()
)

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
    pdi_scores <- numeric(n_iterations)
    
    pdi_test_0 <- numeric(n_iterations)
    pdi_test_1 <- numeric(n_iterations)
    pdi_test_2 <- numeric(n_iterations)
    
    for (itr in 1:n_iterations) {
      idx <- sample(1:n, n, replace = TRUE)
      y_resample <- y_true[idx, ]
      y_score_resample <- y_score[idx, ]
      
      true_labels <- apply(y_resample, 1, which.max)
      
      # PDI
      pdi_cal <- pdi(y = true_labels, d = y_score_resample, method = "prob")
      pdi_scores[itr] <- pdi_cal$measure
      pdi_test_0[itr] <- pdi_cal$table$VALUES[1]
      pdi_test_1[itr] <- pdi_cal$table$VALUES[2]
      pdi_test_2[itr] <- pdi_cal$table$VALUES[3]
      
    }
    
    mean_pdi <- mean(pdi_scores)
    ci_pdi <- quantile(pdi_scores, c(0.025, 0.975))
    mean_pdi_test_0 <- mean(pdi_test_0)
    ci_pdi_test_0 <- quantile(pdi_test_0, c(0.025, 0.975))
    mean_pdi_test_1 <- mean(pdi_test_1)
    ci_pdi_test_1 <- quantile(pdi_test_1, c(0.025, 0.975))
    mean_pdi_test_2 <- mean(pdi_test_2)
    ci_pdi_test_2 <- quantile(pdi_test_2, c(0.025, 0.975))
    
    # Append to dataframe
    pdi_category_results <- rbind(pdi_category_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      pdi = sprintf("%.2f [%.2f, %.2f]", mean_pdi, ci_pdi[1], ci_pdi[2]),
      pdi_0 = sprintf("%.2f [%.2f, %.2f]", mean_pdi_test_0, ci_pdi_test_0[1], ci_pdi_test_0[2]),
      pdi_1 = sprintf("%.2f [%.2f, %.2f]", mean_pdi_test_1, ci_pdi_test_1[1], ci_pdi_test_1[2]),
      pdi_2 = sprintf("%.2f [%.2f, %.2f]", mean_pdi_test_2, ci_pdi_test_2[1], ci_pdi_test_2[2]),
      mean_pdi = mean_pdi,
      lower_pdi = ci_pdi[1],
      upper_pdi = ci_pdi[2],
      mean_pdi_0 = mean_pdi_test_0,
      lower_pdi_0 = ci_pdi_test_0[1],
      upper_pdi_0 = ci_pdi_test_0[2],
      mean_pdi_1 = mean_pdi_test_1,
      lower_pdi_1 = ci_pdi_test_1[1],
      upper_pdi_1 = ci_pdi_test_1[2],
      mean_pdi_2 = mean_pdi_test_2,
      lower_pdi_2 = ci_pdi_test_2[1],
      upper_pdi_2 = ci_pdi_test_2[2],
      stringsAsFactors = FALSE
    ))
    
    # Append to dataframe
    pdi_results <- rbind(pdi_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      pdi = sprintf("%.2f [%.2f, %.2f]", mean_pdi, ci_pdi[1], ci_pdi[2]),
      mean_pdi = mean_pdi,
      lower_pdi = ci_pdi[1],
      upper_pdi = ci_pdi[2],
      stringsAsFactors = FALSE
    ))
  }
}
print(pdi_results)

write.csv(pdi_results, "pdi_results.csv", row.names = FALSE)

write.csv(pdi_category_results, "pdi_category_results.csv", row.names = FALSE)
