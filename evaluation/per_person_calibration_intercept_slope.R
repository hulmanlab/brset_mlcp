library(CalibrationCurves) 
library(ggplot2)
library(mcca)
library(patchwork)
library(dplyr)
# dev.off()
gc()

library(optparse)

option_list <- list(
  make_option("--prob_root", default = "BRSET_TL_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
# prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/BRSET_TL'
setwd(prob_root)
files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|reproduced|ensemble|recalib|retfound|dinov3|visionfm_eval_3class", files)]
files <- sort(files)

# results_list <- vector("list", length(files))
# res_i <- 1

calibration_results <- data.frame(
  model = character(),
  mode = character(),
  intercept = character(),
  slope = character(),
  intercept_0 = character(),
  slope_0 = character(),
  intercept_1 = character(),
  slope_1 = character(),
  intercept_2 = character(),
  slope_2 = character()
)

for (i in seq_along(files)) {
  name <- files[i]
  # if (grepl("^y.*\\.csv$", name) && !grepl("pdi", name) && !grepl("reproduced", name) && !grepl("ensemble", name)) {
    
  print(name)
  mode <- if (grepl("_fine_tune_", name)) "Full Fine-tune" else "Head Fine-tune"

  model <- dplyr::case_when(
    grepl("retfound_d2_s", name) ~ "RETFound DINOv2 Shanghai",
    grepl("retfound_d2_m", name) ~ "RETFound DINOv2 MEH",
    grepl("retfound", name)     ~ "RETFound",
    grepl("dinov3_large", name) ~ "DINOv3 Large",
    grepl("visionfm", name)     ~ "VisionFM",
    grepl("dinov2", name)       ~ "DINOv2 Large",
    grepl("eyeclip", name)      ~ "EyeCLIP",
    grepl("convnext", name)     ~ "ConvNeXt",
    grepl("resnet200d", name)   ~ "ResNet200d",
    grepl("resnet50", name)     ~ "ResNet50",
    TRUE                        ~ "Unknown Model"
  )
  
  df<-read.csv(name)

  # Replace boundary probabilities to avoid exact 0 and 1 values.
  probability_columns <- intersect(c("y_pred", "y_prob_0", "y_prob_1", "y_prob_2"), names(df))
  for (column in probability_columns) {
    df[[column]][df[[column]] == 0] <- 0.00001
    df[[column]][df[[column]] == 1] <- 0.99999
  }
  
  # Extract person ID from image_ids
  # e.g. 1002.3 -> 1002
  if (!"person_id" %in% names(df)) {
    df$person_id <- sub("\\..*$", "", as.character(df$image_ids))
  }
  # Optional: store as numeric
  df$person_id <- as.numeric(df$person_id)

  persons <- unique(df$person_id)
  n_persons <- length(persons)
  
  df$y_camera <- NULL

  # Bootstrap settings
  n_iterations <- 1000
  # n <- nrow(y_true)
  intercept <- numeric(n_iterations)  
  slope <- numeric(n_iterations)
  intercept_0 <- numeric(n_iterations)  
  slope_0 <- numeric(n_iterations)
  intercept_1 <- numeric(n_iterations)  
  slope_1 <- numeric(n_iterations)
  intercept_2 <- numeric(n_iterations)  
  slope_2 <- numeric(n_iterations)

  for (itr in 1:n_iterations) {

    # Per-person sample
    sampled_persons <- sample(
      persons,
      size = n_persons,
      replace = TRUE
    )

    boot_df <- do.call(
      rbind,
      lapply(sampled_persons, function(id) {
        df[df$person_id == id, ]
      })
    )
    
    # intercept and slope
    if (!"y_pred" %in% names(df)) {
      out0 <- val.prob.ci.2(boot_df$y_prob_0, boot_df$y_test_0)
      out1 <- val.prob.ci.2(boot_df$y_prob_1, boot_df$y_test_1)
      out2 <- val.prob.ci.2(boot_df$y_prob_2, boot_df$y_test_2)
      intercept_0[itr] <- out0$Calibration$Intercept[1]
      slope_0[itr] <- out0$Calibration$Slope[1]
      intercept_1[itr] <- out1$Calibration$Intercept[1]
      slope_1[itr] <- out1$Calibration$Slope[1]
      intercept_2[itr] <- out2$Calibration$Intercept[1]
      slope_2[itr] <- out2$Calibration$Slope[1]
    } else {
      out <- val.prob.ci.2(boot_df$y_pred, boot_df$y_test)
      intercept[itr] <- out$Calibration$Intercept[1]
      slope[itr] <- out$Calibration$Slope[1]
    }

  }
  if (!"y_pred" %in% names(df)){
    mean_intercept_0 <- mean(intercept_0)
    ci_intercept_0 <- quantile(intercept_0, c(0.025, 0.975))
    mean_slope_0 <- mean(slope_0)
    ci_slope_0 <- quantile(slope_0, c(0.025, 0.975))
    mean_intercept_1 <- mean(intercept_1)
    ci_intercept_1 <- quantile(intercept_1, c(0.025, 0.975))
    mean_slope_1 <- mean(slope_1)
    ci_slope_1 <- quantile(slope_1, c(0.025, 0.975))
    mean_intercept_2 <- mean(intercept_2)
    ci_intercept_2 <- quantile(intercept_2, c(0.025, 0.975))
    mean_slope_2 <- mean(slope_2)
    ci_slope_2 <- quantile(slope_2, c(0.025, 0.975))
    
    calibration_results <- rbind(calibration_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      intercept_0 = sprintf("%.2f [%.2f, %.2f]", mean_intercept_0, ci_intercept_0[1], ci_intercept_0[2]),
      slope_0 = sprintf("%.2f [%.2f, %.2f]", mean_slope_0, ci_slope_0[1], ci_slope_0[2]),
      intercept_1 = sprintf("%.2f [%.2f, %.2f]", mean_intercept_1, ci_intercept_1[1], ci_intercept_1[2]),
      slope_1 = sprintf("%.2f [%.2f, %.2f]", mean_slope_1, ci_slope_1[1], ci_slope_1[2]),
      intercept_2 = sprintf("%.2f [%.2f, %.2f]", mean_intercept_2, ci_intercept_2[1], ci_intercept_2[2]),
      slope_2 = sprintf("%.2f [%.2f, %.2f]", mean_slope_2, ci_slope_2[1], ci_slope_2[2])
    ))
  } else {
    mean_intercept <- mean(intercept)
    ci_intercept <- quantile(intercept, c(0.025, 0.975))
    mean_slope <- mean(slope)
    ci_slope <- quantile(slope, c(0.025, 0.975))

    calibration_results <- rbind(calibration_results, data.frame(
      model = paste0(model),
      mode = paste0(mode),
      intercept = sprintf("%.2f [%.2f, %.2f]", mean_intercept, ci_intercept[1], ci_intercept[2]),
      slope = sprintf("%.2f [%.2f, %.2f]", mean_slope, ci_slope[1], ci_slope[2])
    ))
    
  }
  rm(df)
  gc(FALSE)
  write.csv(calibration_results, file.path(prob_root, 'summary', 'calibration_intercept&slope_per_person.csv'), row.names = FALSE)
}

write.csv(calibration_results, file.path(prob_root, 'summary', 'calibration_intercept&slope_per_person.csv'), row.names = FALSE)
gc()