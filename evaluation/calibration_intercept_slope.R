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
prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
# prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/mBRSET_TL'
setwd(prob_root)
files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|reproduced|ensemble", files)]
files <- sort(files)

get_calib_metrics_df <- function(prob, true, label, model_name, mode) {
  out <- val.prob.ci.2(prob, true)
  intercept <- out$Calibration$Intercept
  slope     <- out$Calibration$Slope

  data.frame(
    model = model_name,
    mode = mode,
    label = label,
    intercept = sprintf("%.2f [%.2f, %.2f]", intercept[1], intercept[2], intercept[3]),
    slope     = sprintf("%.2f [%.2f, %.2f]", slope[1], slope[2], slope[3]),
    stringsAsFactors = FALSE
  )
}

results_list <- vector("list", length(files))
res_i <- 1

# metrics_results <- data.frame(
#   model = character(),
#   mode = character(),
#   intercept = character(),
#   mean_intercept = numeric(),
#   lower_intercept = numeric(),
#   upper_intercept = numeric(),
#   slope = character(),
#   mean_slope = numeric(),
#   lower_slope = numeric(),
#   upper_slope = numeric()
# )

for (i in seq_along(files)) {
  name <- files[i]
  # if (grepl("^y.*\\.csv$", name) && !grepl("pdi", name) && !grepl("reproduced", name) && !grepl("ensemble", name)) {
    
  # print(name)
  mode <- if (grepl("_fine_tune_", name)) "Full Fine-tune" else "Head Fine-tune"

  model_name <- dplyr::case_when(
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
  
  # Optional: patch missing extreme values in-place
  if (ncol(df) > 3) {
    df$y_prob_0[df$y_prob_0 == 0] <- 1e-5
    df$y_prob_1[df$y_prob_1 == 0] <- 1e-5
    df$y_prob_2[df$y_prob_2 == 1] <- 0.99999

    results_list[[res_i]] <- rbind(
      get_calib_metrics_df(df$y_prob_0, df$y_test_0, "Normal", model_name, mode),
      get_calib_metrics_df(df$y_prob_1, df$y_test_1, "Non-proliferative Retinopathy", model_name, mode),
      get_calib_metrics_df(df$y_prob_2, df$y_test_2, "Proliferative Retinopathy", model_name, mode)
    )
  
  } else {
    df$y_pred[df$y_pred == 0] <- 1e-5
    df$y_pred[df$y_pred == 1] <- 0.99999

    results_list[[res_i]] <-
      get_calib_metrics_df(df$y_pred, df$y_test, "Retinopathy", model_name, mode)
  }
  res_i <- res_i + 1
  rm(df)
  gc(FALSE)
}
  
metrics_results <- do.call(rbind, results_list)
write.csv(metrics_results, file.path(prob_root, 'summary', 'calibration_intercept&slope_results.csv'), row.names = FALSE)
gc()