library(CalibrationCurves) 
library(ggplot2)
library(mcca)
library(patchwork)
library(dplyr)
# dev.off()
rm(list = ls(all.names = TRUE)) 
gc()

# /3class_prob/BRSET_TL
# ^y.*\\.csv

prob_root<-"/home/livieymli/brset_analysis/BRSET/output_predicted_probabilities/mBRSET_EXEVAL"
setwd(prob_root)
files <- list.files(prob_root)
files <- sort(files)

metrics_results <- data.frame(
  model = character(),
  mode = character(),
  intercept = character(),
  mean_intercept = numeric(),
  lower_intercept = numeric(),
  upper_intercept = numeric(),
  slope = character(),
  mean_slope = numeric(),
  lower_slope = numeric(),
  upper_slope = numeric()
)

for (i in seq_along(files)) {
  name <- files[i]
  if (grepl("^y.*\\.csv$", name) && !grepl("pdi", name) && !grepl("reproduced", name) && !grepl("ensemble", name)) {
    
    print(name)
    '
    # name<-"y_visionfm_fine_tune_3class_mBRSET_TL" 
    # mode<-"Full" # Head, Full
    # model_name<-"VisionFM" # ConvNeXtv2, ResNet200d, RETFound, Dinov2, VisionFM
    '
    if (grepl("_fine_tune_", name)) {
      mode<-"Full" # Head, Full
    } else {
      mode<-"Head"
    }
    if (grepl("_convnextv2_large_", name)) {
      model_name<-"ConvNeXtv2"
    } else if (grepl("_dinov2_", name)) {
      model_name<-"DINOv2"
    } else if (grepl("_resnet200d_", name)) {
      model_name<-"ResNet200d"
    } else if (grepl("_retfound_", name)) {
      model_name<-"RETFound"
    } else {
      model_name<-"VisionFM"
    }
    
    if (grepl("_reproduced", name)) {
      title<-sprintf("Reproducing %s on BRSET", model_name)
    } else if (grepl("/BRSET_TL", prob_root)){
      title<-sprintf("%s Fine-tuning %s on BRSET", mode, model_name)
    } else if (grepl("/mBRSET_EXEVAL", prob_root)){
      title<-sprintf("External Validation of %s Fine-tuned %s on mBRSET", mode, model_name)
    } else {
      title<-sprintf("Transfer Learning %s Fine-tuned %s to mBRSET", mode, model_name)
    }
    
    df<-read.csv(name)
    describe(df)
    
    # Optional: patch missing extreme values in-place
    df$y_prob_0[df$y_prob_0 == 0] <- 1e-5
    df$y_prob_1[df$y_prob_1 == 0] <- 1e-5
    df$y_prob_2[df$y_prob_2 == 1] <- 0.99999
    
    ### get label
    one_hot_label <- df[, c(1:3)]
    label <- colnames(one_hot_label)[apply(one_hot_label, 1, which.max)]
    df$label <- label
    
    
    # Define helper function
    
    get_calib_metrics_df <- function(prob, true, label, model_name, mode, swap = FALSE) {
      out <- val.prob.ci.2(prob, true)
      intercept <- out$Calibration$Intercept
      slope     <- out$Calibration$Slope
      
      data.frame(
        model = model_name,
        mode = mode,
        label = label,
        intercept = sprintf("%.2f [%.2f - %.2f]", intercept[1], intercept[2], intercept[3]),
        mean_intercept = intercept[1],
        lower_intercept = intercept[2],
        upper_intercept = intercept[3],
        slope = sprintf("%.2f [%.2f - %.2f]", slope[1], slope[2], slope[3]),
        mean_slope = slope[1],
        lower_slope = slope[2],
        upper_slope = slope[3],
        stringsAsFactors = FALSE
      )
    }
    
    # Collect all results at once
    metrics_results_pertest <- rbind(
      get_calib_metrics_df(df$y_prob_0, df$y_test_0, "Normal", model_name, mode),
      get_calib_metrics_df(df$y_prob_1, df$y_test_1, "Non-proliferative Retinopathy", model_name, mode),
      get_calib_metrics_df(df$y_prob_2, df$y_test_2, "Proliferative Retinopathy", model_name, mode)
    )
    

  metrics_results <- rbind(metrics_results, metrics_results_pertest)  
  }
}
write.csv(metrics_results, "calibration_intercept&slope_results.csv", row.names = FALSE)
