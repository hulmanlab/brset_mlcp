library(CalibrationCurves)
library(ggplot2)
library(mcca)
library(patchwork)
library(dplyr)

rm(list = ls(all.names = TRUE))
gc()

library(optparse)

option_list <- list(
  make_option("--prob_root", default = "mBRSET_EX_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
# prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/mBRSET_EX_b'
setwd(prob_root)

files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|ensemble|_fine_tune_|recalib", files)]
files <- sort(files)

print(files)
for (i in seq_along(files)) {
  name1 <- files[i]
  name2 <- gsub("_eval_", "_fine_tune_", name1)
  mode1 <- "Head Fine-tune"
  mode2 <- "Full Fine-tune"
  model_name <- dplyr::case_when(
    grepl("retfound_d2_s", name1) ~ "RETFound DINOv2 Shanghai",
    grepl("retfound_d2_m", name1) ~ "RETFound DINOv2 MEH",
    grepl("retfound", name1)     ~ "RETFound",
    grepl("dinov3_large", name1) ~ "DINOv3 Large",
    grepl("visionfm", name1)     ~ "VisionFM",
    grepl("dinov2", name1)       ~ "DINOv2 Large",
    grepl("eyeclip", name1)      ~ "EyeCLIP",
    grepl("convnext", name1)     ~ "ConvNeXt",
    grepl("resnet200d", name1)   ~ "ResNet200d",
    grepl("resnet50", name1)     ~ "ResNet50",
    TRUE                        ~ "Unknown Model"
  )

  df1 <- read.csv(name1)
  df2 <- read.csv(name2)

  # Optional: patch missing extreme values in-place

  df1$y_pred[df1$y_pred == 0] <- 1e-5
  df1$y_pred[df1$y_pred == 1] <- 0.99999
  df2$y_pred[df2$y_pred == 0] <- 1e-5
  df2$y_pred[df2$y_pred == 1] <- 0.99999
  label <- df1$y_test


  df1$label <- as.character(label)
  df2$label <- as.character(label)

  # # Define helper function
  get_calib_df <- function(prob, true, label, swap = FALSE) {
    out <- val.prob.ci.2(prob, true)
    cl <- out$CalibrationCurves$FlexibleCalibration
    calib_df <- data.frame(x = cl$x, y = cl$y, class = label)
    return(data.frame(x = cl$x, y = cl$y, class = label))
  }

  # Create long-format calibration data
  df_calib <- bind_rows(
    get_calib_df(df1$y_pred, df1$y_test, "Head fine-tune"),
    get_calib_df(df2$y_pred, df2$y_test, "Full fine-tune"),
    data.frame(x = c(0, 1), y = c(0, 1), class = "Ideal")
  ) %>%
    mutate(class = factor(class,
                          levels = c("Head fine-tune",
                                     "Full fine-tune",
                                     "Ideal")))
  color_map <- setNames(
    c("#E66100", "#5D3A9B", "#808080"),
    c("Head fine-tune", "Full fine-tune", "Ideal")
  )

  # Plot
  cp <- ggplot(df_calib, aes(x = x, y = y, color = class)) +
    geom_line(linewidth = 1.5) +
    scale_color_manual(values = color_map) +
    labs(
      # title = "Calibration curve",
      x = "",
      y = "Observed proportion",
      color = NULL
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_minimal() +
    theme(
      plot.margin = margin(0.5, 0.05, -0.5, 0.001, "cm"),
      axis.text = element_text(size = 24),
      axis.title.y = element_text(size = 30, face = "bold"),
      title = element_text(size = 24, face = "bold"),
      legend.position = c(0.2, 0.91),
      legend.background = element_rect(fill = "transparent"),
      legend.key = element_blank(),
      legend.text = element_text(size = 24)
    )

  name <- strsplit(name1, "\\.")[[1]][1]
  dir.create(
    file.path(prob_root, "calibration_plots"),
    recursive = TRUE,
    showWarnings = FALSE
  )
  ggsave(sprintf(file.path(prob_root, "calibration_plots", "%s_head&full.png"),
                 model_name, mode), cp, width = 10, height = 7.7, dpi = 300)
  rm(df1, df2, df_calib, label)
  graphics.off()
  gc(FALSE)

}
gc()