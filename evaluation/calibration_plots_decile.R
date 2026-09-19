library(PredictABEL)
library(ggplot2)
library(dplyr)
library(tidyr)
library(optparse)

option_list <- list(
  make_option("--prob_root", default = "mBRSET_EX_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
# prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/BRSET_TL_b'
setwd(prob_root)

files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|ensemble|recalibrate", files)]
files <- sort(files)

for (i in seq_along(files)) {
  name <- files[i]
  # print(name)

  mode <- if (grepl("_fine_tune_", name)) "Full Fine-tune" else "Head Fine-tune"

  model_name <- dplyr::case_when(
    grepl("retfound_d2_s", name) ~ "RETFound DINOv2 Shanghai",
    grepl("retfound_d2_m", name) ~ "RETFound DINOv2 MEH",
    grepl("retfound", name)     ~ "RETFound",
    grepl("dinov3_large", name) ~ "DINOv3",
    grepl("visionfm", name)     ~ "VisionFM",
    grepl("dinov2", name)       ~ "DINOv2",
    grepl("eyeclip", name)      ~ "EyeCLIP",
    grepl("convnext", name)     ~ "ConvNeXt",
    grepl("resnet200d", name)   ~ "ResNet200d",
    grepl("resnet50", name)     ~ "ResNet50",
    TRUE                        ~ "Unknown Model"
  )
  df<-read.csv(name)

  df$y_pred[df$y_pred == 0] <- 1e-5
  df$y_pred[df$y_pred == 1] <- 0.99999
  pt <- plotCalibration(df, 1, df$y_pred, group = 10)
  print(pt)
  tab <- as.data.frame(pt$Table_HLtest)
  tab$group <- rownames(tab)
  tab$decile <- 1:nrow(tab)
  # Extract lower & upper bounds from interval labels
  bounds <- gsub("\\[|\\)|\\]", "", rownames(tab))
  bounds_split <- strsplit(bounds, ",")

  tab$lower <- as.numeric(sapply(bounds_split, `[`, 1))
  tab$upper <- as.numeric(sapply(bounds_split, `[`, 2))

  tab_long <- tab %>%
    select(decile, meanpred, meanobs, lower, upper) %>%
    pivot_longer(
      cols = c(meanpred, meanobs),
      names_to = "type",
      values_to = "risk"
    )

  # Force predicted first
  tab_long$type <- factor(
    tab_long$type,
    levels = c("meanpred", "meanobs")
  )

  p <- ggplot(tab_long,
              aes(x = factor(decile),
                  y = risk * 100,
                  fill = type)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    # Add range only for predicted bars
    geom_errorbar(
      data = tab_long,
      aes(x = as.numeric(factor(decile)) - 0.175,
          ymin = lower * 100,
          ymax = upper * 100,
          group = "meanpred"),
      width = 0.2,
      position = position_dodge(width = 0.7),
      inherit.aes = FALSE
    ) +
    scale_fill_manual(values = c(meanpred = "grey70", meanobs  = "black"),
                      labels = c(meanpred = "Predicted", meanobs  = "Observed")) +
    labs(x = "Deciles of predicted risk",
      y = "Observed risk (%)",
      fill = ""
    ) +
    theme_classic(base_size = 26)

  p

  dir.create(
    file.path(prob_root, "calibration_plots_decile"),
    recursive = TRUE,
    showWarnings = FALSE
  )
  # ggsave(sprintf(file.path(prob_root, "calibration_plots_decile", "%s_%s.png"), model_name, mode), p, width = 11, height = 7.7, dpi = 300)
  rm(df, pt)
#   graphics.off()
  gc(FALSE)
  
}