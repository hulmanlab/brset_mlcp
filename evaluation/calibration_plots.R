library(CalibrationCurves) 
library(ggplot2)
library(mcca)
library(patchwork)
library(dplyr)
# dev.off()
rm(list = ls(all.names = TRUE)) 
gc()

library(optparse)

option_list <- list(
  make_option("--prob_root", default = "mBRSET_EX_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
# prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/BRSET_TL'
setwd(prob_root)

files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|ensemble|recalib", files)]
files <- sort(files)

# Create a plotting function
plot_probs <- function(df, prob_col, invert_labels, title_text, subtitle_text, x_text) {
  pd_lx <- if (ncol(df) > 4) 'none' else c(0.15, 1.01)
  df_binned <- df %>%
    mutate(bin_num = .data[[prob_col]]) %>%
    mutate(bin = cut(bin_num, breaks = seq(0, 1, by = 0.01), include.lowest = TRUE)) %>%
    mutate(bin_mid = (as.numeric(sub("\\((.+),.*", "\\1", bin)) + 
                        as.numeric(sub("[^,]*,([^]]*)\\]", "\\1", bin))) / 2) %>%
    group_by(bin_mid, label = factor(label)) %>%
    summarise(count = n(), .groups = "drop") %>%
    left_join(total_per_class, by = "label") %>%  
    mutate(perc = count / class_total * 100) %>%  # class-normalized percentage
    mutate(perc = ifelse(label %in% invert_labels, -perc, perc)) %>%
    filter(!is.na(bin_mid), !is.na(perc))
  tick_length <- max(abs(df_binned$perc)) * 0.015
  label_offset <- tick_length * 2
  gap <- 0.1
  
  ggplot(df_binned, aes(x = bin_mid, y = perc, fill = label)) +
    geom_col(position = "identity", color = "white", alpha = 0.8) +
    geom_hline(yintercept = 0, color = "black", lwd=0.4) +
    scale_y_continuous(labels = abs, breaks = scales::breaks_pretty(n = 3)) +
    coord_cartesian(ylim = range(df_binned$perc, na.rm = TRUE))+
    scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
    labs(title = title_text,
          subtitle = subtitle_text,
          x = x_text,
          # y = "Percentage (%)"
          ) + 
    {
      if (ncol(df) > 4) {
        scale_fill_manual(values = c("#004D40", "#FFC107", "#D81B60"),
                          name = NULL)
      } else {
        scale_fill_manual(values = c("#004D40", "#D81B60"),
                          labels = c("Normal", "Retinopathy"),
                          name = NULL)
        
      }
    } +
    # scale_fill_manual(values = c("#004D40", "#FFC107", "#D81B60")) +
    theme_minimal() +
    theme(
          panel.grid.minor = element_blank(),
          legend.position = pd_lx,
          legend.background = element_rect(fill = "transparent"),
          legend.key = element_blank(),
          legend.text = element_text(size = 24),
          plot.margin = margin(0.001, 0.001, 0.001, 0.001, "cm"),
          axis.text = element_text(vjust = -0.5, size = 24),
          title=element_text(size = 22,face="bold"),
          )
}




for (i in seq_along(files)) {
  name <- files[i]
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
  
  if (grepl("_reproduced", name)) {
    title<-sprintf("Reproducing %s on BRSET", model_name)
  } else if (grepl("/BRSET_TL*", prob_root)){
    title<-sprintf("%s Fine-tuning %s on BRSET", mode, model_name)
  } else if (grepl("/mBRSET_EXEVAL*", prob_root)){
    title<-sprintf("External Validation of %s fine-tune %s on mBRSET", mode, model_name)
  } else {
    title<-sprintf("Transfer Learning %s fine-tune %s to mBRSET", mode, model_name)
  }
  
  df<-read.csv(name)

  # Optional: patch missing extreme values in-place
  if (ncol(df) > 3) {
    df$y_prob_0[df$y_prob_0 == 0] <- 1e-5
    df$y_prob_0[df$y_prob_0 == 1] <- 0.99999
    df$y_prob_1[df$y_prob_1 == 0] <- 1e-5
    df$y_prob_1[df$y_prob_1 == 1] <- 0.99999
    df$y_prob_2[df$y_prob_2 == 0] <- 1e-5
    df$y_prob_2[df$y_prob_2 == 1] <- 0.99999

    one_hot_label <- df[, c(1:3)]
    label <- colnames(one_hot_label)[apply(one_hot_label, 1, which.max)]
  
  } else {
    df$y_pred[df$y_pred == 0] <- 1e-5
    df$y_pred[df$y_pred == 1] <- 0.99999
    label <- df$y_test
  }
  # df$label <- label
  df$label <- as.character(label)
  
  # Get total per class
  total_per_class <- df %>%
    dplyr::count(label) %>%
    dplyr::rename(class_total = n)
  

  

  # Define input
  if (ncol(df) > 4) {
    plot_list <- list(
    list(prob_col = "y_prob_0", invert_labels = c("y_test_1", "y_test_2"), title = "Normal", subtitle = "", x = ""),
    list(prob_col = "y_prob_1", invert_labels = c("y_test_0", "y_test_2"), title = "Non-proliferative retinopathy", subtitle = "", x = ""),
    list(prob_col = "y_prob_2", invert_labels = c("y_test_0", "y_test_1"), title = "Proliferative retinopathy", subtitle = "", x = "")
  )
  } else {
    plot_list <- list(
    list(prob_col = "y_pred", invert_labels = c("0"), title = "", subtitle = "", x = "")
    )
  }
  # Generate and combine plots
  plots <- lapply(plot_list, function(p) {
    plot_probs(df, p$prob_col, p$invert_labels, p$title, p$subtitle, p$x)
  })

  combined <- wrap_plots(plots, ncol = 1) &
    theme(axis.title.y = element_blank()) +
    labs(y = "Percentage (%)")
  
  y_lab <- ggplot() +
  annotate("text", x = 0.5, y = 0.5,
           label = "Percentage (%)",
           angle = 90, size = 10, fontface = "bold") +
  theme_void()


  combined <- y_lab + combined + plot_layout(widths = c(0.06, 1))
  rm(plots)
  gc()
  #-----------------------------------------------
  
  
  # Define helper function
  get_calib_df <- function(prob, true, label, swap = FALSE) {
    out <- val.prob.ci.2(prob, true)
    # intercept <- out$Calibration$Intercept
    cl <- out$CalibrationCurves$FlexibleCalibration
    calib_df <- data.frame(x = cl$x, y = cl$y, class = label)
    return(data.frame(x = cl$x, y = cl$y, class = label))
  }
    
  # Create long-format calibration data
  if (ncol(df) > 4) {
    df_calib <- get_calib_df(df$y_prob_0, df$y_test_0, "Normal")
    df_calib <- rbind(df_calib,
                      get_calib_df(df$y_prob_1, df$y_test_1, "Non-proliferative Retinopathy"))
    df_calib <- rbind(df_calib,
                      get_calib_df(df$y_prob_2, df$y_test_2, "Proliferative Retinopathy"))
    df_calib <- rbind(df_calib,
                      data.frame(x = c(0, 1), y = c(0, 1), class = "Ideal")) %>% 
                      mutate(class = factor(class, levels = c("Normal", "Non-proliferative Retinopathy", "Proliferative Retinopathy", "Ideal")))

    color_map <- setNames(
      c("#004D40", "#FFC107", "#D81B60", "#808080"),
      c("Normal", "Non-proliferative Retinopathy", "Proliferative Retinopathy", "Ideal")
    )
  } else {
    df_calib <- bind_rows(
      get_calib_df(df$y_pred, df$y_test, "Retinopathy"),
      data.frame(x = c(0, 1), y = c(0, 1), class = "Ideal")
    ) %>%
      mutate(class = factor(class, levels = c("Retinopathy", "Ideal")))
    color_map <- setNames(
      c("#D81B60", "#808080"),
      c("Retinopathy", "Ideal")
    )
  }
  
  cp_lx <- if (ncol(df) > 4) 0.3 else 0.16
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
      plot.margin = margin(0.001, 0.001, 0.001, 0.001, "cm"),
      axis.text = element_text(size = 24),
      axis.title.y = element_text(size = 30, face = "bold"),
      title = element_text(size = 24, face = "bold"),
      legend.position = c(cp_lx, 0.93),
      legend.background = element_rect(fill = "transparent"),
      legend.key = element_blank(),
      legend.text = element_text(size = 24)
    )
  
  
  full <- (cp |combined) + 
    plot_annotation(
                    theme = theme(plot.title = element_text(size = 17, face = "bold"))
                    )
  rm(cp, combined)
  gc()
  name <- strsplit(name, "\\.")[[1]][1]
  dir.create(
    file.path(prob_root, "calibration_plots"),
    recursive = TRUE,
    showWarnings = FALSE
  )
  ggsave(sprintf(file.path(prob_root, "calibration_plots", "%s_%s.png"), model_name, mode), full, width = 20, height = 7.7, dpi = 300)
  rm(df, df_calib, full, label)
  graphics.off()
  gc(FALSE)

}
gc()