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

for (i in seq_along(files)) {
  name <- files[i]
  if (grepl("perfect_calibrated.csv$", name) && !grepl("pdi", name) && !grepl("ensemble", name)) {
    
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
    
    # Get total per class
    total_per_class <- df %>%
      dplyr::count(label) %>%
      dplyr::rename(class_total = n)
    
    # Create a plotting function
    plot_probs <- function(df, prob_col, invert_labels, title_text, subtitle_text, x_text) {
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
        scale_y_continuous(labels = abs) +
        coord_cartesian(ylim = range(df_binned$perc, na.rm = TRUE))+
        scale_x_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1)) +
        labs(title = title_text,
             subtitle = subtitle_text,
             x = x_text,
             y = "Percentage (%)") +
        scale_fill_manual(values = c("#004D40", "#FFC107", "#D81B60")) +
        theme_minimal() +
        theme(
              panel.grid.minor = element_blank(),
              legend.position = "none",
              plot.margin = margin(0.001, 0.001, 0.001, 0.001, "cm"),
              axis.text = element_text(vjust = -0.5, size=15),
              title=element_text(size=17,face="bold"),
              )
    }
    

    # Define input
    plot_list <- list(
      list(prob_col = "y_prob_0", invert_labels = c("y_test_1", "y_test_2"), title = "Distribution of predicted probabilities", subtitle = "Normal", x = ""),
      list(prob_col = "y_prob_1", invert_labels = c("y_test_0", "y_test_2"), title = "", subtitle = "Non-proliferative retinopathy", x = ""),
      list(prob_col = "y_prob_2", invert_labels = c("y_test_0", "y_test_1"), title = "", subtitle = "Proliferative retinopathy", x = "Predicted probability")
    )
    
    # Generate and combine plots
    plots <- lapply(plot_list, function(p) {
      plot_probs(df, p$prob_col, p$invert_labels, p$title, p$subtitle, p$x)
    })
    
    combined <- wrap_plots(plots, ncol = 1)
    combined
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
    df_calib <- bind_rows(
      get_calib_df(df$y_prob_0, df$y_test_0, "Normal"),
      get_calib_df(df$y_prob_1, df$y_test_1, "Non-proliferative Retinopathy"),
      get_calib_df(df$y_prob_2, df$y_test_2, "Proliferative Retinopathy"),
      data.frame(x = c(0, 1), y = c(0, 1), class = "Ideal")
    ) %>%
      mutate(class = factor(class, levels = c("Normal", "Non-proliferative Retinopathy", "Proliferative Retinopathy", "Ideal")))

    color_map <- setNames(
      c("#004D40", "#FFC107", "#D81B60", "#808080"),
      c("Normal", "Non-proliferative Retinopathy", "Proliferative Retinopathy", "Ideal")
    )
    
    # Plot
    cp <- ggplot(df_calib, aes(x = x, y = y, color = class)) +
      geom_line(linewidth = 1.5) +
      scale_color_manual(values = color_map) +
      labs(
        title = "Calibration curve",
        x = "Predicted probability",
        y = "Observed proportion",
        color = NULL
      ) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      theme_minimal() +
      theme(
        plot.margin = margin(0.001, 0.001, 0.001, 0.001, "cm"),
        axis.text = element_text(size=15),
        title=element_text(size=17,face="bold"),
        legend.position = c(0.2, 0.93),
        legend.background = element_rect(fill = "transparent"),
        legend.key = element_blank(),
        legend.text = element_text(size = 17)
      )
    
    cp
    
    full <- (cp |combined) + 
      plot_annotation(
                      theme = theme(plot.title = element_text(size = 17, face = "bold"))
                      )
    full
    name <- strsplit(name, "\\.")[[1]][1]
    ggsave(sprintf("%s.png", name), full, width = 20, height = 7.7, dpi = 300)
    
    
    #Transfer learning down-stream network fine-tuned 
    #External Validation of Down-stream Network Fine-tuned VisionFM on mBRSET
    
#     ###-------------PDI-------------------
#     
    # data <- df[, c(5,4,6)]
    # pdi(y = label, d = data,method = "prob")
  }
}
