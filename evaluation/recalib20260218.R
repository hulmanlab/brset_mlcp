
############################################################
# Packages
############################################################

library(optparse)
library(CalibrationCurves)
library(ggplot2)
library(dplyr)
library(tidyr)
library(dcurves)
library(PredictABEL)

option_list <- list(
  make_option("--prob_root", default = "mBRSET_EX_b")
)

opts <- parse_args(OptionParser(option_list = option_list))
path <- opts$prob_root
# prob_root <- file.path(getwd(), "output", "predicted_probabilities", path)
prob_root <- '/home/livieymli/brset_analysis/BRSET/output/predicted_probabilities/mBRSET_TL_b'
setwd(prob_root)
scale <- if (grepl("mBRSET", prob_root, ignore.case = TRUE)) c(-0.05, 0.25) else c(-0.2, 0.1)
model <- "retfound"
mode <- "fine_tune"
files <- list.files(
  prob_root,
  pattern = "^y.*\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("pdi|ensemble|recalib|d2", files)]
files <- sort(files)

name <- files[
  grepl(model, files, ignore.case = TRUE) &
    grepl(mode, files, ignore.case = TRUE)
]
if (length(name) == 0L) {
  stop(sprintf("No matching %s %s file found in %s", model, mode, prob_root))
}
name <- sub("\\.csv$", "", basename(name[1]))
print(name)
test_data <- read.csv(file = file.path(prob_root, paste0(name, ".csv")),
                      header = TRUE, sep = ",")


# # Define helper function
get_calib_df <- function(prob, true, label, swap = FALSE) {
  out <- val.prob.ci.2(prob, true)
  cl <- out$CalibrationCurves$FlexibleCalibration
  return(data.frame(x = cl$x, y = cl$y, class = label))
}

# recalib - full code
# -------------------------------
# Full Recalibration Pipeline with Isotonic Regression + GLM-based methods
# -------------------------------

# 1️⃣ Functions
logit <- function(p) log(p / (1 - p))
inv_logit <- function(x) 1 / (1 + exp(-x))
clip_probs <- function(p, eps = 1e-6) pmin(pmax(p, eps), 1 - eps)

# -------------------------------
# 2️⃣ Split test_data into calibration (1/3) and evaluation (2/3)
# -------------------------------
set.seed(42)

# Split indices by outcome
idx_0 <- which(test_data$y_test == 0)
idx_1 <- which(test_data$y_test == 1)

# Sample 1/3 from each class
cal_0 <- sample(idx_0, size = floor(length(idx_0) / 3))
cal_1 <- sample(idx_1, size = floor(length(idx_1) / 3))

# Combine calibration indices
cal_idx <- c(cal_0, cal_1)

# Create datasets
cal_data  <- test_data[cal_idx, ]
eval_data <- test_data[-cal_idx, ]
# Avoid 0 or 1 probabilities for logit transformation
eps <- 1e-5
# Extract predictions and true labels
cal_pred <- clip_probs(cal_data$y_pred)
cal_pred <- pmin(pmax(cal_pred, eps), 1 - eps)
cal_y    <- cal_data$y_test

eval_pred <- clip_probs(eval_data$y_pred)
eval_pred <- pmin(pmax(eval_pred, eps), 1 - eps)
eval_y    <- eval_data$y_test
eval_ids  <- eval_data$image_ids

# -------------------------------
# 3️⃣ Convert to logits for GLM methods
# -------------------------------
cal_lp  <- logit(cal_pred)
eval_lp <- logit(eval_pred)

# -------------------------------
# 4️⃣ Platt scaling (slope + intercept)
# -------------------------------
platt_model <- glm(cal_y ~ cal_lp, family = binomial)
eval_pred_platt <- predict(platt_model,
                           newdata = data.frame(cal_lp = eval_lp),
                           type = "response")
eval_pred_platt <- clip_probs(eval_pred_platt)

# -------------------------------
# 5️⃣ Intercept-only recalibration (beta = 1)
# -------------------------------
intercept_model <- glm(cal_y ~ 1, offset = cal_lp, family = binomial)
alpha <- coef(intercept_model)[["(Intercept)"]]
eval_pred_intercept <- inv_logit(eval_lp + alpha)
eval_pred_intercept <- clip_probs(eval_pred_intercept)

# -------------------------------
# 6️⃣ Temperature scaling (slope-only) via glm + offset
# -------------------------------
temp_model <- glm(cal_y ~ 0 + cal_lp, offset = rep(0, length(cal_lp)), family = binomial)
beta <- coef(temp_model)[["cal_lp"]]
T_hat <- 1 / beta
print(paste("Estimated temperature from GLM:", T_hat))

# Apply to evaluation set
eval_pred_temp <- inv_logit(beta * eval_lp)
eval_pred_temp <- clip_probs(eval_pred_temp)

# # -------------------------------
# # 7️⃣ Isotonic regression
# # -------------------------------

# # Fit isotonic regression on calibration set
# ord <- order(cal_pred)
# iso_fit <- stats::isoreg(cal_pred[ord], cal_y[ord])

# # Apply isotonic regression to evaluation set using safe interpolation
# eval_pred_iso <- approx(
#   x = iso_fit$x,
#   y = iso_fit$yf,
#   xout = eval_pred,
#   rule = 2
# )$y
# eval_pred_iso <- clip_probs(eval_pred_iso)

# -------------------------------
# 8️⃣ Save recalibrated predictions in eval_data
# -------------------------------
eval_data$y_pred_platt      <- eval_pred_platt
eval_data$y_pred_intercept  <- eval_pred_intercept
eval_data$y_pred_temp       <- eval_pred_temp
# eval_data$y_pred_iso        <- eval_pred_iso

# -------------------------------
# 9️⃣ Evaluation / reliability plots
# -------------------------------
# Assuming val.prob.ci.2 is loaded
out_platt <- val.prob.ci.2(eval_pred_platt, eval_y, main = "Platt Scaling", smooth = 'rcs')
int_platt <- out_platt$Calibration$Intercept
slope_platt <- out_platt$Calibration$Slope
print(paste("Platt Scaling - Intercept:", sprintf("%.2f [%.2f, %.2f]", int_platt[1], int_platt[2], int_platt[3])))
print(paste("Platt Scaling - Slope:", sprintf("%.2f [%.2f, %.2f]", slope_platt[1], slope_platt[2], slope_platt[3])))

# val.prob.ci.2(eval_pred_intercept, eval_y, main = "Intercept-only")
out_temp <- val.prob.ci.2(eval_pred_temp, eval_y, main = "Temperature Scaling")
int_temp <- out_temp$Calibration$Intercept
slope_temp <- out_temp$Calibration$Slope
print(paste("Temperature Scaling - Intercept:", sprintf("%.2f [%.2f, %.2f]", int_temp[1], int_temp[2], int_temp[3])))
print(paste("Temperature Scaling - Slope:", sprintf("%.2f [%.2f, %.2f]", slope_temp[1], slope_temp[2], slope_temp[3])))

# # val.prob.ci.2(eval_pred_iso, eval_y, main = "Isotonic Regression")

write.csv(data.frame(y_test = eval_y, y_pred_platt = eval_pred_platt, image_ids = eval_ids),
      file = file.path(prob_root, paste0(name, "_recalibrated_0.csv")),
      row.names = FALSE)

# # Create long-format calibration data
df_calib <- bind_rows(
  get_calib_df(eval_pred_platt, eval_y, "Platt Scaling"),
  get_calib_df(eval_pred_intercept, eval_y, "Intercept-only"),
  get_calib_df(eval_pred_temp, eval_y, "Temperature Scaling"),
  data.frame(x = c(0, 1), y = c(0, 1), class = "Ideal")
) %>%
  mutate(class = factor(class,
                        levels = c("Platt Scaling",
                                    "Intercept-only",
                                    "Temperature Scaling",
                                    "Ideal")))
color_map <- setNames(
  c("#004D40", "#9C812E", "#286DA9", "#808080"),
  c("Platt Scaling", "Intercept-only", "Temperature Scaling", "Ideal")
)

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
      legend.position = c(0.76, 0.15),
      legend.background = element_rect(fill = "transparent"),
      legend.key = element_blank(),
      legend.text = element_text(size = 24)
    )



#### Decile plot with confidence intervals for Platt scaling only (for now)
# pt <- plotCalibration(eval_data, 1, eval_data$y_pred_platt, group = 10)
# tab <- as.data.frame(pt$Table_HLtest)
# tab$group <- rownames(tab)
# tab$decile <- 1:nrow(tab)
# # Extract lower & upper bounds from interval labels
# bounds <- gsub("\\[|\\)|\\]", "", rownames(tab))
# bounds_split <- strsplit(bounds, ",")

# tab$lower <- as.numeric(sapply(bounds_split, `[`, 1))
# tab$upper <- as.numeric(sapply(bounds_split, `[`, 2))

# tab_long <- tab %>%
#   select(decile, meanpred, meanobs, lower, upper) %>%
#   pivot_longer(
#     cols = c(meanpred, meanobs),
#     names_to = "type",
#     values_to = "risk"
#   )

# # Force predicted first
# tab_long$type <- factor(
#   tab_long$type,
#   levels = c("meanpred", "meanobs")
# )

# p <- ggplot(tab_long,
#             aes(x = factor(decile),
#                 y = risk * 100,
#                 fill = type)) +
#   geom_col(position = position_dodge(width = 0.7), width = 0.6) +
#   # Add range only for predicted bars
#   geom_errorbar(
#     data = tab_long,
#     aes(x = as.numeric(factor(decile)) - 0.175,
#         ymin = lower * 100,
#         ymax = upper * 100,
#         group = "meanpred"),
#     width = 0.2,
#     position = position_dodge(width = 0.7),
#     inherit.aes = FALSE
#   ) +
#   scale_fill_manual(values = c(meanpred = "grey70", meanobs  = "black"),
#                     labels = c(meanpred = "Predicted", meanobs  = "Observed")) +
#   labs(x = "Deciles of predicted risk",
#     y = "Observed risk (%)",
#     fill = ""
#   ) +
#   theme_classic(base_size = 26)

# p




# dca_plot <- dca(eval_y ~ eval_pred + eval_pred_platt,
#     data = eval_data,
#     thresholds = seq(0, 0.5, 0.05),
#     label = list(
#       eval_pred = "Before recalibration",
#       eval_pred_platt = "After recalibration")
#     ) %>%
#     plot(smooth = FALSE) + 
#     coord_cartesian(ylim = scale)


# calibration_dir <- file.path(prob_root, "calibration_plots_recalibrated")
# if (!dir.exists(calibration_dir)) {
#   dir.create(calibration_dir, recursive = TRUE)
# }

# ggsave(sprintf(file.path(calibration_dir, "%s_%s_recalibrated_dca.png"), model, mode), dca_plot, width = 5, height = 3, dpi = 300)

# ggsave(sprintf(file.path(calibration_dir, "%s_%s_recalibrated.png"),
#                 model, mode), cp, width = 11, height = 7.7, dpi = 300)
# ggsave(sprintf(file.path(calibration_dir, "DINOv3_Full fine-tuned_recalibrated_decile.png")), p, width = 11, height = 7.7, dpi = 300)
