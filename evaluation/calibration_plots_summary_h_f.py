#%%# Assemble calibration plots into a single figure
import argparse
import gc
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
gc.collect()

#%%
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

# path = "mBRSET_EX_b"

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plot_dir = os.path.join(DATASET, "output/predicted_probabilities", path, 'calibration_plots')

# arrange by model (rows) and mode (columns)
modes = ["Calibration Curve", "Head Fine-tune", "Full Fine-tune"]
curves = [f for f in os.listdir(plot_dir) if f.endswith(".png") and 'ConvNeXt' not in f and 'ResNet' not in f and '_head&full' in f]
plot_files = [f for f in os.listdir(plot_dir) if f.endswith(".png") and 'ConvNeXt' not in f and 'ResNet' not in f and '_head&full' not in f]

#%%
# # discover model names by stripping the mode suffix from filenames
models_set = set()
for fn in plot_files:
    for m in modes:
        token = f"_{m}.png"
        if token in fn:
            models_set.add(fn.replace(token, ""))
            break
models = sorted(models_set)  # adjust ordering if you want a specific order
models = ['DINOv3', 'RETFound', 'VisionFM']
# print(models)
nrows = max(1, len(models))
ncols = len(modes)

sample_img = Image.open(os.path.join(plot_dir, curves[0]))
img_w, img_h = sample_img.size
sample_img.close()

left_margin = 320    # space for model names
top_margin = 270     # space for mode titles
pad = 100
bottom_margin = 240

canvas_w = left_margin + img_w * ncols
canvas_h = top_margin + img_h * nrows + bottom_margin

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 150)
    font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", 130)
except IOError:
    font_title = font_label = ImageFont.load_default()

for col, mode in enumerate(modes):
    if mode == "Calibration Curve":
        mode = ""
    x = left_margin + col * img_w + img_w // 2
    y = top_margin // 2 + 100
    draw.text((x, y), mode, fill="black", anchor="mm", font=font_label)

for row, model in enumerate(models):
    y = top_margin + row * img_h + 50

    bbox = draw.textbbox((0, 0), model, font=font_label)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    PAD_TXT = 20
    txt = Image.new(
        "RGBA",
        (text_w + 2 * PAD_TXT, text_h + 2 * PAD_TXT),
        (255, 255, 255, 0)
    )

    txt_draw = ImageDraw.Draw(txt)
    txt_draw.text((PAD_TXT, PAD_TXT), model, fill="black", font=font_label)

    txt = txt.rotate(90, expand=True)

    canvas.paste(
        txt,
        (
            left_margin - pad - txt.size[0],
            y + img_h // 2 - txt.size[1] // 2
        ),
        txt
    )
    for col, mode in enumerate(modes):
        fname = os.path.join(plot_dir, f"{model}{' Large' if model == 'DINOv3' else ''}_{'head&full' if mode == 'Calibration Curve' else mode}.png")
        x = left_margin + col * img_w

        if os.path.exists(fname):
            img = Image.open(fname)
            if img.size != (img_w, img_h):
                cropped = img.crop((img.size[0] - img_w + 170, 0, img.size[0], img_h - 140))
                img = Image.new("RGB", (img_w, img_h), "white")
                img.paste(cropped, (0, 90))
            canvas.paste(img, (x, y))
            img.close()

top_label = "Calibration curves                                     Predicted probabilities distribution     "
x = canvas_w // 2
y = 100
draw.text((x, y),
            top_label,
            fill="black",
            anchor="mm",
            font=font_title)


bottom_label = "Predicted probabilities"

for col in range(ncols):
    x = left_margin + col * img_w + img_w // 2
    y = top_margin + img_h * nrows + bottom_margin // 2
    draw.text((x, y),
              bottom_label,
              fill="black",
              anchor="mm",
              font=font_label)
    
canvas.save(os.path.join(os.path.dirname(plot_dir), "summary", "Calibration_Comparison_3c.png"), dpi=(300, 300))

# reduce margins between subplots

gc.collect()
# %%
