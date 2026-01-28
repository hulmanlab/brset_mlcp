# Assemble calibration plots into a single figure
import argparse
import gc
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_root", default="BRSET_TL_b", help="Path to predicted probabilities directory")
path = parser.parse_args().prob_root

DATASET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plot_dir = os.path.join(DATASET, "output/predicted_probabilities", path, 'calibration_plots')

# arrange by model (rows) and mode (columns)
modes = ["Head Fine-tune", "Full Fine-tune"]
plot_files = [f for f in os.listdir(plot_dir) if f.endswith(".png") and 'ConvNeXt' not in f and 'ResNet' not in f]

# discover model names by stripping the mode suffix from filenames
models_set = set()
for fn in plot_files:
    for m in modes:
        token = f"_{m}.png"
        if token in fn:
            models_set.add(fn.replace(token, ""))
            break
models = sorted(models_set)  # adjust ordering if you want a specific order
# print(models)
nrows = max(1, len(models))
ncols = len(modes)

sample_img = Image.open(os.path.join(plot_dir, plot_files[0]))
img_w, img_h = sample_img.size
sample_img.close()

left_margin = 320    # space for model names
top_margin = 120     # space for mode titles
pad = 100

canvas_w = left_margin + img_w * ncols
canvas_h = top_margin + img_h * nrows

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
    font_label = ImageFont.truetype("DejaVuSans.ttf", 100)
except IOError:
    font_title = font_label = ImageFont.load_default()

for col, mode in enumerate(modes):
    x = left_margin + col * img_w + img_w // 2
    y = top_margin // 2
    draw.text((x, y), mode, fill="black", anchor="mm", font=font_title)

for row, model in enumerate(models):
    y = top_margin + row * img_h

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
        fname = os.path.join(plot_dir, f"{model}_{mode}.png")
        x = left_margin + col * img_w

        if os.path.exists(fname):
            img = Image.open(fname)
            canvas.paste(img, (x, y))
            img.close()

canvas.save(os.path.join(os.path.dirname(plot_dir), "summary", "Calibration_Comparison.png"), dpi=(300, 300))

# reduce margins between subplots

gc.collect()
# %%
