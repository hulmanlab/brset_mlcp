
---

# Comparison of Foundation Models and Transfer Learning Strategies for Multi-class Diabetic Retinopathy Classification


In this project we evaluate using deep learning models, mostly foundation models, to predicte diabetic retinopathy(DR). We experiment with different transfer learning strategies and thoroughly evaluated the results focusing on both discrimination and calibration.

There are three experiments in this study: 
* Experiment 1 : Fine-tuning pre-trained models on BRSET
* Experiment 2 : External validation using mBRSET
* Experiment 3 : Fine-tuning BRSET-trained models on mBRSET

---
## Usage

### 1. Dataset 
In this study we used two datasets BRSET, A Brazilian Multilabel Ophthalmological Dataset, and mBRSET, a Mobile Brazilian Retinal Dataset. Datasets and related files can be found in folder `data/`

* Images: `data/fundus_photos`
* **BRSET:** Images starting with `img*`
* **mBRSET:** All other images
* Other: Preprocessing files and a train-test split for the three experiments.
    * xxx_brset_nooverlap.csv are the splits used in this study

### 2. Data Preprocessing

* The preprocessing scripts can be found the `src` folder to prepare the data for training. All functions are intergrated in the training pipelines.
* Functions available: data loading, data analysis, get dataset.

### 3. Training pipelines

* Training templates: Files starting with `template*` in the parent folder.
    * Exp 1. template_3class_DR.py 
    * Exp 2. template_3class_DR_mBRSET_EXEVAL.py
    * Exp 3. template_3class_DR_mBRSET_TL.py
* Training on the GPU cluster: Use bash scripts in the `sbash` folder to run jobs on a cluster.

Great thanks to Luis Nakayama for the source code, https://github.com/luisnakayama/BRSET
Adjustments for local application has been made.   

### 4. Models and Outputs

* **Models:** Saved in `output/models/` after fine-tuning.
* **Output predicted probabilities:** Stored in `output/predicted_probabilities/`, organized by experiment, including analysis results.

### 5. Evaluation

* Scripts in the `evaluation/` folder (Python and R) to analyze predicted probabilities.
* Metrics include:

  * Area under the ROC curve
  * Brier score
  * Expected calibration error
  * Polytomous discrimination index
  * Calibration intercept & slope
  * Calibration curve
  * Distribution of predicted probabilities

---
## Computational setup
### 1. Packages
We used Python (v3.12.5) and PyTorch (v2.5.0) for model training. Pre-trained model weights were loaded from the models’ official GitHub repositories.[DINOv2](https://github.com/facebookresearch/dinov2), [RETFound](https://github.com/rmaphoh/RETFound/tree/main), [VisionFM](https://github.com/ABILab-CUHK/VisionFM)
AUROC was computed with the roc_auc_score from scikit-learn (v1.7.1). ECE was computed using the MulticlassCalibrationError from torchmetrics (v1.8.1). 

For further performance evaluation and visualization, we used R (v4.4.2). The PDI was computed using the [mcca (v0.7.0)](https://cran.r-project.org/web/packages/mcca/index.html) package. The calibration curves were plotted using [CalibrationCurves (v2.0.4)](https://cran.r-project.org/web/packages/CalibrationCurves/index.html), as well as the calculation of calibration-in-the-large and calibration slope. The Brier score was calculated under a multiclass framework using the mbrier function developed by Van Calster et al.(https://github.com/benvancalster/OrdinalCalibration) Confidence intervals (95%) for AUROC, PDI, and ECE were estimated using 1,000 bootstrap samples of the test sets.


### 2. Hardware
The analyses were performed on a GPU cluster node with the following specifications: 16 GB of RAM, 16 CPU cores, and 2 GPUs. Jobs were run on the gpu partition of the cluster with a maximum runtime of 6 hours.

---
### 3. How to run
#### 1. On your local machine
Make sure there's Python installed. Install the required packages using requirements.txt:

```bash
pip install -r requirements.txt
```

Execute the script :

```bash
python template_3class_DR.py -b convnextv2_large -bm fine_tune
```

#### 2. On a SLURM Cluster (with GPU)

```bash
#!/bin/bash
#SBATCH --mem 16g
#SBATCH -c 32
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --time 6:00:00
cd <dir>/BRSET
python template_3class_DR.py -b convnextv2_large -bm fine_tune
```

#### 3. On a SLURM Cluster (without GPU)

```bash
#!/bin/bash
#SBATCH --mem 16g
#SBATCH -c 32
#SBATCH --time 16:00:00
cd <dir>/BRSET
python template_3class_DR.py -b convnextv2_large -bm fine_tune
```

