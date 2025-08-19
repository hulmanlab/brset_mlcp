Great thanks to Luis Nakayama for the source code, https://github.com/luisnakayama/BRSET
Adjustments for local application has been made.   

Here’s a structured draft of your README focusing on the usage section:

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


### 4. Models and Outputs

* **Models:** Saved in `models/` after fine-tuning.
* **Output predicted probabilities:** Stored in `output_predicted_probabilities/`, organized by experiment, including analysis results.

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
