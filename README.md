# Transforming Drug Discovery with Generative AI and Foundation Models: Enhancing High-Throughput Screening

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## Description
This project is associated with the submission of the coursework for the Research Project as part of the MPhil in Data Intensive Science at the University of Cambridge. The requirements for the coursework can be found at [`Instructions.md`](Instructions.md). The associated project writeup can be found under `report` folder. The dissertation report can be found at [`report/Dissertation.pdf`](report/Dissertation.pdf), and the executive summary can be found at [`report/exsummary.pdf`](report/exsummary.pdf).

The primary objective of this project is to reproduce the results presented in https://arxiv.org/abs/2407.09507 and the results presented in https://arxiv.org/abs/2407.17882. However, this work extends the original work by (i) integrating consistency distillation to the CellResDM model, and (ii) doing additional ablation studies.

Word Count: [Dissertation](report/Dissertation.pdf): 6,836 words, [Executive Summary](report/exsummary.pdf): 999 words.

### Projct Overview

CellResDM is a next-generation deep learning framework for synthesizing multiplexed Cell Painting immunofluorescence (IF) images directly from brightfield (BF) microscopy. By leveraging a novel residual-shifted diffusion process and transformer-enhanced UNet architectures, CellResDM enables rapid, high-fidelity prediction of biologically meaningful fluorescence channels from label-free input, dramatically reducing the need for costly and time-consuming experimental staining.

This repository provides a reproducible implementation of CellResDM and its consistency-distilled variant, along with strong baselines (UNet, Pix2Pix, DeepHCS). It includes all code for training, inference, and evaluation on the public Cell Painting Gallery dataset, as well as scripts for extracting and comparing single-cell morphological features. The methods here empower researchers to accelerate high-content screening, improve phenotypic profiling, and explore new frontiers in computational cell biology.

<details>
<summary>Table of Contents</summary>

- [Transforming Drug Discovery with Generative AI and Foundation Models: Enhancing High-Throughput Screening](#transforming-drug-discovery-with-generative-ai-and-foundation-models-enhancing-high-throughput-screening)
  - [Description](#description)
    - [Projct Overview](#projct-overview)
    - [Project Structure 🗂️](#project-structure-️)
  - [Installation](#installation)
  - [Dataset](#dataset)
    - [Channel Information](#channel-information)
  - [Models](#models)
    - [1. CellResDM (Residual-Shifted Diffusion Model)](#1-cellresdm-residual-shifted-diffusion-model)
    - [2. Consistency Distilled CellResDM](#2-consistency-distilled-cellresdm)
    - [3. UNet Baseline](#3-unet-baseline)
    - [4. Pix2Pix Baseline](#4-pix2pix-baseline)
    - [5. DeepHCS Baseline](#5-deephcs-baseline)
    - [6. Latent Diffusion (Optional/Advanced)](#6-latent-diffusion-optionaladvanced)
  - [Training](#training)
    - [Training CellResDM](#training-cellresdm)
    - [Training Consistency Distilled CellResDM](#training-consistency-distilled-cellresdm)
    - [Training Baseline Models](#training-baseline-models)
    - [Training Options](#training-options)
    - [Cluster Submission Scripts](#cluster-submission-scripts)
  - [Inference](#inference)
    - [Model Checkpoints](#model-checkpoints)
    - [Running Inference](#running-inference)
    - [Model Types](#model-types)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Cluster Submission Scripts](#cluster-submission-scripts-1)
  - [CellProfiler Pipeline](#cellprofiler-pipeline)
    - [Pipeline Overview](#pipeline-overview)
    - [Setup and Installation](#setup-and-installation)
    - [Data Preparation](#data-preparation)
    - [Pipeline Execution](#pipeline-execution)
    - [Results Analysis](#results-analysis)
  - [Results](#results)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

</details>


<!-- ## Overview

This project reproduces and extends CellResDM—a residual-shifted diffusion model for efficient cell painting synthesis from brightfield microscopy images. 

High-content phenotypic screening, especially the Cell Painting assay, has revolutionized early-stage drug discovery by enabling rapid morphological profiling of cells under thousands of perturbations. However, fluorescence-based imaging is costly, time-consuming, and prone to artifacts, limiting its scalability.

CellResDM efficiently predicts five-channel Cell Painting immunofluorescence (IF) from a single brightfield (BF) image in as few as 15 steps, over 60× faster than standard diffusion, with strong fidelity to both pixel-level and single-cell morphological features. We further introduce Consistency Distilled CellResDM, which reduces inference to just two steps while maintaining or even improving accuracy.

This repository contains implementations of:
- CellResDM with residual-shifted diffusion
- Consistency Distilled CellResDM
- Several baseline models (UNet, Pix2Pix, DeepHCS)
- Training and inference scripts
- Evaluation metrics including both standard image metrics and biological feature correlations -->

### Project Structure 🗂️

```
DIS_final_project/
├── src/               # Training & inference code
├── models/            # UNet, Swin-UNet, Diffusion, GAN, DeepHCS
├── basicsr/ ld ...    # 3rd-party utilities (truncated)
├── Cell_Profiler/     # CellProfiler pipelines + CSV feature tables
├── notebooks/         # Jupyter notebooks for exploration & plotting
└── report/            # Dissertation & supplementary material
```


## Installation

To set up the environment for this project:

```bash
# Clone the repository
git clone https://github.com/yourusername/cellresdm.git
cd cellresdm

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate m1_env

# Option 2: Using pip with a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt  # 100x faster pip install

# Set up Jupyter kernel (optional)
python -m ipykernel install --user --name=dissertation_venv
```
For logging, we use wandb. To set up wandb, you need to create an account and get the API key. Then, you can run the following command to login:

```bash
wandb login
```
This allows you to track your experiments and see the results in the wandb dashboard.




## Dataset

This project utilizes the public cpg0000 dataset from the Cell Painting Gallery, which is hosted on AWS S3. To download the data:

```bash
# Install AWS CLI if not already installed
pip install awscli


# move the download_plates.sh to the your desired storage path
mv scripts/download_plates.sh <your desired storage path>

# Make the download script executable
chmod +x download_plates.sh

# Edit the LOCAL_ROOT variable in the script to set your desired storage path
nano download_plates.sh

# Run the download script
./download_plates.sh
```

The script will download data for 10 plates: `BR00116991`, `BR00116992`, `BR00116995`, `BR00117024`, `BR00116993`, `BR00117025`, `BR00116994`, `BR00117026`, `BR00118041`, and `BR00118045`. The data will be organized as follows:

```
├───data
│   ├───BR00116991
│   │   ├───Images
│   │   │   ├───r01c01f01p01-ch1sk1fk1fl1.tiff  # Mitochondria (Alexa 647)
│   │   │   ├───r01c01f01p01-ch2sk1fk1fl1.tiff  # AGP (Alexa 568)
│   │   │   ├───r01c01f01p01-ch3sk1fk1fl1.tiff  # ER (Alexa 488 long pass)
│   │   │   ├───r01c01f01p01-ch4sk1fk1fl1.tiff  # RNA (Alexa 488)
│   │   │   ├───r01c01f01p01-ch5sk1fk1fl1.tiff  # DNA (Hoechst 33342)
│   │   │   ├───r01c01f01p01-ch6sk1fk1fl1.tiff  # BFLow (Brightfield z – 5 µm)
│   │   │   ├───r01c01f01p01-ch7sk1fk1fl1.tiff  # BF primary (Brightfield focal plane)
│   │   │   ├───r01c01f01p01-ch8sk1fk1fl1.tiff  # BFHigh (Brightfield z + 5 µm)
│   │   │   ...
│   ├───BR00116992
...
```

After downloading the data, run the script [`record_unique_paths.sh`](scripts/record_unique_paths.sh) to generate a list of unique image paths. This step is necessary for the data loader to correctly locate and load the images.

### Channel Information

| Channel  | Dye / Filter              | Biological target                                  | Notes                                                  |
| -------- | ------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| **ch01** | Alexa 647                 | Mitochondria                                       | MitoTracker Deep Red                                   |
| **ch02** | Alexa 568                 | AGP (Actin cytoskeleton + Golgi + plasma membrane) | Phalloidin (actin) + WGA (Golgi/PM) share this channel |
| **ch03** | Alexa 488 (long pass)     | Endoplasmic reticulum (ER)                         | Concanavalin A/Alexa 488 conjugate                     |
| **ch04** | Alexa 488                 | RNA (nucleoli + cytoplasmic RNA)                   | SYTO 14 green fluorescent nucleic acid stain           |
| **ch05** | Hoechst 33342             | DNA (nucleus)                                      | Hoechst nuclear stain                                  |
| **ch06** | Brightfield (z – 5 µm)    | Reference BF (below focal plane)                   | BFLow                                                  |
| **ch07** | Brightfield (focal plane) | Primary BF (reference focal plane)                 | BF primary                                             |
| **ch08** | Brightfield (z + 5 µm)    | Reference BF (above focal plane)                   | BFHigh                                                 |

In our experiments, we use the central Brightfield channel (ch07) as input to predict the five immunofluorescence channels (ch01-ch05).

## Models

This project implements several models for label-free Cell Painting synthesis. For each, we provide the model class and file, noise scheduling (if used), the Trainer class (for training), the Predictor class (for inference), and usage instructions.

---

### 1. CellResDM (Residual-Shifted Diffusion Model)
- **Model Class & File:** `UNetModelSwin` in [`models/unet.py`](models/unet.py)
- **Noise Scheduling:** `DiffusionScheduler` and `CFG` in [`src/noise_scheduling.py`](src/noise_scheduling.py)
- **Trainer:** `DiffusionTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `SwinUNetPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--use_diffusion` for training, `--model_type swin_unet` for inference
- **Summary:**
  CellResDM is a fast, high-fidelity diffusion model for translating brightfield (BF) images to multiplexed immunofluorescence (IF) images. It leverages a Swin-UNet backbone, which combines the spatial localization of UNet with the global context modeling of Swin Transformers. The core innovation is the residual-shifted diffusion process: instead of starting from pure noise, the model gradually adds noise and a fraction of the residual (difference) between the BF and IF images, allowing the network to focus on learning only the minimal, biologically relevant transformations. The noise schedule is non-uniform and geometric, starting with almost no noise and ramping up steeply, which enables high-quality synthesis in as few as 15 steps (vs. 1000+ for standard diffusion). Training and inference are both efficient, and the model achieves strong fidelity to both pixel-level and single-cell morphological features.

---

### 2. Consistency Distilled CellResDM
- **Model Class & File:** `UNetModelSwin` in [`models/unet.py`](models/unet.py)
- **Noise Scheduling:** `DiffusionScheduler` and `CFG` in [`src/noise_scheduling.py`](src/noise_scheduling.py)
- **Trainer:** `ConsistencyDistillationTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `SwinUNetPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--use_consistency_distillation` for training, `--model_type swin_unet` for inference
- **Summary:**
  Consistency Distilled CellResDM builds on a pretrained CellResDM by applying consistency distillation, a self-distillation technique that enforces the model to produce consistent outputs across adjacent timesteps. This is achieved by using an exponential moving average (EMA) of the model weights as a teacher and training the student to match the teacher's predictions at neighboring timesteps. The result is a model that can perform high-quality denoising in just 2 steps, dramatically accelerating inference. This approach not only speeds up generation but often improves output quality and generalization, as the model learns to be robust to noise and to produce stable, biologically meaningful outputs even under distribution shift. The same Swin-UNet backbone and geometric noise schedule are used, but the training loop is modified to enforce self-consistency.

---

### 3. UNet Baseline
- **Model Class & File:** `UNet` in [`models/plain_unet.py`](models/plain_unet.py)
- **Trainer:** `NormalTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `UNetPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--model_type unet` for inference
- **Summary:**
  The UNet baseline is a classic encoder-decoder architecture with skip connections, widely used for biomedical image segmentation and translation tasks. In this project, it is used for direct image-to-image translation from BF to IF images, without any diffusion or noise scheduling. The model is simple, fast to train, and serves as a strong baseline for pixel-level metrics, but it lacks the generative diversity and robustness of diffusion-based approaches.

---

### 4. Pix2Pix Baseline
- **Model Class & File:** `ImprovedPix2PixModel` in [`models/pix2pix_improved.py`](models/pix2pix_improved.py)
- **Trainer:** `Pix2PixTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `Pix2PixPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--use_pix2pix` for training, `--model_type pix2pix` for inference
- **Summary:**
  Pix2Pix is a conditional GAN framework for image-to-image translation. It uses a UNet-based generator and a PatchGAN discriminator, optimizing a combination of adversarial (GAN) loss and L1 loss. In this project, Pix2Pix is adapted for 5-channel BF-to-IF translation. The adversarial training encourages sharper, more realistic outputs, while the L1 loss ensures pixel-level accuracy. Pix2Pix can produce visually appealing results but may be less robust to out-of-distribution samples and can sometimes hallucinate features not present in the input.

---

### 5. DeepHCS Baseline
- **Model Class & File:** `DeepHCSModel` in [`models/deephcs.py`](models/deephcs.py)
- **Trainer:** `DeepHCSTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `DeepHCSPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--use_deephcs` for training, `--model_type deephcs` for inference
- **Summary:**
  DeepHCS is a two-stage architecture specifically designed for high-content screening. The first stage, the Transform Network, performs a coarse translation from BF to IF. The second stage, the Refinement Network, takes both the transform output and the original input to produce a refined prediction. The refinement stage is trained with a combination of MAE and multi-scale SSIM loss, encouraging both pixel accuracy and perceptual similarity. DeepHCS can be trained in three modes: transform only, refinement only, or both jointly. It is particularly effective at capturing fine morphological details and is competitive with more complex generative models.

---

### 6. Latent Diffusion (Optional/Advanced)
- **Model Class & File:** `UNetModelSwin` in [`models/unet.py`](models/unet.py) (for latent space)
- **Noise Scheduling:** `DiffusionScheduler` and `CFG` in [`src/noise_scheduling.py`](src/noise_scheduling.py)
- **Trainer:** `LatentDiffusionTrainer` in [`src/train.py`](src/train.py)
- **Predictor:** `LatentDiffusionPredictor` in [`src/inference.py`](src/inference.py)
- **How to Use:** `--use_latent_diffusion` for training, `--model_type latent_diffusion` for inference
- **Summary:**
  Latent Diffusion adapts the CellResDM approach to a compressed latent space using a pretrained Stable Diffusion VAE. Images are first encoded to a lower-dimensional latent space, where the diffusion process is run, and then decoded back to image space. This can dramatically reduce memory and compute requirements, enabling larger models or higher resolutions. The same geometric noise schedule and Swin-UNet backbone are used, but all operations are performed in the latent domain. This approach is experimental but can be useful for scaling up or for applications where speed and memory are critical.

---

**Model selection for training and inference is controlled by the `--model_type` flag. See the Training and Inference sections for example commands.**

## Training

The main training script is `src/train.py`, which supports various training configurations through command-line arguments. The script uses the Hugging Face Trainer API with custom trainers for different approaches.

### Training CellResDM

To train the basic CellResDM model:

```bash
accelerate launch src/train.py --output_dir results/cellresdm --use_diffusion
```

### Training Consistency Distilled CellResDM

To train the Consistency Distilled CellResDM model (requires a pretrained CellResDM model):

```bash
accelerate launch src/train.py \
  --output_dir results/cellresdm-distilled \
  --use_consistency_distillation \
  --num_train_epochs 3 \
  --cd_pretrained_path results/cellresdm/checkpoint-69120
```

### Training Baseline Models

```bash
# UNet
accelerate launch src/train.py --output_dir results/unet

# Pix2Pix
accelerate launch src/train.py --output_dir results/pix2pix --use_pix2pix

# DeepHCS
accelerate launch src/train.py --output_dir results/deephcs --use_deephcs --deephcs_training_stage 3
```

### Training Options

Key training parameters:

- `--use_diffusion`: Enable diffusion training (CellResDM)
- `--use_consistency_distillation`: Enable consistency distillation
- `--use_pix2pix`: Enable Pix2Pix training
- `--use_deephcs`: Enable DeepHCS training
- `--use_latent_diffusion`: Enable latent diffusion with SD-VAE

For CellResDM training, you can also use the following options:
- `--use_lpips`: Use LPIPS perceptual loss (recommended)
- `--use_loss_coef`: Use per-timestep loss coefficients (not recommended)
- `--kappa`, `--p`, `--eta_T`, `--T`: Diffusion noise scheduling hyperparameters

For Consistency Distilled CellResDM training, you can also use the following options:
- `--cd_ema_decay`, `--cd_lambda_weight`: Consistency distillation parameters

### Cluster Submission Scripts
For cluster environments, we provide a sample SLURM submission scripts:
- [`submit_train.sh`](scripts/submit_train.sh): Basic CellResDM training

## Inference

The main inference script is `src/inference.py`, which supports different model types and evaluation metrics.

The inference script generates synthetic images using the trained model. During inference, these images are automatically evaluated using pixel-level metrics, and the results are saved in a `metrics.csv` file. Copies of the generated synthetic images are also saved in the `predictions` subfolder within the specified output directory.

### Model Checkpoints

The model checkpoints are available at the following [google drive link](https://drive.google.com/drive/folders/1CvaSDDd7I2fuZUyvyTOo-xV9rvy2Zjdn?usp=sharing)

### Running Inference

```bash
python src/inference.py \
  --model_path <path_to_model> \
  --model_type <model_type> \
  --output_dir <output_dir>
```

### Model Types

The `--model_type` parameter specifies which predictor to use:
- `swin_unet`: For CellResDM and Consistency Distilled CellResDM. This is because the model used for these two models are the swin-transformer augumented Unet.
- `unet`: For UNet baseline
- `pix2pix`: For Pix2Pix baseline
- `deephcs`: For DeepHCS baseline
- `latent_diffusion`: For latent diffusion models

### Evaluation Metrics

The inference script calculates several metrics:
- **MSE**: Mean Squared Error
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio

For more comprehensive evaluation including morphological feature correlations, the Cell Profiler pipeline in `Cell_Profiler/` can be used to extract single-cell features from both real and synthetic images.


### Cluster Submission Scripts

For cluster environments, we provide a sample SLURM submission scripts:
- [`submit_inference.sh`](scripts/submit_inference.sh): Basic CellResDM inference



## CellProfiler Pipeline

The CellProfiler pipeline located in `Cell_Profiler/` enables comprehensive morphological feature extraction from both real and synthetic images for biological validation. This pipeline performs cell segmentation followed by feature extraction, allowing quantitative comparison between synthetic and ground-truth images.

### Pipeline Overview

The main pipeline file is located at [`Cell_Profiler/CPJUMP1_analysis_without_batchfile_406.cpproj`](Cell_Profiler/CPJUMP1_analysis_without_batchfile_406.cpproj). The pipeline workflow consists of:

1. **Image segmentation**: Automated cell boundary detection
2. **Feature extraction**: Quantification of morphological properties from segmented cells
3. **Feature comparison**: Statistical analysis against ground-truth measurements

### Setup and Installation

1. Download and install CellProfiler from the [official website](https://cellprofiler.org/releases/)

2. Launch CellProfiler:
   ```bash
   cellprofiler
   ```

3. Load the provided pipeline file into CellProfiler

### Data Preparation

Before running the CellProfiler pipeline, your folder structure should look like this:

```
Cell_Profiler/
├── CPJUMP1_analysis_without_batchfile_406.cpproj   # Main CellProfiler pipeline
├── cell_profiler_original_files/                   # Place your raw prediction folders here
│   ├── cell_profiler_pre_process.py                # Preprocessing script
|   └── get_real.py                                 # Script to move the target.npy files to the real folder and rename them to pred.npy
│   ├── real/                                       # Ground truth images obtained by running the get_real.py script
│   └── [model_name]/                               # Individual model predictions
├── cell_profiler_input/                            # Auto-generated preprocessed data
|   ├── ...
│   └── [model_name]/                               # CellProfiler-ready format
└── cell_profiler_results/                          # Pipeline outputs and analysis
    ├── analysis.ipynb                              # Feature correlation analysis
    ├── image_overlay_visualization.ipynb           # Segmentation visualization
    └── [model_name]/                               # Individual model results
```

This can be achieved by following the below steps.

**Steps:**

1. **Copy predictions**: Place your model's `predictions` folder into `Cell_Profiler/cell_profiler_original_files/`

2. **Run preprocessing**:
   ```bash
   python Cell_Profiler/cell_profiler_original_files/cell_profiler_pre_process.py
   ```
   This converts images to CellProfiler-compatible TIFF format in `cell_profiler_input/`

3. **Ready for CellProfiler**: Your data is now prepared for the pipeline

### Pipeline Execution

1. **Configure input paths**: In the CellProfiler pipeline, set the input data location to `Cell_Profiler/cell_profiler_input/<predictions_folder_name>/`

2. **Verify data loading**: Click **"Press to view CSV file"** to confirm the data matches the `load_data.csv` file in the input directory

3. **Set output paths**: Configure the output directory to `Cell_Profiler/cell_profiler_results/<predictions_folder_name>/`
   
   *Note: Input and output paths can be easily configured in CellProfiler preferences for convenience*

4. **Execute pipeline**: Run the analysis to generate morphological feature measurements

### Results Analysis

After pipeline execution, two analysis notebooks are available in `Cell_Profiler/cell_profiler_results/`:

- **[`analysis.ipynb`](Cell_Profiler/cell_profiler_results/analysis.ipynb)**: Performs correlation analysis of morphological features between synthetic and real images
- **[`image_overlay_visualization.ipynb`](Cell_Profiler/cell_profiler_results/image_overlay_visualization.ipynb)**: Provides visual inspection of segmentation results and image overlays

These notebooks enable comprehensive evaluation of how well the synthetic images preserve biologically relevant morphological characteristics.


## Results

Our experiments show that:

1. **CellResDM** achieves great performance in synthesizing immunofluorescence images from brightfield, outperforming UNet and Pix2Pix baselines in both pixel-level metrics (MSE, SSIM, PSNR) and biological feature preservation.

2. **Consistency Distilled CellResDM** further improves results while reducing inference from 15 to just 2 steps, making it both more accurate and more efficient than all baseline methods.

3. **Morphological Feature Preservation**: Both CellResDM variants excel at preserving biologically meaningful features, as evidenced by high correlation with ground-truth single-cell metrics and clear clustering in UMAP visualizations.

4. **Out-of-Distribution Generalization**: Consistency Distilled CellResDM shows particularly strong generalization to out-of-distribution data, likely due to the regularizing effect of consistency distillation training.

For detailed results and visualizations, see the notebooks in the `notebooks/` directory:
- [`channel_visualization.ipynb`](notebooks/channel_visualization.ipynb): Visualizes the different channels and model outputs
- [`results_visualization.ipynb`](notebooks/results_visualization.ipynb): Visualizes the results of the different models
- [`umap_analysis.ipynb`](notebooks/umap_analysis.ipynb): Analyzes the feature space using UMAP

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Prof Guang Yang and Jiang Le for their guidance and support on the project
- The Cell Painting Gallery for providing the cpg0000 dataset
- The authors of the original CellResDM paper for their innovative approach
- The Hugging Face team for their Transformers and Diffusers libraries
