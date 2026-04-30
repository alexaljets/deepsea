# deepsea
Recreation of DeepSEA (Zhou, 2015) in PyTorch

# DeepSEA CNN Reproduction — CMSE 410

## Overview
DeepSEA is a deep learning framework developed by Zhou and Troyanskaya at Princeton University in 2015 that uses a Convolutional Neural Network to predict chromatin features from raw DNA sequences, specifically taking 1,000 base-pair sequences as input and outputting probability predictions for 919 distinct chromatin features including transcription factor binding sites, DNase hypersensitivity sites, and histone modifications derived from the ENCODE and Roadmap Epigenomics projects. This project reproduced the DeepSEA architecture from scratch in PyTorch, rebuilding the training dataset from source ENCODE BED files after the original Princeton data bundle was found to be offline, and training the model on 500,000 sequences using an NVIDIA V100 GPU on the MSU HPCC. The reproduction achieved a mean test AUROC of 0.871 across 919 chromatin features on held-out chromosomes 8 and 9, compared to the original paper's reported AUROC of 0.933 on the full 4.4 million sequence dataset, with 374 of 919 features exceeding AUROC 0.9; saliency map analysis confirmed the model learned biologically meaningful sequence features, and a model randomization test verified that these patterns were data-dependent rather than architectural artifacts.

## Results
- Test AUROC: 0.871 (paper: 0.933)
- Test AUPRC: 0.236
- 374/919 features above AUROC 0.9
- Training: 500K sequences, 60 epochs, V100 GPU, ~2 hours

## Repository Structure
model.py        — DeepSEA CNN architecture
train.py        — Training loop
data_utils.py   — Data loading from .npy files
evaluate.py     — AUROC/AUPRC + saliency maps
smoke_test.py   — Pre-training sanity checks
*.sb            — SLURM job scripts for MSU HPCC

## Requirements
pip install -r requirements.txt

## How to Reproduce

### 1. Set up environment (MSU HPCC)
module load Miniforge3/24.3.0-0
conda create -n deepsea python=3.11
conda activate deepsea
pip install -r requirements.txt

### 2. Download and build dataset
sbatch submit_download.sb   # downloads ENCODE BED files + hg19 via rsync from UCSC
sbatch submit_build.sb      # builds train/valid/test .npy files (~45GB total)

### 3. Train
sbatch submit_train.sb      # 60 epochs, 500K samples, ~2 hours on V100

### 4. Evaluate
sbatch run_test_eval.sb     # test set AUROC/AUPRC
sbatch run_evaluate.sb      # saliency maps + randomization test

## Data
Dataset is rebuilt from ENCODE AWG uniform processing pipeline BED files
hosted on UCSC (wgEncodeAwgDnaseUniform, wgEncodeAwgTfbsUniform) using
github.com/jakublipinski/build-deepsea-training-dataset.
Original Princeton bundle was unavailable; rebuilt dataset matches
original to within 0.1% on labels.

## Reference
Zhou & Troyanskaya (2015). Predicting effects of noncoding variants
with deep learning-based sequence model. Nature Methods, 12, 931-934.
