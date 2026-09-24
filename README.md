# Facial Expression Recognition on RAF-DB

### A comparative evaluation of Simple CNN, POSTER_V2, and LibreFace

**CSCI323 | Group 4 | University of Wollongong**

This project compares three approaches to seven-class facial expression recognition on the RAF-DB dataset: a custom Simple CNN, POSTER_V2, and LibreFace. A shared prediction format and evaluation pipeline support comparison using accuracy, macro F1, weighted F1, per-class F1, and confusion matrices.

**Best reported result:** POSTER_V2 achieved **92.18% accuracy** and **87.34% macro F1** on the 3,068-image test set.

> Results below are transcribed from the Group 4 report, not independently reproduced for this README. Simple CNN was trained from scratch, LibreFace was adapted and fine-tuned on RAF-DB, and POSTER_V2 was evaluated using a pretrained RAF-DB checkpoint. This is a comparison of the reported model pipelines, not a controlled comparison with identical pretraining or training budgets.

## Project goals

- Compare a convolutional baseline with more advanced facial expression recognition models.
- Standardize label mapping and prediction outputs across all three models.
- Examine performance on an imbalanced dataset, especially Fear and Disgust.
- Produce aggregate metrics, per-class results, and confusion matrices.

## Dataset

The project uses the **single-label, seven-class subset of RAF-DB**, containing 15,339 images. The report describes an 80:20 split of the original training partition into training and validation data, with the official test partition retained for evaluation.

| Partition | Images |
| --- | ---: |
| Training | 9,816 |
| Validation | 2,455 |
| Test | 3,068 |
| **Total** | **15,339** |

The shared evaluation convention uses **1-indexed labels**:

| Label | Expression | Images across the full subset |
| --- | --- | ---: |
| 1 | Surprise | 1,619 |
| 2 | Fear | 355 |
| 3 | Disgust | 877 |
| 4 | Happiness | 5,957 |
| 5 | Sadness | 2,460 |
| 6 | Anger | 867 |
| 7 | Neutral | 3,204 |

The report describes 224 × 224 RGB inputs with ImageNet normalization for the shared model pipeline. Model-specific preprocessing also applies; for example, POSTER_V2 uses a 112 × 112 input for its MobileFaceNet stream.

Obtain RAF-DB separately under the dataset provider's access and usage terms. A copy of the dataset should not be assumed to be included in this repository.

## Models

| Model | Approach | Use in this project |
| --- | --- | --- |
| Simple CNN | Five convolutional blocks, global average pooling, dropout, and a seven-class classifier | Custom baseline trained from scratch |
| POSTER_V2 | Multi-scale facial features and transformer attention using IR-50 and MobileFaceNet components | Adapted inference wrapper with a pretrained RAF-DB checkpoint |
| LibreFace | Facial analysis framework incorporating MAE teacher / ResNet-18 student methods | Adapted to RAF-DB's seven classes and fine-tuned |

### Simple CNN

Each feature block applies a 3 × 3 convolution, batch normalization, ReLU, and 2 × 2 max pooling. Feature channels increase through 64, 128, 256, 512, and 512. Global average pooling produces a 512-dimensional representation, followed by a classifier with a 256-unit hidden layer and seven output logits.

The reported training pipeline uses AdamW, cross-entropy with label smoothing, augmentation, learning-rate scheduling, early stopping, and best-checkpoint selection.

### POSTER_V2

The architecture combines IR-50 feature maps at three scales with face-specific features from a frozen MobileFaceNet encoder. Attention outputs feed a lightweight vision transformer for seven-class prediction.

Its inference wrapper converts the model's zero-indexed outputs to the project's one-indexed label convention. The report identifies three required weights: a RAF-DB POSTER_V2 checkpoint, IR-50 weights, and MobileFaceNet weights. The conclusion identifies the evaluated POSTER_V2 result as using a pretrained RAF-DB checkpoint; training settings discussed elsewhere in the report should not be treated as proof of a fresh end-to-end training run.

### LibreFace

The project adapts LibreFace to the RAF-DB label space, including mapping from AffectNet's eight-class setting and using a new seven-class classifier. The report describes fine-tuning, checkpoint saving, and test inference, with training stopping at epoch 26 after no further improvement.

## Results

### Overall performance

All values are percentages, as reported for the RAF-DB test set (**n = 3,068**).

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Simple CNN | 76.14 | 65.86 | 75.80 |
| **POSTER_V2** | **92.18** | **87.34** | **92.09** |
| LibreFace | 85.72 | 77.58 | 85.62 |

### Per-class F1

| Expression | Simple CNN | POSTER_V2 | LibreFace |
| --- | ---: | ---: | ---: |
| Surprise | 76.59 | **90.85** | 86.97 |
| Fear | 46.72 | **76.12** | 64.06 |
| Disgust | 45.48 | **75.75** | 55.74 |
| Happiness | 88.43 | **97.05** | 94.30 |
| Sadness | 70.23 | **90.70** | 82.93 |
| Anger | 62.88 | **89.72** | 75.54 |
| Neutral | 70.71 | **91.19** | 83.53 |

POSTER_V2 leads on every reported metric and class. Happiness has the highest F1 for all three models, while Fear and Disgust remain the most difficult classes. Macro F1 is particularly useful here because it gives each class equal weight despite the dataset imbalance.

## Evaluation workflow

1. Prepare the RAF-DB images, split files, and shared label mapping.
2. Load the appropriate model checkpoint and apply its preprocessing.
3. Generate test predictions in the common CSV format.
4. Validate and consolidate predictions across models.
5. Calculate metrics and generate comparison tables and confusion matrices.

The shared CSV schema is:

```csv
image_path,true_label,pred_label,prob_1,prob_2,prob_3,prob_4,prob_5,prob_6,prob_7
```

`prob_1` through `prob_7` must follow the label order above. Evaluation should use the same test images and labels for every model. Synthetic or dummy predictions are suitable only for testing the pipeline, not for reporting model performance.

## Getting started

### Download the project and model files

Install Git and Git LFS. On macOS with Homebrew:

```bash
brew install git git-lfs
git lfs install
```

Clone the repository and retrieve its LFS-managed model files:

```bash
git clone https://github.com/ngynhuyntrang/Project-LibreFace-CNN-Poster-V2.git
cd Project-LibreFace-CNN-Poster-V2
git lfs pull
git lfs ls-files
```

The large model files identified during repository preparation include:

| File | Approximate size |
| --- | ---: |
| `Full 3 models ran already/models/poster_v2/checkpoint/rafdb_best.pth` | 228 MB |
| `Full 3 models ran already/models/poster_v2/models/pretrain/ir50.pth` | 116 MB |

These files are intended to be stored through Git LFS. Confirm that the actual weights have downloaded before inference; an LFS pointer alone is not a usable checkpoint.

### Prepare the execution environment

Enter the project directory and optionally create an isolated Python environment:

```bash
cd "Full 3 models ran already"
python3 -m venv .venv
source .venv/bin/activate
```

Use the dependency manifests and model-specific documentation included with the source code to install compatible packages. Then obtain RAF-DB, configure dataset and checkpoint paths, and verify that all required weights are available, including the MobileFaceNet and LibreFace weights.

**Reproduction status:** exact Python/package versions, command-line arguments, and runnable training/inference commands have not been verified against the source code for this README. The report does not provide a complete installation manifest. Refer to the scripts and their configuration before starting a run; the reported scores are reference results, not a guarantee of an identical rerun.

## Key project files

The following paths were identified from the project files shown during repository preparation. This is a partial directory guide.

| Path | Purpose |
| --- | --- |
| `CSCI323_Group4_Report.pdf` | Full project report, figures, methodology, and references |
| `Full 3 models ran already/` | Main project workspace |
| `Full 3 models ran already/models/poster_v2/` | POSTER_V2 integration and model assets |
| `Full 3 models ran already/scripts/create_rafdb_splits.py` | Dataset split preparation script |
| `Full 3 models ran already/scripts/verify_raf_db.py` | Dataset verification script |
| `Full 3 models ran already/scripts/generate_results.py` | Results-generation script |
| `Full 3 models ran already/scripts/final_comparison.py` | Model-comparison script |

## Limitations and future work

- Class imbalance and visual similarity affect recognition, particularly for Fear and Disgust.
- Models differ in pretraining, architecture, and adaptation strategy, so the results do not isolate architectural effects alone.
- Runtime, hardware efficiency, and real-time capability were not established by the reported accuracy/F1 tables.
- Webcam deployment and testing under blur, lighting changes, pose variation, and occlusion are proposed future work.
- The task predicts dataset-defined facial expression labels; those labels should not be interpreted as a reliable measurement of a person's internal emotional state.

## Team contributions

The report records equal contributions of **20% per member**.

| Member | Main contributions |
| --- | --- |
| Minh Quan Dang | Evaluation framework, Simple CNN, data/results utilities, report writing and final checks |
| Ba Hoang Khanh Phan | POSTER_V2 integration and inference wrapper, RAF-DB preparation, result consolidation and model comparison |
| An Duy Pham | LibreFace workflow, LibreFace and background sections, report consolidation and formatting |
| Ta Hoang Phuc Vo | LibreFace integration, label mapping and fine-tuning, inference, comparison outputs, dataset and conclusion sections |
| Ngoc Huyen Trang Nguyen | POSTER_V2 workflow and code integration with Ba Hoang Khanh Phan, background theory, and introduction |

## Report and acknowledgements

See the [full project report](CSCI323_Group4_Report.pdf) for confusion matrices, implementation discussion, and the complete bibliography.

The project builds on RAF-DB, POSTER_V2 / POSTER++, LibreFace, and their associated research and pretrained models. Key references listed in the report are:

- Li, S., Deng, W., and Du, J. (2017). *Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild.* CVPR.
- Mao, J., et al. (2025). *POSTER++: A simpler and stronger facial expression recognition network.* Pattern Recognition, 157, 110951.
- Chang, D., et al. (2024). *LibreFace: An open-source toolkit for deep facial expression analysis.* WACV.

## License and reuse

No project-wide license is declared in this README. Check the repository's license files before reuse. Third-party source code, datasets, and pretrained weights retain their own licenses and usage conditions; their inclusion does not grant additional redistribution rights.
