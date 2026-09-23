# Predictions Folder

All model inference outputs go here in standard CSV format.

## Naming Convention
{model_name}.csv

## Expected Files
- simple_cnn.csv       (Quan + Phuc)
- poster_v2.csv        (Trang + Khanh)
- libreface.csv        (Duy)

## Required CSV Columns
- `image_path`,
- `true_label`,
- `pred_label` → integers **1 to 7**
- `prob_1`, `prob_2`, ..., `prob_7`

## Rules
- image_path must be RELATIVE (no hardcoded absolute paths)
- true_label and pred_label must be integers 1-7
- prob_1` to `prob_7` must sum to approximately 1.0 per row
- Use the label mapping defined in `data/label_map.json` or `eval/utils.py`
