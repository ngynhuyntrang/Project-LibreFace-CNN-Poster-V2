# How to run (RTX 5070 Laptop GPU)

## Setup (one-time)
- Step 1: Go to the model's directory: `cd "models\simple_cnn"`

- Step 2: Uninstall the current PyTorch build, then reinstall with CUDA 12.8
          (required for RTX 5070 Blackwell architecture — sm_120 support was not added until cu128):
          pip uninstall torch torchvision -y
      pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
      Note: --no-cache-dir is required. Without it, pip may silently reuse a cached incompatible build and report success.

- Step 3: Verify GPU is actually computing (not just detected):
          `python -c "import torch; t = torch.tensor([2.0]).cuda(); print(t * t); print(torch.cuda.get_device_capability())"`
          Expected output:
            tensor([4.], device='cuda:0')    (confirms kernels are running on GPU)
            (12, 0)                          (confirms sm_120 Blackwell compute capability)

          Note: torch.cuda.is_available() returning True is NOT sufficient — it only checks
          that the driver detected a CUDA device, not that sm_120 kernels are compiled in.
          Always verify with an actual tensor computation.

## Training
- Step 4: `python train.py --no-class-weights --lr 3e-4 --epochs 60`
          Training will run up to 60 epochs with early stopping.
          Best checkpoint is saved automatically to models/simple_cnn/checkpoints/simple_cnn_best.pt

## Inference
- Step 5: `python inference.py --split test`
          Output: predictions/simple_cnn.csv

## Validation check
- Step 6: Go back to the project root and run the consolidation check:
      `cd ../../`
      `cd scripts`
      `python consolidate.py`
      If simple_cnn.csv passes all checks, it is ready for the final evaluation.

## Generate results
- Step 7: Generate tables, plots, etc:
      `python generate_results.py`


# Tips for hitting 65%+

- Class weights matter. RAF-DB train is heavily skewed: Happiness has 3,817 samples while Fear
  has only 225 and Disgust 574. Weighting recovers F1 points on minority classes, which lifts
  macro-F1 noticeably.

- Don't over-augment. RAF-DB faces are already aligned. Heavy rotation or aggressive crops
  degrade performance. The current setup (flip + small rotation + mild jitter + small
  RandomErasing) is calibrated for aligned faces.

- BatchNorm + GAP is the cheap win over a vanilla VGG-ish head. GAP cuts parameters by ~10x
  vs flattening 512x7x7 into a linear layer and reduces overfitting on a ~12K-image train set.

- Watch val loss, not val acc, for the scheduler. With class-weighted loss + label smoothing,
  val loss is a more stable plateau signal than val acc.

- If you stall around 55-60%, kill class weights first and rerun. On RAF-DB the choice between
  weighted CE and plain CE flips depending on which metric you care about (macro-F1 vs overall
  accuracy). Overall accuracy on test is the headline number the markers see, so plain CE often
  wins there.

- GPU sanity: the model is ~4M params. Start with batch 64 (the default) and increase to 128
  only if GPU memory allows.