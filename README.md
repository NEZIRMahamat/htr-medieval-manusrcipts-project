# CATMuS Segmentation Data Notes

This README documents the segmentation part of the project.

## Project Scope

The segmentation sub-project detects text lines on full medieval manuscript pages.
It does not transcribe text.

Pipeline covered here:

```text
full manuscript page
-> U-Net binary line segmentation
-> predicted line masks
-> bbox / polygon / reading order
-> line crops
-> segmentation JSON contract
-> HTR module
```

The HTR module will then use the segmentation test outputs as integration input.

## Dataset

Dataset: `CATMuS/medieval-segmentation`

Local path:

```text
data/segment_data
```

Local split sizes:

| Split          | Pages |
| -------------- | ----: |
| `train`      | 1,336 |
| `validation` |   191 |
| `test`       |   178 |

The model uses only line objects from the CATMuS annotations:

```text
objects.type == "line"
```

The line polygons are rasterized into binary masks:

```text
0 = background
1 = text line
```

More details are documented in:

```text
DATA_SOURCES.md
```

## Model

Model: U-Net binary segmentation

Checkpoint:

```text
outputs/segmentation/checkpoints/unet-segmentation.keras
```

Input shape:

```text
image: 256 x 256 x 3
mask:  256 x 256 x 1
```

Training setup used locally:

| Parameter         |     Value |
| ----------------- | --------: |
| train pages       |       480 |
| validation pages  |        60 |
| epochs            |        15 |
| batch size        |         4 |
| optimizer         |     AdamW |
| learning rate     |      1e-4 |
| training hardware |       CPU |
| training time     | 91.51 min |

More details are documented in:

```text
MODEL_CARD.md
```

## Hugging Face Deployment

Model repository:

```text
https://huggingface.co/nzIng/unet-manuscripts-segmentation/tree/main
```

Gradio Space:

```text
https://huggingface.co/spaces/nzIng/unet-manuscripts-segmentation-app
```

The model repository contains the exported Keras checkpoint and a simple inference script.
The Gradio Space provides a web interface to upload a manuscript page and visualize the predicted line segmentation.

## Evaluation Results

Validation on 60 pages:

| Metric    |  Value |
| --------- | -----: |
| Dice      | 0.9748 |
| IoU       | 0.9513 |
| Precision | 0.9767 |
| Recall    | 0.9736 |
| F1        | 0.9748 |

Test on 60 pages:

| Metric    |  Value |
| --------- | -----: |
| Dice      | 0.9379 |
| IoU       | 0.8849 |
| Precision | 0.9296 |
| Recall    | 0.9473 |
| F1        | 0.9379 |

Project target:

```text
validation threshold: IoU > 0.75
excellence threshold: IoU > 0.85
```

The current test IoU is above the excellence threshold on the evaluated 60-page test subset.

## Test Set Sealing

The local `test` split was sealed with SHA-256.

Files:

```text
outputs/splits/test_set_segmentation_manifest.json
outputs/splits/test_set_segmentation_sha256.txt
```

Global SHA-256:

```text
07e3e5a02b09c57e8df42be272b51931bab017a6bec686aeea03e1157cc92c80
```

## Main Source Files

```text
src/segmentation/dataset.py
src/segmentation/utils.py
src/segmentation/model.py
src/segmentation/train.py
src/segmentation/evaluate.py
src/segmentation/crops.py
src/segmentation/data_contract_json.py
src/segmentation/normalize_data_contract.py
src/segmentation/save_original_images.py
```

Main responsibilities:

| File                                            | Role                                                |
| ----------------------------------------------- | --------------------------------------------------- |
| `src/segmentation/dataset.py`                   | dataset download/load/audit helpers                 |
| `src/segmentation/utils.py`                     | mask generation and preprocessing                   |
| `src/segmentation/model.py`                     | U-Net model, losses, metrics                        |
| `src/segmentation/train.py`                     | segmentation training                               |
| `src/segmentation/evaluate.py`                  | validation/test evaluation and line prediction JSON |
| `src/segmentation/crops.py`                     | crop generation from predicted line bboxes          |
| `src/segmentation/data_contract_json.py`        | segmentation JSON schema                            |
| `src/segmentation/normalize_data_contract.py`   | normalize and validate final segmentation contracts |
| `src/segmentation/save_original_images.py`      | save original page images for debug/inspection      |

## How To Run

Run from the project root.

Train the segmentation model:

```powershell
venv\Scripts\python.exe -B src\segmentation\train.py
```

Evaluate the current checkpoint on the configured split:

```powershell
venv\Scripts\python.exe -B src\segmentation\evaluate.py
```

Generate crops from predictions:

```powershell
venv\Scripts\python.exe -B src\segmentation\crops.py
```

Generate and validate segmentation contracts:

```powershell
venv\Scripts\python.exe -B src\segmentation\normalize_data_contract.py
```

Seal the segmentation test split:

```powershell
powershell -ExecutionPolicy Bypass -File notes\sceller_test_set.ps1
```

## Outputs

Validation outputs:

```text
outputs/segmentation/metrics/validation_segmentation_metrics.json
outputs/segmentation/predictions/validation_segmentation_predictions.json
outputs/segmentation/predictions/validation_segmentation_predictions_with_crops.json
outputs/segmentation/data_contract/validation_segmentation_contract.json
outputs/segmentation/crops_line/validation/
```

Test outputs:

```text
outputs/segmentation/metrics/test_segmentation_metrics.json
outputs/segmentation/predictions/test_segmentation_predictions.json
outputs/segmentation/predictions/test_segmentation_predictions_with_crops.json
outputs/segmentation/data_contract/test_segmentation_contract.json
outputs/segmentation/crops_line/test/
```

The old folder below is obsolete and should not be used:

```text
outputs/segmentation/crops_line_split/
```

## HTR Integration

The HTR model is trained and evaluated on the line-level dataset:

```text
CATMuS/medieval
```

For the complete project pipeline, HTR should consume the segmentation test output:

```text
outputs/segmentation/predictions/test_segmentation_predictions_with_crops.json
outputs/segmentation/data_contract/test_segmentation_contract.json
outputs/segmentation/crops_line/test/
```

HTR should add transcription fields to the line-level JSON or to the global data contract.

## Known Limitations

- The model was trained on a subset, not the full CATMuS segmentation training split.
- Training was done on CPU because of local GPU/memory constraints.
- The model predicts at `256 x 256`, then bboxes are rescaled to original page size.
- Exported prediction polygons are rectangular polygons from bboxes.
- Complex layouts, marginalia, decorations, and dense pages may still require manual inspection.
- Final reported test metrics are computed on 60 pages from the sealed test split.
