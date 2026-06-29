# Model Card - CATMuS Medieval Line Segmentation

## Model Details

Model type: U-Net

Task: binary page segmentation for medieval manuscript text lines.

Checkpoint:

```text
outputs/segmentation/checkpoints/unet-segmentation.keras
```

The model predicts a binary mask:

- `0`: background
- `1`: text line

The predicted mask is post-processed into line geometries: `bbox`, rectangular `polygon`, `reading_order`, confidence score, and line crops.

## Deployment

Hugging Face model repository:

```text
https://huggingface.co/nzIng/unet-manuscripts-segmentation/tree/main
```

Hugging Face Gradio Space:

```text
https://huggingface.co/spaces/nzIng/unet-manuscripts-segmentation-app
```

The model repository hosts the exported Keras checkpoint and a simple inference script. The Gradio Space exposes the model through a web interface that displays the input page, the segmented page, the predicted mask, line crops, and a downloadable JSON output.

## Training Data

Dataset: `CATMuS/medieval-segmentation`

Local dataset:

```text
data/segment_data
```

Full local split sizes:

| Split          | Pages |
| -------------- | ----: |
| `train`      | 1,336 |
| `validation` |   191 |
| `test`       |   178 |

Training was done on a local subset because of CPU/RAM constraints:

| Usage                 | Pages |
| --------------------- | ----: |
| train                 |   480 |
| validation            |    60 |
| final test evaluation |    60 |

The full local `test` split is sealed with SHA-256:

```text
07e3e5a02b09c57e8df42be272b51931bab017a6bec686aeea03e1157cc92c80
```

## Training Setup

Input size:

```text
256 x 256 x 3
```

Mask size:

```text
256 x 256 x 1
```

Main hyperparameters:

| Parameter     |           Value |
| ------------- | --------------: |
| epochs        |              15 |
| batch size    |               4 |
| optimizer     |           AdamW |
| learning rate |            1e-4 |
| weight decay  |            1e-4 |
| loss          | BCE + Dice loss |

Hardware:

```text
CPU training only
```

Training time:

```text
91.51 minutes, about 1h31m31s
```

## Evaluation

Post-processing parameters:

| Parameter                  |                                            Value |
| -------------------------- | -----------------------------------------------: |
| threshold                  |                                              0.5 |
| min_area_model_pixels      |                                               20 |
| border_margin_model_pixels |                                                0 |
| post-processing            | connected components + tall component line split |

Validation results on 60 pages:

| Metric    |  Value |
| --------- | -----: |
| Dice      | 0.9748 |
| IoU       | 0.9513 |
| Precision | 0.9767 |
| Recall    | 0.9736 |
| F1        | 0.9748 |

Test results on 60 pages:

| Metric    |  Value |
| --------- | -----: |
| Dice      | 0.9379 |
| IoU       | 0.8849 |
| Precision | 0.9296 |
| Recall    | 0.9473 |
| F1        | 0.9379 |

The target from the project brief was:

- validation threshold: IoU > 0.75
- excellence threshold: IoU > 0.85

The current test result, IoU `0.8849`, is above the excellence threshold on the evaluated 60-page test subset.

## Outputs

Validation outputs:

```text
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

The HTR model is trained and evaluated on its own line-level dataset, `CATMuS/medieval`. For the complete project pipeline, the HTR module should then run on the segmentation test output:

```text
outputs/segmentation/predictions/test_segmentation_predictions_with_crops.json
outputs/segmentation/data_contract/test_segmentation_contract.json
outputs/segmentation/crops_line/test/
```

This keeps the HTR training split independent while using the final segmentation test output as the integration input for transcription.

## Limitations

- The model was trained on a subset, not the full `train` split.
- Training was CPU-only, so image size and batch size were kept small.
- The model predicts masks at `256 x 256`, then geometries are rescaled to original page coordinates.
- Polygons exported after prediction are rectangular polygons from bboxes, not the original curved CATMuS line polygons.
- Complex layouts, decorations, marginalia, tables, and very dense pages may still require manual review.
- The final reported test metrics are computed on 60 test pages, not the full 178-page local test split.

## Intended Use

This model is intended as the segmentation stage of the project pipeline:

```text
manuscript page
-> U-Net line segmentation
-> line bboxes/polygons/reading order
-> line crops
-> HTR transcription
-> NLP module
```

It is not intended as a standalone OCR/HTR model. It does not transcribe text.
