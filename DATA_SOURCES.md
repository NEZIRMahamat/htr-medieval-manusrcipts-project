# CATMuS Segmentation Data Notes

Dataset: `CATMuS/medieval-segmentation`
Config/subset: `default`
Task: page-level line segmentation for medieval manuscript pages.

Local copy used in this project:

```text
data/segment_data
```

Source references used for this audit:

- Hugging Face dataset metadata: `https://datasets-server.huggingface.co/info?dataset=CATMuS/medieval-segmentation`
- Hugging Face first rows: `https://datasets-server.huggingface.co/first-rows?dataset=CATMuS/medieval-segmentation&config=default&split=train`

## Splits and Counts

Counts from the local HuggingFace `DatasetDict` saved in `data/segment_data`.

| Split           |           Pages |
| --------------- | --------------: |
| `train`       |           1,336 |
| `validation`  |             191 |
| `test`        |             178 |
| **Total** | **1,705** |

## Fields

Image field: `image`

Page metadata fields:

- `width`
- `height`
- `shelfmark`
- `century`
- `project`

Annotation field: `objects`

Inside `objects`, the useful annotation fields are:

- `id`
- `bbox`
- `polygons`
- `category`
- `type`
- `parent`
- `baseline`

For the segmentation part, the model input is the full page image in `image`.
The target mask is generated from `objects.polygons`, keeping only objects where `objects.type == "line"`.

## Feature Schema

```json
{
  "image": {
    "_type": "Image"
  },
  "width": {
    "dtype": "int64",
    "_type": "Value"
  },
  "height": {
    "dtype": "int64",
    "_type": "Value"
  },
  "objects": {
    "id": ["string"],
    "bbox": [["int64"]],
    "polygons": [["int64"]],
    "category": ["string"],
    "type": ["string"],
    "parent": ["string"],
    "baseline": [["int64"]]
  },
  "shelfmark": {
    "dtype": "string",
    "_type": "Value"
  },
  "century": {
    "dtype": "int64",
    "_type": "Value"
  },
  "project": {
    "dtype": "string",
    "_type": "Value"
  }
}
```

## Test Set Sealing

The local `test` split was sealed with SHA-256 before final reporting.

Files:

- Manifest: `outputs/splits/test_set_segmentation_manifest.json`
- Global SHA-256: `outputs/splits/test_set_segmentation_sha256.txt`

Current global SHA-256:

```text
07e3e5a02b09c57e8df42be272b51931bab017a6bec686aeea03e1157cc92c80
```

## Report Notes

- `CATMuS/medieval-segmentation` is already page-level segmentation data.
- The dataset provides line polygons and baselines, so line masks can be generated directly from the annotations.
- The project uses binary segmentation: `0 = background`, `1 = text line`.
- The official splits are kept unchanged.
- The final reported test metrics use 60 pages from the sealed test split because of local CPU/RAM constraints.
- The full local test split remains sealed and should not be used for hyperparameter tuning.
