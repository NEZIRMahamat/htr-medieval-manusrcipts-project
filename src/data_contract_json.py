from __future__ import annotations

DATA_CONTRACT_NAME = "catmus_medieval_segmentation_lines"
DATA_CONTRACT_VERSION = "1.0.0"


SEGMENTATION_DATA_CONTRACT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.org/schemas/catmus-medieval-segmentation-lines.schema.json",
    "title": "CATMuS Medieval Segmentation Lines Contract",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_name",
        "contract_version",
        "dataset",
        "generated_at",
        "source",
        "pipeline",
        "summary",
        "pages",
    ],
    "properties": {
        "schema_name": {"const": DATA_CONTRACT_NAME},
        "contract_version": {"type": "string"},
        "dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "task", "local_path", "split"],
            "properties": {
                "name": {"type": "string"},
                "task": {"type": "string"},
                "local_path": {"type": "string"},
                "split": {"type": "string"},
                "max_examples": {"type": ["integer", "null"], "minimum": 1},
            },
        },
        "generated_at": {"type": "string"},
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["predictions_json"],
            "properties": {
                "predictions_json": {"type": "string"},
                "test_set_sha256": {"type": ["string", "null"]},
                "test_set_manifest": {"type": ["string", "null"]},
            },
        },
        "pipeline": {
            "type": "object",
            "additionalProperties": False,
            "required": ["stage", "model", "postprocessing", "crop_generation"],
            "properties": {
                "stage": {"const": "segmentation"},
                "model": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["architecture", "checkpoint_path", "input_size"],
                    "properties": {
                        "architecture": {"type": "string"},
                        "checkpoint_path": {"type": "string"},
                        "input_size": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["height", "width"],
                            "properties": {
                                "height": {"type": "integer", "minimum": 1},
                                "width": {"type": "integer", "minimum": 1},
                            },
                        },
                    },
                },
                "postprocessing": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "threshold", "min_area_model_pixels", "border_margin_model_pixels"],
                    "properties": {
                        "name": {"type": "string"},
                        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "min_area_model_pixels": {"type": "integer", "minimum": 0},
                        "border_margin_model_pixels": {"type": "integer", "minimum": 0},
                    },
                },
                "crop_generation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["crop_margin_pixels", "output_crops_root", "total_crops"],
                    "properties": {
                        "crop_margin_pixels": {"type": "integer", "minimum": 0},
                        "output_crops_root": {"type": "string"},
                        "total_crops": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["page_count", "line_count", "mean_metrics"],
            "properties": {
                "page_count": {"type": "integer", "minimum": 0},
                "line_count": {"type": "integer", "minimum": 0},
                "mean_metrics": {"$ref": "#/$defs/metrics"},
            },
        },
        "pages": {
            "type": "array",
            "items": {"$ref": "#/$defs/page"},
        },
    },
    "$defs": {
        "bbox": {
            "type": "array",
            "prefixItems": [
                {"type": "integer"},
                {"type": "integer"},
                {"type": "integer"},
                {"type": "integer"},
            ],
            "items": False,
            "minItems": 4,
            "maxItems": 4,
        },
        "point": {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "integer"}],
            "items": False,
            "minItems": 2,
            "maxItems": 2,
        },
        "polygon": {
            "type": "array",
            "items": {"$ref": "#/$defs/point"},
            "minItems": 4,
        },
        "metrics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["dice", "iou", "precision", "recall", "f1"],
            "properties": {
                "dice": {"type": "number", "minimum": 0, "maximum": 1},
                "iou": {"type": "number", "minimum": 0, "maximum": 1},
                "precision": {"type": "number", "minimum": 0, "maximum": 1},
                "recall": {"type": "number", "minimum": 0, "maximum": 1},
                "f1": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "page": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "page_id",
                "image_name",
                "split",
                "split_index",
                "width",
                "height",
                "metrics",
                "line_count",
                "lines",
            ],
            "properties": {
                "page_id": {"type": "string"},
                "image_name": {"type": "string"},
                "split": {"type": "string"},
                "split_index": {"type": "integer", "minimum": 0},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "metrics": {"$ref": "#/$defs/metrics"},
                "line_count": {"type": "integer", "minimum": 0},
                "lines": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/line"},
                },
            },
        },
        "line": {
            "type": "object",
            "additionalProperties": False,
            "required": ["line_id", "reading_order", "geometry", "segmentation", "crop"],
            "properties": {
                "line_id": {"type": "string"},
                "reading_order": {"type": "integer", "minimum": 1},
                "geometry": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["bbox", "polygon"],
                    "properties": {
                        "bbox": {"$ref": "#/$defs/bbox"},
                        "polygon": {"$ref": "#/$defs/polygon"},
                    },
                },
                "segmentation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["score", "area_model_pixels", "source_component_bbox"],
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "area_model_pixels": {"type": "integer", "minimum": 0},
                        "source_component_bbox": {"$ref": "#/$defs/bbox"},
                    },
                },
                "crop": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "bbox", "width", "height"],
                    "properties": {
                        "path": {"type": "string"},
                        "bbox": {"$ref": "#/$defs/bbox"},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
    },
}
