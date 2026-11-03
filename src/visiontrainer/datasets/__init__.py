"""数据集管线（05 号票契约）：投影与格式适配、校验、切分清单、派生结构。"""

from .formats import YoloAdapter, adapter_for, yolo_lines
from .projection import conserved, project_file, project_json, shape_matches_task
from .service import (
    DatasetError,
    delete_dataset,
    get_dataset,
    import_dataset,
    list_datasets,
    register_external,
    rename_classes,
    resplit_dataset,
    validate_dataset,
)

__all__ = [
    "DatasetError",
    "YoloAdapter",
    "adapter_for",
    "conserved",
    "delete_dataset",
    "get_dataset",
    "import_dataset",
    "list_datasets",
    "project_file",
    "project_json",
    "register_external",
    "rename_classes",
    "resplit_dataset",
    "shape_matches_task",
    "validate_dataset",
    "yolo_lines",
]
