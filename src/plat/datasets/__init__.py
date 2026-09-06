"""数据集管线（05 号票契约）：格式转换、校验、切分清单、派生结构。"""

from .convert import json_to_yolo_lines, shape_matches_task
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
    "delete_dataset",
    "get_dataset",
    "import_dataset",
    "json_to_yolo_lines",
    "list_datasets",
    "register_external",
    "rename_classes",
    "resplit_dataset",
    "shape_matches_task",
    "validate_dataset",
]
