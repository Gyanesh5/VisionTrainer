"""数据集管线（05 号票契约）：投影与格式适配、校验、切分清单、派生结构。

v1.4 票 03：托管数据集唯一出生入口为 `create_dataset`（三态来源），旧「导入」入口
`import_dataset` 已按硬切裁决删除，不留兼容层。
"""

from .formats import YoloAdapter, adapter_for, yolo_lines
from .projection import conserved, project_file, project_json, shape_matches_task
from .service import (
    DatasetError,
    create_dataset,
    delete_dataset,
    get_dataset,
    list_datasets,
    register_external,
    rename_classes,
    validate_dataset,
)

__all__ = [
    "DatasetError",
    "YoloAdapter",
    "adapter_for",
    "conserved",
    "create_dataset",
    "delete_dataset",
    "get_dataset",
    "list_datasets",
    "project_file",
    "project_json",
    "register_external",
    "rename_classes",
    "shape_matches_task",
    "validate_dataset",
    "yolo_lines",
]
