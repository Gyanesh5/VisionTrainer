"""数据集管线（05 号票契约）：投影与格式适配、校验、切分清单、派生结构。

v1.4 票 03：托管数据集唯一出生入口为 `create_dataset`（三态来源），旧「导入」入口
`import_dataset` 已按硬切裁决删除，不留兼容层。
v1.4 票 04：加图 `add_images`、类别表四动作（追加/改名/重排/删除）与改标回流对账
`reconcile_on_open` 落地；改名是平台第三条标注写路径（ADR 0009，备份 + 原子写 + 审查继承）。
"""

from .formats import YoloAdapter, adapter_for
from .projection import conserved, project_file, project_json, shape_matches_task
from .service import (
    DatasetError,
    add_images,
    append_classes,
    create_dataset,
    delete_class,
    delete_dataset,
    detect_rename_suspects,
    get_dataset,
    list_datasets,
    register_external,
    reconcile_on_open,
    rename_classes,
    reorder_classes,
    rollback_classes,
    validate_dataset,
)

__all__ = [
    "DatasetError",
    "YoloAdapter",
    "adapter_for",
    "add_images",
    "append_classes",
    "conserved",
    "create_dataset",
    "delete_class",
    "delete_dataset",
    "detect_rename_suspects",
    "get_dataset",
    "list_datasets",
    "project_file",
    "project_json",
    "reconcile_on_open",
    "register_external",
    "rename_classes",
    "reorder_classes",
    "rollback_classes",
    "shape_matches_task",
    "validate_dataset",
]
