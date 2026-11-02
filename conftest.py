"""根级 pytest 收集配置：vendor/ 下的第三方子模块（X-AnyLabeling 等）自带测试树，
跑在各自 venv 里，不进平台收集面（其依赖平台 venv 没有，收集即报错）。"""

collect_ignore = ["vendor"]
