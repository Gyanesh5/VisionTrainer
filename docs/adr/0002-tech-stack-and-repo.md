# 0002 - 技术栈定案与独立仓库

Status: accepted（2026-09-06，源自 [08 号票](../../.scratch/visiontrainer-build/issues/08-tech-stack-skeleton.md)）

平台以**独立仓库**承载（不进 X-AnyLabeling 上游仓库——上游需持续同步 CVHub520 更新，混入异构大项目会让同步痛苦）。技术栈：前端 **NiceGUI**（Python 全栈、无 Node 构建链，锁 3.x 小版本，UI 薄层化——业务数据全走框架无关 REST/WS 契约，将来可换 Vue3+Element Plus 重写 UI 而后端不动）；后端 FastAPI + **SQLite + SQLAlchemy 2.0** + 进程内 asyncio 单飞队列（不引入 Celery/Redis）；配置 TOML（`config.toml` 密钥不进 git）；**平台与训练环境分离**——平台 venv 无 torch，训练子进程指向外部配置的 python 路径（现有 X-AnyLabeling/.venv）。Python 3.12，uv 管理依赖。界面中文，一期不做 i18n。

## Considered Options

- 前端 Vue3+Element Plus（Liu Studio 同款）：观感上限更高，但双语言+构建链的维护成本对单人项目不划算，列为后备——契约层保证可迁移。
- 平台/训练同 venv：省一个环境，但 torch 体积巨大且平台重启/依赖变更会波及训练环境，拒绝。
- X-AnyLabeling 仓库子目录：见上，拒绝。
