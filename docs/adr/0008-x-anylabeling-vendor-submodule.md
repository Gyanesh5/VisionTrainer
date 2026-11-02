# 0008 - X-AnyLabeling 以 vendor 子模块入库

Status: accepted（2026-11-02）

X-AnyLabeling 以 git submodule 固定在 `vendor/X-AnyLabeling`（pin v4.0.5，commit 1b6750dd）。动机：平台对 X-AnyLabeling 的依赖已从「功能调起」升级为「契约依赖」——标注 JSON 的 `checked` 字段语义（[ADR 0003](0003-review-state-in-annotation-json.md)）、审查/自动修复写回（ADR 0005）、调起入口（`training/xal_open_entry.py`）都依赖其版本行为；游离克隆不进版本账目，新机部署不可复现、升级无据可查。子模块后：`git clone --recurse-submodules` 即得配套版本，升级=父仓库提交新 gitlink，行为变化有 commit 可溯。

与 [ADR 0002](0002-tech-stack-and-repo.md) 的边界不冲突且互补：0002 裁定**平台不进 XAL 上游**（本仓库不被 XAL 吞并）；本 ADR 是反向——XAL 作为第三方副本（layout.md `vendor/` 语义）进平台仓库，上游同步方向不变（仍从 CVHub520 拉取）。venv 仍在子模块内（`.venv` 由 XAL 自身 .gitignore 忽略，不入库），`config.toml` 的 `training.python` 指向 `<repo>/vendor/X-AnyLabeling/.venv/bin/python`——「平台与训练环境分离」原则不变。

## 升级流程

1. `vendor/X-AnyLabeling` 内 `git fetch && git checkout <新 tag>`；
2. 按其 pyproject 同步 venv（uv，缓存命中时分钟级）；
3. 验证：`checked` 字段往返（审查页↔XAL 图标）、自动修复写回、「在 X-AnyLabeling 中打开」调起链路；
4. 父仓库提交新 gitlink，提交信息注明版本与验证结论。

## Considered Options

- git subtree：免 submodule 的 clone 体验，但上游历史混入父仓库、升级合并冲突面大，单人维护不划算，拒绝。
- 继续游离克隆 + 文档记版本号：零改动，但版本账目靠人记，新机部署断链，拒绝（即本 ADR 动机）。
