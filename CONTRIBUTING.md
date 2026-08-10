# 参与开发

本仓库采用“功能分支 + Pull Request”协作方式。请勿直接向 `main` 推送未验证代码。

## 开始开发

```powershell
git switch main
git pull origin main
git switch -c feature/简短任务名

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

`.env` 只保存在本机。不得提交 API Key、SQLite 数据库、原始录音、真实回答、模型权重或未脱敏的参与者数据。

## 模块边界

- `memory_lantern/assessment/`：题库、回答清洗、安全判断、语义评分、语音特征和报告；
- `memory_lantern/drawing/`：绘图特征与机器辅助评分；
- `memory_lantern/knowledge/`：健康知识库和 RAG；
- `memory_lantern/games/`：认知小游戏；
- `memory_lantern/integrations/`：腾讯云等第三方服务；
- `memory_lantern/persistence/`：SQLite 表结构和数据访问；
- `memory_lantern/core/`：国际化与研究展示等公共能力；
- `app.py`：Flask 路由和模块编排，不在其中重复实现底层算法。

新增逻辑时应放入对应模块，并在 `tests/` 增加测试。不要重新把业务模块放回仓库根目录。

## 提交前检查

```powershell
python -m pytest -c tests/pytest.ini -q
git diff --check
git status
```

提交信息建议使用：`feat:`、`fix:`、`test:`、`docs:`、`refactor:` 或 `style:`。
