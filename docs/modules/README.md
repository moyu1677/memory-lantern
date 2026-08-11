# 语音陪伴与聊天关怀报告：模块交接说明

这两个目录保存了本次同步的可公开代码、部署说明、冻结案例和验证记录。它们不包含 UI、模板、样式、MoCA、真实录音、数据库、`.env` 或模型调用的原始对话记录。

## 语音陪伴

链路为：老人语音 → `asr_service.py` → AI 回复 → 受限语气规划 → `voice_service.py` → 已授权家人音色的 IndexTTS2 WAV。

- 应用侧客户端：`asr_service.py`、`voice_service.py`
- GPU 服务：`remote_asr/`、`remote_tts/`
- 应用环境变量：根目录 `.env.example` 中的 `ASR_*`、`INDEXTTS2_*`
- 部署与验证说明：[`voice-companion/`](voice-companion/)

浏览器不应直接访问远端 ASR 或 TTS；应用服务应通过 SSH 隧道或受控内网转发请求。录入家人音色前必须取得明确授权，撤回授权时应同时删除远端音色。

## 聊天关怀报告

`care_report_engine.py` 仅取老人端回合，要求模型返回固定 JSON，再拒绝诊断性措辞、聊天原话复述、不可验证来源、越界证据和与领域不相符的行动建议。输出给家人的是克制的关怀提示，不是医疗结论。

- 提取与规范化：`care_report_engine.py`
- 公开冻结案例和真实 API 评测脚本：`tools/care-report-benchmarks/`
- 最新公开汇总：`validation/care_report_validation_latest.json`
- 设计、数据边界与验证记录：[`care-report/`](care-report/)

## 与共享应用的接入边界

本次同步没有替换远端已更新的 `app.py`、`database.py`、`templates/` 或 `static/`。接入方需要在共享应用中单独完成：受权人检查、CSRF 与文件验证、家人音色的数据库引用及撤回、仅向已绑定家属展示去原话后的报告，以及管理员可见的验证汇总。

历史模块文档中出现的 `tests/benchmarks/` 路径，在本仓库中对应 `tools/care-report-benchmarks/`；其余路径以本文件列出的实际位置为准。
