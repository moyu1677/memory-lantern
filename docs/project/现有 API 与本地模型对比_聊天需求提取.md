# 聊天需求提取：现有 API 与本地模型对比（当前定案）

> 更新日期：2026-08-09
> 范围：只讨论“从老人—AI 语音聊天的转写文本中提取关怀信号”，不包含 MoCA，也不把聊天功能用于疾病或心理诊断。

## 结论

**当前版本应优先使用项目已经接入的 `deepseek-v4-flash` API，不部署本地大模型作为主链路。**

这不是因为“云端一定比开源好”，而是由这个项目的实际条件决定的：

1. 现有 Flask 代码已经能够调用该 API；
2. 官方资料显示它原生支持 1M context、JSON 输出和思考模式；
3. 本机只有 RTX 4070 Laptop（8GB 显存）。Qwen3-8B 虽然可量化运行，但不能把长上下文稳定地当作可交付能力；
4. 需求提取最怕的是跨会话遗漏、忽略后来的否认/好转，以及含蓄表达。长上下文和后台推理比“本地免费”更直接影响这个任务；
5. Qwen3-8B 留作隐私优先、断网演示或后续对照实验的备选，而不是增加本阶段工作量的主服务。

这里的“选择 API”不等于让 API 诊断老人：它仍只能生成有来源、可拒收、信息不足时保留的**关怀信号**。

## 1. 当前项目实际上有什么可用模型

### 1.1 已有聊天 API：DeepSeek

代码实际采用 OpenAI Python SDK 调用 DeepSeek 的 Chat Completions 接口：

| 位置 | 已核实内容 |
|---|---|
| `忆路明灯/app.py` | 默认 `DEEPSEEK_BASE_URL=https://api.deepseek.com`，默认 `DEEPSEEK_MODEL=deepseek-v4-flash`，60 秒超时 |
| `忆路明灯/app.py` 的 `get_ai_reply` | 所有普通聊天通过该客户端调用；当前显式 `thinking: disabled` |
| `忆路明灯/needs_engine.py` 的 `extract_session_card` | 已使用同一客户端、同一模型和 `response_format={"type": "json_object"}` 做 JSON 卡片提取；当前逻辑仍是“单次会话、老人确认分享后”的旧路径 |
| `cross_language_voice_clone/src/chat/llm_engine.py` | 另一个语音克隆演示也使用 OpenAI-compatible DeepSeek 调用，默认是 `deepseek-chat` |

仓库中没有发现第二家已配置的 LLM API。当前工作环境未发现 `.env` 或 `DEEPSEEK_MODEL` 环境变量，因此本文核对的是**代码默认值和官方能力**，没有读取密钥、没有向 API 发送真实聊天内容。

### 1.2 可本地部署备选：Qwen3-8B

`Qwen/Qwen3-8B` 是公开 Apache-2.0 权重，可由 vLLM、SGLang、Ollama、llama.cpp 等提供 OpenAI-compatible 接口。它不需要把对话发给第三方，但本项目的当前电脑只有 8GB 显存。

官方模型卡标注其原生 context 为 32,768 tokens；YaRN 可扩到 131,072 tokens。扩展 context 不等于本机稳定可承受：上下文越长，KV cache 占用的显存越多。对 8GB 显存而言，4-bit 量化可以做短上下文原型，但不能承诺数周聊天历史的稳定推理。

## 2. 针对本项目能力的直接比较

| 项目真正需要的能力 | DeepSeek-V4-Flash（已有 API） | Qwen3-8B（本地备选） | 结论 |
|---|---|---|---|
| 接入成本 | 已有调用链；只需新增报告专用提示词与校验 | 需下载权重、量化、启动推理服务、排查显存与并发 | API 明显更省时间 |
| 长期、多会话阅读 | 官方标注 1M context | 32K 原生；131K 需 YaRN，且受本机显存限制 | API 明显更适合报告级输入 |
| JSON 结构化输出 | 官方支持 JSON Output；现有代码已使用 `json_object` | 可提示 JSON，也可用推理框架做约束解码 | 都能做；API 已接好 |
| 后台推理 | 支持 thinking，默认 high，可只用于报告 | 同样支持 thinking 模式，但会进一步占用本机资源/等待时间 | API 更适合“后台慢一点、结果稳一点” |
| 老人实时聊天 | thinking 可关闭，现有代码就是这样做的 | 可本地低延迟，但需实际部署后验证 | 当前 API 已满足；本地无明显收益 |
| 隐私与断网 | 对话文本会发给 API；必须明确授权、最小化上传和保存期限 | 对话可完全留在本机 | 本地唯一明确优势；若隐私成为硬性限制再切换 |
| 直接经济成本 | 官方当前价：缓存未命中输入 $0.14 / 1M tokens，输出 $0.28 / 1M tokens；价格可能调整 | 权重免费，但有设备、电力、部署与维护成本 | 低频“每日/每周报告”通常 API 成本很低 |
| 本项目的语义准确性 | 尚未用同一测试集实测 | 尚未用同一测试集实测 | **不能仅凭宣传资料声称任一方一定更懂老人** |

官方资料：

- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)：`deepseek-v4-flash`、1M context、JSON Output、thinking、价格和并发。
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode)：要求 `response_format={"type":"json_object"}`，提示词中包含 JSON 和示例；官方也说明偶发空内容，因此程序必须校验并重试。
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)：thinking 默认开启，可通过 `extra_body={"thinking":{"type":"enabled/disabled"}}` 控制。
- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)：8.2B 参数、32K 原生 context、YaRN 131K 和本地部署方式。
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)：Apache-2.0 开放权重许可。

## 3. 为什么这不是“只比参数”

本项目需要的不是一句话分类，而是：先记录会话中已出现的事实线索，再在报告时判断它们是否反复出现、有没有更新/否定，以及是否不足以下结论。这个任务的关键是“把足够多的上下文带进来，还不把普通聊天病理化”。

因此，DeepSeek-V4-Flash 的 1M context 和后台 thinking 是**已核实的功能优势**，足以让它成为当前更合适的工程选择；但它们不是“真实需求提取准确率”的证明。最终语义质量仍要在同一个测试协议下同本地候选比较。

## 4. 推荐的最小架构

```text
老人语音 → ASR 转写
                 ↓
每次会话：DeepSeek-V4-Flash，thinking 关闭
  输出：只含事实型信号、时间、否定/好转标记、内部来源轮次
                 ↓
报告生成：DeepSeek-V4-Flash，thinking 开启
  输入：近期原文 + 早期会话级信号/趋势
  输出：最多 1–3 条关怀提示或“信息不足/彼此矛盾”
                 ↓
程序校验：JSON、来源轮次、禁止诊断词、矛盾/无信号处理
                 ↓
家人看到简短提示和建议，不展示老人原话
```

这里没有每句向量化，也不需要本地保存无限长的逐字录音。较早会话可保留结构化信号；但报告生成必须重新读到相关的近期原文，避免“上次摘要漏掉一条就永远漏掉”。

## 5. 现有代码的一个重要观察

当前 `needs_engine.py` 的卡片路径与现在的产品方向不完全一致：它只在老人明确同意分享后，提取一张带逐字 `evidence` 的单次会话卡片。这是现有原型的隐私保护设计，不是我们现在要做的“跨会话、潜在关怀信号报告”。

本次没有修改它。后续实现时，应该新建“报告级提取”而不是强行扩大旧卡片函数：

- 不向家人展示老人原话；
- 不要求老人每次确认一张卡；
- 内部保留来源轮次，仅用于程序审计；
- 缺乏证据或出现矛盾时输出保留，不让模型补全故事；
- 聊天回复保持低延迟，报告生成才启用 thinking。

## 6. 如何把“API 比本地好很多”说成可验证结论

目前可以严谨地说：API 在**工程可用性和可用上下文长度**上明显更适合当前项目；不能严谨地说：API 已经被证明在“老人潜在需求”上绝对更准确。

要完成最后一层比较，只需要同一份小型固定测试集，不需要人工给真实老人聊天打标签：

| 测试 | 比较方式 | 指标 |
|---|---|---|
| 长上下文 | DeepSeek-V4-Flash 与本地 Qwen3-8B 同时跑 LongBench 中文子集及项目长度变体 | task score；在不同长度上的通过率 |
| 含蓄/个性化理解 | 同样跑 PrefEval `implicit persona` classification 子集 | accuracy |
| 本项目安全行为 | 相同中文合成对话：无信号、矛盾、好转、无关闲聊插入、明确症状词 | 无信号误报率、矛盾保留率、关键线索通过率、诊断越界率 |
| 系统稳定性 | 相同 JSON Schema、温度 0，重复调用 | 可解析率、来源轮次完整率、空响应/重试率、端到端耗时 |

如果 Flash 在这套对比中不弱于 Qwen3-8B，就没有理由为了“免费”牺牲当前机器无法稳定支持的长上下文；若 Qwen3-8B 明显更好且团队决定“对话绝不出本机”，再选择本地部署。这才是可对老师负责的结论。

## 7. 当前决策与不做的事

**现在决定：**

- 主模型：现有 `deepseek-v4-flash` API；
- 老人聊天：thinking 关闭，短、自然、低等待；
- 家庭报告：后台 thinking 开启，JSON + 程序校验；
- 本地 Qwen3-8B：只作为对照/隐私备选，不加入当前主链路；
- 不训练专用“老人需求模型”，不调用模型诊断疾病。

**这次不做：**

- 不读取或暴露任何 API 密钥；
- 不调用真实 API 消耗额度；
- 不下载或部署本地权重；
- 不修改现有代码。
