# router-relay

一个带智能路由的 OpenAI 兼容 API 中转，灵感来自 OpenSquilla 的
`SquillaRouter` + Ensemble。opencode（或任意 OpenAI 兼容客户端）指向本服务，
relay 把请求转发给上游 OpenAI 兼容 provider（OpenAI / OpenRouter /
marketingforce / …），并可选地**按 turn 难度路由到最省的模型**、和/或**对复杂
turn 融合多个模型**。

当前状态：
- **P0** 透明透传 —— 完成 ✅
- **P1** 规则路由（特征 → `c0..c3` 档位 → 模型覆盖）—— 完成 ✅
- **P2** B5 ensemble 融合（并行 proposer → aggregator LLM）—— 完成 ✅
- **P3 前置** 按日期分文件 JSONL 捕获 + outcome sidecar + 抱怨回溯 + 离线标签回填 —— 完成 ✅
- **P3** LightGBM 训练闭环 —— 待做（需要先攒真实流量 + outcome 数据）

## 目录结构

```
router-relay/
├── pyproject.toml
├── .env.example              # 配置模板（复制成 .env）
├── scripts/
│   ├── realign_labels.py     # 离线标签回填（decisions + outcomes + judge → labeled）
│   ├── judge_labels.py       # LLM-as-judge 绝对难度标注（user msg → optimal_tier）
│   └── train_p3.py           # LightGBM 训练（labeled JSONL → p3_lightgbm.txt）
└── src/relay/
    ├── __init__.py
    ├── __main__.py           # python -m relay  /  router-relay
    ├── app.py                 # FastAPI 应用 + 路由 + outcome 捕获
    ├── auth.py                # Bearer 鉴权依赖
    ├── capture.py             # P3 前置：decisions + outcomes 双文件 JSONL
    ├── config.py              # env 驱动配置（pydantic-settings）
    ├── ensemble.py            # P2：B5 融合（proposer → aggregator）
    ├── errors.py              # RelayError → OpenAI 形状错误信封
    ├── upstream.py            # httpx 客户端 + SSE 透传
    └── router/
        ├── __init__.py
        ├── features.py        # handcrafted 特征提取 + 抱怨检测
        ├── scorer.py          # 规则打分 → 档位 + 置信度
        ├── policy.py          # confidence_gate → complaint_upgrade → large_context_floor → sticky
        ├── tiers.py           # 档位 → 模型映射（marketingforce preset）
        └── runtime.py         # RoutingDecision、历史、有界 apply_router、_derive_source
```

## 1. 前置

Python ≥ 3.10，`uv`（推荐）或 pip。

## 2. 安装

```sh
cd E:\PY_CODE\router-relay
uv sync                       # 建 .venv 并装依赖
#  — 或 —
python -m venv .venv && .venv/Scripts/activate && pip install -e .
```

## 3. 配置（`.env`）

```sh
cp .env.example .env          # 然后编辑 .env
```

至少配这些（上游块是 OpenAI 兼容，OpenAI / OpenRouter / marketingforce 都行）：

```ini
# 入站鉴权（客户端用 Authorization: Bearer <token> 发送）
RELAY_API_KEYS=<你的 relay token>

# 上游 OpenAI 兼容 provider
UPSTREAM_BASE_URL=https://api.openai.com/v1
UPSTREAM_API_KEY=<你的上游 key>

# 服务
LISTEN_HOST=127.0.0.1
LISTEN_PORT=8787
```

全部字段见下方**配置参考**，含路由 / ensemble / 捕获的功能开关（默认全关 =
纯透传）。

## 4. 启动

```sh
cd E:\PY_CODE\router-relay
uv run router-relay
#  — 或 —
uv run python -m relay
#  — 或 —
uv run uvicorn relay.app:app --port 8787
```

服务监听 `http://127.0.0.1:8787`，`Ctrl+C` 停止。

## 5. 连接 opencode

把下面放进 `opencode.json`（项目根或 `~/.config/opencode/opencode.json`）。
`apiKey` 填你的 `RELAY_API_KEYS` 值；`models` 的 key **必须和上游接受的 id 一致**
（下面是 MAAS 示例）。

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "relay/qwen3-max",
  "provider": {
    "relay": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Router Relay (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        "apiKey": "<你的 relay token>"
      },
      "models": {
        "qwen3-max": { "name": "Qwen3 Max" },
        "gpt-5.4": { "name": "GPT-5.4" },
        "claude-opus-4-8": { "name": "Claude Opus 4.8" },
        "deepseek-r1": { "name": "DeepSeek R1" }
      }
    }
  }
}
```

然后 `opencode` 里选 `relay/...` 模型即可。

> 开了 `ROUTER_ENABLED=true` 后，opencode 里选哪个 model 只是个入口——
> **真正模型由路由按 turn 难度决定**（简单 → `claude-3-5-haiku`，复杂 →
> `claude-sonnet-4.5` / `claude-opus-4-8`）。

## 5b. 连接 ZCode（Anthropic 端点）

ZCode 用 **Anthropic** API 格式（`/v1/messages`），不是 OpenAI。relay 已加
`/v1/messages` 端点（透传 + 路由；ensemble 在该路径跳过，因为它调
`/chat/completions`）。DEFAULT_TIERS 也改成**双端点**模型（claude-3-5-haiku /
qwen3.7-plus / claude-sonnet-4.5 / claude-opus-4-8——marketingforce 上同时支持
openai+anthropic），所以同一套 preset 同时适配 opencode(OpenAI) 和 ZCode(Anthropic)。

ZCode 的 provider 配置在 `~/.zcode/v2/config.json` 的 `provider` 对象里。加一条：

```json
"router-relay": {
  "name": "Router Relay (local)",
  "kind": "anthropic",
  "options": {
    "apiKey": "<你 .env 里的 RELAY_API_KEYS 值>",
    "baseURL": "http://127.0.0.1:8787/v1",
    "apiKeyRequired": true
  },
  "enabled": true,
  "source": "custom",
  "models": {
    "GLM-5.2": {
      "limit": {"context": 200000},
      "modalities": {"input": ["text"], "output": ["text"]}
    }
  }
}
```

> 我已帮你加好这条（备份在 `~/.zcode/v2/config.json.bak`）。ZCode 选的 `GLM-5.2`
> 只是入口，relay 会按难度路由覆盖。

步骤：
1. **重启 relay**（你 8787 上跑的是加 `/v1/messages` 之前的旧代码，必须重启）：
   `cd E:\PY_CODE\router-relay && uv run router-relay`
2. **重启 ZCode**（读 config.json）→ Settings → 模型 provider → 选 **Router Relay (local)**。
3. 先用 `ROUTER_OBSERVE_ONLY=true` 跑一阵（ZCode 仍走 GLM-5.2，relay 记录决策），
   验证 `/v1/router/decisions` 合理后再切 `ROUTER_OBSERVE_ONLY=false` 让路由覆盖。
4. 回退：在 Settings 切回原 provider 即可；或还原 `cp ~/.zcode/v2/config.json.bak ~/.zcode/v2/config.json`。

## 6. 自检（curl，不依赖 opencode）

```sh
# 活性（无需鉴权）
curl http://127.0.0.1:8787/healthz
# {"status":"ok"}

# 鉴权门（不带 token 应返回 401）
curl http://127.0.0.1:8787/v1/models

# 透传上游模型列表
curl -H "Authorization: Bearer <你的 relay token>" http://127.0.0.1:8787/v1/models

# 非流式对话
curl -H "Authorization: Bearer <你的 relay token>" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-max","messages":[{"role":"user","content":"hi"}]}' \
  http://127.0.0.1:8787/v1/chat/completions

# 流式对话（SSE）
curl -N -H "Authorization: Bearer <你的 relay token>" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-max","stream":true,"messages":[{"role":"user","content":"count to 3"}]}' \
  http://127.0.0.1:8787/v1/chat/completions

# 查路由决策（需开 ROUTER_ENABLED=true 才有数据）
curl -H "Authorization: Bearer <你的 relay token>" \
  "http://127.0.0.1:8787/v1/router/decisions?limit=10"
```

## 7. 推荐上线路径

1. **先 observe + 捕获**——不改模型行为，验证规则打分，同时开始攒 P3 训练数据：
   ```ini
   ROUTER_ENABLED=true
   ROUTER_OBSERVE_ONLY=true      # 只记录决策、不覆盖模型
   ROUTER_CAPTURE_DIR=./logs    # 每 turn 落一条样本
   ENSEMBLE_ENABLED=false
   ```
   正常用 opencode。决策进 `/v1/router/decisions` 和
   `logs/router-samples-YYYY-MM-DD.jsonl`；客户端发的模型原样转发。
2. **看几天决策对不对**——简单 turn 是不是 → `c0`，复杂 turn 是不是 → `c2/c3`？
   不对就调 `ROUTER_CONFIDENCE_THRESHOLD` 或 `ROUTER_TIERS`。
3. **切 route 模式**——`ROUTER_OBSERVE_ONLY=false`，路由开始真正覆盖模型。
4. **（可选）开 ensemble**——`ENSEMBLE_ENABLED=true`（注意成本；别用
   `deepseek-r1` 这类推理模型当 proposer）。
5. **P3**——`logs/` 样本够了后，训 LightGBM 在同一 `score_features` 调用点替换规则打分。

## P1 路由

每个请求由确定性规则打分（长度 / 语言 / 代码比例 / 关键词桶 / 上下文规模）
分到 `c0..c3` 档，策略链定档，该档模型覆盖客户端发的模型。打分在 worker
线程里有界执行；超时/异常时透明透传客户端原模型，绝不阻塞请求。

策略链：`confidence_gate`（置信度低 → 升档）→ `complaint_upgrade`（用户抱怨
上一轮答案 → 升档）→ `large_context_floor`（上下文很长 → 抬到 `c2`）→
`sticky`/防降档（避免会话中途在档位间反复横跳）。

| 档位 | 默认模型（双端点 preset） | 用途 |
| --- | --- | --- |
| c0 | claude-3-5-haiku | 便宜/快——简单问答、闲聊 |
| c1 | qwen3.7-plus | 中等 |
| c2 | claude-sonnet-4.5 | 强——工程、设计 |
| c3 | claude-opus-4-8 | 最强 |

四个都同时支持 marketingforce 的 openai + anthropic 端点，所以同一套 preset
同时适配 opencode(`/v1/chat/completions`) 和 ZCode(`/v1/messages`)。只用单路径时可
换更便宜的单端点模型（如 `qwen3-max` 仅 OpenAI）。

覆盖某档（JSON）：`ROUTER_TIERS={"c3":{"model":"claude-opus-4-8"}}`。

sticky 路由的 session key 由**第一条 user 消息**派生，故 per-会话粘性无需
客户端配合（opencode 每轮重发完整历史——无状态 OpenAI 协议）。

## P2 Ensemble

两种模式（`ENSEMBLE_MODE`）：

- **`b5_fusion`**（默认）：复杂 turn（路由档位 ≥ `ENSEMBLE_MIN_TIER`）时，路由 anchor
  模型 + 配置的 proposer **并行**（非流式）出草稿，aggregator LLM 用 `<CANDIDATE N>`
  prompt 把草稿融合成最终答案。适合需要综合多视角的任务。
- **`best_of_n`**：同样的并行 proposer 出草稿，但由 scorer LLM 评估选出**最佳单条**
  草稿直接返回（不再调模型生成新答案）。适合有明确正确答案的任务（debugging、
  事实问答），比 b5_fusion 更快更省。

简单 turn（档位 < `ENSEMBLE_MIN_TIER`）仍走单模型，不触发 ensemble。

```ini
ENSEMBLE_ENABLED=true                    # 需先 ROUTER_ENABLED=true
ENSEMBLE_MODE=b5_fusion                  # 或 best_of_n
ENSEMBLE_PROPOSERS=qwen3.7-plus,glm-5.2  # 路由 anchor(gpt-5.5) 自动加入
ENSEMBLE_AGGREGATOR=gpt-5.5              # 融合模型；空则用路由 anchor
ENSEMBLE_SCORER_MODEL=gpt-5.5            # best_of_n 的 scorer；空则用 aggregator
ENSEMBLE_MIN_TIER=c2                     # 只在复杂档触发
ENSEMBLE_MIN_SUCCESSFUL=2                # 聚合前的 quorum
ENSEMBLE_MAX_PROPOSERS=3                 # 限制并发 proposer 数量（成本控制）
```

行为：
- **Quorum**：成功 proposer 少于 `ENSEMBLE_MIN_SUCCESSFUL` → 在路由 anchor 上
  `fallback_single`。
- **非流式**：JSON 响应，usage 是所有 proposer + aggregator/scorer 求和。
- **流式**：proposer 阶段每 5s 发 SSE 注释帧心跳（`: heartbeat\n\n`）防客户端
  超时；proposer 完成后 b5_fusion 转发 aggregator 的 SSE，best_of_n 合成 SSE
  帧从已选草稿直接发出（不再调模型）。
- **tools**：带 `tools` 的请求跳过 ensemble（P2 不融合 tool-calling）。
- **成本提醒**：推理模型当 proposer 会烧不受 `max_tokens` 限制的推理 token，
  proposer 池建议用非推理模型。`ENSEMBLE_MAX_PROPOSERS` 限制并发数量。

## P3 前置捕获

每 turn 一个训练样本，append 到**按日期分文件**的 JSONL
（`logs/router-samples-YYYY-MM-DD.jsonl`，跨天自动切新文件）。只存聚合特征标量
——**绝不存 prompt 明文**。

上游返回后，再追加一条 outcome 记录到 sidecar 文件
（`logs/router-outcomes-YYYY-MM-DD.jsonl`），按 `decision_id` 关联。

```ini
ROUTER_CAPTURE_DIR=./logs      # 空 = 不捕获
```

decision 行格式：
```json
{"decision_id":"...","ts_ms":...,"session_key":"...","tier":"c2","model":"gpt-5.4",
 "client_model":"qwen3-max","confidence":0.62,"difficulty":0.51,
 "source":"rule_scorer:complaint_upgrade","executed_kind":"single",
 "trail":[{"stage":"complaint_upgrade",...}],
 "feature_snapshot":{"char_len":85,"hard_kw_hits":4,"complaint_detected":false,...},
 "signals":{"len_score":0.01,"code_score":0.0,"kw_score":-0.12,"ctx_score":0.05,...},
 "schema_version":1,"label":null}
```

outcome 行格式（sidecar，同一 `decision_id` 可有多条）：
```json
{"decision_id":"...","ts_ms":...,"schema_version":1,
 "outcome":"success","executed_kind":"single","latency_ms":1234,
 "usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150},
 "finish_reason":"stop","upstream_status":200,"label_hint":"appropriate"}
```

`label` 由离线 realignment 脚本回填。标签信号来源（按优先级）：

1. **抱怨回溯**（`outcome="complaint_followup"`）：turn *i* 检测到抱怨 → 给 turn *i−1*
   写 `label_hint="under_routed"`。这是 OpenSquilla 的 `retrospective_under_routing`，
   relay 在线自动写入 outcome 文件，零额外成本。
2. **上游错误**（`outcome="upstream_error"`）：模型无法服务请求 → `under_routed`。
3. **token 效率启发**（`label_hint`）：强档位但输出 < 50 tokens → `over_routed`；
   弱档位但输出 > 2000 tokens → `under_routed`。

离线合并命令（read-only，不改原始捕获文件）：
```sh
uv run python scripts/realign_labels.py --date 2026-07-13
# → 写出 logs/router-labeled-2026-07-13.jsonl（label + optimal_tier 已填充）
uv run python scripts/realign_labels.py --dry-run   # 只打印分布，不写文件
```

`source` 字段现在按 policy 链细分为：`rule_scorer`（无 policy 触发）、
`rule_scorer:confidence_gate`、`rule_scorer:complaint_upgrade`、
`rule_scorer:large_context_floor`、`rule_scorer:sticky`、`passthrough`
（路由超时/异常时的透传）。

这就是未来 P3 LightGBM 的训练数据底座。

### LLM-as-Judge 绝对难度标注

上面的 `label` 是**相对标签**（基于规则打分器自己的选择 ±1），训出来的模型只是
规则打分器的影子。LLM-as-judge 读用户消息文本，独立判断"这个任务的真实难度是
c几"，产出**绝对标签**。

**Step 1：开启 raw content 捕获**（需要重启 server）：
```ini
CAPTURE_RAW_CONTENT=true      # 存最后一条 user 消息文本到 router-raw-*.jsonl
```
只存最后一条 user 消息（judge 判断难度的主信号），不存完整对话/响应。默认关 =
隐私优先。可标注后删除 raw 文件，judge 标签持久保存在单独文件里。

**Step 2：运行 judge 标注**（离线，调上游 API）：
```sh
uv run python scripts/judge_labels.py --date 2026-07-14
uv run python scripts/judge_labels.py --date 2026-07-14 --limit 50 --delay 1.0  # 成本控制
uv run python scripts/judge_labels.py --dry-run        # 只预览，不调 API
uv run python scripts/judge_labels.py --skip-judged    # 跳过已标注的
```
输出 `router-judge-YYYY-MM-DD.jsonl`，每行一个绝对难度判断：
```json
{"decision_id":"...","optimal_tier":"c2","confidence":0.85,"reason":"multi-file refactor","judge_model":"gpt-5.5"}
```

**Step 3：合并标签**（realign 脚本自动合并 judge + outcome 信号）：
```sh
uv run python scripts/realign_labels.py --date 2026-07-14
```
标签优先级（最强信号优先）：
1. `complaint_followup`（用户明确投诉）→ `under_routed`，optimal ≥ actual+1
2. **judge**（绝对难度）→ 直接用 judge 的 `optimal_tier`
3. `upstream_error`（模型无法服务）→ `under_routed`
4. `label_hint`（token 效率启发式）→ ±1

`label_source` 字段标注每条记录的标签来源，便于分析。

### P3 LightGBM 训练 + 推理

标签就绪后，训练一个 LightGBM 4 分类器（c0..c3），作为 `score_features` 的
drop-in 替换。推理 ~0.1ms（CPU），与规则打分器共享同一 `FeatureBundle` 接口。

**训练**（读所有日期的 `router-labeled-*.jsonl`，过滤 `optimal_tier != null`）：
```sh
uv run python scripts/train_p3.py --auto
# 输出分类报告 + 混淆矩阵 + 特征重要性
# 保存 models/p3_lightgbm.txt + models/p3_lightgbm.meta.json
```

**激活**（`.env` 里设，重启 server）：
```ini
ROUTER_ML_MODEL_PATH=models/p3_lightgbm.txt
```
设置后 `apply_router` 自动用 ML head 替代规则打分器；路径无效或加载失败 →
透明回退到规则打分器（不阻塞请求）。`source` 字段标为 `ml_head`。

**迭代重训**（攒更多数据后）：
```sh
# 1. 对新流量跑 judge + realignment
uv run python scripts/judge_labels.py --date 2026-07-20 --skip-judged
uv run python scripts/realign_labels.py --date 2026-07-20

# 2. 用全部日期数据重新训练（--auto 自动发现所有 router-labeled-*.jsonl）
uv run python scripts/train_p3.py --auto
# → 覆盖 models/p3_lightgbm.txt，重启 server 即生效

# 保留多版本对比：
uv run python scripts/train_p3.py --auto --output models/p3_v2.txt
ROUTER_ML_MODEL_PATH=models/p3_v2.txt uv run router-relay
```

建议每攒 200-300 条新带标签数据重训一次，对比验证集准确率。
数据到 1000+ 条后可关掉 `ROUTER_OBSERVE_ONLY=false` 让 ML head 真正接管路由。

## 配置参考（`.env`）

### 基础

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `RELAY_API_KEYS` | — | 客户端须发的 bearer token，逗号分隔。空 = 开放 relay（仅 dev）。 |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | 上游 OpenAI 兼容 base URL。 |
| `UPSTREAM_API_KEY` | — | 上游 key（`Authorization: Bearer`）。 |
| `UPSTREAM_ORGANIZATION` | — | 可选 `OpenAI-Organization` 头。 |
| `DEFAULT_MODEL` | — | 请求缺 `model` 时的兜底。 |
| `LISTEN_HOST` / `LISTEN_PORT` | `127.0.0.1` / `8787` | 绑定。 |
| `UPSTREAM_TIMEOUT` | `600` | 上游请求超时（秒）。 |
| `LOG_LEVEL` | `info` | uvicorn 日志级别。 |

### P1 路由

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `ROUTER_ENABLED` | `false` | 总开关（关 = 纯透传）。 |
| `ROUTER_OBSERVE_ONLY` | `false` | 只记不覆盖模型（安全 rollout）。 |
| `ROUTER_TIMEOUT_SECONDS` | `2.0` | 打分硬预算；超时 → 透传。 |
| `ROUTER_STICKY_TURNS` | `3` | 每会话保留的近期档位数（防横跳）。 |
| `ROUTER_CONFIDENCE_THRESHOLD` | `0.55` | 置信度低于此 → confidence_gate 升一档。 |
| `ROUTER_LARGE_CONTEXT_CHARS` | `64000` | 上下文超此 → large_context_floor 抬到 c2。 |
| `ROUTER_TIERS` | — | JSON 档→模型覆盖；空 = 内置 preset。 |
| `ROUTER_DECISION_DB` | — | 可选 SQLite 路径，持久化决策。 |
| `ROUTER_CAPTURE_DIR` | — | P3 前置：按日期分文件 JSONL 目录。 |
| `CAPTURE_RAW_CONTENT` | `false` | 存用户消息文本到 `router-raw-*.jsonl`（judge 标注用）。默认关=不存 prompt 明文。 |
| `ROUTER_ML_MODEL_PATH` | — | P3：LightGBM 模型路径（`train_p3.py` 输出）。设置后 ML head 替代规则打分器；空或无效=规则打分器。 |
| `ROUTER_LOG_DECISIONS` | `true` | INFO 打路由决策日志。 |

### P2 Ensemble

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `ENSEMBLE_ENABLED` | `false` | 需 `ROUTER_ENABLED=true`。 |
| `ENSEMBLE_MODE` | `b5_fusion` | `b5_fusion`（融合）或 `best_of_n`（选最佳）。 |
| `ENSEMBLE_PROPOSERS` | — | proposer 模型 id，逗号分隔（anchor 自动加入）。 |
| `ENSEMBLE_AGGREGATOR` | — | aggregator 模型；空 = 路由 anchor。 |
| `ENSEMBLE_SCORER_MODEL` | — | best_of_n 的 scorer；空 = 用 aggregator。 |
| `ENSEMBLE_MIN_TIER` | `c2` | 路由档位 ≥ 此才融合。 |
| `ENSEMBLE_MIN_SUCCESSFUL` | `2` | 聚合前 quorum。 |
| `ENSEMBLE_MAX_PROPOSERS` | `3` | 并发 proposer 上限（含 anchor，成本控制）。 |
| `ENSEMBLE_PROPOSER_TIMEOUT` / `ENSEMBLE_AGGREGATOR_TIMEOUT` | `60` / `120` | 各阶段超时（秒）。 |
| `ENSEMBLE_CANDIDATE_MAX_CHARS` | `24000` | aggregator/scorer prompt 里截断每条草稿。 |

## 关键文件

| 路径 | 用途 |
| --- | --- |
| `.env` | 全部运行配置（已 gitignore，不进版本库）。 |
| `logs/router-samples-YYYY-MM-DD.jsonl` | P3 决策记录（路由时写入，`label` 待回填）。 |
| `logs/router-outcomes-YYYY-MM-DD.jsonl` | P3 outcome 记录（上游返回后写入，sidecar）。 |
| `logs/router-raw-YYYY-MM-DD.jsonl` | P3 raw content（user 消息文本，judge 标注用，opt-in）。 |
| `logs/router-judge-YYYY-MM-DD.jsonl` | P3 judge 标签（LLM-as-judge 绝对难度，judge 脚本生成）。 |
| `logs/router-labeled-YYYY-MM-DD.jsonl` | 离线回填后的带标签训练文件（realign 脚本生成）。 |
| `src/relay/router/scorer.py` | 规则打分——P3 LightGBM 替换点。 |
| `src/relay/router/ml_head.py` | P3 ML head：加载 LightGBM 模型 → `FeatureBundle → ScoreResult`。 |
| `src/relay/ensemble.py` | B5 融合逻辑。 |
| `src/relay/capture.py` | decisions + outcomes + raw 三文件 JSONL 捕获。 |
| `scripts/realign_labels.py` | 离线标签回填：`uv run python scripts/realign_labels.py --date YYYY-MM-DD`。 |
| `scripts/judge_labels.py` | LLM-as-judge 标注：`uv run python scripts/judge_labels.py --date YYYY-MM-DD`。 |
| `scripts/train_p3.py` | LightGBM 训练：`uv run python scripts/train_p3.py --auto`。 |
| `models/p3_lightgbm.txt` | 训练好的模型（`train_p3.py` 输出，`ROUTER_ML_MODEL_PATH` 指向）。 |
| `tests/test_router_scoring.py` | 打分 + 路由自测：`uv run python tests/test_router_scoring.py`。 |

## 路线图

- ~~**P2.5**：ensemble proposer 阶段加 SSE 心跳注释帧~~ —— ✅ 已完成。
  流式模式 proposer 阶段每 5s 发 `: heartbeat\n\n`，防客户端超时。
- ~~**P2.5+**：best-of-N 模式 + dynamic proposer 选择~~ —— ✅ 已完成。
  `ENSEMBLE_MODE=best_of_n` + `ENSEMBLE_MAX_PROPOSERS` 成本控制。
- ~~**P3** 自学习~~ —— ✅ 基础完成。LightGBM 训练 + ML head 推理 + judge 标签管线。
  待持续积累数据迭代重训。
- **P3** 自学习：读捕获的 JSONL → 关联会话 → 标签 realignment（抱怨/重试信号）
  → 增量 LightGBM 重训 → session-holdout CV + 成本上限 gate → 原子指针切换 +
  live rollback，替换 `score_features`。
