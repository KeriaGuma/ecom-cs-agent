# 电商售后智能客服 Agent（ecom-cs-agent）

面向电商售后查单、改址、退换货与政策咨询场景的**生产级确定性客服 Agent 系统**。

- **全异步流式单 Agent 主架构**：主线采用纯 Python 原生异步 ReAct 流式引擎（`cs_agent/`），消除多 Agent 间的转述损耗与串行调度开销，支持细粒度工具进度感知与 Token 级 SSE 实时输出。
- **确定性规则与双分支售后状态机**：退货资格、金额、运费与审批阈值完全由规则引擎计算；严格区分**未发货仅退款（确认即打款）**与**已签收退货退款（确认/审批 → 买家回填退货单号 → 仓库验货入库 → 幂等打款）**，配备独立主动对账恢复进程处理支付网关超时掉单。
- **纵深安全与防幻觉护栏**：工具层闭包 / MCP `_meta` 绑定用户身份（HTTP 模式采用 `Bearer Scope` 服务令牌 + 60s `HMAC-SHA256` 用户断言双层鉴权）；内置流式防“谎报完成”守卫（`Claim Guard`）、多模态盲测核验（隔离用户描述防顺从幻觉 + OCR 转义防跨模态注入）与全出口 PII 脱敏。
- **严谨评测与架构边界实证**：在自建盲测基准及 **τ²-bench、ECom-Bench** 公开基准上，保留同工程水平的 Supervisor 多 Agent 对照基线（`bench/multi_agent/`），用配对消融实验量化“工程护栏 vs 多 Agent 架构拆分”的真实收益与成本拐点。

---

## 一分钟体验离线沙箱

沙箱内置合成订单、确定性脚本模型与模拟支付账本，**无需配置任何 API Key** 即可走通真实 HTTP SSE 流式对话、退货退款审批、人工坐席接管与支付掉单对账恢复：

```bash
# 方式一：Docker Compose 启动（含应用容器 + 独立主动对账进程）
make demo

# 方式二：本地 Python 虚拟环境直接启动
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
make demo-local

# 顾客端：http://127.0.0.1:8000
# 坐席与售后工作台：http://127.0.0.1:8000/desk
```

| 角色 | 演示账号 | 演示口令 | 权限说明 |
| --- | --- | --- | --- |
| 买家 | `u001`、`u002` | `demo` | 发起会话、确认退款草稿、填写退货快递单号 |
| 客服坐席 | `s01`、`s02` | `demo` | 领取优先级工单、接管原会话回复买家、关闭工单 |
| 售后审批与验收员 | `a01` | `demo` | 审批大额售后申请、确认退货包裹验收入库、核对异常支付 |

### 演示大额退货退款与支付掉单恢复

1. 在顾客页登录 `u001`，发送「`订单 O1002 的扫地机器人买错了，帮我申请退货退款`」，核对并点击**确认申请**（¥1,899 超出自动额度，进入 `pending_approval` 待人工审核；期间聊天会话保持畅通）。
2. 发送「`我要转人工`」，在 `/desk` 以 `s01` 登录领取工单并回复，验证人工接管期间机器人自动静默、坐席关闭工单后机器人恢复接待。
3. 在 `/desk` 切换 `a01` 登录审批通过该申请：因 `O1002` 为已签收商品，状态流转为 **`awaiting_return`（待寄回退货）**，不会未退货先打款。
4. 回到顾客页在退款卡片填入退货快递单号（如 `SF900123`）并提交；要观察掉单恢复，可先执行 `docker compose stop reconciliation`，在 `a01` 工作台为该单选择「退款成功但回执丢失」后点击**确认退货入库并打款**，状态变为 `unknown`（待核对）。
5. 执行 `docker compose restart sandbox && docker compose start reconciliation`，对账进程自动扫描支付账本将状态恢复为 `succeeded`，且重启后原会话记录与登录态完整保留。

---

## 系统架构

```
客户端 (Web SSE / CLI)
   │  POST /chat (Async, X-User-Id / Signed Cookie, request_id 幂等回执)
   ▼
FastAPI 异步网关 (引用计数会话互斥锁 SessionLockManager)
   ├── 人工坐席接管中 ──► 追加至原会话存档，静默机器人 ──► 结束
   ▼
多模态前置核验 (vision.py：盲测提取商品瑕疵，转义 OCR 阻断跨模态注入)
   │
   ▼
三层上下文组装 (代码提取多订单 Facts + 较早对话滚动摘要 + 最近原文滑动窗口)
   │
   ▼
核心单 Agent 异步 ReAct 引擎 (agent.py + react.py)
   ├── 受控业务工具调用（闭包 / MCP _meta 强绑定 user_id）
   │     ├─ 订单与物流工具 (Local / MCP stdio / MCP HTTP 双层鉴权 + 地址守卫)
   │     ├─ 售后规则与确认单 (rules.py 确定性计算资格/金额/运费 + 生成草稿)
   │     ├─ 混合知识库检索 (BM25 + 向量召回 → 去重并集 → Cross-Encoder 重排)
   │     └─ 转人工工具 (结构化交接摘要 + 投诉/情绪关键词高优兜底)
   ├── 最终回复出口：流式防谎报守卫 (Claim Guard，拦截无写工具依据的虚假承诺)
   ├── 模型/工具异常 ──► 本地零依赖 FAQ 兜底（纯咨询）或规则摘要转人工
   ▼
Token 级 SSE 实时输出 ──► 推送 done 与确认单卡片 ──► 后台异步滚动摘要 + PII 脱敏归档
                                   │
                                   ▼ 用户确认草稿
         ┌─────────────────────────┴─────────────────────────┐
     未发货仅退款 (refund_only)                     已签收退货退款 (return_refund)
   小额直通 / 大额人工审批                           小额直通 / 大额人工审批
         │                                                   │
         │                                          买家回填退货单号 → 仓库验货入库
         └─────────────────────────┬─────────────────────────┘
                                   ▼
                 支付网关幂等打款 (refund_id 幂等键) ◄── 独立对账进程主动恢复
```

### 目录与模块说明

| 模块 | 路径 | 职责说明 |
| --- | --- | --- |
| 核心单 Agent | [cs_agent/agent.py](cs_agent/agent.py) | 系统提示词组装、用户级工具挂载、多订单事实提取与同步/异步入口 |
| 异步流式 ReAct | [cs_agent/react.py](cs_agent/react.py) | 原生异步 Token 流式循环，协调工具进度事件与流式防谎报缓冲校验 |
| 会话编排器 | [cs_agent/runner.py](cs_agent/runner.py) | 串联视觉前置核验、主引擎执行、FAQ 降级兜底、人工交接与记忆落库 |
| 异步 HTTP 网关 | [cs_agent/api.py](cs_agent/api.py) | SSE 流式对话、自动回收会话锁、请求回执重放、售后状态机与工单接口 |
| 售后规则引擎 | [cs_agent/rules.py](cs_agent/rules.py) | 确定性计算退货资格、售后类型（仅退款/退货退款）、金额、运费与审批阈值 |
| 售后与支付状态机 | [cs_agent/refunds.py](cs_agent/refunds.py)、[cs_agent/refund_jobs.py](cs_agent/refund_jobs.py) | 草稿确认、人工审批、退货单号与验货入库、幂等打款及独立主动对账恢复 |
| 业务工具与守卫 | [cs_agent/tools.py](cs_agent/tools.py) | 闭包绑定 `user_id`、订单归属校验、地址混入电话/收件人拦截、操作留痕回显 |
| MCP 服务与鉴权 | [cs_agent/mcp_server.py](cs_agent/mcp_server.py)、[cs_agent/mcp_client.py](cs_agent/mcp_client.py)、[cs_agent/mcp_auth.py](cs_agent/mcp_auth.py) | 订单工具 MCP 化；HTTP 部署采用 Bearer Scope 令牌 + 60s HMAC `_meta` 用户断言 |
| 混合检索管线 | [cs_agent/rag.py](cs_agent/rag.py)、[cs_agent/kb.py](cs_agent/kb.py) | BM25 + 向量召回、去重并集 Cross-Encoder 重排、相关性阈值截断与多级退化 |
| 多模态核验 | [cs_agent/vision.py](cs_agent/vision.py) | 独立视觉模型盲测提取商品瑕疵特征，清洗转义图片内文字防御跨模态注入 |
| 三层会话记忆 | [cs_agent/memory.py](cs_agent/memory.py)、[cs_agent/state.py](cs_agent/state.py) | 代码提取多订单关键事实 + 异步非阻塞滚动摘要 + 最近原文滑动窗口 |
| 防谎报与脱敏 | [cs_agent/claims.py](cs_agent/claims.py)、[cs_agent/pii.py](cs_agent/pii.py) | 基于写工具轨迹拦截虚假办理承诺；日志、工单、对话存档出口统一 PII 打码 |
| 人机协同交接 | [cs_agent/handoff.py](cs_agent/handoff.py)、[cs_agent/desk_service.py](cs_agent/desk_service.py)、[cs_agent/desk.py](cs_agent/desk.py) | LLM 结构化交接摘要（故障时规则降级）、投诉高优插队、坐席工作台与 CLI |
| 存储层 | [cs_agent/store.py](cs_agent/store.py) | 开启 WAL 与忙等待超时的 SQLite 事务存储 |
| 架构对照与基准 | [bench/multi_agent/](bench/multi_agent/)、[bench/ecom/](bench/ecom/)、[bench/tau2/](bench/tau2/) | Supervisor 多 Agent 对照基线，以及 ECom-Bench 与 τ²-bench 公开基准对比实验 |

---

## 核心工程设计

1. **确定性逻辑不交给大模型**：模型只负责理解用户意图与售后原因归类；能否退货、属于仅退款还是退货退款、退多少金额、运费谁承担、是否触发大额审批、工单排队优先级，全部由确定性代码计算。
2. **杜绝“未退货先打款”的电商售后状态机**：未发货订单（`paid`）走 `refund_only`，确认后直接原路退款；已签收订单（`delivered`）强制走 `return_refund`，确认及大额审批通过后停在 `awaiting_return`，必须经买家回填物流单号并由仓库/验收员确认收货后才触发 `PaymentGateway.submit()`。支付超时记为 `unknown`，通过 `refund_id` 幂等键支持安全重试与后台对账恢复。
3. **身份移出模型参数空间 + MCP 双层鉴权**：本地工具通过闭包绑定 `user_id`；MCP HTTP 独立部署时，HTTP 头 `Authorization: Bearer` 校验调用方服务身份与读写 Scope（`orders:read` / `orders:write`），请求 `_meta` 携带 60 秒过期的 `HMAC-SHA256` 用户断言校验“替哪个用户调用”，模型既看不到也无法篡改用户身份。
4. **流式输出与防谎报守卫（`Claim Guard`）协同**：在原生异步流式循环中，若本轮已调用写工具，最终回复零缓冲逐 Token 实时流式推送（低首字延迟）；若本轮未调用任何写工具，先对短文本回复做轻量缓冲校验，一旦检测到无写工具支撑的“已修改/已生成确认单/已退款”表述，静默注入提醒打回重跑，确保前端绝不收到虚假办理承诺。
5. **多模态盲测防顺从 + 跨模态防注入**：视觉模型核验图片时**不接收用户的文字描述**，防止用户声称“裂了”诱导模型产生顺从幻觉；图片中的纸条/截图文字经换行折叠与段落分隔符转义后写入独立状态字段，阻断跨模态伪造系统上下文。
6. **三层记忆与异步非阻塞压缩**：第一层由代码从真实工具返回中提取焦点订单与近期涉及订单列表（自动忽略越权或不存在的订单）；第二层在 `done` 事件发出后异步压缩旧消息生成滚动摘要，不卡用户尾延迟；第三层保留全部尚未摘要的近期原文，消除滑动窗口断层；全量对话经脱敏后独立归档至 `transcripts` 表。
7. **出口统一 PII 脱敏**：通过 Python `logging.setLogRecordFactory` 在日志生成源头统一打码（覆盖第三方库、Uvicorn 访问日志与异常堆栈），并在工单与对话归档写库前脱敏手机号（含空格/短横线/`+86`）、身份证、银行卡、邮箱及详细地址（兼容直辖市省略“市”字写法），同时保留模型推理链路内的地址原文以便向用户核对确认。

---

## 快速开始（接入真实大模型）

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # 填入 LLM_API_KEY（默认 DeepSeek，支持任意 OpenAI 兼容接口）

# 离线全量单元测试（无需 API Key）
.venv/bin/python -m pytest -q

# 启动线上异步 HTTP 服务（浏览器访问 http://127.0.0.1:8000）
.venv/bin/uvicorn --factory cs_agent.api:create_app --port 8000

# 命令行交互对话
.venv/bin/python -m cs_agent.cli --user u001 --session demo --verbose

# 独立部署订单 HTTP MCP Server（双层鉴权）
.venv/bin/python -m cs_agent.mcp_auth new-token --name cs-backend --tokens data/mcp_tokens.json
.venv/bin/python -m cs_agent.mcp_auth new-key
MCP_USER_KEY=... .venv/bin/python -m cs_agent.mcp_server --db data/shop.db --http --port 8765 --tokens data/mcp_tokens.json
ORDER_TOOLS=mcp-http .venv/bin/python -m cs_agent.cli --user u001

# 命令行坐席与售后审批工作台
.venv/bin/python -m cs_agent.desk --approvals
.venv/bin/python -m cs_agent.desk --approve <退款编号> --staff a01
.venv/bin/python -m cs_agent.desk --receive <退款编号> --comment SF900123 --staff a01
```

### HTTP 接口一览

| 接口 | 鉴权依赖 | 说明 |
| --- | --- | --- |
| `POST /chat` | 买家身份 | SSE 流式对话，依次推送 `status`、`token`、`message`、`done`（含本轮确认单或工单号） |
| `GET /sessions/{id}/messages`、`DELETE /sessions/{id}/messages` | 买家身份 | 读取脱敏会话历史与人工接管状态；清理当前会话上下文 |
| `GET /sessions/{id}/refunds`、`GET /refunds/{id}` | 买家身份 | 查询本人售后申请列表与实时核对进度 |
| `POST /refunds/{id}/confirm` | 买家身份 | 确认（`true`）或取消（`false`）退款草稿 |
| `POST /refunds/{id}/return` | 买家身份 | 为 `awaiting_return` 状态的退货单回填快递单号 |
| `GET /approvals` | 审批员身份 | 查看待审批大额申请（`refunds`）与待验货退货单（`returns`） |
| `POST /refunds/{id}/decision` | 审批员身份 | 审批通过或驳回大额申请 |
| `POST /refunds/{id}/receive` | 审批员身份 | 确认退货包裹已验收入库并触发支付网关原路退款 |
| `POST /refunds/{id}/reconcile` | 审批员身份 | 主动查询支付账本，必要时使用原申请编号幂等重试 |
| `GET /tickets`、`POST /tickets/{id}/claim`、`POST /tickets/{id}/messages`、`POST /tickets/{id}/close` | 坐席身份 | 优先级工单队列、领取接管、人工回复买家与结束接管 |

---

## 评测体系与单/多 Agent 架构选型实证

本项目所有评测严格遵守三项纪律：
1. **盲测隔离**：留出测试集（Held-out Test Set）仅依据业务规则文档独立构建，构建过程与系统提示词、开发集物理隔离；
2. **防重跑与污染退役**：评测脚本自动记录提示词哈希、配置串与数据集哈希，同一测试集严禁调参重跑；凡参考过错例做针对性修复的测试集立即登记入 `RETIRED_TEST_SETS` 强制退役；
3. **双异构模型评委校准**：被测系统采用 DeepSeek，回复要点复核与沟通质量评估采用 Kimi + 通义千问交叉核验，要求强制引用原文经代码字面校验，并通过 150 份合成缺陷样本完成检出率校准。

### 1. 为什么线上主架构选择单 Agent 而非多 Agent？

为了客观回答“客服场景是否应该拆分为 Supervisor + 多专员 Agent”，我们在**共享完全相同的工具权限代码、确定性规则引擎、三层记忆与防谎报护栏**的前提下，对工程化单 Agent 与 Supervisor 多 Agent（`bench/multi_agent/`）进行了多维度消融对比：

#### (1) 自建基准与扩展实验对照

```bash
.venv/bin/python -m cs_agent.eval_e2e --system both --split dev --repeat 3
```

| 实验场景 | 单 Agent（主架构） | Supervisor 多 Agent（对照） | 核心工程结论 |
| --- | --- | --- | --- |
| 标准留出集（45 题 × 3 次） | 91.9%（p50 **4.3s**，Token 6592） | 95.6%（p50 6.1s，Token 6296） | 成功率差值 95% CI `[-3.0%, +11.1%]` **统计不显著**；单 Agent 延迟低 **30%** |
| 退款边界与防谎报修复后（test4，45 题 × 3） | **100%**（p50 **5.4s**） | **100%**（p50 7.4s） | 补齐确定性规则与代码守卫后两边均达 100%，单 Agent 显著更快 |
| 追问与操作留痕修复后（test6，36 题 × 3） | **98.1%**（p50 **3.9s**） | 99.1%（p50 4.9s） | 跨轮操作记录消除了单 Agent 历史动作不可见短板 |
| 工具数量扩展（挂满 48 个工具） | 94.3%（误调干扰工具 1 次，Token +113%） | 95.5%（误调 2 次，Token +44%） | 48 个工具规模下单 Agent **并不会选错工具**，差异仅在输入 Token 成本 |
| 业务域扩展（3 → 8 个领域，T2 档） | 96.1%（p50 **4.4s**，Token 9408） | **100%**（p50 6.5s，Token **5885**） | 扩展至 8 个独立领域时，单 Agent Token 为多 Agent 的 **1.6 倍** |

#### (2) 公开基准实证（ECom-Bench 与 τ²-bench）

详见 [bench/ecom/README.md](bench/ecom/README.md) 与 [bench/tau2/README.md](bench/tau2/README.md)。在多意图极度密集（单段对话 4～8 件事）的中文京东客服基准 **ECom-Bench**（25 题 × 3 次）上的三方对照结果：

| 对比实现 | pass^1 | pass^3 | 每轮延迟 p50 | 客服总 Token |
| --- | --- | --- | --- | --- |
| ECom-Bench 上游自带单 Agent | 42.7% | 16.0% | **4.0s** | **79,659** |
| **我们的工程化单 Agent** | **69.3%** | **48.0%** | 4.2s | 90,287 |
| 我们的 Supervisor 多 Agent | 48.0% | 20.0% | 9.2s | 114,814 |

- **工程护栏的真实贡献**（工程化单 Agent − 自带单 Agent）：**+26.7%**（95% CI `+13.3%～+41.3%`，显著）——来自跨轮保留工具结果、防谎报代码拦截与明确的确认规则；
- **拆分多 Agent 的纯架构效应**（多 Agent − 工程化单 Agent）：**−21.3%**（95% CI `−36.0%～−5.3%`，显著变差）——根源在于前台仅看专员转述汇报导致细节丢失、需要用户确认的写操作在前台与专员之间往返拆裂。
- 在 **τ²-bench** 零售（40 题 × 3）与航空（20 题 × 3）官方测试集上结论完全一致：单 Agent 最准且最快（pass^1 90.0% / 80.0%，耗时省 40%）。

> **架构选型结论**：在 10～40 个工具、单一团队维护的客服场景下，**“工程化单 Agent + 确定性代码护栏”在延迟、流式体验与复杂多意图准确性上全面优于多 Agent**；只有当业务膨胀至数十个跨团队独立演进的领域微服务、且全量工具定义导致单轮输入 Token 成本成为主要矛盾时，才值得引入 Supervisor 多 Agent 分层。

---

### 2. 混合检索（RAG）评测

详见 `python -m cs_agent.eval_rag`。知识库共 8 份文档、48 条售后条款；在独立留出测试集（62 题，覆盖口语化同义改写与精确术语提问）上的消融结果：

| 检索配置 | Recall@1 | Recall@3 | MRR@10 | 开发集 p50 延迟 |
| --- | --- | --- | --- | --- |
| 字符二元组匹配（仅用于本地零依赖 FAQ 降级） | 54.8% | 74.2% | 0.646 | <1 ms |
| 纯 BM25（jieba 分词） | 53.2% | 69.4% | 0.629 | <1 ms |
| 纯向量检索（`qwen3.7-text-embedding-flash`） | 93.5% | 98.4% | 0.960 | 111 ms |
| BM25 + 向量（等权 RRF 融合） | 72.6% | 93.5% | 0.838 | 114 ms |
| **BM25 + 向量并集 + 专用 Cross-Encoder 重排（线上配置）** | **98.4%** | **100%** | **0.992** | **208 ms** |

- 口语化提问下，等权 RRF 会被 BM25 的字面词频噪声拉低排名；线上采用**两路召回并集直接送入专用重排模型（`qwen3.7-text-rerank`）**，仅增加约 100ms 即达到 98.4% Recall@1，同时保留重排故障 → 向量优先加权融合 → 纯 BM25 的完整退化链路。

---

### 3. 安全对抗、多模态核验与鲁棒性评测

| 评测维度 | 测试集规模 | 核心指标结果 |
| --- | --- | --- |
| **安全与提示词注入对抗**（[eval/SAFETY_GUIDE.md](eval/SAFETY_GUIDE.md)） | 49 题 × 3 次（含直接注入、商品名/物流轨迹间接注入、越权冒充、套取提示词） | **严重失败（被攻破/越权/泄密）率 0%**（0/147） |
| **故障注入测试** | 45 题 × 2 次（注入 20% 数据库超时 / 30% 模型接口超时） | **写错数据 0 次、未捕获崩溃 0 次**，全部如实告知超时或安全降级转人工 |
| **口语扰动鲁棒性** | 45 题 × 4 类改写（错别字、方言口语、中英夹杂、同义换问） | 变体与原题平均成功率差异 < 2% |
| **防“谎报完成”代码守卫** | 45 题 × 3 次（含 18 题“办理 + 咨询”复合诉求） | 单 Agent 无依据虚假承诺从 **4 次降至 0 次** |
| **地址守卫与不支持功能规则** | `test7` + `test8` 各 30 题 × 3 次 | 电话混入地址写库错误 **30 次 → 0 次**；顺带不支持诉求误转人工 **5 次 → 0 次** |
| **多模态商品照片核验** | 30 题 × 3 次（覆盖破损、完好冒充、拍糊、拍错商品、图片纸条注入） | 相比不看图基线 **+32.2%～+35.6%**（显著）；**完好照片误判为质量问题 0 次** |

---

## 演示订单数据（买家 `u001`）

| 订单号 | 商品 | 金额 | 当前状态 | 典型验证场景 |
| --- | --- | --- | --- | --- |
| `O1001` | 蓝牙耳机 | ¥199 | 已签收 3 天 | 退货退款确认 → 买家填退货单号 → 验货入库打款 |
| `O1002` | 扫地机器人 | ¥1,899 | 已签收 5 天 | 大额退货退款 → 人工审批 → 退货验货入库 → 对账恢复 |
| `O1003` | 羽绒服 | ¥699 | 运输中 | 查询物流轨迹；运输中不支持直接退款（引导拒收或签收后申请） |
| `O1004` | 保温杯 | ¥89 | 已签收 20 天 | 超出 7 天无理由及 15 天质量退货期拦截 |
| `O1005` | 机械键盘 | ¥459 | 待发货 | 未发货仅退款（确认后直接原路打款）；修改收货地址（含电话混入拦截） |
| `O2001` | 手机 | ¥3,999 | 属于 `u002` | 跨用户越权查询/修改/退款工具层拦截 |
