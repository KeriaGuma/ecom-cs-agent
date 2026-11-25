# 客服沙箱验收说明

订单、演示账号、会话和模拟支付账本均来自本地确定性合成数据，无需外部模型密钥即可验证完整 HTTP 流式对话、售后双分支状态机、人工坐席接管与支付掉单对账恢复。

## 核心验证项

| 检查项 | 验证内容 |
| --- | --- |
| 全量离线测试 | 覆盖确定性售后规则、仅退款 vs 退货退款双分支、防谎报流式守卫、MCP stdio / HTTP 双层鉴权、混合检索与重排阈值、多模态盲测与跨模态防注入、PII 出口脱敏及评测统计框架。 |
| 顾客端、坐席端、审批与验收工作台 | 顾客确认退货退款草稿、同会话人工坐席接管与回复、审批通过后进入待退货（`awaiting_return`）、顾客回填退货单号、验收员确认入库并触发打款、坐席结束接管后机器人恢复服务。 |
| 容器冷重启与掉单对账恢复 | 模拟支付落账后回执丢失（`ack_lost`），本地状态记为 `unknown`、订单保持 `delivered`；容器重启后会话登录态与聊天记录保持完整，独立对账进程 `cs_agent.refund_jobs` 扫描账本幂等恢复为 `succeeded`。 |
| 在线一致性备份与恢复 | 基于 SQLite backup API 生成一致性副本，恢复到新实例后原会话存档、退款审计事件与支付账本完全一致。 |

## 复现命令

```bash
# 1. 运行全量离线回归测试
.venv/bin/python -m pytest -q

# 2. 启动容器沙箱并执行端到端故障恢复冒烟测试
make demo
.venv/bin/python -m bench.sandbox_smoke --compose

# 3. 运行并发隔离与热点会话互斥压测
.venv/bin/python -m bench.sandbox_load --requests 64 --concurrency 8
.venv/bin/python -m bench.sandbox_load --requests 32 --concurrency 8 --mode hotspot
```
