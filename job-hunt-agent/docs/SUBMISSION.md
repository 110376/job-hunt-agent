# Inspire 校招 AI Engineer 作业提交说明

## 1. 项目目标

实现一个具备自主规划与执行能力的 Agentic AI 系统，自动收集 AI Engineer 校招/实习岗位，输出结构化数据（JSON/CSV）。

## 2. 功能完成情况

- 自动收集岗位：已完成
- 目标数量 50：已完成（见 `outputs/ai-engineer-20260318-233746.json`）
- 至少 2 个站点：已完成（`boss` + `liepin`）
- 语义筛选（AI + 校招/实习）：已完成
- 去重与结构化：已完成
- 结果导出 JSON/CSV：已完成

## 3. Agent 设计

核心闭环：

1. 目标输入与状态初始化
2. 生成查询词（分层扩圈）
3. 搜索与详情抓取
4. 语义判断与技术标签抽取
5. 去重入库
6. 进度评估与策略调整
7. 达标/预算耗尽时导出

关键模块：

- 主流程：`app/agent_runner.py`
- 工具层：`app/tools.py`
- 查询规划：`app/services/query_planner.py`
- 分类器：`app/services/classifier.py`
- 去重：`app/services/dedup.py`
- 导出：`app/services/exporter.py`

## 4. 关键优化点（对应评分项）

- 三层扩圈策略（高精度 -> 相关岗位 -> 宽泛岗位 + AI 约束）
- 低增长自动扩站点（不是固定 2 个站）
- 站点熔断与恢复（失败后暂停若干轮）
- LLM 降级机制（429 限流和 400/401 不可恢复错误）
- 预算保护（最大轮次 + 查询预算，防无限循环）
- 预采集后启用 Agent（`JOB_AGENT_AGENT_START_COLLECTED`，降低 API 压力）

## 5. 输出格式

输出字段：

- `title`
- `company`
- `location`
- `salary`
- `tech_tags`
- `requirements`
- `source`
- `job_url`

## 6. 运行方式

```bash
python run_job_agent.py --role "AI Engineer" --target 50 --sites "boss,liepin" --max-rounds 12 --non-interactive
```

可选（先预采集再启用 Agent）：

```bash
set JOB_AGENT_AGENT_START_COLLECTED=20
python run_job_agent.py --role "AI Engineer" --target 50 --sites "boss,liepin" --max-rounds 12 --non-interactive
```

## 7. 测试结果

- 命令：`pytest -q`
- 结果：通过（详见本地运行结果）
