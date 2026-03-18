# Job Hunt Agent (Agentic AI)

一个面向校招/实习 AI Engineer 岗位的 Agentic AI 求职系统。  
系统支持任务规划、工具调用、迭代搜索、语义筛选、去重和结构化导出（JSON/CSV）。

## 1. 作业要求映射

- 目标采集：`50` 条岗位，且来源至少 `2` 个站点
- Agent 能力：
  - 任务规划与流程编排：[`app/agent_runner.py`](./app/agent_runner.py)
  - 工具调用：[`app/tools.py`](./app/tools.py)
  - 迭代搜索：[`app/services/query_planner.py`](./app/services/query_planner.py)
  - 语义判断：[`app/services/classifier.py`](./app/services/classifier.py)
  - 数据清洗与去重：[`app/services/dedup.py`](./app/services/dedup.py), [`app/schemas.py`](./app/schemas.py)
  - 结果导出：[`app/services/exporter.py`](./app/services/exporter.py)

## 2. 架构与关键策略

- Agent 主循环：LLM Agent + deterministic fallback 双模式
- 三层扩圈搜索：
  - Tier 1：高精度 AI 校招关键词
  - Tier 2：相关 AI 角色 + 城市扩展
  - Tier 3：宽泛岗位（如 Python/后端）+ AI 上下文约束
- 低增长自动扩站点：结果增长停滞时自动加入新站点
- 失败保护：
  - 模型限流降级（429）
  - 非可恢复 LLM 错误降级（400/401）
  - 站点级熔断（连续失败后暂停若干轮）
  - 查询预算上限（防止无限循环）
- API 压力控制：
  - `JOB_AGENT_AGENT_START_COLLECTED` 支持“先预采集，再启用 Agent”

## 3. 安装

```bash
python -m pip install -r requirements.txt
```

可选（登录态反爬兜底）：

```bash
python -m pip install playwright
python -m playwright install chromium
```

## 4. 配置

在项目根目录创建 `.env`：

```env
JOB_AGENT_PROVIDER=deepseek
JOB_AGENT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_api_key
```

常用可选环境变量：

- `JOB_AGENT_ROLE`
- `JOB_AGENT_TARGET`
- `JOB_AGENT_SITES`
- `JOB_AGENT_AUTO_EXPAND_SITES`（默认 `1`）
- `JOB_AGENT_SITE_EXPAND_NO_GROWTH_THRESHOLD`（默认 `2`）
- `JOB_AGENT_AGENT_START_COLLECTED`（默认 `0`，先采集再启用 Agent）
- `JOB_AGENT_MAX_QUERY_ATTEMPTS`（默认 `120`）

登录态抓取相关：

- `JOB_AGENT_USE_LOGIN_FETCHER=1`
- `JOB_AGENT_LOGIN_STATE=.job_agent_login_state.json`
- `JOB_AGENT_LOGIN_HEADLESS=1`
- `JOB_AGENT_FORCE_LOGIN_FETCHER=1`（可选）

## 5. 运行

交互模式：

```bash
python run_job_agent.py
```

非交互模式（推荐）：

```bash
python run_job_agent.py \
  --role "AI Engineer" \
  --target 50 \
  --sites "boss,liepin" \
  --provider deepseek \
  --model deepseek-chat \
  --max-rounds 12 \
  --non-interactive
```

降低 LLM 调用压力（先预采集 20 条）：

```bash
set JOB_AGENT_AGENT_START_COLLECTED=20
python run_job_agent.py --role "AI Engineer" --target 50 --sites "boss,liepin" --max-rounds 12 --non-interactive
```

准备登录态（仅当目标站点反爬明显时使用）：

```bash
python run_job_agent.py --prepare-login --login-sites "boss,liepin"
```

## 6. 输出

输出目录：`outputs/`

字段：

- `title`
- `company`
- `location`
- `salary`
- `tech_tags`
- `requirements`
- `source`
- `job_url`

当前保留示例：

- [`outputs/ai-engineer-20260318-233746.json`](./outputs/ai-engineer-20260318-233746.json)
- [`outputs/ai-engineer-20260318-233746.csv`](./outputs/ai-engineer-20260318-233746.csv)

## 7. 测试

```bash
pytest -q
```

## 8. 目录结构

```text
app/
  agent_runner.py
  tools.py
  schemas.py
  model_factory.py
  logging_utils.py
  services/
    classifier.py
    query_planner.py
    dedup.py
    exporter.py
    login_fetcher.py
  sites/
    base.py
    registry.py
tests/
outputs/
run_job_agent.py
```
