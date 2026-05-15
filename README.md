# 股票多因子评分系统 (Stock Multi-Factor Scoring System)

全市场扫描 + 多因子评分 + 自主交易 + Web 仪表盘的一站式 A 股分析系统。

## 系统架构

```
stock-system/
├── scripts/
│   ├── strategy_engine.py       # 评分引擎（经典/增强双模式）
│   ├── data_loader.py           # 数据加载与行业映射
│   ├── download_data_v2.py      # 数据下载（新浪 + Baostock）
│   ├── update_kline_incremental.py  # 日K增量更新
│   ├── financial_data.py        # 基本面因子计算
│   ├── realtime_monitor.py      # 实时行情监控
│   ├── web_api.py               # FastAPI 后端（端口 8002）
│   ├── auto_trader.py           # 自主交易模块
│   ├── daily_review.py          # 每日复盘（cron 15:35）
│   ├── trading.py               # 交易/持仓/回测核心逻辑
│   ├── industry_code_map.py     # 行业代码映射
│   └── visualization.py         # K线图绘制
├── web/
│   └── index.html               # 前端仪表盘（单页应用）
├── data/                        # 数据目录（gitignore）
│   ├── daily/                   # 日K线 Parquet（5179只股票）
│   ├── financial/               # 基本面数据
│   ├── info/                    # 股票基础信息
│   └── charts/                  # 生成图表缓存
├── README.md
└── .gitignore
```

## 核心功能

### 1. 双模式评分引擎

| 模式 | 维度 | 底层指标数 | 权重分布 |
|:---|:---|:---:|:---|
| **经典评分** | 技术+基本面+资金 | **8项** | 技术40%+基本面40%+资金20% |
| **增强评分** | 技术+价量+行业+热点+资金 | **15+项** | 技术30%+价量25%+行业20%+热点15%+资金10% |

**经典评分细分指标：**
- 技术面 (40%)：均线趋势(24%) + RSI(6%) + 量比(6%) + MACD(4%)
- 基本面 (40%)：PE百分位(16%) + ROE(12%) + 营收增长率(12%)
- 资金面 (20%)：北向资金评分

**增强评分细分指标：**
- 技术面 (30%)：均线趋势50% + RSI20% + 量比15% + MACD15%
- 价量衍生 (25%)：乖离率35% + 趋势强度35% + 波动率15% + 换手率15%
- 行业动量 (20%)：板块5日/20日涨幅排名综合
- 市场热点 (15%)：资金集中度 + 情绪热度
- 资金流 (10%)：北向资金 + 大单流向

### 2. Web 仪表盘 (port 8002)

前端为单页应用，深色主题（Tokyo Night风格），包含：

| 页面 | 功能 |
|:---|:---|
| 🔍 **全市场扫描** | 评分排序列表，经典/增强模式切换，按行业筛选 |
| 📊 **个股分析** | K线图（日/周/月），技术指标，实时行情 |
| ⭐ **自选股** | 关注的个股评分动态跟踪 |
| 💰 **持仓与交易** | 持仓管理，买卖操作，盈亏统计 |
| 🔥 **热点板块** | 行业动量排名（5日/20日） |
| 📈 **回测** | 策略历史回测 |

### 3. 自主交易（auto_trader）

cron 驱动的自动交易系统（每10分钟盘中执行）：

- **买入条件**：增强评分 ≥ 70 + 价格站上20日均线
- **卖出条件**：评分 < 55 或趋势破位（价格跌破60日均线）
- **止损**：浮亏 -5% 自动止损
- **风控**：单次开仓 ≤ 现金5%，单只 ≤ 总资产15%，行业持仓上限

### 4. 每日复盘（cron 15:35）

自动生成当日持仓分析、系统诊断、交易日志，推送至飞书群。

## 快速开始

### 启动 Web 服务

```bash
cd ~/stock-system
python scripts/web_api.py
```

访问 `http://localhost:8002`

### 更新日K数据

```bash
python scripts/update_kline_incremental.py
```

### 运行全市场扫描

```bash
python scripts/strategy_engine.py scan
```

### 手动交易

```bash
# 查看持仓
python scripts/auto_trader.py
# 早盘风险检查
python scripts/auto_trader.py --morning-scan
# 强制扫描持仓评分
python scripts/auto_trader.py --force-scan
```

## 数据源

- **日K线**：本地 Parquet 文件（~2.8GB/5179只），Baostock 全量下载
- **实时行情**：新浪 API
- **基本面**：本地财务数据（PE/ROE/营收增长率）
- **行业映射**：自定义行业代码映射表，覆盖28个行业

## 技术栈

| 层 | 技术 |
|:---|:---|
| 后端 | Python 3, FastAPI, pandas, numpy, pyarrow |
| 前端 | 原生 HTML/CSS/JS, 深色主题, ECharts (K线) |
| 数据 | Parquet (列存), Baostock, 新浪实时 API |
| 交易 | 模拟交易, JSON 持久化 |
| 自动化 | Linux cron, 每日复盘推送飞书 |
