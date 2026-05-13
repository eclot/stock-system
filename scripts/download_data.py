#!/usr/bin/env python3
"""
股票数据获取脚本
- 全量A股日K线（上市日起至最新）
- 股票基础信息（名称、行业、上市日期）
- 财务数据（PE、PB、ROE、营收增速）
- 存储格式：Parquet
"""

import akshare as ak
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ========== 配置 ==========
DATA_DIR = os.path.expanduser("~/stock-system/data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
INFO_DIR = os.path.join(DATA_DIR, "info")
FINANCIAL_DIR = os.path.join(DATA_DIR, "financial")
PROGRESS_FILE = os.path.join(DATA_DIR, "download_progress.json")

# 沪深300成分股（用于测试验证）
CSI300_FILE = os.path.join(DATA_DIR, "csi300_list.parquet")

os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(INFO_DIR, exist_ok=True)
os.makedirs(FINANCIAL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, "download.log"))
    ]
)
logger = logging.getLogger(__name__)

# AKShare 请求间隔（秒）
REQUEST_INTERVAL = 0.3


def get_all_stock_list():
    """获取全量A股列表（含代码、名称、行业、上市日期）"""
    logger.info("获取A股全量股票列表...")
    df = ak.stock_zh_a_spot_em()
    
    # 标准化字段
    result = df[["代码", "名称", "最新价", "涨跌幅", 
                 "总市值", "流通市值", "市盈率-动态", 
                 "市净率"]].copy()
    
    result.columns = ["symbol", "name", "price", "change_pct",
                      "total_mv", "float_mv", "pe", "pb"]
    
    # 过滤无效数据
    result = result[result["symbol"].notna()].reset_index(drop=True)
    
    logger.info(f"获取到 {len(result)} 只股票")
    return result


def get_stock_industry_info():
    """获取股票行业分类信息"""
    logger.info("获取股票行业分类...")
    try:
        df = ak.stock_board_industry_name_em()
        # 这里只获取板块列表，个股归属需另外获取
        return df
    except Exception as e:
        logger.warning(f"获取行业分类失败: {e}")
        return None


def download_daily_kline(symbol, name=""):
    """
    下载单只股票日K线（前复权）
    返回 DataFrame 或 None
    """
    # 提取纯数字代码（AKShare 需要）
    code = symbol.replace("sh", "").replace("sz", "").strip()
    
    # 临时取消代理（AKShare 直连 eastmoney 不走代理）
    old_env = {}
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        old_env[key] = os.environ.pop(key, None)
    
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date="19900101",
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq"
        )
        
        # 恢复代理
        for key, val in old_env.items():
            if val is not None:
                os.environ[key] = val
        
        if df is None or len(df) == 0:
            logger.warning(f"  [{symbol}] {name} - 无数据")
            return None
            
        # 统一字段命名
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change",
            "换手率": "turnover"
        })
        
        # 只保留需要的列
        keep_cols = ["date", "open", "close", "high", "low", 
                     "volume", "amount", "change_pct", "turnover"]
        df = df[[c for c in keep_cols if c in df.columns]]
        
        # 添加股票代码列
        df["symbol"] = symbol
        
        # 日期排序
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"  [{symbol}] {name} - {len(df)} 条记录 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
        return df
        
    except Exception as e:
        logger.error(f"  [{symbol}] {name} - 下载失败: {e}")
        return None
    
    finally:
        # 始终恢复代理
        for key, val in old_env.items():
            if val is not None:
                os.environ[key] = val


def compute_technical_indicators(df):
    """计算常用技术指标"""
    if df is None or len(df) < 20:
        return df
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    
    # MA均线
    for period in [5, 10, 20, 60]:
        df[f"ma{period}"] = pd.Series(close).rolling(period).mean().values
    
    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd_dif = ema12 - ema26
    macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
    macd_hist = 2 * (macd_dif - macd_dea)
    df["macd_dif"] = macd_dif.values
    df["macd_dea"] = macd_dea.values
    df["macd_hist"] = macd_hist.values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = (100 - (100 / (1 + rs))).values
    
    # KDJ
    low_9 = pd.Series(low).rolling(9).min()
    high_9 = pd.Series(high).rolling(9).max()
    rsv = (pd.Series(close) - low_9) / (high_9 - low_9) * 100
    k = rsv.ewm(com=2).mean()
    d_val = k.ewm(com=2).mean()
    j = 3 * k - 2 * d_val
    df["kdj_k"] = k.values
    df["kdj_d"] = d_val.values
    df["kdj_j"] = j.values
    
    # 布林带(20,2)
    boll_mid = pd.Series(close).rolling(20).mean()
    boll_std = pd.Series(close).rolling(20).std()
    df["boll_mid"] = boll_mid.values
    df["boll_upper"] = (boll_mid + 2 * boll_std).values
    df["boll_lower"] = (boll_mid - 2 * boll_std).values
    
    # 成交量均线
    df["volume_ma5"] = pd.Series(volume).rolling(5).mean().values
    
    return df


def save_to_parquet(df, symbol, directory=DAILY_DIR):
    """保存 DataFrame 为 Parquet 文件"""
    if df is None or len(df) == 0:
        return False
    
    filepath = os.path.join(directory, f"{symbol}.parquet")
    
    # 标准化列顺序
    cols = [c for c in df.columns if c not in ["symbol"]]
    cols = ["symbol"] + cols
    
    table = pa.Table.from_pandas(df[cols], preserve_index=False)
    pq.write_table(table, filepath, compression="zstd")
    
    return True


def load_progress():
    """加载下载进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_update": None}


def save_progress(progress):
    """保存下载进度"""
    progress["last_update"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def download_csi300(mode="test"):
    """下载沪深300成分股数据（用于验证）"""
    logger.info(f"=== [模式: {mode}] 沪深300成分股数据下载 ===")
    
    try:
        # 获取沪深300成分股
        csi300 = ak.index_stock_cons_csindex("000300")
        symbols = csi300["成分券代码"].tolist()
        names = csi300.get("成分券名称", [""] * len(symbols))
        
        logger.info(f"沪深300成分股: {len(symbols)} 只")
        
        # 保存成分股列表
        csi300_df = pd.DataFrame({
            "symbol": [f"sh{s}" if s.startswith("6") else f"sz{s}" for s in symbols],
            "name": names,
            "raw_code": symbols
        })
        csi300_df.to_parquet(CSI300_FILE)
        
        progress = load_progress()
        
        for i, (raw_code, name) in enumerate(zip(symbols, names)):
            symbol = f"sh{raw_code}" if raw_code.startswith("6") else f"sz{raw_code}"
            
            if symbol in progress.get("completed", []):
                logger.info(f"  [{i+1}/{len(symbols)}] {symbol} {name} - 已下载，跳过")
                continue
            
            df = download_daily_kline(symbol, f"{i+1}/{len(symbols)} {name}")
            
            if df is not None:
                df = compute_technical_indicators(df)
                save_to_parquet(df, symbol)
                progress.setdefault("completed", []).append(symbol)
                if symbol in progress.get("failed", []):
                    progress["failed"].remove(symbol)
            else:
                progress.setdefault("failed", []).append(symbol)
            
            save_progress(progress)
            
            # 限速
            time.sleep(REQUEST_INTERVAL)
            
            # 测试模式只跑5只
            if mode == "test" and i >= 4:
                logger.info("测试模式：已下载5只，停止")
                break
        
        logger.info(f"沪深300下载完成！成功: {len(progress.get('completed', []))}, 失败: {len(progress.get('failed', []))}")
        return progress
        
    except Exception as e:
        logger.error(f"下载沪深300失败: {e}")
        return None


def download_all_a_shares():
    """下载全量A股日K线数据"""
    logger.info("=== 全量A股数据下载 ===")
    
    # 获取所有股票列表
    stock_list = get_all_stock_list()
    
    # 保存股票列表
    stock_list.to_parquet(os.path.join(INFO_DIR, "all_stocks.parquet"))
    logger.info(f"股票列表已保存: {len(stock_list)} 只")
    
    progress = load_progress()
    
    total = len(stock_list)
    for i, row in stock_list.iterrows():
        symbol = row["symbol"]
        name = row["name"]
        
        if symbol in progress.get("completed", []):
            if (i + 1) % 500 == 0:
                logger.info(f"[{i+1}/{total}] 进度: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
            continue
        
        df = download_daily_kline(symbol, f"{i+1}/{total} {name}")
        
        if df is not None:
            df = compute_technical_indicators(df)
            save_to_parquet(df, symbol)
            progress.setdefault("completed", []).append(symbol)
            if symbol in progress.get("failed", []):
                progress["failed"].remove(symbol)
        else:
            progress.setdefault("failed", []).append(symbol)
        
        # 每500个保存一次进度
        if (i + 1) % 500 == 0:
            save_progress(progress)
            logger.info(f"[{i+1}/{total}] 进度: {i+1}/{total} ({100*(i+1)/total:.1f}%), "
                       f"成功: {len(progress['completed'])}, 失败: {len(progress.get('failed', []))}")
        
        # 限速
        time.sleep(REQUEST_INTERVAL)
    
    # 最后保存一次
    save_progress(progress)
    
    logger.info(f"全量下载完成！总: {total}, 成功: {len(progress.get('completed', []))}, 失败: {len(progress.get('failed', []))}")


def download_financial_data(symbols):
    """下载财务数据（PE、PB、ROE、营收增速）"""
    logger.info("=== 财务数据下载 ===")
    
    financial_data = []
    
    for i, symbol in enumerate(symbols):
        code = symbol.replace("sh", "").replace("sz", "").strip()
        
        try:
            # 获取最新财务指标
            df = ak.stock_a_lg_indicator(symbol=code)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                financial_data.append({
                    "symbol": symbol,
                    "date": latest.get("日期", None),
                    "pe": latest.get("市盈率-动态", None),
                    "pb": latest.get("市净率", None),
                    "roe": latest.get("净资产收益率", None),
                    "revenue_growth": latest.get("营业收入同比增长率", None),
                    "profit_growth": latest.get("净利润同比增长率", None),
                    "gross_margin": latest.get("销售毛利率", None),
                    "net_margin": latest.get("销售净利率", None),
                })
        except Exception as e:
            logger.warning(f"  [{symbol}] 财务数据失败: {e}")
        
        if (i + 1) % 500 == 0:
            logger.info(f"  财务进度: {i+1}/{len(symbols)}")
        
        time.sleep(REQUEST_INTERVAL / 2)
    
    if financial_data:
        df = pd.DataFrame(financial_data)
        df.to_parquet(os.path.join(FINANCIAL_DIR, "financial_indicators.parquet"))
        logger.info(f"财务数据已保存: {len(df)} 条")
    else:
        logger.warning("无财务数据")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股票数据下载工具")
    parser.add_argument("mode", choices=["test", "csi300", "full", "financial"], 
                       default="test", nargs="?",
                       help="test=测试5只, csi300=沪深300, full=全量, financial=财务数据")
    args = parser.parse_args()
    
    if args.mode == "test":
        download_csi300(mode="test")
    elif args.mode == "csi300":
        download_csi300(mode="full")
    elif args.mode == "full":
        download_all_a_shares()
    elif args.mode == "financial":
        # 先获取股票列表
        stock_list = pd.read_parquet(os.path.join(INFO_DIR, "all_stocks.parquet"))
        download_financial_data(stock_list["symbol"].tolist())
