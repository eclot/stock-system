#!/usr/bin/env python3
"""
股票数据获取脚本 v2
- 数据源: Baostock（稳定可靠）
- 范围: 全量A股日K线（上市日起至最新）
- 股票基础信息（名称、行业、上市日期）
- 存储格式: Parquet（zstd压缩）
- 技术指标: MA、MACD、RSI、KDJ、布林带（预计算）
- 重试机制: 失败自动重试3次
"""

import baostock as bs
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta

# ========== 配置 ==========
DATA_DIR = os.path.expanduser("~/stock-system/data")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
INFO_DIR = os.path.join(DATA_DIR, "info")
FINANCIAL_DIR = os.path.join(DATA_DIR, "financial")
PROGRESS_FILE = os.path.join(DATA_DIR, "download_progress.json")

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

MAX_RETRIES = 3
REQUEST_DELAY = 0.2  # 请求间隔


def bs_login():
    """登录 Baostock"""
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"Baostock登录失败: {lg.error_msg}")
    return lg


def get_all_stock_list():
    """获取全量A股股票列表"""
    bs_login()
    rs = bs.query_all_stock(datetime.now().strftime("%Y-%m-%d"))
    
    stocks = []
    while rs.next():
        row = rs.get_row_data()
        code = row[0]         # sh.600519
        name = row[2]
        
        # 排除指数，只保留真实A股
        # 上证A股: sh.6xxxxx, sh.688xxx
        # 深证A股: sz.0xxxxx, sz.3xxxxx, sz.2xxxxx
        is_real_stock = (
            (code.startswith("sh.") and code[3:].isdigit() and
             (code[3:5] in ("60", "68"))) or
            (code.startswith("sz.") and code[3:].isdigit() and
             code[3] in ("0", "2", "3") and not code[4:].startswith("99"))
        )
        
        if is_real_stock and "指数" not in name and row[1] == "1":
            stocks.append({
                "code": code,
                "code_name": name,
            })
    
    df = pd.DataFrame(stocks)
    
    # 只保留正常上市的股票
    active = df  # is_real_stock + tradeStatus=1 already filtered above
    
    # 转为 Hermes 格式（去掉点）
    active["symbol"] = active["code"].str.replace(".", "")
    
    logger.info(f"获取到 {len(active)} 只真实A股")
    bs.logout()
    return active


def download_daily_kline(symbol, code, name="", max_retries=MAX_RETRIES):
    """
    下载单只股票日K线（前复权）
    使用 Baostock
    """
    errors = []
    
    for attempt in range(1, max_retries + 1):
        try:
            # 获取从上市到现在的所有数据
            rs = bs.query_history_k_data_plus(
                code,  # sh.600519 格式
                "date,open,high,low,close,volume,amount,adjustflag",
                start_date="1990-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2"  # 2=前复权
            )
            
            if rs.error_code != "0":
                raise Exception(f"查询失败: {rs.error_msg}")
            
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            
            if not rows:
                logger.warning(f"  [{symbol}] {name} - 无数据")
                return None
            
            df = pd.DataFrame(rows, columns=rs.fields)
            
            # 过滤无效行（Baostock有时返回空行）
            df = df[df["date"] != ""].copy()
            
            if len(df) == 0:
                logger.warning(f"  [{symbol}] {name} - 有效数据为空")
                return None
            
            # 转换数据类型
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # 去重（按日期）
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            # 添加股票代码
            df["symbol"] = symbol
            
            # 日期排序
            df["date"] = pd.to_datetime(df["date"])
            
            # 只保留基础数据（技术指标由策略引擎实时计算）
            keep_cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in keep_cols if c in df.columns]]
            
            logger.info(f"  [{symbol}] {name} - {len(df)} 条 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
            return df
            
        except Exception as e:
            errors.append(str(e))
            logger.warning(f"  [{symbol}] 第{attempt}次失败: {e}")
            
            # 重试前重新登录
            try:
                bs.logout()
            except:
                pass
            try:
                bs_login()
            except:
                pass
            
            if attempt < max_retries:
                time.sleep(REQUEST_DELAY * 2)
            else:
                logger.error(f"  [{symbol}] {name} - 重试{max_retries}次均失败: {errors[-1]}")
                return None


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
    rs_ = gain / loss
    df["rsi"] = (100 - (100 / (1 + rs_))).values
    
    # KDJ(9,3,3)
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


def save_to_parquet(df, symbol):
    """保存 DataFrame 为 Parquet 文件"""
    if df is None or len(df) == 0:
        return False
    
    filepath = os.path.join(DAILY_DIR, f"{symbol}.parquet")
    
    cols = [c for c in df.columns if c not in ["symbol", "adjustflag", "code"]]
    cols = ["symbol", "date"] + [c for c in cols if c != "date"]
    cols = [c for c in cols if c in df.columns]
    
    table = pa.Table.from_pandas(df[cols], preserve_index=False)
    pq.write_table(table, filepath, compression="zstd")
    
    return True


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_update": None}


def save_progress(progress):
    progress["last_update"] = datetime.now().isoformat()
    progress["total_completed"] = len(progress.get("completed", []))
    progress["total_failed"] = len(progress.get("failed", []))
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def download_stock_data(stock_list, mode="csi300"):
    """下载股票列表数据"""
    progress = load_progress()
    total = len(stock_list)
    
    logger.info(f"\n=== 开始下载 {mode} ({total} 只) ===")
    
    # Baostock 全局登录（一次登录，多次查询）
    bs_login()
    
    success_count = 0
    fail_count = 0
    
    for i, row in stock_list.iterrows():
        code = row["code"]     # sh.600519
        symbol = row["symbol"]  # sh600519
        name = row.get("code_name", row.get("name", ""))
        
        if symbol in progress.get("completed", []):
            success_count += 1
            if (i + 1) % 100 == 0:
                logger.info(f"[{i+1}/{total}] {symbol} {name} - 已完成，跳过")
            continue
        
        df = download_daily_kline(symbol, code, f"{i+1}/{total} {name}")
        
        if df is not None:
            df = compute_technical_indicators(df)
            save_to_parquet(df, symbol)
            progress.setdefault("completed", []).append(symbol)
            if symbol in progress.get("failed", []):
                progress["failed"].remove(symbol)
            success_count += 1
        else:
            progress.setdefault("failed", []).append(symbol)
            fail_count += 1
        
        # 每50个保存一次进度
        if (i + 1) % 50 == 0:
            save_progress(progress)
            done = len(progress.get("completed", []))
            failed = len(progress.get("failed", []))
            logger.info(f"[{i+1}/{total}] 进度: {100*(i+1)/total:.1f}% | 成功: {done} | 失败: {failed}")
        
        time.sleep(REQUEST_DELAY)
    
    # Baostock 登出
    try:
        bs.logout()
    except:
        pass
    
    save_progress(progress)
    
    final_done = len(progress.get("completed", []))
    final_failed = len(progress.get("failed", []))
    logger.info(f"\n=== {mode} 下载完成！===")
    logger.info(f"总: {total} | 成功: {final_done} | 失败: {final_failed}")
    
    return progress


def download_csi300():
    """下载沪深300"""
    bs_login()
    rs = bs.query_hs300_stocks()
    
    stocks = []
    while rs.next():
        row = rs.get_row_data()
        code = row[0]  # sh.600519 or sz.000001
        symbol = code.replace(".", "")
        stocks.append({
            "code": code,
            "symbol": symbol,
            "code_name": row[1],
        })
    bs.logout()
    
    df = pd.DataFrame(stocks)
    logger.info(f"沪深300成分股: {len(df)} 只")
    
    # 保存成分股列表
    df.to_parquet(os.path.join(DATA_DIR, "csi300_list.parquet"))
    
    return download_stock_data(df, "沪深300")


def download_all_a_shares():
    """下载全量A股"""
    stock_list = get_all_stock_list()
    stock_list.to_parquet(os.path.join(INFO_DIR, "all_stocks.parquet"))
    return download_stock_data(stock_list, "全量A股")


def download_financial_data():
    """下载财务数据（PE、PB、ROE、营收增速）"""
    logger.info("\n=== 财务数据下载 ===")
    
    # 获取已下载的股票列表
    completed = load_progress().get("completed", [])
    if not completed:
        logger.warning("请先下载日K线数据")
        return
    
    stocks = completed
    logger.info(f"需要获取财务数据的股票: {len(stocks)} 只")
    
    financial_data = []
    bs_login()
    
    for i, symbol in enumerate(stocks):
        code = symbol[:2] + "." + symbol[2:]  # sh600519 -> sh.600519
        
        try:
            # 获取最近4个季度的财务指标
            rs = bs.query_stock_basic(code)
            if rs.next():
                info = rs.get_row_data()
            
            # 获取成长能力数据
            rs = bs.query_growth_data(code, year=datetime.now().year, quarter=1)
            growth = None
            if rs.next():
                growth = rs.get_row_data()
            
            if growth:
                financial_data.append({
                    "symbol": symbol,
                    "roe": float(growth[3]) if growth[3] != "" else None,
                    "revenue_growth": float(growth[4]) if growth[4] != "" else None,
                    "profit_growth": float(growth[5]) if growth[5] != "" else None,
                })
        except Exception as e:
            logger.warning(f"  [{symbol}] 财务数据失败: {e}")
        
        if (i + 1) % 500 == 0:
            logger.info(f"  财务进度: {i+1}/{len(stocks)}")
        
        time.sleep(REQUEST_DELAY / 2)
    
    bs.logout()
    
    if financial_data:
        df = pd.DataFrame(financial_data)
        df.to_parquet(os.path.join(FINANCIAL_DIR, "financial_indicators.parquet"))
        logger.info(f"财务数据已保存: {len(df)} 条")
    else:
        logger.warning("无财务数据")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股票数据下载工具 v2 (Baostock)")
    parser.add_argument("mode", choices=["test", "csi300", "full", "financial", "resume"],
                       default="test", nargs="?",
                       help="test=测试5只, csi300=沪深300, full=全量, financial=财务数据, resume=继续下载")
    args = parser.parse_args()
    
    try:
        if args.mode == "test":
            # 测试模式：取前5只沪深300
            bs_login()
            rs = bs.query_hs300_stocks()
            stocks = []
            count = 0
            while rs.next() and count < 5:
                r = rs.get_row_data()
                stocks.append({"code": r[1], "symbol": r[1].replace(".", ""), "code_name": r[2]})
                count += 1
            bs.logout()
            download_stock_data(pd.DataFrame(stocks), "测试(5只)")
        
        elif args.mode == "full":
            download_all_a_shares()
        
        elif args.mode == "financial":
            download_financial_data()
        
        elif args.mode == "resume":
            # 续传：获取所有A股列表中未完成的
            stock_list = pd.read_parquet(os.path.join(INFO_DIR, "all_stocks.parquet"))
            progress = load_progress()
            done = set(progress.get("completed", []))
            remaining = stock_list[~stock_list["symbol"].isin(done)].copy()
            logger.info(f"续传: 已完成{len(done)}, 剩余{len(remaining)}")
            if len(remaining) > 0:
                download_stock_data(remaining, f"续传({len(remaining)}只)")
            else:
                logger.info("全部已完成！")
        
        else:
            # 默认: csi300
            download_csi300()
            
    except KeyboardInterrupt:
        logger.info("\n用户中断，已保存当前进度")
        save_progress(load_progress())
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
