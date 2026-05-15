#!/usr/bin/env python3
"""
策略引擎 — 技术指标 + 基本面因子 + 多因子打分 + 买卖信号
依赖: pandas, pyarrow, numpy
数据源: Parquet 文件（由 download_data_v2.py 生成）
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import json
from datetime import datetime, timedelta
from scripts.data_loader import (
    load_industry_mapping, get_industry,
    load_industry_momentum, get_industry_momentum_score,
)

DATA_DIR = os.path.expanduser("~/stock-system/data")
INDUSTRY_TOP30_PATH = os.path.join(DATA_DIR, "industry_top30.parquet")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
INFO_DIR = os.path.join(DATA_DIR, "info")
FINANCIAL_DIR = os.path.join(DATA_DIR, "financial")


# ═══════════════════════════════════════════════
#  1. 技术指标计算
# ═══════════════════════════════════════════════

class TechnicalAnalyzer:
    """从 DataFrame 计算技术指标"""
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """计算全部技术指标"""
        if df is None or len(df) < 20:
            return df
        
        df = df.copy().sort_values("date")
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        
        # MA 均线
        for period in [5, 10, 20, 60, 120]:
            df[f"ma{period}"] = pd.Series(close).rolling(period).mean().values
        
        # MACD (12, 26, 9)
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd_dif = ema12 - ema26
        macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
        df["macd_dif"] = macd_dif.values
        df["macd_dea"] = macd_dea.values
        df["macd_hist"] = (2 * (macd_dif - macd_dea)).values
        
        # RSI(14)
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = (100 - (100 / (1 + rs))).values
        
        # KDJ(9, 3, 3)
        low_9 = pd.Series(low).rolling(9).min()
        high_9 = pd.Series(high).rolling(9).max()
        rsv = (pd.Series(close) - low_9) / (high_9 - low_9) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        df["kdj_k"] = k.values
        df["kdj_d"] = d.values
        df["kdj_j"] = (3 * k - 2 * d).values
        
        # 布林带(20, 2)
        boll_mid = pd.Series(close).rolling(20).mean()
        boll_std = pd.Series(close).rolling(20).std()
        df["boll_mid"] = boll_mid.values
        df["boll_upper"] = (boll_mid + 2 * boll_std).values
        df["boll_lower"] = (boll_mid - 2 * boll_std).values
        
        # 成交量均线
        df["volume_ma5"] = pd.Series(volume).rolling(5).mean().values
        
        return df
    
    @staticmethod
    def get_latest(df: pd.DataFrame) -> dict:
        """获取最新一行的所有指标"""
        if df is None or len(df) == 0:
            return {}
        latest = df.iloc[-1].to_dict()
        return {k: v for k, v in latest.items() if not pd.isna(v)}


# ═══════════════════════════════════════════════
#  2. 股票数据加载
# ═══════════════════════════════════════════════

def load_stock_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """加载单只股票数据，返回带技术指标的 DataFrame
    years=0 表示加载全量历史数据
    """
    filepath = os.path.join(DAILY_DIR, f"{symbol}.parquet")
    if not os.path.exists(filepath):
        return None
    
    df = pq.read_table(filepath).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    
    if years > 0:
        # 只保留最近 N 年
        cutoff = datetime.now() - timedelta(days=years * 365)
        df = df[df["date"] >= cutoff].copy()
    # years=0 → 加载全量数据，不过滤
    
    return TechnicalAnalyzer.compute_all(df)


def load_all_stocks_info() -> pd.DataFrame:
    """加载股票基本信息"""
    path = os.path.join(INFO_DIR, "all_stocks.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_financial_data() -> pd.DataFrame:
    """加载财务数据"""
    path = os.path.join(FINANCIAL_DIR, "financial_indicators.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def get_available_symbols() -> list:
    """获取已下载的股票列表"""
    files = os.listdir(DAILY_DIR)
    return sorted([f.replace(".parquet", "") for f in files if f.endswith(".parquet")])


# ═══════════════════════════════════════════════
#  3. 多因子评分模型
# ═══════════════════════════════════════════════

class ScoringEngine:
    """
    多因子评分模型
    
    两种模式:
    - classic (经典版): 技术面(40%) + 基本面(40%) + 资金面(20%)
    - enhanced (增强版): 技术面(30%) + 价量衍生(25%) + 行业动量(20%) + 热点(15%) + 资金(10%)
    """
    
    SCORING_MODE_CLASSIC = "classic"
    SCORING_MODE_ENHANCED = "enhanced"
    
    WEIGHTS = {
        "pe_percentile": 0.16,    # PE历史分位（低=好）基本面40%×40%
        "roe": 0.12,              # ROE（高=好）基本面40%×30%
        "revenue_growth": 0.12,   # 营收增速（高=好）基本面40%×30%
        "ma_trend": 0.24,         # 均线排列（多头=好）技术面40%×60%
        "rsi_position": 0.06,     # RSI位置（中性偏强=好）技术面40%×15%
        "volume_ratio": 0.06,     # 成交量比（放量=好）技术面40%×15%
        "north_flow": 0.20,       # 北向资金 资金面20%
        "macd_signal": 0.04,      # MACD信号 技术面40%×10%
    }
    
    @classmethod
    def score_stock(cls, df: pd.DataFrame, financial: dict = None,
                    mode: str = SCORING_MODE_CLASSIC,
                    symbol: str = None) -> dict:
        """
        对单只股票评分
        返回: {score, details, signals, latest, mode}
        """
        if df is None or len(df) < 60:
            return {"score": 0, "details": {}, "signals": [], "latest": {"close": 0}, "mode": mode}
        
        if mode == cls.SCORING_MODE_ENHANCED:
            return cls._score_enhanced(df, financial, symbol)
        else:
            return cls._score_classic(df, financial)
    
    @classmethod
    def _score_classic(cls, df: pd.DataFrame, financial: dict = None) -> dict:
        """经典版评分 (原有逻辑完全保留)"""
        latest = TechnicalAnalyzer.get_latest(df)
        details = {}
        signals = []
        
        # ── 均线趋势评分 ──
        ma_trend = cls._score_ma_trend(latest)
        details["ma_trend"] = {"score": ma_trend, "weight": cls.WEIGHTS["ma_trend"]}
        
        # ── RSI位置评分 ──
        rsi_score = cls._score_rsi(latest)
        details["rsi_position"] = {"score": rsi_score, "weight": cls.WEIGHTS["rsi_position"]}
        
        # ── 成交量比 ──
        volume_score = cls._score_volume(latest)
        details["volume_ratio"] = {"score": volume_score, "weight": cls.WEIGHTS["volume_ratio"]}
        
        # ── MACD信号 ──
        macd_score, macd_signal = cls._score_macd(latest)
        details["macd_signal"] = {"score": macd_score, "weight": cls.WEIGHTS["macd_signal"]}
        if macd_signal:
            signals.append(macd_signal)
        
        # ── PE历史分位（如果有基本面数据） ──
        pe_score = cls._score_pe(financial or {}, df)
        details["pe_percentile"] = {"score": pe_score, "weight": cls.WEIGHTS["pe_percentile"]}
        
        # ── ROE评分 ──
        roe_score = cls._score_roe(financial or {})
        details["roe"] = {"score": roe_score, "weight": cls.WEIGHTS["roe"]}
        
        # ── 营收增速 ──
        rev_score = cls._score_revenue_growth(financial or {})
        details["revenue_growth"] = {"score": rev_score, "weight": cls.WEIGHTS["revenue_growth"]}

        # ── 北向资金 ──
        north_score = cls._score_north_flow(financial or {})
        details["north_flow"] = {"score": north_score, "weight": cls.WEIGHTS["north_flow"]}

        # 综合评分 (0~100)
        total = sum(v["score"] * v["weight"] for v in details.values()) * 10
        
        # 买入/卖出信号
        buy_signals, sell_signals = cls._generate_signals(latest, df, total, details)
        signals.extend(buy_signals)
        signals.extend(sell_signals)
        
        return {
            "score": round(total, 1),
            "details": details,
            "signals": signals,
            "latest": latest,
            "mode": cls.SCORING_MODE_CLASSIC,
        }
    
    @staticmethod
    def _score_ma_trend(latest: dict) -> float:
        """均线趋势评分 (0~10)"""
        # 多头排列: MA5 > MA10 > MA20 > MA60 = 10分
        # 粘合震荡: = 5分
        # 空头排列: = 0分
        try:
            ma5 = latest.get("ma5", 0) or 0
            ma10 = latest.get("ma10", 0) or 0
            ma20 = latest.get("ma20", 0) or 0
            ma60 = latest.get("ma60", 0) or 0
            
            if ma5 > ma10 > ma20 > ma60:
                return 10.0
            elif ma5 > ma10 > ma20:
                return 8.0
            elif ma10 > ma20 > ma60:
                # 短期整理，长期向上
                close = latest.get("close", 0) or 0
                return 7.0 if close > ma20 else 5.0
            elif ma5 < ma10 < ma20 < ma60:
                return 0.0
            elif ma10 < ma20 < ma60:
                return 2.0
            else:
                return 4.0
        except:
            return 5.0
    
    @staticmethod
    def _score_rsi(latest: dict) -> float:
        """RSI位置评分 (0~10)"""
        rsi = latest.get("rsi", 50) or 50
        if 40 <= rsi <= 60:
            return 10.0  # 中性区域，上涨空间大
        elif 30 <= rsi < 40:
            return 8.0   # 偏弱但接近超卖
        elif 60 < rsi <= 70:
            return 7.0   # 偏强但未过热
        elif rsi < 30:
            return 6.0   # 超卖，可能反弹
        elif 70 < rsi <= 80:
            return 3.0   # 过热
        else:
            return 0.0   # 极度过热/超卖
    
    @staticmethod
    def _score_volume(latest: dict) -> float:
        """成交量比评分 (0~10)"""
        vol = latest.get("volume", 0) or 0
        vol_ma5 = latest.get("volume_ma5", 1) or 1
        if vol_ma5 == 0:
            return 5.0
        ratio = vol / vol_ma5
        if ratio > 2.0:
            return 10.0  # 显著放量
        elif ratio > 1.5:
            return 8.0
        elif ratio > 1.0:
            return 6.0
        elif ratio > 0.7:
            return 4.0
        else:
            return 2.0  # 缩量
    
    @staticmethod
    def _score_macd(latest: dict) -> tuple:
        """MACD信号评分 + 信号"""
        dif = latest.get("macd_dif", 0) or 0
        dea = latest.get("macd_dea", 0) or 0
        hist = latest.get("macd_hist", 0) or 0
        
        signal = None
        if dif > 0 and dea > 0 and hist > 0:
            score = 10.0  # 多头强势
        elif dif > 0 and hist > 0 and abs(hist) < abs(latest.get("macd_dif", 1)):
            score = 8.0   # 多头，DIFF在DEA上方
        elif dif > 0 and dea > 0:
            score = 7.0   # 多头但动能减弱
        elif dif < 0 and dea < 0 and hist < 0:
            score = 2.0   # 空头
        elif dif > dea:
            score = 6.0   # 刚金叉或即将金叉
            signal = "macd_golden_cross"
        elif dif < dea:
            score = 4.0   # 死叉
            signal = "macd_death_cross"
        else:
            score = 5.0
        
        return score, signal
    
    @staticmethod
    def _score_pe(financial: dict, df: pd.DataFrame) -> float:
        """PE历史分位评分 (0~10)，PE低=高分"""
        pe = financial.get("pe", None)
        if pe is None or pe <= 0:
            return 5.0  # 未知，中性
        
        # 计算PE历史分位（从K线数据估算）
        # 实际上需要历史PE数据，这里用PB/股价替代
        close_values = df["close"].values
        current_price = close_values[-1]
        price_percentile = np.sum(close_values <= current_price) / len(close_values)
        
        # 股价在历史低位 = 好，高位 = 差
        if price_percentile < 0.3:
            return 8.0
        elif price_percentile < 0.5:
            return 6.0
        elif price_percentile < 0.7:
            return 4.0
        else:
            return 2.0
    
    @staticmethod
    def _score_roe(financial: dict) -> float:
        """ROE评分 (0~10)"""
        roe = financial.get("roe", None)
        if roe is None:
            return 5.0
        if roe >= 20:
            return 10.0
        elif roe >= 15:
            return 8.0
        elif roe >= 10:
            return 6.0
        elif roe >= 5:
            return 4.0
        else:
            return 2.0
    
    @staticmethod
    def _score_revenue_growth(financial: dict) -> float:
        """营收增速评分 (0~10)"""
        growth = financial.get("revenue_growth", None)
        if growth is None:
            return 5.0
        if growth >= 30:
            return 10.0
        elif growth >= 20:
            return 8.0
        elif growth >= 10:
            return 6.0
        elif growth >= 0:
            return 4.0
        else:
            return 1.0

    @staticmethod
    def _score_north_flow(financial: dict) -> float:
        """北向资金评分 (0~10) — 预留占位，待接入北向数据"""
        return 5.0

    @staticmethod
    def _generate_signals(latest: dict, df: pd.DataFrame,
                          total_score: float, details: dict) -> tuple:
        """生成买卖信号"""
        buy = []
        sell = []
        
        close = latest.get("close", 0) or 0
        ma20 = latest.get("ma20", 0) or 0
        ma60 = latest.get("ma60", 0) or 0
        rsi = latest.get("rsi", 50) or 50
        
        # 买入信号
        if total_score >= 70:
            if close > ma20 > 0:
                buy.append("strong_buy")
        
        if 60 <= total_score < 70:
            if close > ma20 > 0 and close > ma60 > 0:
                buy.append("watch_buy")
        
        # 卖出信号
        if close < ma60 and ma60 > 0:
            sell.append("trend_broken")
        
        if rsi > 80:
            sell.append("overbought")
        
        # MACD顶背离（简化判断）
        if len(df) >= 60:
            high_30 = df["high"].iloc[-30:].max()
            high_60 = df["high"].iloc[-60:-30].max()
            macd_30 = df["macd_dif"].iloc[-30:].max()
            macd_60 = df["macd_dif"].iloc[-60:-30].max()
            
            if high_30 > high_60 and macd_30 < macd_60:
                sell.append("macd_divergence")
        
        return buy, sell

    # ═══════════════════════════════════════════════
    #  增强版评分 (Phase 1-2 逐步完善)
    # ═══════════════════════════════════════════════

    @classmethod
    def _score_enhanced(cls, df: pd.DataFrame, financial: dict = None,
                        symbol: str = None) -> dict:
        """增强版多因子评分"""
        latest = TechnicalAnalyzer.get_latest(df)
        details = {}
        signals = []
        
        # 1. 技术面 (30%)
        ma_trend = cls._score_ma_trend(latest)
        rsi_score = cls._score_rsi(latest)
        volume_score = cls._score_volume(latest)
        macd_score, macd_signal = cls._score_macd(latest)
        if macd_signal:
            signals.append(macd_signal)
        
        tech_score = ma_trend * 0.50 + rsi_score * 0.20 + volume_score * 0.15 + macd_score * 0.15
        details["technical"] = {"score": tech_score, "weight": 0.30}
        
        # 2. 价量衍生 (25%) — Phase 2 实现
        derived_factors = cls._calc_price_derived_factors(df)
        derived_score = 0
        if derived_factors:
            derived_score = derived_factors.get("_total", 5.0)
            for k, v in derived_factors.items():
                if not k.startswith("_"):
                    details[f"derived_{k}"] = {"score": v, "weight": 0}
        
        # 如果没有行业数据/动量数据，价量衍生权重提高到45%
        details["price_derived"] = {"score": derived_score, "weight": 0.25}
        
        # 3. 行业动量 (20%) — Phase 1 实现
        industry_score = 5.0
        industry_name = None
        if symbol:
            mapping = load_industry_mapping()
            if mapping is not None:
                industry_name = get_industry(symbol, mapping)
                if industry_name:
                    momentum = load_industry_momentum()
                    industry_score = get_industry_momentum_score(industry_name, momentum)
                    details["industry"] = {"score": industry_score, "weight": 0.20}
        
        # 4. 市场热点 (15%) — 基于行业板块动量
        hotspot_score = 5.0
        hotspot_detail = ""
        if symbol and industry_name:
            # 用行业动量数据作为热点评分
            momentum = load_industry_momentum()
            if momentum is not None and industry_name in momentum["industry"].values:
                sorted_m = momentum.sort_values("return_20d", ascending=False).reset_index(drop=True)
                match_idx = sorted_m[sorted_m["industry"] == industry_name].index[0]
                total = len(sorted_m)
                pct = (match_idx + 1) / total
                # 排名越靠前，热点评分越高
                if pct <= 0.1:
                    hotspot_score = 10.0
                    hotspot_detail = f"TOP10%({industry_name})"
                elif pct <= 0.25:
                    hotspot_score = 8.5
                    hotspot_detail = f"TOP25%({industry_name})"
                elif pct <= 0.5:
                    hotspot_score = 6.5
                    hotspot_detail = f"TOP50%({industry_name})"
                elif pct <= 0.75:
                    hotspot_score = 4.5
                    hotspot_detail = f"后50%({industry_name})"
                else:
                    hotspot_score = 2.5
                    hotspot_detail = f"后25%({industry_name})"
        
        details["hotspot"] = {"score": hotspot_score, "weight": 0.15}
        
        # 5. 资金面 (10%)
        fund_score = 5.0
        details["fund_flow"] = {"score": fund_score, "weight": 0.10}
        
        # 综合评分 (0~100)
        total = sum(v["score"] * v["weight"] for v in details.values()) * 10
        
        # 买入/卖出信号（沿用经典版规则）
        buy_signals, sell_signals = cls._generate_signals(latest, df, total, details)
        signals.extend(buy_signals)
        signals.extend(sell_signals)
        
        return {
            "score": round(total, 1),
            "details": details,
            "signals": signals,
            "latest": latest,
            "mode": cls.SCORING_MODE_ENHANCED,
            "industry": industry_name,
        }

    @staticmethod
    def _calc_price_derived_raw(df: pd.DataFrame) -> dict:
        """从K线计算价量衍生因子原始值"""
        if df is None or len(df) < 20:
            return {}

        latest = df.iloc[-1]
        close = latest["close"]
        ma5 = latest.get("ma5")
        ma20 = latest.get("ma20")
        volume = latest.get("volume", 0) or 0
        volume_ma5 = latest.get("volume_ma5", 1) or 1

        # 1. 乖离率 (Bias) — (收盘价 - MA20) / MA20
        bias = (close - ma20) / ma20 * 100 if (ma20 and ma20 > 0) else 0.0

        # 2. 趋势强度 (MA5相对MA20的位置)
        trend_strength = (ma5 - ma20) / ma20 * 100 if (ma5 and ma20 and ma20 > 0) else 0.0

        # 3. 波动率 (20日年化)
        returns = df["close"].pct_change().dropna()
        vol = float(returns.tail(20).std() * (252 ** 0.5) * 100)

        # 4. 换手率比率 (成交量 / 均值，衡量放/缩量程度)
        turnover_ratio = volume / volume_ma5 if volume_ma5 > 0 else 1.0

        return {
            "bias": bias,
            "trend_strength": trend_strength,
            "volatility": vol,
            "turnover_ratio": turnover_ratio,
        }

    @staticmethod
    def _score_price_derived(factors: dict) -> dict:
        """价量衍生因子评分 (每项 0-10)，返回带 _total 的 dict"""
        scores = {}

        # ── 乖离率: ±3%最优，越远越低 ──
        bias = abs(factors.get("bias", 0))
        if bias <= 3:
            scores["bias"] = 10.0 - bias / 3 * 2          # 10~8
        elif bias <= 10:
            scores["bias"] = 8.0 - (bias - 3) / 7 * 3     # 8~5
        elif bias <= 20:
            scores["bias"] = 5.0 - (bias - 10) / 10 * 3   # 5~2
        else:
            scores["bias"] = max(0.0, 2.0 - (bias - 20) / 10 * 2)  # 2~0

        # ── 趋势强度: 正值加分，负值减分 ──
        ts = factors.get("trend_strength", 0)
        if ts >= 5:
            scores["trend_strength"] = 10.0
        elif ts >= 0:
            scores["trend_strength"] = 5.0 + ts / 5 * 5   # 5~10
        elif ts >= -5:
            scores["trend_strength"] = 5.0 - abs(ts) / 5 * 3  # 5~2
        else:
            scores["trend_strength"] = max(0.0, 2.0 - (abs(ts) - 5) / 10 * 2)  # 2~0

        # ── 波动率: 适中最好(15%~30%) ──
        vol = factors.get("volatility", 20)
        if vol <= 15:
            scores["volatility"] = 8.0                     # 波动太低，不活跃
        elif vol <= 30:
            scores["volatility"] = 10.0 - (vol - 15) / 15 * 2  # 10~8
        elif vol <= 50:
            scores["volatility"] = 8.0 - (vol - 30) / 20 * 4  # 8~4
        else:
            scores["volatility"] = max(0.0, 4.0 - (vol - 50) / 30 * 4)  # 4~0

        # ── 换手率比率: 1~3倍为合理放量 ──
        tr = factors.get("turnover_ratio", 1.0)
        if 1.0 <= tr <= 3.0:
            scores["turnover_ratio"] = 10.0
        elif 0.7 <= tr < 1.0:
            scores["turnover_ratio"] = 7.0 + (tr - 0.7) / 0.3 * 3   # 7~10
        elif 3.0 < tr <= 5.0:
            scores["turnover_ratio"] = 10.0 - (tr - 3.0) / 2.0 * 4  # 10~6
        elif 0.3 <= tr < 0.7:
            scores["turnover_ratio"] = 3.0 + (tr - 0.3) / 0.4 * 4   # 3~7
        else:
            scores["turnover_ratio"] = max(0.0, 3.0 - abs(tr - (0.0 if tr < 0.3 else 5.0)) / 5 * 3)

        # 综合：乖离率35% + 趋势强度35% + 波动率15% + 换手率15%
        scores["_total"] = (
            scores["bias"] * 0.35
            + scores["trend_strength"] * 0.35
            + scores["volatility"] * 0.15
            + scores["turnover_ratio"] * 0.15
        )

        return scores

    @staticmethod
    def _calc_price_derived_factors(df: pd.DataFrame) -> dict:
        """价量衍生因子（Phase 2 完整实现）
        返回: {bias: 0-10, trend_strength: 0-10, volatility: 0-10,
               turnover_ratio: 0-10, _total: 0-10}
        """
        raw = ScoringEngine._calc_price_derived_raw(df)
        if not raw:
            return {"_total": 5.0}
        return ScoringEngine._score_price_derived(raw)


# ═══════════════════════════════════════════════
#  4. 全市场扫描
# ═══════════════════════════════════════════════

def scan_all_stocks(top_n: int = 50, min_score: float = 60,
                    mode: str = "classic",
                    symbols_override: list = None) -> pd.DataFrame:
    """
    扫描全市场，返回评分排序结果
    symbols_override: 指定股票列表（用于按行业筛选等场景）
    """
    if symbols_override is not None:
        symbols = symbols_override
    else:
        symbols = get_available_symbols()
    if not symbols:
        return pd.DataFrame()
    
    stocks_info = load_all_stocks_info()
    financial = load_financial_data()
    
    # 两层过滤：第一层基本面筛选
    filtered_symbols = symbols
    try:
        from scripts.financial_data import filter_by_fundamentals
        qualified = filter_by_fundamentals(symbols, financial_df=financial, relaxed=True)
        if qualified:
            filtered_symbols = qualified
    except ImportError:
        pass
    
    results = []
    total = len(filtered_symbols)
    
    print(f"扫描: {len(symbols)}只 → 基本面筛选后 {len(filtered_symbols)}只")
    
    for i, symbol in enumerate(filtered_symbols):
        if (i + 1) % 500 == 0:
            print(f"  扫描进度: {i+1}/{total}")
        
        df = load_stock_data(symbol)
        if df is None:
            continue
        
        # 获取基本面数据
        fin = {}
        if financial is not None:
            match = financial[financial["symbol"] == symbol]
            if len(match) > 0:
                fin = match.iloc[-1].to_dict()
        
        # 评分
        result = ScoringEngine.score_stock(df, fin, mode=mode, symbol=symbol)
        
        # 获取股票名称
        name = symbol
        if stocks_info is not None:
            match = stocks_info[stocks_info["symbol"] == symbol]
            if len(match) > 0:
                name = match.iloc[0].get("code_name", symbol)
        
        entry = {
            "symbol": symbol,
            "name": name,
            "score": result["score"],
            "price": result["latest"].get("close", 0),
            "change_pct": result["latest"].get("change_pct", 0),
            "ma_trend": result["details"].get("ma_trend", {}).get("score", 
                       result["details"].get("technical", {}).get("score", 0)),
            "rsi": result["latest"].get("rsi", 0),
            "volume_ratio": result["latest"].get("volume", 1) / max(result["latest"].get("volume_ma5", 1), 1),
            "signals": ",".join(result["signals"]),
        }
        
        # 行业归属（双模式统一添加）
        try:
            entry["industry"] = get_industry(symbol, load_industry_mapping())
        except:
            entry["industry"] = ""
        
        # 增强版附加字段
        if mode == ScoringEngine.SCORING_MODE_ENHANCED:
            entry["mode"] = "enhanced"
            for k, v in result["details"].items():
                if k not in ("ma_trend", "rsi_position", "volume_ratio", "macd_signal",
                             "pe_percentile", "roe", "revenue_growth", "north_flow"):
                    entry[f"detail_{k}"] = round(v["score"], 1)
        else:
            entry["mode"] = "classic"
        
        results.append(entry)
    
    df_result = pd.DataFrame(results)
    if len(df_result) == 0:
        return df_result
    
    # 按评分排序
    df_result = df_result.sort_values("score", ascending=False).reset_index(drop=True)
    
    # 截取前N
    result = df_result.head(top_n)
    
    # 增强版：缓存每个行业的前30只股票
    if mode == "enhanced" and "industry" in df_result.columns:
        try:
            top_per_industry = df_result.groupby("industry").apply(
                lambda g: g.head(30)
            ).reset_index()
            # drop多余索引列 (level_1)
            top_per_industry = top_per_industry.drop(columns=[c for c in top_per_industry.columns if c.startswith("level_")], errors="ignore")
            # 调整列顺序: industry在首位
            cols = ["industry"] + [c for c in top_per_industry.columns if c != "industry"]
            top_per_industry = top_per_industry[cols]
            top_per_industry.to_parquet(INDUSTRY_TOP30_PATH, compression="zstd")
        except Exception:
            import traceback; traceback.print_exc()
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        print("=== 全市场扫描 ===")
        results = scan_all_stocks(top_n=50)
        if len(results) > 0:
            print(f"\n扫描完成: {len(results)} 只")
            print(results[["symbol", "name", "score", "price", "signals"]].to_string(index=False))
        else:
            print("暂无数据，请先下载股票数据")
