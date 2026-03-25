"""
Pure-pandas technical indicator implementations (no TA-Lib dependency).
All functions operate on a pd.Series or pd.DataFrame and return pd.Series.
"""
import numpy as np
import pandas as pd


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def bb_width(upper: pd.Series, lower: pd.Series, mid: pd.Series) -> pd.Series:
    return (upper - lower) / mid


# ── Channels ──────────────────────────────────────────────────────────────────

def donchian_channel(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


# ── Volume / Price ────────────────────────────────────────────────────────────

def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    timestamps: pd.Series | None = None,
) -> pd.Series:
    """
    Volume Weighted Average Price with daily reset.

    If ``timestamps`` (epoch seconds or a DatetimeIndex-like series) are
    provided, VWAP resets at UTC midnight each day.  Otherwise falls back to
    a global cumulative VWAP (backward-compatible).
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    if timestamps is not None:
        try:
            dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
        except Exception:
            dt_index = pd.to_datetime(timestamps, utc=True)
        day_groups = dt_index.dt.date
        cum_tp_vol = tp_vol.groupby(day_groups).cumsum()
        cum_vol = volume.groupby(day_groups).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    m = series.rolling(period).mean()
    s = series.rolling(period).std()
    return (series - m) / s.replace(0, np.nan)


# ── Statistical ───────────────────────────────────────────────────────────────

def _hurst_rs(ts: np.ndarray, min_lag: int = 2, max_lag: int = 100) -> float:
    """Estimate Hurst exponent via R/S analysis."""
    lags = range(min_lag, min(max_lag, len(ts) // 2))
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.sqrt(np.std(diff)))
    if len(tau) < 2:
        return 0.5
    m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(m[0])


def _hurst_dfa(ts: np.ndarray) -> float:
    """Estimate Hurst exponent via Detrended Fluctuation Analysis (DFA)."""
    n = len(ts)
    y = np.cumsum(ts - ts.mean())

    scales = np.unique(np.logspace(0.5, np.log10(n // 4), num=15).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 2:
        return 0.5

    fluct = []
    for s in scales:
        n_segments = n // s
        if n_segments < 1:
            continue
        rms_list = []
        for v in range(n_segments):
            segment = y[v * s:(v + 1) * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_list:
            fluct.append(np.mean(rms_list))
        else:
            fluct.append(np.nan)

    valid = [(s, f) for s, f in zip(scales, fluct) if f > 0 and np.isfinite(f)]
    if len(valid) < 2:
        return 0.5

    log_s = np.log([v[0] for v in valid])
    log_f = np.log([v[1] for v in valid])
    m = np.polyfit(log_s, log_f, 1)
    return float(m[0])


# Rolling buffer for Hurst smoothing (last 3 values)
_hurst_history: list[float] = []


def hurst_exponent(
    series: pd.Series, min_lag: int = 2, max_lag: int = 100, min_points: int = 100,
) -> float:
    """
    Estimate Hurst exponent using R/S + DFA cross-check with smoothing.

    H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
    Returns H=0.5 (neutral) if fewer than ``min_points`` datapoints.
    Final value is the average of the last 3 estimates (smoothing).
    """
    ts = series.dropna().values

    # Minimum sample-size guard
    if len(ts) < min_points:
        return 0.5

    h_rs = _hurst_rs(ts, min_lag, max_lag)
    h_dfa = _hurst_dfa(ts)

    # Average of R/S and DFA as cross-check
    raw_h = (h_rs + h_dfa) / 2.0

    # Smoothing: rolling average of last 3 values
    _hurst_history.append(raw_h)
    if len(_hurst_history) > 3:
        _hurst_history.pop(0)

    return float(np.mean(_hurst_history))


def detect_rsi_divergence(
    close: pd.Series,
    rsi_values: pd.Series,
    lookback: int = 5,
) -> tuple[bool, bool]:
    """
    Detect RSI divergence over the last `lookback` bars.
    Returns (bullish_divergence, bearish_divergence).
    """
    if len(close) < lookback + 1:
        return False, False

    price_recent = close.iloc[-1]
    price_prev = close.iloc[-(lookback + 1)]
    rsi_recent = rsi_values.iloc[-1]
    rsi_prev = rsi_values.iloc[-(lookback + 1)]

    bullish = (price_recent < price_prev) and (rsi_recent > rsi_prev)
    bearish = (price_recent > price_prev) and (rsi_recent < rsi_prev)
    return bullish, bearish
