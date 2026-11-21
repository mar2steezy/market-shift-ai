"""
shift_rules.py - how Market Shift AI decides if the market is BULLISH, BEARISH or UNCLEAR.

Each rule looks at finished DAILY bars of SPY or QQQ and gives one answer per day:
    +1 = BULLISH  -> own the index (buy QQQ or SPY)
    -1 = BEARISH  -> bet against it (buy the inverse ETF PSQ or SH)
     0 = UNCLEAR  -> sit in cash
The bot acts at the NEXT morning's open, exactly as shift_research.py tested it.
Rules only use data up to and including the current day (no peeking ahead).
"""
import numpy as np
import pandas as pd

# Keep this list SHORT: there is only one real bear market (2022) in the data.
CANDIDATES = [
    ("trend_flip", {"confirm": 3}),    # above 200-day avg 3 days in a row -> bull; below 3 days -> bear
    ("cross3", {}),                    # bull: price AND 50-day avg above 200-day. bear: both below. else cash
    ("trend_rally", {"th": 90}),       # bull: above 200-day. bear: only short bounces inside a downtrend
    ("mom_rally", {"n": 126, "th": 90}),  # bull: up over the last 6 months. bear: same bounce rule
]


def rule_id(rule, params):
    return f"{rule}(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"


def rsi(close, n=2):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _bounce_short(c, ma200, th, max_hold=10):
    """1 while betting against a bounce inside a downtrend, else 0."""
    r2 = rsi(c, 2)
    out = np.zeros(len(c), dtype=int)
    state, held = False, 0
    for i in range(len(c)):
        if np.isnan(ma200.iloc[i]):
            continue
        bear = c.iloc[i] < ma200.iloc[i]
        if not state and bear and r2.iloc[i] > th:
            state, held = True, 0
        elif state:
            held += 1
            if r2.iloc[i] < 30 or held >= max_hold:
                state = False
        out[i] = int(state)
    return out


def regime_series(df, rule, params):
    """Return an int Series of +1 / 0 / -1 (see top of file)."""
    c = df["close"]
    ma200 = c.rolling(200).mean()
    ma50 = c.rolling(50).mean()
    ready = ma200.notna().values
    above = (c > ma200).values
    out = np.zeros(len(c), dtype=int)

    if rule == "trend_flip":
        n = params.get("confirm", 3)
        s = 0
        for i in range(len(c)):
            if not ready[i]:
                continue
            if i >= n - 1 and above[i - n + 1:i + 1].all():
                s = 1
            elif i >= n - 1 and (~above[i - n + 1:i + 1]).all():
                s = -1
            out[i] = s
    elif rule == "cross3":
        f = (ma50 > ma200).values
        out = np.where(ready & above & f, 1, np.where(ready & ~above & ~f, -1, 0))
    elif rule == "trend_rally":
        b = _bounce_short(c, ma200, params.get("th", 90))
        out = np.where(~ready, 0, np.where(above, 1, -b))
    elif rule == "mom_rally":
        up = (c / c.shift(params.get("n", 126)) - 1 > 0).values
        b = _bounce_short(c, ma200, params.get("th", 90))
        out = np.where(~ready, 0, np.where(up, 1, -b))
    else:
        raise ValueError(f"Unknown rule {rule}")
    return pd.Series(out.astype(int), index=df.index)


def simulate(df, regime, cost_bps=10.0, bull_stop=0.10, bear_stop=0.05, cooldown=5, inverse_fee_yr=0.0089):
    """Simulate exactly what shift_bot.py does.

      * decide at day i's close, switch at day i+1's OPEN (sell the old side, buy the new side)
      * BULL trades have a protective stop 10% below entry, BEAR trades 5% against entry
      * after a stop-out, wait `cooldown` trading days before re-entering that same side
      * bear side = inverse ETF, modeled as -1x the index each day minus its yearly fee
    Returns (daily_returns Series, list of (entry_date, side, trade_return)).
    """
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    w = regime.values
    cost = cost_bps / 1e4
    fee = inverse_fee_yr / 252
    rets = np.zeros(len(c))
    trades = []
    p, stop_lvl, t_eq, t_date = 0, 0.0, 1.0, None
    cool_until, cool_side = -1, 0

    def leg(side, a, b):              # return of holding `side` from price a to price b
        return (b / a - 1) if side == 1 else -(b / a - 1)

    for i in range(1, len(c)):
        tgt = int(w[i - 1])
        if tgt != 0 and tgt == cool_side and i <= cool_until:
            tgt = 0
        r = 0.0
        base = c[i - 1]
        if p != 0 and tgt != p:                        # close the old side at the open
            x = leg(p, c[i - 1], o[i]) - cost
            r += x
            t_eq *= 1 + x
            trades.append((t_date, p, t_eq - 1))
            p = 0
        if p == 0 and tgt != 0:                        # open the new side at the open
            p, base, t_date, t_eq = tgt, o[i], df.index[i], 1.0
            stop_lvl = o[i] * (1 - bull_stop) if p == 1 else o[i] * (1 + bear_stop)
            r -= cost
            t_eq *= 1 - cost
        if p != 0:
            hit_gap = base == c[i - 1] and ((p == 1 and o[i] <= stop_lvl) or (p == -1 and o[i] >= stop_lvl))
            hit_day = (p == 1 and l[i] <= stop_lvl) or (p == -1 and h[i] >= stop_lvl)
            if hit_gap or hit_day:
                exit_px = o[i] if hit_gap else stop_lvl
                x = leg(p, base, exit_px) - cost - (fee if p == -1 else 0)
                r += x
                t_eq *= 1 + x
                trades.append((t_date, p, t_eq - 1))
                cool_until, cool_side, p = i + cooldown, p, 0
            else:
                x = leg(p, base, c[i]) - (fee if p == -1 else 0)
                r += x
                t_eq *= 1 + x
        rets[i] = r
    if p != 0:
        trades.append((t_date, p, t_eq - 1))
    return pd.Series(rets, index=df.index), trades


def stats(rets, trades, start=None, end=None):
    r = rets.loc[start:end]
    eq = (1 + r).cumprod()
    s0 = pd.Timestamp(start) if start else None
    s1 = pd.Timestamp(end) if end else None
    tr = [x for d, side, x in trades if (s0 is None or d >= s0) and (s1 is None or d <= s1)]
    return {
        "return %": round(float(eq.iloc[-1] - 1) * 100, 1) if len(eq) else 0.0,
        "max drawdown %": round(float((eq / eq.cummax() - 1).min()) * 100, 1) if len(eq) else 0.0,
        "trades": len(tr),
        "win rate %": round(100 * float(np.mean([x > 0 for x in tr]))) if tr else 0,
    }


def buy_hold(df, start=None, end=None):
    c = df["close"].loc[start:end]
    eq = c / c.iloc[0]
    return {"return %": round(float(eq.iloc[-1] - 1) * 100, 1),
            "max drawdown %": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}
