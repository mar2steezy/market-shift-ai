"""
shift_bot.py - Market Shift AI: a once-a-day PAPER bot that switches sides when the market shifts.

    BULLISH  -> BUYS the index (QQQ or SPY)
    BEARISH  -> SELLS the index and BUYS the inverse ETF (PSQ or SH), which goes UP when the market falls
    UNCLEAR  -> SELLS everything and waits in cash

Run it ONCE each trading morning, a few minutes after the 9:30am Eastern open.
It reads yesterday's finished daily bar, asks the rule picked by shift_research.py which mode the
market is in, and switches at the open if the mode changed. Then it exits.

Safety built in:
  * Research verdict sets size: GO = full, LEAN = half, NO-GO = no trading (unless --experiment).
  * Every trade gets a real STOP order at Alpaca (bull side 10% below entry, bear side 5% against you),
    so it is protected during the day. After a stop-out it waits 5 trading days before re-entering that side.
  * It only ever sells the shares IT bought, never shares your other bots own. It will not start a trade
    in a symbol another bot is already holding.
  * Halts itself (and closes its trade) after 5 losing trades in a row, or if its own losses reach 4%
    of the account. Stays halted until you clear it with --reset-halt.
  * Runs at most once per day. Refuses stale research (over 120 days) or stale price data.
  * New trades are opened with a LIMIT order (default: at most 0.2% worse than the last price), so it
    never chases a sudden spike. If the limit isn't reached within about a minute, the order is
    cancelled and the bot stays in cash until the next run. Exits still use market orders so they
    always go through.

Usage:
    python shift_bot.py --dry-run          # show what it WOULD do today, place nothing
    python shift_bot.py                    # do it (paper money)
    python shift_bot.py --symbol SPY       # trade SPY/SH instead of the research's default pick
    python shift_bot.py --short            # bear side = real short selling instead of an inverse ETF
    python shift_bot.py --alloc 0.25       # fraction of account at full size (default 0.25)
    python shift_bot.py --limit-pct 0.002  # how far above the last price a limit buy may pay (0 = market order)
    python shift_bot.py --experiment       # tiny size if the research verdict is NO-GO
    python shift_bot.py --report
    python shift_bot.py --reset-halt

Needs shift_rules.py + alerts.py + candidate_shift.json, and PAPER keys in env vars
APCA_API_KEY_ID and APCA_API_SECRET_KEY (the same ones your other bots use).
Paper trading only. Educational only. Not financial advice.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from alerts import notify
from shift_rules import regime_series

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state_shift.json")
JOURNAL = os.path.join(HERE, "journal_shift.csv")
EQUITY = os.path.join(HERE, "equity_shift.csv")
CONFIG = os.path.join(HERE, "candidate_shift.json")
TZ = "America/New_York"
MAX_RESEARCH_AGE_DAYS = 120
MAX_DATA_AGE_DAYS = 6
MAX_LOSS_STREAK = 5
MAX_BOT_LOSS = 0.04          # of account equity
EXPERIMENT_MULT = 0.1
NAME = "Market Shift AI"
MODE_NAME = {1: "BULLISH", -1: "BEARISH", 0: "UNCLEAR"}
LEG = {1: "bull", -1: "bear", 0: "cash"}


# ---------------- pure logic (testable without Alpaca) ----------------
def today_et():
    return pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)


def size_mult(cfg, experiment=False):
    m = cfg.get("size_mult")
    if m is None:
        m = 1.0 if cfg.get("verdict") == "GO" else 0.0
    return EXPERIMENT_MULT if (m <= 0 and experiment) else m


def target_qty(equity, alloc, mult, price):
    return int(equity * alloc * mult / price) if price > 0 else 0


def cooldown_ok(bar_dates, stopped_on, cooldown):
    if not stopped_on:
        return True
    return int((bar_dates > pd.Timestamp(stopped_on)).sum()) >= cooldown


def plan(current_leg, target_leg, can_enter, halted):
    """Returns (exit_now, enter_leg). current/target are 'bull', 'bear' or 'cash'."""
    if halted:
        return current_leg != "cash", "cash"
    if current_leg == target_leg:
        return False, "cash" if current_leg == "cash" else None   # None = keep holding
    exit_now = current_leg != "cash"
    enter = target_leg if (target_leg != "cash" and can_enter) else "cash"
    return exit_now, enter


def should_halt(loss_streak, realized_pnl, equity):
    if loss_streak >= MAX_LOSS_STREAK:
        return f"{loss_streak} losing trades in a row"
    if realized_pnl <= -MAX_BOT_LOSS * equity:
        return f"bot losses reached {realized_pnl:,.0f} USD (limit {MAX_BOT_LOSS:.0%} of account)"
    return None


def trade_pnl(side, entry, exit_price, qty):
    """side 'long' or 'short'."""
    return (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty


# ---------------- files ----------------
def load_state():
    base = {"halted": False, "in_trade": False, "realized_pnl": 0.0, "loss_streak": 0}
    if os.path.exists(STATE):
        base.update(json.load(open(STATE)))
    return base


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)


def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def report():
    if os.path.exists(JOURNAL):
        print("Trades:")
        print(pd.read_csv(JOURNAL).tail(20).to_string(index=False))
    if os.path.exists(EQUITY):
        print("\nDaily log:")
        print(pd.read_csv(EQUITY).tail(20).to_string(index=False))
    else:
        print("No runs logged yet.")
    st = load_state()
    print(f"\nBot's own realized P&L: {st.get('realized_pnl', 0):+,.2f} USD | losing streak: {st.get('loss_streak', 0)}")
    if st.get("in_trade"):
        print(f"In a {st.get('leg')} trade: {st.get('side')} {st.get('qty')} {st.get('symbol')} from {st.get('entry_price')}")
    if st.get("halted"):
        print(f"\nBOT IS HALTED: {st.get('reason')}")
    print("Judge it over many months and compare it with just holding the index.")


# ---------------- broker helpers ----------------
def status_of(order):
    return str(getattr(order.status, "value", order.status)).lower()


def wait_fill(trading, order_id, seconds=60):
    for _ in range(seconds // 2):
        o = trading.get_order_by_id(order_id)
        if status_of(o) == "filled" and o.filled_avg_price:
            return float(o.filled_avg_price)
        if status_of(o) in ("canceled", "rejected", "expired"):
            return None
        time.sleep(2)
    return None


def place_stop(trading, sym, side, qty, entry, stop):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest
    if side == "long":
        req = StopOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                               stop_price=round(entry * (1 - stop), 2))
    else:
        req = StopOrderRequest(symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
                               stop_price=round(entry * (1 + stop), 2))
    return str(trading.submit_order(req).id)


def enter_order(trading, sym, side, qty, ref_price, limit_pct):
    """Open a trade. With limit_pct > 0 it's a LIMIT order: a buy pays at most ref_price*(1+limit_pct),
    a short sale gets at least ref_price*(1-limit_pct). Anything not filled within ~60s is cancelled.
    Returns (avg_fill_price or None, shares_filled, limit_price or None)."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    o_side = OrderSide.BUY if side == "long" else OrderSide.SELL
    if limit_pct <= 0:
        limit = None
        o = trading.submit_order(MarketOrderRequest(symbol=sym, qty=qty, side=o_side, time_in_force=TimeInForce.DAY))
    else:
        limit = round(ref_price * (1 + limit_pct), 2) if side == "long" else round(ref_price * (1 - limit_pct), 2)
        o = trading.submit_order(LimitOrderRequest(symbol=sym, qty=qty, side=o_side,
                                                   time_in_force=TimeInForce.DAY, limit_price=limit))
    fill = wait_fill(trading, str(o.id))
    if fill:
        return fill, qty, limit
    # not (fully) filled in time: cancel what's left, then keep whatever part did fill
    try:
        trading.cancel_order_by_id(str(o.id))
    except Exception:
        pass
    time.sleep(2)
    o = trading.get_order_by_id(str(o.id))
    got = int(float(o.filled_qty or 0))
    if got > 0 and o.filled_avg_price:
        return float(o.filled_avg_price), got, limit
    return None, 0, limit


def stop_is_open(trading, order_id):
    if not order_id:
        return False
    try:
        st = status_of(trading.get_order_by_id(order_id))
    except Exception:
        return False
    return st not in ("filled", "canceled", "expired", "rejected", "replaced")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", choices=["QQQ", "SPY"], help="override the research's default symbol")
    ap.add_argument("--alloc", type=float, default=0.25, help="fraction of account at full size")
    ap.add_argument("--limit-pct", type=float, default=0.002,
                    help="limit buys pay at most this much above the last price (0.002 = 0.2%%; 0 = market order)")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--short", action="store_true", help="bear side = real short selling instead of an inverse ETF")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset-halt", action="store_true")
    args = ap.parse_args()

    if args.report:
        return report()
    if args.reset_halt:
        st = load_state()
        st.update(halted=False, reason=None, loss_streak=0)
        save_state(st)
        print("Halt cleared. Losing streak reset. (Realized P&L history is kept.)")
        return

    state = load_state()
    if state.get("halted") and not args.dry_run and not state.get("in_trade"):
        notify(f"{NAME} is HALTED", f"{state.get('reason')}. It will not trade until you clear the halt.", "high")
        sys.exit(f"Bot is HALTED: {state.get('reason')}\nClear with --reset-halt only if you understand why.")

    if not os.path.exists(CONFIG):
        sys.exit("No candidate_shift.json. Run: python shift_research.py")
    full = json.load(open(CONFIG))
    age = (datetime.now() - datetime.fromisoformat(full["generated"])).days
    if age > MAX_RESEARCH_AGE_DAYS and not args.experiment:
        notify(f"{NAME} needs fresh research", f"Research is {age} days old. Rerun shift_research.py --refresh.", "high")
        sys.exit(f"Research is {age} days old. Rerun: python shift_research.py --refresh")

    # an open trade keeps the symbol it was opened with, even if you change --symbol later
    sig_sym = state.get("signal_symbol") if state.get("in_trade") else (args.symbol or full["picked"])
    cfg = full["by_symbol"][sig_sym]
    mult = size_mult(cfg, args.experiment)
    if mult == 0 and not state.get("in_trade"):
        sys.exit(f"Research verdict for {cfg['id']} on {sig_sym} is {cfg['verdict']}: no trading.\n"
                 "To forward-test it with a tiny size on paper anyway: python shift_bot.py --experiment")
    bull_stop, bear_stop = float(full.get("bull_stop", 0.10)), float(full.get("bear_stop", 0.05))
    cooldown = int(full.get("cooldown", 5))

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    key, secret = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    trading = TradingClient(key, secret, paper=True)  # PAPER ONLY
    data = StockHistoricalDataClient(key, secret)
    today = today_et()

    print(f"{NAME} | rule {cfg['id']} on {sig_sym} | verdict {cfg['verdict']} | "
          f"bull = buy {sig_sym}, bear = {'short ' + sig_sym if args.short else 'buy ' + cfg['inverse']} | "
          f"size {mult:.0%} of {args.alloc:.0%}{' | EXPERIMENT' if args.experiment and cfg['verdict'] == 'NO-GO' else ''}")

    if not args.dry_run:
        if str(today.date()) == state.get("last_run_date"):
            sys.exit("Already ran today. Run again on the next trading day.")
        if not trading.get_clock().is_open:
            sys.exit("Market is closed. Run this a few minutes after 9:30am Eastern on a trading day.")

    # ---- which mode is the market in? (finished daily bars only) ----
    req = StockBarsRequest(symbol_or_symbols=sig_sym, timeframe=TimeFrame.Day,
                           start=datetime.now(timezone.utc) - timedelta(days=1000),
                           feed=DataFeed.IEX, adjustment=Adjustment.ALL)
    df = data.get_stock_bars(req).df.xs(sig_sym)[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_convert(TZ).normalize().tz_localize(None)
    df = df[df.index < today]
    if len(df) < 260:
        notify(f"{NAME} skipped", "Not enough price history to evaluate the rule.", "high")
        sys.exit("Not enough daily history to evaluate the rule.")
    if (today - df.index[-1]).days > MAX_DATA_AGE_DAYS:
        notify(f"{NAME} skipped", f"Price data looks stale (latest bar {df.index[-1].date()}). Did nothing.", "high")
        sys.exit(f"Latest daily bar is {df.index[-1].date()}: data looks stale, doing nothing.")
    mode = int(regime_series(df, cfg["rule"], cfg["params"]).iloc[-1])
    target_leg = LEG[mode]

    # ---- account and positions ----
    equity = float(trading.get_account().equity)
    positions = {p.symbol: p for p in trading.get_all_positions()}

    def held(sym):
        p = positions.get(sym)
        return float(p.qty) if p else 0.0           # negative = short

    events = []
    # ---- reconcile: did the broker stop us out (or someone sell our shares) since last run? ----
    if state.get("in_trade"):
        sym, side, own = state["symbol"], state["side"], float(state["qty"])
        have = held(sym)
        exit_price, reason = None, "closed outside the bot"
        if state.get("stop_order_id"):
            try:
                so = trading.get_order_by_id(state["stop_order_id"])
                if status_of(so) == "filled" and so.filled_avg_price:
                    exit_price, reason = float(so.filled_avg_price), "STOP hit"
            except Exception:
                pass
        still = (have >= own) if side == "long" else (have <= -own)
        if exit_price or not still:
            pnl = trade_pnl(side, float(state["entry_price"]), exit_price, own) if exit_price else 0.0
            events.append(("STOPPED" if exit_price else "GONE", state["leg"], sym, own, exit_price, pnl, reason))
            print(f"Since last run: {state['leg']} trade in {sym} {reason}"
                  f"{f' at {exit_price:.2f}, P&L {pnl:+,.2f}' if exit_price else ''}.")
            if not exit_price:
                print("  (Something else sold these shares, maybe another bot. This bot now treats itself as in cash.)")
            if not args.dry_run:
                state["realized_pnl"] = float(state.get("realized_pnl", 0)) + pnl
                state["loss_streak"] = state.get("loss_streak", 0) + 1 if pnl < 0 else 0
                if exit_price:
                    state.update(stopped_on=str(df.index[-1].date()), stopped_leg=state["leg"])
                state.update(in_trade=False, stop_order_id=None)

    in_trade = bool(state.get("in_trade")) and not any(e[0] in ("STOPPED", "GONE") for e in events)
    current_leg = state.get("leg") if in_trade else "cash"
    open_pnl = 0.0
    if in_trade:
        p = positions.get(state["symbol"])
        open_pnl = trade_pnl(state["side"], float(state["entry_price"]), float(p.current_price), float(state["qty"]))

    halt_reason = state.get("reason") if state.get("halted") else should_halt(
        state.get("loss_streak", 0), float(state.get("realized_pnl", 0)) + min(open_pnl, 0), equity)
    can_enter = mult > 0 and (state.get("stopped_leg") != target_leg or
                              cooldown_ok(df.index, state.get("stopped_on"), cooldown))
    exit_now, enter_leg = plan(current_leg, target_leg, can_enter, bool(halt_reason))

    # ---- what would the new trade be? ----
    new_sym, new_side, new_stop, new_qty, price, blocked = None, None, None, 0, 0.0, None
    if enter_leg in ("bull", "bear"):
        if enter_leg == "bull":
            new_sym, new_side, new_stop = sig_sym, "long", bull_stop
        elif args.short:
            new_sym, new_side, new_stop = sig_sym, "short", bear_stop
        else:
            new_sym, new_side, new_stop = cfg["inverse"], "long", bear_stop
        others = held(new_sym)
        if exit_now and state.get("symbol") == new_sym:
            others -= float(state["qty"]) if state["side"] == "long" else -float(state["qty"])
        if others != 0:
            blocked = (f"the account already holds {others:+.0f} {new_sym} that this bot did not buy "
                       f"(probably another bot)")
        else:
            price = float(data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=new_sym, feed=DataFeed.IEX))[new_sym].price)
            new_qty = target_qty(equity, args.alloc, size_mult(cfg, args.experiment), price)
            if new_qty <= 0:
                blocked = "position size rounds to 0 shares"

    # ---- tell the user ----
    holding = f" ({state.get('qty')} {state.get('symbol')})" if in_trade else ""
    print(f"Market mode from {df.index[-1].date()}'s close: {MODE_NAME[mode]} | bot is in: {current_leg.upper()}"
          f"{holding} | account {equity:,.0f}")
    if halt_reason:
        print(f"SELF-HALT: {halt_reason}. Closing any trade and stopping.")
    steps = []
    if exit_now:
        steps.append(f"SELL {state['qty']} {state['symbol']}" if state["side"] == "long" else f"BUY BACK {state['qty']} {state['symbol']}")
    if new_sym and not blocked:
        lim = ""
        if args.limit_pct > 0:
            lp = price * (1 + args.limit_pct) if new_side == "long" else price * (1 - args.limit_pct)
            lim = f" (limit {lp:.2f})"
        steps.append(f"{'BUY' if new_side == 'long' else 'SHORT'} {new_qty} {new_sym}{lim}")
    if not steps:
        steps.append("HOLD" if (in_trade and not exit_now) else "STAY IN CASH")
    if enter_leg == "cash" and target_leg != "cash" and not halt_reason and mult > 0:
        print(f"Was stopped out of a {target_leg} trade on {state.get('stopped_on')}; "
              f"waiting {cooldown} trading days before re-entering.")
    if blocked:
        print(f"Won't open the {enter_leg} trade: {blocked}.")
        if "another bot" in blocked:
            print(f"  Fix: run with --symbol {'SPY' if sig_sym == 'QQQ' else 'QQQ'}, or stop the other bot from trading {new_sym}.")
    print("Plan: " + ", then ".join(steps))

    if args.dry_run:
        print("(Dry run: nothing was ordered, nothing was saved.)")
        return

    # ---- act: 1) close the old side ----
    if exit_now:
        sym, side, own = state["symbol"], state["side"], int(float(state["qty"]))
        if state.get("stop_order_id"):
            try:
                trading.cancel_order_by_id(state["stop_order_id"])
            except Exception:
                pass
            time.sleep(1)
        o = trading.submit_order(MarketOrderRequest(symbol=sym, qty=own, time_in_force=TimeInForce.DAY,
                                                    side=OrderSide.SELL if side == "long" else OrderSide.BUY))
        fill = wait_fill(trading, str(o.id)) or float(positions[sym].current_price)
        pnl = trade_pnl(side, float(state["entry_price"]), fill, own)
        state["realized_pnl"] = float(state.get("realized_pnl", 0)) + pnl
        state["loss_streak"] = state.get("loss_streak", 0) + 1 if pnl < 0 else 0
        reason = "halt" if halt_reason else f"market turned {MODE_NAME[mode]}"
        state.update(in_trade=False, stop_order_id=None)
        events.append(("EXIT", state["leg"], sym, own, fill, pnl, reason))
        save_state(state)
        notify(f"{NAME}: {'SELL' if side == 'long' else 'BUY BACK'} {own} {sym}",
               f"{reason.capitalize()}. Paper P&L on this trade {pnl:+,.0f} USD.", "high")

    # ---- act: 2) open the new side ----
    if new_sym and not blocked:
        state.update(in_trade=True, leg=enter_leg, signal_symbol=sig_sym, symbol=new_sym, side=new_side,
                     qty=new_qty, entry_price=price, stop_order_id=None,
                     opened=datetime.now().isoformat(timespec="seconds"))
        save_state(state)                      # remember the trade even if the next step fails
        fill, got, limit = enter_order(trading, new_sym, new_side, new_qty, price, args.limit_pct)
        lim_note = f", limit {limit:.2f}" if limit else ""
        if fill and got > 0:
            state.update(entry_price=fill, qty=got)
            state["stop_order_id"] = place_stop(trading, new_sym, new_side, got, fill, new_stop)
            part = f" (only {got} of {new_qty} filled before the limit order was cancelled)" if got < new_qty else ""
            events.append(("ENTER", enter_leg, new_sym, got, fill, 0.0,
                           f"{new_side}, stop {new_stop:.0%} away{lim_note}{part}"))
            print(f"Filled {got} {new_sym} at {fill:.2f}{lim_note}{part}. Stop placed.")
            notify(f"{NAME}: {MODE_NAME[mode]} -> {'BUY' if new_side == 'long' else 'SHORT'} {got} {new_sym}",
                   f"Rule {cfg['id']} says the market is {MODE_NAME[mode]}. Paper order filled at {fill:.2f}"
                   f"{lim_note}{part}. Stop placed.", "high")
        else:
            state.update(in_trade=False, stop_order_id=None)
            events.append(("NO FILL", enter_leg, new_sym, new_qty, None, 0.0,
                           f"limit {limit:.2f} not reached; order cancelled" if limit else "order not filled"))
            print(f"Limit order for {new_sym} wasn't filled{lim_note}. Cancelled it. Staying in cash; will try again next run.")
            notify(f"{NAME}: {new_sym} limit not reached",
                   f"Price moved past the limit{lim_note}. Order cancelled; staying in cash until next run.", "high")
    elif in_trade and not exit_now:
        sym, side = state["symbol"], state["side"]
        stop = bull_stop if state["leg"] == "bull" else bear_stop
        if not stop_is_open(trading, state.get("stop_order_id")):
            state["stop_order_id"] = place_stop(trading, sym, side, int(float(state["qty"])),
                                                float(state["entry_price"]), stop)
            print("Protective stop was missing (expired or never placed). Placed a new one.")
        notify(f"{NAME}: holding {sym}", f"Market still {MODE_NAME[mode]}. Open P&L {open_pnl:+,.0f} USD.", "low")
    elif blocked:
        notify(f"{NAME}: trade skipped", f"Market is {MODE_NAME[mode]} but {blocked}.", "high")
    else:
        notify(f"{NAME}: in cash", f"Market mode {MODE_NAME[mode]}. Nothing to do today.", "min")

    for ev in events:
        append_csv(JOURNAL, ["time", "event", "leg", "symbol", "qty", "price", "pnl", "note", "rule"],
                   [datetime.now().isoformat(timespec="seconds"), ev[0], ev[1], ev[2], ev[3],
                    round(ev[4], 4) if ev[4] else "", round(ev[5], 2), ev[6], cfg["id"]])
    append_csv(EQUITY, ["date", "account_equity", "bot_realized_pnl", "open_pnl", "market_mode", "bot_leg",
                        "symbol", "qty", "plan"],
               [today.date(), equity, round(state["realized_pnl"], 2), round(open_pnl, 2), MODE_NAME[mode],
                state.get("leg") if state.get("in_trade") else "cash",
                state.get("symbol") if state.get("in_trade") else "", state.get("qty") if state.get("in_trade") else 0,
                " / ".join(steps)])

    if halt_reason is None:
        halt_reason = should_halt(state.get("loss_streak", 0), float(state.get("realized_pnl", 0)), equity)
        if halt_reason and state.get("in_trade"):
            print(f"Halt limit reached ({halt_reason}). The trade will be closed on the next run.")
    if halt_reason and not state.get("halted"):
        state.update(halted=True, reason=halt_reason, since=datetime.now().isoformat(timespec="seconds"))
        notify(f"{NAME} SELF-HALTED", f"{halt_reason}. It will not trade until you clear the halt.", "urgent")
    state["last_run_date"] = str(today.date())
    save_state(state)
    print("Done. Run again next trading morning.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:
        notify(f"{NAME} ERROR", f"{type(ex).__name__}: {str(ex)[:150]}", "high")
        raise
