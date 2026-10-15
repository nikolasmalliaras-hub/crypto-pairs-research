#!/usr/bin/env python3
"""
HYBRID SCALPING test on MULTI-YEAR HOURLY data  (v8: + results split by BTC VOLATILITY REGIME at entry) (Binance public market data: no account, no API key).

Idea under test (the "core" idea): the big picture decides WHICH pairs may be traded, the hour decides WHEN.
  * Every --rescan-hours the last --gate-days days of hourly data are tested for cointegration (on 4h-sampled data,
    p < --p-max, half-life in a sane range). One coin -> one pair. A pair that stops passing is closed ("decoupled")
    and gets no new positions.
  * Entries on the hourly spread z-score (|z| >= --entry-z); exits on target / stop / time / decoupled.
  * Costs: 4 fills x (fee + slippage) = round trip. PnL in % of ONE leg's notional. Dollar-neutral, no leverage.
Noise checks built in: t-stat of the mean trade (net AND gross), bootstrap interval, first/second half,
yearly breakdown, and a --placebo mode (random correlated pairs, no cointegration test, same rules).

Usage:
  pip install numpy pandas statsmodels
  python hl_regime_v8.py --years 4 --coins 30                 # real test (first run downloads ~7-10 min, run ~10-25 min)
  python hl_regime_v8.py --years 7 --coins 30 --cache cache7h  # out-of-sample check on an earlier period
  python hl_regime_v8.py --synthetic                          # only checks that the code runs
NOT modelled: funding (use --funding-drag-day), intra-bar stops, partial fills, delisted coins (survivorship bias).
"""
import argparse, itertools, json, math, os, time, urllib.request, warnings
import bisect
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    coint = None
EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}
warnings.filterwarnings("ignore")

BIN_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
BINANCE_COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "LTC", "LINK", "DOT", "AVAX", "ATOM", "UNI", "XLM",
                 "ETC", "TRX", "BCH", "FIL", "NEAR", "AAVE", "ALGO", "VET", "THETA", "SAND", "MANA", "AXS", "CRV",
                 "SUSHI", "COMP", "MKR", "SNX", "ENJ", "CHZ", "ZEC", "DASH", "EGLD", "HBAR", "ICP", "RUNE", "KAVA"]


# ---------------------------------------------------------------- data
def get_json(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (400, 404) or k == tries - 1:
                raise
            time.sleep(5 * (k + 1) * (6 if code in (418, 429) else 1))


def fetch_1h(coin, start_ms, cache, years):
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/{coin}_bin_1h_{years:g}y.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    rows, cur, err = [], start_ms, None
    while True:
        ch = None
        for host in BIN_HOSTS:
            try:
                ch = get_json(f"{host}/api/v3/klines?symbol={coin}USDT&interval=1h&limit=1000&startTime={cur}")
                break
            except Exception as e:
                err = e
        if ch is None:
            raise err
        if not ch:
            break
        rows += ch
        if len(ch) < 1000:
            break
        cur = ch[-1][0] + 3_600_000
        time.sleep(0.4)
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms"), name="close")
    s = s[~s.index.duplicated()].sort_index().iloc[:-1]  # drop the still-forming candle
    if s.empty:
        raise ValueError("no data")
    s.to_frame().to_csv(path)
    return s


def load_data(args):
    start = int((time.time() - args.years * 365.25 * 86400) * 1000)
    ser = {}
    for c in BINANCE_COINS[:args.coins]:
        try:
            ser[c] = fetch_1h(c, start, args.cache, args.years)
            print(f"fetched {c}: {len(ser[c])} hourly candles", flush=True)
        except Exception as e:
            print(f"skip {c}: {e}", flush=True)
    px = pd.DataFrame(ser)
    px = px.loc[:, px.notna().mean() > 0.95].ffill().dropna()
    print(f"kept {px.shape[1]} of {len(ser)} coins with full history ({px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d})")
    print("NOTE: only coins that still trade today are in the list (survivorship bias).")
    return px


def synthetic(n, ncoins=24, seed=1):
    rng = np.random.default_rng(seed)
    mkt = np.cumsum(rng.normal(0, 0.004, n))
    cols = {}
    for i in range(0, ncoins, 2):
        base = mkt * rng.uniform(0.7, 1.3) + np.cumsum(rng.normal(0, 0.008, n))
        if (i // 2) % 2 == 0:
            th, sp = rng.uniform(0.03, 0.1), np.zeros(n)
            brk = rng.integers(n // 3, n) if (i // 2) % 3 == 0 else n + 1
            for t in range(1, n):
                sp[t] = sp[t - 1] + rng.normal(0, 0.01) if t >= brk else sp[t - 1] * (1 - th) + rng.normal(0, 0.01)
            other = 0.9 * base + sp
        else:
            other = mkt * rng.uniform(0.7, 1.3) + np.cumsum(rng.normal(0, 0.008, n))
        cols[f"C{i}"], cols[f"C{i+1}"] = np.exp(base + 3), np.exp(other + 3)
    return pd.DataFrame(cols, index=pd.date_range("2022-01-01", periods=n, freq="h"))


# ---------------------------------------------------------------- pair selection (big picture)
def half_life(sp):
    s = sp - sp.mean()
    b = np.polyfit(s[:-1], np.diff(s), 1)[0]
    return -np.log(2) / b if b < 0 else np.inf


def select_pairs(logp_h, args):
    l4 = logp_h.iloc[3::4]
    corr = l4.diff().dropna().corr()
    cand = []
    for a, b in itertools.combinations(logp_h.columns, 2):
        if corr.loc[a, b] < args.corr_min:
            continue
        beta, c0 = np.polyfit(l4[b], l4[a], 1)
        if coint is not None:
            p = coint(l4[a], l4[b], trend="c")[1]
            if p > args.p_max:
                continue
        else:
            u = (l4[a] - beta * l4[b] - c0).values
            du, ul = np.diff(u), u[:-1]
            g = (ul @ du) / (ul @ ul)
            p = g / np.sqrt((du - g * ul).var(ddof=1) / (ul @ ul))
            if p > EG_CRIT.get(args.p_max, -3.90):
                continue
        sp = logp_h[a] - beta * logp_h[b]
        hl = half_life(sp.values)  # hours
        if not (args.hl_min <= hl <= args.hl_max):
            continue
        cand.append((p, a, b, beta, sp.mean(), sp.std(), hl))
    cand.sort()
    used, chosen = set(), []
    for c in cand:  # one coin -> one pair
        if c[1] in used or c[2] in used:
            continue
        used |= {c[1], c[2]}
        chosen.append(c)
    return chosen


def random_pairs(logp_h, args):
    """Placebo: random disjoint pairs among correlated coins, NO cointegration test."""
    corr = logp_h.iloc[3::4].diff().dropna().corr()
    allp = [(a, b) for a, b in itertools.combinations(logp_h.columns, 2) if corr.loc[a, b] >= args.corr_min]
    used, out = set(), []
    for j in args.rng.permutation(len(allp)):
        a, b = allp[j]
        if a in used or b in used:
            continue
        beta, _ = np.polyfit(logp_h[b], logp_h[a], 1)
        sp = logp_h[a] - beta * logp_h[b]
        hl = min(max(half_life(sp.values), args.hl_min), args.hl_max)
        out.append((0.0, a, b, beta, sp.mean(), sp.std(), hl))
        used |= {a, b}
        if len(out) >= args.placebo_pairs:
            break
    return out


def build_schedule(px, args):
    logp, W, step, sched = np.log(px), args.gate_days * 24, args.rescan_hours, {}
    starts = list(range(W, len(px) - 1, step))
    for k, s in enumerate(starts):
        hist = logp.iloc[s - W:s]
        sched[s] = random_pairs(hist, args) if args.placebo else select_pairs(hist, args)
        if k % 20 == 0 or k == len(starts) - 1:
            print(f"rescan {k + 1}/{len(starts)} ({px.index[s]:%Y-%m-%d}): {len(sched[s])} pairs chosen", flush=True)
    return sched


# ---------------------------------------------------------------- simulation
def vol_percentile(px, window_h, min_prior_h):
    """Trailing BTC realized volatility (annualized %) and its percentile among PAST values only (no look-ahead)."""
    lp = np.log(px)
    r = lp["BTC"].diff() if "BTC" in px.columns else lp.diff().mean(axis=1)
    rv = (r.rolling(window_h).std() * math.sqrt(8760) * 100).values
    pct, past = np.full(len(rv), np.nan), []
    for i, v in enumerate(rv):
        if not np.isfinite(v):
            continue
        if len(past) >= min_prior_h:
            pct[i] = bisect.bisect_left(past, v) / len(past)
        bisect.insort(past, v)
    return pct


def simulate_all(px, sched, args, target, stop, pct=None):
    LP, P = np.log(px.to_numpy()), px.to_numpy()
    col = {c: i for i, c in enumerate(px.columns)}
    idx, n = px.index, len(px)
    cost = 4 * (args.fee + args.slip)
    starts = sorted(sched)
    state, cool, trades = {}, {}, []

    def close(key, i, reason):
        st = state.pop(key)
        ia, ib = col[key[0]], col[key[1]]
        gross = st["pos"] * ((P[i, ia] / st["a0"] - 1) - (P[i, ib] / st["b0"] - 1))
        drag = args.funding_drag_day / 100 * (i - st["t0"]) / 24
        trades.append(dict(entry=idx[st["t0"]], exit=idx[i], a=key[0], b=key[1], pos=st["pos"], gross=gross * 100,
                           drag=drag * 100, cost=cost * 100, net=(gross - drag - cost) * 100, hold_h=i - st["t0"],
                           reason=reason, vol_pct=(pct[st["t0"]] if pct is not None else np.nan)))
        cool[key] = i + args.cooldown

    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else n
        pairs = {(c[1], c[2]): c[3:] for c in sched[s]}
        for key in list(state):
            if key not in pairs:
                close(key, s, "decoupled")
        for i in range(s, e):
            for key, (beta, mu, sd, hl) in pairs.items():
                ia, ib = col[key[0]], col[key[1]]
                st = state.get(key)
                if st is None:
                    z = (LP[i, ia] - beta * LP[i, ib] - mu) / sd
                    if i >= cool.get(key, 0) and abs(z) >= args.entry_z and i < n - 1:
                        state[key] = dict(pos=-1 if z > 0 else 1, a0=P[i, ia], b0=P[i, ib], t0=i)
                else:
                    hold = i - st["t0"]
                    gross = st["pos"] * ((P[i, ia] / st["a0"] - 1) - (P[i, ib] / st["b0"] - 1))
                    tot = gross - args.funding_drag_day / 100 * hold / 24
                    reason = None
                    if tot >= target:
                        reason = "target"
                    elif tot <= -stop:
                        reason = "stop"
                    elif hold >= min(args.max_hold_cap_h, max(12, args.max_hold_mult * hl)):
                        reason = "time"
                    elif i == n - 1:
                        reason = "data_end"
                    if reason:
                        close(key, i, reason)
    return pd.DataFrame(trades)


def tstat(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def summarize(tr, mid):
    if tr.empty:
        return dict(trades=0)
    d = tr.sort_values("exit")
    cum = d["net"].cumsum()
    return dict(trades=len(d), win_pct=round((d["net"] > 0).mean() * 100, 1),
                stop_pct=round((d["reason"] == "stop").mean() * 100, 1),
                decoupled_pct=round((d["reason"] == "decoupled").mean() * 100, 1),
                time_pct=round((d["reason"] == "time").mean() * 100, 1),
                avg_hold_h=round(d["hold_h"].mean(), 1), gross_per_trade=round(d["gross"].mean(), 3),
                gross_t=round(tstat(d["gross"]), 2), avg_net_pct=round(d["net"].mean(), 3),
                total_net_pct=round(d["net"].sum(), 1), fees_pct=round(d["cost"].sum(), 1),
                first_half=round(d.loc[d["exit"] < mid, "net"].sum(), 1),
                second_half=round(d.loc[d["exit"] >= mid, "net"].sum(), 1),
                max_dd_pct=round((cum.cummax() - cum).max(), 1), t_stat=round(tstat(d["net"]), 2))


def best_key(rows):
    ok = [r for r in rows if r.get("total_net_pct") is not None]
    b = max(ok, key=lambda r: r["total_net_pct"])
    return (b["target"], b["stop"])


def regime_report(name, tr, mid):
    d = tr[np.isfinite(tr["vol_pct"])].copy()
    print(f"\n=== PAIRS BY BTC VOLATILITY REGIME AT ENTRY: {name} | {len(d)} of {len(tr)} trades have a regime ===")
    if len(d) < 30:
        print("too few trades with a defined regime")
        return
    d["regime"] = np.where(d["vol_pct"] < 1 / 3, "low", np.where(d["vol_pct"] < 2 / 3, "mid", "high"))
    years = sorted(d["exit"].dt.year.unique())
    print(f"{'regime':<7}{'trades':>7}{'gross/trade':>12}{'gross_t':>9}{'net/trade':>10}{'net_t':>7}{'total net':>10}{'1st half':>9}{'2nd half':>9}  "
          + " ".join(f"{y:>7}" for y in years))
    stats = {}
    for reg in ("low", "mid", "high"):
        g = d[d["regime"] == reg]
        if len(g) < 3:
            continue
        yr = g.groupby(g["exit"].dt.year)["net"].sum()
        stats[reg] = dict(gross=g["gross"].mean(), net_t=tstat(g["net"]), h1=g.loc[g["exit"] < mid, "net"].sum(),
                          h2=g.loc[g["exit"] >= mid, "net"].sum(), yrs_pos=int((yr > 0).sum()), n_years=len(yr))
        print(f"{reg:<7}{len(g):>7}{g['gross'].mean():>12.3f}{tstat(g['gross']):>9.2f}{g['net'].mean():>10.3f}{tstat(g['net']):>7.2f}"
              f"{g['net'].sum():>10.1f}{stats[reg]['h1']:>9.1f}{stats[reg]['h2']:>9.1f}  "
              + " ".join(f"{yr.get(y, 0.0):>7.1f}" for y in years))
    print("Gross return per trade by exit year, with the number of trades in brackets:")
    print(f"{'regime':<7}" + " ".join(f"{y:>12}" for y in years))
    for reg in ("low", "mid", "high"):
        g = d[d["regime"] == reg]
        cells = []
        for y in years:
            gy = g[g["exit"].dt.year == y]
            cells.append(f"{gy['gross'].mean():+.2f}({len(gy)})" if len(gy) else "-")
        print(f"{reg:<7}" + " ".join(f"{c:>12}" for c in cells))
    if "low" in stats and "mid" in stats and "high" in stats:
        L = stats["low"]
        mono = L["gross"] > stats["mid"]["gross"] > stats["high"]["gross"]
        print("Criteria for 'pairs work in calm markets' (agreed in advance), LOW-vol regime:")
        print(f"  net t-stat >= 2 .................. {'YES' if L['net_t'] >= 2 else 'NO'} ({L['net_t']:.2f})")
        print(f"  positive in both halves .......... {'YES' if L['h1'] > 0 and L['h2'] > 0 else 'NO'} ({L['h1']:.1f} / {L['h2']:.1f})")
        print(f"  positive in >= 3 of the years .... {'YES' if L['yrs_pos'] >= 3 else 'NO'} ({L['yrs_pos']} of {L['n_years']})")
        print(f"  gross low > mid > high (monotone)  {'YES' if mono else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--coins", type=int, default=30)
    ap.add_argument("--cache", default="cache1h")
    ap.add_argument("--gate-days", type=int, default=60)
    ap.add_argument("--rescan-hours", type=int, default=48)
    ap.add_argument("--p-max", type=float, default=0.01)
    ap.add_argument("--corr-min", type=float, default=0.6)
    ap.add_argument("--hl-min", type=float, default=6, help="hours")
    ap.add_argument("--hl-max", type=float, default=120, help="hours")
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--max-hold-mult", type=float, default=3.0)
    ap.add_argument("--max-hold-cap-h", type=float, default=72.0)
    ap.add_argument("--cooldown", type=int, default=6, help="hours")
    ap.add_argument("--fee", type=float, default=0.00045, help="per fill (Hyperliquid-like taker); check your tier")
    ap.add_argument("--slip", type=float, default=0.0002, help="per fill")
    ap.add_argument("--funding-drag-day", type=float, default=0.0, help="%% of leg notional per day held")
    ap.add_argument("--targets", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--stops", type=float, nargs="+", default=[2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--placebo-pairs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vol-window-days", type=int, default=7, help="trailing window for BTC realized volatility")
    ap.add_argument("--vol-min-prior-days", type=int, default=90, help="warm-up before the volatility percentile is defined")
    ap.add_argument("--synth-days", type=int, default=400)
    a = ap.parse_args()
    a.rng = np.random.default_rng(a.seed)

    px = synthetic(a.synth_days * 24) if a.synthetic else load_data(a)
    print(f"\nPrice matrix: {px.shape[1]} coins x {px.shape[0]} hourly bars | placebo={a.placebo}\n", flush=True)
    sched = build_schedule(px, a)
    if not sched:
        print("Not enough data for the gate window.")
        return
    mid = px.index[(min(sched) + len(px)) // 2]
    cost = 4 * (a.fee + a.slip)
    n_pairs = sum(len(v) for v in sched.values())
    print(f"\nAverage pairs chosen per rescan: {n_pairs / len(sched):.1f} | cost per round trip: {cost*100:.3f}% "
          f"of one-leg notional | halves split at {mid:%Y-%m-%d}\n", flush=True)

    pct = vol_percentile(px, a.vol_window_days * 24, a.vol_min_prior_days * 24)
    print(f"BTC volatility regime: trailing {a.vol_window_days}-day realized vol, percentile vs its own PAST; "
          f"defined for {np.isfinite(pct).mean() * 100:.0f}% of the bars (first {a.vol_min_prior_days} days are warm-up)\n", flush=True)
    rows, best, best_tr, trlog = [], None, None, {}
    for tg, st in itertools.product(a.targets, a.stops):
        tr = simulate_all(px, sched, a, tg / 100, st / 100, pct)
        r = dict(target=tg, stop=st, **summarize(tr, mid))
        rows.append(r)
        trlog[(tg, st)] = tr
        if r.get("total_net_pct") is not None and (best is None or r["total_net_pct"] > best):
            best, best_tr = r["total_net_pct"], tr
    res = pd.DataFrame(rows)
    if "total_net_pct" not in res.columns:
        print("No trades at all. Loosen a filter (--p-max 0.05, --entry-z 1.5, --corr-min 0.4).")
        return
    res = res.sort_values("total_net_pct", ascending=False)
    print(res.to_string(index=False))

    x = best_tr["net"].values
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(x, len(x)).sum() for _ in range(5000)])
    print(f"\nBest combo: {len(x)} trades, total {x.sum():.1f}; without its 5 best trades {x.sum() - np.sort(x)[-5:].sum():.1f}")
    print(f"Bootstrap 95% interval of the total: [{np.percentile(boots, 2.5):.1f}, {np.percentile(boots, 97.5):.1f}] | "
          f"chance total <= 0: {(boots <= 0).mean() * 100:.1f}%")
    print(f"(best of {len(rows)} combos, so optimistic; t_stat of about 2 or more is the usual bar; gross_t is the same "
          f"test BEFORE costs)")
    by_year = best_tr.groupby(best_tr["exit"].dt.year)["net"].agg(["size", "sum"]).round(1)
    by_year.columns = ["trades", "net_pct"]
    print("\nBest combo by year:\n" + by_year.to_string())
    refs = [(best_key(rows), "best combo by total net")] + [(k, "reference combo") for k in [(2.0, 8.0), (1.5, 5.0)] if k in trlog]
    seen = set()
    for key, label in refs:
        if key in seen:
            continue
        seen.add(key)
        regime_report(f"target {key[0]} / stop {key[1]} ({label})", trlog[key], mid)
    res.to_csv("hourly_results.csv", index=False)
    best_tr.to_csv("hourly_trades_best.csv", index=False)
    print("\nSaved hourly_results.csv and hourly_trades_best.csv")


if __name__ == "__main__":
    main()
