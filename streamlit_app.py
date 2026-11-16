"""VIKA Forecast Lab - what stands out, where each series is heading and what moves with what, on the Macro Terminal's DATA export.

The first three tabs answer questions (BRIEFING: what is unusual across everything; THIS SERIES: where it stands, where it is
heading and whether the forecast deserves trust; RELATIONSHIPS: what leads / follows it and what happened next in the past).
The remaining tabs are the workbench behind them: ARIMA family, backtest, GARCH, robust OLS and the unit-root diagnostics.

Run:  streamlit run forecast_lab.py
Upload the ZIP / CSV / Excel you downloaded from the terminal's DATA tab (long or wide layout)."""
from __future__ import annotations

import hashlib
import io
import os
import re
import warnings
import zipfile
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

import statsmodels.api as sm  # noqa: E402
from arch import arch_model  # noqa: E402
from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron  # noqa: E402
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid, het_arch,  # noqa: E402
                                          het_breuschpagan, het_white, linear_reset)
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402
from statsmodels.stats.stattools import durbin_watson, jarque_bera  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, coint, grangercausalitytests, pacf, zivot_andrews  # noqa: E402

BUILD = "forecast lab 16-Nov-2026 b"

# ----------------------------------------------------------------------------- theme
BG, PANEL, GRID, TEXT, MUTED = "#07050f", "#0e0a1c", "#241c3f", "#E8E6F2", "#8d86a8"
GOLD, PURPLE, CYAN, PINK, GREEN, RED, ORANGE = "#F2B825", "#8B5CF6", "#38BDF8", "#F472B6", "#2DD4A0", "#F4577A", "#FB923C"
SERIES_COLORS = [GOLD, PURPLE, CYAN, PINK, GREEN, ORANGE, "#A3E635", "#E879F9"]
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SEASON = {"D": 1, "W": 1, "M": 12, "Q": 4, "A": 1}
RULE = {"W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}
ORDER = ["D", "W", "M", "Q", "A"]
PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}


def layout(fig, height=320, title=None, legend=True):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family=MONO, size=11, color=MUTED), margin=dict(l=48, r=18, t=34 if title else 12, b=34),
                      title=dict(text=title, font=dict(size=13, color=GOLD), x=0.01) if title else None,
                      hovermode="x unified", legend=dict(orientation="h", x=1, xanchor="right", y=1.08, font=dict(color=TEXT, size=10)))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="rgba(0,0,0,0)", zeroline=False)
    return fig


# ----------------------------------------------------------------------------- loading the terminal's export
def _category_from_name(name: str) -> str:
    stem = re.sub(r"\.(csv|xlsx|zip)$", "", os.path.basename(name), flags=re.I)
    stem = re.sub(r"^vika_", "", stem)
    stem = re.sub(r"_(long|wide)(_\d{8})?$", "", stem)
    stem = re.sub(r"_\d{8}$", "", stem)
    return stem.replace("_", " ").strip().title() or "Data"


def _to_long(df: pd.DataFrame, category: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    if {"series", "date", "value"} <= set(low):                    # long layout from the DATA tab
        out = pd.DataFrame({"series": df[low["series"]].astype(str), "date": pd.to_datetime(df[low["date"]], errors="coerce"),
                            "value": pd.to_numeric(df[low["value"]], errors="coerce")})
        for k, default in (("unit", ""), ("frequency", ""), ("source", "")):
            out[k] = df[low[k]].astype(str) if k in low else default
        out["raw_value"] = pd.to_numeric(df[low["raw_value"]], errors="coerce") if "raw_value" in low else np.nan
    else:                                                          # wide layout: first column = date
        dcol = low.get("date", df.columns[0])
        v = df.melt(id_vars=dcol, var_name="series", value_name="value")
        out = pd.DataFrame({"series": v["series"].astype(str), "date": pd.to_datetime(v[dcol], errors="coerce"),
                            "value": pd.to_numeric(v["value"], errors="coerce"), "unit": "", "frequency": "", "source": "", "raw_value": np.nan})
    out = out.dropna(subset=["date", "value"])
    out["category"] = category
    return out


@st.cache_data(show_spinner="Reading files...", max_entries=8)
def read_uploads(files: tuple) -> pd.DataFrame:
    """files: tuple of (name, bytes). Returns one long table over every category found."""
    frames = []

    def add_csv(name, raw):
        d = pd.read_csv(io.BytesIO(raw))
        if name.lower().endswith("catalog.csv") or "observations" in {c.lower() for c in d.columns}:
            return
        frames.append(_to_long(d, _category_from_name(name)))

    for name, raw in files:
        n = name.lower()
        if n.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for m in z.namelist():
                    if m.lower().endswith(".csv"):
                        add_csv(m, z.read(m))
        elif n.endswith(".csv"):
            add_csv(name, raw)
        elif n.endswith((".xlsx", ".xls")):
            for sheet, d in pd.read_excel(io.BytesIO(raw), sheet_name=None).items():
                if sheet.lower() != "catalog":
                    frames.append(_to_long(d, sheet))
    if not frames:
        raise ValueError("no usable data found in the uploaded files")
    return pd.concat(frames, ignore_index=True)


def demo_data() -> pd.DataFrame:
    """Synthetic series shaped like the terminal's export, for trying the app without data."""
    r = np.random.default_rng(7)
    rows = []

    def add(cat, name, unit, freq, idx, vals):
        rows.append(pd.DataFrame({"category": cat, "series": name, "unit": unit, "frequency": freq, "source": "synthetic",
                                  "date": idx, "value": vals, "raw_value": np.nan}))
    d = pd.bdate_range("2016-01-01", "2026-10-01")
    eps = r.standard_t(6, len(d)) / np.sqrt(1.5); h = np.empty(len(d)); h[0] = 1.0; ret = np.empty(len(d))
    for t in range(len(d)):
        if t:
            h[t] = 0.05 + 0.10 * ret[t - 1] ** 2 + 0.86 * h[t - 1]
        ret[t] = 0.04 + np.sqrt(h[t]) * eps[t]
    add("Markets", "Nifty 50 (synthetic)", "index", "D", d, 8000 * np.exp(np.cumsum(ret) / 100))
    m = pd.date_range("2013-01-31", "2026-08-31", freq="ME"); n = len(m)
    infl = np.empty(n); infl[0] = 5
    for t in range(1, n):
        infl[t] = 1.2 + 0.75 * infl[t - 1] + r.normal(0, 0.6)
    add("India macro", "CPI inflation - All items (synthetic)", "% YoY", "M", m, infl)
    seas = 6 * np.sin(2 * np.pi * (np.asarray(m.month) - 3) / 12)
    iip = 100 * np.exp(0.005 * np.arange(n) + 0.02 * np.cumsum(r.normal(0, 1, n)) / 4 + seas / 100)
    covid = (m >= "2020-03-31") & (m <= "2020-07-31"); iip[covid] *= 0.7
    add("India macro", "IIP index - General (synthetic)", "index", "M", m, iip)
    add("India macro", "WPI index - All commodities (synthetic)", "index", "M", m, 100 * np.exp(np.cumsum(0.003 + r.normal(0, 0.01, n))))
    w = pd.date_range("2015-06-05", "2026-10-02", freq="W-FRI"); pe = np.empty(len(w)); pe[0] = 22
    for t in range(1, len(w)):
        pe[t] = pe[t - 1] + 0.04 * (23 - pe[t - 1]) + r.normal(0, 0.5)
    add("Valuations", "Nifty 50 - P/E (synthetic)", "x", "W", w, pe)
    add("Global macro", "US unemployment (synthetic)", "%", "M", m, 5 + np.cumsum(r.normal(0, 0.15, n)) * 0.3)
    gy = 4.6 + 0.45 * pd.Series(infl).shift(3).bfill().values + r.normal(0, 0.12, n)       # follows inflation with a 3-month delay
    add("India macro", "10Y G-sec yield (synthetic)", "%", "M", m, gy)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- series preparation
def detect_freq(idx: pd.DatetimeIndex) -> str:
    if len(idx) < 4:
        return "D"
    d = pd.Series(idx).diff().dt.days.dropna().median()
    return "D" if d <= 4 else "W" if d <= 10 else "M" if d <= 40 else "Q" if d <= 100 else "A"


def regularize(s: pd.Series, target: str, how: str = "last") -> pd.Series:
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if target == "D":
        return s
    r = s.resample(RULE[target])
    out = r.last() if how == "last" else r.mean() if how == "mean" else r.sum(min_count=1)
    out = out.loc[out.first_valid_index():out.last_valid_index()]
    return out.interpolate(limit=3, limit_area="inside")


TRANSFORMS = {
    "level": "Level",
    "log": "Natural log of level",
    "diff": "First difference",
    "logdiff": "Log difference x100 (growth / return)",
    "yoy": "Year-on-year % change",
}


def transform(y: pd.Series, kind: str, m: int) -> pd.Series:
    if kind in ("log", "logdiff", "yoy") and (y <= 0).any():
        raise ValueError("this transformation needs strictly positive values")
    if kind == "level":
        return y
    if kind == "log":
        return np.log(y)
    if kind == "diff":
        return y.diff().dropna()
    if kind == "logdiff":
        return (100 * np.log(y).diff()).dropna()
    if kind == "yoy":
        return (100 * (y / y.shift(max(m, 1)) - 1)).dropna()
    raise ValueError(kind)


def suggest_transform(y: pd.Series, unit: str, name: str) -> tuple[str, str]:
    """Pick a sensible default transformation and say why."""
    u = (unit or "").lower()
    if "%" in u or "yoy" in name.lower() or (y <= 0).any() or u in ("x", "pp"):
        return "level", "a rate / ratio / series with non-positive values: model the level (differencing is chosen by the unit-root tests)"
    return "log", "a strictly positive index or price: take logs so growth is proportional and variance stabilises"


def future_dates(last: pd.Timestamp, h: int, freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return pd.bdate_range(last + pd.Timedelta(days=1), periods=h)
    return pd.date_range(last, periods=h + 1, freq=RULE[freq])[1:]


def dummy_series(index: pd.DatetimeIndex, start, end) -> np.ndarray:
    return ((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))).astype(float)


# ----------------------------------------------------------------------------- stationarity & seasonality
def unitroot_table(y: pd.Series) -> pd.DataFrame:
    rows = []
    specs = [("ADF (constant)", lambda: ADF(y, trend="c", method="aic"), "unit root"),
             ("ADF (const+trend)", lambda: ADF(y, trend="ct", method="aic"), "unit root"),
             ("Phillips-Perron (constant)", lambda: PhillipsPerron(y, trend="c"), "unit root"),
             ("DF-GLS (constant)", lambda: DFGLS(y, trend="c"), "unit root"),
             ("KPSS (constant)", lambda: KPSS(y, trend="c"), "stationary"),
             ("KPSS (const+trend)", lambda: KPSS(y, trend="ct"), "stationary")]
    for name, mk, null in specs:
        try:
            t = mk()
            rej = t.pvalue < 0.05
            rows.append({"Test": name, "H0": null, "Statistic": float(t.stat), "p-value": float(t.pvalue), "Lags": int(t.lags),
                         "At 5%": ("reject H0" if rej else "cannot reject H0"),
                         "Reads as": ("stationary" if rej else "non-stationary") if null == "unit root" else ("non-stationary" if rej else "stationary")})
        except Exception as e:  # noqa: BLE001
            rows.append({"Test": name, "H0": null, "Statistic": np.nan, "p-value": np.nan, "Lags": 0, "At 5%": "n/a", "Reads as": f"failed: {str(e)[:40]}"})
    return pd.DataFrame(rows)


def is_stationary(y: pd.Series, trend: str = "c") -> tuple[bool, dict]:
    """Vote of ADF, PP, DF-GLS (H0 unit root) and KPSS (H0 stationary) at 5%. >=3 of 4 -> stationary; a 2-2 split follows ADF."""
    v = {}
    try:
        v["ADF"] = ADF(y, trend=trend, method="aic").pvalue < 0.05
    except Exception:  # noqa: BLE001
        v["ADF"] = False
    try:
        v["PP"] = PhillipsPerron(y, trend=trend).pvalue < 0.05
    except Exception:  # noqa: BLE001
        v["PP"] = v["ADF"]
    try:
        v["DFGLS"] = DFGLS(y, trend=trend).pvalue < 0.05
    except Exception:  # noqa: BLE001
        v["DFGLS"] = v["ADF"]
    try:
        v["KPSS"] = KPSS(y, trend=trend).pvalue >= 0.05
    except Exception:  # noqa: BLE001
        v["KPSS"] = v["ADF"]
    votes = sum(v.values())
    return (votes >= 3 or (votes == 2 and v["ADF"])), {**v, "votes": votes}


def has_trend(y: pd.Series) -> bool:
    """Is there a significant deterministic linear trend (HAC t-test)? Used to choose constant vs constant+trend."""
    if len(y) < 30:
        return False
    X = sm.add_constant(np.arange(len(y), dtype=float))
    r = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})
    return bool(r.pvalues[1] < 0.01 and r.rsquared > 0.4)


def integration_order(y: pd.Series, max_d: int = 2, trend: str = "auto") -> tuple[int, list[dict]]:
    z, hist = y.dropna(), []
    for d in range(max_d + 1):
        tr = trend if trend != "auto" else ("ct" if d == 0 and has_trend(z) else "c")
        ok, v = is_stationary(z, tr)
        hist.append({"Differences": d, "Deterministics": tr, **{k: (bool(x) if k != "votes" else int(x)) for k, x in v.items()}, "Stationary": ok})
        if ok:
            return d, hist
        z = z.diff().dropna()
    return max_d, hist


def seasonal_strength(y: pd.Series, m: int) -> float:
    """Hyndman's STL strength of seasonality (0..1). > 0.64 is treated as 'strongly seasonal'."""
    if m <= 1 or len(y) < 2 * m + 4:
        return 0.0
    try:
        f = STL(y.reset_index(drop=True), period=m, robust=True).fit()
        return float(max(0.0, 1 - np.var(f.resid) / np.var(f.seasonal + f.resid)))
    except Exception:  # noqa: BLE001
        return 0.0


def nw_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb for the HAC bandwidth."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def ljung_box(r: np.ndarray, lags: int, model_df: int = 0) -> tuple[float, float]:
    lags = int(max(lags, model_df + 1))
    t = acorr_ljungbox(r, lags=[lags], model_df=model_df, return_df=True)
    return float(t["lb_stat"].iloc[0]), float(t["lb_pvalue"].iloc[0])


def arch_lm(r: np.ndarray, lags: int) -> tuple[float, float]:
    lm, p, _f, _fp = het_arch(r, nlags=lags)
    return float(lm), float(p)


def describe(y: pd.Series) -> dict:
    jb, jbp, sk, ku = jarque_bera(y.values)
    return {"Observations": len(y), "Mean": y.mean(), "Std dev": y.std(), "Min": y.min(), "Max": y.max(),
            "Skewness": sk, "Kurtosis": ku, "Jarque-Bera p": jbp}


def outliers(y: pd.Series, k: float = 5.0) -> pd.Series:
    med = y.median()
    mad = (y - med).abs().median() * 1.4826 or y.std()
    z = (y - med) / mad
    return y[z.abs() > k]


def diagnostics_bundle(y: pd.Series, freq: str, trend: str = "auto") -> dict:
    """Everything the DIAGNOSTICS tab and the model defaults rely on."""
    m = SEASON[freq]
    d, hist = integration_order(y, 2, trend)
    yd = y.copy()
    for _ in range(d):
        yd = yd.diff().dropna()
    fs = seasonal_strength(y if d == 0 else yd, m)
    D = 1 if (m > 1 and fs > 0.64) else 0
    out = {"d": d, "D": D, "m": m, "seas_strength": fs, "history": hist, "unitroot": unitroot_table(y), "yd": yd}
    L = int(min(max(2 * m, 10), len(yd) // 5)) if len(yd) > 25 else 5
    z = yd.values - yd.values.mean()
    try:
        out["lb"] = ljung_box(z, L)
        out["arch"] = arch_lm(z, min(L, 12))
    except Exception:  # noqa: BLE001
        out["lb"] = out["arch"] = (np.nan, np.nan)
    out["za"] = None
    if len(y) >= 40:
        try:
            za = zivot_andrews(y.values, regression="c", autolag="AIC")
            out["za"] = {"stat": float(za[0]), "p": float(za[1]), "bp": int(za[4])}
        except Exception:  # noqa: BLE001
            pass
    return out


# ----------------------------------------------------------------------------- ARIMA family
def trend_for(d: int, D: int, trend_stationary: bool, drift: bool) -> str:
    """Deterministic terms that are identified for a given integration order (statsmodels ARIMA treats trend as a regressor
    on the undifferenced series): d+D=0 -> constant (or constant+trend); d+D=1 -> drift only; d+D>=2 -> none."""
    k = d + D
    if k == 0:
        return "ct" if trend_stationary else "c"
    if k == 1:
        return "t" if drift else "n"
    return "n"


def drift_significant(y: pd.Series, d: int, D: int, m: int) -> bool:
    """HAC t-test that the mean of the differenced series differs from zero (decides drift when d+D=1)."""
    z = y.copy()
    for _ in range(d):
        z = z.diff()
    if D and m > 1:
        z = z.diff(m)
    z = z.dropna()
    if len(z) < 20:
        return False
    r = sm.OLS(z.values, np.ones(len(z))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(z))})
    return bool(r.pvalues[0] < 0.05)


def fit_arima(y: pd.Series, order, seasonal=(0, 0, 0, 0), trend="n", exog=None, cov_type=None):
    ys = pd.Series(np.asarray(y, dtype=float))
    ex = None if exog is None else pd.DataFrame(np.asarray(exog, dtype=float).reshape(len(ys), -1), columns=[f"dummy{i}" if i else "dummy" for i in range(np.asarray(exog).reshape(len(ys), -1).shape[1])])
    mod = ARIMA(ys, exog=ex, order=order, seasonal_order=seasonal, trend=trend)
    return mod.fit(cov_type=cov_type) if cov_type else mod.fit()


@st.cache_data(show_spinner=False, max_entries=24)
def arima_search(yv: np.ndarray, d: int, D: int, m: int, trend: str, pmax: int, qmax: int, Pmax: int, Qmax: int,
                 ic: str, exog: np.ndarray | None) -> pd.DataFrame:
    """Grid search of (p,q)(P,Q) at fixed d, D and trend; ranked by AICc/BIC/AIC. Only models that converge and are
    stationary/invertible (enforced) are kept."""
    rows = []
    seasonal = m > 1 and (Pmax > 0 or Qmax > 0 or D > 0)
    for p, q, P, Q in product(range(pmax + 1), range(qmax + 1), range(Pmax + 1) if seasonal else [0], range(Qmax + 1) if seasonal else [0]):
        so = (P, D, Q, m) if seasonal else (0, 0, 0, 0)
        try:
            r = fit_arima(yv, (p, d, q), so, trend, exog)
            if not np.isfinite(r.llf):
                continue
            rows.append({"p": p, "d": d, "q": q, "P": P, "D": D if seasonal else 0, "Q": Q, "m": m if seasonal else 0, "trend": trend,
                         "k": len(r.params), "AIC": r.aic, "AICc": r.aicc, "BIC": r.bic})
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        raise ValueError("no ARIMA specification converged")
    return pd.DataFrame(rows).sort_values(ic).reset_index(drop=True)


def residual_checks(res, y_n: int, d: int, D: int, m: int, n_arma: int) -> pd.DataFrame:
    burn = d + D * max(m, 1) if D else d
    r = np.asarray(res.resid, dtype=float)[burn:]
    L = int(min(max(2 * m if m > 1 else 10, 10), max(len(r) // 5, n_arma + 2)))
    rows = []

    def add(name, stat, p, h0, bad_means):
        rows.append({"Check": name, "Statistic": stat, "p-value": p, "H0": h0,
                     "Result": ("PASS" if p >= 0.05 else "FAIL") if np.isfinite(p) else "n/a", "If FAIL": bad_means})
    s, p = ljung_box(r, L, n_arma)
    add(f"Ljung-Box (lag {L}) on residuals", s, p, "no autocorrelation", "mean equation misspecified: add AR/MA terms, a dummy, or seasonal terms")
    s, p = ljung_box(r ** 2, L)
    add(f"Ljung-Box (lag {L}) on squared residuals", s, p, "no conditional heteroskedasticity", "volatility clusters: use robust SEs and consider ARIMA-GARCH")
    s, p = arch_lm(r, min(L, 12))
    add("ARCH-LM", s, p, "no ARCH effects", "forecast intervals are too narrow in calm and too wide in stormy periods; see GARCH tab")
    jb, jbp, sk, ku = jarque_bera(r)
    add(f"Jarque-Bera (skew {sk:.2f}, kurt {ku:.2f})", jb, jbp, "residuals are normal", "fat tails/skew: Gaussian intervals unreliable, use robust SEs")
    t = stats.ttest_1samp(r, 0.0)
    add("Mean of residuals = 0", float(t.statistic), float(t.pvalue), "zero mean", "bias: check trend/drift term")
    half = len(r) // 2
    if half > 15:
        f = np.var(r[half:], ddof=1) / np.var(r[:half], ddof=1)
        pf = 2 * min(stats.f.cdf(f, half - 1, len(r) - half - 1), 1 - stats.f.cdf(f, half - 1, len(r) - half - 1))
        add("Variance ratio, 2nd half / 1st half", float(f), float(pf), "constant variance", "variance changed over time: consider a log transform or GARCH")
    return pd.DataFrame(rows)


def param_table(res) -> pd.DataFrame:
    """Parameters with robust (sandwich) and OPG standard errors side by side."""
    names = list(res.param_names)
    p = np.asarray(res.params, dtype=float)
    se_r = np.asarray(res.bse, dtype=float)
    try:
        se_o = np.sqrt(np.diag(res.cov_params_opg)) if hasattr(res, "cov_params_opg") else np.full(len(p), np.nan)
    except Exception:  # noqa: BLE001
        se_o = np.full(len(p), np.nan)
    z = p / se_r
    return pd.DataFrame({"Parameter": names, "Estimate": p, "Robust SE": se_r, "OPG SE": se_o, "z (robust)": z,
                         "p (robust)": 2 * (1 - stats.norm.cdf(np.abs(z)))})


def run_arima(yv: np.ndarray, order, seasonal, trend, exog, exog_future, h: int, alphas=(0.2, 0.05)) -> dict:
    res = fit_arima(yv, order, seasonal, trend, exog, cov_type="robust")
    n_arma = order[0] + order[2] + seasonal[0] + seasonal[2]
    fc = res.get_forecast(h, exog=None if exog_future is None else pd.DataFrame(exog_future, columns=["dummy"]))
    out = {"res": res, "mean": np.asarray(fc.predicted_mean), "sigma": float(np.sqrt(res.params.get("sigma2", np.nan))) if hasattr(res.params, "get") else np.nan}
    for a in alphas:
        ci = np.asarray(fc.conf_int(alpha=a))
        out[f"lo{int((1 - a) * 100)}"], out[f"hi{int((1 - a) * 100)}"] = ci[:, 0], ci[:, 1]
    out["checks"] = residual_checks(res, len(yv), order[1], seasonal[1], seasonal[3] or 1, n_arma)
    out["params"] = param_table(res)
    out["roots"] = {"ar": np.abs(res.arroots) if len(res.arroots) else np.array([]), "ma": np.abs(res.maroots) if len(res.maroots) else np.array([])}
    out["fitted"] = np.asarray(res.fittedvalues)
    out["resid"] = np.asarray(res.resid)
    out["aicc"], out["bic"], out["llf"] = float(res.aicc), float(res.bic), float(res.llf)
    return out


# ----------------------------------------------------------------------------- benchmarks, ETS, backtest
def f_naive(tr, h, m):
    return np.repeat(tr[-1], h)


def f_snaive(tr, h, m):
    if m <= 1 or len(tr) < m:
        return f_naive(tr, h, m)
    return np.array([tr[-m + (i % m)] for i in range(h)])


def f_drift(tr, h, m):
    return tr[-1] + (tr[-1] - tr[0]) / (len(tr) - 1) * np.arange(1, h + 1)


def ets_choose(yv: np.ndarray, m: int) -> dict:
    best, rows = None, []
    for trend, damped in ((None, False), ("add", False), ("add", True)):
        for seas in ([None, "add"] if m > 1 and len(yv) >= 2 * m + 2 else [None]):
            try:
                r = ExponentialSmoothing(yv, trend=trend, damped_trend=damped, seasonal=seas, seasonal_periods=m if seas else None,
                                         initialization_method="estimated").fit()
                rows.append({"trend": trend, "damped": damped, "seasonal": seas, "AICc": r.aicc})
            except Exception:  # noqa: BLE001
                continue
    if not rows:
        return {"trend": None, "damped": False, "seasonal": None}
    best = min(rows, key=lambda r: r["AICc"])
    return {k: best[k] for k in ("trend", "damped", "seasonal")}


def f_ets_factory(spec: dict):
    def f(tr, h, m):
        r = ExponentialSmoothing(tr, trend=spec["trend"], damped_trend=spec["damped"], seasonal=spec["seasonal"],
                                 seasonal_periods=m if spec["seasonal"] else None, initialization_method="estimated").fit()
        return np.asarray(r.forecast(h))
    return f


def f_arima_factory(order, seasonal, trend):
    def f(tr, h, m):
        r = fit_arima(tr, order, seasonal, trend)
        return np.asarray(r.get_forecast(h).predicted_mean)
    return f


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction. d = loss_a - loss_b; a negative statistic
    means model A has the lower loss. Two-sided p-value from Student t(n-1)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 6 or np.allclose(d, d[0]):
        return np.nan, np.nan
    dm = d - d.mean()
    gam = [np.sum(dm[k:] * dm[:n - k]) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), n - 1)))


@st.cache_data(show_spinner=False, max_entries=12)
def run_backtest(yv: np.ndarray, m: int, h: int, n_origins: int, step: int, arima_spec: tuple | None, use_ets: bool = True) -> dict:
    """Expanding-window, fixed-specification backtest: at each origin every model is refit on the data so far and forecasts h steps.
    (Orders are not re-selected at each origin; that would be slow and the selection would leak into the evaluation.)"""
    n = len(yv)
    last_origin = n - h
    origins = [last_origin - i * step for i in range(n_origins)][::-1]
    origins = [o for o in origins if o >= max(40, 3 * max(m, 4))]
    if len(origins) < 3:
        raise ValueError("series too short for this horizon / number of origins")
    models = {"Naive": f_naive, "Drift": f_drift}
    if m > 1:
        models["Seasonal naive"] = f_snaive
    if use_ets:
        models["ETS"] = f_ets_factory(ets_choose(yv[:origins[0]], m))
    if arima_spec:
        models["ARIMA"] = f_arima_factory(*arima_spec)
    errs = {k: np.full((len(origins), h), np.nan) for k in models}
    for i, o in enumerate(origins):
        tr, act = yv[:o], yv[o:o + h]
        for k, f in models.items():
            try:
                errs[k][i] = act - f(tr, h, m)
            except Exception:  # noqa: BLE001
                pass
    scale_lag = m if m > 1 else 1
    insample = yv[:origins[0]]
    scale = np.mean(np.abs(insample[scale_lag:] - insample[:-scale_lag])) or 1.0
    rows = []
    for k, e in errs.items():
        for hh in sorted({1, max(1, h // 2), h}):
            col = e[:, hh - 1]
            col = col[np.isfinite(col)]
            rows.append({"Model": k, "Horizon": hh, "RMSE": float(np.sqrt(np.mean(col ** 2))), "MAE": float(np.mean(np.abs(col))),
                         "MASE": float(np.mean(np.abs(col)) / scale), "Bias": float(np.mean(col)), "Origins": len(col)})
    by_h = pd.DataFrame({k: np.sqrt(np.nanmean(e ** 2, axis=0)) for k, e in errs.items()}, index=np.arange(1, h + 1))
    dm_rows = []
    base = "Seasonal naive" if "Seasonal naive" in errs else "Naive"
    for k in errs:
        if k == base:
            continue
        for hh in sorted({1, h}):
            a, b = errs[k][:, hh - 1] ** 2, errs[base][:, hh - 1] ** 2
            ok = np.isfinite(a) & np.isfinite(b)
            stat, p = dm_test(a[ok], b[ok], hh)
            dm_rows.append({"Model": k, "vs": base, "Horizon": hh, "DM stat": stat, "p-value": p,
                            "Reading": ("better" if stat < 0 else "worse") + (" (significant)" if p < 0.05 else " (not significant)") if np.isfinite(p) else "n/a"})
    return {"table": pd.DataFrame(rows), "by_h": by_h, "dm": pd.DataFrame(dm_rows), "origins": len(origins), "base": base}


# ----------------------------------------------------------------------------- GARCH family
GARCH_SPECS = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
}


def engle_ng(z: np.ndarray) -> dict:
    """Engle-Ng sign-bias tests on standardised residuals: z_t^2 on const, S-_{t-1}, S-_{t-1}z_{t-1}, S+_{t-1}z_{t-1}."""
    zt, zl = z[1:] ** 2, z[:-1]
    neg = (zl < 0).astype(float)
    X = sm.add_constant(np.column_stack([neg, neg * zl, (1 - neg) * zl]))
    r = sm.OLS(zt, X).fit(cov_type="HC3")
    return {"sign": float(r.pvalues[1]), "neg_size": float(r.pvalues[2]), "pos_size": float(r.pvalues[3]), "joint": float(r.f_test(np.eye(4)[1:]).pvalue)}


def fit_garch(r: np.ndarray, name: str, dist: str, ar_lags: int):
    spec = GARCH_SPECS[name]
    am = arch_model(r, mean="AR" if ar_lags else "Constant", lags=ar_lags if ar_lags else 0, dist=dist, rescale=False, **spec)
    return am.fit(disp="off", show_warning=False)


@st.cache_data(show_spinner=False, max_entries=12)
def run_garch(rv: np.ndarray, ar_lags: int, h: int, periods_per_year: int, dists: tuple, names: tuple) -> dict:
    """Fit {GARCH, GJR, EGARCH} x {normal, t, skew-t} on percentage returns; keep the lowest BIC. Standard errors are
    Bollerslev-Wooldridge robust. Diagnostics are computed on standardised residuals."""
    rows, fits = [], {}
    for nm, dist in product(names, dists):
        try:
            f = fit_garch(rv, nm, dist, ar_lags)
            rows.append({"Model": nm, "Distribution": dist, "LogLik": f.loglikelihood, "AIC": f.aic, "BIC": f.bic, "Params": len(f.params)})
            fits[(nm, dist)] = f
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        raise ValueError("no GARCH specification converged")
    comp = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    nm, dist = comp.loc[0, "Model"], comp.loc[0, "Distribution"]
    f = fits[(nm, dist)]
    z = np.asarray(f.std_resid, dtype=float)
    z = z[np.isfinite(z)]
    lb_z, lb_zp = ljung_box(z, 10)
    lb_z2, lb_z2p = ljung_box(z ** 2, 10)
    lm, lmp = arch_lm(z, 10)
    checks = pd.DataFrame([
        {"Check": "Ljung-Box (10) on standardised residuals", "Statistic": lb_z, "p-value": lb_zp, "Result": "PASS" if lb_zp >= .05 else "FAIL",
         "If FAIL": "mean equation: raise AR lags"},
        {"Check": "Ljung-Box (10) on squared standardised residuals", "Statistic": lb_z2, "p-value": lb_z2p, "Result": "PASS" if lb_z2p >= .05 else "FAIL",
         "If FAIL": "variance equation too simple: try another family / more lags"},
        {"Check": "ARCH-LM (10) on standardised residuals", "Statistic": lm, "p-value": lmp, "Result": "PASS" if lmp >= .05 else "FAIL",
         "If FAIL": "ARCH effects remain"}])
    eng = engle_ng(z)
    checks = pd.concat([checks, pd.DataFrame([{"Check": "Engle-Ng sign-bias (joint)", "Statistic": np.nan, "p-value": eng["joint"],
                                               "Result": "PASS" if eng["joint"] >= .05 else "FAIL",
                                               "If FAIL": "leverage not captured: prefer GJR / EGARCH"}])], ignore_index=True)
    classic = None
    try:
        am = arch_model(rv, mean="AR" if ar_lags else "Constant", lags=ar_lags if ar_lags else 0, dist=dist, rescale=False, **GARCH_SPECS[nm])
        classic = am.fit(disp="off", show_warning=False, cov_type="classic").std_err
    except Exception:  # noqa: BLE001
        pass
    pars = pd.DataFrame({"Parameter": f.params.index, "Estimate": f.params.values, "Robust SE": f.std_err.values,
                         "Classic SE": classic.values if classic is not None else np.nan, "t (robust)": f.tvalues.values,
                         "p (robust)": f.pvalues.values})
    par = f.params
    alpha = par.filter(like="alpha[").sum()
    beta = par.filter(like="beta[").sum()
    gamma = par.filter(like="gamma[").sum()
    pers = beta if nm.startswith("EGARCH") else alpha + beta + (gamma / 2 if gamma else 0)
    omega = par.get("omega", np.nan)
    unc = np.sqrt(omega / (1 - pers)) if (not nm.startswith("EGARCH") and pers < 1 and np.isfinite(omega)) else np.nan
    half = np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.nan
    try:
        fc = f.forecast(horizon=h, reindex=False, method="analytic") if not nm.startswith("EGARCH") else f.forecast(horizon=h, reindex=False, method="simulation", simulations=1000)
    except Exception:  # noqa: BLE001
        fc = f.forecast(horizon=h, reindex=False, method="simulation", simulations=1000)
    var_path = np.asarray(fc.variance.values[-1], dtype=float)
    mu_next = float(np.asarray(fc.mean.values[-1])[0])
    q = np.quantile(z, [0.01, 0.05])
    sig1 = float(np.sqrt(var_path[0]))
    return {"comparison": comp, "name": nm, "dist": dist, "params": pars, "checks": checks, "engle_ng": eng,
            "cond_vol": np.asarray(f.conditional_volatility), "std_resid": z, "persistence": float(pers), "half_life": float(half),
            "uncond_vol": float(unc), "var_path": var_path, "ann": periods_per_year,
            "var1": float(mu_next + q[1] * sig1), "var99": float(mu_next + q[0] * sig1), "mu_next": mu_next, "sig1": sig1,
            "bic": float(f.bic), "loglik": float(f.loglikelihood)}


# ----------------------------------------------------------------------------- OLS lab
def auto_stationary(s: pd.Series, unit: str, name: str) -> tuple[pd.Series, str, int]:
    """log if strictly positive and not a rate; then difference by the unit-root verdict (max 1)."""
    z, lab_ = s.dropna(), ""
    kind, _ = suggest_transform(z, unit, name)
    if kind == "log":
        z, lab_ = np.log(z), "ln "
    d, _ = integration_order(z, 1, "c")
    if d:
        z = z.diff().dropna()
        lab_ = ("D" + lab_) if lab_ else "D"
    return z, (lab_ + name) if lab_ else name, d


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    cols_ = [c for c in X.columns if c != "const"]
    if len(cols_) < 2:
        return pd.DataFrame({"Variable": cols_, "VIF": [1.0] * len(cols_)})
    Xv = sm.add_constant(X[cols_])
    return pd.DataFrame({"Variable": cols_, "VIF": [variance_inflation_factor(Xv.values, i + 1) for i in range(len(cols_))]})


def ols_lab(y: pd.Series, X: pd.DataFrame, cov_choice: str, hac_lags: int) -> dict:
    """OLS with every standard diagnostic; reports classic / HC3 / HAC standard errors side by side and picks one by the tests."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    Y, XX = d["y"], sm.add_constant(d.drop(columns="y"), has_constant="add")
    n, k = XX.shape
    base = sm.OLS(Y, XX).fit()
    L = hac_lags or nw_lags(n)
    ses = {"Classic": base, "HC3": sm.OLS(Y, XX).fit(cov_type="HC3"),
           f"HAC (NW, {L} lags)": sm.OLS(Y, XX).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})}
    resid = base.resid.values
    tests = []
    try:
        bp = het_breuschpagan(resid, XX.values)
        tests.append(("Breusch-Pagan (studentised)", bp[0], bp[1], "homoskedastic", "heteroskedasticity"))
    except Exception:  # noqa: BLE001
        pass
    try:
        wt = het_white(resid, XX.values)
        tests.append(("White", wt[0], wt[1], "homoskedastic", "heteroskedasticity"))
    except Exception:  # noqa: BLE001
        pass
    try:
        bg_l = int(min(max(L, 4), n // 4))
        bg = acorr_breusch_godfrey(base, nlags=bg_l)
        tests.append((f"Breusch-Godfrey ({bg_l} lags)", bg[0], bg[1], "no serial correlation", "serial correlation"))
    except Exception:  # noqa: BLE001
        pass
    try:
        lb = ljung_box(resid, min(10, n // 5), 0)
        tests.append(("Ljung-Box (10) on residuals", lb[0], lb[1], "no autocorrelation", "serial correlation"))
    except Exception:  # noqa: BLE001
        pass
    try:
        lm, lmp = arch_lm(resid, min(5, n // 8))
        tests.append(("ARCH-LM (5) on residuals", lm, lmp, "no ARCH", "conditional heteroskedasticity"))
    except Exception:  # noqa: BLE001
        pass
    jb, jbp, _sk, _ku = jarque_bera(resid)
    tests.append(("Jarque-Bera", jb, jbp, "normal residuals", "non-normal residuals (inference relies on asymptotics)"))
    try:
        rs = linear_reset(base, power=[2, 3], use_f=True)
        tests.append(("Ramsey RESET (powers 2-3)", float(rs.statistic), float(rs.pvalue), "correct functional form", "functional-form misspecification / omitted nonlinearity"))
    except Exception:  # noqa: BLE001
        pass
    try:
        cu = breaks_cusumolsresid(resid, ddof=k)
        tests.append(("CUSUM (OLS residuals)", float(cu[0]), float(cu[1]), "stable parameters", "structural break / unstable parameters"))
    except Exception:  # noqa: BLE001
        pass
    tbl = pd.DataFrame([{"Test": a, "Statistic": b, "p-value": c, "H0": e, "Result": "PASS" if c >= .05 else "FAIL", "If FAIL": f}
                        for a, b, c, e, f in tests])
    het = bool(any((t[0].startswith(("Breusch-Pagan", "White", "ARCH")) and t[2] < .05) for t in tests))
    ser = bool(any((t[0].startswith(("Breusch-Godfrey", "Ljung")) and t[2] < .05) for t in tests))
    pick = f"HAC (NW, {L} lags)" if ser else "HC3"
    if cov_choice != "auto":
        pick = {"Classic": "Classic", "HC3": "HC3", "HAC": f"HAC (NW, {L} lags)"}[cov_choice]
    chosen = ses[pick]
    coef = pd.DataFrame({"Variable": XX.columns, "Coefficient": base.params.values})
    for nm, r_ in ses.items():
        coef[f"SE {nm}"] = r_.bse.values
    coef["t (chosen)"], coef["p (chosen)"] = chosen.tvalues.values, chosen.pvalues.values
    return {"n": n, "k": k, "r2": base.rsquared, "adj_r2": base.rsquared_adj, "dw": float(durbin_watson(resid)), "aic": base.aic, "bic": base.bic,
            "coef": coef, "tests": tbl, "picked": pick, "heterosked": het, "serial": ser, "hac_lags": L, "vif": vif_table(XX),
            "cond": float(np.linalg.cond(XX.values)), "fitted": base.fittedvalues, "resid": base.resid,
            "wald_p": float(chosen.f_pvalue) if chosen.f_pvalue is not None else np.nan, "y": Y}


# ----------------------------------------------------------------------------- insight layer
SHORT_LONG = {"D": (63, 252), "W": (13, 52), "M": (3, 12), "Q": (1, 4), "A": (1, 2)}      # ~3 months, ~12 months in periods
PERIOD_WORD = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "A": "year"}
EXPECTED_GAP = {"D": 1, "W": 7, "M": 31, "Q": 92, "A": 366}
LMAX = {"D": 20, "W": 13, "M": 12, "Q": 6, "A": 3}
RATE_WORDS = re.compile(r"yield|inflation|spread|p/e|p/b|\bpe\b|\bpb\b|ratio|unemployment|pmi|vix|margin|yoy|growth|deficit|"
                        r"balance|sentiment|confidence|repo|\brate\b", re.I)


def fnum(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    a = abs(v)
    return f"{v:,.0f}" if a >= 10000 else f"{v:,.1f}" if a >= 100 else f"{v:,.2f}"


def is_rate_like(y: pd.Series, unit: str, name: str) -> bool:
    """Rates, ratios, spreads and surveys are read as levels; indices and prices are read through their growth."""
    if suggest_transform(y, unit, name)[0] == "level":
        return True
    return bool(RATE_WORDS.search(name)) and "exchange" not in name.lower()


def robust_z(x: pd.Series) -> pd.Series:
    """Distance from the median in robust standard deviations (MAD), so one spike cannot distort the yardstick."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        mad = x.std()
    return (x - med) / mad if (np.isfinite(mad) and mad > 0) else x * 0.0


def gauge_of(s: pd.Series, unit: str, name: str, freq: str) -> tuple[pd.Series, str, bool]:
    """The number worth comparing with history: the level for a rate / ratio, the year-on-year % change for an index or
    price (whose level only ever trends). Returns (gauge, label, rate_like)."""
    rate = is_rate_like(s, unit, name)
    k = PERIODS_PER_YEAR[freq]
    if rate:
        return s, (unit or "level"), True
    if len(s) < k + max(12, k // 2) or (s <= 0).any():
        return s, (unit or "level"), False
    return (100 * (s / s.shift(k) - 1)).dropna(), "% YoY", False


def since_when(g: pd.Series) -> str:
    """'highest since Jun-2019' style statement for the latest reading; empty if it says nothing (under a year)."""
    last, prev = g.iloc[-1], g.iloc[:-1]
    if len(prev) < 8:
        return ""
    hi = last >= prev.median()
    beyond = prev[prev >= last] if hi else prev[prev <= last]
    word = "highest" if hi else "lowest"
    if beyond.empty:
        return f"{word} in the sample (from {g.index[0]:%b-%Y})" if (g.index[-1] - g.index[0]).days >= 365 else ""
    return f"{word} since {beyond.index[-1]:%b-%Y}" if (g.index[-1] - beyond.index[-1]).days >= 365 else ""


def change_k(s: pd.Series, k: int, rate: bool) -> pd.Series:
    """Change over k periods: points for a rate, % (log) for an index / price."""
    if rate or (s <= 0).any():
        return s.diff(k)
    return 100 * np.log(s).diff(k)


def snapshot_reg(s: pd.Series, unit: str, name: str, freq: str) -> dict | None:
    """Where a (regularised) series stands against its own history, how it is moving, and how unusual the latest move is."""
    s = s.dropna()
    if len(s) < 24:
        return None
    g, glabel, rate = gauge_of(s, unit, name, freq)
    if len(g) < 12:
        return None
    ks, kl = SHORT_LONG[freq]
    last = float(g.iloc[-1])
    base = change_k(s, 1, rate).dropna()
    dS, dL = (g - g.shift(ks)).dropna(), (g - g.shift(kl)).dropna()
    mom_z = float(dS.iloc[-1] / dS.std()) if len(dS) > 8 and dS.std() > 0 else np.nan
    long_z = float(dL.iloc[-1] / dL.std()) if len(dL) > 8 and dL.std() > 0 else np.nan
    if not np.isfinite(mom_z) or abs(mom_z) < 0.5:
        mom = "Flat"
    elif np.isfinite(long_z) and abs(long_z) >= 0.5 and np.sign(long_z) != np.sign(mom_z):
        mom = "Turning up" if mom_z > 0 else "Turning down"
    else:
        mom = "Rising" if mom_z > 0 else "Falling"
    hlv = {"D": 21, "W": 8, "M": 6, "Q": 4, "A": 3}[freq]                      # recent turbulence: EWMA size of one-period moves
    ev = np.sqrt(((base - base.mean()) ** 2).ewm(halflife=hlv).mean()).iloc[3 * hlv:]
    vol_pct = float((ev.iloc[:-1] < ev.iloc[-1]).mean() * 100) if len(ev) > 20 else np.nan
    tail = g.iloc[-max(3 * PERIODS_PER_YEAR[freq], 12):]
    spark = [float(v) for v in tail.iloc[::max(1, len(tail) // 60)].values]
    return {"vol_pct": vol_pct, "vol_now": float(ev.iloc[-1] * np.sqrt(PERIODS_PER_YEAR[freq])) if len(ev) else np.nan, "spark": spark,
            "freq": freq, "asof": s.index[-1], "latest": float(s.iloc[-1]), "unit": unit or "", "glabel": glabel, "gval": last,
            "pct": float((g.iloc[:-1] < last).mean() * 100), "z": float(robust_z(g).iloc[-1]), "median": float(g.median()),
            "p10": float(g.quantile(.1)), "p90": float(g.quantile(.9)), "since": since_when(g), "mom": mom, "mom_z": mom_z,
            "mom_chg": float(dS.iloc[-1]) if len(dS) else np.nan, "long_z": long_z,
            "move": float(base.iloc[-1]) if len(base) else np.nan, "move_z": float(robust_z(base).iloc[-1]) if len(base) > 8 else np.nan,
            "move_unit": "pts" if (rate or (s <= 0).any()) else "%", "years": (s.index[-1] - s.index[0]).days / 365.25,
            "rate": rate, "gauge": g}


@st.cache_data(show_spinner="Scanning every series...", max_entries=4)
def briefing_table(_data: pd.DataFrame, sig: tuple) -> pd.DataFrame:
    """One row per series in the upload: standing vs own history, momentum and the size of the latest move."""
    rows, newest = [], _data["date"].max()
    for (cat, name), sub in _data.groupby(["category", "series"], sort=True):
        try:
            sn = sub.groupby("date")["value"].last().dropna().sort_index()
            if len(sn) < 24:
                continue
            fq = detect_freq(sn.index)
            sp = snapshot_reg(regularize(sn, fq), str(sub["unit"].iloc[0]), name, fq)
        except Exception:  # noqa: BLE001
            sp = None
        if sp is None:
            continue
        sp.pop("gauge")
        rows.append({"category": cat, "series": name, **sp, "stale": (newest - sp["asof"]).days > 3 * EXPECTED_GAP[sp["freq"]] + 10})
    return pd.DataFrame(rows)


def headline_lists(B: pd.DataFrame, n: int = 6) -> dict:
    """The three things worth a look first: readings at an extreme, unusually large latest moves, changes of direction."""
    F = B[~B["stale"]]
    ext = F[(F["pct"] >= 95) | (F["pct"] <= 5)].assign(a=lambda d: (d["pct"] - 50).abs(), b=lambda d: d["z"].abs())
    ext = ext.sort_values(["a", "b"], ascending=False).head(n)
    mv = F[F["move_z"].abs() >= 2.5].assign(a=lambda d: d["move_z"].abs()).sort_values("a", ascending=False).head(n)
    tn = F[F["mom"].str.startswith("Turning") & (F["mom_z"].abs() >= 1)].assign(a=lambda d: d["mom_z"].abs()).sort_values("a", ascending=False).head(n)
    out = {"ext": [], "mv": [], "tn": []}
    for r in ext.itertuples():
        out["ext"].append(f"<b>{r.series}</b> at {fnum(r.gval)} {r.glabel} - " + (r.since if r.since else f"{'above' if r.pct > 50 else 'below'} {max(r.pct, 100 - r.pct):.0f}% of its history"))
    for r in mv.itertuples():
        out["mv"].append(f"<b>{r.series}</b> {r.move:+.2f} {r.move_unit} in the latest {PERIOD_WORD[r.freq]} - {abs(r.move_z):.1f}x a normal move ({r.asof:%d %b})")
    for r in tn.itertuples():
        out["tn"].append(f"<b>{r.series}</b> {r.mom.lower()} after a {'rise' if r.long_z > 0 else 'fall'} over the past year (now {fnum(r.gval)} {r.glabel})")
    return out


@st.cache_data(show_spinner=False, max_entries=24)
def track_record(yv: np.ndarray, order: tuple, seasonal: tuple, trend: str, h: int, m: int, n_or: int = 12) -> dict | None:
    """Would this model have been worth listening to? Refit at past dates, forecast h ahead, and compare with what happened:
    error vs simply assuming 'no change', share of direction calls that were right, and how often the 80% band held."""
    n = len(yv)
    min_train, last = max(40, 3 * max(m, 4)), n - h
    if last - min_train < 3:
        return None
    step = max(1, min(h, (last - min_train) // max(n_or - 1, 1)))
    rows = []
    for o in sorted(o for o in (last - i * step for i in range(n_or)) if o >= min_train):
        try:
            fc = fit_arima(yv[:o], order, seasonal, trend).get_forecast(h)
            lo, hi = np.asarray(fc.conf_int(alpha=0.2))[-1]
            rows.append((yv[o - 1], float(np.asarray(fc.predicted_mean)[-1]), lo, hi, yv[o + h - 1], o))
        except Exception:  # noqa: BLE001
            continue
    if len(rows) < 4:
        return None
    a = np.array(rows, dtype=float)
    start, mu, lo, hi, act, oo = a.T
    rm, rn = np.sqrt(np.mean((act - mu) ** 2)), np.sqrt(np.mean((act - start) ** 2))
    called = np.abs(mu - start) > 1e-9 * np.maximum(np.abs(start), 1)
    hits = int(np.sum(np.sign(mu - start)[called] == np.sign(act - start)[called]))
    return {"n": len(a), "skill": float(1 - rm / rn) if rn > 0 else np.nan, "hits": hits, "calls": int(called.sum()),
            "cover": int(np.sum((act >= lo) & (act <= hi))), "step": step, "o": oo.astype(int), "start": start, "mu": mu, "lo": lo, "hi": hi, "act": act}


def track_fig(y: pd.Series, rec: dict, h: int, inv, unit: str, height=330):
    """Each past replay drawn on the series: where the model said it would go (with its 8-in-10 range) and where it went."""
    oo = rec["o"]
    hist = y.iloc[max(0, int(oo.min()) - 1 - 2 * h):]
    fig = go.Figure(go.Scatter(x=hist.index, y=inv(hist.values), name="what happened", line=dict(color=TEXT, width=1.6)))
    xs, ys = [], []
    for i, o in enumerate(oo):
        xs += [y.index[o - 1], y.index[o + h - 1], None]
        ys += [float(inv(rec["start"][i])), float(inv(rec["mu"][i])), None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="forecast made at the time", line=dict(color=GOLD, width=1.3, dash="dot"), hoverinfo="skip"))
    tx = [y.index[o + h - 1] for o in oo]
    mu_, lo_, hi_, act_ = (np.asarray(inv(rec[k_]), dtype=float) for k_ in ("mu", "lo", "hi", "act"))
    inside = (rec["act"] >= rec["lo"]) & (rec["act"] <= rec["hi"])
    fig.add_trace(go.Scatter(x=tx, y=mu_, mode="markers", name="forecast and its 8-in-10 range", marker=dict(color=GOLD, size=8, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False, array=hi_ - mu_, arrayminus=mu_ - lo_, color=GOLD, thickness=1.2, width=4)))
    fig.add_trace(go.Scatter(x=tx, y=act_, mode="markers", name="outcome (green = inside the range, red = outside)",
                             marker=dict(size=10, color=[GREEN if i else RED for i in inside], line=dict(color=BG, width=1))))
    layout(fig, height, "Would the model have been worth listening to? Past forecasts against what happened")
    fig.update_layout(hovermode="closest", margin=dict(t=58), legend=dict(y=1.13))
    fig.update_yaxes(title=unit)
    return fig


def episode_paths(signal: pd.Series, level: pd.Series, k: int, lo: float, hi: float, pct: bool):
    """Separate (non-overlapping) past visits of `signal` to the band [lo, hi]: the level's path from k periods before to k
    periods after each visit, rebased to zero at the visit. Also today's run-up, for comparison."""
    d = pd.concat([signal.rename("s"), level.rename("l")], axis=1).dropna(subset=["l"])
    lv, sv, n = d["l"].values.astype(float), d["s"].values.astype(float), len(d)
    rebase = (lambda seg, b: 100 * (seg / b - 1)) if pct else (lambda seg, b: seg - b)
    paths, last_p = [], -10 ** 9
    for p_ in np.where((sv >= lo) & (sv <= hi))[0]:
        if p_ - last_p >= k and p_ + k < n:
            a = max(0, p_ - k)
            paths.append((d.index[p_], np.arange(a - p_, k + 1), rebase(lv[a:p_ + k + 1], lv[p_])))
            last_p = p_
    a = max(0, n - 1 - k)
    return paths, (np.arange(a - (n - 1), 1), rebase(lv[a:], lv[-1]))


def episode_fig(paths, today, k: int, unit: str, word: str, height=300):
    fig = go.Figure()
    for i, (dt, x, v) in enumerate(paths):
        fig.add_trace(go.Scatter(x=x, y=v, mode="lines", line=dict(color=PURPLE, width=1.1), opacity=0.55, name="past episodes", legendgroup="p",
                                 showlegend=(i == 0), hovertemplate=f"from {dt:%b-%Y}: " + "%{y:+.2f} after %{x} " + word + "s<extra></extra>"))
    fwd = np.array([v[x >= 0] for _dt, x, v in paths])
    fig.add_trace(go.Scatter(x=np.arange(0, k + 1), y=np.median(fwd, axis=0), mode="lines", name="typical path after", line=dict(color=GOLD, width=3)))
    fig.add_trace(go.Scatter(x=today[0], y=today[1], mode="lines", name="this time, so far", line=dict(color=TEXT, width=2.6)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    layout(fig, height, f"The {len(paths)} separate times it stood here: the run-up and what followed")
    fig.update_layout(hovermode="closest", margin=dict(t=74), legend=dict(y=1.2))
    fig.update_xaxes(title=f"{word}s before / after standing at this level")
    fig.update_yaxes(title=unit)
    return fig


def kupiec_p(x: int, n: int, p: float) -> float:
    """Kupiec proportion-of-failures test: is the share of VaR breaches consistent with the promised rate p?"""
    if n <= 0:
        return np.nan
    ph = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - ph) if x < n else 0.0) + (x * np.log(ph) if x > 0 else 0.0)
    return float(1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1))


@st.cache_data(show_spinner=False, max_entries=12)
def garch_validation(rv: np.ndarray, name: str, dist: str, ar_lags: int, lam: float) -> dict | None:
    """Honest test of the chosen volatility model: estimate it on the first 70% of the sample only, then let it forecast the
    remaining 30% one period at a time with those parameters frozen. Compared with (a) a plain EWMA of squared moves
    (RiskMetrics) on QLIKE loss, Diebold-Mariano tested, and (b) its own VaR promises (Kupiec test on breaches)."""
    n = len(rv)
    split = int(n * 0.7)
    if split < 80 or n - split < 40:
        return None
    am = arch_model(rv, mean="AR" if ar_lags else "Constant", lags=ar_lags if ar_lags else 0, dist=dist, rescale=False, **GARCH_SPECS[name])
    f = am.fit(last_obs=split, disp="off", show_warning=False)
    fc = f.forecast(horizon=1, start=split - 1, reindex=False)
    hv = np.asarray(fc.variance.values[:, 0], dtype=float)[:-1]          # row t holds the forecast for t+1
    mf = np.asarray(fc.mean.values[:, 0], dtype=float)[:-1]
    r_out = rv[split:]
    e2 = (r_out - mf) ** 2
    mu = rv[:split].mean()
    d2 = (rv - mu) ** 2
    s2 = np.empty(n)
    s2[0] = d2[:split].mean()
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * d2[t - 1]
    he = s2[split:]
    ok = np.isfinite(hv) & (hv > 0) & (he > 0)
    if ok.sum() < 30:
        return None
    ql_g, ql_e = np.log(hv[ok]) + e2[ok] / hv[ok], np.log(he[ok]) + e2[ok] / he[ok]
    _dm, dmp = dm_test(ql_g, ql_e, 1)
    z = np.asarray(f.std_resid, dtype=float)
    z = z[np.isfinite(z)]
    q5, q1 = np.quantile(z, [0.05, 0.01])
    v95, v99 = mf + q5 * np.sqrt(hv), mf + q1 * np.sqrt(hv)
    x95, x99, no = int(np.sum(r_out[ok] < v95[ok])), int(np.sum(r_out[ok] < v99[ok])), int(ok.sum())
    return {"split": split, "n_out": no, "ql_g": float(ql_g.mean()), "ql_e": float(ql_e.mean()), "dm_p": dmp, "x95": x95, "x99": x99,
            "p95": kupiec_p(x95, no, 0.05), "p99": kupiec_p(x99, no, 0.01), "v95": v95, "r_out": r_out, "sig_g": np.sqrt(hv), "sig_e": np.sqrt(he), "lam": lam}


def prob_above(mean, lo95, hi95, thr: float) -> np.ndarray:
    """P(value > thr) at each horizon from the forecast mean and 95% band (Gaussian, in the modelled units)."""
    se = (np.asarray(hi95) - np.asarray(lo95)) / (2 * 1.959964)
    return 1 - stats.norm.cdf((thr - np.asarray(mean)) / np.where(se > 0, se, np.nan))


def conditional_outcomes(signal: pd.Series, level: pd.Series, k: int, pct_change: bool, nb: int = 5) -> dict | None:
    """What happened to `level` over the next k periods, grouped by where `signal` stood at the time (history split into
    nb equal-count bands). Windows overlap, so the number of independent episodes is roughly observations / k."""
    fwd = (100 * (level.shift(-k) / level - 1)) if pct_change else (level.shift(-k) - level)
    d = pd.concat([signal.rename("sig"), fwd.rename("fwd")], axis=1).dropna()
    if len(d) < max(40, 4 * nb) or d["sig"].nunique() < nb:
        return None
    try:
        d["b"], edges = pd.qcut(d["sig"], nb, labels=False, retbins=True, duplicates="drop")
    except Exception:  # noqa: BLE001
        return None
    nbk = len(edges) - 1
    if nbk < 3:
        return None
    rows = []
    for b in range(nbk):
        f = d.loc[d["b"] == b, "fwd"]
        rows.append({"Band": f"{fnum(edges[b])} to {fnum(edges[b + 1])}", "lo": edges[b], "hi": edges[b + 1], "Cases": len(f),
                     "Median": f.median(), "p25": f.quantile(.25), "p75": f.quantile(.75), "Rose %": float((f > 0).mean() * 100)})
    now = float(signal.dropna().iloc[-1])
    cur = int(np.clip(np.searchsorted(edges, now, side="right") - 1, 0, nbk - 1))
    return {"table": pd.DataFrame(rows), "current": cur, "now": now, "episodes": len(d) / max(k, 1), "n": len(d),
            "all_median": float(d["fwd"].median()), "all_rose": float((d["fwd"] > 0).mean() * 100)}


def outcome_fig(co: dict, title: str, unit: str, height=300):
    t = co["table"]
    fig = go.Figure(go.Bar(x=t["Band"], y=t["Median"], marker_color=[GOLD if i == co["current"] else MUTED for i in range(len(t))],
                           error_y=dict(type="data", symmetric=False, array=t["p75"] - t["Median"], arrayminus=t["Median"] - t["p25"], color=TEXT, thickness=1.2),
                           customdata=np.column_stack([t["Cases"], t["Rose %"]]),
                           hovertemplate="%{x}<br>median %{y:.2f}<br>%{customdata[0]:.0f} cases, rose in %{customdata[1]:.0f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    layout(fig, height, title, legend=False)
    fig.update_xaxes(title="where the signal stood (gold = where it is now)", type="category")
    fig.update_yaxes(title=unit)
    return fig


def xcorr(a: np.ndarray, b: np.ndarray, maxlag: int) -> dict:
    """corr(a_t, b_{t-L}) for L in -maxlag..maxlag on a common grid (NaN allowed). L > 0: b moves first."""
    out = {}
    for L in range(-maxlag, maxlag + 1):
        ya, xb = (a[L:], b[:len(b) - L]) if L > 0 else (a[:len(a) + L], b[-L:]) if L < 0 else (a, b)
        ok = np.isfinite(ya) & np.isfinite(xb)
        out[L] = float(np.corrcoef(ya[ok], xb[ok])[0, 1]) if ok.sum() >= 20 and np.std(ya[ok]) > 0 and np.std(xb[ok]) > 0 else np.nan
    return out


def pair_changes(y_nat: pd.Series, y_unit: str, y_name: str, x_nat: pd.Series, x_unit: str, x_name: str, freq: str, months: int):
    """Both series on the coarser of the two calendars, each as its change over the chosen span. Returns (frame, freq, k)."""
    nat = detect_freq(x_nat.index)
    cf = freq if ORDER.index(freq) >= ORDER.index(nat) else nat
    ya, xa = regularize(y_nat, cf), regularize(x_nat, cf)
    k = 1 if months == 0 else max(1, int(round(PERIODS_PER_YEAR[cf] * months / 12)))
    df = pd.concat([change_k(ya, k, is_rate_like(ya, y_unit, y_name)).rename("y"), change_k(xa, k, is_rate_like(xa, x_unit, x_name)).rename("x")], axis=1)
    df = df.dropna() if cf == "D" else df.loc[max(ya.index[0], xa.index[0]):min(ya.index[-1], xa.index[-1])]
    return df, cf, k, ya, xa


def lag_reading(df: pd.DataFrame, cf: str, k: int) -> dict | None:
    a, b = df["y"].values.astype(float), df["x"].values.astype(float)
    n = int((np.isfinite(a) & np.isfinite(b)).sum())
    L = LMAX[cf]
    if n < max(30, 3 * k + L):
        return None
    cc = xcorr(a, b, L)
    v = {l: c for l, c in cc.items() if np.isfinite(c)}
    if not v:
        return None
    best = max(v, key=lambda l: abs(v[l]))
    # noise level for the BEST of 2L+1 timings: overlapping k-period changes leave about n/k independent observations and
    # about (2L+1)/k independent timings, so the 5% bar is raised accordingly (Bonferroni) instead of the usual 2/sqrt(n)
    thr = float(stats.norm.ppf(1 - 0.025 / max(1.0, (2 * L + 1) / k)) / np.sqrt(max(n / k, 4)))
    ya, xb = (a[best:], b[:len(b) - best]) if best > 0 else (a[:len(a) + best], b[-best:]) if best < 0 else (a, b)
    ok = np.isfinite(ya) & np.isfinite(xb)
    ya, xb = ya[ok], xb[ok]
    h = len(ya) // 2
    try:
        c1, c2 = np.corrcoef(ya[:h], xb[:h])[0, 1], np.corrcoef(ya[h:], xb[h:])[0, 1]
        stable = bool(np.sign(c1) == np.sign(c2) == np.sign(v[best]) and min(abs(c1), abs(c2)) >= 0.1)
    except Exception:  # noqa: BLE001
        stable = False
    return {"cc": cc, "best": int(best), "corr": float(v[best]), "corr0": float(cc.get(0, np.nan)), "thr": float(thr), "n": n, "stable": stable}


@st.cache_data(show_spinner="Comparing with every other series...", max_entries=8)
def leadlag_scan(_data: pd.DataFrame, sig: tuple, cat: str, sname: str, freq: str, months: int, win: tuple) -> pd.DataFrame:
    """For the chosen series, find which other series' changes move before, with, or after it."""
    me = _data[(_data["category"] == cat) & (_data["series"] == sname)]
    y_nat = me.groupby("date")["value"].last().dropna().sort_index().loc[win[0]:win[1]]
    y_unit = str(me["unit"].iloc[0])
    rows = []
    for (c, s), sub in _data.groupby(["category", "series"], sort=True):
        if (c, s) == (cat, sname):
            continue
        try:
            x_nat = sub.groupby("date")["value"].last().dropna().sort_index()
            if len(x_nat) < 30:
                continue
            df, cf, k, _ya, _xa = pair_changes(y_nat, y_unit, sname, x_nat, str(sub["unit"].iloc[0]), s, freq, months)
            r = lag_reading(df, cf, k)
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            continue
        b, w = r["best"], PERIOD_WORD[cf]
        rows.append({"Category": c, "Series": s, "Timing": "moves together" if b == 0 else f"leads by {b} {w}{'s' if b > 1 else ''}" if b > 0 else f"follows by {-b} {w}{'s' if b < -1 else ''}",
                     "Lag": b, "Link": r["corr"], "Same-time link": r["corr0"], "Noise level": r["thr"], "Above noise": abs(r["corr"]) >= r["thr"],
                     "Holds in both halves": r["stable"], "Pairs": r["n"], "Freq": cf})
    if not rows:
        return pd.DataFrame()
    T = pd.DataFrame(rows)
    T["_rank"] = (T["Above noise"] & T["Holds in both halves"]).astype(int) * 10 + T["Link"].abs()
    return T.sort_values("_rank", ascending=False).drop(columns="_rank").reset_index(drop=True)


# ----------------------------------------------------------------------------- UI helpers
CSS = f"""
<style>
.stApp{{background:{BG};color:{TEXT}}}
[data-testid="stSidebar"]{{background:{PANEL};border-right:1px solid {GRID}}}
h1,h2,h3{{color:{GOLD}!important;font-family:{MONO}!important;letter-spacing:.02em}}
.lab-hdr{{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid {GRID};padding:4px 0 10px 0;margin-bottom:8px}}
.lab-hdr b{{color:{GOLD};font-family:{MONO};font-size:20px;letter-spacing:.12em}}
.lab-hdr span{{color:{MUTED};font-family:{MONO};font-size:12px}}
.card{{background:{PANEL};border:1px solid {GRID};border-left:3px solid {GOLD};border-radius:6px;padding:10px 14px;margin:6px 0;font-size:13px;line-height:1.55}}
.card.bad{{border-left-color:{RED}}} .card.good{{border-left-color:{GREEN}}}
.card.info{{border-left-color:{CYAN}}} .card.mute{{border-left-color:{MUTED}}}
.card b{{color:{GOLD}}}
.card ul{{margin:4px 0 0 0;padding-left:18px}} .card li{{margin:3px 0}}
.card .hd{{color:{MUTED};font-family:{MONO};font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
.chip{{background:{PANEL};border:1px solid {GRID};border-radius:6px;padding:10px 12px;margin:4px 0;min-height:104px}}
.chip .ct{{color:{MUTED};font-family:{MONO};font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
.chip .cv{{font-size:19px;font-weight:700;margin:3px 0 4px 0;line-height:1.2}}
.chip .ch{{color:{MUTED};font-size:12px;line-height:1.4}}
button[data-baseweb="tab"]{{font-family:{MONO};font-size:12px;letter-spacing:.06em}}
</style>"""


def card(html: str, kind: str = ""):
    st.markdown(f'<div class="card {kind}">{html}</div>', unsafe_allow_html=True)


def show(df: pd.DataFrame, fmt: dict | None = None, height: int | None = None, key: str | None = None):
    sty = df.style.format(fmt or {}, na_rep="-")
    for c in ("Result", "Stationary", "Reads as"):
        if c in df.columns:
            sty = sty.map(lambda v: f"color:{GREEN}" if v in ("PASS", True, "stationary") else f"color:{RED}" if v in ("FAIL", False, "non-stationary") else "", subset=[c])
    for c in [c for c in df.columns if c.startswith("p ") or c.startswith("p-") or c == "p-value" or c.endswith("p")]:
        if df[c].dtype.kind == "f":
            sty = sty.map(lambda v: f"color:{GOLD}" if (isinstance(v, float) and v < 0.05) else "", subset=[c])
    kw = {"height": height} if height else {}
    st.dataframe(sty, hide_index=True, width="stretch", key=key, **kw)


def acf_fig(x, nlags: int, title: str, partial=False, height=230):
    x = np.asarray(x, dtype=float)
    nlags = int(min(nlags, len(x) // 2 - 1))
    v = pacf(x, nlags=nlags, method="ywm") if partial else acf(x, nlags=nlags, fft=True)
    fig = go.Figure(go.Bar(x=list(range(1, nlags + 1)), y=v[1:], marker_color=[PURPLE if abs(a) > 1.96 / np.sqrt(len(x)) else MUTED for a in v[1:]]))
    b = 1.96 / np.sqrt(len(x))
    for s in (b, -b):
        fig.add_hline(y=s, line=dict(color=GOLD, width=1, dash="dot"))
    layout(fig, height, title, legend=False)
    fig.update_xaxes(title="lag")
    return fig


def line_fig(series: dict, height=300, title=None):
    fig = go.Figure()
    for i, (k, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=k, mode="lines", line=dict(color=SERIES_COLORS[i % 8], width=1.6)))
    return layout(fig, height, title, legend=len(series) > 1)


def fan_fig(hist: pd.Series, fdates, mean, bands: dict, inv, title, unit="", height=360):
    fig = go.Figure()
    h = hist.iloc[-min(len(hist), 72):]
    fig.add_trace(go.Scatter(x=h.index, y=inv(h.values), name="history", line=dict(color=TEXT, width=1.6)))
    for (lo, hi, name, col) in bands.get("list", []):
        fig.add_trace(go.Scatter(x=list(fdates) + list(fdates)[::-1], y=list(inv(hi)) + list(inv(lo))[::-1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor=col, name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[h.index[-1]] + list(fdates), y=[inv(h.values[-1])] + list(inv(mean)), name="forecast", line=dict(color=GOLD, width=2.4)))
    layout(fig, height, title)
    fig.update_yaxes(title=unit)
    return fig


def style_pass(df):
    return df


STATUS_COL = {"ok": GREEN, "warn": GOLD, "bad": RED, "info": CYAN}


def chips(items, per_row: int = 4):
    """Cards with a coloured top edge: (title, headline value, status, one-line hint)."""
    for i in range(0, len(items), per_row):
        row = items[i:i + per_row]
        for c, (t, v, stt, hint) in zip(st.columns(per_row)[:len(row)], row):
            col = STATUS_COL[stt]
            c.markdown(f'<div class="chip" style="border-top:3px solid {col}"><div class="ct">{t}</div>'
                       f'<div class="cv" style="color:{col}">{v}</div><div class="ch">{hint}</div></div>', unsafe_allow_html=True)


def check_chips(df: pd.DataFrame, per_row: int = 3):
    items = []
    for r in df.to_dict("records"):
        ok = r["Result"] == "PASS"
        name = r.get("Check", r.get("Test", "")).split(" (")[0].replace(" on residuals", "").replace("Ljung-Box", "Autocorrelation").replace("squared", "squared (volatility)")
        pv = r.get("p-value", float("nan"))
        items.append((name, "PASS" if ok else "FAIL", "ok" if ok else "bad", f"p = {pv:.3f}. " + ("" if ok else str(r.get("If FAIL", ""))[:90])))
    chips(items, per_row)


def forest(labels, est, lo, hi, pv, title, height=None):
    n = len(labels)
    col = [GREEN if p < .05 else MUTED for p in pv]
    fig = go.Figure(go.Scatter(x=est, y=labels, mode="markers", marker=dict(color=col, size=11),
                               error_x=dict(type="data", symmetric=False, array=np.asarray(hi) - np.asarray(est), arrayminus=np.asarray(est) - np.asarray(lo), color=MUTED, thickness=1.6),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=GOLD, width=1, dash="dot"))
    layout(fig, height or max(180, 46 * n + 70), title, legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


@st.cache_data(show_spinner=False, max_entries=16)
def cached_diag(y: pd.Series, freq: str, trend: str) -> dict:
    return diagnostics_bundle(y, freq, trend)


def ar_lags_by_bic(r: np.ndarray, maxlag: int = 5) -> int:
    best, bl = np.inf, 0
    for p in range(maxlag + 1):
        try:
            rr = ARIMA(pd.Series(r), order=(p, 0, 0)).fit()
            if rr.bic < best:
                best, bl = rr.bic, p
        except Exception:  # noqa: BLE001
            continue
    return bl


# ----------------------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="VIKA Forecast Lab", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="lab-hdr"><b>VIKA WEALTH</b><span>FORECAST LAB · what stands out · where it is heading · what moves with what · {BUILD}</span></div>', unsafe_allow_html=True)

    # ---------------- data
    with st.sidebar:
        st.markdown("### 1 · Data")
        ups = st.file_uploader("Upload the DATA-tab export (ZIP, CSV or Excel)", type=["zip", "csv", "xlsx"], accept_multiple_files=True)
        path = st.text_input("...or a local file path (when running on your Mac)", value=os.environ.get("FORECAST_LAB_DATA", ""))
        demo = st.checkbox("Use synthetic demo data", value=False)
    files = [(u.name, u.getvalue()) for u in ups]
    if path and os.path.isfile(path):
        files.append((os.path.basename(path), open(path, "rb").read()))
    try:
        data = demo_data() if (demo and not files) else (read_uploads(tuple(files)) if files else None)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the files: {e}")
        return
    if data is None:
        card("<b>Start here.</b> In the Macro Terminal open the <b>DATA</b> tab and download the ZIP (long layout works best), "
             "then upload it in the sidebar. Or tick <i>Use synthetic demo data</i> to try the lab.")
        return

    dsig = (len(data), str(data["date"].max()), int(data["series"].nunique()), float(np.nansum(data["value"].values)))
    pick = st.session_state.pop("pending_pick", None)          # a row clicked on the BRIEFING / RELATIONSHIPS tables
    if pick:
        st.session_state["cat_sel"], st.session_state["series_q"], st.session_state["series_sel"] = pick[0], "", pick[1]
    with st.sidebar:
        cats = sorted(data["category"].unique())
        if st.session_state.get("cat_sel") not in cats:
            st.session_state.pop("cat_sel", None)
        cat = st.selectbox("Category", cats, key="cat_sel")
        names = sorted(data.loc[data["category"] == cat, "series"].unique())
        q = st.text_input("Filter series", "", key="series_q")
        shown = [n for n in names if q.lower() in n.lower()] or names
        if st.session_state.get("series_sel") not in shown:
            st.session_state.pop("series_sel", None)
        sname = st.selectbox("Series", shown, key="series_sel")
    sub = data[(data["category"] == cat) & (data["series"] == sname)].sort_values("date")
    unit = sub["unit"].iloc[0] if len(sub) else ""
    s_native = sub.groupby("date")["value"].last()
    native = detect_freq(s_native.index)
    with st.sidebar:
        st.markdown("### 2 · Frequency & transformation")
        options = [f for f in ORDER if ORDER.index(f) >= ORDER.index(native)]
        freq = st.selectbox("Model frequency", options, index=0, help="Series can be aggregated to a lower frequency; D = observed trading days")
        how = st.radio("Aggregate with", ["last", "mean"], horizontal=True) if freq != native else "last"
        y_lvl = regularize(s_native, freq, how)
        sug, why = suggest_transform(y_lvl, unit, sname)
        kinds = list(TRANSFORMS)
        kind = st.selectbox("Transformation", kinds, index=kinds.index(sug), format_func=TRANSFORMS.get)
        st.caption(f"Suggested: **{TRANSFORMS[sug]}** - {why}.")
        if len(y_lvl) < 30:
            st.error(f"'{sname}' has only {len(y_lvl)} observation(s) at {freq} frequency - too few to model. Pick another series.")
            return
        lo, hi = y_lvl.index.min().to_pydatetime(), y_lvl.index.max().to_pydatetime()
        win = st.slider("Sample window", lo, hi, (lo, hi), format="YYYY-MM-DD")
        trend_mode = st.selectbox("Deterministics in unit-root tests", ["auto", "c", "ct"], help="auto = constant, plus a trend if a significant linear trend is found")
    m = SEASON[freq]
    y_lvl = y_lvl.loc[win[0]:win[1]]
    try:
        y = transform(y_lvl, kind, m)
    except ValueError as e:
        st.error(f"{e}. Pick another transformation.")
        return
    y = y.dropna()
    if len(y) < 30:
        st.error(f"Only {len(y)} observations after the transformation and window - need at least 30.")
        return
    gaps = int(y_lvl.isna().sum())
    inv = np.exp if kind == "log" else (lambda a: np.asarray(a))
    ylabel = {"level": unit or "level", "log": f"ln({unit or 'level'})", "diff": "change", "logdiff": "% growth", "yoy": "% YoY"}[kind]
    ppy = PERIODS_PER_YEAR[freq]

    (t_brief, t_series, t_rel, t_fc, t_bt, t_vol, t_ols, t_diag, t_data, t_guide) = st.tabs(
        ["BRIEFING", "THIS SERIES", "RELATIONSHIPS", "FORECAST (ARIMA)", "BACKTEST", "VOLATILITY (GARCH)", "REGRESSION (OLS)", "DIAGNOSTICS", "DATA", "GUIDE"])

    def jump_on_click(event, frame: pd.DataFrame, ccol: str, scol: str, token: str):
        """Clicking a table row selects that series in the sidebar (once per click)."""
        rows_ = list(getattr(getattr(event, "selection", None), "rows", []) or [])
        if not rows_ or rows_[0] >= len(frame):
            return
        tgt = (frame.iloc[rows_[0]][ccol], frame.iloc[rows_[0]][scol])
        if st.session_state.get(token) != tgt:
            st.session_state[token] = tgt
            if tgt != (cat, sname):
                st.session_state["pending_pick"] = tgt
                st.rerun()

    # ================================================================ BRIEFING
    with t_brief:
        B = briefing_table(data, dsig)
        if B.empty:
            card("No series has enough history (24 observations) for the scan.", "bad")
        else:
            fresh = B[~B["stale"]]
            card(f"<b>{len(B)} series scanned</b>, latest data {data['date'].max():%d %b %Y}. Each is compared only with <b>its own history</b>: "
                 f"rates, ratios and surveys by their level; indices and prices by their year-on-year growth (the level of an index only trends, so it says nothing). "
                 f"<br>In the top or bottom 5% of own history: <b>{int((fresh['pct'] >= 95).sum() + (fresh['pct'] <= 5).sum())}</b> · "
                 f"unusually large latest move: <b>{int((fresh['move_z'].abs() >= 2.5).sum())}</b> · "
                 f"changing direction: <b>{int((fresh['mom'].str.startswith('Turning') & (fresh['mom_z'].abs() >= 1)).sum())}</b> · "
                 f"unusually turbulent (top 10% of own history): <b>{int((fresh['vol_pct'] >= 90).sum())}</b>."
                 + (f" {int(B['stale'].sum())} series have stopped updating and are left out of the lists below." if B["stale"].any() else ""), "info")
            H = headline_lists(B)
            hc = st.columns(3)
            for col_, key_, ttl, none_ in ((hc[0], "ext", "At an extreme", "Nothing in the top / bottom 5% of its history."),
                                           (hc[1], "mv", "Unusual latest move", "No move beyond 2.5x normal."),
                                           (hc[2], "tn", "Changing direction", "No clear turn against the past year's trend.")):
                with col_:
                    card(f'<span class="hd">{ttl}</span><ul>' + ("".join(f"<li>{x}</li>" for x in H[key_]) if H[key_] else f"<li style='color:{MUTED}'>{none_}</li>") + "</ul>",
                         "" if H[key_] else "mute")
            fc1, fc2, fc3 = st.columns([2, 1, 1])
            bcats = fc1.multiselect("Categories", sorted(B["category"].unique()), default=[], placeholder="All categories", key="b_cats")
            bview = fc2.selectbox("Show", ["Everything", "At an extreme (top / bottom 10%)", "Unusual latest move", "Changing direction"], key="b_view")
            bsort = fc3.selectbox("Sort by", ["Most unusual first", "Highest vs history", "Lowest vs history", "Biggest latest move", "Category"], key="b_sort")
            V = B[B["category"].isin(bcats)] if bcats else B
            if bview.startswith("At an"):
                V = V[(V["pct"] >= 90) | (V["pct"] <= 10)]
            elif bview.startswith("Unusual"):
                V = V[V["move_z"].abs() >= 2]
            elif bview.startswith("Changing"):
                V = V[V["mom"].str.startswith("Turning")]
            V = V.assign(_u=np.maximum((V["pct"] - 50).abs() / 50 * 3, V["move_z"].abs().fillna(0)))
            V = {"Most unusual first": V.sort_values("_u", ascending=False), "Highest vs history": V.sort_values("pct", ascending=False),
                 "Lowest vs history": V.sort_values("pct"), "Biggest latest move": V.sort_values("move_z", key=lambda s_: s_.abs(), ascending=False),
                 "Category": V.sort_values(["category", "series"])}[bsort].reset_index(drop=True)
            if len(V):
                fig = go.Figure()
                for i, (c_, g_) in enumerate(V[~V["stale"]].groupby("category")):
                    fig.add_trace(go.Scatter(x=g_["pct"], y=g_["mom_z"], mode="markers", name=c_, text=g_["series"],
                                             customdata=np.column_stack([g_["gval"].map(fnum), g_["glabel"], g_["mom"]]),
                                             marker=dict(size=10, color=SERIES_COLORS[i % 8], line=dict(color=BG, width=1)),
                                             hovertemplate="<b>%{text}</b><br>%{customdata[0]} %{customdata[1]}<br>above %{x:.0f}% of its history · %{customdata[2]}<extra></extra>"))
                fig.add_vline(x=50, line=dict(color=GRID, width=1)); fig.add_hline(y=0, line=dict(color=GRID, width=1))
                yr = float(np.nanmax(np.abs(V["mom_z"].values))) if V["mom_z"].notna().any() else 1.0
                for x_, y_, t_ in ((97, yr, "HIGH & STILL RISING"), (97, -yr, "HIGH BUT FALLING"), (3, yr, "LOW BUT RISING"), (3, -yr, "LOW & STILL FALLING")):
                    fig.add_annotation(x=x_, y=y_ * 1.08, text=t_, showarrow=False, font=dict(color=MUTED, size=10), xanchor="right" if x_ > 50 else "left")
                layout(fig, 450, "Where everything stands (left-right) and which way it is moving (up-down)")
                fig.update_layout(hovermode="closest", margin=dict(t=64), legend=dict(y=1.12))
                fig.update_xaxes(title="share of its own history that was lower than today (%)", range=[-2, 102])
                fig.update_yaxes(title="3-month move (x typical)")
                st.plotly_chart(fig, width="stretch")
                tbl = pd.DataFrame({"Category": V["category"], "Series": V["series"], "As of": V["asof"].dt.strftime("%d %b %y") + np.where(V["stale"], " (stale)", ""),
                                    "Latest": V["latest"], "Read as": V["gval"].map(fnum) + " " + V["glabel"], "Last 3 years": V["spark"], "Above % of history": V["pct"],
                                    "Standing": V["since"].where(V["since"] != "", "-"), "Direction": V["mom"],
                                    "Latest move": V["move"].map(lambda v: f"{v:+.2f}" if np.isfinite(v) else "-") + " " + V["move_unit"],
                                    "x normal": V["move_z"], "Turbulence": V["vol_pct"], "Years": V["years"]})
                ev = st.dataframe(tbl, hide_index=True, width="stretch", height=min(560, 38 + 35 * len(tbl)), on_select="rerun", selection_mode="single-row",
                                  key=f"b_table_{abs(hash((tuple(bcats), bview, bsort))) % 10**8}",
                                  column_config={"Above % of history": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                                                 "Last 3 years": st.column_config.LineChartColumn(width="small", help="The 'Read as' number over the last three years, each row on its own scale."),
                                                 "Latest": st.column_config.NumberColumn(format="localized"),
                                                 "x normal": st.column_config.NumberColumn(format="%+.1f", help="Latest one-period move divided by a normal one-period move (robust). Beyond +/-2.5 is rare."),
                                                 "Turbulence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", help="How large recent moves have been: share of the series' own history that was calmer than now."),
                                                 "Years": st.column_config.NumberColumn(format="%.0f", help="Length of history the comparison uses")})
                st.caption("Tick a row to open that series in THIS SERIES and RELATIONSHIPS. 'High' and 'low' are relative to the series' own past, not a judgement: "
                           "a short history makes any reading look extreme, so check the Years column. Direction for a growth gauge means growth is speeding up or slowing.")
                jump_on_click(ev, V, "category", "series", "b_last_pick")
            else:
                card("Nothing matches this filter.", "mute")

    # ================================================================ DATA
    with t_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(y):,}")
        c2.metric("Frequency", f"{freq} (native {native})")
        c3.metric("Range", f"{y.index.min():%b-%Y} → {y.index.max():%b-%Y}")
        c4.metric("Latest value", f"{y.iloc[-1]:,.3f}")
        st.plotly_chart(line_fig({"modelled series": y}, 300, f"{sname} - {TRANSFORMS[kind]}"), width="stretch")
        if kind != "level":
            st.plotly_chart(line_fig({"original level": y_lvl}, 220, "original series"), width="stretch")
        card(f"Source unit: <b>{unit or 'n/a'}</b>. Transformation chosen: <b>{TRANSFORMS[kind]}</b>. "
             f"{'Interior gaps were interpolated (max 3 periods).' if gaps else 'No gaps after regularising.'} "
             + ("<br><b>Overlapping observations:</b> a YoY series shares 11 of 12 months with its neighbour, so its residuals carry an MA(11) pattern by construction - "
                "robust (HAC) standard errors are essential, or model the index in log differences instead." if kind == "yoy" or "yoy" in sname.lower() or "% yoy" in unit.lower() else ""))
        show(pd.DataFrame([describe(y)]).T.reset_index().rename(columns={"index": "Statistic", 0: "Value"}), {"Value": "{:,.4f}"}, 330)

    diag = cached_diag(y, freq, trend_mode)
    d_auto, D_auto = diag["d"], diag["D"]

    # ================================================================ THIS SERIES
    with t_series:
        hsum = 12 if freq in ("M", "W") else 8 if freq == "Q" else 5 if freq == "A" else 20
        pw = PERIOD_WORD[freq]
        DD0 = D_auto if m > 1 else 0
        tr0 = trend_for(d_auto, DD0, has_trend(y) if d_auto == 0 and DD0 == 0 else False, drift_significant(y, d_auto, DD0, m))
        out0, rec = None, None
        try:
            with st.spinner("Fitting the best model and checking its track record..."):
                yv0 = y.values.astype(float)
                cand0 = arima_search(yv0, int(d_auto), int(DD0), m, tr0, 2, 2, 1 if DD0 else 0, 1 if DD0 else 0, "AICc", None)
                b0 = cand0.iloc[0]
                order0 = (int(b0.p), int(b0.d), int(b0.q))
                seas0 = (int(b0.P), int(b0.D), int(b0.Q), int(b0.m)) if b0.m else (0, 0, 0, 0)
                out0 = run_arima(yv0, order0, seas0, tr0, None, None, hsum)
                rec = track_record(yv0, order0, seas0, tr0, hsum, m)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Automatic model could not be fitted ({type(e).__name__}); use the other tabs.")
        fd0 = future_dates(y.index[-1], hsum, freq)
        last_v = float(inv(np.array([y.iloc[-1]]))[0])
        snap = snapshot_reg(y_lvl, unit, sname, freq)
        zp = diag["za"]["p"] if diag["za"] else 1.0
        arp = diag["arch"][1]
        nout = len(outliers(diag["yd"]))
        phi = None
        if out0 is not None:
            arr = out0["params"][out0["params"]["Parameter"].str.startswith("ar.L")]["Estimate"]
            if len(arr):
                phi = float(arr.sum())
        hl = (np.log(.5) / np.log(phi)) if (d_auto == 0 and phi is not None and 0 < phi < 1) else None

        # ---- the reading, in sentences
        pts = []
        if snap:
            where = f"<b>Where it stands.</b> {fnum(snap['gval'])} {snap['glabel']} as of {snap['asof']:%d %b %Y} - above <b>{snap['pct']:.0f}%</b> of its own {snap['years']:.0f}-year history"
            where += f", the <b>{snap['since']}</b>." if snap["since"] else "."
            where += f" Typical (median) reading: {fnum(snap['median'])}; 8 readings in 10 fell between {fnum(snap['p10'])} and {fnum(snap['p90'])}."
            pts.append(where)
            mv = {"Flat": "Little changed", "Rising": "Rising", "Falling": "Falling", "Turning up": "Turning up", "Turning down": "Turning down"}[snap["mom"]]
            mom_s = f"<b>Direction.</b> {mv} over the last ~3 months" + (f", against the past year's {'fall' if snap['mom'] == 'Turning up' else 'rise'}" if snap["mom"].startswith("Turning") else "")
            if np.isfinite(snap["mom_z"]):
                mom_s += f" ({snap['mom_chg']:+.2f}, about {abs(snap['mom_z']):.1f}x a typical 3-month move)"
            if np.isfinite(snap["move_z"]) and abs(snap["move_z"]) >= 2:
                mom_s += f". The latest {pw}'s move of {snap['move']:+.2f} {snap['move_unit']} was <b>{abs(snap['move_z']):.1f}x a normal one</b>"
            pts.append(mom_s + ".")
            if np.isfinite(snap["vol_pct"]):
                pts.append(f"<b>Turbulence.</b> Recent {pw}-to-{pw} moves have been larger than <b>{snap['vol_pct']:.0f}%</b> of its history"
                           + (" - an unusually stormy stretch, so expect wider outcomes than the ranges below suggest." if snap["vol_pct"] >= 85
                              else " - an unusually calm stretch; calm rarely lasts, and the ranges below already assume average conditions." if snap["vol_pct"] <= 15 else "."))
        if out0 is not None:
            end_v = float(inv(out0["mean"])[-1]); lo_v = float(inv(out0["lo80"])[-1]); hi_v = float(inv(out0["hi80"])[-1])
            move = end_v - last_v
            implied = ""
            if snap and snap["glabel"] == "% YoY" and kind in ("level", "log"):       # say it in the same terms as 'where it stands'
                path = np.concatenate([y_lvl.dropna().values, np.asarray(inv(out0["mean"]), dtype=float)])
                if len(path) > ppy and path[-1 - ppy] > 0:
                    implied = f" That path means <b>{100 * (path[-1] / path[-1 - ppy] - 1):+.1f}% YoY</b> by then, against {snap['gval']:+.1f}% now."
            pts.append(f"<b>Where it is heading.</b> The model's central path is <b>{fnum(end_v)}</b> ({ylabel if kind != 'log' else (unit or 'level')}) by {fd0[-1]:{'%d %b %Y' if freq in ('D', 'W') else '%b %Y'}} from {fnum(last_v)} now - "
                       f"{'essentially unchanged' if abs(move) < 0.005 * max(abs(last_v), 1e-9) else ('a rise of ' if move > 0 else 'a fall of ') + fnum(abs(move))}; "
                       f"8-in-10 range {fnum(lo_v)} to {fnum(hi_v)}.{implied}")
        verdict, vkind = None, "mute"
        if rec:
            sk = rec["skill"]
            if np.isfinite(sk) and sk >= 0.10:
                verdict, vkind = f"it has earned some trust: its misses were <b>{sk * 100:.0f}% smaller</b> than simply assuming no change", "good"
            elif np.isfinite(sk) and sk >= -0.05:
                verdict, vkind = "it has <b>no real edge</b> over simply assuming no change - read the central path as 'about where it is now' and take the range as the message", "mute"
            else:
                verdict, vkind = f"it did <b>worse than assuming no change</b> (misses {abs(sk) * 100:.0f}% larger) - do not lean on the central path", "bad"
            trk = f"<b>Can the forecast be trusted?</b> Replayed at {rec['n']} past dates, {hsum} {pw}s ahead each time, {verdict}."
            if rec["calls"] >= 4:
                trk += f" It called the direction right {rec['hits']} times out of {rec['calls']}."
            trk += f" The 8-in-10 range held the outcome {rec['cover']} times out of {rec['n']}" + (" - too narrow, so widen it in your head." if rec["cover"] / rec["n"] < 0.6 else ".")
            pts.append(trk)
        elif out0 is not None:
            pts.append("<b>Can the forecast be trusted?</b> The history is too short to replay the model at past dates - treat the path as indicative only.")
        if d_auto == 0:
            pts.append("<b>Behaviour.</b> It keeps returning to its average" + (f"; a surprise fades by half in about {hl:.1f} {pw}s, so extremes here have not lasted." if hl else "."))
        else:
            pts.append("<b>Behaviour.</b> It does not return to an average: a shock stays in the level, so 'high' or 'low' against history is not by itself a reason to expect a reversal.")
        flags = []
        if arp < .05:
            flags.append("calm and stormy spells cluster (see VOLATILITY for the current regime)")
        if zp < .05 and diag["za"]:
            flags.append(f"a possible regime shift around {y.index[min(diag['za']['bp'], len(y) - 1)]:%b-%Y}, so older history may be a poor guide")
        if nout:
            flags.append(f"{nout} extreme jump{'s' if nout > 1 else ''} in the sample")
        if flags:
            ftxt = "; ".join(flags)
            pts.append("<b>Handle with care.</b> " + ftxt[0].upper() + ftxt[1:] + ".")
        card(f'<span class="hd">{sname}</span><ul>' + "".join(f"<li>{p}</li>" for p in pts) + "</ul>", vkind if rec else "")

        # ---- standing vs history, and the forecast
        g1, g2 = st.columns(2)
        with g1:
            if snap:
                gs = snap["gauge"].iloc[-min(len(snap["gauge"]), 15 * ppy):]
                fig = go.Figure()
                fig.add_hrect(y0=snap["p10"], y1=snap["p90"], fillcolor="rgba(139,92,246,0.14)", line_width=0)
                fig.add_trace(go.Scatter(x=gs.index, y=gs.values, name=snap["glabel"], line=dict(color=TEXT, width=1.6)))
                fig.add_hline(y=snap["median"], line=dict(color=GOLD, width=1, dash="dot"), annotation_text="typical", annotation_font_color=GOLD)
                fig.add_trace(go.Scatter(x=[gs.index[-1]], y=[gs.values[-1]], mode="markers", marker=dict(color=GOLD, size=11), name="now"))
                layout(fig, 400, f"Where it stands - {snap['glabel']} against its usual range", legend=False)
                fig.update_yaxes(title=snap["glabel"])
                st.plotly_chart(fig, width="stretch")
                st.caption("Shaded band = where 8 readings in 10 have fallen; dotted line = the typical (median) reading; gold dot = now.")
        with g2:
            if out0 is not None:
                bands = {"list": [(out0["lo95"], out0["hi95"], "19-in-20 range", "rgba(139,92,246,0.18)"), (out0["lo80"], out0["hi80"], "8-in-10 range", "rgba(139,92,246,0.34)")]}
                fig = fan_fig(y, fd0, out0["mean"], bands, inv, "Where it is heading", ylabel if kind != "log" else unit, 400)
                if d_auto == 0:
                    fig.add_hline(y=float(inv(np.array([y.mean()]))[0]), line=dict(color=GOLD, width=1, dash="dot"), annotation_text="long-run average")
                st.plotly_chart(fig, width="stretch")
                st.caption("Gold line = central path; bands = likely ranges, assuming volatility stays as it has been on average.")

        if rec:
            st.plotly_chart(track_fig(y, rec, hsum, inv, ylabel if kind != "log" else unit), width="stretch")
            st.caption(f"Each dotted line is a forecast made with only the data available then, {hsum} {pw}s ahead. Diamonds with whiskers = what it expected and its 8-in-10 range; "
                       f"dots = what happened ({rec['cover']} of {rec['n']} landed inside). Outcomes that keep falling on one side of the diamonds mean the model is biased.")

        # ---- two questions a forecast can actually answer
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("##### What are the odds it crosses a level?")
            if out0 is not None:
                dflt = float(inv(np.array([y.median()]))[0]) if d_auto == 0 else last_v
                o1, o2 = st.columns(2)
                thr = o1.number_input(f"Level ({ylabel if kind != 'log' else (unit or 'level')})", value=float(f"{dflt:.4g}"), format="%.4f", key=f"thr_{sname}_{kind}")
                hq = o2.slider(f"Within how many {pw}s", 1, hsum, hsum, key="odds_h")
                thr_t = np.log(thr) if (kind == "log" and thr > 0) else thr
                if kind == "log" and thr <= 0:
                    card("Enter a level above zero.", "bad")
                else:
                    pa = prob_above(out0["mean"], out0["lo95"], out0["hi95"], thr_t) * 100
                    p_ = float(pa[hq - 1])
                    chips([(f"Above {fnum(thr)} on {fd0[hq - 1]:%d %b %Y}", f"{p_:.0f}%", "info", f"Below: {100 - p_:.0f}%. Now at {fnum(last_v)}.")], 1)
                    fig = go.Figure(go.Scatter(x=fd0, y=pa, mode="lines+markers", line=dict(color=GOLD, width=2.4)))
                    fig.add_hline(y=50, line=dict(color=MUTED, width=1, dash="dot"))
                    layout(fig, 230, f"Chance of being above {fnum(thr)} at each date (%)", legend=False)
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Odds of the reading on that date, not of touching the level at any point before it. They come from the model's range, so they are only as good as its track record above"
                               + (" - which says the range has been too narrow, so push these odds toward 50%." if rec and rec["cover"] / rec["n"] < 0.6 else "."))
        with q2:
            st.markdown("##### When it stood here before, what came next?")
            if snap:
                co = conditional_outcomes(snap["gauge"], y_lvl, hsum, pct_change=not snap["rate"] and (y_lvl > 0).all())
                pc0 = (not snap["rate"]) and bool((y_lvl > 0).all())
                ou, ous = ("% change", "%") if pc0 else ("change in points", "pts")
                if co is None:
                    card("Not enough history to split into bands.", "mute")
                else:
                    r_ = co["table"].iloc[co["current"]]
                    kindc = "bad" if co["episodes"] < 8 else ""
                    card(f"From readings between <b>{r_['Band']}</b> {snap['glabel']} (it is {fnum(co['now'])} now), the series over the next {hsum} {pw}s "
                         f"<b>rose in {r_['Rose %']:.0f}%</b> of {int(r_['Cases'])} cases; median change <b>{r_['Median']:+.2f} {ous}</b> "
                         f"(middle half {r_['p25']:+.2f} to {r_['p75']:+.2f}). From any starting point: rose in {co['all_rose']:.0f}%, median {co['all_median']:+.2f} {ous}."
                         + (f"<br><b>Thin evidence:</b> the windows overlap, so this rests on only about {co['episodes']:.0f} separate episodes." if co["episodes"] < 8 else ""), kindc)
                    eps, today_ = episode_paths(snap["gauge"], y_lvl, hsum, float(r_["lo"]), float(r_["hi"]), pc0)
                    if len(eps) >= 2:
                        st.plotly_chart(episode_fig(eps, today_, hsum, ou, pw, 320), width="stretch")
                        st.caption(f"Purple = each separate past episode (at least {hsum} {pw}s apart), all set to zero on the day the series stood in today's band; gold = their typical path afterwards; "
                                   "white = the run-up this time. If the white run-up looks unlike the purple ones, the past episodes are a weaker guide.")
                        with st.expander("Outcomes from every starting band"):
                            st.plotly_chart(outcome_fig(co, f"Median {ou} over the next {hsum} {pw}s, by starting point", ou, 255), width="stretch")
                    else:
                        st.plotly_chart(outcome_fig(co, f"Median {ou} over the next {hsum} {pw}s, by starting point", ou, 255), width="stretch")
                        st.caption(f"Bars = median outcome, whiskers = middle half of outcomes. About {co['episodes']:.0f} non-overlapping episodes in total; a pattern across the bars matters more than any one bar.")

        with st.expander("Model behind this page"):
            items = [("Behaviour", "Returns to average" if d_auto == 0 else "Wanders", "ok" if d_auto == 0 else "warn",
                      (f"Shock half-life about {hl:.1f} {pw}s." if hl else "Stable around a fixed level.") if d_auto == 0 else "Shocks stay in the level; modelled in changes.")]
            if m > 1:
                ss = diag["seas_strength"]
                items.append(("Seasonality", "Strong" if ss > .64 else "Weak", "info" if ss > .64 else "ok", f"Seasonal strength {ss:.2f} (above 0.64 is treated as seasonal)."))
            items.append(("Volatility clusters", "Yes" if arp < .05 else "No", "warn" if arp < .05 else "ok", "See the VOLATILITY tab." if arp < .05 else "Variance looks steady."))
            items.append(("Structural break", "Possible" if zp < .05 else "None obvious", "warn" if zp < .05 else "ok", "Try the outlier dummy on the FORECAST tab." if zp < .05 else "No dominant break found."))
            if out0 is not None:
                f0 = int((out0["checks"]["Result"] == "FAIL").sum()); n0 = len(out0["checks"])
                items.append(("Model health", f"{n0 - f0}/{n0} checks pass", "ok" if f0 == 0 else "warn" if f0 <= 2 else "bad", "Residual tests: ok means the model captured the pattern."))
                items.append(("Chosen model", f"ARIMA{order0}" + (f"x{seas0}" if seas0[3] else ""), "info", f"trend term '{tr0}'; picked by AICc on {TRANSFORMS[kind].lower()}."))
            if rec:
                items.append(("Track record", f"{rec['skill'] * 100:+.0f}% vs no-change" if np.isfinite(rec["skill"]) else "n/a", "ok" if rec["skill"] >= .1 else "warn" if rec["skill"] >= -.05 else "bad",
                              f"{rec['n']} past dates, {rec['step']} {pw}s apart, {hsum} ahead. Positive = smaller misses than assuming no change."))
            chips(items, 4)

    # ================================================================ RELATIONSHIPS
    with t_rel:
        nser = int(data[["category", "series"]].drop_duplicates().shape[0])
        if nser < 2:
            card("Upload more than one series to compare them.", "mute")
        else:
            r1, r2 = st.columns([1, 2])
            span_lbl = r1.selectbox("Compare changes over", ["1 period", "3 months", "12 months"], index=1 if freq in ("D", "W", "M") else 0, key="rel_span",
                                    help="Longer spans show slow co-movement but overlap, so fewer independent observations stand behind each number.")
            months = {"1 period": 0, "3 months": 3, "12 months": 12}[span_lbl]
            span_txt = {"1 period": "one-period", "3 months": "3-month", "12 months": "12-month"}[span_lbl]
            rcats = r2.multiselect("Compare against", sorted(data["category"].unique()), default=[], placeholder="All categories", key="rel_cats")
            T = leadlag_scan(data, dsig, cat, sname, freq, months, (pd.Timestamp(win[0]), pd.Timestamp(win[1])))
            if len(T) and rcats:
                T = T[T["Category"].isin(rcats)].reset_index(drop=True)
            if T.empty:
                card("No other series overlaps this one for long enough to compare.", "mute")
            else:
                good = T[T["Above noise"] & T["Holds in both halves"]]
                lead, foll, tog = good[good["Lag"] > 0], good[good["Lag"] < 0], good[good["Lag"] == 0]

                def _names(df_, n_=3):
                    return ", ".join(f"<b>{r.Series}</b> ({r.Timing.split(' by ')[-1] if ' by ' in r.Timing else 'same time'}, link {r.Link:+.2f})" for r in df_.head(n_).itertuples())
                lines = [f"<b>Moves first</b> (possible early signals for {sname}): {_names(lead)}." if len(lead) else f"<b>Moves first:</b> nothing reliably leads {sname} in this data.",
                         f"<b>Moves after</b> ({sname} is the early signal): {_names(foll)}." if len(foll) else "<b>Moves after:</b> nothing reliably follows it.",
                         f"<b>Moves together:</b> {_names(tog)}." if len(tog) else "<b>Moves together:</b> no dependable same-time partner."]
                card(f'<span class="hd">{sname} against {len(T)} other series, {span_txt} changes</span><ul>' + "".join(f"<li>{x}</li>" for x in lines) + "</ul>"
                     f"<span style='color:{MUTED}'>{len(good)} of {len(T)} links clear the noise level (already raised for the search across timings) and hold in both halves of the sample. "
                     f"A few unrelated series in a hundred still pass by luck, so keep only links with an economic reason behind them.</span>", "info")
                tv = T.copy()
                ev = st.dataframe(tv, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(tv)), on_select="rerun", selection_mode="single-row",
                                  key=f"rel_table_{abs(hash((cat, sname, freq, months, tuple(rcats)))) % 10**8}",
                                  column_config={"Lag": None, "Link": st.column_config.NumberColumn(format="%+.2f", help="Correlation of changes at the best timing. +1 = move in step, -1 = opposite, 0 = unrelated."),
                                                 "Same-time link": st.column_config.NumberColumn(format="%+.2f"),
                                                 "Noise level": st.column_config.NumberColumn(format="%.2f", help="A link smaller than this is indistinguishable from chance for this much data, allowing for the search across timings."),
                                                 "Pairs": st.column_config.NumberColumn(help="Observations both series share")})
                st.caption("'Leads' = the other series' changes line up best with LATER changes in the selected series. Tick a row to study that pair below.")
                rows_ = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
                pr = T.iloc[rows_[0]] if rows_ and rows_[0] < len(T) else T.iloc[0]
                oc, on_ = pr["Category"], pr["Series"]
                osub = data[(data["category"] == oc) & (data["series"] == on_)]
                x_nat = osub.groupby("date")["value"].last().dropna().sort_index()
                x_unit = str(osub["unit"].iloc[0])
                dfp, cf, kk_, ya_, xa_ = pair_changes(s_native.loc[win[0]:win[1]], unit, sname, x_nat, x_unit, on_, freq, months)
                LR = lag_reading(dfp, cf, kk_)
                st.markdown(f"##### {on_}  vs  {sname}")
                if LR:
                    b_, w_ = LR["best"], PERIOD_WORD[cf]
                    timing = "at the same time as" if b_ == 0 else f"{abs(b_)} {w_}{'s' if abs(b_) > 1 else ''} {'before' if b_ > 0 else 'after'}"
                    strength = "strong" if abs(LR["corr"]) >= .6 else "moderate" if abs(LR["corr"]) >= .35 else "weak"
                    if abs(LR["corr"]) < LR["thr"]:
                        card(f"<b>No dependable link</b> between <b>{on_}</b> and <b>{sname}</b>. The best timing found ({on_} moving {timing} it, link {LR['corr']:+.2f}) "
                             f"is within the noise level of {LR['thr']:.2f} - what unrelated series produce when {2 * LMAX[cf] + 1} timings are searched over {LR['n']} shared observations.", "bad")
                    else:
                        card(f"<b>{on_}</b> tends to move <b>{timing}</b> <b>{sname}</b>, {'in the same direction' if LR['corr'] > 0 else 'in the <b>opposite</b> direction'}. "
                             f"The link is <b>{strength}</b> ({LR['corr']:+.2f}; anything under {LR['thr']:.2f} is noise for {LR['n']} shared observations) and "
                             + ("<b>held in both halves</b> of the sample." if LR["stable"] else "<b>did not hold in both halves</b> of the sample - it may be an accident of one period.")
                             + " Timing alone is not cause: a third force can drive both.", "good" if LR["stable"] else "")
                    p1, p2 = st.columns(2)
                    with p1:
                        lags_ = sorted(LR["cc"])
                        fig = go.Figure(go.Bar(x=lags_, y=[LR["cc"][l] for l in lags_],
                                               marker_color=[GOLD if l == b_ else (PURPLE if abs(LR["cc"][l] or 0) >= LR["thr"] else MUTED) for l in lags_]))
                        for s_ in (LR["thr"], -LR["thr"]):
                            fig.add_hline(y=s_, line=dict(color=GOLD, width=1, dash="dot"))
                        fig.add_vline(x=0, line=dict(color=GRID, width=1))
                        layout(fig, 300, "Link at each timing (gold = strongest)", legend=False)
                        fig.update_xaxes(title=f"<- {on_[:28]} moves after   |   {w_}s   |   moves first ->")
                        st.plotly_chart(fig, width="stretch")
                    with p2:
                        zz = pd.DataFrame({sname: robust_z(dfp["y"].dropna()), on_ + (f" (shifted {b_} {w_}s)" if b_ else ""): robust_z(dfp["x"].dropna()).shift(b_) if cf != "D" else robust_z(dfp["x"]).shift(b_)})
                        zz = zz.iloc[-min(len(zz), 12 * PERIODS_PER_YEAR[cf]):]
                        figl = line_fig({c_: zz[c_].dropna() for c_ in zz.columns}, 300, f"{span_txt.capitalize()} changes, each in its own normal-move units")
                        st.plotly_chart(figl, width="stretch")
                    wroll = {"D": 250, "W": 104, "M": 36, "Q": 16, "A": 8}[cf]
                    rc = dfp["y"].rolling(wroll, min_periods=int(wroll * .7)).corr(dfp["x"].shift(b_)).dropna()
                    if len(rc) > 5:
                        fig = line_fig({"link": rc}, 220, f"Has the link been steady? (rolling {wroll}-{w_} window at the best timing)")
                        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
                        st.plotly_chart(fig, width="stretch")
                else:
                    card("These two do not share enough observations to measure the timing.", "mute")
                st.markdown(f"##### When {on_} stood at different levels, what did {sname} do next?")
                gx, gxl, _gr = gauge_of(xa_.dropna(), x_unit, on_, cf)
                y_rate = is_rate_like(ya_.dropna(), unit, sname)
                kh = st.number_input(f"Look ahead ({PERIOD_WORD[cf]}s)", 1, 10 * PERIODS_PER_YEAR[cf], SHORT_LONG[cf][1], key=f"rel_h_{cf}")
                pc_ = (not y_rate) and bool((ya_.dropna() > 0).all())
                co = conditional_outcomes(gx, ya_.dropna(), int(kh), pct_change=pc_)
                ou, ous = ("% change", "%") if pc_ else ("change in points", "pts")
                if co is None:
                    card("Not enough shared history for this look-ahead.", "mute")
                else:
                    t_ = co["table"]; r_ = t_.iloc[co["current"]]
                    spread = float(t_["Median"].iloc[-1] - t_["Median"].iloc[0])
                    mono = stats.spearmanr(np.arange(len(t_)), t_["Median"]).correlation if len(t_) >= 3 else np.nan
                    pattern = ("there is a <b>clear pattern</b>: the higher the signal stood, the " + ("higher" if spread > 0 else "lower") + " what followed") if (np.isfinite(mono) and abs(mono) >= .9) else "there is <b>no orderly pattern</b>, so the signal's level has not told much about what followed"
                    card(f"<b>{on_}</b> is at {fnum(co['now'])} {gxl} now (band {r_['Band']}). From that band, <b>{sname}</b> over the next {int(kh)} {PERIOD_WORD[cf]}s "
                         f"<b>rose in {r_['Rose %']:.0f}%</b> of {int(r_['Cases'])} cases, median <b>{r_['Median']:+.2f} {ous}</b> (middle half {r_['p25']:+.2f} to {r_['p75']:+.2f}); "
                         f"from any starting point it rose in {co['all_rose']:.0f}%, median {co['all_median']:+.2f} {ous}. Across the bands {pattern}."
                         + (f"<br><b>Thin evidence:</b> only about {co['episodes']:.0f} separate episodes stand behind this." if co["episodes"] < 8 else ""),
                         "bad" if co["episodes"] < 8 else "")
                    st.plotly_chart(outcome_fig(co, f"{sname}: median {ou} over the next {int(kh)} {PERIOD_WORD[cf]}s, by where {on_[:40]} stood", ou, 300), width="stretch")
                st.caption("To test a link formally with other drivers held fixed, use REGRESSION (OLS).")

    # ================================================================ DIAGNOSTICS
    with t_diag:
        verdict = f"integrated of order <b>d = {d_auto}</b>" + (f", seasonally integrated (strength {diag['seas_strength']:.2f} &gt; 0.64): <b>D = 1</b>" if D_auto else
                                                                (f"; seasonal strength {diag['seas_strength']:.2f} - no seasonal differencing" if m > 1 else ""))
        card(f"<b>In plain words:</b> {'this series is stable around an average - it can be modelled as it is.' if d_auto == 0 else 'this series wanders (its level has no fixed home), so it must be converted to changes before modelling.'} "
             f"<br><span style='color:{MUTED}'>Technically: {verdict}.</span>", "good" if d_auto == 0 else "")
        st.markdown("##### 1 · Does it keep returning to an average? (differencing ladder)")
        hist = diag["history"]
        titles = []
        for h_ in hist:
            nmx = "Original" if h_["Differences"] == 0 else f"Change #{h_['Differences']}"
            titles.append(f"{nmx}: {'stable ✓' if h_['Stationary'] else 'wanders ✗'} ({h_['votes']}/4 tests)")
        fig = make_subplots(rows=1, cols=len(hist), subplot_titles=titles, horizontal_spacing=0.06)
        for i, h_ in enumerate(hist, 1):
            sx = y
            for _ in range(h_["Differences"]):
                sx = sx.diff()
            sx = sx.dropna()
            fig.add_trace(go.Scatter(x=sx.index, y=sx.values, line=dict(color=GREEN if h_["Stationary"] else RED, width=1.3), showlegend=False), row=1, col=i)
            if h_["Stationary"]:
                fig.add_hline(y=float(sx.mean()), line=dict(color=GOLD, width=1, dash="dot"), row=1, col=i)
        layout(fig, 300, None, legend=False)
        fig.update_layout(margin=dict(t=60))
        st.plotly_chart(fig, width="stretch")
        st.caption("Green + dotted gold average = the series hovers around a fixed level (modelling can start there). Red = it drifts away; the next panel shows it after taking changes.")

        st.markdown("##### 2 · What the four tests say")
        ur = diag["unitroot"]
        fig = go.Figure(go.Bar(y=ur["Test"], x=ur["p-value"], orientation="h", marker_color=[GREEN if r == "stationary" else RED for r in ur["Reads as"]],
                               text=[f"p = {p:.3f} → {r}" for p, r in zip(ur["p-value"], ur["Reads as"])], textposition="outside", cliponaxis=False))
        fig.add_vline(x=.05, line=dict(color=GOLD, width=1.5, dash="dash"), annotation_text="5% line", annotation_position="top")
        layout(fig, 300, None, legend=False)
        fig.update_xaxes(range=[0, 1.25], title="p-value"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        st.caption("Green = this test calls the series stable, red = wandering. ADF / PP / DF-GLS say 'stable' when their bar is SHORT (p below the gold line); KPSS says 'stable' when its bar is LONG (p above it). The final call is the vote.")
        with st.expander("Full test table"):
            show(ur, {"Statistic": "{:.3f}", "p-value": "{:.4f}"})
            show(pd.DataFrame(hist))

        st.markdown("##### 3 · Memory and volatility")
        c1, c2 = st.columns(2)
        lb, lbp = diag["lb"]; ar_s, ar_p = diag["arch"]
        with c1:
            chips([("Memory (autocorrelation)", "Present" if lbp < .05 else "Weak", "info" if lbp < .05 else "ok",
                    "Past values help predict the next: AR/MA terms are useful." if lbp < .05 else "Little carry-over from past values."),
                   ("Volatility clustering", "Yes" if ar_p < .05 else "No", "warn" if ar_p < .05 else "ok",
                    "Big moves cluster: use robust errors / GARCH." if ar_p < .05 else "Size of moves is fairly constant.")], 2)
        with c2:
            if diag["za"]:
                za = diag["za"]; bp = y.index[min(za["bp"], len(y) - 1)]
                out = outliers(diag["yd"])
                chips([("Possible break", f"{bp:%b-%Y}" if za["p"] < .05 else "None", "warn" if za["p"] < .05 else "ok",
                        "Zivot-Andrews: the series may have changed regime here." if za["p"] < .05 else "No dominant regime change."),
                       ("Outliers", str(len(out)), "warn" if len(out) else "ok",
                        (", ".join(f"{i:%b-%y}" for i in out.head(3).index) + ". Use a dummy window.") if len(out) else "No extreme jumps.")], 2)
        yd = diag["yd"]
        yd_name = "changes in the series" if d_auto else "the series"
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(acf_fig(yd.values, 36 if m > 1 else 30, f"ACF of {yd_name}"), width="stretch")
        with cc2:
            st.plotly_chart(acf_fig(yd.values, 36 if m > 1 else 30, f"PACF of {yd_name}", partial=True), width="stretch")
        st.caption("Each bar = how strongly today is linked to N periods ago. Purple bars poke outside the dotted band = real links. PACF suggests how many AR terms, ACF how many MA terms (the search on the FORECAST tab decides for you).")
        if m in (4, 12):
            lab = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if m == 12 else ["Q1", "Q2", "Q3", "Q4"]
            keyv = yd.index.month if m == 12 else yd.index.quarter
            prof = yd.groupby(keyv).mean()
            fig = go.Figure(go.Bar(x=[lab[i - 1] for i in prof.index], y=prof.values, marker_color=[GREEN if v >= 0 else RED for v in prof.values]))
            fig.add_hline(y=0, line=dict(color=MUTED, width=1))
            st.plotly_chart(layout(fig, 250, f"Seasonal pattern - average of {yd_name} by {'month' if m == 12 else 'quarter'} (strength {diag['seas_strength']:.2f})", legend=False), width="stretch")
        if m > 1:
            try:
                f = STL(y.reset_index(drop=True), period=m, robust=True).fit()
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                for i, (nm, v, c) in enumerate((("trend", f.trend, GOLD), ("seasonal", f.seasonal, PURPLE), ("remainder", f.resid, CYAN)), 1):
                    fig.add_trace(go.Scatter(x=y.index, y=v, name=nm, line=dict(color=c, width=1.4)), row=i, col=1)
                with st.expander("Trend / seasonal / remainder split (STL)"):
                    st.plotly_chart(layout(fig, 380, f"STL decomposition (period {m})"), width="stretch")
            except Exception:  # noqa: BLE001
                pass
        rs = y.rolling(max(12, ppy // 2 if ppy > 12 else 12)).std()
        st.plotly_chart(line_fig({"rolling std dev": rs.dropna()}, 220, "Is the size of moves steady? (rolling volatility)"), width="stretch")

    # ================================================================ ARIMA
    with t_fc:
        st.markdown("##### AR / MA / ARMA / ARIMA / SARIMA with robust inference")
        fam = st.selectbox("Model family", ["Auto - search by information criterion", "AR(p)", "MA(q)", "ARMA(p,q)", "ARIMA(p,d,q)", "SARIMA(p,d,q)(P,D,Q)m"])
        c = st.columns(6)
        dd = c[0].number_input("d", 0, 2, d_auto) if fam in ("ARIMA(p,d,q)", "SARIMA(p,d,q)(P,D,Q)m") else (d_auto if fam.startswith("Auto") else 0)
        DD = c[1].number_input("D", 0, 1, D_auto) if fam.startswith("SARIMA") else (D_auto if (fam.startswith("Auto") and m > 1) else 0)
        p_ = c[2].number_input("p (max)" if fam.startswith("Auto") else "p", 0, 8, 2 if fam.startswith("Auto") else 1) if not fam.startswith("MA") else 0
        q_ = c[3].number_input("q (max)" if fam.startswith("Auto") else "q", 0, 8, 2 if fam.startswith("Auto") else 1) if not fam.startswith("AR(") else 0
        P_ = c[4].number_input("P (max)" if fam.startswith("Auto") else "P", 0, 3, 1 if fam.startswith("Auto") else 0) if (fam.startswith("SARIMA") or (fam.startswith("Auto") and m > 1)) else 0
        Q_ = c[5].number_input("Q (max)" if fam.startswith("Auto") else "Q", 0, 3, 1 if fam.startswith("Auto") else 1) if (fam.startswith("SARIMA") or (fam.startswith("Auto") and m > 1)) else 0
        if fam in ("AR(p)", "MA(q)", "ARMA(p,q)") and d_auto > 0:
            card(f"The tests say this series is I({d_auto}); a pure {fam} on it will be mis-specified (spurious persistence). Switch to ARIMA, or change the transformation to a difference / growth rate.", "bad")
        c1, c2, c3, c4 = st.columns(4)
        h = c1.number_input("Forecast horizon", 1, 120, 12 if freq in ("M", "W") else 8 if freq == "Q" else 5 if freq == "A" else 20)
        ic = c2.selectbox("Selection criterion", ["AICc", "BIC", "AIC"], disabled=not fam.startswith("Auto"))
        use_dummy = c3.checkbox("Outlier dummy window")
        trend_override = c4.selectbox("Deterministic term", ["auto", "n", "c", "t", "ct"])
        dstart, dend = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30")
        if use_dummy:
            dc1, dc2 = st.columns(2)
            dstart = pd.Timestamp(dc1.date_input("Dummy from", dstart.date()))
            dend = pd.Timestamp(dc2.date_input("Dummy to", dend.date()))
        seasonal_on = m > 1 and (fam.startswith("SARIMA") or (fam.startswith("Auto") and (DD > 0 or P_ > 0 or Q_ > 0)))
        tstat = has_trend(y) if dd == 0 and DD == 0 else False
        drift = drift_significant(y, dd, DD if seasonal_on else 0, m)
        auto_trend = trend_for(dd, DD if seasonal_on else 0, tstat, drift)
        trend = auto_trend if trend_override == "auto" else trend_override
        kk = dd + (DD if seasonal_on else 0)
        valid = {0: ("n", "c", "t", "ct"), 1: ("n", "t"), 2: ("n",)}[min(kk, 2)] if kk else ("n", "c", "t", "ct")
        if kk and trend not in valid:
            st.warning(f"Trend '{trend}' is not identified with d+D = {kk}; using '{auto_trend}'.")
            trend = auto_trend
        why_t = {"c": "stationary in levels → constant (mean)", "ct": "stationary around a significant linear trend → constant + trend",
                 "t": "one difference with a significant drift (HAC t-test on mean of Δ) → drift term", "n": "differenced without significant drift → no deterministic term"}[auto_trend]
        card(f"<b>Specification logic.</b> d = {dd}{f', D = {DD}, m = {m}' if seasonal_on else ''} from the unit-root vote; deterministic term <b>'{trend}'</b> ({why_t if trend_override == 'auto' else 'manual override'}). "
             f"Standard errors are <b>robust (sandwich)</b>; OPG SEs are shown alongside for comparison.")
        ex_all = dummy_series(y.index, dstart, dend) if use_dummy else None
        ex_fut = np.zeros(int(h)) if use_dummy else None
        if use_dummy and ex_all.sum() == 0:
            st.warning("The dummy window does not overlap the sample; it will be ignored.")
            ex_all = ex_fut = None
        if st.button("Run model", type="primary", key="run_arima"):
            try:
                yv = y.values.astype(float)
                with st.spinner("Fitting..."):
                    if fam.startswith("Auto"):
                        cand = arima_search(yv, int(dd), int(DD) if seasonal_on else 0, m, trend, int(p_), int(q_), int(P_) if seasonal_on else 0,
                                            int(Q_) if seasonal_on else 0, ic, ex_all)
                        b = cand.iloc[0]
                        order, seas = (int(b.p), int(b.d), int(b.q)), ((int(b.P), int(b.D), int(b.Q), int(b.m)) if b.m else (0, 0, 0, 0))
                    else:
                        cand = None
                        order = (int(p_), int(dd), int(q_))
                        seas = (int(P_), int(DD), int(Q_), m) if fam.startswith("SARIMA") else (0, 0, 0, 0)
                    out = run_arima(yv, order, seas, trend, ex_all, ex_fut, int(h))
                st.session_state["arima"] = dict(out=out, order=order, seas=seas, trend=trend, cand=cand, key=(sname, freq, kind, win), h=int(h), fam=fam)
            except Exception as e:  # noqa: BLE001
                st.error(f"Model failed: {type(e).__name__}: {e}")
        A = st.session_state.get("arima")
        if A and A["key"] == (sname, freq, kind, win):
            out, order, seas = A["out"], A["order"], A["seas"]
            nm = f"ARIMA{order}" + (f"x{seas}" if seas[3] else "")
            fails = int((out["checks"]["Result"] == "FAIL").sum()); nchk = len(out["checks"])
            chips([("Model", nm, "info", f"deterministic term '{A['trend']}'; " + ("picked by search" if A["cand"] is not None else "set by you")),
                   ("Fit score (AICc)", f"{out['aicc']:,.1f}", "info", "Lower is better. Only comparable between models with the same differencing."),
                   ("Residual checks", f"{nchk - fails}/{nchk} pass", "ok" if fails == 0 else "warn" if fails <= 2 else "bad",
                    "All clean: nothing systematic left." if fails == 0 else "Some structure is left: see the red boxes below.")], 3)
            roots = np.concatenate([out["roots"]["ar"], out["roots"]["ma"]])
            if len(roots) and roots.min() < 1.02:
                card("A characteristic root is within 2% of the unit circle - the model is close to non-stationary / non-invertible; consider more differencing or fewer terms.", "bad")
            fd = future_dates(y.index[-1], A["h"], freq)
            bands = {"list": [(out["lo95"], out["hi95"], "95% range", "rgba(139,92,246,0.18)"), (out["lo80"], out["hi80"], "80% range", "rgba(139,92,246,0.34)")]}
            fig = fan_fig(y, fd, out["mean"], bands, inv, f"Forecast - {nm}", ylabel if kind != "log" else unit, 380)
            if order[1] == 0 and seas[1] == 0:
                fig.add_hline(y=float(inv(np.array([y.mean()]))[0]), line=dict(color=GOLD, width=1, dash="dot"), annotation_text="average")
            st.plotly_chart(fig, width="stretch")
            if kind == "log":
                st.caption("Back-transformed with exp(): these are medians; the mean is slightly higher.")
            st.markdown("##### What the model learned")
            pr = out["params"]; pr = pr[pr["Parameter"] != "sigma2"].copy()
            if len(pr):
                pr["lo"] = pr["Estimate"] - 1.96 * pr["Robust SE"]; pr["hi"] = pr["Estimate"] + 1.96 * pr["Robust SE"]
                st.plotly_chart(forest(pr["Parameter"], pr["Estimate"], pr["lo"], pr["hi"], pr["p (robust)"], "Estimates with 95% robust intervals (green = statistically real, grey = indistinguishable from zero)"), width="stretch")
            ar = pr[pr["Parameter"].str.startswith("ar.")]["Estimate"]; ma = pr[pr["Parameter"].str.startswith("ma.")]["Estimate"]
            notes = []
            if len(ar):
                ps = float(ar.sum())
                notes.append(f"<b>Persistence {ps:.2f}</b> - " + ("very sticky: shocks take long to fade." if abs(ps) > .9 else "moderate: shocks fade over a few periods." if abs(ps) > .5 else "low: values quickly forget the past."))
            if len(ma):
                notes.append("<b>MA terms</b> - recent surprises still echo into the next forecasts.")
            if order[1] or seas[1]:
                notes.append("The model works on <b>changes</b>, so forecasts of the level accumulate them (hence the widening cone).")
            else:
                notes.append("The model works on the <b>level</b>, so forecasts converge toward the average over time.")
            card("<br>".join(notes))
            st.markdown("##### Health check - is anything left unexplained?")
            check_chips(out["checks"], 3)
            if (out["checks"]["Check"].str.contains("squared|ARCH") & (out["checks"]["Result"] == "FAIL")).any():
                card("<b>Volatility is not constant.</b> Coefficients and robust errors remain valid, but the forecast bands above assume constant variance. See the VOLATILITY tab for volatility-aware ranges.", "bad")
            burn = order[1] + (seas[1] * seas[3] if seas[3] else 0)
            r = out["resid"][burn:]
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.plotly_chart(layout(go.Figure(go.Scatter(x=y.index[burn:], y=r, line=dict(color=CYAN, width=1))), 230, "Residuals (should look like noise)", legend=False), width="stretch")
            with cc2:
                st.plotly_chart(acf_fig(r, 24, "Residual ACF (no purple = good)"), width="stretch")
            with cc3:
                fig = go.Figure(go.Histogram(x=r, nbinsx=40, marker_color=PURPLE, histnorm="probability density"))
                xs = np.linspace(r.min(), r.max(), 200)
                fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), r.std()), line=dict(color=GOLD)))
                st.plotly_chart(layout(fig, 230, "Residuals vs bell curve", legend=False), width="stretch")
            with st.expander("Tables: search ranking, parameters, residual tests"):
                if A["cand"] is not None:
                    st.caption("Top specifications by the chosen criterion")
                    show(A["cand"].head(12), {"AIC": "{:.1f}", "AICc": "{:.1f}", "BIC": "{:.1f}"})
                show(out["params"], {"Estimate": "{:.4f}", "Robust SE": "{:.4f}", "OPG SE": "{:.4f}", "z (robust)": "{:.2f}", "p (robust)": "{:.4f}"})
                show(out["checks"], {"Statistic": "{:.3f}", "p-value": "{:.4f}"})

    # ================================================================ BACKTEST
    with t_bt:
        st.markdown("##### Out-of-sample comparison against benchmarks")
        A = st.session_state.get("arima")
        have = A and A["key"] == (sname, freq, kind, win)
        if not have:
            card("Run a model on the ARIMA tab first; its specification is then backtested against naive, drift, seasonal-naive and ETS. "
                 "(Without it the benchmarks still run.)")
        c1, c2, c3 = st.columns(3)
        hb = c1.number_input("Horizon", 1, 60, 12 if freq in ("M", "W") else 4 if freq == "Q" else 3 if freq == "A" else 10, key="bt_h")
        nor = c2.number_input("Number of forecast origins", 6, 120, 24)
        stp = c3.number_input("Step between origins", 1, 24, 1)
        if st.button("Run backtest", type="primary"):
            try:
                with st.spinner("Re-fitting every model at every origin..."):
                    spec = (A["order"], A["seas"], A["trend"]) if have else None
                    st.session_state["bt"] = dict(res=run_backtest(y.values.astype(float), m, int(hb), int(nor), int(stp), spec, True),
                                                  key=(sname, freq, kind, win), spec=spec)
            except Exception as e:  # noqa: BLE001
                st.error(f"Backtest failed: {type(e).__name__}: {e}")
        B = st.session_state.get("bt")
        if B and B["key"] == (sname, freq, kind, win):
            R = B["res"]
            tb = R["table"]
            avg = tb.groupby("Model")[["RMSE", "MAE", "MASE"]].mean().sort_values("RMSE")
            best = avg.index[0]
            bench = avg.drop(index="ARIMA", errors="ignore")
            if "ARIMA" in avg.index and len(bench):
                gain = (1 - avg.loc["ARIMA", "RMSE"] / bench["RMSE"].min()) * 100
                card(f"<b>Result:</b> over {R['origins']} out-of-sample tests the most accurate model was <b>{best}</b>. "
                     f"Your ARIMA is <b>{abs(gain):.0f}% {'more' if gain > 0 else 'less'} accurate</b> than the best simple benchmark ({bench['RMSE'].idxmin()}). "
                     + ("A complicated model that does not beat 'tomorrow = today' is not adding value." if gain <= 0 else "That is the value the model adds."), "good" if gain > 0 else "bad")
            else:
                card(f"<b>Result:</b> the most accurate benchmark was <b>{best}</b>.")
            fig = go.Figure(go.Bar(y=avg.index, x=avg["RMSE"], orientation="h",
                                   marker_color=[GOLD if i == "ARIMA" else (GREEN if i == best else MUTED) for i in avg.index],
                                   text=[f"{v:.3f}" for v in avg["RMSE"]], textposition="outside", cliponaxis=False))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(layout(fig, 60 + 46 * len(avg), "Average forecast error (RMSE) - shorter bar = better. Gold = your ARIMA, green = winner", legend=False), width="stretch")
            fig = go.Figure()
            for i, col in enumerate(R["by_h"].columns):
                fig.add_trace(go.Scatter(x=R["by_h"].index, y=R["by_h"][col], name=col, mode="lines+markers",
                                         line=dict(color=GOLD if col == "ARIMA" else SERIES_COLORS[i % 8], width=3 if col == "ARIMA" else 1.5)))
            layout(fig, 320, "How the error grows with the forecast horizon (lower = better)")
            fig.update_xaxes(title="periods ahead")
            st.plotly_chart(fig, width="stretch")
            if "ARIMA" in avg.index:
                dmA = R["dm"][R["dm"]["Model"] == "ARIMA"]
                if len(dmA):
                    items = [(f"ARIMA vs {R['base']}, h={int(r['Horizon'])}", r["Reading"].split(" (")[0].title(), "ok" if ("better" in r["Reading"] and r["p-value"] < .05) else "info" if "better" in r["Reading"] else "warn", f"p = {r['p-value']:.3f}")
                             for r in dmA.to_dict("records")]
                    st.markdown("##### Is the difference real? (Diebold-Mariano test)")
                    chips(items[:4], 4)
            with st.expander("Tables"):
                st.caption(f"{R['origins']} expanding-window origins. Orders fixed at the full-sample choice; ETS chosen on the first training window.")
                show(tb, {"RMSE": "{:.4f}", "MAE": "{:.4f}", "MASE": "{:.3f}", "Bias": "{:.4f}"})
                show(R["dm"], {"DM stat": "{:.3f}", "p-value": "{:.4f}"})
            st.caption("RMSE = typical size of the miss. MASE below 1 beats the naive forecast measured in-sample. Negative DM statistic = the model has lower squared error than the benchmark.")

    # ================================================================ GARCH
    with t_vol:
        st.markdown("##### Volatility: GARCH / GJR-GARCH / EGARCH with robust (Bollerslev-Wooldridge) standard errors")
        if kind == "logdiff":
            ret = y.copy()
        elif (y_lvl > 0).all() and kind in ("level", "log"):
            ret = (100 * np.log(y_lvl).diff()).dropna()
        else:
            ret = y.copy()
            card("This series cannot be turned into log returns; using the modelled series as is. GARCH is meant for returns / growth rates.", "bad")
        a_s, a_p = arch_lm(ret.values - ret.values.mean(), 10)
        card(f"<b>Is GARCH warranted?</b> ARCH-LM(10) on demeaned returns: p = {a_p:.4f} → "
             + ("<b>yes</b>, volatility clusters." if a_p < .05 else "no clear ARCH effect; a constant-variance model with robust SEs is probably enough."), "good" if a_p < .05 else "bad")
        c1, c2, c3, c4 = st.columns(4)
        arl = c1.selectbox("AR lags in the mean", ["auto (BIC)", 0, 1, 2, 3, 4, 5])
        dists = c2.multiselect("Distributions", ["normal", "t", "skewt"], default=["normal", "t", "skewt"])
        fams = c3.multiselect("Variance models", list(GARCH_SPECS), default=list(GARCH_SPECS))
        hg = c4.number_input("Vol forecast horizon", 1, 250, 20 if freq == "D" else 12)
        if len(ret) < (250 if freq == "D" else 100):
            st.warning(f"{len(ret)} observations is on the short side for GARCH estimation; treat parameters with caution.")
        if st.button("Fit volatility models", type="primary"):
            try:
                with st.spinner("Fitting GARCH family..."):
                    rv = ret.values.astype(float)
                    lags = ar_lags_by_bic(rv) if arl == "auto (BIC)" else int(arl)
                    G = run_garch(rv, lags, int(hg), ppy, tuple(dists), tuple(fams))
                    try:
                        Vv = garch_validation(rv, G["name"], G["dist"], lags, 0.94 if freq in ("D", "W") else 0.97)
                    except Exception:  # noqa: BLE001
                        Vv = None
                st.session_state["garch"] = dict(G=G, key=(sname, freq, kind, win), lags=lags, idx=ret.index, V=Vv)
            except Exception as e:  # noqa: BLE001
                st.error(f"GARCH failed: {type(e).__name__}: {e}")
        Gs = st.session_state.get("garch")
        if Gs and Gs["key"] == (sname, freq, kind, win):
            G = Gs["G"]
            cv = G["cond_vol"] * np.sqrt(ppy)
            pct = float((cv < cv[-1]).mean() * 100)
            lr = G["uncond_vol"] * np.sqrt(ppy) if np.isfinite(G["uncond_vol"]) else None
            card(f"<b>Right now</b> annualised volatility is about <b>{cv[-1]:.1f}%</b>, higher than <b>{pct:.0f}%</b> of history"
                 + (f" and {'above' if cv[-1] > lr else 'below'} its long-run level of {lr:.1f}%" if lr else "")
                 + (f". A shock to volatility loses half its strength in about <b>{G['half_life']:.0f} periods</b>." if np.isfinite(G["half_life"]) else ".")
                 + f" Best model: <b>{G['name']} / {G['dist']}</b> (AR({Gs['lags']}) mean).", "bad" if pct > 80 else "good" if pct < 40 else "")
            chips([("Persistence", f"{G['persistence']:.3f}", "warn" if G["persistence"] > .98 else "info", "Near 1 = volatility spells last very long."),
                   ("Shock half-life", f"{G['half_life']:.1f} periods" if np.isfinite(G["half_life"]) else "n/a", "info", "How fast a volatility spike fades."),
                   ("Long-run vol (ann.)", f"{lr:.1f}%" if lr else "n/a", "info", "Where volatility settles in the long run."),
                   ("1-period VaR 95% / 99%", f"{G['var1']:.2f}% / {G['var99']:.2f}%", "info", "Loss that should be exceeded only 5% / 1% of the time.")], 4)
            idx = Gs["idx"][-len(G["cond_vol"]):]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx, y=np.abs(ret.values[-len(idx):]) * np.sqrt(ppy), name="actual |move| (annualised)", line=dict(color=MUTED, width=0.8), opacity=0.5))
            fig.add_trace(go.Scatter(x=idx, y=cv, name="model volatility (annualised %)", line=dict(color=GOLD, width=2)))
            if lr:
                fig.add_hline(y=lr, line=dict(color=CYAN, width=1, dash="dot"), annotation_text="long-run level")
            st.plotly_chart(layout(fig, 340, "Calm and stormy spells - model volatility vs actual moves"), width="stretch")
            lvl_ = y_lvl.reindex(idx).dropna()
            if len(lvl_) > 20:
                cvs = pd.Series(cv, index=idx)
                sm_ = cvs.rolling(5, min_periods=1).median() if freq == "D" else cvs
                state = np.where(sm_ >= cvs.quantile(.8), 1, np.where(sm_ <= cvs.quantile(.2), -1, 0))
                shapes, a_ = [], 0
                for t_ in range(1, len(state) + 1):
                    if t_ == len(state) or state[t_] != state[a_]:
                        if state[a_] != 0:
                            shapes.append(dict(type="rect", xref="x", yref="paper", x0=idx[a_], x1=idx[t_ - 1], y0=0, y1=1, line_width=0, layer="below",
                                               fillcolor="rgba(244,87,122,0.22)" if state[a_] > 0 else "rgba(45,212,160,0.14)"))
                        a_ = t_
                fig = go.Figure(go.Scatter(x=lvl_.index, y=lvl_.values, line=dict(color=TEXT, width=1.5)))
                layout(fig, 320, f"What the stormy (red) and calm (green) stretches looked like on {sname[:60]} itself", legend=False)
                fig.update_layout(shapes=shapes)
                fig.update_yaxes(title=unit)
                st.plotly_chart(fig, width="stretch")
                now_s = "a stormy stretch" if state[-1] > 0 else "a calm stretch" if state[-1] < 0 else "neither extreme"
                st.caption(f"Red = model volatility in its top fifth of history, green = bottom fifth. Right now: {now_s}. Look at what the series did inside and just after the red bands.")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Histogram(x=cv, nbinsx=40, marker_color=PURPLE))
                fig.add_vline(x=cv[-1], line=dict(color=GOLD, width=3), annotation_text="now")
                st.plotly_chart(layout(fig, 260, "Where today's volatility sits in its history", legend=False), width="stretch")
            with c2:
                fdt = future_dates(idx[-1], len(G["var_path"]), freq)
                fig = go.Figure(go.Scatter(x=fdt, y=np.sqrt(G["var_path"]) * np.sqrt(ppy), line=dict(color=PURPLE, width=2.6)))
                if lr:
                    fig.add_hline(y=lr, line=dict(color=CYAN, width=1, dash="dot"), annotation_text="long-run level")
                st.plotly_chart(layout(fig, 260, "Volatility forecast (annualised %)", legend=False), width="stretch")
            st.markdown("##### Health check on the model")
            check_chips(G["checks"], 3)
            eng = G["engle_ng"]
            if eng["joint"] < .05 and G["name"] == "GARCH(1,1)":
                card("Bad news raises volatility more than good news (sign-bias). The symmetric GARCH misses this - prefer GJR or EGARCH.", "bad")
            st.markdown("##### Does it hold up on data it has not seen?")
            Vv = Gs.get("V")
            if not Vv:
                card("The sample is too short to hold back 30% for an honest test (needs about 80 observations to fit and 40 to test).", "mute")
            else:
                better = Vv["ql_g"] < Vv["ql_e"]
                sigd = np.isfinite(Vv["dm_p"]) and Vv["dm_p"] < .05

                def _breach(x_, p_, rate):
                    exp_ = rate * Vv["n_out"]
                    okk = (not np.isfinite(p_)) or p_ >= .05
                    return (f"{x_} of {Vv['n_out']}", "ok" if okk else "bad",
                            f"About {exp_:.0f} expected. " + ("In line with the promise." if okk else "Too many: risk is understated." if x_ > exp_ else "Too few: risk is overstated."))
                b95, b99 = _breach(Vv["x95"], Vv["p95"], .05), _breach(Vv["x99"], Vv["p99"], .01)
                card(f"Fitted on the first 70% of the sample, then left to forecast the last <b>{Vv['n_out']} {pw}s</b> one at a time with its settings frozen. "
                     + (f"<b>{G['name']} forecast volatility better than the simple benchmark</b>" + (", and the gap is statistically real." if sigd else ", but the gap is small enough to be luck.") if better
                        else f"<b>{G['name']} did no better than the simple benchmark</b> (a weighted average of recent squared moves) - the extra modelling is not buying accuracy here; use it for the regime reading, not for precision.")
                     + f" Its 95% VaR was breached {Vv['x95']} times against about {0.05 * Vv['n_out']:.0f} expected.",
                     "good" if (better and b95[1] == "ok") else "bad" if (not better and b95[1] == "bad") else "")
                chips([("vs simple EWMA benchmark", ("Better" if better else "Worse") + ("" if sigd else " (not significant)"), "ok" if (better and sigd) else "info" if better else "warn",
                        f"Forecast loss {Vv['ql_g']:.3f} vs {Vv['ql_e']:.3f} (lower is better); p = {Vv['dm_p']:.3f}. EWMA weight {Vv['lam']}."),
                       ("95% VaR breaches", b95[0], b95[1], b95[2]), ("99% VaR breaches", b99[0], b99[1], b99[2])], 3)
                oi = Gs["idx"][-len(Vv["r_out"]):]
                br = Vv["r_out"] < Vv["v95"]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=oi, y=Vv["r_out"], name=f"actual {'daily' if pw == 'day' else pw + 'ly'} move (%)", marker_color=MUTED, marker_line_width=0))
                fig.add_trace(go.Scatter(x=oi, y=Vv["v95"], name="95% VaR promised beforehand", line=dict(color=GOLD, width=1.6)))
                fig.add_trace(go.Scatter(x=oi[br], y=Vv["r_out"][br], mode="markers", name="breach", marker=dict(color=RED, size=8)))
                layout(fig, 320, "The VaR test: moves in the held-back period against the loss limit set a period earlier")
                fig.update_layout(bargap=0, margin=dict(t=58), legend=dict(y=1.13))
                st.plotly_chart(fig, width="stretch")
                st.caption("A sound 95% VaR is broken about 1 time in 20, and the red dots should be scattered, not bunched. Bunched breaches mean the model reacts too slowly when a storm starts.")
            with st.expander("Tables: model comparison and parameters"):
                show(G["comparison"], {"LogLik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
                show(G["params"], {"Estimate": "{:.4f}", "Robust SE": "{:.4f}", "Classic SE": "{:.4f}", "t (robust)": "{:.2f}", "p (robust)": "{:.4f}"})
            st.caption("VaR uses filtered historical simulation (empirical quantiles of standardised residuals x next-period volatility).")

    # ================================================================ OLS
    with t_ols:
        st.markdown("##### OLS with heteroskedasticity- and autocorrelation-robust inference")
        allser = {f"{c} | {s}": (c, s) for c, s in data[["category", "series"]].drop_duplicates().itertuples(index=False) if (c, s) != (cat, sname)}
        xs_sel = st.multiselect("Explanatory series (optional - leave empty for an AR / trend regression)", list(allser))
        c1, c2, c3, c4 = st.columns(4)
        pyl = c1.number_input("Lags of the dependent variable", 0, 12, 1)
        kxl = c2.number_input("Lags of each regressor (0 = contemporaneous only)", 0, 12, 0)
        trend_in = c3.checkbox("Linear time trend", value=False)
        seas_in = c4.checkbox("Seasonal dummies", value=False, disabled=(m == 1))
        c1, c2, c3 = st.columns(3)
        make_stat = c1.checkbox("Make every variable stationary automatically", value=True,
                                help="log if strictly positive & not a rate, then first-difference those that the unit-root vote calls I(1)")
        covc = c2.selectbox("Covariance estimator", ["auto", "Classic", "HC3", "HAC"], help="auto: HAC if serial correlation is detected, otherwise HC3")
        hlag = c3.number_input("HAC lags (0 = Newey-West rule)", 0, 40, 0)
        use_d2 = st.checkbox("Outlier dummy window", key="ols_dummy")
        d2s, d2e = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30")
        if use_d2:
            e1, e2 = st.columns(2)
            d2s = pd.Timestamp(e1.date_input("From", d2s.date(), key="o_d1")); d2e = pd.Timestamp(e2.date_input("To", d2e.date(), key="o_d2"))
        if st.button("Estimate", type="primary", key="run_ols"):
            try:
                variables, info = {}, []
                ydf = y_lvl if kind in ("level", "log", "diff", "logdiff", "yoy") else y_lvl
                base_y = ydf
                if make_stat:
                    ys, yl, yd_ = auto_stationary(base_y, unit, sname)
                else:
                    ys, yl, yd_ = y, f"{sname} ({TRANSFORMS[kind]})", d_auto
                info.append({"Variable": "y: " + yl, "Differenced": yd_, "Unit-root vote": "I(%d)" % yd_ if make_stat else f"I({d_auto})"})
                X = {}
                for lbl in xs_sel:
                    c_, s_ = allser[lbl]
                    sx = data[(data["category"] == c_) & (data["series"] == s_)].groupby("date")["value"].last()
                    ux = data[(data["category"] == c_) & (data["series"] == s_)]["unit"].iloc[0]
                    nat = detect_freq(sx.index)
                    if ORDER.index(nat) > ORDER.index(freq):
                        info.append({"Variable": lbl, "Differenced": "-", "Unit-root vote": f"skipped: native {nat} is coarser than {freq}"})
                        continue
                    sx = regularize(sx, freq, "last")
                    if make_stat:
                        zx, nx, dx = auto_stationary(sx, ux, s_)
                    else:
                        zx, nx, dx = sx, s_, "-"
                    X[nx] = zx
                    info.append({"Variable": "x: " + nx, "Differenced": dx, "Unit-root vote": f"I({dx})" if make_stat else "levels (not tested)"})
                Xd = pd.DataFrame(index=ys.index)
                for k, v in X.items():
                    for L_ in range(0, int(kxl) + 1):
                        Xd[f"{k}" + (f" (t-{L_})" if L_ else "")] = v.reindex(ys.index).shift(L_)
                for L_ in range(1, int(pyl) + 1):
                    Xd[f"y (t-{L_})"] = ys.shift(L_)
                if trend_in:
                    Xd["trend"] = np.arange(len(ys), dtype=float)
                if seas_in and m > 1:
                    mon = ys.index.month if m == 12 else ys.index.quarter
                    dm_ = pd.get_dummies(mon, prefix="season", drop_first=True, dtype=float)
                    dm_.index = ys.index
                    Xd = pd.concat([Xd, dm_], axis=1)
                if use_d2:
                    dv = dummy_series(ys.index, d2s, d2e)
                    if dv.sum():
                        Xd["outlier dummy"] = dv
                R = ols_lab(ys, Xd, covc, int(hlag))
                gr, co = [], []
                if X:
                    dfg = pd.concat([ys.rename("y")] + [v.rename(k) for k, v in X.items()], axis=1).dropna()
                    for k in X:
                        try:
                            g1 = grangercausalitytests(dfg[["y", k]], maxlag=min(4, len(dfg) // 10), verbose=False)
                            g2 = grangercausalitytests(dfg[[k, "y"]], maxlag=min(4, len(dfg) // 10), verbose=False)
                            gr.append({"Regressor": k, **{f"{k_}→y lag{L_}": g1[L_][0]["ssr_ftest"][1] for L_ in g1 for k_ in [k]},
                                       **{f"y→{k} lag{L_}": g2[L_][0]["ssr_ftest"][1] for L_ in g2}})
                        except Exception:  # noqa: BLE001
                            pass
                    if not make_stat:
                        for k in X:
                            try:
                                cs = coint(dfg["y"], dfg[k], trend="c")
                                co.append({"Pair": f"y ~ {k}", "Engle-Granger stat": cs[0], "p-value": cs[1],
                                           "Reading": "cointegrated (levels regression is meaningful; consider an ECM)" if cs[1] < .05 else "NOT cointegrated - a levels regression is likely spurious; difference the variables"})
                            except Exception:  # noqa: BLE001
                                pass
                st.session_state["ols"] = dict(R=R, info=pd.DataFrame(info), gr=pd.DataFrame(gr), co=pd.DataFrame(co), key=(sname, freq, kind, win))
            except Exception as e:  # noqa: BLE001
                st.error(f"Regression failed: {type(e).__name__}: {e}")
        O = st.session_state.get("ols")
        if O and O["key"] == (sname, freq, kind, win):
            R = O["R"]
            tfail = int((R["tests"]["Result"] == "FAIL").sum())
            chips([("Explained", f"{R['r2'] * 100:.0f}%", "ok" if R["r2"] > .3 else "info", f"R² {R['r2']:.3f} (adj. {R['adj_r2']:.3f}); {R['n']} observations."),
                   ("Standard errors", R["picked"], "info", ("Heteroskedasticity found. " if R["heterosked"] else "") + ("Serial correlation found." if R["serial"] else "Corrected only where needed.")),
                   ("Durbin-Watson", f"{R['dw']:.2f}", "ok" if 1.5 < R["dw"] < 2.5 else "warn", "About 2 = no autocorrelation in residuals."),
                   ("Specification tests", f"{len(R['tests']) - tfail}/{len(R['tests'])} pass", "ok" if tfail == 0 else "warn" if tfail <= 2 else "bad", "Red boxes below explain what failed.")], 4)
            cf = R["coef"].copy()
            cf = cf[cf["Variable"] != "const"]
            if len(cf):
                se = (cf["Coefficient"] / cf["t (chosen)"]).abs()
                st.plotly_chart(forest(cf["Variable"], cf["Coefficient"], cf["Coefficient"] - 1.96 * se, cf["Coefficient"] + 1.96 * se, cf["p (chosen)"],
                                       f"Effect on y of +1 unit in each variable, with 95% intervals ({R['picked']} errors; green = significant)"), width="stretch")
                sig = cf[cf["p (chosen)"] < .05]
                card("<b>Significant drivers:</b> " + (", ".join(f"<b>{r.Variable}</b> ({'+' if r.Coefficient > 0 else '-'}{abs(r.Coefficient):.3g})" for r in sig.itertuples()) if len(sig) else "none at the 5% level.") +
                     " Coefficients are in the (possibly differenced) units shown in the table above." )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=R["y"].index, y=R["y"].values, name="actual", line=dict(color=TEXT, width=1.3)))
            fig.add_trace(go.Scatter(x=R["fitted"].index, y=R["fitted"].values, name="fitted", line=dict(color=GOLD, width=1.8)))
            st.plotly_chart(layout(fig, 300, "Actual vs fitted"), width="stretch")
            st.markdown("##### Health check")
            check_chips(R["tests"], 4)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(layout(go.Figure(go.Scatter(x=R["resid"].index, y=R["resid"].values, line=dict(color=CYAN, width=1))), 230, "Residuals", legend=False), width="stretch")
            with cc2:
                st.plotly_chart(acf_fig(R["resid"].values, 24, "Residual ACF"), width="stretch")
            if len(O["co"]):
                st.markdown("##### Cointegration (levels regression)")
                show(O["co"], {"Engle-Granger stat": "{:.3f}", "p-value": "{:.4f}"})
            with st.expander("Tables: variables used, full coefficients (classic / HC3 / HAC), multicollinearity, Granger causality"):
                show(O["info"])
                show(R["coef"], {c: "{:.4f}" for c in R["coef"].columns if c != "Variable"})
                if len(R["vif"]):
                    show(R["vif"], {"VIF": "{:.2f}"})
                if len(O["gr"]):
                    show(O["gr"], {c: "{:.4f}" for c in O["gr"].columns if c != "Regressor"})
                show(R["tests"], {"Statistic": "{:.3f}", "p-value": "{:.4f}"})

    # ================================================================ METHOD
    with t_guide:
        st.markdown("""
##### How to use the lab

**1 · BRIEFING - what deserves attention.** Every series in the upload is compared with its own history. Rates, ratios and surveys are read by their level; indices and prices by their year-on-year growth. Three lists come out: readings in the top or bottom 5% of history, latest moves more than 2.5x a normal move, and series turning against their past year's direction. Tick a row to open that series.

**2 · THIS SERIES - what to make of it.** Where it stands, which way it is moving, where the model expects it to go, and - most important - whether that model would have been worth listening to in the past (replayed at up to 12 earlier dates against simply assuming no change). Then two questions a forecast can answer: the odds of being above a level you choose, and what followed the last times the series stood where it stands now.

**3 · RELATIONSHIPS - what moves with it.** The selected series is compared with every other one to find what moves first, together or after, keeping only links that are above noise and hold in both halves of the sample. Pick a pair to see the timing, whether the link has been steady, and what the selected series did after the other one stood at different levels.

**The rest is the workbench** for changing the model (FORECAST), testing it harder (BACKTEST), modelling risk (VOLATILITY), testing drivers formally (REGRESSION) and seeing the statistical tests (DIAGNOSTICS).

**VOLATILITY is tested, not just fitted.** The chosen GARCH model is re-estimated on the first 70% of the sample and made to forecast the rest. It is compared with a plain weighted average of recent squared moves (EWMA, weight 0.94 for daily / weekly and 0.97 otherwise), and its 95% and 99% VaR are checked against how often they were actually broken. "Turbulence" on the first two tabs is that same simple EWMA measure, ranked against the series' own history - no model fitting needed.

**Three habits that keep the readings honest.** A short history makes any reading look extreme. Overlapping windows (12-month changes, look-ahead studies) repeat the same episodes, so count episodes rather than observations. And a link found by scanning many series needs an economic reason before it is believed.
""")
        st.markdown("##### Plain-English glossary")
        show(pd.DataFrame([
            ("Stationary / stable", "Hovers around a fixed average with steady swings. Can be modelled as is."),
            ("Unit root / wandering", "No fixed home: shocks are permanent. Model the changes instead."),
            ("d, D", "How many times to take changes (d) or year-on-year changes (D) before modelling."),
            ("AR / MA", "AR = today depends on past values. MA = today depends on past surprises."),
            ("AICc / BIC", "Model scores that reward fit and punish complexity. Lower is better."),
            ("Robust standard errors", "Error bars that stay honest when variance changes (HC3) or errors are linked over time (HAC)."),
            ("Heteroskedasticity", "Variance is not constant - big moves cluster. Classic error bars become wrong."),
            ("Ljung-Box / ARCH-LM", "Tests for leftover patterns in the residuals / in their volatility."),
            ("GARCH", "A model of how volatility itself evolves: calm and stormy spells."),
            ("RMSE / MASE", "Typical forecast miss. MASE < 1 means better than a naive forecast."),
            ("Diebold-Mariano", "Test of whether two models' forecast errors really differ."),
            ("Cointegration", "Two wandering series that move together long-term; regressing them is then meaningful."),
        ], columns=["Term", "Meaning"]))
        st.markdown(f"""
**How specifications are chosen**

1. **Frequency & transformation.** Series are regularised to a calendar grid (month-end, quarter-end, Friday weeks) or kept on trading days. Strictly positive indices/prices default to logs; rates, ratios and YoY series stay in levels.
2. **Order of integration (d).** ADF, Phillips-Perron, DF-GLS (H0: unit root) and KPSS (H0: stationary) are voted at 5%; 3 of 4, or a 2-2 split that ADF breaks, means stationary. Repeated on differences, max 2. A constant-plus-trend test is used only when a significant HAC trend exists.
3. **Seasonality (D).** STL seasonal strength; above 0.64 (Wang-Smith-Hyndman) a seasonal difference is applied. Weekly/daily/annual data are treated as non-seasonal.
4. **Deterministic terms.** ARIMA treats trends as regressors on the undifferenced series, so: d+D = 0 → constant (or constant+trend if a trend is significant); d+D = 1 → drift only if the mean of the differenced series is significantly non-zero (HAC t-test); d+D ≥ 2 → none.
5. **Orders (p,q,P,Q).** Grid search at fixed d, D and trend by AICc (or BIC/AIC). Information criteria are only comparable at the same differencing, which is why d and D are fixed first.
6. **Inference.** ARIMA/SARIMA use the sandwich ("robust") covariance, shown next to OPG. OLS shows classic, HC3 and Newey-West HAC (bandwidth ⌊4(n/100)^(2/9)⌋); the reported t/p use HAC when Breusch-Godfrey or Ljung-Box flag serial correlation, otherwise HC3. GARCH uses Bollerslev-Wooldridge robust SEs.
7. **Checks.** Ljung-Box (residuals and squares, degrees of freedom adjusted), ARCH-LM, Jarque-Bera, mean-zero, variance-ratio; for OLS also Breusch-Pagan, White, Breusch-Godfrey, RESET, CUSUM, VIF. Each failing check says what it implies.
8. **Backtest.** Expanding window, models refit at each origin, orders fixed; RMSE/MAE/MASE and Diebold-Mariano (HLN-corrected) against the best naive benchmark.
9. **Regression hygiene.** If variables are integrated and not cointegrated the levels regression is spurious; the lab differences them automatically (or runs Engle-Granger when you keep levels).

**Caveats.** Forecast bands assume constant variance and Gaussian errors; with ARCH effects they are too wide in calm and too narrow in stressed periods. Short macro series (≈50 quarterly GDP points) support only low-order models. 2020 is a structural outlier for IIP, GDP, WPI and others - use the dummy window rather than trusting a model that treats it as ordinary noise. Overlapping YoY series have built-in MA(11) errors.
""")


if os.environ.get("FORECAST_LAB_NO_UI") != "1":
    main()
