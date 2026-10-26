"""VIKA Forecast Lab - time-series diagnostics, ARIMA-family models, GARCH and robust OLS on the Macro Terminal's DATA export.

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

BUILD = "forecast lab 26-Oct-2026 c"

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
.card b{{color:{GOLD}}}
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
    st.markdown(f'<div class="lab-hdr"><b>VIKA WEALTH</b><span>FORECAST LAB · time-series diagnostics, ARIMA family, GARCH, robust OLS · {BUILD}</span></div>', unsafe_allow_html=True)

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

    with st.sidebar:
        cats = sorted(data["category"].unique())
        cat = st.selectbox("Category", cats)
        names = sorted(data.loc[data["category"] == cat, "series"].unique())
        q = st.text_input("Filter series", "")
        shown = [n for n in names if q.lower() in n.lower()] or names
        sname = st.selectbox("Series", shown)
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

    tabs = st.tabs(["SUMMARY", "DATA", "DIAGNOSTICS", "FORECAST (ARIMA)", "BACKTEST", "VOLATILITY (GARCH)", "REGRESSION (OLS)", "GUIDE"])

    # ================================================================ DATA
    with tabs[1]:
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

    # ================================================================ SUMMARY
    with tabs[0]:
        hsum = 12 if freq in ("M", "W") else 8 if freq == "Q" else 5 if freq == "A" else 20
        DD0 = D_auto if m > 1 else 0
        tr0 = trend_for(d_auto, DD0, has_trend(y) if d_auto == 0 and DD0 == 0 else False, drift_significant(y, d_auto, DD0, m))
        out0 = None
        try:
            with st.spinner("Fitting the best model automatically..."):
                yv0 = y.values.astype(float)
                cand0 = arima_search(yv0, int(d_auto), int(DD0), m, tr0, 2, 2, 1 if DD0 else 0, 1 if DD0 else 0, "AICc", None)
                b0 = cand0.iloc[0]
                order0 = (int(b0.p), int(b0.d), int(b0.q))
                seas0 = (int(b0.P), int(b0.D), int(b0.Q), int(b0.m)) if b0.m else (0, 0, 0, 0)
                out0 = run_arima(yv0, order0, seas0, tr0, None, None, hsum)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Automatic model could not be fitted ({type(e).__name__}); use the other tabs.")
        fd0 = future_dates(y.index[-1], hsum, freq)
        last_v = float(inv(np.array([y.iloc[-1]]))[0])
        if out0 is not None:
            end_v = float(inv(out0["mean"])[-1]); lo_v = float(inv(out0["lo80"])[-1]); hi_v = float(inv(out0["hi80"])[-1])
            move = end_v - last_v
            card(f"<b>{sname}</b> last read <b>{last_v:,.2f}</b> ({ylabel}) on {y.index[-1]:%d %b %Y}. "
                 f"The best-fitting model expects <b>{end_v:,.2f}</b> by {fd0[-1]:%b %Y} - {'essentially unchanged' if abs(move) < 0.005 * max(abs(last_v), 1e-9) else ('a rise of ' if move > 0 else 'a fall of ') + format(abs(move), ',.2f')} "
                 f"(80% chance it lands between {lo_v:,.2f} and {hi_v:,.2f}).")
        zp = diag["za"]["p"] if diag["za"] else 1.0
        arp = diag["arch"][1]
        nout = len(outliers(diag["yd"]))
        phi = None
        if out0 is not None:
            arr = out0["params"][out0["params"]["Parameter"].str.startswith("ar.L")]["Estimate"]
            if len(arr):
                phi = float(arr.sum())
        items = []
        if d_auto == 0:
            hl = (np.log(.5) / np.log(phi)) if (phi is not None and 0 < phi < 1) else None
            items.append(("Behaviour", "Mean-reverting", "ok", "It keeps coming back to its average" + (f"; a shock fades by half in about {hl:.1f} periods." if hl else ".")))
        else:
            items.append(("Behaviour", "Wanders" if d_auto == 1 else "Very smooth", "warn", "Shocks do not fade: levels drift, so forecasts widen over time. Modelled in changes."))
        tr_sig = has_trend(y) if d_auto == 0 else drift_significant(y, d_auto, 0, 1)
        items.append(("Trend", "Yes" if tr_sig else "None clear", "info", "A steady up/down drift is statistically visible." if tr_sig else "No steady drift detected."))
        if m > 1:
            ss = diag["seas_strength"]
            items.append(("Seasonality", "Strong" if ss > .64 else "Weak", "info" if ss > .64 else "ok", f"Seasonal strength {ss:.2f} (above 0.64 is treated as seasonal)."))
        items.append(("Volatility clusters", "Yes" if arp < .05 else "No", "warn" if arp < .05 else "ok", "Calm and stormy spells: see the VOLATILITY tab." if arp < .05 else "Variance looks steady."))
        items.append(("Structural break", "Possible" if zp < .05 else "None obvious", "warn" if zp < .05 else "ok", "A regime shift may distort the model; try the outlier dummy." if zp < .05 else "No dominant break found."))
        items.append(("Outliers", str(nout), "warn" if nout else "ok", "Extreme jumps (e.g. COVID) - consider a dummy window." if nout else "No extreme jumps."))
        if out0 is not None:
            f0 = int((out0["checks"]["Result"] == "FAIL").sum()); n0 = len(out0["checks"])
            items.append(("Model health", f"{n0 - f0}/{n0} checks pass", "ok" if f0 == 0 else "warn" if f0 <= 2 else "bad", "Residual tests: ok means the model captured the pattern."))
            items.append(("Chosen model", f"ARIMA{order0}" + (f"x{seas0}" if seas0[3] else ""), "info", f"trend term '{tr0}'; picked by AICc."))
        chips(items, 4)
        if out0 is not None:
            bands = {"list": [(out0["lo95"], out0["hi95"], "95% range", "rgba(139,92,246,0.18)"), (out0["lo80"], out0["hi80"], "80% range", "rgba(139,92,246,0.34)")]}
            fig = fan_fig(y, fd0, out0["mean"], bands, inv, f"Where it is heading - {sname}", ylabel, 400)
            if d_auto == 0:
                fig.add_hline(y=float(inv(np.array([y.mean()]))[0]), line=dict(color=GOLD, width=1, dash="dot"), annotation_text="long-run average")
            st.plotly_chart(fig, width="stretch")
            st.caption("Gold line = central forecast; shaded bands = 80% / 95% ranges. The bands assume steady volatility. "
                       "Use the other tabs to change the model, test it against simple benchmarks (BACKTEST) or model the volatility (VOLATILITY).")

    # ================================================================ DIAGNOSTICS
    with tabs[2]:
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
    with tabs[3]:
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
    with tabs[4]:
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
    with tabs[5]:
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
                st.session_state["garch"] = dict(G=G, key=(sname, freq, kind, win), lags=lags, idx=ret.index)
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
            with st.expander("Tables: model comparison and parameters"):
                show(G["comparison"], {"LogLik": "{:.1f}", "AIC": "{:.1f}", "BIC": "{:.1f}"})
                show(G["params"], {"Estimate": "{:.4f}", "Robust SE": "{:.4f}", "Classic SE": "{:.4f}", "t (robust)": "{:.2f}", "p (robust)": "{:.4f}"})
            st.caption("VaR uses filtered historical simulation (empirical quantiles of standardised residuals x next-period volatility).")

    # ================================================================ OLS
    with tabs[6]:
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
    with tabs[7]:
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
