`asteroid_feedstock.py`. To run it, install the packages it needs with `pip install pandas numpy scipy matplotlib tabulate`, then run `python asteroid_feedstock.py --fetch`. If the Inter font isn't installed on your computer, the charts fall back to a default font.

```python
#!/usr/bin/env python3
"""
Asteroid feedstock analysis for in-space fabrication
====================================================

Research question
    Moon or asteroids: where should in-space fabrication get its metal?
    Which accessible near-Earth asteroids (NEAs) are likely metal-rich?

Data (NASA / JPL)
    1. SBDB Query API: every known NEA with size, albedo, spectral type, orbit
       https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=spkid,full_name,pdes,H,
       diameter,albedo,spec_B,spec_T,rot_per,a,e,i,q,ad,moid,class,pha&sb-kind=a&sb-group=neo
    2. NHATS API: NEAs reachable round trip from a 400 km Earth orbit for
       <= 12 km/s total delta-v (NASA's "accessible" list)
       https://ssd-api.jpl.nasa.gov/nhats.api

What the script does
    1. Data gap: how many NEAs have a measured composition at all
    2. Albedo test: can brightness alone separate carbon-rich (C) from rocky (S)
       asteroids? Validated on asteroids whose spectral type IS known
    3. Correlations: what orbital/physical properties predict delta-v
    4. Moon comparison: how many asteroids beat a lunar round trip on delta-v
    5. Metal candidates: accessible asteroids that look metal-rich
    6. Iron estimate: tonnes of iron in a metal-rich asteroid, with uncertainty
    7. Energy: melting asteroid metal vs smelting oxide ore

Usage
    python asteroid_feedstock.py --sbdb data/sbdb.json --nhats data/nhats.json --out out
    python asteroid_feedstock.py --fetch --out out      # downloads the data itself
"""

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants and assumptions (all stated so they can go in the report)
# ---------------------------------------------------------------------------
SBDB_URL = ("https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=spkid,full_name,pdes,H,"
            "diameter,albedo,spec_B,spec_T,rot_per,a,e,i,q,ad,moid,class,pha"
            "&sb-kind=a&sb-group=neo")
NHATS_URL = "https://ssd-api.jpl.nasa.gov/nhats.api"

# Round-trip delta-v from low Earth orbit for lunar destinations, on the same
# basis as NHATS (Barbee et al. 2015, NASA NTRS 20150009390)
DV_LUNAR_ORBIT_RT = 5.0    # km/s, low lunar orbit and back
DV_LUNAR_SURFACE_RT = 9.0  # km/s, lunar surface and back

# Tholen-style albedo split of the featureless X-complex
X_DARK_MAX = 0.10   # below: P-type (dark, carbon-rich)
X_METAL_MAX = 0.30  # 0.10-0.30: M-type (metal-rich); above: E-type (enstatite)

# Iron-content assumptions
METAL_FRAC_M = 0.85            # Sanchez et al. 2021: 1986 DA surface ~85% metal
FE_IN_METAL = 0.90             # iron-nickel metal is ~90% iron
DENSITY_M = {"low": 3000, "mid": 4500, "high": 7000}   # kg/m3: rubble pile .. solid iron
ISS_MASS_KG = 420_000          # International Space Station, ~420 t

# Energy per tonne of iron (GJ/t)
ENERGY = {
    # Heat iron 25 C -> 1538 C (~59 kJ/mol) + latent heat of fusion (13.8 kJ/mol)
    "melt": 72.7e3 / 55.845e-3 / 1e9 * 1e3,           # ~1.30 GJ/t
    # Thermodynamic minimum to strip oxygen = enthalpy of formation per kg Fe
    "smelt_floor_FeO": 272.0e3 / 55.845e-3 / 1e9 * 1e3,        # ~4.9 GJ/t
    "smelt_floor_Fe2O3": 824.2e3 / 2 / 55.845e-3 / 1e9 * 1e3,  # ~7.4 GJ/t
    # worldsteel 2024 global average, whole ore-to-steel route (per t crude steel)
    "steel_route_avg": 20.95,
}

# Chart style: dark "mission briefing" look to match the slide deck.
# Transparent background (the slide's near-black shows through), Inter font,
# minimal axes, light text. Colours: validated reference palette, dark-mode steps
# (first three categorical slots pass all-pairs CVD checks on #0B0D12).
SURFACE = "#0B0D12"     # slide background (ring colour around overlapping marks)
PANEL = "#141922"       # legend / note boxes
INK = "#EEF1F5"
INK_2 = "#A3ACB9"
GRID = "#262D38"
AXIS = "#3A4250"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
GRAY = "#4B5361"
RED_DIV = "#e66767"
BAR_NEUTRAL = "#5B6474"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "legend.title_fontsize": 13,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.transparent": True,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.top": False, "ytick.right": False,
    "xtick.major.size": 5, "ytick.major.size": 5, "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": True, "legend.edgecolor": GRID, "legend.facecolor": PANEL, "legend.fancybox": False,
    "legend.framealpha": 0.95, "legend.borderpad": 0.6, "legend.labelcolor": INK,
    "savefig.dpi": 300, "axes.axisbelow": True,
})
FIG_WIDE = (11, 6.2)    # main result figure
FIG_HALF = (8, 5.6)     # figures that share a slide with text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def fetch(url, path):
    with urllib.request.urlopen(url, timeout=120) as r:
        Path(path).write_bytes(r.read())


def load_sbdb(path):
    j = json.loads(Path(path).read_text())
    df = pd.DataFrame(j["data"], columns=j["fields"])
    for c in ["H", "diameter", "albedo", "rot_per", "a", "e", "i", "q", "ad", "moid"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["full_name"] = df["full_name"].str.strip()
    df["pdes"] = df["pdes"].str.strip()
    return df


def load_nhats(path):
    j = json.loads(Path(path).read_text())
    rows = []
    for d in j["data"]:
        mdv = d.get("min_dv") or {}
        rows.append({
            "pdes": str(d.get("des", "")).strip(),
            "dv": pd.to_numeric(mdv.get("dv"), errors="coerce"),
            "dv_trip_days": pd.to_numeric(mdv.get("dur"), errors="coerce"),
            "size_min_m": pd.to_numeric(d.get("min_size"), errors="coerce"),
            "size_max_m": pd.to_numeric(d.get("max_size"), errors="coerce"),
            "n_traj": pd.to_numeric(d.get("n_via_traj"), errors="coerce"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composition classes
# ---------------------------------------------------------------------------
def _clean(code):
    if not isinstance(code, str):
        return ""
    m = re.match(r"[A-Za-z]+", code.strip())
    return m.group(0) if m else ""


def spectral_class(spec_b, spec_t, albedo):
    """Map SMASS (Bus) / Tholen types to a feedstock class.

    Returns (class, used_albedo). used_albedo=True means albedo was needed to
    decide (X-complex), so those rows are excluded from the albedo validation.
    """
    b, t = _clean(spec_b), _clean(spec_t).upper()
    if not b and not t:
        return None, False
    if t.startswith("M"):
        return "Metal-rich (M)", False
    letter = (b[0] if b else t[0]).upper()
    if letter in "CBDTFGP":
        return "Carbon-rich (C)", False
    if letter in "SQVAKLRO":
        return "Rocky (S)", False
    if letter in "XE":
        if b and b.lower() == "xe":      # Bus Xe: enstatite-like, defined by spectrum
            return "Enstatite (E)", False
        if letter == "E":                # Tholen E: defined using high albedo
            return "Enstatite (E)", True
        if pd.isna(albedo):
            return "X, albedo unknown", False
        if albedo < X_DARK_MAX:
            return "Carbon-rich (C)", True
        if albedo <= X_METAL_MAX:
            return "Metal-rich (M)", True
        return "Enstatite (E)", True
    return "Other", False


def diameter_m(row):
    """Best diameter in metres: measured, else from H with class-typical albedo."""
    if pd.notna(row["diameter"]):
        return row["diameter"] * 1000
    p = row["albedo"] if pd.notna(row["albedo"]) else np.nan
    if pd.isna(p):
        return np.nan
    return 1329.0 / math.sqrt(p) * 10 ** (-row["H"] / 5) * 1000


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse(sbdb, nhats):
    R = {}  # results for the summary

    # Classes
    cls = sbdb.apply(lambda r: spectral_class(r["spec_B"], r["spec_T"], r["albedo"]), axis=1)
    sbdb["spec_class"] = [c[0] for c in cls]
    sbdb["class_used_albedo"] = [c[1] for c in cls]
    sbdb["has_type"] = sbdb["spec_class"].notna()
    sbdb["D_m"] = sbdb.apply(diameter_m, axis=1)

    acc = nhats.merge(sbdb, on="pdes", how="left", validate="one_to_one")
    acc["D_m"] = acc["D_m"].fillna(np.sqrt(acc["size_min_m"] * acc["size_max_m"]))
    R["n_matched"] = int(acc["spkid"].notna().sum())

    # 1. Data gap ----------------------------------------------------------
    R["n_nea"] = len(sbdb)
    R["n_nea_albedo"] = int(sbdb["albedo"].notna().sum())
    R["n_nea_type"] = int(sbdb["has_type"].sum())
    R["n_acc"] = len(acc)
    R["n_acc_albedo"] = int(acc["albedo"].notna().sum())
    R["n_acc_type"] = int(acc["has_type"].fillna(False).astype(bool).sum())
    R["median_D_all"] = float(sbdb["D_m"].median())
    R["median_D_acc_est"] = float(acc["D_m"].median())

    # 2. Albedo test --------------------------------------------------------
    val = sbdb[sbdb["albedo"].notna()
               & sbdb["spec_class"].isin(["Carbon-rich (C)", "Rocky (S)"])
               & ~sbdb["class_used_albedo"]].copy()
    val["is_C"] = val["spec_class"].eq("Carbon-rich (C)")
    sweep = []
    for th in np.round(np.arange(0.03, 0.251, 0.005), 3):
        pred = val["albedo"] < th
        tpr = (pred & val.is_C).sum() / val.is_C.sum()           # C correctly called C
        tnr = (~pred & ~val.is_C).sum() / (~val.is_C).sum()      # S correctly called S
        acc_ = (pred == val.is_C).mean()
        sweep.append((th, acc_, (tpr + tnr) / 2, tpr, tnr))
    sweep = pd.DataFrame(sweep, columns=["threshold", "accuracy", "balanced", "C_recall", "S_recall"])
    best = sweep.loc[sweep["balanced"].idxmax()]
    at10 = sweep.loc[(sweep["threshold"] - 0.10).abs().idxmin()]
    R["val_n"] = len(val)
    R["val_nC"] = int(val.is_C.sum())
    R["val_nS"] = int((~val.is_C).sum())
    R["val_best"] = best.to_dict()
    R["val_at10"] = at10.to_dict()
    R["median_albedo"] = (sbdb[sbdb["albedo"].notna() & sbdb["has_type"]]
                          .groupby("spec_class")["albedo"].agg(["median", "count"]).round(3))
    # Mann-Whitney: are C and S albedos different populations?
    R["mw_p"] = float(stats.mannwhitneyu(val.loc[val.is_C, "albedo"],
                                         val.loc[~val.is_C, "albedo"]).pvalue)
    # Use the literature cut-off (0.10) rather than the tuned one: a threshold
    # tuned on the same 121 asteroids it is scored on would overstate accuracy.
    threshold = X_DARK_MAX
    R["threshold"] = threshold

    # Inferred class for every accessible asteroid
    def inferred(r):
        if isinstance(r["spec_class"], str) and r["spec_class"] not in ("Other", "X, albedo unknown"):
            if r["spec_class"] == "Enstatite (E)":
                return "Rocky or metal (bright)"
            if r["spec_class"] == "Rocky (S)":
                return "Rocky or metal (bright)"
            return r["spec_class"]
        if pd.notna(r["albedo"]):
            return "Carbon-rich (C)" if r["albedo"] < threshold else "Rocky or metal (bright)"
        return "Unknown"
    acc["inferred"] = acc.apply(inferred, axis=1)
    sbdb_inf = sbdb.copy()
    sbdb_inf["inferred"] = sbdb_inf.apply(inferred, axis=1)
    R["acc_inferred_counts"] = acc["inferred"].value_counts()
    R["all_inferred_counts"] = sbdb_inf["inferred"].value_counts()

    # 3. Correlations with delta-v -----------------------------------------
    acc["abs_a_minus_1"] = (acc["a"] - 1).abs()
    feats = {
        "Inclination (i)": "i",
        "Eccentricity (e)": "e",
        "Distance from Earth-like orbit |a - 1 AU|": "abs_a_minus_1",
        "Min. distance to Earth's orbit (MOID)": "moid",
        "Size (diameter)": "D_m",
        "Albedo": "albedo",
    }
    corr = []
    for label, col in feats.items():
        sub = acc[["dv", col]].dropna()
        rho, p = stats.spearmanr(sub["dv"], sub[col])
        corr.append((label, rho, p, len(sub)))
    R["corr"] = pd.DataFrame(corr, columns=["feature", "spearman_rho", "p_value", "n"])

    # Size vs dv split: are the most accessible asteroids small?
    lo = acc[acc["dv"] < DV_LUNAR_SURFACE_RT]
    hi = acc[acc["dv"] >= DV_LUNAR_SURFACE_RT]
    R["median_D_lowdv"] = float(lo["D_m"].median())
    R["median_D_highdv"] = float(hi["D_m"].median())
    R["frac_type_lowdv"] = float(lo["has_type"].fillna(False).astype(bool).mean())
    R["frac_albedo_lowdv"] = float(lo["albedo"].notna().mean())

    # 4. Moon comparison ---------------------------------------------------
    R["n_below_surface"] = int((acc["dv"] < DV_LUNAR_SURFACE_RT).sum())
    R["n_below_orbit"] = int((acc["dv"] < DV_LUNAR_ORBIT_RT).sum())
    R["n_below_surface_known_comp"] = int(((acc["dv"] < DV_LUNAR_SURFACE_RT)
                                           & (acc["inferred"] != "Unknown")).sum())
    R["n_below_surface_D100"] = int(((acc["dv"] < DV_LUNAR_SURFACE_RT) & (acc["D_m"] >= 100)).sum())

    # 5. Metal candidates --------------------------------------------------
    metal = acc[acc["inferred"].eq("Metal-rich (M)")].copy()
    bright_lowdv = acc[acc["inferred"].eq("Rocky or metal (bright)")
                       & (acc["dv"] < DV_LUNAR_SURFACE_RT)].copy()
    R["metal"] = metal
    R["bright_lowdv"] = bright_lowdv
    known = {"6178": "1986 DA", "843295": "2016 ED85", "436724": "2011 UW158"}
    R["known_metal_check"] = {
        name: (float(acc.loc[acc.pdes == p, "dv"].iloc[0]) if (acc.pdes == p).any() else None)
        for p, name in known.items()}
    R["known_metal_sbdb"] = sbdb[sbdb.pdes.isin(known)][
        ["full_name", "spec_B", "spec_T", "albedo", "diameter", "a", "e", "i"]]

    # 6. Iron estimate -----------------------------------------------------
    def iron_t(D_m, dens):
        vol = 4 / 3 * math.pi * (D_m / 2) ** 3
        return vol * dens * METAL_FRAC_M * FE_IN_METAL / 1000  # tonnes
    R["iron_100m"] = {k: iron_t(100, v) for k, v in DENSITY_M.items()}
    for k, v in DENSITY_M.items():
        metal[f"iron_t_{k}"] = metal["D_m"].apply(lambda d: iron_t(d, v) if pd.notna(d) else np.nan)
    R["metal"] = metal

    return acc, sbdb, sweep, val, R


# ---------------------------------------------------------------------------
# Charts (no titles: captions live on the slide / in the report)
# ---------------------------------------------------------------------------
def _save(fig, out, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def _category_axis(ax):
    ax.tick_params(axis="y", which="both", length=0, right=False, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def chart_data_gap(R, out):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    labels = [f"All near-Earth\nasteroids\n($n$ = {R['n_nea']:,})",
              f"Accessible\n(NASA NHATS)\n($n$ = {R['n_acc']:,})"]
    pct_alb = [R["n_nea_albedo"] / R["n_nea"] * 100, R["n_acc_albedo"] / R["n_acc"] * 100]
    pct_typ = [R["n_nea_type"] / R["n_nea"] * 100, R["n_acc_type"] / R["n_acc"] * 100]
    y = np.arange(2)
    h = 0.36
    b1 = ax.barh(y - h / 2 - 0.02, pct_alb, height=h, color=BLUE, label="Albedo measured")
    b2 = ax.barh(y + h / 2 + 0.02, pct_typ, height=h, color=ORANGE, label="Spectral type measured")
    for bars, vals, counts in [(b1, pct_alb, [R["n_nea_albedo"], R["n_acc_albedo"]]),
                               (b2, pct_typ, [R["n_nea_type"], R["n_acc_type"]])]:
        for bar, v, c in zip(bars, vals, counts):
            ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}% ({c:,})", va="center", fontsize=14)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.2)
    ax.set_xlabel("Asteroids with composition data (%)")
    _category_axis(ax)
    ax.legend(loc="lower right")
    _save(fig, out, "fig1_data_gap")


def chart_albedo_validation(sbdb, R, out):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    groups = [("Carbon-rich (C)", BLUE), ("Rocky (S)", ORANGE), ("Metal-rich (M)*", AQUA)]
    keys = ["Carbon-rich (C)", "Rocky (S)", "Metal-rich (M)"]
    rng = np.random.default_rng(7)
    d = sbdb[sbdb["albedo"].notna() & sbdb["has_type"]]
    for k, ((g, col), key) in enumerate(zip(groups, keys)):
        vals = d.loc[d["spec_class"] == key, "albedo"].values
        yj = k + rng.uniform(-0.2, 0.2, len(vals))
        ax.scatter(vals, yj, s=42, color=col, edgecolor=SURFACE, linewidth=0.7, zorder=3)
        if len(vals):
            med = np.median(vals)
            ax.plot([med, med], [k - 0.32, k + 0.32], color=INK, lw=2.2, zorder=4)
            ax.text(med, k - 0.38, f"median {med:.2f}", ha="center", va="bottom", fontsize=13)
    th = R["threshold"]
    ax.axvline(th, color=INK_2, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.text(th * 1.03, -0.6, f"cut-off {th:.2f}", ha="left", va="top", fontsize=13, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.10", "0.20", "0.50"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(3), [g for g, _ in groups])
    ax.set_ylim(2.75, -0.65)
    ax.set_xlabel("Geometric albedo, $p_V$")
    _category_axis(ax)
    fig.text(0.015, 0.012, "*M-types are partly defined by albedo; shown for comparison, not tested.",
             fontsize=11.5, color=INK_2)
    _save(fig, out, "fig2_albedo_test", rect=(0, 0.04, 1, 1))


def chart_accessible_map(acc, R, out):
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    order = [("Unknown", GRAY, 9, 0.45, "Composition unknown"),
             ("Carbon-rich (C)", BLUE, 40, 1, "Carbon-rich (dark)"),
             ("Rocky or metal (bright)", ORANGE, 40, 1, "Rocky or metal (bright)"),
             ("Metal-rich (M)", AQUA, 110, 1, "Likely metal-rich (M)")]
    for g, col, s, a, lab in order:
        d = acc[acc["inferred"] == g]
        ax.scatter(d["D_m"], d["dv"], s=s, color=col, alpha=a, label=f"{lab} ({len(d):,})",
                   edgecolor=SURFACE if g != "Unknown" else "none", linewidth=0.7,
                   zorder=2 if g == "Unknown" else 3, rasterized=(g == "Unknown"))
    for yv, txt in [(DV_LUNAR_SURFACE_RT, "Lunar surface round trip"),
                    (DV_LUNAR_ORBIT_RT, "Low lunar orbit round trip")]:
        ax.axhline(yv, color=INK_2, lw=1.2, ls=(0, (5, 3)), zorder=1)
        ax.text(0.012, yv + 0.12, txt, transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=13.5, color=INK_2,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    for _, r in R["metal"].iterrows():
        name = re.sub(r"^\d+\s*", "", r["full_name"]).strip("() ")
        left = r["D_m"] < 400
        ax.annotate(name, (r["D_m"], r["dv"]), xytext=(-11 if left else 11, 0),
                    textcoords="offset points", fontsize=14, weight="bold",
                    ha="right" if left else "left", va="center",
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xlim(0.4, 4000)
    ax.set_ylim(3, 12.3)
    ax.set_xlabel("Diameter, $D$ (m)")
    ax.set_ylabel(r"Round-trip $\Delta v$ from low Earth orbit (km s$^{-1}$)")
    ax.legend(loc="lower right", title=f"NASA NHATS accessible list ($n$ = {R['n_acc']:,})",
              markerscale=1.0)
    _save(fig, out, "fig3_accessible_map")


def chart_correlations(R, out):
    c = R["corr"].sort_values("spearman_rho").copy()
    pretty = {"Inclination (i)": "Inclination, $i$",
              "Eccentricity (e)": "Eccentricity, $e$",
              "Distance from Earth-like orbit |a - 1 AU|": r"$|a - 1\,\mathrm{AU}|$",
              "Min. distance to Earth's orbit (MOID)": "Earth MOID",
              "Size (diameter)": "Diameter, $D$",
              "Albedo": "Albedo, $p_V$"}
    c["label"] = c["feature"].map(pretty)
    fig, ax = plt.subplots(figsize=FIG_HALF)
    cols = [(BLUE if v < 0 else RED_DIV) if p < 0.05 else BAR_NEUTRAL
            for v, p in zip(c["spearman_rho"], c["p_value"])]
    bars = ax.barh(c["label"], c["spearman_rho"], color=cols, height=0.62)
    for bar, (_, r) in zip(bars, c.iterrows()):
        x = bar.get_width()
        txt = f"{x:+.2f}" + ("" if r["p_value"] < 0.05 else " (n.s.)")
        ax.text(x + (0.02 if x >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, txt,
                va="center", ha="left" if x >= 0 else "right", fontsize=14)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_xlim(-0.5, 0.9)
    ax.set_xlabel(r"Spearman's $\rho$ with $\Delta v$")
    _category_axis(ax)
    _save(fig, out, "fig4_deltav_correlations")


def chart_energy(out):
    """Stacked bars: energy to strip oxygen (smelt) + energy to melt, per tonne of iron."""
    melt = ENERGY["melt"]
    rows = [("Metallic asteroid", 0.0, melt),
            ("Lunar iron (FeO),\ntheoretical minimum", ENERGY["smelt_floor_FeO"], melt),
            ("Earth ore (Fe$_2$O$_3$),\ntheoretical minimum", ENERGY["smelt_floor_Fe2O3"], melt)]
    real = ("Steelmaking on Earth,\n2024 global average", ENERGY["steel_route_avg"])
    labels = [r[0] for r in rows] + [real[0]]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIG_HALF)
    h = 0.6
    gap = 0.12  # GJ/t of surface between stacked segments (2px-style spacer)
    smelt_v = [r[1] for r in rows]
    melt_v = [r[2] for r in rows]
    ax.barh(y[:3], smelt_v, height=h, color=ORANGE, label="Strip oxygen (smelt)")
    ax.barh(y[:3], [m - (gap if sv > 0 else 0) for m, sv in zip(melt_v, smelt_v)], height=h,
            left=[sv + (gap if sv > 0 else 0) for sv in smelt_v], color=BLUE, label="Melt to shape")
    ax.barh(y[3], real[1], height=h, color=BAR_NEUTRAL, label="Real process, all steps")
    totals = [sv + mv for sv, mv in zip(smelt_v, melt_v)] + [real[1]]
    for yi, t in zip(y, totals):
        ax.text(t + 0.35, yi, f"{t:.1f}", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(melt + 2.2, y[0], "smelting step: 0", va="center", fontsize=13, color=INK_2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.set_xlabel(r"Energy (GJ t$^{-1}$ Fe)")
    _category_axis(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.24))
    _save(fig, out, "fig5_energy")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_summary(R, sweep, out):
    L = []
    p = L.append
    p("# Asteroid feedstock analysis: results\n")
    p("## 1. Data gap")
    p(f"- Known near-Earth asteroids: {R['n_nea']:,}")
    p(f"- With measured albedo: {R['n_nea_albedo']:,} ({R['n_nea_albedo']/R['n_nea']*100:.1f}%)")
    p(f"- With spectral type: {R['n_nea_type']:,} ({R['n_nea_type']/R['n_nea']*100:.2f}%)")
    p(f"- Accessible (NHATS, <= 12 km/s round trip): {R['n_acc']:,} "
      f"(matched to SBDB: {R['n_matched']:,})")
    p(f"- Accessible with albedo: {R['n_acc_albedo']:,} ({R['n_acc_albedo']/R['n_acc']*100:.1f}%); "
      f"with spectral type: {R['n_acc_type']:,} ({R['n_acc_type']/R['n_acc']*100:.2f}%)")
    p(f"- Median estimated diameter of accessible asteroids: {R['median_D_acc_est']:.0f} m\n")

    p("## 2. Albedo test (C vs S, validated on asteroids with known spectral type)")
    p(f"- Validation set: {R['val_n']} asteroids ({R['val_nC']} carbon-rich, {R['val_nS']} rocky)")
    b, t = R["val_best"], R["val_at10"]
    p(f"- USED: literature cut-off albedo < 0.10 -> carbon-rich. C recall {t['C_recall']*100:.0f}%, "
      f"S recall {t['S_recall']*100:.0f}%, balanced accuracy {t['balanced']*100:.0f}%, "
      f"raw accuracy {t['accuracy']*100:.0f}%")
    p(f"- For reference only, cut-off tuned on this same set: {b['threshold']:.3f} "
      f"(balanced accuracy {b['balanced']*100:.0f}%); not used because it is fit to the test data")
    p(f"- C and S albedos are different populations (Mann-Whitney p = {R['mw_p']:.1e})")
    p("- Median albedo by class:\n")
    p(R["median_albedo"].to_markdown())
    p("\n- Limitation: albedo cannot separate rocky (S) from metal-rich (M); both are 'bright'.\n")

    p("## 3. What predicts delta-v (accessible asteroids, Spearman)")
    p(R["corr"].assign(spearman_rho=lambda d: d.spearman_rho.round(3),
                       p_value=lambda d: d.p_value.map(lambda v: f"{v:.1e}")).to_markdown(index=False))
    p(f"\n- Median diameter, delta-v < 9 km/s: {R['median_D_lowdv']:.0f} m vs >= 9 km/s: "
      f"{R['median_D_highdv']:.0f} m")
    p(f"- Of asteroids cheaper than a lunar surface round trip, {R['frac_albedo_lowdv']*100:.1f}% have "
      f"an albedo and {R['frac_type_lowdv']*100:.1f}% have a spectral type\n")

    p("## 4. Moon comparison (round trip from low Earth orbit)")
    p(f"- Cheaper than a lunar SURFACE round trip (< {DV_LUNAR_SURFACE_RT} km/s): {R['n_below_surface']:,}")
    p(f"- Cheaper than a low lunar ORBIT round trip (< {DV_LUNAR_ORBIT_RT} km/s): {R['n_below_orbit']:,}")
    p(f"- Of the < 9 km/s group, {R['n_below_surface_D100']:,} are 100 m or larger, and "
      f"{R['n_below_surface_known_comp']:,} have any composition data\n")

    p("## 5. Metal-rich candidates on the accessible list")
    m = R["metal"]
    if len(m):
        cols = ["full_name", "spec_B", "spec_T", "albedo", "D_m", "dv", "dv_trip_days",
                "iron_t_low", "iron_t_mid", "iron_t_high"]
        p(m[cols].sort_values("dv").round(3).to_markdown(index=False))
    else:
        p("- None")
    p(f"\n- Known metal-rich NEAs on the NHATS list (delta-v, km/s): {R['known_metal_check']}")
    p("\n" + R["known_metal_sbdb"].to_markdown(index=False))
    p(f"\n- Bright (rocky OR metal) asteroids cheaper than a lunar surface round trip: "
      f"{len(R['bright_lowdv'])}, of which {int(R['bright_lowdv']['spec_class'].isna().sum())} have no "
      f"spectral type yet. Those are the pool where follow-up spectra could find metal.\n")
    bl = R["bright_lowdv"].sort_values("dv")[["full_name", "spec_B", "albedo", "D_m", "dv"]].round(3)
    p(bl.to_markdown(index=False))

    p("\n## 6. Iron in a 100 m metal-rich asteroid")
    for k, v in R["iron_100m"].items():
        p(f"- {k} density ({DENSITY_M[k]} kg/m3): {v/1e6:.2f} million t iron "
          f"= {v*1000/ISS_MASS_KG:,.0f} ISS masses")
    p(f"- Assumes {METAL_FRAC_M:.0%} metal (Sanchez et al. 2021) and {FE_IN_METAL:.0%} iron in the metal\n")

    p("## 7. Energy per tonne of iron")
    for k, v in ENERGY.items():
        p(f"- {k}: {v:.2f} GJ/t")
    (out / "results.md").write_text("\n".join(L))
    sweep.round(4).to_csv(out / "albedo_threshold_sweep.csv", index=False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sbdb", default="data/sbdb.json")
    ap.add_argument("--nhats", default="data/nhats.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fetch", action="store_true", help="download fresh data from NASA/JPL first")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.fetch:
        Path(a.sbdb).parent.mkdir(parents=True, exist_ok=True)
        fetch(SBDB_URL, a.sbdb)
        fetch(NHATS_URL, a.nhats)

    sbdb, nhats = load_sbdb(a.sbdb), load_nhats(a.nhats)
    acc, sbdb, sweep, val, R = analyse(sbdb, nhats)

    chart_data_gap(R, out)
    chart_albedo_validation(sbdb, R, out)
    chart_accessible_map(acc, R, out)
    chart_correlations(R, out)
    chart_energy(out)
    write_summary(R, sweep, out)
    acc.to_csv(out / "accessible_asteroids_classified.csv", index=False)
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
```