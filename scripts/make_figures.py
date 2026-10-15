"""
Figures for the v2 manuscript, generated only from committed result files.

Run (web venv): python scripts/make_figures.py
Writes figures/fig*.png (300 dpi)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "figures"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"physics": "#7f7f7f", "physics_lin": "#bcbd22", "ml": "#1f77b4", "hybrid": "#d62728"}
LABEL = {"physics": "Physics (edge theory)", "physics_lin": "Physics, linear calib.",
         "ml": "ML (Magpie + T)", "hybrid": "Hybrid (physics + ML residual)"}


def fig_data():
    d = pd.read_csv("data/processed/rhea_strength.csv")
    d = d[d.physics_domain]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for h, c in (("train", "#1f77b4"), ("future", "#ff7f0e")):
        q = d[d.holdout == h]
        ax[0].scatter(q.T_C, q.YS_MPa, s=10, alpha=0.7, c=c,
                      label=f"{h} ({q.key.nunique()} alloys, {len(q)} tests)")
    ax[0].set_xlabel("Test temperature (°C)"); ax[0].set_ylabel("Yield strength (MPa)"); ax[0].legend(frameon=False, fontsize=7)
    el = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
    a = d.drop_duplicates("key")
    ax[1].bar(el, [(a[e] > 0).sum() for e in el], color="#555")
    ax[1].set_ylabel("Alloys containing element")
    ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/fig1_dataset.png"); plt.close(fig)


def fig_parity():
    P = pd.read_csv("results/strength_predictions.csv")
    fig, ax = plt.subplots(1, 3, figsize=(8.5, 2.9), sharex=True, sharey=True)
    for i, m in enumerate(["physics", "ml", "hybrid"]):
        q = P[(P.split == "grouped") & (P.model == m)]
        sc = ax[i].scatter(q.YS_MPa, q.pred, c=q.T_C, s=9, cmap="viridis", vmin=0, vmax=1600)
        lim = [0, 2600]; ax[i].plot(lim, lim, "k--", lw=0.8); ax[i].set_xlim(lim); ax[i].set_ylim(lim)
        e = q.pred - q.YS_MPa
        r2 = 1 - (e ** 2).sum() / ((q.YS_MPa - q.YS_MPa.mean()) ** 2).sum()
        ax[i].set_title(f"({'abc'[i]}) {LABEL[m]}\nR² = {r2:.2f}, MAE = {e.abs().mean():.0f} MPa", fontsize=8, loc="left")
        ax[i].set_xlabel("Measured (MPa)")
    ax[0].set_ylabel("Predicted (MPa)")
    fig.colorbar(sc, ax=ax, label="T (°C)", shrink=0.85)
    fig.savefig(f"{OUT}/fig3_parity_grouped.png"); plt.close(fig)


def fig_models():
    R = pd.read_csv("results/strength_models.csv")
    splits = ["grouped", "system", "future"]
    names = {"grouped": "Unseen alloys\n(grouped CV)", "system": "Unseen systems\n(system CV)",
             "future": "2019 alloys\n(future holdout)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8))
    w = 0.2
    for j, m in enumerate(["physics", "physics_lin", "ml", "hybrid"]):
        q = R[R.model == m].set_index("split").loc[splits]
        x = np.arange(3) + (j - 1.5) * w
        ax[0].bar(x, q.R2, w, color=C[m], label=LABEL[m]); ax[1].bar(x, q.MAE, w, color=C[m])
    for a, t in zip(ax, ["R²", "MAE (MPa)"]):
        a.set_xticks(range(3)); a.set_xticklabels([names[s] for s in splits], fontsize=7); a.set_ylabel(t)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=7, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.08)); ax[0].set_ylim(0, 0.85)
    ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/fig4_model_comparison.png"); plt.close(fig)


def fig_residual_T():
    P = pd.read_csv("results/strength_predictions.csv")
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for m in ("physics", "hybrid"):
        q = P[(P.split == "grouped") & (P.model == m)].copy()
        q["bin"] = pd.cut(q.T_C, [-300, 30, 600, 900, 1200, 1700])
        g = q.assign(err=q.pred - q.YS_MPa).groupby("bin", observed=True).err
        x = [b.mid if b.left > -300 else 25 for b in g.mean().index]
        ax.errorbar(x, g.mean(), yerr=g.std(), fmt="o-", color=C[m], label=LABEL[m], capsize=2, ms=4)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Test temperature (°C)"); ax.set_ylabel("Prediction − measured (MPa)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(f"{OUT}/fig5_residual_vs_T.png"); plt.close(fig)


def fig_chgnet_benchmark():
    b = pd.read_csv("results/chgnet_benchmark.csv")
    g = b.groupby("alloy", sort=False).agg(err=("a_err_pct_vs_exp", "mean"), sd=("a_err_pct_vs_exp", "std"),
                                            dv=("vol_change_pct_vs_rom", "mean"))
    fig, ax = plt.subplots(figsize=(4, 2.6))
    x = np.arange(len(g))
    ax.bar(x - 0.18, g.err, 0.36, yerr=g.sd, label="a vs experiment (%)", color="#1f77b4", capsize=2)
    ax.bar(x + 0.18, g.dv, 0.36, label="V vs Vegard (%)", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(g.index, fontsize=7)
    ax.legend(frameon=False, fontsize=7); ax.set_ylabel("Deviation (%)")
    fig.savefig(f"{OUT}/fig2_chgnet_benchmark.png"); plt.close(fig)


def fig_pareto():
    path = "results/design_pareto.csv"
    if not os.path.exists(path):
        return
    p = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    sc = ax.scatter(p.VEC, p.spec_lcb_1000C, c=p.density, cmap="plasma", s=18)
    ax.axvspan(p.VEC.min() - 0.1, 4.5, color="#2ca02c", alpha=0.08, lw=0)
    ax.set_xlabel("VEC (ductility proxy; ≤ 4.4 ductile, Sheikh 2016)")
    ax.set_ylabel("Lower 90% bound, σy(1000 °C)/ρ\n(GPa cm³ g⁻¹)")
    fig.colorbar(sc, label="Density (g cm⁻³)")
    fig.savefig(f"{OUT}/fig7_pareto.png"); plt.close(fig)


def fig_mlip_elements():
    path = "results/mlip_elements.csv"
    if not os.path.exists(path):
        return
    E = pd.read_csv(path)
    pots = E.potential.unique()
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    el = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
    w = 0.8 / len(pots)
    for j, pot in enumerate(pots):
        q = E[E.potential == pot].set_index("element").loc[el]
        x = np.arange(len(el)) + (j - (len(pots) - 1) / 2) * w
        ax[0].bar(x, 100 * (q.a - q.a_ref) / q.a_ref, w, label=pot)
        cp, cpr = (q.C11 - q.C12) / 2, (q.C11_ref - q.C12_ref) / 2
        ax[1].bar(x, cp - cpr, w, label=pot)
    for a in ax:
        a.set_xticks(range(len(el))); a.set_xticklabels(el); a.axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("a error vs reference (%)"); ax[1].set_ylabel("C′ error vs reference (GPa)")
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
    fig.savefig(f"{OUT}/fig6_mlip_elements.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_data, fig_chgnet_benchmark, fig_parity, fig_models, fig_residual_T, fig_mlip_elements, fig_pareto):
        f(); print("ok", f.__name__)
