"""
qpce.plots
============

The two physics figures.  Both are drawn from the DEPLOYED generator on
fresh germ draws and scored against the HELD-OUT test set.

Figure 1  Rank-correlation heatmaps: Geant4, model, and the residual.
          Spearman rather than Pearson, because rank correlation is
          invariant under the per-pixel unit conversion and therefore
          isolates dependence from marginal shape.  Diverging blue-to-red on
          a neutral midpoint, symmetric about zero, one shared scale across
          all three panels so the residual panel is comparable rather than
          auto-scaled into looking dramatic.  Residual cells carry the
          bootstrap z-score, so a reader can see what is outside sampling
          noise.

Figure 2  The n marginal pdfs with ratio panels, LHC convention: the points
          carry the reference error, the band about unity carries the model
          error, never both on the same object.  Error bars are exact
          multinomial standard errors at fixed bin edges -- the
          infinite-resample limit of the histogram bootstrap, with no Monte
          Carlo noise and no runtime cost.

          In QPCE-F these marginals ARE produced by the circuit, so the
          figure tests the device.  In the published two-stage pipeline the
          same figure was a calibration diagnostic that pure uniform noise
          passed (W1 0.00221 against a trained model's 0.00148); that
          caveat does not apply here, and its absence is the point.

Bins are fixed a priori.  Tuning binning until a pixel passes is p-hacking,
so ``bins`` is a parameter of the figure, not of the fit.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap          # noqa: E402
from scipy.stats import wasserstein_distance                   # noqa: E402

from .metrics import spearman_matrix                           # noqa: E402

DIVERGING = LinearSegmentedColormap.from_list(
    "qpce_div", ["#2a78d6", "#8fb6e8", "#f0efec", "#eda2a2", "#d03b3b"])
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8981"
C_DATA, C_MODEL = "#0b0b0b", "#2a78d6"

STYLE = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
def figure_correlations(Y_ref, Y_gen, out, n_boot=300, seed=0,
                        title="Rank-correlation structure",
                        name="correlations", model_label="QPCE-F"):
    plt.rcParams.update(STYLE)
    n = Y_ref.shape[1]
    iu = np.triu_indices(n, 1)
    Cd, Cm = spearman_matrix(Y_ref), spearman_matrix(Y_gen)
    rng = np.random.default_rng(seed)
    boot = np.stack([spearman_matrix(Y_ref[rng.integers(0, len(Y_ref),
                                                        len(Y_ref))])
                     for _ in range(n_boot)])
    Z = (Cm - Cd) / np.maximum(boot.std(0, ddof=1), 1e-12)
    np.fill_diagonal(Z, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9),
                             gridspec_kw={"wspace": 0.28})
    panels = [(Cd, "Geant4 reference"), (Cm, f"{model_label} (deployed)"),
              (Cm - Cd, "model − reference")]
    for ax, (Mx, ttl) in zip(axes, panels):
        im = ax.imshow(Mx, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_title(ttl, fontsize=10, pad=8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel" if ax is axes[0] else "")
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        resid = ttl.startswith("model")
        for i in range(n):
            for j in range(n):
                if resid and i == j:
                    continue
                txt = f"{Z[i, j]:+.0f}" if resid else \
                    f"{Mx[i, j]:.2f}".replace("0.", ".")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                        color="white" if abs(Mx[i, j]) > 0.55 else INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.015)
    cb.set_label("Spearman rank correlation", fontsize=9)
    cb.outline.set_edgecolor(INK2)
    cb.outline.set_linewidth(0.8)
    fig.suptitle(title, fontsize=9.5, color=INK2, y=1.005)
    axes[2].text(0.5, -0.22, "cell labels: bootstrap z-score of the residual",
                 transform=axes[2].transAxes, ha="center", fontsize=7.5,
                 color=MUTED)
    _save(fig, out, name)
    return {"frobenius": float(np.linalg.norm(Cm - Cd)),
            "mean_abs_rho_model": float(np.abs(Cm[iu]).mean()),
            "mean_abs_rho_data": float(np.abs(Cd[iu]).mean()),
            "max_abs_z": float(np.abs(Z).max()),
            "frac_beyond_2sigma": float((np.abs(Z[iu]) > 2).mean())}


# =====================================================================
def figure_marginals(Y_ref, Y_gen, out, bins=30, min_count=10,
                     title=None):
    plt.rcParams.update(STYLE)
    n = Y_ref.shape[1]
    rows = int(np.ceil(n / 4))
    fig = plt.figure(figsize=(11.2, 3.1 * rows))
    gs = fig.add_gridspec(2 * rows, 4, hspace=0.55, wspace=0.28,
                          height_ratios=[3, 1.15] * rows)
    stats = []
    for j in range(n):
        r, c = divmod(j, 4)
        ax = fig.add_subplot(gs[2 * r, c])
        axr = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        edges = np.histogram_bin_edges(
            np.concatenate([Y_ref[:, j], Y_gen[:, j]]), bins=bins)
        h = np.diff(edges)
        nr, ng = len(Y_ref), len(Y_gen)
        cr, _ = np.histogram(Y_ref[:, j], edges)
        cg, _ = np.histogram(Y_gen[:, j], edges)
        fr, fg = cr / (nr * h), cg / (ng * h)
        ser = np.sqrt(cr / nr * (1 - cr / nr) / nr) / h
        seg = np.sqrt(cg / ng * (1 - cg / ng) / ng) / h
        ctr = 0.5 * (edges[1:] + edges[:-1])

        ax.stairs(fg, edges, color=C_MODEL, lw=1.6, label="QPCE-F")
        ax.stairs(fg + seg, edges, baseline=fg - seg, fill=True,
                  color=C_MODEL, alpha=0.20, lw=0)
        ax.errorbar(ctr, fr, yerr=ser, fmt="o", ms=2.6, lw=0, elinewidth=0.8,
                    color=C_DATA, label="Geant4")
        ax.set_ylabel("density" if c == 0 else "", fontsize=8.5)
        ax.set_title(f"pixel {j}", fontsize=9, pad=4)
        ax.tick_params(labelbottom=False, labelsize=7.5)
        ax.set_ylim(0, max(np.max(fr + ser), np.max(fg + seg)) * 1.32)
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper right", handlelength=1.2,
                      borderaxespad=0.3)

        ok = (cr >= min_count) & (cg >= min_count)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ok, fr / np.where(fg > 0, fg, np.nan), np.nan)
            rerr = np.where(ok, ser / np.where(fg > 0, fg, np.nan), np.nan)
            band = np.where(ok, seg / np.where(fg > 0, fg, np.nan), np.nan)
        axr.axhline(1.0, color=INK2, lw=0.8)
        axr.stairs(1 + band, edges, baseline=1 - band, fill=True,
                   color=C_MODEL, alpha=0.20, lw=0)
        axr.errorbar(ctr, ratio, yerr=rerr, fmt="o", ms=2.4, lw=0,
                     elinewidth=0.8, color=C_DATA)
        axr.set_ylim(0.45, 1.55)
        axr.set_ylabel("ref/gen" if c == 0 else "", fontsize=7.5)
        axr.set_xlabel("intensity", fontsize=8.5)
        axr.tick_params(labelsize=7.5)

        chi2 = float(np.nansum((fr[ok] - fg[ok]) ** 2
                               / (ser[ok] ** 2 + seg[ok] ** 2)))
        ndf = max(int(ok.sum()) - 1, 1)
        w1 = float(wasserstein_distance(Y_ref[:, j], Y_gen[:, j]))
        stats.append({"pixel": j, "W1": w1, "chi2_ndf": chi2 / ndf, "ndf": ndf,
                      "bins_dropped": int((~ok).sum())})
        ax.text(0.035, 0.97, f"$W_1$ {w1:.4f}\n$\\chi^2$/ndf {chi2 / ndf:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color=MUTED,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    fig.suptitle(title or
                 "Per-pixel marginals — fully quantum readout "
                 "$Y_k = m_k + s_k\\langle Z_k\\rangle$:\nno recalibration, "
                 "no inverse PIT — these are produced by the circuit and "
                 "test it directly",
                 fontsize=9.5, color=INK2, y=0.995)
    _save(fig, out, "marginals")
    return stats


# =====================================================================
def figure_convergence(histories, out, name="convergence"):
    """Training curve.  One loss, so one curve per run."""
    plt.rcParams.update(STYLE)
    if not isinstance(histories, dict):
        histories = {"QPCE-F": histories}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for lab, h in histories.items():
        ax.plot(np.arange(1, len(h) + 1), h, lw=1.6, label=lab)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("loss evaluation")
    ax.set_ylabel("energy distance  $\\mathcal{E}(P,Q)$")
    ax.set_title("Training: one loss, exact adjoint gradients",
                 fontsize=10, pad=8)
    _save(fig, out, name)
    return {k: {"n_evaluations": len(v), "loss_initial": float(v[0]),
                "loss_final": float(v[-1])} for k, v in histories.items()}
