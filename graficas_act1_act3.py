#!/usr/bin/env python3
"""
Gráficas del informe de Interferometría:
  - Actividad 1: d_N vs N (ajuste b=0)
  - Actividad 3: ΔP vs N (ajuste b=0) y n vs P

Requisitos: numpy, matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def wls_b0(x, y, sy):
    """Ajuste ponderado y = m x (b = 0)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sy = np.asarray(sy, float)
    w = 1.0 / sy**2
    m = np.sum(w * x * y) / np.sum(w * x * x)
    sm = np.sqrt(1.0 / np.sum(w * x * x))
    yfit = m * x
    return m, sm, yfit


def plot_fit_residuals(x, y, sy, yfit, xlabel, ylabel, title, fname, text=None, xerr=None):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )
    ax.errorbar(
        x, y, yerr=sy, xerr=xerr, fmt="o", ms=5, color="#1f77b4",
        ecolor="#1f77b4", capsize=3, label="Datos",
    )
    order = np.argsort(x)
    ax.plot(x[order], yfit[order], "-", color="#d62728", lw=1.8, label="Ajuste")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    if text:
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )

    rw = (y - yfit) / sy
    axr.axhline(0, color="k", lw=0.8)
    axr.axhline(1, color="gray", ls="--", lw=0.7)
    axr.axhline(-1, color="gray", ls="--", lw=0.7)
    axr.errorbar(
        x, rw, yerr=np.ones_like(rw), fmt="o", ms=5, color="#2ca02c",
        ecolor="#2ca02c", capsize=3,
    )
    axr.set_xlabel(xlabel)
    axr.set_ylabel(r"Residuales ponderados $(y-\hat{y})/\sigma$")
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {OUT / fname}")


def actividad_1():
    # d_N acumulado (μm) y N acumulado (mínimos por tramo sumados)
    d_interval = np.array([10, 20, 30, 40, 50, 53, 56, 57, 59, 61, 63, 64, 65], float)
    delta_N = np.array([28, 21, 3, 43, 30, 5, 10, 5, 8, 5, 5, 10, 6], float)
    N = np.cumsum(delta_N)
    d = d_interval
    sd = np.full_like(d, 0.5)  # μm

    m, sm, yfit = wls_b0(N, d, sd)
    lam_nm = 2 * m * 1e3
    slam_nm = 2 * sm * 1e3

    plot_fit_residuals(
        N, d, sd, yfit,
        r"$N$ (mínimos acumulados)",
        r"$d_N$ ($\mu$m)",
        r"Actividad 1: $d_N$ vs $N$ (ajuste $b=0$)",
        "act1_dN_vs_N_b0.png",
        text=(
            fr"$m={m:.4f}\pm{sm:.4f}\,\mu$m" + "\n"
            + fr"$\lambda=2m={lam_nm:.1f}\pm{slam_nm:.1f}$ nm"
        ),
    )
    print(f"Act.1 b=0: m = {m:.6f} ± {sm:.6f} μm → λ = {lam_nm:.1f} ± {slam_nm:.1f} nm")


def actividad_3():
    # Presión acumulada (cmHg) y N acumulado
    P = np.array([6, 10, 14, 18, 24, 30, 36], float)
    N = np.array([1, 2, 3, 4, 6, 8, 10], float)
    sP = np.full_like(P, 0.05)  # cmHg

    m, sm, yfit = wls_b0(N, P, sP)

    plot_fit_residuals(
        N, P, sP, yfit,
        r"$N$ (número de mínimos)",
        r"$\Delta P$ (cmHg)",
        r"Actividad 3: $\Delta P$ vs $N$ (ajuste $b=0$)",
        "act3_dP_vs_N_b0.png",
        text=fr"$m=\Delta P/N={m:.3f}\pm{sm:.3f}$ cmHg",
    )
    print(f"Act.3 b=0: m = {m:.5f} ± {sm:.5f} cmHg")

    # n vs P
    lam_m = 633e-9
    t_m = 3.24e-2
    cmHg_to_Pa = 1333.22
    m_Pa = m * cmHg_to_Pa
    dn_dP = lam_m / (2 * t_m * m_Pa)

    P_Pa = P * cmHg_to_Pa
    sP_Pa = sP * cmHg_to_Pa
    n = 1 + P_Pa * dn_dP
    # incertidumbre formal desde m (sin σ_N); solo para barras de la figura
    sm_Pa = sm * cmHg_to_Pa
    sdn = abs(dn_dP * (sm_Pa / m_Pa))
    sn = np.sqrt((P_Pa * sdn) ** 2 + (dn_dP * sP_Pa) ** 2)

    n_bog = 1 + 56.0 * cmHg_to_Pa * dn_dP

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )
    P_line = np.linspace(0, 60, 200)
    n_line = 1 + P_line * cmHg_to_Pa * dn_dP
    ax.errorbar(
        P, n, yerr=sn, xerr=sP, fmt="o", ms=5, color="#1f77b4",
        ecolor="#1f77b4", capsize=3, label="Datos",
    )
    ax.plot(P_line, n_line, "-", color="#d62728", lw=1.8, label=r"$n=1+P\,\mathrm{d}n/\mathrm{d}P$")
    ax.plot([56], [n_bog], "s", color="#ff7f0e", ms=8, label=fr"$n(56\,\mathrm{{cmHg}})={n_bog:.6f}$")
    ax.set_ylabel(r"$n$")
    ax.set_title(r"Actividad 3: $n$ del aire vs $P$ ($\lambda=633$ nm)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    slope_i = (n - 1) / P_Pa
    rw = (slope_i - dn_dP) / (sn / P_Pa)
    axr.axhline(0, color="k", lw=0.8)
    axr.axhline(1, color="gray", ls="--", lw=0.7)
    axr.axhline(-1, color="gray", ls="--", lw=0.7)
    axr.errorbar(P, rw, yerr=1, fmt="o", ms=5, color="#2ca02c", ecolor="#2ca02c", capsize=3)
    axr.set_xlabel(r"$P$ (cmHg)")
    axr.set_ylabel("Residuales ponderados")
    axr.grid(True, alpha=0.3)
    fig.savefig(OUT / "act3_n_vs_P.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {OUT / 'act3_n_vs_P.png'}")
    print(f"dn/dP = {dn_dP:.6e} /Pa ; n(56 cmHg) = {n_bog:.7f}")


if __name__ == "__main__":
    actividad_1()
    actividad_3()
