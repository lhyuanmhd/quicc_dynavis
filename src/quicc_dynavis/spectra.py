# src/quicc_dynavis/spectra.py
import os
import matplotlib.pyplot as plt
import numpy as np
from .io import read_single_spectrum,read_single_n_spectrum, avgSpectra_new 
from .timeseries_utils import input_params_from_path
import re

from .spectra_utils import calculate_flow_degree

def _load_km_spectra(
    folder_file,
    mode="single",
    start_time=None,
    stop_time=None,
    which="last",
):
    if mode == "single":
        kinetic = read_single_spectrum(
            folder_file,
            "kinetic",
            which=which,
        )
        magnetic = read_single_spectrum(
            folder_file,
            "magnetic",
            which=which,
        )

        time_k = kinetic[-1]
        time_m = magnetic[-1]

        label_k = f"Single t={time_k:.3E}"
        label_m = f"Single t={time_m:.3E}"

        return kinetic[:-1], magnetic[:-1], label_k, label_m

    if mode == "average":
        if start_time is None or stop_time is None:
            start_time = 0
            stop_time = 1

        kinetic = avgSpectra_new(
            folder_file,
            "kinetic",
            start_time,
            stop_time,
        )
        magnetic = avgSpectra_new(
            folder_file,
            "magnetic",
            start_time,
            stop_time,
        )

        return kinetic, magnetic, "average", "average"

    raise ValueError("mode must be 'single' or 'average'")

def plot_spectra_km(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True, ref_scaling=False):
    """
    Plot kinetic and magnetic spectra (single or averaged).
    Top row: kinetic, Bottom row: magnetic.
    """

    kinetic, magnetic, label_k, label_m = _load_km_spectra(
    folderFile,
    mode=mode,
    start_time=start_time,
    stop_time=stop_time,
    which=which,
    )

    (
        lk,
        mk,
        ltot_k,
        ltor_k,
        lpol_k,
        mtot_k,
        mtor_k,
        mpol_k,
    ) = kinetic

    flow_degree = calculate_flow_degree(
        lk,
        ltot_k,
    )

    #flow_degree_over_pi = flow_degree / np.pi

    (
        lm,
        mm,
        ltot_m,
        ltor_m,
        lpol_m,
        mtot_m,
        mtor_m,
        mpol_m,
    ) = magnetic

    # --- create figure ---
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), dpi=180)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, left=0.12, top=0.92, right=0.97, bottom=0.12)
    

  
    # l-spetra
    ax1, ax2 = axes[0]
    ax1.plot(lk[1:], ltot_k[1:], '.-', label='total')
    ax1.plot(lk[1:], ltor_k[1:], '.-', label='toroidal')
    ax1.loglog(lk[1:], lpol_k[1:], '.-', label='poloidal')
    ax1.set_xlabel('$l$')
    ax1.set_ylabel('Energy')
    #ax1.set_xscale('log'); 
    ax1.set_yscale('log')
    ax1.set_title(f'Kinetic $l$-spectrum({label_k})')
    ax1.legend()
    ax1.text(
        0.97,
        0.95,
        rf"$\ell_u={flow_degree:.2f}$",
        transform=ax1.transAxes,
        ha="right",
        va="top",
    )
    
    # m-spectra
    ax2.plot(mk, mtot_k, '.-', label='total')
    ax2.plot(mk, mtor_k, '.-', label='toroidal')
    ax2.loglog(mk, mpol_k, '.-', label='poloidal')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('Energy')
    #ax2.set_xscale('log'); 
    ax2.set_yscale('log')
    ax2.set_title(f'Kinetic $m$-spectrum({label_k})')


    # Magnetic (bottom row)
    ax3, ax4 = axes[1]
    lm = np.asarray(lm)
    Em = np.asarray(ltot_m)

    # ignore l=0 if present
    lm_use = lm[1:]
    Em_use = Em[1:]

    # index of spectral peak
    ipeak = np.argmax(Em_use)

    l_ref = lm_use[ipeak]
    E_ref = Em_use[ipeak]

    n = len(lm_use)
    sid = int(0.4 * n)
    eid = int(1 * n)

    ax3.plot(lm[1:], ltot_m[1:], '.-', label='total')
    ax3.plot(lm[1:], ltor_m[1:], '.-', label='toroidal')
    ax3.loglog(lm[1:], lpol_m[1:], '.-', label='poloidal')
    ax3.set_xlabel('$l$')
    ax3.set_ylabel('Energy')
    #ax3.set_xscale('log'); 
    ax3.set_yscale('log')
    ax3.set_title(f'Magnetic $l$-spectrum ({label_m})')
    ax3.legend()

    ax4.plot(mm, mtot_m, '.-', label='total')
    ax4.plot(mm, mtor_m, '.-', label='toroidal')
    ax4.loglog(mm, mpol_m, '.-', label='poloidal')
    ax4.set_xlabel('$m$')
    ax4.set_ylabel('Energy')
    #ax4.set_xscale('log'); 
    ax4.set_yscale('log')
    ax4.set_title(f'Magnetic $m$-spectrum({label_m})')

    # Save figure
    tag = 'single' if mode == 'single' else 'average'

    Ek,q,Ra = input_params_from_path(folderFile)
    save_path = os.path.join(save_dir, f'Ek_{Ek}_q{q}_Ra{Ra}_spectra_{tag}.png')
    plt.savefig(save_path, dpi=270)
    

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, save_path

def plot_spectra_kt(
    folderFile,
    save_dir,
    mode="single",
    start_time=None,
    stop_time=None,
    which="last",
    show=True,
    ref_scaling=False,
):
    """
    Plot kinetic and temperature spectra for purely hydrodynamic cases.

    Top row:
        Kinetic l- and m-spectra.

    Bottom row:
        Temperature l- and m-spectra.
    """
    if mode == "single":
        (
            lk,
            mk,
            ltot_k,
            ltor_k,
            lpol_k,
            mtot_k,
            mtor_k,
            mpol_k,
            time_k,
        ) = read_single_spectrum(
            folderFile,
            "kinetic",
            which=which,
        )

        (
            lt,
            mt,
            ltot_t,
            _,
            _,
            mtot_t,
            _,
            _,
            time_t,
        ) = read_single_spectrum(
            folderFile,
            "temperature",
            which=which,
        )

        label_k = f"Single t={time_k:.3E}"
        label_t = f"Single t={time_t:.3E}"

    elif mode == "average":
        if start_time is None or stop_time is None:
            start_time = 0
            stop_time = 1

        (
            lk,
            mk,
            ltot_k,
            ltor_k,
            lpol_k,
            mtot_k,
            mtor_k,
            mpol_k,
        ) = avgSpectra_new(
            folderFile,
            "kinetic",
            start_time,
            stop_time,
        )

        (
            lt,
            mt,
            ltot_t,
            _,
            _,
            mtot_t,
            _,
            _,
        ) = avgSpectra_new(
            folderFile,
            "temperature",
            start_time,
            stop_time,
        )

        label_k = f"Average t=[{start_time}, {stop_time}]"
        label_t = f"Average t=[{start_time}, {stop_time}]"

    else:
        raise ValueError("mode must be 'single' or 'average'")

    plt.close("all")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 9),
        dpi=180,
    )

    fig.subplots_adjust(
        wspace=0.3,
        hspace=0.35,
        left=0.12,
        top=0.92,
        right=0.97,
        bottom=0.12,
    )

    # Kinetic l-spectrum
    ax1, ax2 = axes[0]

    ax1.plot(lk[1:], ltot_k[1:], ".-", label="total")
    ax1.plot(lk[1:], ltor_k[1:], ".-", label="toroidal")
    ax1.plot(lk[1:], lpol_k[1:], ".-", label="poloidal")

    ax1.set_xlabel(r"$l$")
    ax1.set_ylabel("Energy")
    ax1.set_yscale("log")
    ax1.set_title(f"Kinetic $l$-spectrum ({label_k})")
    ax1.legend()

    # Kinetic m-spectrum
    ax2.plot(mk, mtot_k, ".-", label="total")
    ax2.plot(mk, mtor_k, ".-", label="toroidal")
    ax2.plot(mk, mpol_k, ".-", label="poloidal")

    ax2.set_xlabel(r"$m$")
    ax2.set_ylabel("Energy")
    ax2.set_yscale("log")
    ax2.set_title(f"Kinetic $m$-spectrum ({label_k})")
    ax2.legend()

    # Temperature spectra
    ax3, ax4 = axes[1]

    ax3.plot(lt[1:], ltot_t[1:], ".-", label="total")
    ax3.set_xlabel(r"$l$")
    ax3.set_ylabel("Energy")
    ax3.set_yscale("log")
    ax3.set_title(f"Temperature $l$-spectrum ({label_t})")
    ax3.legend()

    ax4.plot(mt, mtot_t, ".-", label="total")
    ax4.set_xlabel(r"$m$")
    ax4.set_ylabel("Energy")
    ax4.set_yscale("log")
    ax4.set_title(f"Temperature $m$-spectrum ({label_t})")
    ax4.legend()

    tag = "single" if mode == "single" else "average"

    Ek, q, Ra = input_params_from_path(folderFile)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{Ek}_q{q}_Ra{Ra}_spectra_kt_{tag}.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, save_path

def plot_spectra(
    folderFile,
    save_dir,
    mode="single",
    start_time=None,
    stop_time=None,
    which="last",
    show=True,
):
    """
    Plot kinetic, magnetic, and temperature spectra.

    Rows:
        1. Kinetic l- and m-spectra
        2. Magnetic l- and m-spectra
        3. Temperature l- and m-spectra

    Parameters
    ----------
    folderFile : str or path-like
        Simulation run directory.
    save_dir : str or path-like
        Directory in which the figure is saved.
    mode : {"single", "average"}
        Plot one spectrum or a time-averaged spectrum.
    start_time, stop_time : float, optional
        Averaging interval when mode="average".
    which : str
        Spectrum selection passed to read_single_spectrum.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
    save_path : str
    """
    if mode == "single":
        (
            lk,
            mk,
            ltot_k,
            ltor_k,
            lpol_k,
            mtot_k,
            mtor_k,
            mpol_k,
            time_k,
        ) = read_single_spectrum(
            folderFile,
            "kinetic",
            which=which,
        )

        (
            lm,
            mm,
            ltot_m,
            ltor_m,
            lpol_m,
            mtot_m,
            mtor_m,
            mpol_m,
            time_m,
        ) = read_single_spectrum(
            folderFile,
            "magnetic",
            which=which,
        )

        (
            lt,
            mt,
            ltot_t,
            _,
            _,
            mtot_t,
            _,
            _,
            time_t,
        ) = read_single_spectrum(
            folderFile,
            "temperature",
            which=which,
        )

        label_k = f"t={time_k:.3E}"
        label_m = f"t={time_m:.3E}"
        label_t = f"t={time_t:.3E}"

    elif mode == "average":
        if start_time is None or stop_time is None:
            start_time = 0
            stop_time = 1

        (
            lk,
            mk,
            ltot_k,
            ltor_k,
            lpol_k,
            mtot_k,
            mtor_k,
            mpol_k,
        ) = avgSpectra_new(
            folderFile,
            "kinetic",
            start_time,
            stop_time,
        )

        (
            lm,
            mm,
            ltot_m,
            ltor_m,
            lpol_m,
            mtot_m,
            mtor_m,
            mpol_m,
        ) = avgSpectra_new(
            folderFile,
            "magnetic",
            start_time,
            stop_time,
        )

        (
            lt,
            mt,
            ltot_t,
            _,
            _,
            mtot_t,
            _,
            _,
        ) = avgSpectra_new(
            folderFile,
            "temperature",
            start_time,
            stop_time,
        )

        label_k = f"average t=[{start_time}, {stop_time}]"
        label_m = f"average t=[{start_time}, {stop_time}]"
        label_t = f"average t=[{start_time}, {stop_time}]"

    else:
        raise ValueError("mode must be 'single' or 'average'")

    plt.close("all")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8, 9),
        dpi=180,
    )

    fig.subplots_adjust(
        wspace=0.3,
        hspace=0.35,
        left=0.12,
        top=0.92,
        right=0.97,
        bottom=0.12,
    )

    # Kinetic spectra
    ax1, ax2 = axes[0]

    ax1.plot(lk[1:], ltot_k[1:], ".-", label="total")
    ax1.plot(lk[1:], ltor_k[1:], ".-", label="toroidal")
    ax1.plot(lk[1:], lpol_k[1:], ".-", label="poloidal")
    ax1.set_xlabel(r"$l$")
    ax1.set_ylabel("Energy")
    ax1.set_yscale("log")
    ax1.set_title(f"Kinetic $l$-spectrum ({label_k})")
    ax1.legend()

    ax2.plot(mk, mtot_k, ".-", label="total")
    ax2.plot(mk, mtor_k, ".-", label="toroidal")
    ax2.plot(mk, mpol_k, ".-", label="poloidal")
    ax2.set_xlabel(r"$m$")
    ax2.set_ylabel("Energy")
    ax2.set_yscale("log")
    ax2.set_title(f"Kinetic $m$-spectrum ({label_k})")
    ax2.legend()

    # Magnetic spectra
    ax3, ax4 = axes[1]

    ax3.plot(lm[1:], ltot_m[1:], ".-", label="total")
    ax3.plot(lm[1:], ltor_m[1:], ".-", label="toroidal")
    ax3.plot(lm[1:], lpol_m[1:], ".-", label="poloidal")
    ax3.set_xlabel(r"$l$")
    ax3.set_ylabel("Energy")
    ax3.set_yscale("log")
    ax3.set_title(f"Magnetic $l$-spectrum ({label_m})")
    ax3.legend()

    ax4.plot(mm, mtot_m, ".-", label="total")
    ax4.plot(mm, mtor_m, ".-", label="toroidal")
    ax4.plot(mm, mpol_m, ".-", label="poloidal")
    ax4.set_xlabel(r"$m$")
    ax4.set_ylabel("Energy")
    ax4.set_yscale("log")
    ax4.set_title(f"Magnetic $m$-spectrum ({label_m})")
    ax4.legend()

    # Temperature spectra
    ax5, ax6 = axes[2]

    ax5.plot(lt[1:], ltot_t[1:], ".-", label="total")
    ax5.set_xlabel(r"$l$")
    ax5.set_ylabel("Energy")
    ax5.set_yscale("log")
    ax5.set_title(f"Temperature $l$-spectrum ({label_t})")
    ax5.legend()

    ax6.plot(mt, mtot_t, ".-", label="total")
    ax6.set_xlabel(r"$m$")
    ax6.set_ylabel("Energy")
    ax6.set_yscale("log")
    ax6.set_title(f"Temperature $m$-spectrum ({label_t})")
    ax6.legend()

    tag = "single" if mode == "single" else "average"

    Ek, q, Ra = input_params_from_path(folderFile)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{Ek}_Ra{Ra}_q{q}_spectra_{tag}.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, save_path

def plot_lmn_spectra(
    folderFile,
    save_dir,
    mode="single",
    start_time=None,
    stop_time=None,
    which="last",
    show=True,
):
    """
    Plot kinetic, magnetic, and temperature spectra together with
    two-dimensional radial-degree energy maps E(n, l).

    Columns:
        1. l-spectrum
        2. m-spectrum
        3. E(n, l) map

    Rows:
        1. Kinetic
        2. Magnetic
        3. Temperature

    Notes
    -----
    Averaging of the E(n, l) spectra is not currently implemented.
    """
    from matplotlib.colors import LogNorm

    if mode != "single":
        raise NotImplementedError(
            "Averaging mode for n-spectra is not yet implemented."
        )

    # Kinetic l- and m-spectra
    (
        lk,
        mk,
        ltot_k,
        ltor_k,
        lpol_k,
        mtot_k,
        mtor_k,
        mpol_k,
        time_k,
    ) = read_single_spectrum(
        folderFile,
        "kinetic",
        which=which,
    )

    # Magnetic l- and m-spectra
    (
        lm,
        mm,
        ltot_m,
        ltor_m,
        lpol_m,
        mtot_m,
        mtor_m,
        mpol_m,
        time_m,
    ) = read_single_spectrum(
        folderFile,
        "magnetic",
        which=which,
    )

    # Temperature l- and m-spectra
    (
        lt,
        mt,
        ltot_t,
        _,
        _,
        mtot_t,
        _,
        _,
        time_t,
    ) = read_single_spectrum(
        folderFile,
        "temperature",
        which=which,
    )

    # Two-dimensional E(n, l) spectra
    (
        nk,
        lk_n,
        e_tot_k,
        _,
        _,
        time_nk,
    ) = read_single_n_spectrum(
        folderFile,
        "kinetic",
        which=which,
    )

    (
        nm,
        lm_n,
        e_tot_m,
        _,
        _,
        time_nm,
    ) = read_single_n_spectrum(
        folderFile,
        "magnetic",
        which=which,
    )

    (
        nt,
        lt_n,
        e_tot_t,
        _,
        _,
        time_nt,
    ) = read_single_n_spectrum(
        folderFile,
        "temperature",
        which=which,
    )

    label_k = f"t={time_k:.3E}"
    label_m = f"t={time_m:.3E}"
    label_t = f"t={time_t:.3E}"

    plt.close("all")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13, 9),
        dpi=180,
    )

    fig.subplots_adjust(
        wspace=0.35,
        hspace=0.35,
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.10,
    )

    # Shift m by one so m=0 can be shown on a logarithmic axis.
    mk_plot = np.asarray(mk) + 1
    mm_plot = np.asarray(mm) + 1
    mt_plot = np.asarray(mt) + 1

    # ------------------------------------------------------------------
    # Kinetic spectra
    # ------------------------------------------------------------------
    ax_lk, ax_mk, ax_nk = axes[0]

    ax_lk.plot(lk[1:], ltot_k[1:], ".-", label="total")
    ax_lk.plot(lk[1:], ltor_k[1:], ".-", label="toroidal")
    ax_lk.plot(lk[1:], lpol_k[1:], ".-", label="poloidal")
    ax_lk.set(
        xlabel=r"$l$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Kinetic $l$-spectrum ({label_k})",
    )
    ax_lk.legend()

    ax_mk.plot(mk_plot, mtot_k, ".-", label="total")
    ax_mk.plot(mk_plot, mtor_k, ".-", label="toroidal")
    ax_mk.plot(mk_plot, mpol_k, ".-", label="poloidal")
    ax_mk.set(
        xlabel=r"$m+1$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Kinetic $m$-spectrum ({label_k})",
    )
    ax_mk.legend()

    if e_tot_k is not None:
        e_tot_k = np.asarray(e_tot_k)

        positive_k = e_tot_k[e_tot_k > 0]
        if positive_k.size > 0:
            image_k = ax_nk.pcolormesh(
                lk_n,
                nk,
                e_tot_k,
                norm=LogNorm(
                    vmin=positive_k.min(),
                    vmax=positive_k.max(),
                ),
                cmap="magma",
                shading="auto",
            )
            fig.colorbar(
                image_k,
                ax=ax_nk,
                label="Energy",
            )

    ax_nk.set(
        xlabel=r"$l$",
        ylabel=r"$n$",
        title=rf"Kinetic $E(n,l)$ ($t={time_nk:.3E}$)",
    )

    # ------------------------------------------------------------------
    # Magnetic spectra
    # ------------------------------------------------------------------
    ax_lm, ax_mm, ax_nm = axes[1]

    ax_lm.plot(lm[1:], ltot_m[1:], ".-", label="total")
    ax_lm.plot(lm[1:], ltor_m[1:], ".-", label="toroidal")
    ax_lm.plot(lm[1:], lpol_m[1:], ".-", label="poloidal")
    ax_lm.set(
        xlabel=r"$l$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Magnetic $l$-spectrum ({label_m})",
    )
    ax_lm.legend()

    ax_mm.plot(mm_plot, mtot_m, ".-", label="total")
    ax_mm.plot(mm_plot, mtor_m, ".-", label="toroidal")
    ax_mm.plot(mm_plot, mpol_m, ".-", label="poloidal")
    ax_mm.set(
        xlabel=r"$m+1$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Magnetic $m$-spectrum ({label_m})",
    )
    ax_mm.legend()

    if e_tot_m is not None:
        e_tot_m = np.asarray(e_tot_m)

        positive_m = e_tot_m[e_tot_m > 0]
        if positive_m.size > 0:
            image_m = ax_nm.pcolormesh(
                lm_n,
                nm,
                e_tot_m,
                norm=LogNorm(
                    vmin=positive_m.min(),
                    vmax=positive_m.max(),
                ),
                cmap="magma",
                shading="auto",
            )
            fig.colorbar(
                image_m,
                ax=ax_nm,
                label="Energy",
            )

    ax_nm.set(
        xlabel=r"$l$",
        ylabel=r"$n$",
        title=rf"Magnetic $E(n,l)$ ($t={time_nm:.3E}$)",
    )

    # ------------------------------------------------------------------
    # Temperature spectra
    # ------------------------------------------------------------------
    ax_lt, ax_mt, ax_nt = axes[2]

    ax_lt.plot(lt[1:], ltot_t[1:], ".-", label="total")
    ax_lt.set(
        xlabel=r"$l$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Temperature $l$-spectrum ({label_t})",
    )
    ax_lt.legend()

    ax_mt.plot(mt_plot, mtot_t, ".-", label="total")
    ax_mt.set(
        xlabel=r"$m+1$",
        ylabel="Energy",
        xscale="log",
        yscale="log",
        title=f"Temperature $m$-spectrum ({label_t})",
    )
    ax_mt.legend()

    if e_tot_t is not None:
        e_tot_t = np.asarray(e_tot_t)

        positive_t = e_tot_t[e_tot_t > 0]
        if positive_t.size > 0:
            image_t = ax_nt.pcolormesh(
                lt_n,
                nt,
                e_tot_t,
                norm=LogNorm(
                    vmin=positive_t.min(),
                    vmax=positive_t.max(),
                ),
                cmap="magma",
                shading="auto",
            )
            fig.colorbar(
                image_t,
                ax=ax_nt,
                label="Energy",
            )

    ax_nt.set(
        xlabel=r"$l$",
        ylabel=r"$n$",
        title=rf"Temperature $E(n,l)$ ($t={time_nt:.3E}$)",
    )

    Ek, q, Ra = input_params_from_path(folderFile)

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"Ek_{Ek}_q{q}_Ra{Ra}_spectra_with_nlmap.png",
    )

    fig.savefig(
        save_path,
        dpi=270,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, save_path

def plot_single_n_spectrum(filename, show=True, l_indices=None):
    """
    Plot E(n,l) vs n for selected l values from a QUICC-style n-spectrum file.

    Parameters
    ----------
    filename : str
        Path to the n-spectrum file.
    show : bool
        Whether to display the figure interactively.
    l_indices : list of int, optional
        Indices of l-columns to plot (default: 6 evenly spaced).
    """
     
    # --- Extract field name from filename ---
    base = os.path.basename(filename)
    match = re.match(r'(temperature|kinetic|magnetic)', base)
    if match:
        field = match.group(1).lower()
    else:
        raise ValueError(f"Cannot determine field type from filename: {filename}")

    print(f"Detected field: {field}")

    with open(filename, "r") as f:
        lines = f.readlines()

    # --- Extract l-values from header lines ---
    l_values = []
    for line in lines:
        if line.strip().startswith("# l ="):
            match = re.search(r"l\s*=\s*(\d+)", line)
            if match:
                l_values.append(int(match.group(1)))

    # --- Find where numeric data starts ---
    data_start = None
    for i, line in enumerate(lines):
        if not line.strip().startswith("#") and line.strip():
            data_start = i
            break
    if data_start is None:
        raise ValueError("No numeric data block found in file.")

    # --- Load numeric block ---
    data = np.loadtxt(lines[data_start:], ndmin=2)
    n = data[:, 0]
    E = data[:, 1:]  # shape: (N_n, N_l)

    # --- Choose which l to plot ---
    n_l = len(l_values)
    if l_indices is None:
        # default: pick up to 6 evenly spaced l values
        n_plot = min(6, n_l)
        l_indices = np.linspace(0, n_l - 1, n_plot, dtype=int)
    l_selected = [l_values[i] for i in l_indices]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for idx in l_indices:
        ax.plot(n, E[:, idx], '.-', label=f'$l={l_values[idx]}$')

    # RMS over all l (for convergence check)
    E_rms = np.sqrt(np.mean(E**2, axis=1))
    ax.plot(n, E_rms, 'k--', lw=2, label='RMS over $l$')

    # --- Axes formatting ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'Energy')
    ax.set_title(fr'{field} $n$-spectrum')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='best')

    if show:
        plt.show()

    return n, E, np.array(l_values)

def plot_normalized_spectra_single_run(run_path, save_dir=None, which='last', show=True):
    """
    Plot a single run's kinetic and magnetic spectra (l and m), normalized so that 
    each spectrum has max = 1. Useful for comparing shapes directly.

    Args:
        run_path: path to run folder (e.g. ".../run0")
        save_dir: where to save plot
        which: "last" or an integer index (passed to read_single_spectrum)
        show: display figure
    """

    # ----- Load kinetic spectra -----
    lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = \
        read_single_spectrum(run_path, 'kinetic', which=which)

    # ----- Load magnetic spectra -----
    lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = \
        read_single_spectrum(run_path, 'magnetic', which=which)

    # ----- Normalize spectra so max = 1 -----
    def _normalize(arr):
        if np.max(arr) == 0:
            return arr
        return arr / np.max(arr)

    ltot_k_n = _normalize(ltot_k)
    mtot_k_n = _normalize(mtot_k)
    ltot_m_n = _normalize(ltot_m)
    mtot_m_n = _normalize(mtot_m)

    # ----- Create figure -----
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
    ax_l, ax_m = axes

    # Colors
    cK = "tab:blue"
    cM = "tab:red"

    # ---------- l-spectrum ----------
    ax_l.plot(lk, ltot_k_n, '.-', color=cK, label='Kinetic')
    ax_l.plot(lm, ltot_m_n, '.-', color=cM, label='Magnetic')
    ax_l.set_xlabel("$l$")
    ax_l.set_ylabel("Normalized Energy")
    ax_l.set_yscale("log")
    ax_l.set_title(f"Normalized $l$-spectra (t={time_k:.3E})")
    ax_l.grid(alpha=0.3)
    ax_l.legend()

    # ---------- m-spectrum ----------
    ax_m.plot(mk, mtot_k_n, '.-', color=cK, label='Kinetic')
    ax_m.plot(mm, mtot_m_n, '.-', color=cM, label='Magnetic')
    ax_m.set_xlabel("$m$")
    ax_m.set_ylabel("Normalized Energy")
    ax_m.set_yscale("log")
    ax_m.set_title(f"Normalized $m$-spectra (t={time_m:.3E})")
    ax_m.grid(alpha=0.3)
    ax_m.legend()

    plt.tight_layout()

    # ----- Save -----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "normalized_spectra.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectra plot to: {save_path}")
    else:
        save_path = None

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, save_path
