# src/quicc_dynavis/spectra.py
import os
import matplotlib.pyplot as plt
from .io import get_parameters, read_spectra
from .io import read_single_spectrum, avgSpectra_new  # assuming you have it in io or move it here


def plot_spectra(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True):
    """
    Plot kinetic and magnetic spectra (single or averaged).
    Top row: kinetic, Bottom row: magnetic.
    """
    # --- load spectra ---
    if mode == 'single':
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = read_single_spectrum(folderFile, 'kinetic', which=which)
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = read_single_spectrum(folderFile, 'magnetic', which=which)
        label_k = f"Single t={time_k:.2e}"
        label_m = f"Single t={time_m:.2e}"

    elif mode == "average":
        # --- kinetic ---
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k = avgSpectra_new(folderFile, "kinetic", start_time, stop_time)
        # --- magnetic ---
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m = avgSpectra_new(folderFile, "magnetic", start_time, stop_time)
        label_k = f"average over t [{start_time}, {stop_time}]"
        label_m = f"average over t [{start_time}, {stop_time}]"

    else:
        raise ValueError("mode must be 'single' or 'average'")

    # --- create figure ---
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.subplots_adjust(wspace=0.3, hspace=0.35, left=0.12, top=0.92, right=0.97, bottom=0.12)

    # Kinetic (top row)
    ax1, ax2 = axes[0]
    ax1.plot(lk[1:], ltot_k[1:], '.-', label='total')
    ax1.plot(lk[1:], ltor_k[1:], '.-', label='toroidal')
    ax1.plot(lk[1:], lpol_k[1:], '.-', label='poloidal')
    ax1.set_xlabel('$l$')
    ax1.set_ylabel('Energy')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_title(f'Kinetic $l$-spectrum({label_k})')
    ax1.legend()

    ax2.plot(mk, mtot_k, '.-', label='total')
    ax2.plot(mk, mtor_k, '.-', label='toroidal')
    ax2.plot(mk, mpol_k, '.-', label='poloidal')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('Energy')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_title(f'Kinetic $m$-spectrum({label_k})')

    # Magnetic (bottom row)
    ax3, ax4 = axes[1]
    ax3.plot(lm[1:], ltot_m[1:], '.-', label='total')
    ax3.plot(lm[1:], ltor_m[1:], '.-', label='toroidal')
    ax3.plot(lm[1:], lpol_m[1:], '.-', label='poloidal')
    ax3.set_xlabel('$l$')
    ax3.set_ylabel('Energy')
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_title(f'Magnetic $l$-spectrum ({label_m})')
    ax3.legend()

    ax4.plot(mm, mtot_m, '.-', label='total')
    ax4.plot(mm, mtor_m, '.-', label='toroidal')
    ax4.plot(mm, mpol_m, '.-', label='poloidal')
    ax4.set_xlabel('$m$')
    ax4.set_ylabel('Energy')
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.set_title(f'Magnetic $m$-spectrum({label_m})')

    # Save figure
    tag = 'single' if mode == 'single' else 'average'

    Ek,Pm,Pr,q,Ra,Ro=get_parameters(folderFile+'/run0/parameters.cfg','no')

    save_path = os.path.join(save_dir, f'{Ek}_{Ra}_{q}_{tag}_spectra.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()

    return fig, axes, save_path

