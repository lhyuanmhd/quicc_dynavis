# src/quicc_dynavis/spectra.py
import os
import matplotlib.pyplot as plt
import numpy as np
from .io import get_parameters, read_spectra
from .io import read_single_spectrum,read_single_n_spectrum, avgSpectra_new  # assuming you have it in io or move it here
from .timeseries import input_params_from_path
import re


def plot_spectra_km(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True, ref_scaling=False):
    """
    Plot kinetic and magnetic spectra (single or averaged).
    Top row: kinetic, Bottom row: magnetic.
    """
    # --- load spectra ---
    
    if mode == 'single':
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = read_single_spectrum(folderFile, 'kinetic', which=which)
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = read_single_spectrum(folderFile, 'magnetic', which=which)
        #lt, mt, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t, time_t = read_single_spectrum(folderFile, 'temperature', which=which)   
        
        # add temperature spectra here if needed
        label_k = f"Single t={time_k:.3E}"
        label_m = f"Single t={time_m:.3E}"
        #label_t = f"Single t={time_m:.3E}"

    elif mode == "average":
        if start_time == None or stop_time == None:
            start_time = 0
            stop_time = 1
            
        # --- kinetic ---
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k = avgSpectra_new(folderFile, "kinetic", start_time, stop_time)
        # --- magnetic ---
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m = avgSpectra_new(folderFile, "magnetic", start_time, stop_time)
        #--- temperature ---
        #lm, mm, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t = avgSpectra_new(folderFile, "temperature", start_time, stop_time)
        label_k = f"average"
        label_m = f"average"
        label_t = f"average"



    else:
        raise ValueError("mode must be 'single' or 'average'")

    # --- create figure ---
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), dpi=180)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, left=0.12, top=0.92, right=0.97, bottom=0.12)
    
    #mk,mm,mt = mk+1, mm+1, mt+1  # shift m for plotting
    #lk,lm,lt = lk+1, lm+1, lt+1  # shift m for plotting

    # Kinetic (top row)
    # l-spetra
    ax1, ax2 = axes[0]
    ax1.plot(lk[1:], ltot_k[1:], '.-', label='total')
    ax1.plot(lk[1:], ltor_k[1:], '.-', label='toroidal')
    ax1.loglog(lk[1:], lpol_k[1:], '.-', label='poloidal')

#    # --------------------------------------------------
#     # Reference -5/3 scaling anchored at spectral peak
#     # --------------------------------------------------
#     if ref_scaling:
#         lk = np.asarray(lk)
#         Ek = np.asarray(ltot_k)

#         # ignore l=0 if present
#         lk_use = lk[1:]
#         Ek_use = Ek[1:]

#         # index of spectral peak
#         ipeak = np.argmax(Ek_use)

#         l_ref = lk_use[ipeak]
#         E_ref = Ek_use[ipeak]

#         # scaling range (purely for plotting)
#         n = len(lk_use)
#         sid = int(0.25 * n)
#         eid = int(0.8 * n)

#         # -5/3 reference curve
#         E_kolmo = 3 * E_ref * (lk_use / l_ref)**(-5/3)

#         ax1.plot(
#             lk_use[sid:eid],
#             E_kolmo[sid:eid],
#             'k--',
#             lw=1.5,
#             label=r'$l^{-5/3}$'
#         )

#         # 3/4 reference curve
#         E_scal = 1.5* E_ref * (lk_use / l_ref)**(4/5) 
#         sid = int(0.01 * n)
#         eid = int(0.2 * n)

#         ax1.plot(
#             lk_use[sid:eid],
#             E_scal[sid:eid],
#             '--',
#             lw=1.5,
#             label=r'$l^{4/5}$'
#         )


#         # -15/2 reference curve
#         E_scal = 5e2* E_ref * (lk_use / l_ref)**(-8) 
#         sid = int(0.6 * n)
#         eid = int(1 * n)

#         ax1.plot(
#             lk_use[sid:eid],
#             E_scal[sid:eid],
#             '--',
#             lw=1.5,
#             label=r'$l^{-8}$'
#         )

     

    ax1.set_xlabel('$l$')
    ax1.set_ylabel('Energy')
    #ax1.set_xscale('log'); 
    ax1.set_yscale('log')
    ax1.set_title(f'Kinetic $l$-spectrum({label_k})')
    ax1.legend()
    
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

    # # reference curve
    # E_kolmo = 8e4* E_ref * (lm_use / l_ref)**(-4)

    # ax3.plot(
    #     lm_use[sid:eid],
    #     E_kolmo[sid:eid],
    #     'k--',
    #     lw=1.5,
    #     label=r'$l^{-4}$'
    # )


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



def plot_spectra_kt(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True, ref_scaling=False):
    """
    Plot kinetic and magnetic spectra (single or averaged).
    Top row: kinetic, Bottom row: magnetic.
    """
    # --- load spectra ---
    
    if mode == 'single':
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = read_single_spectrum(folderFile, 'kinetic', which=which)
        #lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = read_single_spectrum(folderFile, 'magnetic', which=which)
        #lt, mt, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t, time_t = read_single_spectrum(folderFile, 'temperature', which=which)   
        
        # add temperature spectra here if needed
        label_k = f"Single t={time_k:.3E}"
        #label_m = f"Single t={time_m:.3E}"
        #label_t = f"Single t={time_m:.3E}"

    elif mode == "average":
        if start_time == None or stop_time == None:
            start_time = 0
            stop_time = 1
            
        # --- kinetic ---
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k = avgSpectra_new(folderFile, "kinetic", start_time, stop_time)
        # --- magnetic ---
        #lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m = avgSpectra_new(folderFile, "magnetic", start_time, stop_time)
        #--- temperature ---
        #lm, mm, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t = avgSpectra_new(folderFile, "temperature", start_time, stop_time)
        
        label_k = f"average"
        label_m = f"average"
        label_t = f"average"



    else:
        raise ValueError("mode must be 'single' or 'average'")

    # --- create figure ---
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), dpi=180)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, left=0.12, top=0.92, right=0.97, bottom=0.12)
    
    #mk,mm,mt = mk+1, mm+1, mt+1  # shift m for plotting
    #lk,lm,lt = lk+1, lm+1, lt+1  # shift m for plotting

    # Kinetic (top row)
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
    
    # m-spectra
    ax2.plot(mk, mtot_k, '.-', label='total')
    ax2.plot(mk, mtor_k, '.-', label='toroidal')
    ax2.loglog(mk, mpol_k, '.-', label='poloidal')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('Energy')
    #ax2.set_xscale('log'); 
    ax2.set_yscale('log')
    ax2.set_title(f'Kinetic $m$-spectrum({label_k})')


    # # Magnetic (bottom row)
    # ax3, ax4 = axes[1]

    # lm = np.asarray(lm)
    # Em = np.asarray(ltot_m)

    # # ignore l=0 if present
    # lm_use = lm[1:]
    # Em_use = Em[1:]

    # # index of spectral peak
    # ipeak = np.argmax(Em_use)

    # l_ref = lm_use[ipeak]
    # E_ref = Em_use[ipeak]

    # n = len(lm_use)
    # sid = int(0.4 * n)
    # eid = int(1 * n)

    # # # reference curve
    # # E_kolmo = 8e4* E_ref * (lm_use / l_ref)**(-4)

    # # ax3.plot(
    # #     lm_use[sid:eid],
    # #     E_kolmo[sid:eid],
    # #     'k--',
    # #     lw=1.5,
    # #     label=r'$l^{-4}$'
    # # )


    # ax3.plot(lm[1:], ltot_m[1:], '.-', label='total')
    # ax3.plot(lm[1:], ltor_m[1:], '.-', label='toroidal')
    # ax3.loglog(lm[1:], lpol_m[1:], '.-', label='poloidal')
    # ax3.set_xlabel('$l$')
    # ax3.set_ylabel('Energy')
    # #ax3.set_xscale('log'); 
    # ax3.set_yscale('log')
    # ax3.set_title(f'Magnetic $l$-spectrum ({label_m})')
    # ax3.legend()

    # ax4.plot(mm, mtot_m, '.-', label='total')
    # ax4.plot(mm, mtor_m, '.-', label='toroidal')
    # ax4.loglog(mm, mpol_m, '.-', label='poloidal')
    # ax4.set_xlabel('$m$')
    # ax4.set_ylabel('Energy')
    # #ax4.set_xscale('log'); 
    # ax4.set_yscale('log')
    # ax4.set_title(f'Magnetic $m$-spectrum({label_m})')

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


def plot_spectra(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True):
    """
    Plot kinetic and magnetic spectra (single or averaged).
    Top row: kinetic, Bottom row: magnetic.
    """
    # --- load spectra ---
    if mode == 'single':
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = read_single_spectrum(folderFile, 'kinetic', which=which)
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = read_single_spectrum(folderFile, 'magnetic', which=which)
        lt, mt, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t, time_t = read_single_spectrum(folderFile, 'temperature', which=which)   
        
        # add temperature spectra here if needed
        label_k = f"t={time_k:.3E}"
        label_m = f"t={time_m:.3E}"
        label_t = f"t={time_m:.3E}"

    elif mode == "average":
        # --- kinetic ---
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k = avgSpectra_new(folderFile, "kinetic", start_time, stop_time)
        # --- magnetic ---
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m = avgSpectra_new(folderFile, "magnetic", start_time, stop_time)
        #--- temperature ---
        lm, mm, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t = avgSpectra_new(folderFile, "temperature", start_time, stop_time)
        
        #label_k = f"average over t [{start_time}, {stop_time}]"
        #label_m = f"average over t [{start_time}, {stop_time}]"
        #label_t = f"average over t [{start_time}, {stop_time}]"
    else:
        raise ValueError("mode must be 'single' or 'average'")

    # --- create figure ---
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(8, 9), dpi=180)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, left=0.12, top=0.92, right=0.97, bottom=0.12)
    
    #mk,mm,mt = mk+1, mm+1, mt+1  # shift m for plotting
    #lk,lm,lt = lk+1, lm+1, lt+1  # shift m for plotting

    # Kinetic (top row)
    ax1, ax2 = axes[0]
    ax1.plot(lk[1:], ltot_k[1:], '.-', label='total')
    ax1.plot(lk[1:], ltor_k[1:], '.-', label='toroidal')
    ax1.plot(lk[1:], lpol_k[1:], '.-', label='poloidal')
    ax1.set_xlabel('$l$')
    ax1.set_ylabel('Energy')
    #ax1.set_xscale('log'); 
    ax1.set_yscale('log')
    ax1.set_title(f'Kinetic ({label_k})')
    ax1.legend()

    ax2.plot(mk, mtot_k, '.-', label='total')
    ax2.plot(mk, mtor_k, '.-', label='toroidal')
    ax2.plot(mk, mpol_k, '.-', label='poloidal')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('Energy')
    #ax2.set_xscale('log'); 
    ax2.set_yscale('log')
    ax2.set_title(f'Kinetic ({label_k})')

    # Magnetic (bottom row)
    ax3, ax4 = axes[1]
    ax3.plot(lm[1:], ltot_m[1:], '.-', label='total')
    ax3.plot(lm[1:], ltor_m[1:], '.-', label='toroidal')
    ax3.plot(lm[1:], lpol_m[1:], '.-', label='poloidal')
    ax3.set_xlabel('$l$')
    ax3.set_ylabel('Energy')
    #ax3.set_xscale('log'); 
    ax3.set_yscale('log')
    ax3.set_title(f'Magnetic ({label_m})')
    ax3.legend()

    ax4.plot(mm, mtot_m, '.-', label='total')
    ax4.plot(mm, mtor_m, '.-', label='toroidal')
    ax4.plot(mm, mpol_m, '.-', label='poloidal')
    ax4.set_xlabel('$m$')
    ax4.set_ylabel('Energy')
    #ax4.set_xscale('log'); 
    ax4.set_yscale('log')
    ax4.set_title(f'Magnetic ({label_m})')

    #tempeature spectra (added third row)
    ax5, ax6 = axes[2]
    ax5.plot(lt, ltot_t, '.-', label='total')
    ax5.set_xlabel('$l$')
    ax5.set_ylabel('Energy')
    #ax5.set_xscale('log'); 
    ax5.set_yscale('log')
    ax5.set_title(f'Temperature $l$-spectrum ({label_t})')
    ax5.legend()

    ax6.plot(mt, mtot_t, '.-', label='total')
    ax6.set_xlabel('$m$')
    ax6.set_ylabel('Energy')
    #ax6.set_xscale('log'); 
    ax6.set_yscale('log')
    ax6.set_title(f'Temperature $m$-spectrum({label_t})')


    # Save figure
    tag = 'single' if mode == 'single' else 'average'

    # Ek,Pm,Pr,q,Ra,Ro=get_parameters(folderFile+'/run0/parameters.cfg','no')
    # Ek = f"{Ek:.1e}"

    Ek,q,Ra = input_params_from_path(folderFile)
    save_path = os.path.join(save_dir, f'Ek_{Ek}_Ra{Ra}_q{q}_spectra.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()

    return fig, axes, save_path

def plot_lmn_spectra(folderFile, save_dir, mode='single', start_time=None, stop_time=None, which='last', show=True):
    """
    Plot kinetic, magnetic, and temperature spectra (l-, m-spectra),
    including 2D (n, l) energy maps.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from matplotlib.colors import LogNorm

    # --- Load spectra ---
    if mode == 'single':
        # l and m spectra
        lk, mk, ltot_k, ltor_k, lpol_k, mtot_k, mtor_k, mpol_k, time_k = read_single_spectrum(folderFile, 'kinetic', which=which)
        lm, mm, ltot_m, ltor_m, lpol_m, mtot_m, mtor_m, mpol_m, time_m = read_single_spectrum(folderFile, 'magnetic', which=which)
        lt, mt, ltot_t, ltor_t, lpol_t, mtot_t, mtor_t, mpol_t, time_t = read_single_spectrum(folderFile, 'temperature', which=which)

        # n spectra (now full 2D data)
        nk, lk2, e_tot_k, e_tor_k, e_pol_k, time_nk = read_single_n_spectrum(folderFile, 'kinetic', which=which)
        nm, lm2, e_tot_m, e_tor_m, e_pol_m, time_nm = read_single_n_spectrum(folderFile, 'magnetic', which=which)
        nt, lt2, e_tot_t,       _,       _, time_nt = read_single_n_spectrum(folderFile, 'temperature', which=which)

        label_k = f"t={time_k:.3E}"
        label_m = f"t={time_m:.3E}"
        label_t = f"t={time_t:.3E}"

    else:
        raise NotImplementedError("Averaging mode for n-spectra not yet implemented.")

    # --- Create figure ---
    plt.close('all')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), dpi=180)
    fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.98, top=0.93, bottom=0.10)

    mk, mm, mt = mk + 1, mm + 1, mt + 1
    nkp1, nmp1, ntp1 = nk, nm, nt  # shifted n for labeling

    # === Kinetic spectra ===
    axes[0, 0].plot(lk[1:], ltot_k[1:], '.-', label='total')
    axes[0, 0].plot(lk[1:], ltor_k[1:], '.-', label='toroidal')
    axes[0, 0].plot(lk[1:], lpol_k[1:], '.-', label='poloidal')
    axes[0, 0].set(xlabel='$l$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Kinetic $l$-spectrum ({label_k})')
    axes[0, 0].legend()

    axes[0, 1].plot(mk, mtot_k, '.-', label='total')
    axes[0, 1].plot(mk, mtor_k, '.-', label='toroidal')
    axes[0, 1].plot(mk, mpol_k, '.-', label='poloidal')
    axes[0, 1].set(xlabel='$m+1$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Kinetic $m$-spectrum ({label_k})')

    # === Kinetic n-l map ===
    if e_tot_k is not None:
        im = axes[0, 2].pcolormesh(lk2, nkp1, e_tot_k, norm=LogNorm(),cmap='magma')#cmap='hot')# cmap='viridis')
        axes[0, 2].set(xlabel='$l$', ylabel='$n$', title='Kinetic $E(n,l)$')
        fig.colorbar(im, ax=axes[0, 2], label='Energy')

    # === Magnetic spectra ===
    axes[1, 0].plot(lm[1:], ltot_m[1:], '.-', label='total')
    axes[1, 0].plot(lm[1:], ltor_m[1:], '.-', label='toroidal')
    axes[1, 0].plot(lm[1:], lpol_m[1:], '.-', label='poloidal')
    axes[1, 0].set(xlabel='$l$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Magnetic $l$-spectrum ({label_m})')
    axes[1, 0].legend()

    axes[1, 1].plot(mm, mtot_m, '.-', label='total')
    axes[1, 1].plot(mm, mtor_m, '.-', label='toroidal')
    axes[1, 1].plot(mm, mpol_m, '.-', label='poloidal')
    axes[1, 1].set(xlabel='$m+1$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Magnetic $m$-spectrum ({label_m})')

    # === Magnetic n-l map ===
    if e_tot_m is not None:
        im = axes[1, 2].pcolormesh(lm2, nmp1, e_tot_m, norm=LogNorm(), cmap='magma')#cmap='hot')#cmap='plasma')
        axes[1, 2].set(xlabel='$l$', ylabel='$n$', title='Magnetic $E(n,l)$')
        fig.colorbar(im, ax=axes[1, 2], label='Energy')

    # === Temperature spectra ===
    axes[2, 0].plot(lt[1:], ltot_t[1:], '.-', label='total')
    axes[2, 0].set(xlabel='$l$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Temperature $l$-spectrum ({label_t})')

    axes[2, 1].plot(mt, mtot_t, '.-', label='total')
    axes[2, 1].set(xlabel='$m+1$', ylabel='Energy', xscale='log', yscale='log',
                   title=f'Temperature $m$-spectrum ({label_t})')

    # === Temperature n-l map ===
    if e_tot_t is not None:
        im = axes[2, 2].pcolormesh(lt2, ntp1, e_tot_t, norm=LogNorm(),cmap='magma')#cmap= 'hot')#cmap='inferno')
        axes[2, 2].set(xlabel='$l$', ylabel='$n$', title='Temperature $E(n,l)$')
        fig.colorbar(im, ax=axes[2, 2], label='Energy')

    # === Save ===
    # Ek, Pm, Pr, q, Ra, Ro = get_parameters(os.path.join(folderFile, 'run0/parameters.cfg'), 'no')
    # Ek = f"{Ek:.1e}"
    # Ra = f'{Ra:.1e}'
    # q = f'{q:.1e}'
    
    Ek,q,Ra = input_params_from_path(folderFile)
    save_path = os.path.join(save_dir, f'Ek_{Ek}_q{q}_Ra{Ra}_spectra_with_nlmap.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()

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
