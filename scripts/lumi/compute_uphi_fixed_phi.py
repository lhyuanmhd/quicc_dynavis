#!/usr/bin/env python3
"""Compare instantaneous u_phi(s,z=0,phi=0) at several magnetic truncations.

Also plots the separate, optional column-averaged geostrophic profile Ug(s).
"""

import sys
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Point to the source tree in which you installed flow_profile.py.
PACKAGE_SRC = Path('/scratch/project_465001528/lhyuan/codes/quicc_dynavis/src')
sys.path.insert(0, str(PACKAGE_SRC))

from quicc_dynavis import flow_profile  # noqa: E402

BASE_DIR = Path('/scratch/project_465001528/lhyuan/InviscidDyn/E_eta_1e-6/visu')
OUTPUT_DIR = Path(__file__).resolve().parent

CASES = {
    r'$L_B=40$': 'lb40/runs/run000/visu/vis_fields_0000.npz',
    r'$L_B=60$': 'lb60/runs/run000/visu/vis_fields_0000.npz',
    r'$L_B=80$': 'lb80/runs/run000/visu_0002/vis_fields_0000.npz',
}

# The original instantaneous fixed-ray plot is the main figure.
PHI0 = 0.0
Z0 = 0.0
N_S = 400
S_MAX = 0.98  # Diagnostic: omit the last 2% near the outer boundary.
UPHI_YLIM = (-600, 600)  # Set None for auto scaling.
MAKE_GEOSTROPHIC_FIGURE = True

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 0.9,
})


def main():
    print('flow_profile loaded from:', flow_profile.__file__)

    with ExitStack() as stack:
        data_sources = {}
        for label, relative_path in CASES.items():
            filename = BASE_DIR / relative_path
            if not filename.is_file():
                raise FileNotFoundError(filename)
            data_sources[label] = stack.enter_context(np.load(filename))

        # Figure 1: fixed-phi instantaneous velocity (NOT averaged in phi).
        flow_profile.plot_uphi_vs_s_fixed_z_phi(
            data_sources,
            z_list=[Z0],
            phi0=PHI0,
            n_s=N_S,
            s_max=S_MAX,
            ylim=UPHI_YLIM,
            savefig=OUTPUT_DIR / 'uphi_s_fixed_phi.pdf',
            show=False,
            title=False,
            figsize=(7.2, 4.4),
        )
        plt.close('all')

        if MAKE_GEOSTROPHIC_FIGURE:
            flow_profile.plot_column_averaged_geostrophic_flow(
                data_sources,
                n_s=250,
                n_z=300,
                s_max=S_MAX,
                savefig=OUTPUT_DIR / 'Ug_vs_s.pdf',
                show=False,
                figsize=(7.2, 4.4),
            )
            plt.close('all')

    print('Figures saved in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
