import os
import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from .io import get_parameters, get_resolution, get_boundary_conditions
from .io import F_conc_timeseries, F_read_dipolarity, F_read_energyQCC, F_read_Nusselt
from matplotlib.ticker import FuncFormatter    
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import LogLocator, FuncFormatter

import matplotlib
#matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'cm'   # use Computer Modern for math
#matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
#matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
#matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def safe_conc_timeseries(RunFolders, tag):
    """
    Call F_conc_timeseries; if the underlying files are missing, return empty arrays.
    Assumes F_conc_timeseries returns (t, tot, tor, pol) for these tags.
    """
    try:
        out = F_conc_timeseries(RunFolders, tag)
        if out is None:
            return np.array([]), np.array([]), np.array([]), np.array([])
        t, tot, tor, pol = out
        if t is None or len(t) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return t, tot, tor, pol
    except (FileNotFoundError, OSError, ValueError, IndexError) as e:
        print(f"[WARN] timeseries '{tag}' not found in any run folder (or unreadable): {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])



def _discover_run_folders(folderFile):
    # New layout: folderFile/runs/run*
    runs_new = sorted(glob.iglob(os.path.join(folderFile, "runs", "run*")))
    runs_new = [r for r in runs_new if os.path.isdir(r)]
    # filter only run<digits> to avoid run0_abort etc.
    runs_new = [r for r in runs_new if os.path.basename(r).startswith("run") and os.path.basename(r)[3:].isdigit()]

    if len(runs_new) > 0:
        return runs_new

    # Old layout: folderFile/run*
    runs_old = sorted(glob.iglob(os.path.join(folderFile, "run*")))
    runs_old = [r for r in runs_old if os.path.isdir(r)]
    runs_old = [r for r in runs_old if os.path.basename(r).startswith("run") and os.path.basename(r)[3:].isdigit()]
    return runs_old


def input_params_from_path(folderFile):
    from pathlib import Path

    runs_dir = Path(folderFile) / "runs"

    if (runs_dir / "run0").exists():
        run0_dir = runs_dir / "run0"
    elif (runs_dir / "run000").exists():
        run0_dir = runs_dir / "run000"
    else:
        raise FileNotFoundError("Neither run0 nor run000 exists")

    Ek, Pm, Pr, q, Ra, Ro = get_parameters(
        str(run0_dir / "parameters.cfg"),
        'no'
    )
    
    # Ek in format 1e-4
    Ek = f'{Ek:.1e}'
    Ra = f'{Ra:.1e}'
    q  = f'{q:.1e}'

    return Ek,q,Ra

def extract_Ek_root(path):
    """
    Extract the directory up to and including the Ek folder (e.g. E1e-5, E4.225e-5).

    Example:
    /.../CattaneoHuges/E1e-5/q_1.2/Ra2e3/run0
        --> /.../CattaneoHuges/E1e-5
    /.../IlessDyn/Fixed_flux/E4.225e-5/JonesE2e-4
        --> /.../IlessDyn/Fixed_flux/E4.225e-5
    """
    parts = os.path.normpath(path).split(os.sep)

    for i, p in enumerate(parts):
        # Updated regex to handle optional decimal point and digits
        if re.match(r"E\d+(?:\.\d+)?e[-+]?\d+", p):
            return os.sep.join(parts[: i + 1])

    raise ValueError(f"Could not find Ek folder in path:\n{path}")


import os
import csv
import numpy as np

# keep updating according to the need
""" def write_dynamo_summary_csv(
    csv_path,
    q, Ra, Ek,
    E0mag, # New intial magnetic energy
    dynamo,
    dipolarity,
    Elsasser,
    visDis,   # new viscous dissipation
    ohmDis,   # new ohmic dissipation 
    fohm,
    L_u,      #new
    L_b,
    T_perb,
    nusselt,
    reversal,
    Rm,
    relative_std_fdip,   # fluctuation of the dipolarity 
    bc_mag,bc_temp,bc_vel,
    N,M,L,
    rtol=1e-4
):
    ""
    Write or update dynamo diagnostics in a CSV file.

    If (q, Ra, Ek) already exists (within tolerance), overwrite that row.
    Otherwise, append a new row.

    magEtoti is the intial magnetic energy

    Final table is sorted by:
        1) q
        2) Ra
    ""

    # -------- Compact, readable header --------
    header = [
        "q",
        "Ra",
        "Ek",
        "E0mag",
        "dyn",
        "fdip",
        "Lambda",
        "visDis", 
        "ohmDis", 
        "fohm",
        "L_u",
        "L_b",
        "T_perb",
        "Nu",
        "rev",
        "Rm",
        "relative_std_fdip", # new added standard devation of dipolarity/ average dipolarity 
        "bc_mag", 
        "bc_temp",
        "bc_vel",
        "N",
        "M",
        "L"
    ]

    new_row = [
        f"{q:.2f}",
        f"{Ra:.2e}",
        f"{Ek:.2e}",
        f"{E0mag:.2e}", # added
        int(dynamo),
        f"{dipolarity:.2f}",
        f"{Elsasser:.2e}",
        f"{visDis:.2e}", # added
        f"{ohmDis:.2e}", # added
        f"{fohm:.3f}", # added
        f"{L_u:.2e}", #added
        f"{L_b:.2e}", #added
        f"{T_perb:.3e}", #added
        f"{nusselt:.2f}", #added
        int(reversal),
        f"{Rm:.2f}",
        f"{relative_std_fdip:.2f}", #added
        bc_mag,
        bc_temp,
        bc_vel,
        int(N),
        int(M),
        int(L)
    ]

    rows = []

    # ---------- Read existing file ----------
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if rows and rows[0] != header:
            raise ValueError(f"CSV header mismatch in {csv_path}")
    else:
        rows.append(header)

    updated = False
    # expected bc strings for the new_row / current case
    # (set these from your parser; examples: "insulating", "fixed_flux", "no_slip")
    bc_mag = bc_mag.strip()
    bc_temp = bc_temp.strip()
    bc_vel = bc_vel.strip()

    BC_COLS = (17, 18, 19)

    for i in range(1, len(rows)):
        q_i, Ra_i, Ek_i, E0mag_i = map(float, rows[i][:4])

        same_params = (
            np.isclose(Ra_i, Ra, rtol=rtol, atol=0.0) and
            np.isclose(q_i,  q,  rtol=rtol, atol=0.0) and
            np.isclose(Ek_i, Ek, rtol=rtol, atol=0.0)
        )

        # same initial magnetic energy E0mag?
        same_E0mag = np.isclose(E0mag_i, E0mag, rtol=1e-2, atol=1e-12)

        # same boundary conditions?
        row_bc_mag  = rows[i][BC_COLS[0]].strip()
        row_bc_temp = rows[i][BC_COLS[1]].strip()
        row_bc_vel  = rows[i][BC_COLS[2]].strip()

        same_bc = (row_bc_mag == bc_mag and row_bc_temp == bc_temp and row_bc_vel == bc_vel)

        # same input parameters + same E0mag + same BCs
        if same_params: #and same_E0mag and same_bc:
            rows[i] = new_row
            updated = True
            break

    # ---------- Append if new ----------
    if not updated:
        rows.append(new_row)

    # ---------- Sort by q, then Ra ----------
    data_rows = rows[1:]

    data_rows.sort(
        key=lambda r: (float(r[0]), float(r[1]))  # q first, then Ra
    )

    rows = [header] + data_rows

    # ---------- Write back ----------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("========================================")
    print(f"✅ Dynamo summary CSV updated & sorted: {csv_path}")
"""


def write_dynamo_summary_csv(csv_path, q, Ra, Ek, E0mag,  # New intial magnetic energy 
                              dynamo, dipolarity, Elsasser, visDis,  # new viscous dissipation 
                              ohmDis,  # new ohmic dissipation 
                              fohm, L_u,  # new 
                              L_b, T_perb, nusselt, reversal, Rm, 
                              relative_std_fdip,  # fluctuation of the dipolarity 
                              bc_mag, bc_temp, bc_vel, N, M, L, 
                              atol=1e-10):  # 改用绝对容差
    """ 
    Write or update dynamo diagnostics in a CSV file.
    If (q, Ra, Ek) already exists (exact match), overwrite that row.
    Otherwise, append a new row.
    """
    
    # -------- Compact, readable header --------
    header = [
        "q", "Ra", "Ek", "E0mag", "dyn", "fdip", "Lambda", 
        "visDis", "ohmDis", "fohm", "L_u", "L_b", "T_perb", 
        "Nu", "rev", "Rm", "relative_std_fdip",
        "bc_mag", "bc_temp", "bc_vel", "N", "M", "L"
    ]
    
    # 格式化的字符串表示，用于精确比较
    q_str = f"{q:.2f}"
    Ra_str = f"{Ra:.2e}"
    Ek_str = f"{Ek:.2e}"
    
    new_row = [
        q_str, Ra_str, Ek_str, f"{E0mag:.2e}",
        int(dynamo), f"{dipolarity:.2f}", f"{Elsasser:.2e}", 
        f"{visDis:.2e}", f"{ohmDis:.2e}", f"{fohm:.3f}", 
        f"{L_u:.2e}", f"{L_b:.2e}", f"{T_perb:.3e}", 
        f"{nusselt:.2f}" if not np.isnan(nusselt) else "nan", 
        int(reversal) if not np.isnan(reversal) else "nan", 
        f"{Rm:.2f}", 
        f"{relative_std_fdip:.2f}",
        bc_mag, bc_temp, bc_vel, 
        int(N), int(M), int(L)
    ]
    
    # ---------- Read existing file ----------
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # If file is empty or has no header, initialize
        if len(rows) == 0:
            rows = [header]
    
    # Find and update matching row - 使用精确字符串匹配
    updated = False
    for i in range(1, len(rows)):
        # Skip rows that are too short
        if len(rows[i]) < 3:
            continue
            
        # 使用字符串精确匹配（最可靠）
        q_match = (rows[i][0] == q_str)
        Ra_match = (rows[i][1] == Ra_str)
        Ek_match = (rows[i][2] == Ek_str)
        
        # 或者使用数值比较（备用方案）
        if not (q_match and Ra_match and Ek_match):
            try:
                q_i = float(rows[i][0])
                Ra_i = float(rows[i][1])
                Ek_i = float(rows[i][2])
                
                # 使用更小的容差
                if (abs(q_i - q) < 1e-6 and 
                    abs(Ra_i - Ra) < 1e-6 * abs(Ra) and 
                    abs(Ek_i - Ek) < 1e-6 * abs(Ek)):
                    q_match = Ra_match = Ek_match = True
            except (ValueError, IndexError):
                pass
        
        # 如果匹配，更新这一行
        if q_match and Ra_match and Ek_match:
            # 保留原有的行，只更新数值部分
            rows[i] = new_row.copy()
            updated = True
            print(f"🔄 Updated: q={q_str}, Ra={Ra_str}, Ek={Ek_str}")
            break
    
    # Append if not found
    if not updated:
        rows.append(new_row)
        print(f"➕ Added: q={q_str}, Ra={Ra_str}, Ek={Ek_str}")
    
    # ---------- Sort by q, then Ra ----------
    if len(rows) > 1:
        # Extract and sort data rows (skip header)
        data_rows = rows[1:]
        
        # Sort using numerical values
        try:
            data_rows.sort(key=lambda r: (float(r[0]), float(r[1])))
        except (ValueError, IndexError):
            # Fallback: sort only valid rows
            valid_rows = []
            invalid_rows = []
            for r in data_rows:
                try:
                    if len(r) >= 2:
                        q_val = float(r[0])
                        ra_val = float(r[1])
                        valid_rows.append((q_val, ra_val, r))
                    else:
                        invalid_rows.append(r)
                except ValueError:
                    invalid_rows.append(r)
            
            valid_rows.sort(key=lambda x: (x[0], x[1]))
            data_rows = [r[2] for r in valid_rows] + invalid_rows
        
        rows = [header] + data_rows
    
    # ---------- Write back ----------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"✅ CSV updated: {csv_path} (total entries: {len(rows)-1})")
    print("========================================")




def plot_timeseries(folderFile, save_dir, show=True, xlim=None, ylim=None):
    """
    Plot full timeseries of kinetic/magnetic energy, dipolarity, dipole angle, and dissipations.

    Args:
        folderFile: path containing run folders (run0, run1, ...)
        show: whether to display the figure immediately
        xlim
        ylim 

    Returns:
        fig, axes
    """
    # Get all run folders
    #RunFolders = sorted(glob.iglob(os.path.join(folderFile, 'run*')))
    
    RunFolders = _discover_run_folders(folderFile)


    # Read timeseries
    tkin, kinEtot, kinEtor, kinEpol    = F_conc_timeseries(RunFolders, 'kinE')
    tmag, magEtot, magEtor, magEpol    = F_conc_timeseries(RunFolders, 'magE')
    ttem, temEtot                      = F_conc_timeseries(RunFolders, 'temE')
    tnus, nusselt                      = F_conc_timeseries(RunFolders, 'Nusselt')     
    tdip, fdip, g10, g11, h11          = F_conc_timeseries(RunFolders, 'Dip')

    # Read dissipation 
    tkinDis, kinDtot, kinDtor, kinDpol = safe_conc_timeseries(RunFolders, 'kinDis')
    tmagDis, magDtot, magDtor, magDpol = safe_conc_timeseries(RunFolders, 'magDis')

    #tkinDis, kinDtot, kinDtor, kinDpol = F_conc_timeseries(RunFolders, 'kinDis')
    #tmagDis, magDtot, magDtor, magDpol = F_conc_timeseries(RunFolders, 'magDis')
   
    # Calculate dipole angle
    dipangle = np.arccos(g10 / np.sqrt(g10**2 + g11**2 + h11**2)) * 180 / np.pi

    # Read simulation parameters
    Ek, Pm, Pr, q, Ra, Ro   = get_parameters(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    Nres, Mres, Lres        = get_resolution(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    bc_mag, bc_temp, bc_vel = get_boundary_conditions(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
  
    # Compute time-averaged Rossby number
    t = tkin
    Ro = Ek * np.sqrt(2 * kinEtot)
    #startindex = int(0.3 * len(Ro))
    startindex = np.where(t >= t[0] + 0.3 * (t[-1] - t[0]))[0][0]
    timeavg_Ro = np.round(np.mean(Ro[startindex:]), 2)
 

    #physcial kientic energy
    if  Pm !=0 and Pr !=0: 
        kinEtot = Ek/Pm * kinEtot 
       
    # ------ Create figure------#
    has_dis = (len(tkinDis) > 0) or (len(tmagDis) > 0)

    plt.close('all')
    if has_dis:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 12), sharex=True, dpi=180)
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi=180)
        ax5 = None

    #plt.close('all')
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi =180)
    
    
    ax1.set_title(f'$E: {Ek:.1e}, Ra: {Ra:.2e}, q: {q:.2f}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')
    
    # compute magnetic Reynolds number
    Rm = np.sqrt(np.mean(kinEtot))

    print('Input parameters:')
    print('Ek:', Ek)
    print('Ra:', Ra)
    print('q:', q)
    print('-------------------------------')
    print('output diagnostics:')
    print('magnetic Reynolds number Rm=', int(Rm))

    #print('Reynolds number', '%.2E' %(Rm/Pm))
    #print('Rossby number', '%.2E' %((2*Rm*Ek)/Pm))
   

    # kinetic Energy plot
    ax1.plot(tkin, kinEtot, label=r'$\mathcal{E}_{kin}$')
    #ax1.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')

    ax1.set_yscale('log')
    ax1.set_ylabel('Energy density')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax1.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax1.set_xlim(xlim)        

    if ylim is not None:
       ax1.set_ylim(ylim)    
    ax1.legend()


   
    # # magnetic Energy plot
    # ax2.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')
    # ax2.set_yscale('log')
    # ax2.set_ylabel('Energy density')
    # if xlim is None:
    #    xlim = (np.max(t) - np.min(t)) * 0.05
    #    ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    # else:
    #    ax2.set_xlim(xlim)        

    # if ylim is not None:
    #    ax2.set_ylim(ylim)    
    # ax2.legend()


    # thermal perturbation 
    T_perb = 3/5* (q*Ra)**2 * temEtot  #Et

    timeavg_T_perb = np.mean(T_perb[startindex:])
    print(f"time-averaged  T_perb = {timeavg_T_perb:.2e}")
    ax2.plot(ttem, Ek*T_perb, label=r'$E \mathcal{E}_{t}$')
    #ax2.set_ylabel('Thermal perturbation')
    
    ax2.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')
    ax2.set_ylabel('Energy density')
    ax2.set_yscale('log')
    if xlim is None:
        xlim = (np.max(t) - np.min(t)) * 0.05
        ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
        ax2.set_xlim(xlim)        

    if ylim is not None:
        ax2.set_ylim(ylim)    
    ax2.legend()



    #intial magnetic energy
    magEtoti = magEtot[0]

    #compute time-averaged Energy kinEtot
    #startindex = int(0.3 * len(kinEtot))
    startindex = np.where(t >= t[0] + 0.3 * (t[-1] - t[0]))[0][0]
    timeavg_kinEtot = np.mean(kinEtot[startindex:])
    print(f"time-averaged kinetic energy  = {timeavg_kinEtot:.2e}")

    #compute time-averaged Energy magEtot
    timeavg_magEtot = np.mean(magEtot[startindex:])
    print(f"time-averaged magnetic energy = {timeavg_magEtot:.2e}")

    #Time averaged Elssaser number
    Lambda = 2*timeavg_magEtot
    print(f"Elsasser number Lambda= {Lambda:.2e}")

    #compute time-averaged Nusselt
    timeavg_nusselt = np.mean(nusselt[startindex:])
    print(f"time-averaged nusselt = {timeavg_nusselt:.2e}")

 
    #Determine if dynamo is active
    dynamo_threshold = 1e-4
    if magEtot[-1] < dynamo_threshold or Rm < 30:
        dynamo =  0
        print(f"Dynamo is dead : {dynamo}")
        if Rm < 50:
            print(f"Rm is too small: {Rm}")
        else:
            print(f"Em is too small: {magEtot[-1]}")   
    else:
        dynamo =  1
        print(f"Dynamo is active ✅ : {dynamo} ")

    # Dipolarity
    if len(tdip) > 0:
        ax3.plot(tdip, fdip, color='red', label='g10', alpha=0.6)
    ax3.set_ylabel('Dipolarity')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax3.set_xlim(xlim)  
    ax3.set_ylim(0, 1)

    # compute time-averaged Dipolarity
    timeavg_fdip = np.mean(fdip[startindex:])  

    # compute relative fluctuation: std(fdip) / mean(fdip)
    relative_std_fdip = np.std(fdip[startindex:]) / timeavg_fdip

    print(f"Time-averaged dipolarity: {timeavg_fdip:.2e}")
    print(f"std(dipolarity)/mean(dipolarity): {relative_std_fdip:.2e}")

    # Dipole latitude
    if len(g10) > 0:
        ax4.plot(tdip, dipangle, 'k', alpha=0.7)
    ax4.set_ylabel('Dipole angle (deg)')
    ax4.set_ylim(-5, 185)
    ax4.axhline(90, color='gray', linestyle='--', alpha=0.4, label=r'$90^{\circ}$')
    #ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax4.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax4.set_xlim(xlim)  
    ax4.legend()

    # Determine if reversal occurs by looking dipole angle crossing 90 degrees
    # from < 90 to >90 or vice versa 
    # only use data from startindex to the end
    #reversal = 0    
    #dipangle_subset = dipangle[startindex:]
    #for i in range(1, len(dipangle_subset)):
    #    if (dipangle_subset[i-1] < 90 and dipangle_subset[i] >= 90) or \
    #       (dipangle_subset[i-1] > 90 and dipangle_subset[i] <= 90):
    #        reversal = 1
    #        break
    
    reversal = 0
    excursion = 0

    # only use data from startindex to the end
    dipangle_subset = dipangle[startindex:]

    # check for excursion
    for i in range(1, len(dipangle_subset)):
        if (dipangle_subset[i-1] < 90 and dipangle_subset[i] >= 90) or \
           (dipangle_subset[i-1] > 90 and dipangle_subset[i] <= 90):
            excursion = 1
            break

    print('Excursion:', excursion) 
    
    # check for reversal (need updating)
    # Earth-like criteria: dipole angle goes beyond 150 deg and below 30 deg, and dipolarity >0.35 (empirical)
    if excursion == 1:
        if np.max(dipangle_subset) > 150 and np.min(dipangle_subset) < 30:
            if  timeavg_fdip > 0.30:
                reversal = 1

    print('Reversal (timeavg_fdip > 0.30):', reversal)

  
    if has_dis and ax5 is not None:
        if len(tkinDis) > 0:
            ax5.plot(tkinDis, Ek * kinDtot, alpha=0.7, label='viscous dissipation')
        if len(tmagDis) > 0:
            ax5.plot(tmagDis, magDtot, alpha=0.7, label='ohmic dissipation')

        ax5.set_ylabel('Dissipation')
        ax5.set_xlabel('Time')
        ax5.set_yscale('log')

        if xlim is None:
            xlim5 = (np.max(t) - np.min(t)) * 0.05
            ax5.set_xlim(np.min(t) - xlim5, np.max(t) + xlim5)
        else:
            ax5.set_xlim(xlim)

        ax5.legend()

    
    # averaged viscous dissipation and ohmic dissipation
    kinDtot_avg  =  Ek*np.mean(kinDtot[:])
    magDtot_avg  =     np.mean(magDtot[:])
    fohm =  magDtot_avg/( magDtot_avg+kinDtot_avg) 

    # compute typical lemgth scale:
    L_u = np.sqrt(timeavg_kinEtot*Ek/kinDtot_avg)
    L_b = np.sqrt(timeavg_magEtot/magDtot_avg)

    print(f"viscous dissipation: {kinDtot_avg:.2e}" )
    print(f"ohmic   dissipation: {magDtot_avg :.2e}" )
    print(f"fraction of ohmic Dis: {fohm :.2e}" )
    print(f" typical length scale of u: {L_u :.2e}" )
    print(f" typical length scale of B: {L_b :.2e}" )

    # ---------- Write summary CSV ---------- #
    # --- determine Ek root automatically ---
    Ek_root = extract_Ek_root(folderFile)
    os.makedirs(Ek_root, exist_ok=True)

    csv_name = "diagnostics/"+f"data_E_{Ek:.1e}_E0mag_Dis_L_T_perb_Nu_Bc_stdfdip.csv"
    csv_path = os.path.join(Ek_root, csv_name)

    write_dynamo_summary_csv(
        csv_path=csv_path,
        q=q,
        Ra=Ra,
        Ek=Ek,
        E0mag = magEtoti,
        dynamo=dynamo,
        dipolarity=timeavg_fdip,
        Elsasser=Lambda,
        visDis = kinDtot_avg,
        ohmDis = magDtot_avg,
        fohm = fohm,
        L_u = L_u,          #lengthscale for u
        L_b = L_b,          #lengthscale for b
        T_perb= timeavg_T_perb,     #new added thermal perturbation
        nusselt= timeavg_nusselt,
        reversal=reversal,
        Rm=Rm,
        relative_std_fdip = relative_std_fdip,    # std of dipolarity / average dipolarity
        bc_mag = bc_mag,  # boundary condition
        bc_temp= bc_temp, 
        bc_vel = bc_vel,
        N = Nres,
        M = Mres,
        L = Lres
    )

    #save figure
    Ek,q,Ra = input_params_from_path(folderFile)
    
    save_path = os.path.join(save_dir, f'Ek_{Ek}_q{q}_Ra{Ra}_timeseries.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()
    
    if has_dis:
        return fig, (ax1, ax2, ax3, ax4, ax5)
    else:
        return fig, (ax1, ax2, ax3, ax4)
#-----------------------------------------



def plot_timeseries_dipolarity(folderFile, save_dir, show=True, xlim=None, ylim=None):
    """
    Plot full timeseries of kinetic/magnetic energy, dipolarity, dipole angle, and dissipations.


    Returns:
        fig, axes
    """
    # Get all run folders
    #RunFolders = sorted(glob.iglob(os.path.join(folderFile, 'run*')))
    
    RunFolders = _discover_run_folders(folderFile)


    # Read timeseries
    tkin, kinEtot, kinEtor, kinEpol    = F_conc_timeseries(RunFolders, 'kinE')
    tmag, magEtot, magEtor, magEpol    = F_conc_timeseries(RunFolders, 'magE')
    ttem, temEtot                      = F_conc_timeseries(RunFolders, 'temE')
    tnus, nusselt                      = F_conc_timeseries(RunFolders, 'Nusselt')     
    tdip, fdip, g10, g11, h11          = F_conc_timeseries(RunFolders, 'Dip')

    # Read dissipation 
    tkinDis, kinDtot, kinDtor, kinDpol = safe_conc_timeseries(RunFolders, 'kinDis')
    tmagDis, magDtot, magDtor, magDpol = safe_conc_timeseries(RunFolders, 'magDis')

    #tkinDis, kinDtot, kinDtor, kinDpol = F_conc_timeseries(RunFolders, 'kinDis')
    #tmagDis, magDtot, magDtor, magDpol = F_conc_timeseries(RunFolders, 'magDis')
   
    # Calculate dipole angle
    dipangle = np.arccos(g10 / np.sqrt(g10**2 + g11**2 + h11**2)) * 180 / np.pi

    # Read simulation parameters
    Ek, Pm, Pr, q, Ra, Ro   = get_parameters(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    Nres, Mres, Lres        = get_resolution(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    bc_mag, bc_temp, bc_vel = get_boundary_conditions(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
  
    # Compute time-averaged Rossby number
    t = tkin
    Ro = Ek * np.sqrt(2 * kinEtot)
    #startindex = int(0.3 * len(Ro))
    startindex = np.where(t >= t[0] + 0.3 * (t[-1] - t[0]))[0][0]
    timeavg_Ro = np.round(np.mean(Ro[startindex:]), 2)
 

    #physcial kientic energy
    if  Pm !=0 and Pr !=0: 
        kinEtot = Ek/Pm * kinEtot 
       
    # ------ Create figure------#
    has_dis = (len(tkinDis) > 0) or (len(tmagDis) > 0)

    plt.close('all')
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 4.5), sharex=True, dpi=180)

    #plt.close('all')
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True, dpi =180)
    
    
    ax0.set_title(f'$E: {Ek:.1e}, Ra: {Ra:.2e}, q: {q:.2f}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')
    
    # compute magnetic Reynolds number
    Rm = np.sqrt(np.mean(kinEtot))

    print('Input parameters:')
    print('Ek:', Ek)
    print('Ra:', Ra)
    print('q:', q)
    print('-------------------------------')
    print('output diagnostics:')
    print('magnetic Reynolds number Rm=', int(Rm))

    
    ax0.plot(tmag, magEtot, color= 'darkorange', label=r'$\mathcal{E}_{mag}$')
    ax0.set_ylabel(r'$E_{mag}$')
    ax0.set_yscale('log')
    if xlim is None:
        xlim = (np.max(t) - np.min(t)) * 0.05
        ax0.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
        ax0.set_xlim(xlim)        

    if ylim is not None:
        ax0.set_ylim(ylim)    
    ax0.legend()


    # Dipolarity
    if len(tdip) > 0:
        ax1.plot(tdip, fdip, color='red', label='g10', alpha=0.6)
    ax1.set_ylabel(r'$f_{dip}$')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax1.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax1.set_xlim(xlim)  
    ax1.set_ylim(0, 1)

    # compute time-averaged Dipolarity
    timeavg_fdip = np.mean(fdip[startindex:])  

    # compute relative fluctuation: std(fdip) / mean(fdip)
    relative_std_fdip = np.std(fdip[startindex:]) / timeavg_fdip

    print(f"Time-averaged dipolarity: {timeavg_fdip:.2e}")
    print(f"std(dipolarity)/mean(dipolarity): {relative_std_fdip:.2e}")

    # Dipole latitude
    if len(g10) > 0:
        ax2.plot(tdip, dipangle, 'k', alpha=0.7)
    ax2.set_ylabel(r'$\theta$ (deg)')
    ax2.set_ylim(-5, 185)
    ax2.axhline(90, color='gray', linestyle='--', alpha=0.4, label=r'$90^{\circ}$')
    #ax3.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax2.set_xlim(xlim)  
    ax2.legend()

    ax2.set_xlabel('Time')

    # Determine if reversal occurs by looking dipole angle crossing 90 degrees
    # from < 90 to >90 or vice versa 
    # only use data from startindex to the end
    #reversal = 0    
    #dipangle_subset = dipangle[startindex:]
    #for i in range(1, len(dipangle_subset)):
    #    if (dipangle_subset[i-1] < 90 and dipangle_subset[i] >= 90) or \
    #       (dipangle_subset[i-1] > 90 and dipangle_subset[i] <= 90):
    #        reversal = 1
    #        break
    
    reversal = 0
    excursion = 0

    # only use data from startindex to the end
    dipangle_subset = dipangle[startindex:]

    # check for excursion
    for i in range(1, len(dipangle_subset)):
        if (dipangle_subset[i-1] < 90 and dipangle_subset[i] >= 90) or \
           (dipangle_subset[i-1] > 90 and dipangle_subset[i] <= 90):
            excursion = 1
            break

    print('Excursion:', excursion) 
    
    # check for reversal (need updating)
    # Earth-like criteria: dipole angle goes beyond 150 deg and below 30 deg, and dipolarity >0.35 (empirical)
    if excursion == 1:
        if np.max(dipangle_subset) > 150 and np.min(dipangle_subset) < 30:
            if  timeavg_fdip > 0.30:
                reversal = 1

    print('Reversal (timeavg_fdip > 0.30):', reversal)

    #save figure
    Ek,q,Ra = input_params_from_path(folderFile)
    
    save_path = os.path.join(save_dir, f'Ek_{Ek}_q{q}_Ra{Ra}_timeseries_Emag_fip_tiltAngle.png')
   
    plt.savefig( save_path,dpi=270, bbox_inches='tight')

    if show:
        plt.show()
    
    return fig, (ax0, ax1, ax2)
#-----------------------------------------


#---------------------------------------
def compare_energy(folderFile1, folderFile2, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare kinetic and magnetic energies between two simulation runs.

    Args:
        folderFile1: Path to the first run folder or a directory containing multiple runs (e.g., 'run0', 'run1', ...)
        folderFile2: Path to the second run folder or directory
        save_dir: Directory to save the comparison figure (optional)
        show: Whether to display the figure
        xlim: (xmin, xmax) for time axis
        ylim: (ymin, ymax) for energy axis

    Returns:
        fig, axes
    """
    # Get all run folders inside each directory
    RunFolders1 = sorted(glob.iglob(os.path.join(folderFile1, 'run*')))
    RunFolders2 = sorted(glob.iglob(os.path.join(folderFile2, 'run*')))

    # Read concatenated timeseries
    tkin1, kinEtot1, kinEtor1, kinEpol1 = F_conc_timeseries(RunFolders1, 'kinE')
    tmag1, magEtot1, magEtor1, magEpol1 = F_conc_timeseries(RunFolders1, 'magE')

    tkin2, kinEtot2, kinEtor2, kinEpol2 = F_conc_timeseries(RunFolders2, 'kinE')
    tmag2, magEtot2, magEtor2, magEpol2 = F_conc_timeseries(RunFolders2, 'magE')

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    # Plot kinetic energy comparison
    axes[0].plot(tkin1, kinEtot1, label='Run 1: Kinetic_total', color='tab:blue', lw=1.8)
    axes[0].plot(tkin2, kinEtot2, label='Run 2: Kinetic_total', color='tab:orange', lw=1.8, ls='--')
    axes[0].set_ylabel('Kinetic Energy')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot magnetic energy comparison
    axes[1].plot(tmag1, magEtot1, label='Run 1: Magnetic_total', color='tab:green', lw=1.8)
    axes[1].plot(tmag2, magEtot2, label='Run 2: Magnetic_total', color='tab:red', lw=1.8, ls='--')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Magnetic Energy')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Set limits
    if xlim: axes[1].set_xlim(xlim)
    if ylim: 
        for ax in axes:
            ax.set_ylim(ylim)

    # Save figure
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'compare_energy.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Energy comparison figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def compare_energy_multi(run_paths, labels=None, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare kinetic and magnetic energies from multiple runs.

    Args:
        run_paths: list of paths (each can be a single run or a folder containing run0, run1, ...)
        labels: list of legend labels (optional). If None, folder names will be used.
        save_dir: directory to save the comparison figure
        show: display figure
        xlim: (xmin, xmax)
        ylim: (ymin, ymax) applied to ALL subplots

    Returns:
        fig, axes
    """

    # Validate input
    if not isinstance(run_paths, (list, tuple)):
        raise ValueError("run_paths must be a list of paths")
    if labels and len(labels) != len(run_paths):
        raise ValueError("labels length must match run_paths length")

    # Auto-generate labels if needed
    if labels is None:
        labels = [os.path.basename(p.rstrip('/')) for p in run_paths]

    # Prepare plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    # Colormap for multiple runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_paths)))

    # ---------- Loop over ALL runs ----------
    for idx, path in enumerate(run_paths):

        # Find run folders
        if os.path.isdir(path):
            runs = sorted(glob.iglob(os.path.join(path, "run*")))
            if len(runs) == 0:
                # Maybe user passed a single file instead of folder
                runs = [path]
        else:
            runs = [path]

        # Read timeseries
        tkin, kinEtot, _, _ = F_conc_timeseries(runs, "kinE")
        tmag, magEtot, _, _ = F_conc_timeseries(runs, "magE")
        
        # Read simulation parameters
        Ek, Pm, Pr, q, Ra, Ro = get_parameters(os.path.join(runs[0], 'parameters.cfg'), 'no')
        if  Pm !=0 and Pr !=0: 
            kinEtot = Ek/Pm * kinEtot 

        # ---------- Plot kinetic ----------
        axes[0].semilogy(tkin, kinEtot,
                     label=f"{labels[idx]} - Kinetic",
                     #color=colors[idx], 
                     lw=1.8)

        # ---------- Plot magnetic ----------
        axes[1].semilogy(tmag, magEtot,
                     label=f"{labels[idx]} - Magnetic",
                     #color=colors[idx], 
                     lw=1.8)

    # ---------- Styling ----------
    axes[0].set_ylabel("Kinetic Energy")
    axes[1].set_ylabel("Magnetic Energy")
    axes[1].set_xlabel("Time")

    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.35)

        if ylim:
            ax.set_ylim(ylim)


    if xlim is None:
        xlim = (np.max(tkin) - np.min(tkin)) * 0.05
        axes[1].set_xlim(xlim)
    else:
        axes[1].set_xlim(xlim)


    # ---------- Save figure ----------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, "compare_energy_multi.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


##### timestep ######
def read_cfl_file(fname):
    """
    Read a cfl.dat file.
    
    Returns:
        time: array
        dt: array
    """
    time = []
    dt = []

    if not os.path.exists(fname):
        return np.array([]), np.array([])

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            time.append(float(parts[0]))
            dt.append(float(parts[1]))

    return np.array(time), np.array(dt)



def F_conc_cfl(run_folders):
    """
    Concatenate cfl.dat time series across multiple restart folders.

    This is the CFL analogue of F_conc_timeseries.
    
    Returns:
        time, dt
    """

    time = np.array([])
    dt = np.array([])

    for i, folder in enumerate(run_folders):

        lastval = time[-1] if time.size > 0 else -np.inf

        tnew, dtnew = read_cfl_file(os.path.join(folder, "cfl.dat"))
        if tnew.size == 0:
            continue

        mask = tnew > lastval
        if not np.any(mask):
            continue

        start_idx = np.argmax(mask)

        time = np.concatenate((time, tnew[start_idx:]))
        dt = np.concatenate((dt, dtnew[start_idx:]))

    return time, dt



def compare_timestep_multi(run_paths, labels=None, save_dir=None, show=True, xlim=None, ylim=None):
    """
    Compare CFL time step evolution from multiple runs.

    Args:
        run_paths: list of paths (each can be a single run or a folder with run0, run1,...)
        labels: legend labels
        save_dir: optional output directory
        show: show plot
        xlim: (xmin, xmax)
        ylim: (ymin, ymax)

    Returns:
        fig, ax
    """

    if not isinstance(run_paths, (list, tuple)):
        raise ValueError("run_paths must be a list")

    if labels and len(labels) != len(run_paths):
        raise ValueError("labels length must match run_paths")

    if labels is None:
        labels = [os.path.basename(p.rstrip('/')) for p in run_paths]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_paths)))

    for idx, path in enumerate(run_paths):

        # discover run folders
        if os.path.isdir(path):
            runs = sorted(glob.iglob(os.path.join(path, "run*")))
            if len(runs) == 0:
                runs = [path]
        else:
            runs = [path]

        # read concatenated CFL
        t, dt = F_conc_cfl(runs)

        if t.size == 0:
            print(f"Warning: no cfl.dat found in {path}")
            continue

        # plot
        ax.semilogy(t, dt, lw=1.8, label=labels[idx])

    # style
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep (dt)")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=7)

    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, "compare_timestep_multi.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax




def plot_timeseries_hydro(folderFile, save_dir, show=True, xlim=None, ylim=None):
    """
    Plot full timeseries for pure hydro case (thermal convection)

    Args:
        folderFile: path containing run folders (run0, run1, ...)
        show: whether to display the figure immediately
        xlim
        ylim 

    Returns:
        fig, axes
    """
    # Get all run folders
    #RunFolders = sorted(glob.iglob(os.path.join(folderFile, 'run*')))
    RunFolders = _discover_run_folders(folderFile)

    # Read timeseries
    tkin, kinEtot, kinEtor, kinEpol    = F_conc_timeseries(RunFolders, 'kinE')
    ttem, temEtot                      = F_conc_timeseries(RunFolders, 'temE')
    tnus, nusselt                      = F_conc_timeseries(RunFolders, 'Nusselt')     

    # Read dissipation 
    tkinDis, kinDtot, kinDtor, kinDpol = safe_conc_timeseries(RunFolders, 'kinDis')
   
    # Read simulation parameters
    Ek, Pm, Pr, q, Ra, Ro   = get_parameters(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
    Nres, Mres, Lres        = get_resolution(os.path.join(RunFolders[0], 'parameters.cfg'), 'no')
  
    t = tkin
    #startindex = int(0.3 * len(Ro))
    startindex = np.where(t >= t[0] + 0.3 * (t[-1] - t[0]))[0][0]

    #physcial kientic energy
    if  Pm !=0 and Pr !=0: 
        kinEtot = Ek/Pm * kinEtot 
       
    # ------ Create figure------#
    has_dis = (len(tkinDis) > 0) 

    plt.close('all')
    if has_dis:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, dpi=180)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=180)
        ax5 = None
    

    ax1.set_title(f'Hydro, $E: {Ek:.1e}, Ra: {Ra:.2e}, (N,L,M): ({Nres:.0f},{Mres:.0f},{Lres:.0f})$')
    
    # compute magnetic Reynolds number
    Rm = np.sqrt(np.mean(kinEtot))

    print('Input parameters:')
    print('Ek:', Ek)
    print('Ra:', Ra)
    print('-------------------------------')
    print('output diagnostics:')


    # kinetic Energy plot
    ax1.plot(tkin, kinEtot, label=r'$\mathcal{E}_{kin}$')

    ax1.set_yscale('log')
    ax1.yaxis.set_major_locator(LogLocator(base=10))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1e}"))

    ax1.set_ylabel('Energy density')
    if xlim is None:
       xlim = (np.max(t) - np.min(t)) * 0.05
       ax1.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
       ax1.set_xlim(xlim)        

    if ylim is not None:
       ax1.set_ylim(ylim)    
    ax1.legend()


    # thermal perturbation 
    T_perb = 3/5* (q*Ra)**2 * temEtot  #Et
    timeavg_T_perb = np.mean(T_perb[startindex:])

    print(f"time-averaged  T_perb = {timeavg_T_perb:.2e}")
    ax2.plot(ttem, Ek*T_perb, label=r'$E \mathcal{E}_{t}$', color = 'orange')
    
    #ax2.plot(tmag, magEtot, label=r'$\mathcal{E}_{mag}$')
    ax2.set_ylabel('Energy density')
    ax2.set_yscale('log')

    ax2.yaxis.set_major_locator(LogLocator(base=10))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1e}"))
   
    #ax2.yaxis.set_major_formatter(LogFormatterMathtext())

    if xlim is None:
        xlim = (np.max(t) - np.min(t)) * 0.05
        ax2.set_xlim(np.min(t)-xlim, np.max(t)+xlim)
    else:
        ax2.set_xlim(xlim)        

    if ylim is not None:
        ax2.set_ylim(ylim)    
    ax2.legend()
    #ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2e}"))

    #compute time-averaged Energy kinEtot
    startindex = np.where(t >= t[0] + 0.3 * (t[-1] - t[0]))[0][0]
    timeavg_kinEtot = np.mean(kinEtot[startindex:])
    print(f"time-averaged kinetic energy  = {timeavg_kinEtot:.2e}")

    #compute time-averaged Nusselt
    timeavg_nusselt = np.mean(nusselt[startindex:])
    print(f"time-averaged nusselt = {timeavg_nusselt:.2e}")


    if has_dis and ax3 is not None:
        if len(tkinDis) > 0:
            ax5.plot(tkinDis, Ek * kinDtot, alpha=0.7, label='viscous dissipation')


        ax3.set_ylabel('Dissipation')
        ax3.set_xlabel('Time')
        ax3.set_yscale('log')

        if xlim is None:
            xlim5 = (np.max(t) - np.min(t)) * 0.05
            ax3.set_xlim(np.min(t) - xlim5, np.max(t) + xlim5)
        else:
            ax3.set_xlim(xlim)

        ax3.legend()

    save_path = os.path.join(save_dir, f'Ek_{Ek}_Ra{Ra}_timeseries.png')
    plt.savefig(save_path, dpi=270)

    if show:
        plt.show()
    
    if has_dis:
        return fig, (ax1, ax2, ax3)
    else:
        return fig, (ax1, ax2)
#-----------------------------------------


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot timeseries from QuICC runs.")
    parser.add_argument("folder", help="Path to the start folder containing run subfolders")
    parser.add_argument("--save", help="Path to save the figure", default=None)
    args = parser.parse_args()

    plot_timeseries(start_folder=args.folder, show=True, save_path=args.save)
    
    # TO RUN it in terminal:
    #python -m quicc_dynavis.timeseries /path/to/my/simulation --save timeseries.png
