import numpy as np
import matplotlib.pyplot as plt

field_latex = {
    "u_r": r"u_r",
    "u_theta": r"u_\theta",
    "u_phi": r"u_\phi",
    "B_r": r"B_r",
    "B_theta": r"B_\theta",
    "B_phi": r"B_\phi",
    "temperature": r"T"
}


def _get_color_limits(field, sym_cbar):
    """
        Return vmin, vmax (symmetric if requested).
    """
    if sym_cbar:
        absmax = np.nanmax(np.abs(field))
        return -absmax, absmax
    else:
        return np.nanmin(field), np.nanmax(field)
    

def apply_temperature_background(field_name, field_data, r, include_background=False):
    """
        Add background temperature profile (nondimensional)
                    T0 = 0.5 * (1-r^2) 
        if requested.
    """
    if field_name == "T" and include_background:
        T0 = 0.5 * (1 - r**2)
        field_data = field_data + T0[:, None]
    return field_data    

def _add_dashed_circles(ax):
    """
        Add dashed circles at r=1 

    """
    for r in (1.0,):
        circle = plt.Circle((0, 0), r, color='black', fill=False,
                            linestyle='-', linewidth=1.0,
                            zorder=1, clip_on=True)
        ax.add_artist(circle)


def plot_equatorial(data, field_name, title="Equatorial slice", cmap="RdBu_r",
                    ax=None, savefig=None, sym_cbar=True, include_background=False):
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of "u_r", "u_theta", "u_phi", "B_r", "B_theta", "B_phi", "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure

    """
    r, theta, phi = data["r"], data["theta"], data["phi"]
    eq_idx = np.argmin(np.abs(theta - np.pi/2))
    field = data[field_name][:, eq_idx, :]
    field = apply_temperature_background(field_name, field, r, include_background)

     
    vmin, vmax = _get_color_limits(field, sym_cbar)

    R, Phi = np.meshgrid(r, phi, indexing="ij")
    X, Y = R * np.cos(Phi), R * np.sin(Phi)

    # Mask values outside outer boundary (r > 1)
    mask = R > 1.0
    field = np.ma.masked_where(mask, field)   

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")

    _add_dashed_circles(ax)

    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if savefig:
        plt.savefig(f"{savefig}/{field_name}_equ_slice.png", dpi=300, bbox_inches="tight")


def plot_meridional(data, field_name, title="Meridional slice", cmap="RdBu_r", ax=None, 
                    savefig=None, sym_cbar=True, include_background=False):
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of "u_r", "u_theta", "u_phi", "B_r", "B_theta", "B_phi", "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure

    """
    
    r, theta, phi = data["r"], data["theta"], data["phi"]
    mid_phi = np.argmin(np.abs(phi - np.pi/2))
    field = data[field_name][:, :, mid_phi]
    field = apply_temperature_background(field_name, field, r, include_background)


    vmin, vmax = _get_color_limits(field, sym_cbar)

    R, Theta = np.meshgrid(r, theta, indexing="ij")
    X, Z = R * np.sin(Theta), R * np.cos(Theta)

    # Mask values outside outer boundary (r > 1)
    mask = R > 1.0
    field = np.ma.masked_where(mask, field)   

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(X, Z, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add dashed circles
    _add_dashed_circles(ax)

    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)



    if savefig:
        plt.savefig(f"{savefig}/{field_name}_merid_slice.png", dpi=300, bbox_inches="tight")


def plot_cmb(data, field_name, title="CMB", cmap="RdBu_r", ax=None, 
             savefig=None, sym_cbar=True, include_background=False):
    
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of "u_r", "u_theta", "u_phi", "B_r", "B_theta", "B_phi", "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure

    """
        
    r, theta, phi = data["r"], data["theta"], data["phi"]
    field = data[field_name][-1, :, :]
    field = apply_temperature_background(field_name, field, r, include_background)

    vmin, vmax = _get_color_limits(field, sym_cbar)

    lon, lat = phi - np.pi, np.pi/2 - theta
    Lon, Lat = np.meshgrid(lon, lat, indexing="ij")

    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection="mollweide")

    im = ax.pcolormesh(Lon, Lat, field.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()


    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    
    if savefig:
        plt.savefig(f"{savefig}/{field_name}_cmb.png", dpi=300, bbox_inches="tight")
