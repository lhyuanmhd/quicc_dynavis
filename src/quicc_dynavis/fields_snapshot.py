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

def plot_equatorial(data, field_name, title="Equatorial slice", cmap="RdBu_r", ax=None, savefig=None):
    r, theta, phi = data["r"], data["theta"], data["phi"] 
    eq_idx = np.argmin(np.abs(theta - np.pi/2))
    field = data[field_name][:, eq_idx, :]

    R, Phi = np.meshgrid(r, phi, indexing="ij")
    X, Y = R * np.cos(Phi), R * np.sin(Phi)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap)
    ax.set_aspect("equal")
    ax.axis("off")
    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if savefig is not None:
        plt.savefig(savefig+'/'+field_name+'_equ_slice_.png', dpi=300, bbox_inches="tight")


def plot_meridional(data, field_name, title="Meridional slice", cmap="RdBu_r", ax=None, savefig=None):
    """
    Plot a meridional slice (half-circle) of the field at mid-phi.
    """
    r, theta, phi = data["r"], data["theta"], data["phi"] 
    mid_phi = np.argmin(np.abs(phi - np.pi/2)) #phi=pi/2
    field = data[field_name][:, :, mid_phi]  # shape (len(r), len(theta))

    R, Theta = np.meshgrid(r, theta, indexing="ij")
    X, Z = R * np.sin(Theta), R * np.cos(Theta)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(X, Z, field, shading="auto", cmap=cmap)
    ax.set_aspect("equal")
    ax.axis("off")
    # LaTeX label mapping
    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if savefig is not None:
       plt.savefig(savefig+'/'+field_name+'_merid_slice_.png', dpi=300, bbox_inches="tight")



def plot_cmb(data, field_name, title="CMB", cmap="RdBu_r", ax=None, savefig=None):
    r, theta, phi = data["r"], data["theta"], data["phi"] 
    field = data[field_name][-1, :, :]
    lon, lat = phi - np.pi, np.pi/2 - theta
    Lon, Lat = np.meshgrid(lon, lat, indexing="ij")

    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection="mollweide")
    im = ax.pcolormesh(Lon, Lat, field.T, shading="auto", cmap=cmap)
    ax.set_axis_off()
    label_str = field_latex.get(field_name, field_name)
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if savefig is not None:
        plt.savefig(savefig+'/'+field_name+'_cmb_.png', dpi=300, bbox_inches="tight")