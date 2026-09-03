import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import RegularGridInterpolator
from .timeseries_utils import input_params_from_path
from matplotlib.ticker import ScalarFormatter

# field_latex = {
#     "u_r": r"u_r",
#     "u_theta": r"u_\theta",
#     "u_phi": r"u_\phi",
#     "B_r": r"B_r",
#     "B_theta": r"B_\theta",
#     "B_phi": r"B_\phi",
#     "temperature": r"T"
# }

field_latex = {
    # velocity
    "u_r": r"u_r",
    "u_theta": r"u_\theta",
    "u_phi": r"u_\phi",
    "u_phi_zonal_3d": r"\langle u_\phi \rangle_\phi",

    "thermal_wind_r": r"$u_{T,r}",
    "magnetic_wind_r": r"u_{M,r}",

    # magnetic field
    "B_r": r"B_r",
    "B_theta": r"B_\theta",
    "B_phi": r"B_\phi",

    # temperature
    "temperature": r"T",

    # vorticity / curl(u)
    "curl_u_r": r"(\nabla\times\mathbf{u})_r",
    "curl_u_theta": r"(\nabla\times\mathbf{u})_\theta",
    "curl_u_phi": r"(\nabla\times\mathbf{u})_\phi",
    "curl_u_axial": r"(\nabla\times\mathbf{u})_z",

    # magnetic curl (if you plot these)
    "curlB_r": r"(\nabla\times\mathbf{B})_r",
    "curlB_theta": r"(\nabla\times\mathbf{B})_\theta",
    "curlB_phi": r"(\nabla\times\mathbf{B})_\phi",


    
    # nonlinear advection
    #"u_dot_grad_u_magnitude": r"\left|E_\eta \mathbf{u}\cdot\nabla\mathbf{u}\right|",
    
    #coriolis
    #"coriolis_magnitude": r"\left|\hat{z} \times \mathbf{u}\right|", 

    }


def savefig_field_snapshot(folderFile, field_name, savefig, type="meridional"):
   Ek,q,Ra = input_params_from_path(folderFile)
   savepath = f"{savefig}/Ek_{Ek}_q{q}_Ra{Ra}_{field_name}_{type}.png"
   plt.savefig(savepath, dpi=300, bbox_inches="tight")
   print(f"Saved figure: {savepath}")


# set colomap for each field
def cmap_for_field(field_name):
    if field_name in ["u_r", "u_theta", "u_phi", "u_phi_zonal_3d", "thermal_wind_r",  "magnetic_wind_r" ]:
        return "RdBu_r"
    elif field_name in ["B_r", "B_theta", "B_phi"]:
        return "PuOr"
    elif field_name == "temperature":
        return "gist_heat"
    elif field_name in [ "curl_u_r",  "curl_u_theta",  "curl_u_phi", "curl_u_axial"]:
        return "PRGn"
    elif field_name in ["inertia_magnitude", 
                        "coriolis_magnitude", 
                        "lorentz_magnitude", 
                        "buoyancy_magnitude",
                        "viscous_magnitude"]:
        return "cividis" 

    #elif field_name == "coriolis_magnitude":
    #    return "magma"    


#def _get_color_limits(field, sym_cbar):
#    """
#        Return vmin, vmax (symmetric if requested).
#    """
#    if sym_cbar:
#        absmax = np.nanmax(np.abs(field))
#        if field in ["B_r", "B_theta", "B_phi"]:
#            fc = 0.5
#            return -fc*absmax, fc*absmax
#        else:
#            return -absmax, absmax
#    else:
#        return np.nanmin(field), np.nanmax(field)

def _get_color_limits(field, sym_cbar, name=None, q=0.99):
    """
    Return vmin, vmax.
    If sym_cbar: symmetric around zero.
    q: quantile used to clip outliers (e.g. 0.98 or 0.95)
    """
    data = field[np.isfinite(field)]

    if sym_cbar:
        abs_q = np.quantile(np.abs(data), q)

        if name in ["B_r", "B_theta", "B_phi"]:
            fc = 1.0   
            return -fc * abs_q, fc * abs_q
        else:
            return -abs_q, abs_q
    else:
        return np.quantile(data, 1 - q), np.quantile(data, q)

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


def plot_equatorial(folderFile, data, field_name, title=None, cmap="RdBu_r",
                    ax=None, savefig=None, sym_cbar=True, include_background=False, vmin=None, vmax=None):
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of "u_r", "u_theta", "u_phi", "B_r", "B_theta", "B_phi", "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure

    """
    # set colormap
    cmap = cmap_for_field(field_name)

    r, theta, phi = data["r"], data["theta"], data["phi"]
    eq_idx = np.argmin(np.abs(theta - np.pi/2))
    field = data[field_name][:, eq_idx, :]
    field = apply_temperature_background(field_name, field, r, include_background)
    
    if field_name ==  "inertia_magnitude":
        field  = 1e-9 * field

    if len(r) < 110:
        # Make phi periodic
        phi_periodic = np.hstack([phi, 2*np.pi])
        field_periodic = np.hstack([field, field[:, 0:1]])  # append first column

        # Double r and phi resolution
        r_new = np.linspace(r[0], r[-1], 2*len(r))
        phi_new = np.linspace(phi[0], 2*np.pi, 2*len(phi), endpoint=False)

        # Interpolator on periodic data
        interp_func = RegularGridInterpolator((r, phi_periodic), field_periodic)

        # Create meshgrid for new points
        R_new, Phi_new = np.meshgrid(r_new, phi_new, indexing="ij")
        points_new = np.array([R_new.ravel(), Phi_new.ravel()]).T
        # Interpolate
        field_new = interp_func(points_new).reshape(R_new.shape)
        r, phi, field = r_new, phi_new, field_new


    # if vmin is None and vmax is None: 
    #     q = 0.98
    #     if field_name == "T":
    #         cmap="gist_heat"
    #         sym_cbar = False  # temperature is positive definite
    #     elif field_name == "inertia_magnitude" or 'coriolis_magnitude' or 'viscous_magnitude',or 'lorentz_magnitude' or 'buoyancy_magnitude'
    #         q = 0.95
    #         sym_cbar = False
    #     vmin, vmax = _get_color_limits(field, sym_cbar, q)

    # Define field categories for better organization
    POSITIVE_DEFINITE_FIELDS = {"T"}
    MAGNITUDE_FIELDS = {
        "inertia_magnitude", "coriolis_magnitude", 
        "lorentz_magnitude", "buoyancy_magnitude"
    }
    BOUNDARY_FIELDS= {"viscous_magnitude"}

    if vmin is None and vmax is None:
        if field_name in BOUNDARY_FIELDS:
            max_val = np.max(field)
            vmax_10percent = 0.5 * max_val 
            print(f"\n{field_name} 统计:")
            print(f"  max = {max_val:.6e}")
            print(f"  10% of max = {vmax_10percent:.6e}")
            print(f"  99th percentile = {np.percentile(field, 99):.6e}")
            print(f"  95th percentile = {np.percentile(field, 95):.6e}")
            print(f"  90th percentile = {np.percentile(field, 90):.6e}")
            print(f"  50th percentile (median) = {np.percentile(field, 50):.6e}")
            print(f"  10th percentile = {np.percentile(field, 10):.6e}")
            print(f"  1st percentile = {np.percentile(field, 1):.6e}")
            #90th percentile for viscous magnitud
            #vmax = np.percentile(field, 95)
            
            q = 0.95
            sym_cbar = False
            vmin, vmax = _get_color_limits(field, sym_cbar, q)
        elif field_name in POSITIVE_DEFINITE_FIELDS:
            q = 0.98
            cmap = "gist_heat"
            sym_cbar = False  
            vmin, vmax = _get_color_limits(field, sym_cbar, q)
        elif field_name in MAGNITUDE_FIELDS:
            q = 0.95
            sym_cbar = False
            vmin, vmax = _get_color_limits(field, sym_cbar, q)
        else:
            q = 0.95  
            sym_cbar = True
            vmin, vmax = _get_color_limits(field, sym_cbar, q)    
    
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
    #plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Force scientific notation (x10^n)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))  # always use scientific notation
    cbar.formatter = fmt
    cbar.update_ticks()
    
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if title is not None:
        ax.set_title(rf"${title}$", pad=10, fontsize=16)

    # Save file
    if savefig is not None:
        savefig_field_snapshot(folderFile, field_name, savefig, type="equ")




def plot_meridional(folderFile, data, field_name, title="Meridional slice", cmap="RdBu_r", atphi = 0.5, ax=None, 
                    savefig=None, sym_cbar=True, include_background=False, vmin=None, vmax=None):
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of "u_r", "u_theta", "u_phi", "B_r", "B_theta", "B_phi", "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure

        atphi: float, between 0 and 2, position in phi (0 = 0 degrees, 0.5 = 90 degrees, 1 = 180 degrees)

    """
    
    # set colormap
    cmap = cmap_for_field(field_name)
    
    r, theta, phi = data["r"], data["theta"], data["phi"]
    atphi = atphi * np.pi
    mid_phi = np.argmin(np.abs(phi - atphi)) # meridional slice at phi = 90 degrees

    if field_name == "u_phi_zonal_3d":
        field = data[field_name][:, :, 0]
    else:
        field = data[field_name][:, :, mid_phi]
        field = apply_temperature_background(field_name, field, r, include_background)
    
    if field_name ==  "inertia_magnitude":
        field  = 1e-9 * field

    if len(r) < 120:
        # Make phi periodic
        #phi_periodic = np.hstack([phi, 2*np.pi])
        #field_periodic = np.hstack([field, field[:, 0:1]])  # append first column

        # Double r and phi resolution
        r_new = np.linspace(r[0], r[-1], 2*len(r))
        theta_new = np.linspace(theta[0], theta[-1], 2*len(theta), endpoint=False)

        # Interpolator on periodic data
        interp_func = RegularGridInterpolator((r, theta), field)

        # Create meshgrid for new points
        R_new, Theta_new = np.meshgrid(r_new, theta_new, indexing="ij")
        points_new = np.array([R_new.ravel(), Theta_new.ravel()]).T
        # Interpolate
        field_new = interp_func(points_new).reshape(R_new.shape)
        r, theta, field = r_new, theta_new, field_new

 
    # Define field categories for better organization
    POSITIVE_DEFINITE_FIELDS = {"T"}
    MAGNITUDE_FIELDS = {
        "inertia_magnitude", "coriolis_magnitude", 
        "lorentz_magnitude", "buoyancy_magnitude"
    }
    BOUNDARY_FIELDS= {"viscous_magnitude"}

    if vmin is None and vmax is None:
        if field_name in POSITIVE_DEFINITE_FIELDS:
            q = 0.98
            cmap = "gist_heat"
            sym_cbar = False  # temperature is positive definite
        elif field_name in MAGNITUDE_FIELDS:
            q = 0.95
            sym_cbar = False
        elif field_name in  BOUNDARY_FIELDS:
            q = 0.90
            sym_cbar = False
           # vmax = 0.01*np.max(field) 
        else:
            q = 0.95  # default quantile
            sym_cbar = True


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
    #plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Force scientific notation (x 10^n)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))  # always use scientific notation
    cbar.formatter = fmt
    cbar.update_ticks()
    
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    if savefig is not None:
        savefig_field_snapshot(folderFile, field_name, savefig, type="meridional")
        #plt.savefig(f"{savefig}/{field_name}_merid_slice.png", dpi=300, bbox_inches="tight")

    
def plot_cmb(
    folderFile,    
    data, field_name="div_uh", title="CMB", cmap="RdBu_r", ax=None, 
    savefig=None, sym_cbar=True, include_background=False,
    vmin=None, vmax=None, show_grid=False,
    discrete_cbar=False, N_levels=24
    , at_r = 1.0, abs=False
):    
    """
        data: dictionary with keys "r", "theta", "phi" and field_name
        
        field_name: string, one of 
           "u_r", "u_theta", "u_phi", 
           "B_r", "B_theta", "B_phi", 
           "temperature"
        
        include_background: bool, if True and field_name is "temperature", add background profile
        
        sym_cbar: bool, if True, colorbar is symmetric around zero
        
        show_grid: bool, if True, show latitude/longitude grid
        
        savefig: str or None, if str, path to save figure
        
        ax: matplotlib axis or None, if None, create new figure
    """
    # set colormap
    cmap = cmap_for_field(field_name)
    
    r, theta, phi = data["r"], data["theta"], data["phi"]

    if at_r >= r[0] and at_r <= r[-1]:
        r_id = np.argmin(np.abs(r - at_r))

    if field_name == "div uh":
        # compute divgence of velocity field at CMB
        u_phi = data["u_phi"][r_id, :, :]
        u_theta = data["u_theta"][r_id, :, :]
        div_uh = (1/(r[r_id]*np.sin(theta[:, None]))) * (
            np.gradient(u_phi, phi, axis=1) + 
            np.gradient(u_theta * np.sin(theta[:, None]), theta, axis=0)
        )
        field = div_uh
        # only show negative values
        #field = np.where(field < 0, field, 0)

    else:
        field = data[field_name][r_id, :, :]
        field = apply_temperature_background(field_name, field, r, include_background)    
    
    if abs:
        field = np.abs(field)

    if vmin is None and vmax is None:
        vmin, vmax = _get_color_limits(field, sym_cbar)
    
    if field_name == "B_r" or field_name == "B_theta" or field_name == "B_phi":
        print('max magnetic ' + field_name + ':', str(np.max(field)))
    
    if len(phi) < 200:
        # ------------------------------
        # Increase resolution for plotting
        # ------------------------------
        phi_periodic = np.hstack([phi, 2*np.pi])
        field_periodic = np.hstack([field, field[:, 0:1]])  # append first column for periodicity

        # Optional: double resolution in theta and phi
        theta_new = np.linspace(theta[0], theta[-1], 2*len(theta))
        phi_new = np.linspace(phi[0], 2*np.pi, 2*len(phi), endpoint=False)

        # Interpolator
        interp_func = RegularGridInterpolator((theta, phi_periodic), field_periodic)

        # Meshgrid for new points
        Theta_new, Phi_new = np.meshgrid(theta_new, phi_new, indexing="ij")
        points_new = np.array([Theta_new.ravel(), Phi_new.ravel()]).T

        # Interpolate
        field_new = interp_func(points_new).reshape(Theta_new.shape)

        # Replace old grids with high-res
        theta, phi, field = theta_new, phi_new, field_new

    # Convert spherical coords
    lon = phi - np.pi
    lat = np.pi/2 - theta
    Lon, Lat = np.meshgrid(lon, lat, indexing="ij")

    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection="mollweide")

    # Plot field
    im = ax.pcolormesh(Lon, Lat, field.T, shading="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()

    # -----------------------------
    # DISCRETE OR CONTINUOUS COLORBAR
    # -----------------------------
    if discrete_cbar:
        # Generate N+1 boundaries between vmin and vmax
        boundaries = np.linspace(vmin, vmax, N_levels + 1)
        norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=plt.get_cmap(cmap).N)

        im = ax.pcolormesh(
            Lon, Lat, field.T,
            shading="auto",
            cmap=cmap,
            norm=norm
        )

        cbar = plt.colorbar(
            im, ax=ax, orientation="horizontal",
            pad=0.05, fraction=0.05,
            boundaries=boundaries,
            #ticks=boundaries
            ticks=[vmin, vmax] 
        )

    else:
        # continuous colorbar 
        im = ax.pcolormesh(
            Lon, Lat, field.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax
        )
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                            pad=0.05, fraction=0.05)

    # label and title
    label_str = field_latex.get(field_name, field_name)
    ax.set_title(rf"${label_str}$", pad=10, fontsize=16)

    # -----------------------------
    # Add lat/lon grid 
    # -----------------------------
    if show_grid:
        ax.set_axis_on()
        ax.grid(True, alpha=0.4)

        # -----------------------------
        # Longitudes: 0° → 360° every 30°
        # -----------------------------
        lon_deg = np.arange(0, 361, 30)
        # Convert 0–360 to Mollweide-centered ticks (shift by -180°)
        lon_rad = np.radians(lon_deg - 180)
        ax.set_xticks(lon_rad)
        ax.set_xticklabels([f"{int(d)}°" for d in lon_deg])

        # -----------------------------
        # Latitudes: -90° → 90° every 30°
        # -----------------------------
        lat_deg = np.arange(-90, 91, 30)
        lat_rad = np.radians(lat_deg)
        ax.set_yticks(lat_rad)
        ax.set_yticklabels([f"{int(d)}°" for d in lat_deg])


    # Save file
    if savefig is not None:
        savefig_field_snapshot(folderFile, field_name, savefig, type="cmb")
    
    return ax











