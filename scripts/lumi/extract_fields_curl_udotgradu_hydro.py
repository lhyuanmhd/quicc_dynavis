import h5py
import numpy as np
import matplotlib.pyplot as plt

def _read_scalar(ds):
    val = ds[()]
    if isinstance(val, np.ndarray) and val.shape == ():
        return val.item()
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def _read_string_dataset(ds):
    val = ds[()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.shape == ():
            v = val.item()
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode("utf-8", errors="replace")
            return str(v)
        if val.size == 1:
            v = val.ravel()[0]
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode("utf-8", errors="replace")
            return str(v)
    return str(val)

def _d_dphi_periodic(a, dphi, axis=2):
    """Second-order centered periodic derivative in phi (uniform spacing)."""
    return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * dphi)

def extract_fields_with_curl_fixed(
    hdf5_file,
    output_file="vis_fields_fixed.npz",
    include_curl=True,
    include_magnetic_curl=False,
    include_udotgradu_mag=True,
    save_udotgradu_components=False,
    verbose=True,
    r_cutoff=1e-2,
    theta_cutoff=1e-2,
):
    """完全修正版本：处理 r→0 和 θ→0,π 的奇点"""
    
    with h5py.File(hdf5_file, "r") as f:
        # ----------------------------
        # Mesh grids (1D)
        # ----------------------------
        r = f["/mesh/grid_r"][:]                 # (nr,)
        theta = f["/mesh/grid_theta"][:]         # colatitude (ntheta,)
        phi = f["/mesh/grid_phi"][:]             # (nphi,)

        latitude = np.pi / 2.0 - theta
        nr, ntheta, nphi = len(r), len(theta), len(phi)
        
        if verbose:
            print(f"网格信息:")
            print(f"  r: {nr}点, 范围 [{r.min():.3e}, {r.max():.3e}]")
            print(f"  θ: {ntheta}点, 范围 [{theta.min():.3f}, {theta.max():.3f}] rad")
            print(f"  φ: {nphi}点, 范围 [{phi.min():.3f}, {phi.max():.3f}] rad")
            print(f"  半径截止值: {r_cutoff:.3e}")
            print(f"  角度截止值: {theta_cutoff:.3e} rad")

        # ----------------------------
        # Fields
        # ----------------------------
        u_r     = f["/velocity/velocity_r"][:]
        u_theta = f["/velocity/velocity_theta"][:]
        u_phi   = f["/velocity/velocity_phi"][:]

        #B_r     = f["/magnetic/magnetic_r"][:]
        #B_theta = f["/magnetic/magnetic_theta"][:]
        #B_phi   = f["/magnetic/magnetic_phi"][:]

        T = f["/temperature/temperature"][:]

        # 形状检查
        def _check_shape(name, arr):
            if arr.ndim != 3:
                raise ValueError(f"{name} should be 3D, got shape {arr.shape}")
            if arr.shape != (nr, ntheta, nphi):
                raise ValueError(
                    f"{name} shape mismatch. Expected ({nr}, {ntheta}, {nphi}), got {arr.shape}"
                )

        for nm, a in [
            ("u_r", u_r), ("u_theta", u_theta), ("u_phi", u_phi),
            #("B_r", B_r), ("B_theta", B_theta), ("B_phi", B_phi),
            ("T", T),
        ]:
            _check_shape(nm, a)

        # 时间等信息
        time = _read_scalar(f["/run/time"])
        timestep = _read_scalar(f["/run/timestep"])

        out = dict(
            r=r, theta=theta, phi=phi, latitude=latitude,
            u_r=u_r, u_theta=u_theta, u_phi=u_phi,
            #B_r=B_r, B_theta=B_theta, B_phi=B_phi,
            T=T, time=time, timestep=timestep,
        )

        # ----------------------------
        # Curl fields
        # ----------------------------
        if include_curl:
            w_r     = f["/velocity_curl/velocity_curl_r"][:]
            w_theta = f["/velocity_curl/velocity_curl_theta"][:]
            w_phi   = f["/velocity_curl/velocity_curl_phi"][:]

            cosT = np.cos(theta)[None, :, None]
            sinT = np.sin(theta)[None, :, None]
            w_z_theta = w_r * cosT - w_theta * sinT
            out["curl_u_axial"] = w_z_theta

        # ----------------------------
        # | (u grad) u | 完全修正版本
        # ----------------------------
        if include_udotgradu_mag:
            # 创建网格
            rr = r[:, None, None]                         # (nr,1,1)
            theta_2d = theta[None, :, None]               # (1,ntheta,1)
            sinT = np.sin(theta_2d)
            cosT = np.cos(theta_2d)
            
            # 1. 处理极点问题 - 使用截止
            near_north_pole = (theta_2d < theta_cutoff)
            near_south_pole = (theta_2d > (np.pi - theta_cutoff))
            
            # 安全计算 1/sinθ
            safe_inv_sin = np.zeros_like(sinT)
            # 正常区域
            normal_region = ~(near_north_pole | near_south_pole)
            safe_inv_sin[normal_region] = 1.0 / np.abs(sinT[normal_region])
            
            # 北极附近：使用截止值
            safe_inv_sin[near_north_pole] = 1.0 / theta_cutoff
            # 南极附近
            safe_inv_sin[near_south_pole] = 1.0 / theta_cutoff
            
            # 处理符号
            safe_inv_sin = np.where(sinT >= 0, safe_inv_sin, -safe_inv_sin)
            
            # 安全计算 cotθ，在极点附近使用0
            safe_cot = cosT * safe_inv_sin
            safe_cot = np.where(near_north_pole | near_south_pole, 0.0, safe_cot)
            
            # 2. 处理球心问题
            safe_inv_r = 1.0 / np.maximum(rr, r_cutoff)
            
            if verbose:
                print(f"\n奇点处理:")
                print(f"  北极附近点数: {np.sum(near_north_pole)}")
                print(f"  南极附近点数: {np.sum(near_south_pole)}")
                print(f"  球心附近点数: {np.sum(rr < r_cutoff)}")
                print(f"  sinθ 最小值(正常区域): {np.min(np.abs(sinT[normal_region])):.6e}")
                print(f"  1/r 最大值: {np.max(safe_inv_r):.6e}")

            # 计算导数
            du_r_dr, du_r_dtheta, du_r_dphi = np.gradient(u_r, r, theta, phi, edge_order=2)
            du_theta_dr, du_theta_dtheta, du_theta_dphi = np.gradient(u_theta, r, theta, phi, edge_order=2)
            du_phi_dr, du_phi_dtheta, du_phi_dphi = np.gradient(u_phi, r, theta, phi, edge_order=2)

            # 安全版本：使用修正后的 1/r 和 1/sinθ
            inv_r_safe = safe_inv_r
            inv_r_sin_safe = safe_inv_r * safe_inv_sin

            # 计算对流加速度分量
            adv_r = (
                u_r * du_r_dr
                + (u_theta * inv_r_safe) * du_r_dtheta
                + (u_phi * inv_r_sin_safe) * du_r_dphi
                - (u_theta**2 + u_phi**2) * inv_r_safe
            )

            adv_theta = (
                u_r * du_theta_dr
                + (u_theta * inv_r_safe) * du_theta_dtheta
                + (u_phi * inv_r_sin_safe) * du_theta_dphi
                + (u_r * u_theta) * inv_r_safe
                - (u_phi**2) * safe_cot * inv_r_safe
            )

            adv_phi = (
                u_r * du_phi_dr
                + (u_theta * inv_r_safe) * du_phi_dtheta
                + (u_phi * inv_r_sin_safe) * du_phi_dphi
                + (u_r * u_phi) * inv_r_safe
                + (u_theta * u_phi) * safe_cot * inv_r_safe
            )

            adv_mag = np.sqrt(adv_r**2 + adv_theta**2 + adv_phi**2)

            if verbose:
                print("\n修正后的结果统计:")
                print(f"  adv_r:     mean={np.mean(adv_r):.6e}, median={np.median(np.abs(adv_r)):.6e}")
                print(f"  adv_theta: mean={np.mean(adv_theta):.6e}, median={np.median(np.abs(adv_theta)):.6e}")
                print(f"  adv_phi:   mean={np.mean(adv_phi):.6e}, median={np.median(np.abs(adv_phi)):.6e}")
                print(f"  |adv|:     mean={np.mean(adv_mag):.6e}")
                print(f"  |adv|:     median={np.median(adv_mag):.6e}")
                print(f"  |adv|:     max={np.max(adv_mag):.6e}")
                
                # 使用分位数统计，避免掩码问题
                percentiles = [50, 75, 90, 95, 99, 99.9]
                print(f"\n|adv| 分位数统计:")
                for p in percentiles:
                    val = np.percentile(adv_mag, p)
                    print(f"  {p}%: {val:.6e}")
                
                # 量纲分析验证
                u_mag = np.sqrt(u_r**2 + u_theta**2 + u_phi**2)
                u_scale = np.mean(u_mag)
                adv_estimate = u_scale**2
                
                print(f"\n量纲分析验证:")
                print(f"  速度尺度 U = {u_scale:.6e}")
                print(f"  估计值 U² = {adv_estimate:.6e}")
                print(f"  平均值/估计值 = {np.mean(adv_mag)/adv_estimate:.3f}")
                print(f"  中位数/估计值 = {np.median(adv_mag)/adv_estimate:.3f}")
                print(f"  75%分位数/估计值 = {np.percentile(adv_mag, 75)/adv_estimate:.3f}")
                print(f"  90%分位数/估计值 = {np.percentile(adv_mag, 90)/adv_estimate:.3f}")

            out["u_dot_grad_u_magnitude"] = adv_mag

            if save_udotgradu_components:
                out["u_dot_grad_u_r"] = adv_r
                out["u_dot_grad_u_theta"] = adv_theta
                out["u_dot_grad_u_phi"] = adv_phi

    np.savez_compressed(output_file, **out)
    print(f"\n修正后的结果保存到: {output_file}")
    
    return out


# 使用示例
if __name__ == "__main__":
    result = extract_fields_with_curl_fixed(
        "visState0000.hdf5",
        "vis_fields_0000.npz",
        include_curl=True,
        include_magnetic_curl=False,
        include_udotgradu_mag=True,
        save_udotgradu_components=True,
        verbose=True,
        r_cutoff=1e-2,
        theta_cutoff=1e-2,
    )
    
    print("\n" + "="*60)
    print("RMS值估计:")
    
    adv_mag = result["u_dot_grad_u_magnitude"]
    
    # 使用不同的统计量来估计特征尺度
    mean_adv = np.mean(adv_mag)
    median_adv = np.median(adv_mag)
    p75_adv = np.percentile(adv_mag, 75)
    p90_adv = np.percentile(adv_mag, 90)
    
    print(f"  平均值: {mean_adv:.6e}")
    print(f"  中位数: {median_adv:.6e}")
    print(f"  75%分位数: {p75_adv:.6e}")
    print(f"  90%分位数: {p90_adv:.6e}")
    
    # 量纲分析估计
    u_mag = np.sqrt(result["u_r"]**2 + result["u_theta"]**2 + result["u_phi"]**2)
    u_scale = np.mean(u_mag)
    adv_estimate = u_scale**2
    
    print(f"\n量纲分析估计值 U² = {adv_estimate:.6e}")
    print(f"\n结论: |adv| 的特征尺度在 {median_adv:.2e} 到 {p75_adv:.2e} 之间")
    print(f"      这与量纲分析估计值 {adv_estimate:.2e} 相差约 {median_adv/adv_estimate:.1f} 倍")

