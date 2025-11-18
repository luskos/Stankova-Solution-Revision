#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purely numerical wormhole + BD engine (4D + 10D uplift, radial scan)
with Born–Infeld-like field and shell decomposition.

No symbolic algebra, only numpy + finite differences.
"""

import numpy as np
from math import sin, cos, exp, sqrt, pi

########################################
# 1. Parameters
########################################

A       = 0.01     # depth of potential
R0      = 2.0     # throat radius (large)
w       = 0.2      # width of bump
eps     = 0.04      # regularization near r=0
omegaBD = 100.0    # Brans–Dicke parameter
L_int   = 1.0      # internal length scale for 10D uplift

b_BI    = 0.5      # Born–Infeld scale (adjust as you like)
E0      = 0.2      # amplitude of E(r) profile
sigmaE  = 5.0      # width of E(r) profile around R0

# step for finite differences
h_default = 1e-3

########################################
# 2. Phi(r) and helpers (numeric)
########################################

def rreg(r):
    return sqrt(r*r + eps*eps)

def Phi(r):
    rr = rreg(r)
    return -A * (1.0 - R0/rr) * exp(- (rr - R0)**2 / (w*w))

def dPhi_dr(r, h=h_default):
    return (Phi(r+h) - Phi(r-h)) / (2.0*h)

def d2Phi_dr2(r, h=h_default):
    return (Phi(r+h) - 2.0*Phi(r) + Phi(r-h)) / (h*h)

########################################
# 3. 4D metric g_{μν} and inverse
########################################

def metric4(x):
    """
    x = [t, r, theta, phi]
    returns 4x4 numpy array g_{μν}
    """
    _, r, th, _ = x
    Ph = Phi(r)
    e2P  = exp(2.0*Ph)
    em2P = exp(-2.0*Ph)

    g = np.zeros((4,4), dtype=float)
    g[0,0] = -e2P
    g[1,1] = em2P
    g[2,2] = r*r
    g[3,3] = r*r*(sin(th)**2)
    return g

def inv_metric4(x):
    return np.linalg.inv(metric4(x))

########################################
# 4. Finite difference helper
########################################

def partial_derivative_metric(x, alpha, mu, nu, h=h_default):
    """
    ∂ g_{μν} / ∂ x^alpha at point x, numeric
    alpha, mu, nu = 0..3
    """
    xp = np.array(x, dtype=float)
    xm = np.array(x, dtype=float)
    xp[alpha] += h
    xm[alpha] -= h

    gp = metric4(xp)[mu, nu]
    gm = metric4(xm)[mu, nu]
    return (gp - gm) / (2.0*h)

########################################
# 5. Christoffel symbols Γ^μ_{νρ} (numeric)
########################################

def christoffel4(x, h=h_default):
    """
    returns Gamma[mu,nu,ro] at point x
    """
    g = metric4(x)
    ginv = np.linalg.inv(g)
    Gamma = np.zeros((4,4,4), dtype=float)

    for mu in range(4):
        for nu in range(4):
            for ro in range(4):
                s = 0.0
                for sig in range(4):
                    d_sig_nu = partial_derivative_metric(x, ro, sig, nu, h)
                    d_sig_ro = partial_derivative_metric(x, nu, sig, ro, h)
                    d_nu_ro  = partial_derivative_metric(x, sig, nu, ro, h)
                    s += ginv[mu, sig] * (d_sig_nu + d_sig_ro - d_nu_ro)
                Gamma[mu,nu,ro] = 0.5 * s
    return Gamma

########################################
# 6. Riemann tensor R^μ_{νρσ} (numeric)
########################################

def partial_derivative_gamma(x, alpha, mu, nu, ro, h=h_default):
    """
    ∂ Γ^μ_{νro} / ∂ x^alpha at point x
    """
    xp = np.array(x, dtype=float)
    xm = np.array(x, dtype=float)
    xp[alpha] += h
    xm[alpha] -= h

    Gp = christoffel4(xp, h)[mu,nu,ro]
    Gm = christoffel4(xm, h)[mu,nu,ro]
    return (Gp - Gm) / (2.0*h)

def riemann4(x, h=h_default):
    """
    Riemann R^μ_{νρσ}(x)
    """
    Gamma = christoffel4(x, h)
    Riem = np.zeros((4,4,4,4), dtype=float)

    for mu in range(4):
        for nu in range(4):
            for ro in range(4):
                for sg in range(4):
                    d_ro_G = partial_derivative_gamma(x, ro, mu, nu, sg, h)
                    d_sg_G = partial_derivative_gamma(x, sg, mu, nu, ro, h)
                    s = d_ro_G - d_sg_G
                    for la in range(4):
                        s += Gamma[mu,ro,la]*Gamma[la,nu,sg] \
                           - Gamma[mu,sg,la]*Gamma[la,nu,ro]
                    Riem[mu,nu,ro,sg] = s
    return Riem

########################################
# 7. Ricci, scalar curvature, Einstein tensor
########################################

def ricci4(x, h=h_default):
    Riem = riemann4(x, h)
    Ric = np.zeros((4,4), dtype=float)
    for nu in range(4):
        for sg in range(4):
            s = 0.0
            for mu in range(4):
                s += Riem[mu,nu,mu,sg]
            Ric[nu,sg] = s
    return Ric

def scalar_curvature4(x, h=h_default):
    Ric = ricci4(x, h)
    ginv = inv_metric4(x)
    R = 0.0
    for nu in range(4):
        for sg in range(4):
            R += ginv[nu,sg] * Ric[nu,sg]
    return R

def einstein4(x, h=h_default):
    g = metric4(x)
    Ric = ricci4(x, h)
    R = scalar_curvature4(x, h)
    G = np.zeros((4,4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            G[mu,nu] = Ric[mu,nu] - 0.5*g[mu,nu]*R
    return G

########################################
# 8. Total T_{μν}, Brans–Dicke B(r), T_req(r)
########################################

def T_total4(x, h=h_default):
    """
    T_{μν} = (1/8π) G_{μν}
    """
    G = einstein4(x, h)
    return G / (8.0*pi)

def B_BD(r, h=h_default):
    return d2Phi_dr2(r, h) + 2.0*dPhi_dr(r, h)/r

def T_req(r, h=h_default):
    return (3.0 + 2.0*omegaBD)/(8.0*pi) * B_BD(r, h)

########################################
# 9. Null vector and NEC
########################################

def null_vector_k(x):
    """
    Radial null vector in coordinate basis:
    k^μ = (e^{-Φ}, e^{Φ}, 0, 0)
    """
    _, r, _, _ = x
    Ph = Phi(r)
    return np.array([exp(-Ph), exp(Ph), 0.0, 0.0], dtype=float)

def NEC_flux(T, x):
    """
    T_{μν} k^μ k^ν
    """
    k = null_vector_k(x)
    s = 0.0
    for mu in range(4):
        for nu in range(4):
            s += T[mu,nu] * k[mu] * k[nu]
    return s

########################################
# 10. 10D uplift metric (4D wormhole × 6D flat)
########################################

def metric10(x10):
    """
    x10 = [t,r,th,ph,y1..y6]
    10D block-diagonal metric
    """
    x4 = x10[:4]
    g4 = metric4(x4)
    g10 = np.zeros((10,10), dtype=float)

    # 4D block
    g10[:4,:4] = g4

    # 6D flat block L_int^2
    for i in range(6):
        g10[4+i, 4+i] = L_int**2

    return g10

def inv_metric10(x10):
    return np.linalg.inv(metric10(x10))

def einstein10_from_4D(x4, h=h_default):
    """
    For M_4 × T^6 with flat internal T^6:
    - R10_MN = (R4_mu_nu, 0)
    - R10 = R4
    - G10_MN = R10_MN - 1/2 g10_MN R4
    So we can build G10 from the 4D Einstein and scalar curvature.
    """
    G4 = einstein4(x4, h)
    R4 = scalar_curvature4(x4, h)
    x10 = np.concatenate([x4, np.zeros(6)])  # y_i = 0
    g10 = metric10(x10)

    G10 = np.zeros((10,10), dtype=float)
    # upper-left 4×4 block = G4
    G10[:4,:4] = G4

    # internal 6×6 block: Ricci_int = 0, but we have -1/2 g_int * R4
    for i in range(4,10):
        G10[i,i] = -0.5 * g10[i,i] * R4

    return G10

########################################
# 11. Born–Infeld field: E(r) profile and T_BI
########################################

def E_profile(r):
    """
    Simple localized electric field around the throat.
    You can change this profile freely.
    """
    return E0 * exp(- (r - R0)**2 / (sigmaE**2))

def BI_rho_pr_pt(r):
    """
    BI-like energy density and pressures for pure radial E(r).

    ρ_BI = b^2( sqrt(1 + E^2/b^2) - 1 )
    p_r_BI = -ρ_BI
    p_t_BI = ρ_BI
    """
    E = E_profile(r)
    if b_BI == 0.0:
        return 0.0, 0.0, 0.0
    s = sqrt(1.0 + (E*E)/(b_BI*b_BI))
    rho = b_BI*b_BI*(s - 1.0)
    pr  = -rho
    pt  = rho
    return rho, pr, pt

def T_BI4(x, h=h_default):
    """
    BI stress tensor in coordinate basis, approx diag(-ρ, p_r, p_t, p_t).
    """
    _, r, _, _ = x
    rho, pr, pt = BI_rho_pr_pt(r)
    T = np.zeros((4,4), dtype=float)
    T[0,0] = -rho
    T[1,1] = pr
    T[2,2] = pt
    T[3,3] = pt
    return T

########################################
# 12. Shell T_{μν} from T_req(r) (isotropic)
########################################

def shell_rho_p(r, h=h_default):
    """
    Shell chosen so that trace(shell) ≈ T_req(r) in Minkowski-like frame:
    -rho_sh + 3 p_sh = T_req(r)

    We choose:
      rho_sh = |T_req|
      p_sh   = (T_req + rho_sh)/3

    This guarantees ρ_sh >= 0 and NEC_shell >= 0 (for simple EOS).
    """
    Tt = T_req(r, h)
    rho_sh = abs(Tt)
    p_sh   = (Tt + rho_sh)/3.0
    return rho_sh, p_sh

def T_shell4(x, h=h_default):
    _, r, _, _ = x
    rho_sh, p_sh = shell_rho_p(r, h)
    T = np.zeros((4,4), dtype=float)
    T[0,0] = -rho_sh
    T[1,1] = p_sh
    T[2,2] = p_sh
    T[3,3] = p_sh
    return T

########################################
# 13. Scalar sector T_phi = T_total - T_BI - T_shell
########################################

def T_scalar4(x, h=h_default):
    Ttot = T_total4(x, h)
    Tbi  = T_BI4(x, h)
    Tsh  = T_shell4(x, h)
    return Ttot - Tbi - Tsh

########################################
# 14. Radial scan for NEC and BD trace (all sectors)
########################################

def radial_scan(num_points=11, r_factor_min=0.5, r_factor_max=2.0, h=h_default):
    """
    Scan r from r_factor_min*R0 to r_factor_max*R0

    Returns:
      dict with arrays for:
      r, Phi, B_BD, T_req, NEC_tot, NEC_BI, NEC_shell, NEC_scalar
    """
    r_values = np.linspace(r_factor_min*R0, r_factor_max*R0, num_points)
    Phi_vals    = np.zeros(num_points)
    BBD_vals    = np.zeros(num_points)
    Treq_vals   = np.zeros(num_points)
    NEC_tot     = np.zeros(num_points)
    NEC_BI_vals = np.zeros(num_points)
    NEC_sh_vals = np.zeros(num_points)
    NEC_phi_vals= np.zeros(num_points)

    for i, rv in enumerate(r_values):
        x = np.array([0.0, rv, pi/2, 0.0], dtype=float)
        Phi_vals[i]  = Phi(rv)
        BBD_vals[i]  = B_BD(rv, h)
        Treq_vals[i] = T_req(rv, h)

        Ttot = T_total4(x, h)
        Tbi  = T_BI4(x, h)
        Tsh  = T_shell4(x, h)
        Tphi = Ttot - Tbi - Tsh

        NEC_tot[i]  = NEC_flux(Ttot, x)
        NEC_BI_vals[i] = NEC_flux(Tbi, x)
        NEC_sh_vals[i] = NEC_flux(Tsh, x)
        NEC_phi_vals[i]= NEC_flux(Tphi, x)

    return dict(
        r = r_values,
        Phi = Phi_vals,
        BBD = BBD_vals,
        Treq = Treq_vals,
        NEC_tot = NEC_tot,
        NEC_BI = NEC_BI_vals,
        NEC_shell = NEC_sh_vals,
        NEC_phi = NEC_phi_vals
    )

########################################
# 15. Simple test at the throat + scan
########################################

#############################################################
# APPENDIX MODULE: Visualization, spectral analysis,
# curvature invariants, and 10D diagnostics.
#############################################################

import matplotlib.pyplot as plt

########################################
# A. Plot NEC components vs radius
########################################

def plot_NEC(data):
    r = data["r"]
    plt.figure(figsize=(10,6))
    plt.plot(r, data["NEC_tot"],   label="NEC_total")
    plt.plot(r, data["NEC_BI"],    label="NEC_BI")
    plt.plot(r, data["NEC_shell"], label="NEC_shell")
    plt.plot(r, data["NEC_phi"],   label="NEC_scalar")
    plt.axhline(0, color='black', linewidth=0.7)
    plt.xlabel("r")
    plt.ylabel("NEC")
    plt.title("NEC Components vs Radius")
    plt.legend()
    plt.grid(True)
    plt.show()


########################################
# B. Plot Phi(r), B_BD(r), T_req(r)
########################################

def plot_scalar_sector(data):
    r = data["r"]
    plt.figure(figsize=(10,6))
    plt.plot(r, data["Phi"],  label="Phi(r)")
    plt.plot(r, data["BBD"],  label="B_BD(r)")
    plt.plot(r, data["Treq"], label="T_req(r)")
    plt.xlabel("r")
    plt.ylabel("Value")
    plt.title("Scalar Sector Profiles")
    plt.legend()
    plt.grid(True)
    plt.show()


########################################
# C. Spectral analysis of tensors (eigenvalues)
########################################

def eigenvalues_tensor(T):
    return np.linalg.eigvals(T)

def analyze_tensor_at_throat(x):
    print("\n--- Eigenvalues at the throat ---")

    Ttot = T_total4(x)
    Tbi  = T_BI4(x)
    Tsh  = T_shell4(x)
    Tphi = T_scalar4(x)

    print("\nEigenvalues(T_total):", eigenvalues_tensor(Ttot))
    print("\nEigenvalues(T_BI):   ", eigenvalues_tensor(Tbi))
    print("\nEigenvalues(T_shell):", eigenvalues_tensor(Tsh))
    print("\nEigenvalues(T_scalar):", eigenvalues_tensor(Tphi))

    # also 4D Einstein tensor
    G = einstein4(x)
    print("\nEigenvalues(G_4D):    ", eigenvalues_tensor(G))


########################################
# D. Curvature invariants (4D)
########################################

def kretschmann4(x, h=h_default):
    """
    K = R_{μνρσ} R^{μνρσ}
    """
    Riem = riemann4(x, h)
    ginv = inv_metric4(x)
    K = 0.0
    for mu in range(4):
        for nu in range(4):
            for ro in range(4):
                for sg in range(4):
                    # raise first index
                    R_up = 0.0
                    for al in range(4):
                        R_up += ginv[mu,al]*Riem[al,nu,ro,sg]
                    K += R_up * Riem[mu,nu,ro,sg]
    return K

def ricci_norm4(x, h=h_default):
    Ric = ricci4(x, h)
    ginv = inv_metric4(x)
    val = 0.0
    for mu in range(4):
        for nu in range(4):
            # raise first index
            R_up = 0.0
            for al in range(4):
                R_up += ginv[mu,al]*Ric[al,nu]
            val += R_up * Ric[mu,nu]
    return val


def curvature_report(x):
    print("\n=== Curvature invariants at throat ===")
    print("Kretschmann K = ", kretschmann4(x))
    print("Ricci norm    = ", ricci_norm4(x))
    print("Scalar R      = ", scalar_curvature4(x))


########################################
# E. Heatmap of T-sector decomposition
########################################

def T_heatmap_at_throat(x):
    Ttot = T_total4(x)
    Tbi  = T_BI4(x)
    Tsh  = T_shell4(x)
    Tphi = T_scalar4(x)

    def show(T, title):
        plt.figure(figsize=(4,4))
        plt.imshow(T, cmap="seismic", interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.show()

    show(Ttot, "T_total at throat")
    show(Tbi,  "T_BI at throat")
    show(Tsh,  "T_shell at throat")
    show(Tphi, "T_scalar at throat")


########################################
# F. Embedded geometry z(r) of spatial slice
########################################

def embedding_z_r(r_range=3.0, num=200):
    """
    Embedding diagram for spatial slice t=const, θ=π/2.
    z'(r) = sqrt(g_rr - 1)
    g_rr = e^{-2Φ(r)}.
    """
    rs = np.linspace(R0 - r_range, R0 + r_range, num)
    zs = np.zeros(num)
    for i in range(1, num):
        r_mid = 0.5*(rs[i] + rs[i-1])
        g_rr = np.exp(-2*Phi(r_mid))
        dzdr = np.sqrt(max(g_rr - 1.0, 0))
        zs[i] = zs[i-1] + dzdr*(rs[i]-rs[i-1])
    return rs, zs

def plot_embedding():
    rs, zs = embedding_z_r()
    plt.figure(figsize=(8,6))
    plt.plot(rs, zs, 'b')
    plt.plot(rs, -zs, 'b')
    plt.xlabel("r")
    plt.ylabel("z(r)")
    plt.title("Embedded Wormhole Spatial Slice")
    plt.grid(True)
    plt.show()

#############################################################
# APPENDIX_FLUX: 10D RR/NS flux toy model, warp factor
# and simple "swampland-style" diagnostics
#############################################################

########################################
#  H. Warp factor A(y) and warped 10D metric
########################################

def warp_A(yvec):
    """
    Simple toy warp factor A(y) depending on internal radius ρ.
    You can tune A0, alpha to your liking.
    """
    A0    = 0.0   # base warp (set to 0 to recover unwarped case)
    alpha = 0.0   # small warping if you want (e.g. 0.01)
    rho2  = np.dot(yvec, yvec)
    return A0 + alpha * rho2


def metric10_warped(x10):
    """
    Warped metric:
      ds^2_10 = e^{2A(y)} g_4 + e^{-2A(y)} δ_6
    with A(y) from warp_A.
    """
    x4  = x10[:4]
    y6  = x10[4:]
    Awar = warp_A(y6)

    g4  = metric4(x4)
    g10 = np.zeros((10,10), dtype=float)

    # external block with e^{2A}
    g10[:4,:4] = np.exp(2.0*Awar) * g4

    # internal block with e^{-2A}
    for i in range(6):
        g10[4+i, 4+i] = np.exp(-2.0*Awar) * L_int**2

    return g10


########################################
#  I. Simple RR/NS flux toy stress tensor in 10D
########################################

# Constants controlling flux energy scales
Lambda4_flux  = 0.0     # effective 4D "cosmological" piece from flux
Lambda6_flux  = 1.0e-4  # internal flux energy scale (same order as G10_internal)

def T_flux10(x10):
    """
    Toy model for combined RR/NS flux stress tensor in 10D.

    We assume:
      - 4D part:   T_{μν}^{flux} = -Lambda4_flux * g_{μν}
      - internal:  T_{ab}^{flux} = +Lambda6_flux * g_{ab}
    (sign choice mimics some IIB setups where internal flux
     lays mostly in compact space and gives positive internal pressure.)
    """
    g10 = metric10(x10)  # use unwarped metric for simplicity here
    T10 = np.zeros((10,10), dtype=float)

    # external 4D block
    for mu in range(4):
        for nu in range(4):
            T10[mu,nu] = -Lambda4_flux * g10[mu,nu]

    # internal 6D block
    for a in range(4,10):
        for b in range(4,10):
            if a == b:
                T10[a,b] = Lambda6_flux * g10[a,b]

    return T10


########################################
#  J. 10D residual stress tensor after subtracting flux
########################################

def T10_residual_from_flux(x4, h=h_default):
    """
    We build:
      G10_{MN} from 4D geometry (einstein10_from_4D),
      then T10_total = G10/(8π),
      T10_flux  from T_flux10,
      T10_res   = T10_total - T10_flux.

    T10_res is what must be supplied by "everything else":
    - branes
    - scalar moduli
    - BI fields uplift
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    G10 = einstein10_from_4D(x4, h)
    T10_total = G10 / (8.0 * pi)
    T10_flux  = T_flux10(x10)
    return T10_total - T10_flux, T10_total, T10_flux


def print_T10_residual_report(x4):
    T10_res, T10_tot, T10_flux = T10_residual_from_flux(x4)

    print("\n=== 10D flux decomposition at this point ===")
    print("\nT10_total (from geometry) = G10/(8π):")
    print(T10_tot)

    print("\nT10_flux (toy RR/NS flux model):")
    print(T10_flux)

    print("\nT10_residual = T10_total - T10_flux:")
    print(T10_res)

    # eigenvalues to see signature
    eig_tot   = np.linalg.eigvals(T10_tot)
    eig_flux  = np.linalg.eigvals(T10_flux)
    eig_res   = np.linalg.eigvals(T10_res)

    print("\nEigenvalues(T10_total):   ", eig_tot)
    print("Eigenvalues(T10_flux):    ", eig_flux)
    print("Eigenvalues(T10_residual):", eig_res)


########################################
#  K. Simple "swampland-style" diagnostics
########################################

def swampland_diagnostics_at_throat(x4):
    """
    Crude diagnostics inspired by swampland conjectures.
    We don't have an explicit scalar potential V(φ),
    but we can use scalar-energy scale vs total scale as a proxy.

    We compute at the throat:
      - ratios of scalar T vs total T
      - fraction of NEC carried by scalar sector
    """
    print("\n=== Swampland-style diagnostics at throat ===")

    Ttot = T_total4(x4)
    Tbi  = T_BI4(x4)
    Tsh  = T_shell4(x4)
    Tphi = T_scalar4(x4)

    # rough "energy scales"
    Etot = np.max(np.abs(Ttot))
    Ephi = np.max(np.abs(Tphi))
    Ebi  = np.max(np.abs(Tbi))
    Esh  = np.max(np.abs(Tsh))

    print("Max |T_total|  =", Etot)
    print("Max |T_scalar| =", Ephi)
    print("Max |T_BI|     =", Ebi)
    print("Max |T_shell|  =", Esh)

    print("Scalar fraction (max)|T_scalar| / (max)|T_total| =", Ephi / Etot if Etot != 0 else np.nan)

    # NEC fractions
    nec_tot = NEC_flux(Ttot, x4)
    nec_bi  = NEC_flux(Tbi,  x4)
    nec_sh  = NEC_flux(Tsh,  x4)
    nec_phi = NEC_flux(Tphi, x4)

    print("\nNEC_total =", nec_tot)
    print("NEC_BI    =", nec_bi)
    print("NEC_shell =", nec_sh)
    print("NEC_scalar=", nec_phi)

    if abs(nec_tot) < 1e-8:
        print("\nNEC_total ≈ 0 → wormhole sits on NEC boundary (good for traversability).")
    if nec_phi > 0 and (nec_bi + nec_sh) < 0:
        print("Scalar sector compensates NEC violation of BI+shell → consistent with moduli-driven NEC balancing.")


########################################
#  L. Helper to run the full 10D/flux/swampland check at throat
########################################

def full_10D_flux_and_swampland_report():
    """
    Convenience wrapper:
    - builds x4 at throat
    - prints 10D flux decomposition
    - prints swampland-style diagnostics
    """
    x4 = np.array([0.0, R0, pi/2, 0.0], dtype=float)
    print_T10_residual_report(x4)
    swampland_diagnostics_at_throat(x4)

    #############################################################
# APPENDIX_D3_AXION: D3 / O3 / O7 sources, axion-like flux
# and AdS/CFT-inspired diagnostics in 10D Einstein frame
#############################################################

########################################
# M. Parameters for D3 / O3 / O7 & axion
########################################

# Dimensionless "tensions" in our units (toy values)
tau_D3 = 1.0e-3    # positive tension D3
tau_O3 = -0.5e-3   # negative tension O3 (half in magnitude)
tau_O7 = -2.0e-3   # negative tension O7, typically larger

# Axion (C0) effective energy scale in 10D
Lambda_axion = 1.0e-4  # comparable to internal flux scale


########################################
# N. D3 / O3 / O7 effective 10D stress tensors
########################################

def T10_D3(x4):
    """
    Toy D3-brane stack extended along 4D spacetime, localized in y.
    In 10D Einstein frame we model:
      T_{μν}^{D3} = -tau_D3 * g_{μν}  (μ,ν=0..3)
      T_{ab}^{D3} = 0                 (internal)
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    g10 = metric10(x10)
    T = np.zeros((10,10), dtype=float)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                T[mu,nu] = -tau_D3 * g10[mu,nu]
    return T


def T10_O3(x4):
    """
    Toy O3-plane: like D3 but with negative tension tau_O3.
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    g10 = metric10(x10)
    T = np.zeros((10,10), dtype=float)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                T[mu,nu] = -tau_O3 * g10[mu,nu]
    return T


def T10_O7(x4):
    """
    Toy O7-plane: extended in 4D + 4 internal directions.
    We approximate:
      T_{μν}^{O7} = -tau_O7 * g_{μν}  (μ,ν=0..3)
      T_{ab}^{O7} = -tau_O7 * g_{ab}  (for a,b = 4..7)
      (last 2 internal directions left free)
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    g10 = metric10(x10)
    T = np.zeros((10,10), dtype=float)

    # external 4D
    for mu in range(4):
        T[mu,mu] = -tau_O7 * g10[mu,mu]

    # wrap first 4 internal directions by O7
    for a in range(4, 8):
        T[a,a] = -tau_O7 * g10[a,a]

    return T


########################################
# O. Axion-like RR field C0 and its 10D stress
########################################

def C0_profile(r):
    """
    Simple axion C0(r) profile around the throat.
    You can tune A_ax and w_ax for different behaviours.
    """
    A_ax = 0.1    # amplitude
    w_ax = 5.0    # width
    return A_ax * np.tanh((r - R0)/w_ax)


def dC0_dr(r, h=h_default):
    return (C0_profile(r+h) - C0_profile(r-h)) / (2.0*h)


def T10_axion(x4, h=h_default):
    """
    Treat C0 as a 10D scalar with gradient only along r-direction.
    Lagrangian (Einstein frame):
      L = -1/2 (∂C0)^2
    So:
      T_MN = ∂_M C0 ∂_N C0 - 1/2 g_MN (∂C0)^2

    Here we approximate derivative only along x^1 = r.
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    g10  = metric10(x10)
    g10i = np.linalg.inv(g10)

    r = x4[1]
    dC = dC0_dr(r, h)

    # contravariant gradient (only r-component)
    grad = np.zeros(10)
    grad[1] = dC

    # (∂C)^2 = g^{MN} grad_M grad_N (using contravariant vs covariant simplification)
    dC2 = 0.0
    for M in range(10):
        for N in range(10):
            dC2 += g10i[M,N] * grad[M] * grad[N]

    T = np.zeros((10,10), dtype=float)
    for M in range(10):
        for N in range(10):
            T[M,N] = grad[M]*grad[N] - 0.5*g10[M,N]*dC2

    # rescale with Lambda_axion to tune its overall strength
    return Lambda_axion * T


########################################
# P. Combined 10D sources & residual after all sources
########################################

def T10_sources(x4, include_flux=True, include_D3=True,
                include_O3=True, include_O7=True, include_axion=True):
    """
    Sum of all 10D sources we model:
      - flux (H3/F3/F5 effective)
      - D3 stack
      - O3 plane
      - O7 plane
      - axion C0
    """
    x10 = np.concatenate([x4, np.zeros(6)])
    Tsum = np.zeros((10,10), dtype=float)

    if include_flux:
        Tsum += T_flux10(x10)
    if include_D3:
        Tsum += T10_D3(x4)
    if include_O3:
        Tsum += T10_O3(x4)
    if include_O7:
        Tsum += T10_O7(x4)
    if include_axion:
        Tsum += T10_axion(x4)

    return Tsum


def full_10D_source_decomposition_report():
    """
    Compare:
      T10_total (geometry)  = G10 / (8π)
      T10_sources           = flux + D3 + O3 + O7 + axion
      T10_leftover          = T10_total - T10_sources
    and show eigenvalues and 10D scalar curvature.
    """
    x4 = np.array([0.0, R0, pi/2, 0.0], dtype=float)
    x10 = np.concatenate([x4, np.zeros(6)])

    G10 = einstein10_from_4D(x4)
    T10_total = G10 / (8.0*pi)

    T10_src = T10_sources(x4)
    T10_left = T10_total - T10_src

    print("\n=== FULL 10D SOURCE DECOMPOSITION AT THROAT ===")
    print("\nT10_total (from geometry):")
    print(T10_total)
    print("\nT10_sources (flux + D3 + O3 + O7 + axion):")
    print(T10_src)
    print("\nT10_leftover = T10_total - T10_sources:")
    print(T10_left)

    eig_tot  = np.linalg.eigvals(T10_total)
    eig_src  = np.linalg.eigvals(T10_src)
    eig_left = np.linalg.eigvals(T10_left)

    print("\nEigenvalues(T10_total):   ", eig_tot)
    print("Eigenvalues(T10_sources): ", eig_src)
    print("Eigenvalues(T10_leftover):", eig_left)

    # 10D scalar curvature for product space: R10 = R4
    R4 = scalar_curvature4(x4)
    R10 = R4   # since internal is flat in our model
    print("\n10D scalar curvature R10 (product ansatz) =", R10)

    # Effective "D3 charge" proxy: compare 4D energy scale to tau_D3
    max_T4 = max(abs(T10_total[0,0]), abs(T10_total[1,1]),
                 abs(T10_total[2,2]), abs(T10_total[3,3]))
    Q_D3_eff = max_T4 / (abs(tau_D3) + 1e-30)
    print("Effective D3-charge proxy Q_D3_eff ~ max(T_4D)/|tau_D3| =", Q_D3_eff)

    # AdS-like effective scale from 4D energy
    if max_T4 > 0:
        L_eff = 1.0/np.sqrt(max_T4)
        print("AdS-like effective length scale L_eff ~ 1/sqrt(max_T_4D) =", L_eff)
    else:
        print("AdS-like effective length scale undefined (max_T_4D = 0)")


if __name__ == "__main__":
    # point: t=0, r=R0, th=pi/2, ph=0
    x = np.array([0.0, R0, pi/2, 0.0], dtype=float)

    print("=== Throat (4D, total) ===")
    print("Phi(R0)   =", Phi(R0))
    print("dPhi_dr   =", dPhi_dr(R0))
    print("d2Phi_dr2 =", d2Phi_dr2(R0))

    B_val = B_BD(R0)
    Treq_val = T_req(R0)
    print("B_BD(R0)  =", B_val)
    print("T_req(R0) =", Treq_val)

    g = metric4(x)
    print("\n4D metric g_{μν} at throat:\n", g)

    G = einstein4(x)
    print("\nEinstein tensor G_{μν} at throat:\n", G)

    Ttot = T_total4(x)
    print("\nT_total_{μν} at throat:\n", Ttot)

    nec_val = NEC_flux(Ttot, x)
    print("\nNEC[T_total] at throat =", nec_val)

    print("\n=== Throat: decomposition (BI + shell + scalar) ===")
    Tbi  = T_BI4(x)
    Tsh  = T_shell4(x)
    Tphi = T_scalar4(x)

    print("\nT_BI at throat:\n", Tbi)
    print("NEC[T_BI]    =", NEC_flux(Tbi, x))

    print("\nT_shell at throat:\n", Tsh)
    print("NEC[T_shell] =", NEC_flux(Tsh, x))

    print("\nT_scalar at throat:\n", Tphi)
    print("NEC[T_scalar]=", NEC_flux(Tphi, x))

    print("\n=== Radial scan around the throat ===")
    data = radial_scan(
        num_points=11,
        r_factor_min=0.5,
        r_factor_max=2.0,
        h=h_default
    )

    print("r\tPhi\t\tB_BD\t\tT_req\t\tNEC_tot\t\tNEC_BI\t\tNEC_sh\t\tNEC_phi")
    for i in range(len(data["r"])):
        rv  = data["r"][i]
        phv = data["Phi"][i]
        bbv = data["BBD"][i]
        trv = data["Treq"][i]
        nt  = data["NEC_tot"][i]
        nb  = data["NEC_BI"][i]
        ns  = data["NEC_shell"][i]
        np_ = data["NEC_phi"][i]
        print(f"{rv:.3f}\t{phv:.3e}\t{bbv:.3e}\t{trv:.3e}\t{nt:.3e}\t{nb:.3e}\t{ns:.3e}\t{np_:.3e}")

    print("\n=== 10D uplift at the throat ===")
    x10 = np.array([0.0, R0, pi/2, 0.0, 0,0,0,0,0,0], dtype=float)
    g10 = metric10(x10)
    print("\n10D metric g10 at throat x internal:\n", g10)

    G10 = einstein10_from_4D(x)
    print("\n10D Einstein tensor G10 at throat x internal:\n", G10)

    # Visualization Appendix
    plot_NEC(data)
    plot_scalar_sector(data)
    analyze_tensor_at_throat(x)
    curvature_report(x)
    T_heatmap_at_throat(x)
    plot_embedding()

    # 10D flux + swampland report
    full_10D_flux_and_swampland_report()

    # Full 10D sources (flux + branes + axion) vs geometry
    full_10D_source_decomposition_report()
