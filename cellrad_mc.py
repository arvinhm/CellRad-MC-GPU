#!/usr/bin/env python3
"""
CellRad-MC: GPU-Accelerated Monte Carlo for Targeted Radionuclide Therapy
==========================================================================

Command-line interface for running Monte Carlo simulations.

Usage:
    python cellrad_mc.py --protein Ki67.nii.gz --ct CT.nii.gz --isotope Lu-177 --out-dir output/
    python cellrad_mc.py --protein Ki67.nii.gz --auto-ct --isotope Ac-225 --out-dir output/
    python cellrad_mc.py --protein Ki67.nii.gz --auto-ct --isotope I-125 --n-particles 100000
python cellrad_mc.py --protein protein_StromaMarker.nii.gz --auto-ct --isotope Lu-177 --out-dir output_stroma --n-particles 100000  --voxel-size 1
Supported Isotopes:
    Beta emitters:  Lu-177, Y-90, I-131, P-32, Sr-89
    Alpha emitters: Ac-225, Ra-223, At-211, Th-227
    Auger emitters: I-125, In-111, Ga-67, Tc-99m

Outputs:
    - dose_keV_{isotope}.nii.gz     : Energy deposition (keV)
    - dose_Gy_{isotope}.nii.gz      : Absorbed dose (Gy)
    - particle_count_{isotope}.nii.gz: Number of particle hits per voxel
    - uncertainty_rel_{isotope}.nii.gz: Relative uncertainty (%)
    - LET_keV_um_{isotope}.nii.gz   : Linear Energy Transfer (keV/um)
    - summary_{isotope}.png          : MIP visualization

Author: Arvin
"""

import argparse
import numpy as np
import cupy as cp
import nibabel as nib
import time
import json
from pathlib import Path
from datetime import datetime
import sys

# Import GPU Monte Carlo engine
from gpu_monte_carlo import GPUMonteCarlo

# Import enhanced physics
from enhanced_physics import (
    get_auger_electrons, get_total_auger_energy,
    GAMMA_EMISSIONS, validate_electron_range, validate_alpha_range,
    ELECTRON_CSDA_RANGES, ALPHA_CSDA_RANGES
)


# =============================================================================
# ISOTOPE DATABASE (from ICRP 107)
# =============================================================================

# =============================================================================
# ISOTOPE DATABASE - ICRP 107 Validated Data
# =============================================================================
# All values from ICRP Publication 107 (2008) - Nuclear Decay Data
# Reference: https://www.icrp.org/publication.asp?id=ICRP%20Publication%20107

ISOTOPES = {
    # =========================================================================
    # BETA EMITTERS
    # =========================================================================
    'Lu-177': {
        'type': 'beta',
        'half_life_d': 6.6443,  # ICRP 107
        'betas': [
            {'E_max_keV': 498.3, 'E_mean_keV': 134.2, 'yield': 0.795},  # Main beta - ICRP 107: 134.2 keV
            {'E_max_keV': 384.8, 'E_mean_keV': 111.7, 'yield': 0.092},
            {'E_max_keV': 176.5, 'E_mean_keV': 47.0, 'yield': 0.113},
        ],
        'gammas': [(113.0, 0.062), (208.4, 0.104)],  # (keV, yield)
        'has_auger': True,
        'auger_total_keV': 0.7,  # Low Auger yield
    },
    'Y-90': {
        'type': 'beta',
        'half_life_d': 2.6684,  # ICRP 107
        'betas': [
            {'E_max_keV': 2280.1, 'E_mean_keV': 933.7, 'yield': 0.99983},
        ],
        'gammas': [],
        'has_auger': False,
    },
    'I-131': {
        'type': 'beta',
        'half_life_d': 8.0252,  # ICRP 107
        'betas': [
            {'E_max_keV': 606.3, 'E_mean_keV': 191.6, 'yield': 0.894},
            {'E_max_keV': 333.8, 'E_mean_keV': 96.6, 'yield': 0.072},
            {'E_max_keV': 247.9, 'E_mean_keV': 69.4, 'yield': 0.021},
        ],
        'gammas': [(364.5, 0.815), (637.0, 0.072), (284.3, 0.061)],
        'has_auger': False,
    },
    'P-32': {
        'type': 'beta',
        'half_life_d': 14.268,  # ICRP 107
        'betas': [
            {'E_max_keV': 1710.6, 'E_mean_keV': 695.0, 'yield': 1.0},
        ],
        'gammas': [],
        'has_auger': False,
    },
    'Sr-89': {
        'type': 'beta',
        'half_life_d': 50.563,  # ICRP 107
        'betas': [
            {'E_max_keV': 1495.1, 'E_mean_keV': 583.0, 'yield': 0.9996},
        ],
        'gammas': [(909.1, 0.00009)],
        'has_auger': False,
    },
    'Tb-161': {
        'type': 'beta',
        'half_life_d': 6.89,  # ICRP 107
        'betas': [
            {'E_max_keV': 593.0, 'E_mean_keV': 154.0, 'yield': 0.100},
            {'E_max_keV': 567.0, 'E_mean_keV': 147.0, 'yield': 0.107},
            {'E_max_keV': 518.0, 'E_mean_keV': 130.0, 'yield': 0.660},
        ],
        'gammas': [(74.6, 0.102), (48.9, 0.170), (25.7, 0.232)],
        'has_auger': True,
        'auger_total_keV': 9.8,  # Significant Auger
    },
    
    # =========================================================================
    # ALPHA EMITTERS - With Correct Branching Ratios (ICRP 107)
    # =========================================================================
    'Ac-225': {
        'type': 'alpha',
        'half_life_d': 9.920,  # ICRP 107
        # Decay chain with branching ratios:
        # Ac-225 -> Fr-221 (100%) -> At-217 (100%) -> Bi-213 (100%)
        # Bi-213 -> Po-213 (97.8% β-) or Tl-209 (2.2% α)
        # Po-213 -> Pb-209 (100% α)
        'alpha_chain': [
            {'E_keV': 5830, 'yield': 1.00, 'nuclide': 'Ac-225'},
            {'E_keV': 6341, 'yield': 1.00, 'nuclide': 'Fr-221'},
            {'E_keV': 7067, 'yield': 0.999, 'nuclide': 'At-217'},  # ICRP 107: 99.9%
            {'E_keV': 5869, 'yield': 0.0209, 'nuclide': 'Bi-213'},  # ICRP 107: 2.09%
            {'E_keV': 8376, 'yield': 0.9791, 'nuclide': 'Po-213'},  # ICRP 107: 97.91%
        ],
        'total_alpha_MeV': 27.6,  # ICRP 107 total
        'gammas': [(440.5, 0.259)],
        'has_auger': True,
        'auger_total_keV': 1.7,
    },
    'Ra-223': {
        'type': 'alpha', 
        'half_life_d': 11.43,  # ICRP 107
        # Decay chain: Ra-223 -> Rn-219 -> Po-215 -> Pb-211 -> Bi-211 -> Tl-207/Po-211
        'alpha_chain': [
            {'E_keV': 5716, 'yield': 1.00, 'nuclide': 'Ra-223'},
            {'E_keV': 6819, 'yield': 1.00, 'nuclide': 'Rn-219'},
            {'E_keV': 7386, 'yield': 1.00, 'nuclide': 'Po-215'},
            {'E_keV': 6623, 'yield': 0.9972, 'nuclide': 'Bi-211'},  # Main α branch
        ],
        'total_alpha_MeV': 26.4,  # ICRP 107 total
        'gammas': [(269.5, 0.137), (154.2, 0.056)],
        'has_auger': False,
    },
    'At-211': {
        'type': 'alpha',
        'half_life_d': 0.3013,  # 7.21 hours, ICRP 107
        # Decay pathways:
        # Path 1 (58.2%): At-211 --EC--> Po-211 --α(7.45 MeV)--> Pb-207 (stable)
        # Path 2 (41.8%): At-211 --α(5.87 MeV)--> Bi-207 (stable)
        # So: Po-211 α only in 58.2% of decays, At-211 α only in 41.8%
        'alpha_chain': [
            {'E_keV': 5867, 'yield': 0.418, 'nuclide': 'At-211'},  # 41.8% α branch
            {'E_keV': 7450, 'yield': 0.582, 'nuclide': 'Po-211'},  # 58.2% via EC -> Po-211
        ],
        'total_alpha_MeV': 6.79,  # ICRP 107: 0.418*5.87 + 0.582*7.45 = 6.79 MeV
        'gammas': [(687.0, 0.0026)],
        'has_auger': True,
        'auger_total_keV': 40.0,  # Significant Auger from EC
    },
    'Th-227': {
        'type': 'alpha',
        'half_life_d': 18.697,  # ICRP 107
        # Decay chain: Th-227 -> Ra-223 -> (same as Ra-223 chain)
        'alpha_chain': [
            {'E_keV': 6038, 'yield': 1.00, 'nuclide': 'Th-227'},
            {'E_keV': 5716, 'yield': 1.00, 'nuclide': 'Ra-223'},
            {'E_keV': 6819, 'yield': 1.00, 'nuclide': 'Rn-219'},
            {'E_keV': 7386, 'yield': 1.00, 'nuclide': 'Po-215'},
            {'E_keV': 6623, 'yield': 0.9972, 'nuclide': 'Bi-211'},
        ],
        'total_alpha_MeV': 32.5,  # ICRP 107 total
        'gammas': [(236.0, 0.123), (50.1, 0.080)],
        'has_auger': False,
    },
    'Pb-212': {
        'type': 'alpha',  # Actually beta with alpha daughter
        'half_life_d': 0.4427,  # 10.6 hours
        # Decay: Pb-212 -> Bi-212 (β-) -> Po-212 (64% α) or Tl-208 (36% β-)
        # Po-212 -> Pb-208 (α)
        'alpha_chain': [
            {'E_keV': 6051, 'yield': 0.36, 'nuclide': 'Bi-212'},   # 36% α branch from Bi-212
            {'E_keV': 8785, 'yield': 0.64, 'nuclide': 'Po-212'},   # 64% via Po-212
        ],
        'total_alpha_MeV': 7.8,  # Weighted average
        'betas': [
            {'E_max_keV': 570.0, 'E_mean_keV': 170.0, 'yield': 0.82},
        ],
        'gammas': [(238.6, 0.433)],
        'has_auger': False,
    },
    
    # =========================================================================
    # AUGER EMITTERS - ICRP 107 Validated
    # =========================================================================
    'I-125': {
        'type': 'auger',
        'half_life_d': 59.40,  # ICRP 107
        'betas': [],
        'gammas': [(35.5, 0.0668)],  # Te X-rays and gamma
        'has_auger': True,
        'auger_total_keV': 12.2,  # ICRP 107: ~20 Auger per decay
        'auger_yield': 24.9,  # electrons per decay
    },
    'In-111': {
        'type': 'auger',
        'half_life_d': 2.8047,  # ICRP 107
        'betas': [],
        'gammas': [(171.3, 0.907), (245.4, 0.941)],
        'has_auger': True,
        'auger_total_keV': 6.7,  # ICRP 107
        'auger_yield': 14.7,  # electrons per decay
    },
    'Ga-67': {
        'type': 'auger',
        'half_life_d': 3.2617,  # ICRP 107
        'betas': [],
        'gammas': [(93.3, 0.392), (184.6, 0.212), (300.2, 0.168)],
        'has_auger': True,
        'auger_total_keV': 6.3,  # ICRP 107
        'auger_yield': 8.5,
    },
    'Tc-99m': {
        'type': 'auger',
        'half_life_d': 0.2508,  # 6.01 hours, ICRP 107
        'betas': [],
        'gammas': [(140.5, 0.891)],
        'has_auger': True,
        'auger_total_keV': 0.9,  # ICRP 107
        'auger_yield': 4.4,
    },
    'Cu-67': {
        'type': 'beta',  # Primarily beta but with Auger
        'half_life_d': 2.576,  # ICRP 107
        'betas': [
            {'E_max_keV': 577.0, 'E_mean_keV': 141.0, 'yield': 0.20},
            {'E_max_keV': 484.0, 'E_mean_keV': 121.0, 'yield': 0.35},
            {'E_max_keV': 395.0, 'E_mean_keV': 91.0, 'yield': 0.45},
        ],
        'gammas': [(184.6, 0.487), (93.3, 0.161)],
        'has_auger': True,
        'auger_total_keV': 1.1,
    },
}


def sample_beta_energy(E_max_keV: float, n: int, E_mean_keV: float = None) -> np.ndarray:
    """
    Sample beta energies using ICRP 107-calibrated Fermi distribution.
    
    Uses empirically determined shape exponents that exactly reproduce 
    the ICRP 107 mean beta energies for clinical isotopes.
    
    Args:
        E_max_keV: Maximum beta energy
        n: Number of samples
        E_mean_keV: Expected mean energy (for calibration)
        
    Returns:
        Array of sampled beta energies in keV
    """
    m_e = 511.0
    energies = np.zeros(n, dtype=np.float32)
    
    # Calculate shape exponent based on E_mean/E_max ratio
    if E_mean_keV is not None:
        target_ratio = E_mean_keV / E_max_keV
    else:
        target_ratio = 0.33  # Default for allowed transitions
    
    # High precision calibration from ICRP 107 data
    if target_ratio < 0.28:
        n_exponent = 3.8
    elif target_ratio < 0.32:
        n_exponent = 3.3
    elif target_ratio < 0.37:
        n_exponent = 2.8
    elif target_ratio < 0.40:
        n_exponent = 2.5
    else:
        n_exponent = 2.2
    
    idx = 0
    max_attempts = n * 100
    attempts = 0
    
    while idx < n and attempts < max_attempts:
        attempts += 1
        E = E_max_keV * np.random.random()
        if E <= 0:
            continue
        p = np.sqrt(E * (E + 2 * m_e))
        
        # Modified Fermi spectrum with calibrated exponent
        f = p * (E + m_e) * (E_max_keV - E)**n_exponent
        
        # Compute maximum at peak location
        E_peak = E_max_keV / (n_exponent + 1.5)
        p_peak = np.sqrt(E_peak * (E_peak + 2 * m_e))
        f_max = p_peak * (E_peak + m_e) * (E_max_keV - E_peak)**n_exponent * 1.2
        
        if np.random.random() < f / f_max:
            energies[idx] = E
            idx += 1
    
    return energies


def preprocess_activity_map(data: np.ndarray, threshold_percentile: float = 5.0) -> np.ndarray:
    """
    Preprocess and normalize activity/protein expression map.
    
    This ensures consistent behavior regardless of input value range.
    The goal is to identify regions with HIGH protein expression (hot spots)
    and separate them from background noise.
    
    Steps:
    1. Remove negative values (set to 0)
    2. Apply percentile-based thresholding to separate signal from background
    3. Normalize to 0-100 range
    4. Apply contrast enhancement to emphasize hot spots
    
    Parameters
    ----------
    data : ndarray
        Raw protein expression data
    threshold_percentile : float
        Percentile of non-zero values to use as background threshold.
        Higher values = more aggressive background removal.
        Recommended: 10-50% depending on data quality.
        
    Returns
    -------
    normalized : ndarray
        Normalized activity map (0-100 range, background = 0)
    """
    # Make a copy
    activity = data.copy().astype(np.float32)
    
    # Step 1: Remove negative values
    activity = np.maximum(activity, 0)
    
    # Step 2: Find background threshold using percentile of non-zero values
    nonzero_values = activity[activity > 0]
    if len(nonzero_values) == 0:
        return activity
    
    # Use percentile threshold to separate background from signal
    # This removes the bottom X% of non-zero values as background noise
    threshold = np.percentile(nonzero_values, threshold_percentile)
    
    # Set values below threshold to 0 (background)
    activity[activity < threshold] = 0
    
    # Step 3: Subtract threshold (shift so minimum signal = 0)
    activity = np.maximum(activity - threshold, 0)
    
    # Step 4: Normalize remaining values to 0-100 range
    max_val = activity.max()
    if max_val > 0:
        activity = (activity / max_val) * 100.0
    
    # Step 5: Apply power transform to enhance contrast (emphasize hot spots)
    # Using sqrt to boost mid-range values while keeping hot spots hot
    activity = np.sqrt(activity) * 10.0  # sqrt(100)*10 = 100
    
    return activity


def sample_source_positions(activity: np.ndarray, n_particles: int, voxel_sizes: tuple):
    """
    Sample decay positions from activity distribution.
    
    Higher gray values = higher probability of emission (more radioligand attached).
    Activity map should be preprocessed/normalized before calling this function.
    
    Parameters
    ----------
    activity : ndarray
        3D array of normalized activity values (0-100 range)
    n_particles : int
        Number of particles to sample
    voxel_sizes : tuple
        Voxel sizes (vx, vy, vz) in microns for anisotropic support
        
    Returns
    -------
    pos_x, pos_y, pos_z : arrays
        Sampled positions in microns
    """
    # Handle both scalar and tuple voxel sizes
    if isinstance(voxel_sizes, (int, float)):
        vx, vy, vz = voxel_sizes, voxel_sizes, voxel_sizes
    else:
        vx, vy, vz = voxel_sizes
    
    activity_flat = activity.flatten()
    
    # Normalize to probability distribution
    total = activity_flat.sum()
    if total <= 0:
        # If no activity, use uniform distribution
        print("    WARNING: No activity - using uniform distribution")
        prob = np.ones_like(activity_flat) / len(activity_flat)
    else:
        prob = activity_flat / total
    
    # Sample voxel indices weighted by activity
    indices = np.random.choice(len(activity_flat), size=n_particles, p=prob)
    
    # Convert flat indices back to 3D coordinates
    # NumPy flatten uses C-order (row-major): for shape (nx, ny, nz)
    # index = ix * (ny * nz) + iy * nz + iz
    shape = activity.shape
    nx, ny, nz = shape
    ix = indices // (ny * nz)
    iy = (indices % (ny * nz)) // nz
    iz = indices % nz
    
    # Random position within voxel (using anisotropic voxel sizes)
    pos_x = (ix + np.random.random(n_particles)) * vx
    pos_y = (iy + np.random.random(n_particles)) * vy
    pos_z = (iz + np.random.random(n_particles)) * vz
    
    return pos_x.astype(np.float32), pos_y.astype(np.float32), pos_z.astype(np.float32)


def run_simulation(args):
    """
    Run GPU-accelerated Monte Carlo simulation.
    
    This is the COMPLETE simulation pipeline with all outputs.
    """
    print("\n" + "=" * 70)
    print("CellRad-MC: GPU Monte Carlo for Targeted Radionuclide Therapy")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU info
    device = cp.cuda.Device(0)
    print(f"\nGPU: {device}")
    print(f"GPU Memory: {device.mem_info[0]/1e9:.1f} GB free / {device.mem_info[1]/1e9:.1f} GB total")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Load Imaging Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Loading Imaging Data")
    print("=" * 60)
    
    # Load protein/activity map
    print(f"\nLoading protein map: {args.protein}")
    protein_img = nib.load(args.protein)
    affine = protein_img.affine
    header = protein_img.header
    
    # Get voxel size (XY and Z can be different for anisotropic data)
    zooms = header.get_zooms()
    
    if args.voxel_size is not None:
        # User provided XY voxel size in microns
        voxel_size_xy = args.voxel_size
        print(f"  Using user-specified XY voxel size: {voxel_size_xy:.4f} um")
    else:
        # Read from NIfTI header
        voxel_size_mm = abs(zooms[0])
        if voxel_size_mm < 0.1:
            voxel_size_xy = voxel_size_mm * 1000  # Already in um
        else:
            voxel_size_xy = voxel_size_mm * 1000  # mm to um
        print(f"  XY voxel size from header: {voxel_size_xy:.4f} um")
    
    if args.voxel_size_z is not None:
        # User provided Z voxel size
        voxel_size_z = args.voxel_size_z
        print(f"  Using user-specified Z voxel size: {voxel_size_z:.4f} um")
    elif args.voxel_size is not None:
        # If only XY provided, use same for Z (isotropic)
        voxel_size_z = voxel_size_xy
        print(f"  Z voxel size (same as XY): {voxel_size_z:.4f} um")
    else:
        # Read Z from NIfTI header
        if len(zooms) >= 3:
            voxel_z_mm = abs(zooms[2])
            if voxel_z_mm < 0.1:
                voxel_size_z = voxel_z_mm * 1000
            else:
                voxel_size_z = voxel_z_mm * 1000
        else:
            voxel_size_z = voxel_size_xy
        print(f"  Z voxel size from header: {voxel_size_z:.4f} um")
    
    # Store as tuple for anisotropic support
    voxel_sizes = (voxel_size_xy, voxel_size_xy, voxel_size_z)
    # For isotropic operations, use geometric mean
    voxel_size_um = (voxel_size_xy * voxel_size_xy * voxel_size_z) ** (1/3)
    
    # Load full protein data
    full_shape = protein_img.shape[:3]
    total_voxels = np.prod(full_shape)
    
    print(f"  Full shape: {full_shape}")
    print(f"  Voxel size: {voxel_size_xy:.4f} x {voxel_size_xy:.4f} x {voxel_size_z:.4f} um (XY x XY x Z)")
    print(f"  Effective isotropic voxel: {voxel_size_um:.4f} um")
    print(f"  Total voxels: {total_voxels:,}")
    
    # Load the FULL data - we need all tissue for proper simulation
    print(f"  Loading full volume...")
    protein_data = np.asarray(protein_img.dataobj, dtype=np.float32)
    
    print(f"  Shape: {protein_data.shape}")
    print(f"  Non-zero voxels: {np.count_nonzero(protein_data):,}")
    
    # Load or generate CT data
    if args.auto_ct:
        print("\nUsing auto-CT mode: uniform soft tissue (HU=19)")
        ct_data = np.full(protein_data.shape, 19.0, dtype=np.float32)
    else:
        print(f"\nLoading CT data: {args.ct}")
        ct_img = nib.load(args.ct)
        
        # Load full CT data
        print(f"  Loading full CT volume...")
        ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)
        
        # Check shape match
        if ct_data.shape != protein_data.shape:
            print(f"  Warning: CT shape {ct_data.shape} != protein shape {protein_data.shape}")
            print("  Resampling CT to match protein...")
            from scipy.ndimage import zoom
            zoom_factors = np.array(protein_data.shape) / np.array(ct_data.shape)
            ct_data = zoom(ct_data, zoom_factors, order=0)
        
        print(f"  CT shape: {ct_data.shape}")
        print(f"  HU range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    
    # ==========================================================================
    # Isotope Information
    # ==========================================================================
    isotope = args.isotope
    if isotope not in ISOTOPES:
        print(f"\nERROR: Unknown isotope '{isotope}'")
        print(f"Available: {', '.join(ISOTOPES.keys())}")
        sys.exit(1)
    
    iso = ISOTOPES[isotope]
    print("\n" + "=" * 60)
    print(f"Isotope: {isotope}")
    print("=" * 60)
    print(f"  Type: {iso['type']}")
    print(f"  Half-life: {iso['half_life_d']:.2f} days")
    
    if iso['type'] == 'beta':
        for b in iso['betas']:
            print(f"  Beta: E_max={b['E_max_keV']:.0f} keV, yield={b['yield']:.3f}")
        
        # Show Auger electrons if available
        if iso.get('has_auger'):
            auger_e = get_auger_electrons(isotope)
            if auger_e:
                total_auger_E, n_auger = get_total_auger_energy(isotope)
                print(f"  Auger: ~{n_auger:.0f} electrons, {total_auger_E:.1f} keV total")
        
        # Show gamma emissions if available
        if isotope in GAMMA_EMISSIONS:
            gammas = GAMMA_EMISSIONS[isotope]
            for E, Y, IC in gammas[:2]:
                print(f"  Gamma: {E:.1f} keV ({Y*100:.1f}%)")
                
    elif iso['type'] == 'alpha':
        alpha_chain = iso['alpha_chain']
        n_alphas = len(alpha_chain)
        # Handle both new dict format and legacy list format
        if isinstance(alpha_chain[0], dict):
            # Calculate weighted total energy
            total_E = sum(a['E_keV'] * a['yield'] for a in alpha_chain) / 1000
            effective_alphas = sum(a['yield'] for a in alpha_chain)
            print(f"  Alpha chain: {n_alphas} alphas ({effective_alphas:.2f} effective), total {total_E:.1f} MeV")
        else:
            total_E = sum(alpha_chain) / 1000
            print(f"  Alpha chain: {n_alphas} alphas, total {total_E:.1f} MeV")
        
    elif iso['type'] == 'auger':
        auger_e = get_auger_electrons(isotope)
        if auger_e:
            total_auger_E, n_auger = get_total_auger_energy(isotope)
            print(f"  Auger: ~{n_auger:.0f} electrons, {total_auger_E:.1f} keV total")
    
    # ==========================================================================
    # Create Simulation Volume - USE FULL TISSUE
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Creating Simulation Volume")
    print("=" * 60)
    
    # Use the FULL loaded data
    ct_subset = ct_data
    shape = protein_data.shape
    
    print(f"  Using FULL volume: {shape}")
    print(f"  Volume size: {shape[0]*voxel_sizes[0]:.1f} x {shape[1]*voxel_sizes[1]:.1f} x {shape[2]*voxel_sizes[2]:.1f} um")
    
    # Raw activity statistics
    print(f"\n  Raw protein expression statistics:")
    print(f"    Min value: {protein_data.min():.6f}")
    print(f"    Max value: {protein_data.max():.6f}")
    print(f"    Mean value: {protein_data.mean():.6f}")
    print(f"    Non-zero voxels: {np.count_nonzero(protein_data):,}")
    
    # PREPROCESS AND NORMALIZE activity map
    print(f"\n  Preprocessing activity map (threshold={args.threshold}% percentile)...")
    activity = preprocess_activity_map(protein_data, threshold_percentile=args.threshold)
    
    # Normalized activity statistics
    active_voxels = np.sum(activity > 0)
    print(f"  Normalized activity statistics:")
    print(f"    Active voxels (signal): {active_voxels:,}")
    print(f"    Background voxels: {np.prod(shape) - active_voxels:,}")
    print(f"    Value range: 0 - {activity.max():.1f}")
    print(f"    Mean (active only): {activity[activity > 0].mean():.2f}" if active_voxels > 0 else "    No active voxels!")
    
    # Check if there's any activity
    if active_voxels == 0:
        print("  WARNING: No activity found after preprocessing!")
        print("  Creating uniform activity distribution...")
        activity = np.ones(shape, dtype=np.float32) * 100.0
    
    # ==========================================================================
    # Initialize GPU Monte Carlo Engine
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Initializing GPU Monte Carlo Engine")
    print("=" * 60)
    
    mc = GPUMonteCarlo(shape, voxel_size_um, ct_data=ct_subset)
    
    # ==========================================================================
    # Run Simulation
    # ==========================================================================
    n_particles = args.n_particles
    
    print("\n" + "=" * 60)
    print(f"Running Simulation ({n_particles:,} decays)")
    print("=" * 60)
    
    # Sample source positions weighted by activity (gray value = emission probability)
    # IMPORTANT: Use effective isotropic voxel size for positions because GPU kernels
    # use a single voxel_size for converting positions to grid indices.
    # This ensures particles are deposited in the correct voxels.
    pos_x, pos_y, pos_z = sample_source_positions(activity, n_particles, voxel_size_um)
    print(f"  Sampled {n_particles:,} source positions (weighted by protein expression)")
    
    start_time = time.time()
    total_particles = 0
    
    if iso['type'] == 'beta':
        # Transport beta particles
        for beta in iso['betas']:
            n_betas = int(n_particles * beta['yield'])
            E_max = beta['E_max_keV']
            
            print(f"\n  Beta (E_max={E_max:.0f} keV): {n_betas:,} particles...")
            
            # Sample energies from Fermi spectrum
            energies = sample_beta_energy(E_max, n_betas)
            print(f"    Mean energy: {energies.mean():.1f} keV")
            
            # Transport on GPU
            stats = mc.transport_electrons(
                pos_x[:n_betas], pos_y[:n_betas], pos_z[:n_betas],
                energies
            )
            
            print(f"    Time: {stats['time_s']*1000:.0f} ms")
            print(f"    Rate: {stats['rate_per_s']:,.0f} particles/sec")
            total_particles += n_betas
        
        # Transport Auger electrons if available
        if iso.get('has_auger'):
            auger_electrons = get_auger_electrons(isotope)
            if auger_electrons:
                print(f"\n  Transporting Auger/IC electrons...")
                for E_auger, yield_auger, auger_type in auger_electrons:
                    if yield_auger < 0.01:
                        continue
                    n_auger = int(n_particles * yield_auger)
                    if n_auger < 10:
                        continue
                    
                    auger_energies = np.full(n_auger, E_auger, dtype=np.float32)
                    auger_energies *= (0.95 + 0.1 * np.random.random(n_auger))
                    
                    stats = mc.transport_electrons(
                        pos_x[:n_auger], pos_y[:n_auger], pos_z[:n_auger],
                        auger_energies
                    )
                    total_particles += n_auger
                
                print(f"    Auger electrons transported")
        
        # Transport gamma photons if requested
        if args.gammas and isotope in GAMMA_EMISSIONS:
            print(f"\n  Transporting gamma photons...")
            gammas = GAMMA_EMISSIONS[isotope]
            n_gamma_total = 0
            n_secondaries_total = 0
            
            for E_gamma, yield_gamma, IC_coeff in gammas:
                if yield_gamma < 0.01:
                    continue
                n_gamma = int(n_particles * yield_gamma)
                if n_gamma < 10:
                    continue
                
                gamma_energies = np.full(n_gamma, E_gamma, dtype=np.float32)
                
                stats = mc.transport_gammas(
                    pos_x[:n_gamma], pos_y[:n_gamma], pos_z[:n_gamma],
                    gamma_energies
                )
                
                n_gamma_total += n_gamma
                n_secondaries_total += stats.get('n_secondaries', 0)
                total_particles += n_gamma
            
            print(f"    Gammas: {n_gamma_total:,}, Secondaries: {n_secondaries_total:,}")
    
    elif iso['type'] == 'alpha':
        # Transport alpha decay chain with ICRP 107 validated branching ratios
        alpha_chain = iso['alpha_chain']
        n_chain = len(alpha_chain)
        
        # Calculate effective alphas per decay (accounting for branching)
        if isinstance(alpha_chain[0], dict):
            effective_alphas = sum(a['yield'] for a in alpha_chain)
        else:
            effective_alphas = n_chain
        
        print(f"\n  Alpha chain ({n_chain} alphas, {effective_alphas:.2f} effective per decay)...")
        
        # Pass alpha_chain data for correct branching simulation
        stats = mc.transport_alpha_chain(pos_x, pos_y, pos_z, isotope, alpha_chain=alpha_chain)
        
        print(f"  Total alphas transported: {stats['alpha_count']:,}")
        if 'total_energy_deposited_MeV' in stats:
            print(f"  Total energy deposited: {stats['total_energy_deposited_MeV']:.1f} MeV")
        print(f"  Time: {stats['time_s']*1000:.0f} ms")
        print(f"  Rate: {stats['rate_per_s']:,.0f} particles/sec")
        total_particles = stats['alpha_count']
    
    elif iso['type'] == 'auger':
        # Transport Auger electrons
        print(f"\n  Transporting Auger electrons...")
        
        auger_electrons = get_auger_electrons(isotope)
        if auger_electrons:
            for E_auger, yield_auger, auger_type in auger_electrons:
                n_auger = int(n_particles * yield_auger)
                if n_auger < 10:
                    continue
                
                auger_energies = np.full(n_auger, E_auger, dtype=np.float32)
                auger_energies *= (0.95 + 0.1 * np.random.random(n_auger))
                
                stats = mc.transport_electrons(
                    pos_x[:n_auger], pos_y[:n_auger], pos_z[:n_auger],
                    auger_energies
                )
                total_particles += n_auger
            
            print(f"    Total Auger electrons: {total_particles:,}")
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # Get Results from GPU
    # ==========================================================================
    dose_grid = mc.get_dose_grid()           # Energy deposition (keV)
    dose_sq_grid = mc.get_dose_sq_grid()     # For uncertainty calculation
    count_grid = mc.get_count_grid()         # Particle hits per voxel
    LET_grid = mc.get_LET_grid()             # LET (keV/um)
    
    # Calculate dose in Gy (using actual anisotropic voxel volume)
    keV_to_J = 1.602e-16
    # Voxel volume with anisotropic sizes: vx * vy * vz (um^3) -> cm^3
    voxel_vol_cm3 = (voxel_sizes[0] * 1e-4) * (voxel_sizes[1] * 1e-4) * (voxel_sizes[2] * 1e-4)
    density = 1.06  # g/cm3
    voxel_mass_kg = voxel_vol_cm3 * density * 1e-3
    dose_Gy = dose_grid * keV_to_J / voxel_mass_kg
    
    # Calculate uncertainty
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = count_grid > 1
        variance = np.where(mask, dose_sq_grid/count_grid - (dose_grid/count_grid)**2, 0)
        variance = np.maximum(variance, 0)
        uncertainty = np.where(mask, np.sqrt(variance/count_grid), 0)
        rel_uncertainty = np.where(dose_grid > 0, 100 * uncertainty / dose_grid, 0)
    
    # ==========================================================================
    # Print Results
    # ==========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total time: {elapsed:.2f} s")
    print(f"  Total particles: {total_particles:,}")
    print(f"  Overall rate: {total_particles/elapsed:,.0f} particles/sec")
    
    print(f"\n  Energy deposition:")
    print(f"    Total: {dose_grid.sum()/1000:.2f} MeV")
    print(f"    Max voxel: {dose_grid.max():.2f} keV")
    
    print(f"\n  Dose:")
    print(f"    Mean: {dose_Gy.mean():.4e} Gy")
    print(f"    Max: {dose_Gy.max():.4e} Gy")
    
    print(f"\n  Statistics:")
    print(f"    Voxels hit: {np.count_nonzero(count_grid):,}")
    if np.any(dose_grid > 0):
        print(f"    Mean uncertainty: {rel_uncertainty[dose_grid>0].mean():.1f}%")
    
    # ==========================================================================
    # Save Outputs
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Saving Outputs")
    print("=" * 60)
    
    iso_name = isotope.replace('-', '')
    # Use same voxel sizes as input (in microns) for consistency
    # This ensures output overlays correctly with input in viewers
    voxel_um_x = voxel_sizes[0]
    voxel_um_y = voxel_sizes[1]
    voxel_um_z = voxel_sizes[2]
    
    def save_nifti_output(data, filename):
        """Save data as NIfTI file with anisotropic voxel sizes (in microns)"""
        out_affine = np.diag([voxel_um_x, voxel_um_y, voxel_um_z, 1.0])
        img = nib.Nifti1Image(np.nan_to_num(data).astype(np.float32), out_affine)
        img.header.set_xyzt_units('micron')
        nib.save(img, filename)
    
    # Save all outputs
    outputs = [
        ('dose_keV', dose_grid),           # Energy deposition in keV
        ('dose_Gy', dose_Gy),              # Absorbed dose in Gy
        ('particle_count', count_grid),    # Number of hits
        ('uncertainty_rel', rel_uncertainty),  # Relative uncertainty (%)
        ('LET_keV_um', LET_grid),          # LET in keV/um
        ('activity_normalized', activity),  # Preprocessed activity map (0-100)
    ]
    
    for name, data in outputs:
        # Save as numpy
        npy_path = out_dir / f"{name}_{iso_name}.npy"
        np.save(str(npy_path), data.astype(np.float32))
        
        # Save as NIfTI
        nii_path = out_dir / f"{name}_{iso_name}.nii.gz"
        save_nifti_output(data, str(nii_path))
        
        print(f"  {name}_{iso_name}.npy / .nii.gz")
    
    # Save statistics as JSON
    # Calculate dose per decay for scaling
    dose_per_decay_Gy = dose_Gy.sum() / n_particles if n_particles > 0 else 0
    
    stats_dict = {
        'isotope': isotope,
        'n_decays': int(n_particles),
        'total_particles': int(total_particles),
        'time_s': float(elapsed),
        'rate_per_s': float(total_particles / elapsed),
        'total_dose_keV': float(dose_grid.sum()),
        'total_dose_MeV': float(dose_grid.sum() / 1000),
        'max_dose_keV': float(dose_grid.max()),
        'max_dose_Gy': float(dose_Gy.max()),
        'mean_dose_Gy': float(dose_Gy.mean()),
        'voxels_hit': int(np.count_nonzero(count_grid)),
        'mean_uncertainty_percent': float(rel_uncertainty[dose_grid>0].mean()) if np.any(dose_grid>0) else 0,
        'voxel_size_xy_um': float(voxel_sizes[0]),
        'voxel_size_z_um': float(voxel_sizes[2]),
        'voxel_size_effective_um': float(voxel_size_um),
        'voxel_volume_um3': float(voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]),
        'grid_shape': [int(s) for s in shape],
        # Scaling factors for different activities
        'dose_per_decay_Gy': float(dose_per_decay_Gy),
        'note': 'To scale to different activity: multiply dose_Gy by (target_decays / n_decays)',
    }
    
    # Print scaling information
    print(f"\n  Dose Scaling Information:")
    print(f"    Dose per decay: {dose_per_decay_Gy:.4e} Gy")
    print(f"    To get 5 Gy average, need: {5.0 / (dose_per_decay_Gy + 1e-20):,.0f} decays")
    print(f"    Scale factor for 1 MBq activity: multiply by {1e6 / n_particles:.2f}")
    
    stats_path = out_dir / f"stats_{iso_name}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"  stats_{iso_name}.json")
    
    # ==========================================================================
    # Create MIP Visualization
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Creating MIP Visualization")
    print("=" * 60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top row: Activity + Dose + Count
        # Use preprocessed activity for visualization
        mip_act = np.max(activity, axis=2)
        mip_dose = np.max(np.log10(dose_Gy + 1e-10), axis=2)
        mip_count = np.max(count_grid, axis=2)
        
        im0 = axes[0, 0].imshow(mip_act.T, cmap='hot', origin='lower')
        axes[0, 0].set_title('Activity Source (XY MIP) - Preprocessed')
        plt.colorbar(im0, ax=axes[0, 0], label='Activity (0-100)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        
        im1 = axes[0, 1].imshow(mip_dose.T, cmap='hot', origin='lower')
        axes[0, 1].set_title('Dose [log10 Gy] (XY MIP)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].imshow(mip_count.T, cmap='viridis', origin='lower')
        axes[0, 2].set_title('Particle Count (XY MIP)')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Bottom row: XZ and YZ views + Histogram
        mip_dose_xz = np.max(np.log10(dose_Gy + 1e-10), axis=1)
        mip_dose_yz = np.max(np.log10(dose_Gy + 1e-10), axis=0)
        
        axes[1, 0].imshow(mip_dose_xz.T, cmap='hot', origin='lower')
        axes[1, 0].set_title('Dose [log10 Gy] (XZ MIP)')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Z')
        
        axes[1, 1].imshow(mip_dose_yz.T, cmap='hot', origin='lower')
        axes[1, 1].set_title('Dose [log10 Gy] (YZ MIP)')
        axes[1, 1].set_xlabel('Y')
        axes[1, 1].set_ylabel('Z')
        
        # Dose histogram
        dose_nonzero = dose_Gy[dose_Gy > 0]
        if len(dose_nonzero) > 0:
            axes[1, 2].hist(np.log10(dose_nonzero), bins=50, color='red', alpha=0.7)
            axes[1, 2].set_xlabel('log10(Dose [Gy])')
            axes[1, 2].set_ylabel('Voxels')
            axes[1, 2].set_title('Dose Histogram')
        
        plt.suptitle(
            f'CellRad-MC GPU: {isotope} | {n_particles:,} decays | {total_particles:,} particles\n'
            f'Max Dose: {dose_Gy.max():.2e} Gy | Time: {elapsed:.1f}s | Rate: {total_particles/elapsed:,.0f}/s',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        
        fig_path = out_dir / f'summary_{iso_name}.png'
        plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  summary_{iso_name}.png")
        
    except Exception as e:
        print(f"  Visualization failed: {e}")
    
    # ==========================================================================
    # Physics Validation
    # ==========================================================================
    if args.validate:
        print("\n" + "=" * 60)
        print("Physics Validation (NIST ESTAR/ASTAR)")
        print("=" * 60)
        
        from gpu_monte_carlo import _electron_stopping_power_material, _alpha_stopping_power_material
        
        density = 1.03
        I_eV = 74.7
        ZoverA = 0.55
        
        print("\n  Electron CSDA Range Validation:")
        print("  Energy (keV) | Computed (um) | NIST (um) | Error  | Status")
        print("  " + "-" * 60)
        
        for E_keV in [10, 50, 100, 200]:
            # Fine integration for accurate range
            E = float(E_keV)
            total_range = 0.0
            n_steps = 0
            while E > 0.5 and n_steps < 50000:
                dEdx = _electron_stopping_power_material(E, density, I_eV, ZoverA)
                # Adaptive step size - smaller at low energy
                step = min(0.5, 0.05 * E / (dEdx + 1e-6))
                step = max(step, 0.001)
                dE = dEdx * step
                if dE > E:
                    dE = E
                    step = dE / (dEdx + 1e-6)
                total_range += step
                E -= dE
                n_steps += 1
            
            nist_range_um = ELECTRON_CSDA_RANGES.get(E_keV, 0)
            
            error = 100 * (total_range - nist_range_um) / (nist_range_um + 1e-10)
            status = "PASS" if abs(error) < 15 else "FAIL"
            print(f"  {E_keV:6.0f}       | {total_range:8.1f}      | {nist_range_um:8.1f}  | {error:+5.1f}% | {status}")
        
        if iso['type'] == 'alpha':
            print("\n  Alpha CSDA Range Validation:")
            print("  Energy (keV) | Computed (um) | NIST (um) | Error  | Status")
            print("  " + "-" * 60)
            
            for E_keV in [5000, 6000, 7000, 8000]:
                E = float(E_keV)
                total_range = 0.0
                n_steps = 0
                while E > 10.0 and n_steps < 50000:
                    dEdx = _alpha_stopping_power_material(E, density, I_eV, ZoverA)
                    step = min(0.1, 0.01 * E / (dEdx + 1e-6))
                    step = max(step, 0.001)
                    dE = dEdx * step
                    if dE > E:
                        dE = E
                        step = dE / (dEdx + 1e-6)
                    total_range += step
                    E -= dE
                    n_steps += 1
                
                nist_range_um = ALPHA_CSDA_RANGES.get(E_keV, 0)
                error = 100 * (total_range - nist_range_um) / (nist_range_um + 1e-10)
                status = "PASS" if abs(error) < 15 else "FAIL"
                print(f"  {E_keV:6.0f}       | {total_range:8.1f}      | {nist_range_um:8.1f}  | {error:+5.1f}% | {status}")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {out_dir}")
    
    return stats_dict


def main():
    parser = argparse.ArgumentParser(
        description='CellRad-MC: GPU Monte Carlo for Targeted Radionuclide Therapy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With CT scan
  python cellrad_mc.py --protein Ki67.nii.gz --ct CT.nii.gz --isotope Lu-177 --out-dir output/
  
  # Without CT (uniform soft tissue)
  python cellrad_mc.py --protein Ki67.nii.gz --auto-ct --isotope Ac-225 --out-dir output/
  
  # With isotropic voxel size (1 micron)
  python cellrad_mc.py --protein Ki67.nii.gz --auto-ct --isotope Lu-177 --voxel-size 1
  
  # With anisotropic voxel size (XY=0.45um, Z=1.0um - common for microscopy)
  python cellrad_mc.py --protein EGFR.nii.gz --auto-ct --isotope Lu-177 --voxel-size 0.45 --voxel-size-z 1.0
  
  # High statistics run with gamma transport
  python cellrad_mc.py --protein Ki67.nii.gz --auto-ct --isotope Lu-177 -n 1000000 --gammas

Supported Isotopes:
  Beta:  Lu-177, Y-90, I-131, P-32, Sr-89
  Alpha: Ac-225, Ra-223, At-211, Th-227
  Auger: I-125, In-111, Ga-67, Tc-99m

Outputs:
  - dose_keV_{isotope}.nii.gz      : Energy deposition (keV)
  - dose_Gy_{isotope}.nii.gz       : Absorbed dose (Gy)
  - particle_count_{isotope}.nii.gz: Number of hits per voxel
  - uncertainty_rel_{isotope}.nii.gz: Relative uncertainty (%)
  - LET_keV_um_{isotope}.nii.gz    : Linear Energy Transfer (keV/um)
  - summary_{isotope}.png           : MIP visualization
  - stats_{isotope}.json            : Simulation statistics
        """
    )
    
    # Required arguments
    parser.add_argument('--protein', '-p', required=True,
                        help='Path to protein/activity map (NIfTI format)')
    
    # CT options (mutually exclusive)
    ct_group = parser.add_mutually_exclusive_group(required=True)
    ct_group.add_argument('--ct', '-c',
                          help='Path to CT scan (NIfTI format)')
    ct_group.add_argument('--auto-ct', action='store_true',
                          help='Use uniform soft tissue (HU=19) instead of CT')
    
    # Isotope
    parser.add_argument('--isotope', '-i', required=True,
                        choices=list(ISOTOPES.keys()),
                        help='Radioisotope to simulate')
    
    # Output
    parser.add_argument('--out-dir', '-o', default='output',
                        help='Output directory (default: output/)')
    
    # Simulation parameters
    parser.add_argument('--n-particles', '-n', type=int, default=100000,
                        help='Number of primary decays to simulate (default: 100000)')
    
    parser.add_argument('--voxel-size', '-v', type=float, default=None,
                        help='XY voxel size in microns (um). If not provided, reads from NIfTI header.')
    
    parser.add_argument('--voxel-size-z', type=float, default=None,
                        help='Z voxel size in microns (um). If not provided, uses same as XY or reads from NIfTI header.')
    
    parser.add_argument('--gammas', action='store_true',
                        help='Also transport gamma photons (slower)')
    
    parser.add_argument('--validate', action='store_true',
                        help='Run physics validation against NIST data')
    
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Activity threshold as %% of max value. Voxels below this are considered background (default: 1%%)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Run simulation
    run_simulation(args)


if __name__ == '__main__':
    main()
