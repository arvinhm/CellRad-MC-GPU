#!/usr/bin/env python3
"""
CellRad-MC Comprehensive Cell-Level Analysis
=============================================

This script:
1. Runs cellrad_mc.py to get GPU Monte Carlo simulation results
2. Loads the simulation outputs
3. Performs DNA damage and cell survival analysis per cell
4. Generates Nature-level figures and CSV outputs

Usage:
    python run_comprehensive_analysis.py --nuclei nuclei.nii.gz --protein protein.nii.gz \
        --output results/ --isotope Lu-177 --n-particles 10000000 --voxel-size 1.0

Author: Arvin
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import ndimage, stats
from scipy.ndimage import labeled_comprehension, center_of_mass, sum as nd_sum, mean as nd_mean
from scipy.ndimage import maximum as nd_max, minimum as nd_min
from pathlib import Path
import json
import time
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import argparse

# =============================================================================
# MATPLOTLIB SETUP FOR NATURE-QUALITY FIGURES
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,              # Increased from 11
    'axes.labelsize': 16,         # Increased from 12
    'axes.titlesize': 18,         # Increased from 14
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.5,        # Increased from 1.2
    'xtick.labelsize': 13,        # Increased from 10
    'ytick.labelsize': 13,        # Increased from 10
    'legend.fontsize': 12,        # Increased from 10
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Consistent color scheme for spatial regions
REGION_COLORS = {
    'core': '#E31A1C',        # Red - tumor core (most active)
    'intermediate': '#FF7F00', # Orange - transition zone
    'periphery': '#33A02C',    # Green - tumor edge (less active)
}

COLORS = {
    'primary': '#1976D2',
    'secondary': '#E53935', 
    'accent': '#43A047',
    'warning': '#FB8C00',
    'purple': '#8E24AA',
}

# =============================================================================
# DNA DAMAGE PARAMETERS - PEER-REVIEWED LITERATURE VALUES
# =============================================================================
# 
# PRIMARY REFERENCES:
# 1. Hong et al. (2024) Theranostics 14(18):7122-7139 - RPT dosimetry with ST
# 2. Nikjoo et al. (2001) Radiat. Res. 156:577-583 - DSB yields
# 3. Rothkamm & Löbrich (2003) PNAS 100:5057-5062 - DSB repair
# 4. van Leeuwen et al. (2018) Radiat. Oncol. 13:96 - LQ parameters
# 5. ICRU Report 90 (2016) - LET values
# 6. Sgouros et al. (2010) J Nucl Med 51:311-328 - Alpha particle dosimetry
# 7. Kassis (2004) J Nucl Med 45:1342-1349 - Auger electron radiobiology
# 8. Friedland et al. (2003) Radiat. Res. 159:401-410 - Track structure
#
# LET VALUES (ICRU Report 90, Sgouros 2010):
# - Beta (Lu-177, 134 keV mean): ~0.2-0.5 keV/μm (low LET)
# - Beta (Y-90, 934 keV mean): ~0.2 keV/μm (low LET)
# - Alpha (5-8 MeV): 80-120 keV/μm at Bragg peak, ~100 keV/μm average
# - Auger (I-125): 4-26 keV/μm within nm range (Kassis 2004)
#
# DSB YIELDS (Nikjoo et al. 2001, Friedland et al. 2003):
# - Low-LET (γ, β): 35-40 DSB/Gy/Gbp
# - High-LET (α): 100-200 DSB/Gy/Gbp (due to clustered damage)
#
# SSB:DSB RATIO (Nikjoo et al. 2001):
# - Low-LET: 20-30:1
# - High-LET: 4-8:1
#
# COMPLEX DSB FRACTION (Friedland et al. 2003):
# - γ-rays: 15-25%
# - α-particles: 60-80%
#
# LQ MODEL PARAMETERS (van Leeuwen et al. 2018):
# - Tumor cells, low-LET: α = 0.20-0.45 Gy⁻¹, β = 0.02-0.05 Gy⁻²
# - α/β ratio typically 8-12 Gy for tumors
# - For high-LET: α dominates (β → 0), use D37 model
#
# RBE VALUES (Sgouros et al. 2010, ICRP 103):
# - β-particles: RBE = 1.0
# - α-particles: RBE = 3-8 (typically 5 for therapeutic applications)
# - Auger (nuclear): RBE = 4-8 (depends on internalization)
#
# =============================================================================

RADIATION_PARAMS = {
    # =========================================================================
    # BETA EMITTERS (Low LET: 0.2-0.5 keV/μm)
    # =========================================================================
    # LQ parameters from van Leeuwen et al. (2018) meta-analysis of clinical data
    # DSB yields from Nikjoo et al. (2001) - 35 DSB/Gy/Gbp for low-LET
    
    'Lu-177': {
        'type': 'beta',
        'E_mean_keV': 134.2,           # ICRP 107
        'E_max_keV': 498.3,            # ICRP 107
        'LET_keV_um': 0.34,            # ICRU 90 for ~134 keV electrons in water
        'alpha_LQ': 0.30,              # van Leeuwen 2018: tumor cells
        'beta_LQ': 0.030,              # α/β = 10 Gy (typical for tumor)
        'dsb_per_Gy_Gbp': 35,          # Nikjoo 2001
        'ssb_dsb_ratio': 25,           # Nikjoo 2001
        'complex_frac': 0.20,          # Friedland 2003
        'RBE': 1.0,                    # Reference radiation
    },
    'Y-90': {
        'type': 'beta',
        'E_mean_keV': 933.7,           # ICRP 107
        'E_max_keV': 2280.0,           # ICRP 107
        'LET_keV_um': 0.22,            # ICRU 90 for ~934 keV electrons
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 35,
        'ssb_dsb_ratio': 25,
        'complex_frac': 0.18,          # Slightly lower due to higher energy
        'RBE': 1.0,
    },
    'I-131': {
        'type': 'beta',
        'E_mean_keV': 191.6,           # ICRP 107
        'E_max_keV': 606.3,            # ICRP 107
        'LET_keV_um': 0.38,
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 35,
        'ssb_dsb_ratio': 25,
        'complex_frac': 0.20,
        'RBE': 1.0,
    },
    'P-32': {
        'type': 'beta',
        'E_mean_keV': 695.0,
        'E_max_keV': 1710.0,
        'LET_keV_um': 0.24,
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 35,
        'ssb_dsb_ratio': 25,
        'complex_frac': 0.18,
        'RBE': 1.0,
    },
    'Sr-89': {
        'type': 'beta',
        'E_mean_keV': 583.0,
        'E_max_keV': 1495.0,
        'LET_keV_um': 0.25,
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 35,
        'ssb_dsb_ratio': 25,
        'complex_frac': 0.18,
        'RBE': 1.0,
    },
    'Tb-161': {
        'type': 'beta',
        'E_mean_keV': 154.0,
        'E_max_keV': 593.0,
        'LET_keV_um': 0.36,
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 35,
        'ssb_dsb_ratio': 25,
        'complex_frac': 0.20,
        'RBE': 1.0,
    },
    
    # =========================================================================
    # ALPHA EMITTERS (High LET: 80-120 keV/μm)
    # =========================================================================
    # From Sgouros et al. (2010) MIRD Pamphlet No. 22
    # Alpha particles: 5-8 MeV → LET 80-120 keV/μm at Bragg peak
    # DSB yields 4-5x higher than low-LET (Nikjoo 2001)
    # β_LQ → 0 for high-LET (single-track killing dominates)
    # Use D37 model: S = exp(-D/D37) where D37 = 1/α
    
    # =========================================================================
    # ALPHA EMITTERS - ICRP 107 Validated with Branching Ratios
    # =========================================================================
    'Ac-225': {
        'type': 'alpha',
        # Corrected with ICRP 107 branching ratios
        'alpha_chain_MeV': [5.83, 6.34, 7.07, 0.13, 8.19],  # Bi-213: 5.87*0.022=0.13, Po-213: 8.38*0.978=8.19
        'total_alpha_E_MeV': 27.6,     # ICRP 107 weighted total
        'LET_keV_um': 97,              # Average LET (Sgouros 2010)
        'alpha_LQ': 0.80,              # D37 ≈ 1.25 Gy (Pouget et al. 2015)
        'beta_LQ': 0.0,                # β→0 for high-LET
        'dsb_per_Gy_Gbp': 140,         # ~4x higher than low-LET (Nikjoo 2001)
        'ssb_dsb_ratio': 5,            # Much lower for high-LET
        'complex_frac': 0.70,          # High clustered damage (Friedland 2003)
        'RBE': 5.0,                    # ICRP 103, Sgouros 2010
    },
    'At-211': {
        'type': 'alpha',
        # At-211: 41.8% α branch, Po-211 always α
        'alpha_chain_MeV': [2.45, 7.45],  # 5.87*0.418=2.45, 7.45*1.0=7.45
        'total_alpha_E_MeV': 6.79,     # ICRP 107 weighted
        'LET_keV_um': 99,
        'alpha_LQ': 0.85,              # D37 ≈ 1.18 Gy
        'beta_LQ': 0.0,
        'dsb_per_Gy_Gbp': 150,
        'ssb_dsb_ratio': 5,
        'complex_frac': 0.72,
        'RBE': 5.0,
    },
    'Ra-223': {
        'type': 'alpha',
        # Ra-223 chain: 4 alphas, ~99.7% yield for Bi-211
        'alpha_chain_MeV': [5.72, 6.82, 7.39, 6.60],  # Bi-211: 6.62*0.997=6.60
        'total_alpha_E_MeV': 26.4,     # ICRP 107 weighted
        'LET_keV_um': 93,
        'alpha_LQ': 0.75,              # D37 ≈ 1.33 Gy
        'beta_LQ': 0.0,
        'dsb_per_Gy_Gbp': 135,
        'ssb_dsb_ratio': 6,
        'complex_frac': 0.68,
        'RBE': 5.0,
    },
    'Th-227': {
        'type': 'alpha',
        # Th-227 -> Ra-223 chain: 5 alphas
        'alpha_chain_MeV': [6.04, 5.72, 6.82, 7.39, 6.60],
        'total_alpha_E_MeV': 32.5,     # ICRP 107 weighted
        'LET_keV_um': 95,
        'alpha_LQ': 0.78,
        'beta_LQ': 0.0,
        'dsb_per_Gy_Gbp': 138,
        'ssb_dsb_ratio': 5,
        'complex_frac': 0.70,
        'RBE': 5.0,
    },
    
    # =========================================================================
    # AUGER EMITTERS (Variable LET: 4-26 keV/μm locally)
    # =========================================================================
    # From Kassis (2004) J Nucl Med - Auger electron radiobiology
    # Auger electrons: very short range (nm), high LET locally
    # RBE depends on nuclear localization (1-8x)
    # When internalized in nucleus, effective LET is very high
    
    'I-125': {
        'type': 'auger',
        'n_auger_per_decay': 21,       # ICRP 107
        'total_auger_E_keV': 12.2,     # ICRP 107
        'LET_keV_um': 26,              # Effective LET when nuclear (Kassis 2004)
        'alpha_LQ': 0.50,              # Intermediate (Kassis 2004)
        'beta_LQ': 0.020,
        'dsb_per_Gy_Gbp': 80,          # ~2.3x low-LET when nuclear
        'ssb_dsb_ratio': 10,
        'complex_frac': 0.50,
        'RBE': 4.0,                    # Depends on localization (Kassis 2004)
    },
    'In-111': {
        'type': 'auger',
        'n_auger_per_decay': 14.7,
        'total_auger_E_keV': 6.8,
        'LET_keV_um': 12,
        'alpha_LQ': 0.42,
        'beta_LQ': 0.025,
        'dsb_per_Gy_Gbp': 60,
        'ssb_dsb_ratio': 12,
        'complex_frac': 0.40,
        'RBE': 2.5,
    },
    'Ga-67': {
        'type': 'auger',
        'n_auger_per_decay': 4.7,
        'total_auger_E_keV': 1.1,
        'LET_keV_um': 8,
        'alpha_LQ': 0.38,
        'beta_LQ': 0.028,
        'dsb_per_Gy_Gbp': 50,
        'ssb_dsb_ratio': 15,
        'complex_frac': 0.35,
        'RBE': 1.8,
    },
    'Tc-99m': {
        'type': 'auger',
        'n_auger_per_decay': 4.0,
        'total_auger_E_keV': 0.9,
        'LET_keV_um': 4,
        'alpha_LQ': 0.32,
        'beta_LQ': 0.030,
        'dsb_per_Gy_Gbp': 38,
        'ssb_dsb_ratio': 22,
        'complex_frac': 0.22,
        'RBE': 1.2,
    },
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CellAnalysis:
    """Complete analysis for a single cell"""
    cell_id: int
    volume_um3: float
    centroid_x: float
    centroid_y: float
    centroid_z: float
    mean_expression: float
    max_expression: float
    total_expression: float
    mean_dose_Gy: float
    max_dose_Gy: float
    min_dose_Gy: float
    total_dose_Gy: float
    dose_std_Gy: float
    total_hits: int
    hits_per_um3: float
    total_energy_keV: float
    mean_energy_keV: float
    n_SSB: int
    n_DSB: int
    n_complex_DSB: int
    ssb_dsb_ratio: float
    survival_fraction: float
    lethal_dose_fraction: float
    mean_LET_keV_um: float


def run_cellrad_mc(protein_path: str, output_dir: str, isotope: str, 
                   n_particles: int, voxel_size: float,
                   ct_path: str = None, auto_ct: bool = True) -> str:
    """
    Run cellrad_mc.py to perform GPU Monte Carlo simulation.
    
    Args:
        protein_path: Path to protein expression NIfTI file
        output_dir: Output directory for results
        isotope: Radioisotope name (e.g., 'Lu-177', 'Ac-225')
        n_particles: Number of particles to simulate
        voxel_size: Voxel size in microns
        ct_path: Path to CT scan NIfTI file (optional)
        auto_ct: Use auto-generated soft tissue CT (default: True)
    
    Returns the output directory path.
    """
    cmd = [
        sys.executable, 'cellrad_mc.py',
        '--protein', protein_path,
        '--isotope', isotope,
        '--out-dir', output_dir,
        '--n-particles', str(n_particles),
        '--voxel-size', str(voxel_size)
    ]
    
    # Add CT argument
    if ct_path:
        cmd.extend(['--ct', ct_path])
    elif auto_ct:
        cmd.append('--auto-ct')
    else:
        cmd.append('--auto-ct')  # Default to auto-ct
    
    print(f"\n{'='*60}")
    print("Running CellRad-MC GPU Monte Carlo Simulation")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Run the simulation
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"cellrad_mc.py failed with return code {result.returncode}")
    
    return output_dir


def load_simulation_results(output_dir: str, isotope: str) -> Dict[str, np.ndarray]:
    """Load simulation results from cellrad_mc.py output."""
    iso_name = isotope.replace('-', '')
    
    results = {}
    
    # Load dose in Gy
    dose_path = Path(output_dir) / f'dose_Gy_{iso_name}.nii.gz'
    if dose_path.exists():
        results['dose_Gy'] = nib.load(str(dose_path)).get_fdata()
        print(f"  Loaded dose_Gy: shape={results['dose_Gy'].shape}, max={results['dose_Gy'].max():.4e} Gy")
    
    # Load energy deposition
    energy_path = Path(output_dir) / f'dose_keV_{iso_name}.nii.gz'
    if energy_path.exists():
        results['energy_keV'] = nib.load(str(energy_path)).get_fdata()
        print(f"  Loaded energy_keV: total={results['energy_keV'].sum()/1000:.2f} MeV")
    
    # Load particle counts
    count_path = Path(output_dir) / f'particle_count_{iso_name}.nii.gz'
    if count_path.exists():
        results['hits'] = nib.load(str(count_path)).get_fdata()
        print(f"  Loaded particle_count: total={results['hits'].sum():.0f} hits")
    
    # Load uncertainty
    unc_path = Path(output_dir) / f'uncertainty_rel_{iso_name}.nii.gz'
    if unc_path.exists():
        results['uncertainty'] = nib.load(str(unc_path)).get_fdata()
    
    # Load LET
    let_path = Path(output_dir) / f'LET_keV_um_{iso_name}.nii.gz'
    if let_path.exists():
        results['LET'] = nib.load(str(let_path)).get_fdata()
    
    # Load stats
    stats_path = Path(output_dir) / f'stats_{iso_name}.json'
    if stats_path.exists():
        with open(stats_path) as f:
            results['stats'] = json.load(f)
    
    return results


def analyze_cells_vectorized(nuclei_mask: np.ndarray,
                              expression_map: np.ndarray,
                              dose_Gy: np.ndarray,
                              energy_keV: np.ndarray,
                              hits: np.ndarray,
                              voxel_size_um: float,
                              isotope: str) -> List[CellAnalysis]:
    """
    Analyze all cells using vectorized scipy.ndimage operations for speed.
    """
    # Get cell IDs
    cell_ids = np.unique(nuclei_mask)
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = len(cell_ids)
    
    print(f"\nAnalyzing {n_cells} cells (vectorized)...")
    
    labels = nuclei_mask.astype(np.int32)
    
    # Compute all statistics at once
    print("  Computing cell metrics...")
    volumes = nd_sum(np.ones_like(labels), labels, cell_ids) * (voxel_size_um ** 3)
    
    mean_expr = nd_mean(expression_map, labels, cell_ids)
    max_expr = nd_max(expression_map, labels, cell_ids)
    sum_expr = nd_sum(expression_map, labels, cell_ids)
    
    mean_dose = nd_mean(dose_Gy, labels, cell_ids)
    max_dose = nd_max(dose_Gy, labels, cell_ids)
    min_dose = nd_min(dose_Gy, labels, cell_ids)
    sum_dose = nd_sum(dose_Gy, labels, cell_ids)
    dose_var = labeled_comprehension(dose_Gy, labels, cell_ids, np.var, float, 0)
    dose_std = np.sqrt(dose_var)
    
    sum_hits = nd_sum(hits, labels, cell_ids)
    sum_energy = nd_sum(energy_keV, labels, cell_ids)
    mean_energy = nd_mean(energy_keV, labels, cell_ids)
    
    centroids = center_of_mass(np.ones_like(labels), labels, cell_ids)
    
    # DNA damage calculations based on radiation type
    # Using peer-reviewed parameters from literature (see RADIATION_PARAMS documentation)
    print("  Computing DNA damage and survival...")
    params = RADIATION_PARAMS.get(isotope, RADIATION_PARAMS['Lu-177'])
    
    # DNA content per diploid human cell: 6.4 Gbp (3.2 Gbp per haploid genome x 2)
    dna_content = 6.4  # Gbp per cell (Friedland et al. 2003)
    
    # Extract radiation parameters
    alpha_LQ = params['alpha_LQ']            # Linear component of LQ model (Gy⁻¹)
    beta_LQ = params['beta_LQ']              # Quadratic component of LQ model (Gy⁻²)
    dsb_per_Gy_per_Gbp = params['dsb_per_Gy_Gbp']  # DSB yield (Nikjoo 2001)
    ssb_dsb_ratio_val = params['ssb_dsb_ratio']    # SSB:DSB ratio
    complex_fraction = params['complex_frac']      # Complex DSB fraction
    let_keV_um = params['LET_keV_um']              # Linear Energy Transfer
    rad_type = params['type']
    rbe = params['RBE']                            # Relative Biological Effectiveness
    
    # Calculate DSBs and SSBs
    n_DSB = (mean_dose * dna_content * dsb_per_Gy_per_Gbp).astype(int)
    n_SSB = (n_DSB * ssb_dsb_ratio_val).astype(int)
    n_complex_DSB = (n_DSB * complex_fraction).astype(int)
    ssb_dsb_ratios = n_SSB / np.maximum(n_DSB, 1)
    
    # Cell survival
    if rad_type == 'alpha':
        D37 = 1.0 / alpha_LQ
        survival = np.exp(-mean_dose / D37)
    else:
        survival = np.exp(-alpha_LQ * mean_dose - beta_LQ * mean_dose**2)
    
    survival = np.clip(survival, 0, 1)
    
    if beta_LQ > 0:
        D37 = (-alpha_LQ + np.sqrt(alpha_LQ**2 + 4*beta_LQ)) / (2*beta_LQ)
    else:
        D37 = 1.0 / alpha_LQ
    
    lethal_fraction = mean_dose / D37
    hits_per_um3 = sum_hits / volumes
    
    # Build results
    print("  Building results...")
    results = []
    for i, cell_id in enumerate(cell_ids):
        cx, cy, cz = centroids[i] if isinstance(centroids[i], tuple) else (0, 0, 0)
        
        results.append(CellAnalysis(
            cell_id=int(cell_id),
            volume_um3=float(volumes[i]),
            centroid_x=float(cx * voxel_size_um),
            centroid_y=float(cy * voxel_size_um),
            centroid_z=float(cz * voxel_size_um),
            mean_expression=float(mean_expr[i]),
            max_expression=float(max_expr[i]),
            total_expression=float(sum_expr[i]),
            mean_dose_Gy=float(mean_dose[i]),
            max_dose_Gy=float(max_dose[i]),
            min_dose_Gy=float(min_dose[i]),
            total_dose_Gy=float(sum_dose[i]),
            dose_std_Gy=float(dose_std[i]),
            total_hits=int(sum_hits[i]),
            hits_per_um3=float(hits_per_um3[i]),
            total_energy_keV=float(sum_energy[i]),
            mean_energy_keV=float(mean_energy[i]),
            n_SSB=int(n_SSB[i]),
            n_DSB=int(n_DSB[i]),
            n_complex_DSB=int(n_complex_DSB[i]),
            ssb_dsb_ratio=float(ssb_dsb_ratios[i]),
            survival_fraction=float(survival[i]),
            lethal_dose_fraction=float(lethal_fraction[i]),
            mean_LET_keV_um=float(let_keV_um)
        ))
    
    print(f"  Completed: {len(results)} cells analyzed")
    return results


def analyze_spatial_heterogeneity(cell_analyses: List[CellAnalysis], nuclei_mask: np.ndarray,
                                   voxel_size_um: float) -> Dict:
    """
    Perform sophisticated spatial analysis: tumor core vs periphery.
    
    Based on Hong et al. 2024 methodology for spatial transcriptomics dosimetry.
    """
    print("\n--- Spatial Heterogeneity Analysis ---")
    
    # Extract cell positions and metrics
    cx = np.array([c.centroid_x for c in cell_analyses])
    cy = np.array([c.centroid_y for c in cell_analyses])
    cz = np.array([c.centroid_z for c in cell_analyses])
    doses = np.array([c.mean_dose_Gy for c in cell_analyses])
    survivals = np.array([c.survival_fraction for c in cell_analyses])
    dsbs = np.array([c.n_DSB for c in cell_analyses])
    expressions = np.array([c.mean_expression for c in cell_analyses])
    
    # Calculate tumor centroid
    tumor_cx = np.mean(cx)
    tumor_cy = np.mean(cy)
    tumor_cz = np.mean(cz)
    
    # Calculate distance from tumor centroid (3D)
    distances = np.sqrt((cx - tumor_cx)**2 + (cy - tumor_cy)**2 + (cz - tumor_cz)**2)
    
    # Define core vs periphery (using percentiles)
    # Core: cells within 33rd percentile of distance
    # Intermediate: 33-66th percentile
    # Periphery: >66th percentile
    d_33 = np.percentile(distances, 33)
    d_66 = np.percentile(distances, 66)
    
    core_mask = distances <= d_33
    intermediate_mask = (distances > d_33) & (distances <= d_66)
    periphery_mask = distances > d_66
    
    print(f"  Tumor centroid: ({tumor_cx:.1f}, {tumor_cy:.1f}, {tumor_cz:.1f}) μm")
    print(f"  Core cells (<{d_33:.1f} μm): {np.sum(core_mask)}")
    print(f"  Intermediate cells: {np.sum(intermediate_mask)}")
    print(f"  Periphery cells (>{d_66:.1f} μm): {np.sum(periphery_mask)}")
    
    # Compute statistics for each region
    results = {
        'tumor_centroid': (tumor_cx, tumor_cy, tumor_cz),
        'distances': distances,
        'core_threshold': d_33,
        'periphery_threshold': d_66,
        'regions': {
            'core': {
                'n_cells': int(np.sum(core_mask)),
                'mean_dose': float(np.mean(doses[core_mask])) if np.any(core_mask) else 0,
                'std_dose': float(np.std(doses[core_mask])) if np.any(core_mask) else 0,
                'mean_survival': float(np.mean(survivals[core_mask])) if np.any(core_mask) else 0,
                'mean_dsb': float(np.mean(dsbs[core_mask])) if np.any(core_mask) else 0,
                'mean_expression': float(np.mean(expressions[core_mask])) if np.any(core_mask) else 0,
            },
            'intermediate': {
                'n_cells': int(np.sum(intermediate_mask)),
                'mean_dose': float(np.mean(doses[intermediate_mask])) if np.any(intermediate_mask) else 0,
                'std_dose': float(np.std(doses[intermediate_mask])) if np.any(intermediate_mask) else 0,
                'mean_survival': float(np.mean(survivals[intermediate_mask])) if np.any(intermediate_mask) else 0,
                'mean_dsb': float(np.mean(dsbs[intermediate_mask])) if np.any(intermediate_mask) else 0,
                'mean_expression': float(np.mean(expressions[intermediate_mask])) if np.any(intermediate_mask) else 0,
            },
            'periphery': {
                'n_cells': int(np.sum(periphery_mask)),
                'mean_dose': float(np.mean(doses[periphery_mask])) if np.any(periphery_mask) else 0,
                'std_dose': float(np.std(doses[periphery_mask])) if np.any(periphery_mask) else 0,
                'mean_survival': float(np.mean(survivals[periphery_mask])) if np.any(periphery_mask) else 0,
                'mean_dsb': float(np.mean(dsbs[periphery_mask])) if np.any(periphery_mask) else 0,
                'mean_expression': float(np.mean(expressions[periphery_mask])) if np.any(periphery_mask) else 0,
            }
        },
        'masks': {
            'core': core_mask,
            'intermediate': intermediate_mask,
            'periphery': periphery_mask
        }
    }
    
    # Radial dose profile (binned)
    n_bins = 20
    bin_edges = np.linspace(0, distances.max(), n_bins + 1)
    radial_dose = []
    radial_survival = []
    radial_dsb = []
    radial_r = []
    
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        if np.any(mask):
            radial_dose.append(np.mean(doses[mask]))
            radial_survival.append(np.mean(survivals[mask]))
            radial_dsb.append(np.mean(dsbs[mask]))
            radial_r.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    results['radial_profile'] = {
        'r': radial_r,
        'dose': radial_dose,
        'survival': radial_survival,
        'dsb': radial_dsb
    }
    
    # Heterogeneity indices
    # Gini coefficient (0=uniform, 1=maximum heterogeneity)
    def gini_coefficient(x):
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0
    
    results['heterogeneity'] = {
        'dose_cv': float(np.std(doses) / np.mean(doses)) if np.mean(doses) > 0 else 0,
        'dose_gini': float(gini_coefficient(doses)),
        'survival_cv': float(np.std(survivals) / np.mean(survivals)) if np.mean(survivals) > 0 else 0,
        'dsb_cv': float(np.std(dsbs) / np.mean(dsbs)) if np.mean(dsbs) > 0 else 0,
    }
    
    print(f"  Dose heterogeneity (CV): {results['heterogeneity']['dose_cv']*100:.1f}%")
    print(f"  Dose Gini coefficient: {results['heterogeneity']['dose_gini']:.3f}")
    
    return results


def create_figures(nuclei_mask, expression_map, dose_Gy, cell_analyses, output_dir, isotope,
                   spatial_results=None):
    """Create Nature-quality analysis figures."""
    print("\n--- Generating Figures ---")
    
    out_path = Path(output_dir)
    
    # Extract data from cell analyses
    doses = np.array([c.mean_dose_Gy for c in cell_analyses])
    survivals = np.array([c.survival_fraction for c in cell_analyses])
    dsbs = np.array([c.n_DSB for c in cell_analyses])
    ssbs = np.array([c.n_SSB for c in cell_analyses])
    expressions = np.array([c.mean_expression for c in cell_analyses])
    volumes = np.array([c.volume_um3 for c in cell_analyses])
    cx = np.array([c.centroid_x for c in cell_analyses])
    cy = np.array([c.centroid_y for c in cell_analyses])
    
    # Figure 1: Data Overview
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # MIP of nuclei
    mip_nuclei = np.max(nuclei_mask, axis=2)
    im0 = axes[0, 0].imshow(mip_nuclei.T, cmap='nipy_spectral', origin='lower')
    axes[0, 0].set_title('A) Nuclei Mask (XY MIP)')
    axes[0, 0].set_xlabel('X (voxels)')
    axes[0, 0].set_ylabel('Y (voxels)')
    
    # MIP of expression
    mip_expr = np.max(expression_map, axis=2)
    im1 = axes[0, 1].imshow(mip_expr.T, cmap='hot', origin='lower')
    axes[0, 1].set_title('B) Protein Expression (XY MIP)')
    plt.colorbar(im1, ax=axes[0, 1], label='Expression')
    
    # MIP of dose
    mip_dose = np.max(dose_Gy, axis=2)
    im2 = axes[0, 2].imshow(mip_dose.T, cmap='hot', origin='lower', 
                            norm=LogNorm(vmin=max(1e-10, mip_dose[mip_dose>0].min()) if np.any(mip_dose>0) else 1e-10,
                                        vmax=max(1e-9, mip_dose.max())))
    axes[0, 2].set_title('C) Dose Distribution (XY MIP)')
    plt.colorbar(im2, ax=axes[0, 2], label='Dose (Gy)')
    
    # XZ views
    mip_nuclei_xz = np.max(nuclei_mask, axis=1)
    axes[1, 0].imshow(mip_nuclei_xz.T, cmap='nipy_spectral', origin='lower')
    axes[1, 0].set_title('D) Nuclei Mask (XZ MIP)')
    axes[1, 0].set_xlabel('X (voxels)')
    axes[1, 0].set_ylabel('Z (voxels)')
    
    mip_expr_xz = np.max(expression_map, axis=1)
    im4 = axes[1, 1].imshow(mip_expr_xz.T, cmap='hot', origin='lower')
    axes[1, 1].set_title('E) Protein Expression (XZ MIP)')
    plt.colorbar(im4, ax=axes[1, 1], label='Expression')
    
    mip_dose_xz = np.max(dose_Gy, axis=1)
    im5 = axes[1, 2].imshow(mip_dose_xz.T, cmap='hot', origin='lower',
                            norm=LogNorm(vmin=max(1e-10, mip_dose_xz[mip_dose_xz>0].min()) if np.any(mip_dose_xz>0) else 1e-10,
                                        vmax=max(1e-9, mip_dose_xz.max())))
    axes[1, 2].set_title('F) Dose Distribution (XZ MIP)')
    plt.colorbar(im5, ax=axes[1, 2], label='Dose (Gy)')
    
    plt.tight_layout()
    plt.savefig(out_path / 'Figure1_Data_Overview.png', dpi=300)
    plt.close()
    print("  Saved: Figure1_Data_Overview.png")
    
    # Figure 2: Dose-Response
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dose-response curve
    axes[0, 0].scatter(doses, survivals, alpha=0.5, s=10, c=COLORS['primary'])
    dose_range = np.linspace(0, doses.max() * 1.1, 100)
    params = RADIATION_PARAMS.get(isotope, RADIATION_PARAMS['Lu-177'])
    if params['type'] == 'alpha':
        survival_theory = np.exp(-dose_range / (1.0 / params['alpha_LQ']))
    else:
        survival_theory = np.exp(-params['alpha_LQ'] * dose_range - params['beta_LQ'] * dose_range**2)
    axes[0, 0].plot(dose_range, survival_theory, 'r-', lw=2, label='LQ model')
    axes[0, 0].set_xlabel('Mean Dose (Gy)')
    axes[0, 0].set_ylabel('Survival Fraction')
    axes[0, 0].set_title('A) Dose-Response Curve')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylim([1e-4, 1])
    
    # DSB vs Dose
    axes[0, 1].scatter(doses, dsbs, alpha=0.5, s=10, c=COLORS['secondary'])
    slope = np.polyfit(doses[doses > 0], dsbs[doses > 0], 1)[0] if np.any(doses > 0) else 0
    axes[0, 1].plot(dose_range, slope * dose_range, 'k--', lw=2, label=f'~{slope:.0f} DSB/Gy')
    axes[0, 1].set_xlabel('Mean Dose (Gy)')
    axes[0, 1].set_ylabel('Number of DSBs')
    axes[0, 1].set_title('B) DSB Induction')
    axes[0, 1].legend(frameon=False)
    
    # Expression vs Dose correlation
    axes[1, 0].scatter(expressions, doses, alpha=0.5, s=10, c=COLORS['accent'])
    r = np.corrcoef(expressions, doses)[0, 1] if len(expressions) > 1 else 0
    axes[1, 0].set_xlabel('Mean Expression (a.u.)')
    axes[1, 0].set_ylabel('Mean Dose (Gy)')
    axes[1, 0].set_title(f'C) Expression-Dose Correlation')
    axes[1, 0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[1, 0].transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    # Survival distribution
    axes[1, 1].hist(survivals, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0.5, color='orange', linestyle='--', lw=2, label='50% survival')
    axes[1, 1].axvline(0.1, color='red', linestyle='--', lw=2, label='10% survival')
    axes[1, 1].axvline(0.01, color='darkred', linestyle='--', lw=2, label='1% survival')
    axes[1, 1].set_xlabel('Survival Fraction')
    axes[1, 1].set_ylabel('Number of Cells')
    axes[1, 1].set_title('D) Survival Distribution')
    axes[1, 1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path / 'Figure2_Dose_Response.png', dpi=300)
    plt.close()
    print("  Saved: Figure2_Dose_Response.png")
    
    # Figure 3: DNA Damage
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SSB vs DSB
    scatter = axes[0, 0].scatter(dsbs, ssbs, c=doses, cmap='hot', alpha=0.6, s=10)
    axes[0, 0].plot([0, dsbs.max()], [0, dsbs.max() * params['ssb_dsb_ratio']], 
                    'k--', lw=2, label=f'{params["ssb_dsb_ratio"]}:1 ratio')
    axes[0, 0].set_xlabel('Number of DSBs')
    axes[0, 0].set_ylabel('Number of SSBs')
    axes[0, 0].set_title('A) SSB vs DSB per Cell')
    axes[0, 0].legend(frameon=False)
    plt.colorbar(scatter, ax=axes[0, 0], label='Dose (Gy)')
    
    # Complex DSB fraction histogram
    complex_dsbs = np.array([c.n_complex_DSB for c in cell_analyses])
    complex_frac = complex_dsbs / np.maximum(dsbs, 1)
    axes[0, 1].hist(complex_frac, bins=30, color=COLORS['purple'], alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0.20, color='blue', linestyle='--', lw=2, label='γ-rays (~20%)')
    axes[0, 1].axvline(0.70, color='red', linestyle='--', lw=2, label='α-particles (~70%)')
    axes[0, 1].set_xlabel('Complex DSB Fraction')
    axes[0, 1].set_ylabel('Number of Cells')
    axes[0, 1].set_title('B) Complex Damage Distribution')
    axes[0, 1].legend(frameon=False)
    
    # DSB distribution
    axes[1, 0].hist(dsbs, bins=50, color=COLORS['purple'], alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(dsbs), color='red', linestyle='-', lw=2, label=f'Mean: {np.mean(dsbs):.0f}')
    axes[1, 0].set_xlabel('Number of DSBs per Cell')
    axes[1, 0].set_ylabel('Number of Cells')
    axes[1, 0].set_title('C) DSB Distribution')
    axes[1, 0].legend(frameon=False)
    
    # Lethal dose fraction
    lethal_fracs = np.array([c.lethal_dose_fraction for c in cell_analyses])
    axes[1, 1].hist(lethal_fracs, bins=50, color=COLORS['warning'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(1.0, color='darkred', linestyle='--', lw=2, label='D₃₇ (lethal dose)')
    axes[1, 1].set_xlabel('Fraction of Lethal Dose (D/D₃₇)')
    axes[1, 1].set_ylabel('Number of Cells')
    axes[1, 1].set_title('D) Lethal Dose Fraction')
    axes[1, 1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path / 'Figure3_DNA_Damage.png', dpi=300)
    plt.close()
    print("  Saved: Figure3_DNA_Damage.png")
    
    # Figure 4: Spatial Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spatial dose
    scatter1 = axes[0, 0].scatter(cx, cy, c=doses, cmap='hot', s=10, alpha=0.7)
    axes[0, 0].set_xlabel('X (μm)')
    axes[0, 0].set_ylabel('Y (μm)')
    axes[0, 0].set_title('A) Spatial Dose Distribution')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Dose (Gy)')
    
    # Spatial survival
    scatter2 = axes[0, 1].scatter(cx, cy, c=survivals, cmap='RdYlGn', s=10, alpha=0.7)
    axes[0, 1].set_xlabel('X (μm)')
    axes[0, 1].set_ylabel('Y (μm)')
    axes[0, 1].set_title('B) Spatial Survival Distribution')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Survival')
    
    # Cell size vs dose
    scatter3 = axes[1, 0].scatter(volumes, doses, c=survivals, cmap='RdYlGn', s=10, alpha=0.7)
    axes[1, 0].set_xlabel('Cell Volume (μm³)')
    axes[1, 0].set_ylabel('Mean Dose (Gy)')
    axes[1, 0].set_title('C) Cell Size vs Dose')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Survival')
    
    # Dose histogram
    axes[1, 1].hist(doses, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(doses), color='red', linestyle='-', lw=2, label=f'Mean: {np.mean(doses):.2f} Gy')
    axes[1, 1].axvline(np.median(doses), color='orange', linestyle='--', lw=2, label=f'Median: {np.median(doses):.2f} Gy')
    axes[1, 1].set_xlabel('Mean Dose per Cell (Gy)')
    axes[1, 1].set_ylabel('Number of Cells')
    axes[1, 1].set_title('D) Dose Distribution')
    axes[1, 1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path / 'Figure4_Spatial_Analysis.png', dpi=300)
    plt.close()
    print("  Saved: Figure4_Spatial_Analysis.png")
    
    # Figure 5: Summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Pie chart of survival categories
    viable = np.sum(survivals > 0.5)
    damaged = np.sum((survivals > 0.1) & (survivals <= 0.5))
    dying = np.sum((survivals > 0.01) & (survivals <= 0.1))
    dead = np.sum(survivals <= 0.01)
    
    sizes = [viable, damaged, dying, dead]
    labels = [f'Viable (>50%)\n{viable} cells', f'Damaged (10-50%)\n{damaged} cells',
              f'Dying (1-10%)\n{dying} cells', f'Dead (<1%)\n{dead} cells']
    colors = ['#43A047', '#FB8C00', '#E53935', '#8B0000']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('A) Cell Survival Distribution')
    
    # Key metrics bar chart
    metrics = ['Mean Dose\n(Gy)', 'Mean DSB\n(per cell)', 'Mean Survival\n(fraction)', 
               'Dose-Survival\nCorrelation']
    values = [np.mean(doses), np.mean(dsbs), np.mean(survivals), 
              abs(np.corrcoef(doses, survivals)[0, 1]) if len(doses) > 1 else 0]
    
    bars = axes[0, 1].bar(metrics, values, color=[COLORS['primary'], COLORS['secondary'], 
                                                   COLORS['accent'], COLORS['purple']])
    axes[0, 1].set_title('B) Key Population Metrics')
    for bar, val in zip(bars, values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10)
    
    # Summary statistics text
    axes[1, 0].axis('off')
    summary_text = f"""
    POPULATION SUMMARY STATISTICS
    {'='*40}
    
    Total Cells Analyzed:   {len(cell_analyses):,}
    Total Volume:           {sum(volumes):,.0f} μm³
    
    DOSIMETRY
    {'-'*40}
    Mean Dose:              {np.mean(doses):.4f} ± {np.std(doses):.4f} Gy
    
    DNA DAMAGE
    {'-'*40}
    Total DSBs:             {sum(dsbs):,}
    Mean DSBs per Cell:     {np.mean(dsbs):.1f}
    
    CELL SURVIVAL
    {'-'*40}
    Mean Survival:          {np.mean(survivals):.4f} ({np.mean(survivals)*100:.2f}%)
    Cells < 50% survival:   {np.sum(survivals < 0.5)} ({100*np.sum(survivals < 0.5)/len(survivals):.1f}%)
    Cells < 10% survival:   {np.sum(survivals < 0.1)} ({100*np.sum(survivals < 0.1)/len(survivals):.1f}%)
    Cells < 1% survival:    {np.sum(survivals < 0.01)} ({100*np.sum(survivals < 0.01)/len(survivals):.1f}%)
    
    CORRELATIONS
    {'-'*40}
    Dose-Survival:          r = {np.corrcoef(doses, survivals)[0,1] if len(doses)>1 else 0:.3f}
    Expression-Dose:        r = {np.corrcoef(expressions, doses)[0,1] if len(doses)>1 else 0:.3f}
    """
    axes[1, 0].text(0.1, 0.9, summary_text, transform=axes[1, 0].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 0].set_title('C) Summary Statistics')
    
    # Dose distribution
    axes[1, 1].hist(doses, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(doses), color='red', linestyle='-', lw=2, label=f'Mean: {np.mean(doses):.3f} Gy')
    axes[1, 1].axvline(np.median(doses), color='orange', linestyle='--', lw=2, label=f'Median: {np.median(doses):.3f} Gy')
    axes[1, 1].set_xlabel('Mean Dose per Cell (Gy)')
    axes[1, 1].set_ylabel('Number of Cells')
    axes[1, 1].set_title('D) Dose Distribution')
    axes[1, 1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(out_path / 'Figure5_Summary.png', dpi=300)
    plt.close()
    print("  Saved: Figure5_Summary.png")
    
    # =========================================================================
    # Figure 6: Spatial Heterogeneity Analysis (Core vs Periphery)
    # =========================================================================
    if spatial_results is not None:
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        distances = spatial_results['distances']
        core_mask = spatial_results['masks']['core']
        intermediate_mask = spatial_results['masks']['intermediate']
        periphery_mask = spatial_results['masks']['periphery']
        
        # Panel A: Spatial map colored by region
        ax1 = fig.add_subplot(gs[0, 0])
        # Plot each region with consistent colors
        ax1.scatter(cx[periphery_mask], cy[periphery_mask], c=REGION_COLORS['periphery'], 
                   s=20, alpha=0.7, label='Periphery')
        ax1.scatter(cx[intermediate_mask], cy[intermediate_mask], c=REGION_COLORS['intermediate'], 
                   s=20, alpha=0.7, label='Intermediate')
        ax1.scatter(cx[core_mask], cy[core_mask], c=REGION_COLORS['core'], 
                   s=20, alpha=0.7, label='Core')
        ax1.set_xlabel('X (μm)', fontsize=16)
        ax1.set_ylabel('Y (μm)', fontsize=16)
        ax1.set_title('A) Tumor Spatial Regions', fontsize=18, fontweight='bold')
        ax1.legend(loc='upper right', frameon=False, fontsize=12, markerscale=2)
        
        # Panel B: Radial dose profile with ERROR BANDS (Nature-style like Fig 1g)
        ax2 = fig.add_subplot(gs[0, 1])
        radial = spatial_results['radial_profile']
        
        # Calculate SD for each bin for error bands
        n_bins = len(radial['r'])
        bin_edges = np.linspace(0, distances.max(), n_bins + 1)
        dose_std = []
        for i in range(n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if np.any(mask):
                dose_std.append(np.std(doses[mask]))
            else:
                dose_std.append(0)
        dose_std = np.array(dose_std)
        
        # Plot with shaded SD band
        r_arr = np.array(radial['r'])
        dose_arr = np.array(radial['dose'])
        ax2.fill_between(r_arr, dose_arr - dose_std, dose_arr + dose_std, 
                        alpha=0.3, color=COLORS['primary'], label='±SD')
        ax2.plot(r_arr, dose_arr, '-', color=COLORS['primary'], lw=3, label='Mean')
        ax2.plot(r_arr, dose_arr, 'o', color='white', markersize=6, markeredgecolor=COLORS['primary'], 
                markeredgewidth=2)
        
        # Add region shading
        ax2.axvspan(0, spatial_results['core_threshold'], alpha=0.1, color=REGION_COLORS['core'], label='Core')
        ax2.axvspan(spatial_results['core_threshold'], spatial_results['periphery_threshold'], 
                   alpha=0.1, color=REGION_COLORS['intermediate'])
        ax2.axvspan(spatial_results['periphery_threshold'], r_arr.max(), 
                   alpha=0.1, color=REGION_COLORS['periphery'], label='Periphery')
        
        ax2.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax2.set_ylabel('Mean Dose (Gy)', fontsize=16)
        ax2.set_title('B) Radial Dose Profile', fontsize=18, fontweight='bold')
        ax2.legend(frameon=False, fontsize=11, loc='upper right')
        ax2.tick_params(labelsize=13)
        ax2.set_xlim([0, r_arr.max()])
        
        # Panel C: Radial survival profile with ERROR BANDS
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Calculate SD for survival
        surv_std = []
        for i in range(n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if np.any(mask):
                surv_std.append(np.std(survivals[mask]))
            else:
                surv_std.append(0)
        surv_std = np.array(surv_std)
        surv_arr = np.array(radial['survival'])
        
        # Plot with shaded SD band
        ax3.fill_between(r_arr, surv_arr - surv_std, surv_arr + surv_std, 
                        alpha=0.3, color=COLORS['secondary'], label='±SD')
        ax3.plot(r_arr, surv_arr, '-', color=COLORS['secondary'], lw=3, label='Mean')
        ax3.plot(r_arr, surv_arr, 'o', color='white', markersize=6, markeredgecolor=COLORS['secondary'], 
                markeredgewidth=2)
        
        # Add region shading
        ax3.axvspan(0, spatial_results['core_threshold'], alpha=0.1, color=REGION_COLORS['core'])
        ax3.axvspan(spatial_results['core_threshold'], spatial_results['periphery_threshold'], 
                   alpha=0.1, color=REGION_COLORS['intermediate'])
        ax3.axvspan(spatial_results['periphery_threshold'], r_arr.max(), 
                   alpha=0.1, color=REGION_COLORS['periphery'])
        
        ax3.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax3.set_ylabel('Mean Survival Fraction', fontsize=16)
        ax3.set_title('C) Radial Survival Profile', fontsize=18, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.legend(frameon=False, fontsize=11, loc='lower right')
        ax3.tick_params(labelsize=13)
        ax3.set_xlim([0, r_arr.max()])
        
        # Panel D: VIOLIN PLOT - Dose distribution by region (Nature-style)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Prepare data for violin plot
        dose_data = [doses[core_mask], doses[intermediate_mask], doses[periphery_mask]]
        positions = [1, 2, 3]
        
        # Create violin plot
        parts = ax4.violinplot(dose_data, positions=positions, showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(REGION_COLORS.values())[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Style the lines
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels(['Core', 'Intermediate', 'Periphery'], fontsize=14)
        ax4.set_ylabel('Dose (Gy)', fontsize=16)
        ax4.set_title('D) Dose Distribution by Region', fontsize=18, fontweight='bold')
        ax4.tick_params(labelsize=13)
        
        # Add mean values as text
        for i, (pos, data) in enumerate(zip(positions, dose_data)):
            ax4.text(pos, np.max(data) * 1.05, f'μ={np.mean(data):.4f}', 
                    ha='center', fontsize=11, fontweight='bold')
        
        # Panel E: VIOLIN PLOT - Survival distribution (Nature-style)
        ax5 = fig.add_subplot(gs[1, 1])
        surv_data = [survivals[core_mask], survivals[intermediate_mask], survivals[periphery_mask]]
        
        parts = ax5.violinplot(surv_data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(REGION_COLORS.values())[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        
        ax5.set_xticks(positions)
        ax5.set_xticklabels(['Core', 'Intermediate', 'Periphery'], fontsize=14)
        ax5.set_ylabel('Survival Fraction', fontsize=16)
        ax5.set_title('E) Survival Distribution by Region', fontsize=18, fontweight='bold')
        ax5.set_ylim([0, 1.1])
        ax5.tick_params(labelsize=13)
        
        for i, (pos, data) in enumerate(zip(positions, surv_data)):
            ax5.text(pos, 1.02, f'μ={np.mean(data):.3f}', ha='center', fontsize=11, fontweight='bold')
        
        # Panel F: VIOLIN PLOT - DSB distribution (Nature-style)
        ax6 = fig.add_subplot(gs[1, 2])
        dsb_data = [dsbs[core_mask], dsbs[intermediate_mask], dsbs[periphery_mask]]
        
        parts = ax6.violinplot(dsb_data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(REGION_COLORS.values())[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        
        ax6.set_xticks(positions)
        ax6.set_xticklabels(['Core', 'Intermediate', 'Periphery'], fontsize=14)
        ax6.set_ylabel('DSBs per Cell', fontsize=16)
        ax6.set_title('F) DNA Damage Distribution by Region', fontsize=18, fontweight='bold')
        ax6.tick_params(labelsize=13)
        
        for i, (pos, data) in enumerate(zip(positions, dsb_data)):
            ax6.text(pos, np.max(data) * 1.05, f'μ={np.mean(data):.1f}', 
                    ha='center', fontsize=11, fontweight='bold')
        
        # Panel G: Dose vs Distance scatter with region colors
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(distances[periphery_mask], doses[periphery_mask], c=REGION_COLORS['periphery'], 
                   s=25, alpha=0.6, label='Periphery')
        ax7.scatter(distances[intermediate_mask], doses[intermediate_mask], c=REGION_COLORS['intermediate'], 
                   s=25, alpha=0.6, label='Intermediate')
        ax7.scatter(distances[core_mask], doses[core_mask], c=REGION_COLORS['core'], 
                   s=25, alpha=0.6, label='Core')
        ax7.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax7.set_ylabel('Mean Dose (Gy)', fontsize=16)
        ax7.set_title('G) Dose vs Radial Distance', fontsize=18, fontweight='bold')
        ax7.legend(frameon=False, fontsize=12, markerscale=1.5)
        ax7.tick_params(labelsize=13)
        
        # Panel H: Survival vs Distance with region colors
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(distances[periphery_mask], survivals[periphery_mask], c=REGION_COLORS['periphery'], 
                   s=25, alpha=0.6, label='Periphery')
        ax8.scatter(distances[intermediate_mask], survivals[intermediate_mask], c=REGION_COLORS['intermediate'], 
                   s=25, alpha=0.6, label='Intermediate')
        ax8.scatter(distances[core_mask], survivals[core_mask], c=REGION_COLORS['core'], 
                   s=25, alpha=0.6, label='Core')
        ax8.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax8.set_ylabel('Survival Fraction', fontsize=16)
        ax8.set_title('H) Survival vs Radial Distance', fontsize=18, fontweight='bold')
        ax8.legend(frameon=False, fontsize=12, markerscale=1.5)
        ax8.tick_params(labelsize=13)
        
        # Panel I: Heterogeneity summary with better formatting
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        het = spatial_results['heterogeneity']
        
        # Get region doses for comparison
        core_dose = spatial_results['regions']['core']['mean_dose']
        peri_dose = spatial_results['regions']['periphery']['mean_dose']
        
        summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     SPATIAL HETEROGENEITY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TUMOR GEOMETRY
  ● Centroid: ({spatial_results['tumor_centroid'][0]:.0f}, {spatial_results['tumor_centroid'][1]:.0f}, {spatial_results['tumor_centroid'][2]:.0f}) μm
  ● Core boundary: {spatial_results['core_threshold']:.1f} μm
  ● Periphery boundary: {spatial_results['periphery_threshold']:.1f} μm

CELL DISTRIBUTION
  ● Core (red): {spatial_results['regions']['core']['n_cells']} cells
  ● Intermediate (orange): {spatial_results['regions']['intermediate']['n_cells']} cells  
  ● Periphery (green): {spatial_results['regions']['periphery']['n_cells']} cells

HETEROGENEITY INDICES
  ● Dose CV: {het['dose_cv']*100:.1f}%
  ● Dose Gini: {het['dose_gini']:.3f}
  ● Survival CV: {het['survival_cv']*100:.1f}%
  ● DSB CV: {het['dsb_cv']*100:.1f}%

CLINICAL INTERPRETATION
  {'High dose heterogeneity detected' if het['dose_cv'] > 0.3 else 'Uniform dose distribution'}
  {'Core receives higher dose - good targeting' if core_dose > peri_dose else 'Periphery higher dose - crossfire dominant'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9))
        ax9.set_title('I) Summary & Clinical Interpretation', fontsize=18, fontweight='bold')
        
        plt.suptitle(f'Spatial Heterogeneity Analysis - {isotope}\nTumor Core vs Periphery Dosimetry', 
                    fontsize=20, fontweight='bold', y=0.99)
        plt.savefig(out_path / 'Figure6_Spatial_Heterogeneity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: Figure6_Spatial_Heterogeneity.png")
        
        # =====================================================================
        # Figure 7: Dose-Volume Histogram and Advanced Dosimetry
        # =====================================================================
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Panel A: Cumulative DVH
        sorted_doses = np.sort(doses)[::-1]
        volume_frac = np.arange(1, len(sorted_doses) + 1) / len(sorted_doses) * 100
        axes[0, 0].plot(sorted_doses, volume_frac, 'b-', lw=2)
        axes[0, 0].fill_betweenx(volume_frac, 0, sorted_doses, alpha=0.3)
        axes[0, 0].set_xlabel('Dose (Gy)', fontsize=12)
        axes[0, 0].set_ylabel('Volume (%)', fontsize=12)
        axes[0, 0].set_title('A) Dose-Volume Histogram (DVH)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlim([0, doses.max() * 1.1])
        axes[0, 0].set_ylim([0, 105])
        # Add D90, D50, D10 lines
        D90 = np.percentile(doses, 10)
        D50 = np.percentile(doses, 50)
        D10 = np.percentile(doses, 90)
        axes[0, 0].axvline(D90, color='red', linestyle='--', label=f'D90={D90:.3f} Gy')
        axes[0, 0].axvline(D50, color='orange', linestyle='--', label=f'D50={D50:.3f} Gy')
        axes[0, 0].axvline(D10, color='green', linestyle='--', label=f'D10={D10:.3f} Gy')
        axes[0, 0].legend(frameon=False, fontsize=9)
        
        # Panel B: Differential DVH
        axes[0, 1].hist(doses, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black', density=True)
        axes[0, 1].set_xlabel('Dose (Gy)', fontsize=12)
        axes[0, 1].set_ylabel('Probability Density', fontsize=12)
        axes[0, 1].set_title('B) Differential DVH', fontsize=14, fontweight='bold')
        
        # Panel C: Log dose distribution
        log_doses = np.log10(doses + 1e-10)
        axes[0, 2].hist(log_doses, bins=50, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('log₁₀(Dose [Gy])', fontsize=12)
        axes[0, 2].set_ylabel('Number of Cells', fontsize=12)
        axes[0, 2].set_title('C) Log Dose Distribution', fontsize=14, fontweight='bold')
        
        # Panel D: Dose vs Expression (targeting efficiency)
        axes[1, 0].hexbin(expressions, doses, gridsize=30, cmap='YlOrRd', mincnt=1)
        axes[1, 0].set_xlabel('Protein Expression (a.u.)', fontsize=12)
        axes[1, 0].set_ylabel('Mean Dose (Gy)', fontsize=12)
        axes[1, 0].set_title('D) Targeting Efficiency', fontsize=14, fontweight='bold')
        r_expr_dose = np.corrcoef(expressions, doses)[0, 1] if len(expressions) > 1 else 0
        axes[1, 0].text(0.05, 0.95, f'r = {r_expr_dose:.3f}', transform=axes[1, 0].transAxes,
                       fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
        
        # Panel E: DSB-Survival relationship
        axes[1, 1].scatter(dsbs, survivals, c=doses, cmap='hot', s=15, alpha=0.6)
        axes[1, 1].set_xlabel('Number of DSBs', fontsize=12)
        axes[1, 1].set_ylabel('Survival Fraction', fontsize=12)
        axes[1, 1].set_title('E) DSB-Survival Relationship', fontsize=14, fontweight='bold')
        # Add theoretical curve
        dsb_range = np.linspace(0, dsbs.max(), 100)
        # Each unrepaired DSB has ~3% lethality (Rothkamm & Löbrich 2003)
        surv_theory = np.exp(-0.03 * dsb_range)
        axes[1, 1].plot(dsb_range, surv_theory, 'r--', lw=2, label='Theoretical (~3%/DSB)')
        axes[1, 1].legend(frameon=False)
        
        # Panel F: Cell volume effect
        volumes = np.array([c.volume_um3 for c in cell_analyses])
        axes[1, 2].scatter(volumes, doses / np.mean(doses), c=survivals, cmap='RdYlGn', s=15, alpha=0.6)
        axes[1, 2].set_xlabel('Cell Volume (μm³)', fontsize=12)
        axes[1, 2].set_ylabel('Relative Dose', fontsize=12)
        axes[1, 2].set_title('F) Volume-Dose Relationship', fontsize=14, fontweight='bold')
        axes[1, 2].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Advanced Dosimetry Analysis - {isotope}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path / 'Figure7_Advanced_Dosimetry.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: Figure7_Advanced_Dosimetry.png")
    
    # =========================================================================
    # Figure 8: ADVANCED NATURE-STYLE Multi-Parameter Visualization
    # =========================================================================
    if spatial_results is not None:
        from scipy import stats
        from scipy.ndimage import gaussian_filter
        
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        distances = spatial_results['distances']
        core_mask = spatial_results['masks']['core']
        intermediate_mask = spatial_results['masks']['intermediate']
        periphery_mask = spatial_results['masks']['periphery']
        
        # Panel A: 2D Density hexbin - Dose vs Expression
        ax1 = fig.add_subplot(gs[0, 0])
        hb = ax1.hexbin(expressions, doses, gridsize=30, cmap='YlOrRd', mincnt=1)
        ax1.set_xlabel('Target Expression (a.u.)', fontsize=16)
        ax1.set_ylabel('Absorbed Dose (Gy)', fontsize=16)
        ax1.set_title('A) Dose-Expression Density Map', fontsize=18, fontweight='bold')
        cb = plt.colorbar(hb, ax=ax1, label='Cell Count')
        cb.ax.tick_params(labelsize=12)
        
        # Add regression line
        slope, intercept, r, p, se = stats.linregress(expressions, doses)
        x_line = np.linspace(expressions.min(), expressions.max(), 100)
        ax1.plot(x_line, slope * x_line + intercept, 'k--', lw=2, 
                label=f'R²={r**2:.3f}, p={p:.2e}')
        ax1.legend(frameon=False, fontsize=12, loc='upper left')
        ax1.tick_params(labelsize=13)
        
        # Panel B: 2D Contour - Survival landscape
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create 2D histogram for contours
        H, xedges, yedges = np.histogram2d(doses, survivals, bins=25)
        H = gaussian_filter(H.T, sigma=1.5)  # Smooth
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        
        contour = ax2.contourf(X, Y, H, levels=15, cmap='viridis')
        ax2.contour(X, Y, H, levels=8, colors='white', linewidths=0.5, alpha=0.5)
        ax2.set_xlabel('Absorbed Dose (Gy)', fontsize=16)
        ax2.set_ylabel('Survival Fraction', fontsize=16)
        ax2.set_title('B) Survival Landscape Contour', fontsize=18, fontweight='bold')
        cb = plt.colorbar(contour, ax=ax2, label='Cell Density')
        cb.ax.tick_params(labelsize=12)
        ax2.tick_params(labelsize=13)
        
        # Panel C: Stacked area - Cumulative cell distribution
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Sort by distance and calculate cumulative
        sort_idx = np.argsort(distances)
        sorted_dist = distances[sort_idx]
        cumul_core = np.cumsum(core_mask[sort_idx]) / len(doses) * 100
        cumul_inter = np.cumsum(intermediate_mask[sort_idx]) / len(doses) * 100
        cumul_peri = np.cumsum(periphery_mask[sort_idx]) / len(doses) * 100
        
        ax3.fill_between(sorted_dist, 0, cumul_core, 
                        color=REGION_COLORS['core'], alpha=0.8, label='Core')
        ax3.fill_between(sorted_dist, cumul_core, cumul_core + cumul_inter, 
                        color=REGION_COLORS['intermediate'], alpha=0.8, label='Intermediate')
        ax3.fill_between(sorted_dist, cumul_core + cumul_inter, 
                        cumul_core + cumul_inter + cumul_peri,
                        color=REGION_COLORS['periphery'], alpha=0.8, label='Periphery')
        
        ax3.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax3.set_ylabel('Cumulative Cell Distribution (%)', fontsize=16)
        ax3.set_title('C) Spatial Cell Distribution', fontsize=18, fontweight='bold')
        ax3.legend(frameon=False, fontsize=12, loc='center right')
        ax3.set_ylim([0, 100])
        ax3.tick_params(labelsize=13)
        
        # Panel D: Radar/Spider chart - Multi-parameter comparison
        ax4 = fig.add_subplot(gs[1, 0], projection='polar')
        
        categories = ['Dose', 'Survival', 'DSB', 'Expression', 'Cell Count']
        n_cats = len(categories)
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Normalize data for each region
        def normalize_val(arr, arr_all):
            return arr / np.max(arr_all) if np.max(arr_all) > 0 else 0
        
        for region, mask, color in [('Core', core_mask, REGION_COLORS['core']),
                                     ('Intermediate', intermediate_mask, REGION_COLORS['intermediate']),
                                     ('Periphery', periphery_mask, REGION_COLORS['periphery'])]:
            values = [
                normalize_val(np.mean(doses[mask]), doses),
                normalize_val(np.mean(survivals[mask]), survivals),
                normalize_val(np.mean(dsbs[mask]), dsbs),
                normalize_val(np.mean(expressions[mask]), expressions),
                normalize_val(np.sum(mask), len(doses))
            ]
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=region, color=color)
            ax4.fill(angles, values, alpha=0.25, color=color)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=12)
        ax4.set_title('D) Regional Characteristic Profiles', fontsize=18, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=False, fontsize=12)
        
        # Panel E: Box + Strip plot - DSB by region (Nature style)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Create data for boxplot
        dsb_data = [dsbs[core_mask], dsbs[intermediate_mask], dsbs[periphery_mask]]
        positions = [1, 2, 3]
        
        # Boxplot with custom style
        bp = ax5.boxplot(dsb_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, showmeans=True, 
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8))
        
        for patch, color in zip(bp['boxes'], REGION_COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add strip plot (jittered points)
        for i, (data, color) in enumerate(zip(dsb_data, REGION_COLORS.values())):
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax5.scatter(positions[i] + jitter, data, c=color, alpha=0.3, s=15, edgecolors='none')
        
        ax5.set_xticks(positions)
        ax5.set_xticklabels(['Core', 'Intermediate', 'Periphery'], fontsize=14)
        ax5.set_ylabel('DSBs per Cell', fontsize=16)
        ax5.set_title('E) DNA Damage Distribution', fontsize=18, fontweight='bold')
        ax5.tick_params(labelsize=13)
        
        # Panel F: Correlation heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Calculate correlations
        variables = ['Dose', 'Survival', 'DSB', 'Expression', 'Distance']
        data_matrix = np.column_stack([doses, survivals, dsbs, expressions, distances])
        corr_matrix = np.corrcoef(data_matrix.T)
        
        im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add correlation values
        for i in range(len(variables)):
            for j in range(len(variables)):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax6.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color=color)
        
        ax6.set_xticks(range(len(variables)))
        ax6.set_yticks(range(len(variables)))
        ax6.set_xticklabels(variables, fontsize=12, rotation=45, ha='right')
        ax6.set_yticklabels(variables, fontsize=12)
        ax6.set_title('F) Parameter Correlation Matrix', fontsize=18, fontweight='bold')
        cb = plt.colorbar(im, ax=ax6, label='Pearson r')
        cb.ax.tick_params(labelsize=12)
        
        # Panel G: Multi-line trend with confidence - Expression profiles
        ax7 = fig.add_subplot(gs[2, 0])
        
        # Bin by distance and calculate mean ± SD
        n_bins = 15
        bin_edges = np.linspace(0, distances.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for name, var, color in [('Dose', doses, COLORS['primary']), 
                                  ('Survival', survivals, COLORS['secondary']),
                                  ('DSB (scaled)', dsbs / dsbs.max(), COLORS['accent'])]:
            means = []
            stds = []
            for i in range(n_bins):
                mask_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
                if np.any(mask_bin):
                    means.append(np.mean(var[mask_bin]))
                    stds.append(np.std(var[mask_bin]))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            means = np.array(means)
            stds = np.array(stds)
            
            ax7.fill_between(bin_centers, means - stds, means + stds, alpha=0.2, color=color)
            ax7.plot(bin_centers, means, '-', lw=2.5, color=color, label=name)
            ax7.plot(bin_centers, means, 'o', markersize=5, color='white', 
                    markeredgecolor=color, markeredgewidth=1.5)
        
        ax7.set_xlabel('Distance from Tumor Center (μm)', fontsize=16)
        ax7.set_ylabel('Normalized Value', fontsize=16)
        ax7.set_title('G) Multi-Parameter Radial Profiles', fontsize=18, fontweight='bold')
        ax7.legend(frameon=False, fontsize=12)
        ax7.tick_params(labelsize=13)
        
        # Panel H: 2D scatter with marginal histograms concept
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Scatter colored by survival
        scatter = ax8.scatter(doses, dsbs, c=survivals, cmap='RdYlGn', s=30, alpha=0.6,
                             edgecolors='none')
        
        # Add trend line
        z = np.polyfit(doses, dsbs, 2)  # Quadratic fit
        p = np.poly1d(z)
        x_fit = np.linspace(doses.min(), doses.max(), 100)
        ax8.plot(x_fit, p(x_fit), 'k--', lw=2, label='Quadratic fit')
        
        ax8.set_xlabel('Absorbed Dose (Gy)', fontsize=16)
        ax8.set_ylabel('DSBs per Cell', fontsize=16)
        ax8.set_title('H) Dose-Damage Relationship', fontsize=18, fontweight='bold')
        cb = plt.colorbar(scatter, ax=ax8, label='Survival')
        cb.ax.tick_params(labelsize=12)
        ax8.legend(frameon=False, fontsize=12)
        ax8.tick_params(labelsize=13)
        
        # Panel I: KDE density - Survival distribution by region
        ax9 = fig.add_subplot(gs[2, 2])
        
        from scipy.stats import gaussian_kde
        
        for region, mask, color in [('Core', core_mask, REGION_COLORS['core']),
                                     ('Intermediate', intermediate_mask, REGION_COLORS['intermediate']),
                                     ('Periphery', periphery_mask, REGION_COLORS['periphery'])]:
            data = survivals[mask]
            if len(data) > 10:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), min(data.max(), 1.0), 200)
                ax9.fill_between(x_range, kde(x_range), alpha=0.4, color=color)
                ax9.plot(x_range, kde(x_range), lw=2.5, color=color, label=f'{region} (n={len(data)})')
        
        ax9.set_xlabel('Survival Fraction', fontsize=16)
        ax9.set_ylabel('Probability Density', fontsize=16)
        ax9.set_title('I) Survival Distribution by Region (KDE)', fontsize=18, fontweight='bold')
        ax9.legend(frameon=False, fontsize=12)
        ax9.tick_params(labelsize=13)
        ax9.set_xlim([0, 1.05])
        
        plt.suptitle(f'Advanced Multi-Parameter Analysis - {isotope}\nNature Communications Style Visualization', 
                    fontsize=22, fontweight='bold', y=0.99)
        plt.savefig(out_path / 'Figure8_Advanced_Analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: Figure8_Advanced_Analysis.png")


def export_csv_files(cell_analyses: List[CellAnalysis], output_dir: str):
    """Export analysis results to CSV files."""
    print("\nExporting CSV files...")
    out_path = Path(output_dir)
    
    # Complete cell analysis
    import csv
    csv_path = out_path / 'cell_analysis_complete.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if cell_analyses:
            writer = csv.DictWriter(f, fieldnames=asdict(cell_analyses[0]).keys())
            writer.writeheader()
            for cell in cell_analyses:
                writer.writerow(asdict(cell))
    print(f"  Saved: cell_analysis_complete.csv ({len(cell_analyses)} cells)")
    
    # Dose-survival data
    csv_path = out_path / 'dose_survival_data.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id', 'mean_dose_Gy', 'survival_fraction', 'n_DSB'])
        for cell in cell_analyses:
            writer.writerow([cell.cell_id, cell.mean_dose_Gy, cell.survival_fraction, cell.n_DSB])
    print("  Saved: dose_survival_data.csv")
    
    # DNA damage summary
    csv_path = out_path / 'dna_damage_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id', 'n_SSB', 'n_DSB', 'n_complex_DSB', 'ssb_dsb_ratio'])
        for cell in cell_analyses:
            writer.writerow([cell.cell_id, cell.n_SSB, cell.n_DSB, cell.n_complex_DSB, cell.ssb_dsb_ratio])
    print("  Saved: dna_damage_summary.csv")


def export_spatial_analysis_csv(cell_analyses: List[CellAnalysis], spatial_results: Dict, output_dir: str):
    """Export spatial analysis results to CSV files."""
    out_path = Path(output_dir)
    import csv
    
    # Cell spatial data with region assignment
    csv_path = out_path / 'cell_spatial_regions.csv'
    
    core_mask = spatial_results['masks']['core']
    intermediate_mask = spatial_results['masks']['intermediate']
    periphery_mask = spatial_results['masks']['periphery']
    distances = spatial_results['distances']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id', 'centroid_x', 'centroid_y', 'centroid_z', 
                        'distance_from_center', 'region', 'mean_dose_Gy', 
                        'survival_fraction', 'n_DSB', 'mean_expression'])
        
        for i, cell in enumerate(cell_analyses):
            if core_mask[i]:
                region = 'core'
            elif intermediate_mask[i]:
                region = 'intermediate'
            else:
                region = 'periphery'
            
            writer.writerow([
                cell.cell_id, cell.centroid_x, cell.centroid_y, cell.centroid_z,
                distances[i], region, cell.mean_dose_Gy, cell.survival_fraction,
                cell.n_DSB, cell.mean_expression
            ])
    
    print(f"  Saved: cell_spatial_regions.csv")
    
    # Region summary statistics
    csv_path = out_path / 'region_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['region', 'n_cells', 'mean_dose_Gy', 'std_dose_Gy', 
                        'mean_survival', 'mean_dsb', 'mean_expression'])
        
        for region_name in ['core', 'intermediate', 'periphery']:
            r = spatial_results['regions'][region_name]
            writer.writerow([
                region_name, r['n_cells'], r['mean_dose'], r['std_dose'],
                r['mean_survival'], r['mean_dsb'], r['mean_expression']
            ])
    
    print(f"  Saved: region_summary.csv")
    
    # Radial profile data
    csv_path = out_path / 'radial_profile.csv'
    radial = spatial_results['radial_profile']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['distance_um', 'mean_dose_Gy', 'mean_survival', 'mean_dsb'])
        for i in range(len(radial['r'])):
            writer.writerow([radial['r'][i], radial['dose'][i], 
                           radial['survival'][i], radial['dsb'][i]])
    
    print(f"  Saved: radial_profile.csv")
    
    # Heterogeneity indices
    csv_path = out_path / 'heterogeneity_indices.csv'
    het = spatial_results['heterogeneity']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'interpretation'])
        writer.writerow(['dose_cv', het['dose_cv'], 'Coefficient of variation of dose'])
        writer.writerow(['dose_gini', het['dose_gini'], 'Gini coefficient (0=uniform, 1=max heterogeneity)'])
        writer.writerow(['survival_cv', het['survival_cv'], 'Coefficient of variation of survival'])
        writer.writerow(['dsb_cv', het['dsb_cv'], 'Coefficient of variation of DSBs'])
    
    print(f"  Saved: heterogeneity_indices.csv")


def save_analysis_report(cell_analyses: List[CellAnalysis], sim_results: Dict, 
                         isotope: str, output_dir: str, spatial_results: Dict = None):
    """Generate comprehensive analysis report with peer-reviewed references."""
    out_path = Path(output_dir)
    
    doses = np.array([c.mean_dose_Gy for c in cell_analyses])
    survivals = np.array([c.survival_fraction for c in cell_analyses])
    dsbs = np.array([c.n_DSB for c in cell_analyses])
    ssbs = np.array([c.n_SSB for c in cell_analyses])
    complex_dsbs = np.array([c.n_complex_DSB for c in cell_analyses])
    
    params = RADIATION_PARAMS.get(isotope, RADIATION_PARAMS['Lu-177'])
    
    # Calculate D37 (dose for 37% survival)
    if params['beta_LQ'] > 0:
        D37 = (-params['alpha_LQ'] + np.sqrt(params['alpha_LQ']**2 + 4*params['beta_LQ'])) / (2*params['beta_LQ'])
    else:
        D37 = 1.0 / params['alpha_LQ']
    
    report = f"""# CellRad-MC Comprehensive Cellular Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Simulation Overview

### 1.1 Radiopharmaceutical Parameters
| Parameter | Value | Reference |
|-----------|-------|-----------|
| Isotope | {isotope} | ICRP 107 |
| Radiation Type | {params['type'].upper()} | |
| LET | {params['LET_keV_um']:.1f} keV/μm | ICRU Report 90 |
| RBE | {params['RBE']:.1f} | Sgouros et al. 2010 |

### 1.2 Analysis Summary
| Metric | Value |
|--------|-------|
| Total Cells Analyzed | {len(cell_analyses):,} |
| Mean Dose | {np.mean(doses):.4f} ± {np.std(doses):.4f} Gy |
| Max Dose | {np.max(doses):.4f} Gy |
| Dose Range | {np.min(doses):.4f} - {np.max(doses):.4f} Gy |

---

## 2. DNA Damage Analysis

### 2.1 Methodology
DNA damage was calculated using validated models from peer-reviewed literature:
- **DSB yield**: {params['dsb_per_Gy_Gbp']} DSB/Gy/Gbp (Nikjoo et al. 2001)
- **SSB:DSB ratio**: {params['ssb_dsb_ratio']}:1 (Nikjoo et al. 2001)
- **Complex DSB fraction**: {params['complex_frac']*100:.0f}% (Friedland et al. 2003)
- **DNA content**: 6.4 Gbp per diploid cell

### 2.2 Results
| Damage Type | Total | Per Cell (mean) | Per Cell (std) |
|-------------|-------|-----------------|----------------|
| DSB (Double Strand Breaks) | {sum(dsbs):,} | {np.mean(dsbs):.1f} | {np.std(dsbs):.1f} |
| SSB (Single Strand Breaks) | {sum(ssbs):,} | {np.mean(ssbs):.0f} | {np.std(ssbs):.0f} |
| Complex DSB | {sum(complex_dsbs):,} | {np.mean(complex_dsbs):.1f} | {np.std(complex_dsbs):.1f} |

### 2.3 Interpretation
- **DSBs** are the primary lethal DNA lesions. Each unrepaired DSB can cause cell death.
- **SSB:DSB ratio** of {params['ssb_dsb_ratio']}:1 is characteristic of {'high' if params['LET_keV_um'] > 10 else 'low'}-LET radiation.
- **Complex DSBs** ({params['complex_frac']*100:.0f}%) are clustered damage sites that are difficult to repair.

---

## 3. Cell Survival Analysis

### 3.1 Linear-Quadratic Model
Survival calculated using: **S = exp(-αD - βD²)**

| Parameter | Value | Source |
|-----------|-------|--------|
| α (linear) | {params['alpha_LQ']:.3f} Gy⁻¹ | van Leeuwen et al. 2018 |
| β (quadratic) | {params['beta_LQ']:.4f} Gy⁻² | van Leeuwen et al. 2018 |
| α/β ratio | {f"{params['alpha_LQ']/params['beta_LQ']:.1f}" if params['beta_LQ'] > 0 else '∞'} Gy | |
| D₃₇ | {D37:.2f} Gy | Calculated |

{'**Note:** For high-LET alpha particles, β→0 (single-track killing dominates).' if params['type'] == 'alpha' else ''}

### 3.2 Survival Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| Viable (>50% survival) | {np.sum(survivals > 0.5)} | {100*np.sum(survivals > 0.5)/len(survivals):.1f}% |
| Damaged (10-50% survival) | {np.sum((survivals > 0.1) & (survivals <= 0.5))} | {100*np.sum((survivals > 0.1) & (survivals <= 0.5))/len(survivals):.1f}% |
| Dying (1-10% survival) | {np.sum((survivals > 0.01) & (survivals <= 0.1))} | {100*np.sum((survivals > 0.01) & (survivals <= 0.1))/len(survivals):.1f}% |
| Dead (<1% survival) | {np.sum(survivals <= 0.01)} | {100*np.sum(survivals <= 0.01)/len(survivals):.1f}% |

### 3.3 Key Statistics
- **Mean survival**: {np.mean(survivals)*100:.2f}%
- **Median survival**: {np.median(survivals)*100:.2f}%
- **Cells receiving >D₃₇**: {np.sum(doses > D37)} ({100*np.sum(doses > D37)/len(doses):.1f}%)

---

## 4. Radiation Biology Interpretation

### 4.1 {params['type'].upper()} Emitter Characteristics
{'**Alpha particles** (high-LET, ~100 keV/μm):' if params['type'] == 'alpha' else '**Beta particles** (low-LET, ~0.3 keV/μm):' if params['type'] == 'beta' else '**Auger electrons** (variable LET, depends on localization):'}
{'- Cause dense ionization clusters along short tracks (~50-80 μm)' if params['type'] == 'alpha' else '- Cause sparse ionization along longer tracks (~800-3500 μm)' if params['type'] == 'beta' else '- Cause extremely dense ionization in nm range when internalized'}
{'- ~5x more DSBs per Gy compared to gamma/beta radiation' if params['type'] == 'alpha' else '- "Crossfire effect" delivers dose to nearby cells' if params['type'] == 'beta' else '- RBE depends on nuclear localization (1-8x)'}
{'- RBE = {params["RBE"]:.1f} (ICRP 103)' if params['type'] in ['alpha', 'auger'] else '- RBE = 1.0 (reference radiation)'}
{'- Effective even in hypoxic tumors (direct DNA damage dominates)' if params['type'] == 'alpha' else '- Somewhat affected by hypoxia (indirect damage component)'}

### 4.2 Therapeutic Implications
- **{100*np.sum(survivals < 0.01)/len(survivals):.1f}%** of cells predicted to receive lethal dose
- Mean dose of **{np.mean(doses):.3f} Gy** corresponds to **{np.mean(doses)/D37:.1f}x D₃₇**
- {'Alpha therapy is particularly effective for micrometastases' if params['type'] == 'alpha' else 'Beta therapy provides good crossfire for heterogeneous tumors' if params['type'] == 'beta' else 'Auger therapy requires nuclear internalization for efficacy'}

---

## 5. Spatial Heterogeneity Analysis

### 5.0 Rationale and Purpose

**Why analyze tumor core vs periphery?**

Tumor microenvironment heterogeneity is a critical factor in radiopharmaceutical therapy response. 
Based on Hong et al. (2024) methodology, we analyze spatial dose distribution because:

1. **Target expression varies spatially**: PSMA/EGFR expression often differs between tumor core 
   (hypoxic, necrotic) and periphery (well-vascularized, proliferating cells).

2. **Beta particle crossfire effect**: Lu-177 beta particles travel ~800 μm, causing "crossfire" 
   where cells not expressing the target still receive dose from nearby decays.

3. **Therapeutic optimization**: Understanding spatial dose gradients helps optimize 
   treatment planning and predict regions at risk of undertreating.

4. **Resistance prediction**: Peripheral cells in well-oxygenated regions may be more 
   radiosensitive, while hypoxic core cells may require higher doses.

**Color Scheme:**
- 🔴 **Core (Red)**: Inner tumor region (closest to centroid, typically hypoxic)
- 🟠 **Intermediate (Orange)**: Transition zone
- 🟢 **Periphery (Green)**: Outer tumor region (well-vascularized, proliferating)

{'### 5.1 Tumor Geometry' + chr(10) + f'''
| Parameter | Value | Clinical Significance |
|-----------|-------|----------------------|
| Tumor Centroid | ({spatial_results['tumor_centroid'][0]:.0f}, {spatial_results['tumor_centroid'][1]:.0f}, {spatial_results['tumor_centroid'][2]:.0f}) μm | Reference point for radial analysis |
| Core Boundary | {spatial_results['core_threshold']:.1f} μm | Inner 33% of cells |
| Periphery Boundary | {spatial_results['periphery_threshold']:.1f} μm | Outer 33% of cells |

### 5.2 Regional Dosimetry Comparison
| Region | N Cells | Mean Dose (Gy) | Std Dev | Mean Survival | Mean DSBs |
|--------|---------|----------------|---------|---------------|-----------|
| 🔴 Core | {spatial_results['regions']['core']['n_cells']} | {spatial_results['regions']['core']['mean_dose']:.4f} | {spatial_results['regions']['core']['std_dose']:.4f} | {spatial_results['regions']['core']['mean_survival']:.4f} | {spatial_results['regions']['core']['mean_dsb']:.1f} |
| 🟠 Intermediate | {spatial_results['regions']['intermediate']['n_cells']} | {spatial_results['regions']['intermediate']['mean_dose']:.4f} | {spatial_results['regions']['intermediate']['std_dose']:.4f} | {spatial_results['regions']['intermediate']['mean_survival']:.4f} | {spatial_results['regions']['intermediate']['mean_dsb']:.1f} |
| 🟢 Periphery | {spatial_results['regions']['periphery']['n_cells']} | {spatial_results['regions']['periphery']['mean_dose']:.4f} | {spatial_results['regions']['periphery']['std_dose']:.4f} | {spatial_results['regions']['periphery']['mean_survival']:.4f} | {spatial_results['regions']['periphery']['mean_dsb']:.1f} |

### 5.3 Heterogeneity Indices

**Purpose:** Quantify dose non-uniformity to assess treatment quality.

| Index | Value | Interpretation | Clinical Impact |
|-------|-------|----------------|-----------------|
| Dose CV | {spatial_results['heterogeneity']['dose_cv']*100:.1f}% | {'High' if spatial_results['heterogeneity']['dose_cv'] > 0.3 else 'Moderate' if spatial_results['heterogeneity']['dose_cv'] > 0.15 else 'Low'} heterogeneity | {'May lead to treatment failure in underdosed regions' if spatial_results['heterogeneity']['dose_cv'] > 0.3 else 'Acceptable for beta-emitters'} |
| Dose Gini | {spatial_results['heterogeneity']['dose_gini']:.3f} | 0=uniform, 1=max inequality | Higher values indicate dose concentration |
| Survival CV | {spatial_results['heterogeneity']['survival_cv']*100:.1f}% | Variation in cell kill | Predicts response heterogeneity |

### 5.4 Clinical Interpretation

**Dose Distribution Pattern:**
- {"✅ **Core receives higher dose** - This indicates effective targeting where the radiopharmaceutical concentrates at the tumor center. This is typical for well-vascularized tumors with high target expression in the core." if spatial_results['regions']['core']['mean_dose'] > spatial_results['regions']['periphery']['mean_dose'] else "⚠️ **Periphery receives higher dose** - This suggests the crossfire effect dominates, where beta particles from peripheral decays deposit energy throughout. This is common for Lu-177 with its ~800 μm range."}

**Therapeutic Coverage:**
- {"✅ **Uniform dose coverage** - The low dose CV ({spatial_results['heterogeneity']['dose_cv']*100:.1f}%) indicates homogeneous irradiation, which is favorable for tumor control." if spatial_results['heterogeneity']['dose_cv'] < 0.2 else "⚠️ **Heterogeneous dose coverage** - The CV of {spatial_results['heterogeneity']['dose_cv']*100:.1f}% suggests dose hot/cold spots. Consider combination therapy or fractionation."}

**Survival Prediction:**
- Core survival: {spatial_results['regions']['core']['mean_survival']*100:.1f}%
- Periphery survival: {spatial_results['regions']['periphery']['mean_survival']*100:.1f}%
- {"Core cells more likely to survive (potential resistance)" if spatial_results['regions']['core']['mean_survival'] > spatial_results['regions']['periphery']['mean_survival'] else "Periphery cells more likely to survive"}
''' if spatial_results else '*(Spatial analysis not performed)*'}

---

## 6. References

1. Hong J, et al. (2024) Deciphering the effects of radiopharmaceutical therapy in the tumor microenvironment. **Theranostics** 14(18):7122-7139.

2. Nikjoo H, et al. (2001) Computational modelling of low-energy electron-induced DNA damage. **Radiat Res** 156:577-583.

3. Rothkamm K & Löbrich M (2003) Evidence for a lack of DNA double-strand break repair in human cells exposed to very low x-ray doses. **PNAS** 100:5057-5062.

4. van Leeuwen CM, et al. (2018) The alfa and beta of tumours: a review of parameters of the linear-quadratic model. **Radiat Oncol** 13:96.

5. Sgouros G, et al. (2010) MIRD Pamphlet No. 22: Radiobiology and dosimetry of alpha-particle emitters. **J Nucl Med** 51:311-328.

6. Friedland W, et al. (2003) Monte Carlo simulation of the production of short DNA fragments. **Radiat Res** 159:401-410.

7. Kassis AI (2004) The amazing world of Auger electrons. **J Nucl Med** 45:1342-1349.

8. ICRU Report 90 (2016) Key Data for Ionizing-Radiation Dosimetry.

9. ICRP Publication 107 (2008) Nuclear Decay Data for Dosimetric Calculations.

---

*Report generated by CellRad-MC - GPU-Accelerated Monte Carlo for Targeted Radionuclide Therapy*
"""
    
    report_path = out_path / 'ANALYSIS_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved: ANALYSIS_REPORT.md")


def check_existing_simulation(output_dir: str, isotope: str) -> bool:
    """Check if simulation results already exist."""
    iso_name = isotope.replace('-', '')
    required_files = [
        f'dose_Gy_{iso_name}.nii.gz',
        f'dose_keV_{iso_name}.nii.gz',
        f'particle_count_{iso_name}.nii.gz'
    ]
    
    for f in required_files:
        if not (Path(output_dir) / f).exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='CellRad-MC Comprehensive Cell Analysis')
    parser.add_argument('--nuclei', required=True, help='Nuclei mask NIfTI file')
    parser.add_argument('--protein', required=True, help='Protein expression NIfTI file')
    parser.add_argument('--output', '-o', default='cell_analysis', help='Output directory')
    parser.add_argument('--isotope', '-i', default='Lu-177', help='Radioisotope')
    parser.add_argument('--n-particles', '-n', type=int, default=10000000, help='Number of particles')
    parser.add_argument('--voxel-size', '-v', type=float, default=1.0, help='Voxel size in microns')
    parser.add_argument('--ct', help='CT scan NIfTI file for material mapping')
    parser.add_argument('--auto-ct', action='store_true', help='Auto-generate uniform soft tissue CT (HU=19)')
    parser.add_argument('--skip-simulation', action='store_true', help='Skip MC simulation, load existing results')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-run simulation even if exists')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CellRad-MC Comprehensive Cellular Analysis")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load nuclei mask
    print("\n--- Loading Data ---")
    nuclei_img = nib.load(args.nuclei)
    nuclei_mask = np.asarray(nuclei_img.dataobj).astype(np.int32)
    n_cells = len(np.unique(nuclei_mask)) - 1  # Exclude background
    print(f"  Nuclei mask: {nuclei_mask.shape}, {n_cells} cells")
    
    # Load expression map
    expr_img = nib.load(args.protein)
    expression_map = np.asarray(expr_img.dataobj).astype(np.float32)
    print(f"  Expression map: {expression_map.shape}")
    
    # Run or load simulation with AUTO-RESUME
    mc_output_dir = str(out_dir / 'mc_simulation')
    
    # Check if simulation already exists
    sim_exists = check_existing_simulation(mc_output_dir, args.isotope)
    
    if sim_exists and not args.force and not args.skip_simulation:
        print("\n" + "=" * 60)
        print("AUTO-RESUME: Found existing simulation results!")
        print("=" * 60)
        print(f"  Location: {mc_output_dir}")
        print("  Skipping simulation, loading existing results...")
        print("  (Use --force to re-run simulation)")
    elif not args.skip_simulation:
        # Run cellrad_mc.py
        run_cellrad_mc(args.protein, mc_output_dir, args.isotope, 
                       args.n_particles, args.voxel_size,
                       ct_path=args.ct, auto_ct=args.auto_ct)
    
    # Load simulation results
    print("\n--- Loading Simulation Results ---")
    sim_results = load_simulation_results(mc_output_dir, args.isotope)
    
    dose_Gy = sim_results.get('dose_Gy', np.zeros_like(nuclei_mask, dtype=np.float32))
    energy_keV = sim_results.get('energy_keV', np.zeros_like(nuclei_mask, dtype=np.float32))
    hits = sim_results.get('hits', np.zeros_like(nuclei_mask, dtype=np.float32))
    
    # Analyze cells
    print("\n--- Analyzing Cells ---")
    cell_analyses = analyze_cells_vectorized(
        nuclei_mask, expression_map, dose_Gy, energy_keV, hits,
        args.voxel_size, args.isotope
    )
    
    # Perform spatial heterogeneity analysis
    spatial_results = analyze_spatial_heterogeneity(cell_analyses, nuclei_mask, args.voxel_size)
    
    # Create figures (including spatial analysis)
    print("\n--- Generating Figures ---")
    create_figures(nuclei_mask, expression_map, dose_Gy, cell_analyses, str(out_dir), 
                   args.isotope, spatial_results)
    
    # Export CSVs
    export_csv_files(cell_analyses, str(out_dir))
    
    # Export spatial analysis CSV
    export_spatial_analysis_csv(cell_analyses, spatial_results, str(out_dir))
    
    # Save comprehensive report with spatial analysis
    save_analysis_report(cell_analyses, sim_results, args.isotope, str(out_dir), spatial_results)
    
    # Summary
    doses = np.array([c.mean_dose_Gy for c in cell_analyses])
    survivals = np.array([c.survival_fraction for c in cell_analyses])
    dsbs = np.array([c.n_DSB for c in cell_analyses])
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {out_dir}")
    print(f"\nKey Results:")
    print(f"  - Cells analyzed: {len(cell_analyses):,}")
    print(f"  - Mean dose: {np.mean(doses):.4f} Gy")
    print(f"  - Mean survival: {np.mean(survivals):.4f} ({np.mean(survivals)*100:.2f}%)")
    print(f"  - Total DSBs: {sum(dsbs):,}")
    print(f"  - Cells predicted dead (<1%): {np.sum(survivals < 0.01)} ({100*np.sum(survivals < 0.01)/len(survivals):.1f}%)")


if __name__ == '__main__':
    main()
