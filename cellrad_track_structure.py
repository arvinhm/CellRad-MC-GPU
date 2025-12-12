#!/usr/bin/env python3
"""
CellRad-MC Track-Structure CLI
==============================

Advanced track-structure Monte Carlo simulation with DNA damage prediction.
Supports DAPI nuclear masks for cell-specific damage tracking.

Usage:
    python cellrad_track_structure.py --protein protein.nii.gz --dapi dapi.nii.gz \
        --isotope Lu-177 --out-dir output --n-particles 10000000

Features:
- Full track-structure physics (event-by-event)
- DNA geometry overlay on DAPI-stained nuclei
- SSB/DSB damage classification
- Per-cell dose and damage reporting
- Cell survival prediction (LQ model)
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

# CellRad-MC modules
from track_structure_mc import TrackStructureMC, DNADamageCalculator
from dna_geometry import DNAGeometryBuilder, estimate_dna_damage_yields
from enhanced_physics import ISOTOPE_DATABASE

# =============================================================================
# ISOTOPE EMISSION DATA
# =============================================================================

def get_isotope_emissions(isotope: str) -> dict:
    """Get emission data for radioisotope"""
    
    if isotope not in ISOTOPE_DATABASE:
        raise ValueError(f"Unknown isotope: {isotope}")
    
    iso = ISOTOPE_DATABASE[isotope]
    
    return {
        'half_life_days': iso.half_life_days,
        'beta_particles': iso.beta_particles,
        'alpha_chain': iso.alpha_chain,
        'auger_electrons': getattr(iso, 'auger_electrons', []),
        'gamma_emissions': getattr(iso, 'gamma_emissions', []),
    }


# =============================================================================
# PARTICLE SAMPLING
# =============================================================================

def sample_beta_energy(E_max: float, shape_factor: float = 3.8) -> float:
    """Sample beta particle energy from Fermi distribution"""
    while True:
        E = np.random.uniform(0, E_max)
        W = E / 511.0 + 1.0  # Total energy in units of m_e c^2
        W_max = E_max / 511.0 + 1.0
        p = np.sqrt(W**2 - 1)
        
        # Fermi function (approximate)
        f = p * (W_max - W)**shape_factor * W
        f_max = (E_max / 3) * ((2 * E_max / 3 / 511.0 + 1) ** 2 - 1) ** 0.5 * (W_max - E_max / 3 / 511.0 - 1) ** shape_factor * (E_max / 3 / 511.0 + 1)
        
        if np.random.random() < f / (f_max * 1.5):
            return E


def sample_isotropic_direction() -> np.ndarray:
    """Sample random isotropic direction"""
    cos_theta = 2.0 * np.random.random() - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.random()
    
    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])


def sample_source_positions(
    activity_map: np.ndarray,
    n_particles: int,
    voxel_size_nm: np.ndarray
) -> np.ndarray:
    """Sample particle source positions from activity distribution"""
    
    # Normalize activity
    activity = activity_map.astype(np.float64)
    activity = np.maximum(activity, 0)
    total = activity.sum()
    
    if total == 0:
        raise ValueError("Activity map is all zeros")
    
    # Flatten for sampling
    flat = activity.flatten()
    flat /= flat.sum()
    
    # Sample voxel indices
    indices = np.random.choice(len(flat), size=n_particles, p=flat)
    
    # Convert to 3D coordinates
    nx, ny, nz = activity_map.shape
    iz = indices // (nx * ny)
    iy = (indices % (nx * ny)) // nx
    ix = indices % nx
    
    # Random position within voxel, in nm
    positions = np.zeros((n_particles, 3), dtype=np.float32)
    positions[:, 0] = (ix + np.random.random(n_particles)) * voxel_size_nm[0]
    positions[:, 1] = (iy + np.random.random(n_particles)) * voxel_size_nm[1]
    positions[:, 2] = (iz + np.random.random(n_particles)) * voxel_size_nm[2]
    
    return positions


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_track_structure_simulation(args):
    """Run full track-structure simulation with DNA damage"""
    
    print("\n" + "="*70)
    print("CellRad-MC Track-Structure Monte Carlo")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activity map (protein expression)
    print(f"\nLoading activity map: {args.protein}")
    protein_nii = nib.load(args.protein)
    activity_map = protein_nii.get_fdata()
    
    # Get voxel sizes
    voxel_sizes = protein_nii.header.get_zooms()[:3]
    voxel_size_xy = args.voxel_size if args.voxel_size else voxel_sizes[0]
    voxel_size_z = args.voxel_size_z if args.voxel_size_z else (voxel_sizes[2] if len(voxel_sizes) > 2 else voxel_size_xy)
    
    # Convert to nm
    voxel_size_nm = np.array([
        voxel_size_xy * 1000,  # um to nm
        voxel_size_xy * 1000,
        voxel_size_z * 1000
    ])
    
    print(f"  Shape: {activity_map.shape}")
    print(f"  Voxel size: {voxel_size_xy:.3f} x {voxel_size_xy:.3f} x {voxel_size_z:.3f} um")
    print(f"  Voxel size: {voxel_size_nm[0]:.1f} x {voxel_size_nm[1]:.1f} x {voxel_size_nm[2]:.1f} nm")
    
    # Load DAPI nuclear mask if provided
    dapi_mask = None
    dna_geometry = None
    
    if args.dapi:
        print(f"\nLoading DAPI nuclear mask: {args.dapi}")
        dapi_nii = nib.load(args.dapi)
        dapi_mask = dapi_nii.get_fdata().astype(np.uint8)
        
        print(f"  Shape: {dapi_mask.shape}")
        print(f"  Unique labels: {np.unique(dapi_mask)}")
        
        # Build DNA geometry
        print("\nBuilding DNA geometry from nuclear masks...")
        dna_geometry = DNAGeometryBuilder(
            voxel_size_um=voxel_size_xy,
            voxel_size_z_um=voxel_size_z
        )
        n_nuclei = dna_geometry.load_nuclear_mask(dapi_mask)
        
        if n_nuclei > 0:
            dna_density = dna_geometry.build_dna_density_map(activity_map.shape, dapi_mask)
            print(f"  Found {n_nuclei} nuclei with DNA content")
    
    # Get isotope data
    print(f"\nIsotope: {args.isotope}")
    emissions = get_isotope_emissions(args.isotope)
    print(f"  Half-life: {emissions['half_life_days']:.2f} days")
    print(f"  Beta particles: {len(emissions['beta_particles'])}")
    print(f"  Alpha chain: {len(emissions['alpha_chain'])} particles")
    
    # Preprocess activity map
    print("\nPreprocessing activity map...")
    activity = activity_map.astype(np.float64)
    activity = np.maximum(activity, 0)
    
    if args.threshold > 0:
        thresh = np.percentile(activity[activity > 0], args.threshold)
        activity[activity < thresh] = 0
        print(f"  Applied {args.threshold}th percentile threshold")
    
    # Normalize
    if activity.max() > 0:
        activity = activity / activity.max() * 100.0
    
    # Initialize simulation
    print(f"\nInitializing track-structure simulation...")
    print(f"  Grid shape: {activity_map.shape}")
    print(f"  Particles: {args.n_particles:,}")
    
    # Use effective voxel size for track-structure (nanometer scale)
    ts_voxel_nm = min(voxel_size_nm)  # Use smallest dimension
    
    mc = TrackStructureMC(
        grid_shape=activity_map.shape,
        voxel_size_nm=ts_voxel_nm,
        dna_density=dna_density if dna_geometry else None
    )
    
    # Sample source positions
    print("\nSampling source positions...")
    positions = sample_source_positions(activity, args.n_particles, voxel_size_nm)
    
    # Initialize results
    total_energy_dep = np.zeros(activity_map.shape, dtype=np.float32)
    total_hits = np.zeros(activity_map.shape, dtype=np.int32)
    
    start_time = time.time()
    
    # Run beta particle simulation
    if emissions['beta_particles']:
        print("\n--- Simulating Beta Particles ---")
        
        for beta in emissions['beta_particles']:
            E_max = beta[0]  # keV
            intensity = beta[1]
            n_beta = int(args.n_particles * intensity)
            
            if n_beta == 0:
                continue
            
            print(f"  E_max={E_max:.1f} keV, n={n_beta:,}")
            
            # Sample energies
            energies = np.array([sample_beta_energy(E_max) * 1000 for _ in range(n_beta)])  # keV to eV
            
            # Sample directions
            directions = np.array([sample_isotropic_direction() for _ in range(n_beta)])
            
            # Select positions
            pos_idx = np.random.choice(len(positions), size=n_beta, replace=True)
            pos = positions[pos_idx]
            
            # Run electron tracking
            result = mc.run_electrons(pos, directions, energies, max_steps=10000)
            
            total_energy_dep += result['energy_dep_eV']
            total_hits += result['hit_count']
            
            print(f"    Energy deposited: {result['total_energy_eV']:.2e} eV")
    
    # Run alpha particle simulation
    if emissions['alpha_chain']:
        print("\n--- Simulating Alpha Particles ---")
        
        # Reset GPU arrays for alpha
        mc.energy_dep = cuda.to_device(np.zeros(activity_map.shape, dtype=np.float32))
        mc.hit_count = cuda.to_device(np.zeros(activity_map.shape, dtype=np.int32))
        
        for alpha in emissions['alpha_chain']:
            E_alpha = alpha[0]  # keV
            intensity = alpha[1]
            n_alpha = int(args.n_particles * intensity)
            
            if n_alpha == 0:
                continue
            
            print(f"  E={E_alpha/1000:.1f} MeV, n={n_alpha:,}")
            
            # All alphas at same energy
            energies = np.full(n_alpha, E_alpha, dtype=np.float32)
            
            # Sample directions
            directions = np.array([sample_isotropic_direction() for _ in range(n_alpha)])
            
            # Select positions
            pos_idx = np.random.choice(len(positions), size=n_alpha, replace=True)
            pos = positions[pos_idx]
            
            # Run alpha tracking
            result = mc.run_alphas(pos, directions, energies, max_steps=50000)
            
            total_energy_dep += result['energy_dep_eV']
            total_hits += result['hit_count']
            
            print(f"    Energy deposited: {result['total_energy_eV']/1e6:.2f} MeV")
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f} seconds")
    print(f"  Particle rate: {args.n_particles/elapsed:,.0f} particles/second")
    
    # Calculate DNA damage if DAPI mask provided
    damage_results = None
    if dna_geometry and dapi_mask is not None:
        print("\n--- Calculating DNA Damage ---")
        damage_results = dna_geometry.calculate_damage_per_cell(total_energy_dep, dapi_mask)
        
        # Aggregate statistics
        total_ssb = sum(r['expected_SSB'] for r in damage_results)
        total_dsb = sum(r['expected_DSB'] for r in damage_results)
        
        print(f"  Total cells analyzed: {len(damage_results)}")
        print(f"  Total expected SSBs: {total_ssb:,}")
        print(f"  Total expected DSBs: {total_dsb:,}")
    
    # Convert energy to dose
    print("\n--- Converting to Dose ---")
    
    voxel_vol_cm3 = (voxel_size_xy * 1e-4) * (voxel_size_xy * 1e-4) * (voxel_size_z * 1e-4)
    voxel_mass_kg = voxel_vol_cm3 * 1.0 * 1e-3  # Assume water density
    
    energy_J = total_energy_dep * 1.602e-19  # eV to J
    dose_Gy = energy_J / voxel_mass_kg
    
    print(f"  Total energy deposited: {total_energy_dep.sum():.2e} eV")
    print(f"  Mean dose: {dose_Gy.mean():.4f} Gy")
    print(f"  Max dose: {dose_Gy.max():.4f} Gy")
    
    # Save outputs
    print("\n--- Saving Outputs ---")
    
    # Dose map
    dose_path = out_dir / f"dose_Gy_{args.isotope.replace('-', '')}.nii.gz"
    dose_nii = nib.Nifti1Image(dose_Gy.astype(np.float32), protein_nii.affine, protein_nii.header)
    nib.save(dose_nii, str(dose_path))
    print(f"  Dose map: {dose_path}")
    
    # Energy deposition
    energy_path = out_dir / f"energy_eV_{args.isotope.replace('-', '')}.nii.gz"
    energy_nii = nib.Nifti1Image(total_energy_dep, protein_nii.affine, protein_nii.header)
    nib.save(energy_nii, str(energy_path))
    print(f"  Energy map: {energy_path}")
    
    # Hit count
    hits_path = out_dir / f"hits_{args.isotope.replace('-', '')}.nii.gz"
    hits_nii = nib.Nifti1Image(total_hits.astype(np.float32), protein_nii.affine, protein_nii.header)
    nib.save(hits_nii, str(hits_path))
    print(f"  Hit count: {hits_path}")
    
    # Statistics JSON
    stats = {
        'timestamp': datetime.now().isoformat(),
        'isotope': args.isotope,
        'n_particles': args.n_particles,
        'grid_shape': list(activity_map.shape),
        'voxel_size_xy_um': voxel_size_xy,
        'voxel_size_z_um': voxel_size_z,
        'total_energy_eV': float(total_energy_dep.sum()),
        'mean_dose_Gy': float(dose_Gy.mean()),
        'max_dose_Gy': float(dose_Gy.max()),
        'simulation_time_s': elapsed,
        'particle_rate': args.n_particles / elapsed,
    }
    
    if damage_results:
        stats['n_cells'] = len(damage_results)
        stats['total_SSB'] = sum(r['expected_SSB'] for r in damage_results)
        stats['total_DSB'] = sum(r['expected_DSB'] for r in damage_results)
        stats['cells'] = damage_results
    
    stats_path = out_dir / f"stats_{args.isotope.replace('-', '')}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Statistics: {stats_path}")
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CellRad-MC Track-Structure Monte Carlo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with Lu-177
  python cellrad_track_structure.py --protein protein.nii.gz --isotope Lu-177 --out-dir output

  # With DAPI nuclear mask for DNA damage
  python cellrad_track_structure.py --protein protein.nii.gz --dapi dapi.nii.gz \\
      --isotope Lu-177 --out-dir output --n-particles 10000000

  # Alpha emitter simulation
  python cellrad_track_structure.py --protein protein.nii.gz --dapi dapi.nii.gz \\
      --isotope Ac-225 --out-dir output --n-particles 1000000
        """
    )
    
    # Input files
    parser.add_argument('--protein', required=True, help='Protein/activity map (NIfTI)')
    parser.add_argument('--dapi', help='DAPI nuclear mask (NIfTI) for DNA damage calculation')
    parser.add_argument('--ct', help='CT scan (NIfTI) for tissue heterogeneity')
    parser.add_argument('--auto-ct', action='store_true', help='Use uniform soft tissue (HU=19)')
    
    # Isotope
    parser.add_argument('--isotope', required=True, 
                       choices=['Lu-177', 'Y-90', 'I-131', 'Ac-225', 'At-211', 'Tb-161', 'I-125', 'In-111', 'Ga-67', 'Tc-99m'],
                       help='Radioisotope')
    
    # Output
    parser.add_argument('--out-dir', required=True, help='Output directory')
    
    # Simulation parameters
    parser.add_argument('--n-particles', type=int, default=1000000, help='Number of particles')
    parser.add_argument('--voxel-size', type=float, help='XY voxel size in microns')
    parser.add_argument('--voxel-size-z', type=float, help='Z voxel size in microns')
    parser.add_argument('--threshold', type=float, default=50, help='Activity threshold percentile')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Validation
    parser.add_argument('--validate', action='store_true', help='Run physics validation')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    if args.validate:
        # Run validation tests
        from track_structure_mc import test_track_structure
        from dna_geometry import test_dna_geometry
        
        test_track_structure()
        test_dna_geometry()
        estimate_dna_damage_yields()
        return
    
    # Run simulation
    try:
        # Import numba cuda here to check GPU availability
        from numba import cuda
        
        if not cuda.is_available():
            print("ERROR: CUDA GPU not available")
            print("Track-structure simulation requires GPU")
            sys.exit(1)
        
        stats = run_track_structure_simulation(args)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

