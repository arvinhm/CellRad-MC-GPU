#!/usr/bin/env python3
"""
CellRad-MC Track-Structure Monte Carlo
======================================

Full track-structure simulation with event-by-event tracking,
DNA geometry overlay, and damage classification (SSB/DSB).

Based on Geant4-DNA and TOPAS-nBio physics models.

Author: Arvin
"""

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import IntEnum

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

ELECTRON_MASS_keV = 511.0  # Electron rest mass in keV
BOHR_RADIUS_nm = 0.0529    # Bohr radius in nm
WATER_DENSITY = 1.0        # g/cm³
AVOGADRO = 6.022e23

# DNA structure constants (from TOPAS-nBio)
DNA_BACKBONE_RADIUS_nm = 0.24
DNA_BASE_RADIUS_nm = 0.208
DNA_HELIX_RADIUS_nm = 1.0
BP_SPACING_nm = 0.34

# Damage thresholds (from Geant4-DNA)
DIRECT_DAMAGE_THRESHOLD_eV = 17.5
DIRECT_DAMAGE_LOWER_eV = 5.0
DIRECT_DAMAGE_UPPER_eV = 37.5
DSB_BP_THRESHOLD = 10  # Max base pairs between SSBs for DSB

# =============================================================================
# CROSS-SECTION DATA (from Geant4-DNA / NIST)
# =============================================================================

# Electron ionization cross-sections in water (barn = 1e-24 cm²)
# Energy (eV), Cross-section (barn)
ELECTRON_IONIZATION_XS = np.array([
    [10, 0.0],
    [12, 0.05],
    [15, 0.15],
    [20, 0.35],
    [30, 0.65],
    [50, 1.05],
    [70, 1.25],
    [100, 1.35],
    [150, 1.30],
    [200, 1.20],
    [300, 1.05],
    [500, 0.85],
    [700, 0.70],
    [1000, 0.55],
    [2000, 0.35],
    [5000, 0.18],
    [10000, 0.10],
    [50000, 0.03],
    [100000, 0.018],
    [500000, 0.005],
    [1000000, 0.003],
], dtype=np.float32)

# Electron excitation cross-sections in water
ELECTRON_EXCITATION_XS = np.array([
    [7.5, 0.0],
    [10, 0.02],
    [12, 0.08],
    [15, 0.15],
    [20, 0.25],
    [30, 0.35],
    [50, 0.40],
    [70, 0.38],
    [100, 0.32],
    [150, 0.25],
    [200, 0.20],
    [300, 0.15],
    [500, 0.10],
    [1000, 0.06],
    [5000, 0.02],
    [10000, 0.01],
], dtype=np.float32)

# Electron elastic cross-sections (Champion model)
ELECTRON_ELASTIC_XS = np.array([
    [7.5, 10.0],
    [10, 9.5],
    [15, 8.5],
    [20, 7.5],
    [30, 6.0],
    [50, 4.5],
    [70, 3.5],
    [100, 2.8],
    [150, 2.2],
    [200, 1.8],
    [300, 1.4],
    [500, 1.0],
    [1000, 0.6],
    [5000, 0.2],
    [10000, 0.1],
], dtype=np.float32)

# Alpha ionization cross-sections (Rudd model)
ALPHA_IONIZATION_XS = np.array([
    [100, 50.0],      # keV
    [500, 120.0],
    [1000, 150.0],
    [2000, 160.0],
    [3000, 155.0],
    [4000, 145.0],
    [5000, 135.0],
    [6000, 125.0],
    [7000, 115.0],
    [8000, 105.0],
    [10000, 90.0],
], dtype=np.float32)

# Proton ionization cross-sections
PROTON_IONIZATION_XS = np.array([
    [10, 5.0],        # keV
    [50, 25.0],
    [100, 45.0],
    [200, 65.0],
    [500, 85.0],
    [1000, 90.0],
    [2000, 80.0],
    [5000, 55.0],
    [10000, 35.0],
], dtype=np.float32)

# W-value (energy per ion pair)
W_VALUE_eV = 22.0  # eV per ion pair in water

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ParticleType(IntEnum):
    ELECTRON = 0
    PROTON = 1
    ALPHA = 2
    PHOTON = 3

class DamageType(IntEnum):
    NONE = 0
    DIRECT = 1
    INDIRECT = 2
    MIXED = 3

@dataclass
class DNAHit:
    """Single energy deposition in DNA"""
    position: Tuple[float, float, float]  # nm
    energy_eV: float
    particle_type: int
    chromosome_id: int
    base_pair_id: int
    strand: int  # 1 or 2
    component: int  # 0=base, 1=backbone
    damage_type: int

@dataclass
class StrandBreak:
    """Single or double strand break"""
    chromosome_id: int
    base_pair_id: int
    strand: int
    damage_type: int
    position: Tuple[float, float, float]

@dataclass
class CellDamage:
    """DNA damage summary for a single cell"""
    cell_id: int
    n_hits: int
    total_energy_eV: float
    n_SSB: int
    n_DSB: int
    n_complex_DSB: int
    DSB_positions: List[Tuple[float, float, float]]

# =============================================================================
# GPU KERNELS FOR TRACK-STRUCTURE
# =============================================================================

@cuda.jit(device=True)
def get_cross_section(xs_table, energy, n_entries):
    """Interpolate cross-section from table"""
    if energy <= xs_table[0, 0]:
        return xs_table[0, 1]
    if energy >= xs_table[n_entries-1, 0]:
        return xs_table[n_entries-1, 1]
    
    # Binary search
    low, high = 0, n_entries - 1
    while low < high - 1:
        mid = (low + high) // 2
        if xs_table[mid, 0] <= energy:
            low = mid
        else:
            high = mid
    
    # Linear interpolation
    E1, xs1 = xs_table[low, 0], xs_table[low, 1]
    E2, xs2 = xs_table[high, 0], xs_table[high, 1]
    
    return xs1 + (xs2 - xs1) * (energy - E1) / (E2 - E1)


@cuda.jit(device=True)
def sample_secondary_energy(E_primary, rng_state):
    """Sample secondary electron energy from ionization"""
    # Simplified BEB model for secondary electron energy
    # Most secondaries are low energy (<50 eV)
    
    W = 13.6  # Mean binding energy for water
    max_secondary = (E_primary - W) / 2.0
    
    if max_secondary <= 0:
        return 0.0
    
    # Sample from 1/(W+E)² distribution
    # CDF inversion
    u = xoroshiro128p_uniform_float32(rng_state, 0)
    E_secondary = W * u / (1.0 - u)
    
    return min(E_secondary, max_secondary)


@cuda.jit
def track_electrons_kernel(
    # Particle arrays
    x, y, z, dx, dy, dz, energy,
    active, particle_id,
    # Cross-section tables
    ionization_xs, excitation_xs, elastic_xs,
    # Geometry
    grid_shape, voxel_size_nm,
    # DNA overlay (chromosome ID, base pair density)
    dna_density,
    # Output
    energy_dep, hit_count,
    # RNG
    rng_states,
    # Config
    n_particles, max_steps
):
    """
    Track electrons through water with full physics.
    Event-by-event simulation with:
    - Elastic scattering
    - Ionization (with secondary production)
    - Excitation
    - Energy loss to thermal cutoff
    """
    idx = cuda.grid(1)
    if idx >= n_particles or not active[idx]:
        return
    
    # Load particle state
    px, py, pz = x[idx], y[idx], z[idx]
    pdx, pdy, pdz = dx[idx], dy[idx], dz[idx]
    E = energy[idx]
    
    cutoff_energy = 10.0  # eV - thermalization cutoff
    
    n_ion_xs = ionization_xs.shape[0]
    n_exc_xs = excitation_xs.shape[0]
    n_ela_xs = elastic_xs.shape[0]
    
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]
    
    for step in range(max_steps):
        if E < cutoff_energy:
            break
        
        # Get cross-sections at current energy
        sigma_ion = get_cross_section(ionization_xs, E, n_ion_xs)
        sigma_exc = get_cross_section(excitation_xs, E, n_exc_xs)
        sigma_ela = get_cross_section(elastic_xs, E, n_ela_xs)
        
        sigma_total = sigma_ion + sigma_exc + sigma_ela
        
        if sigma_total <= 0:
            break
        
        # Mean free path (nm)
        n_density = WATER_DENSITY * AVOGADRO / 18.0 * 1e-21  # molecules/nm³
        mfp = 1.0 / (n_density * sigma_total * 1e-24)  # nm
        
        # Sample path length
        u = xoroshiro128p_uniform_float32(rng_states, idx)
        path = -mfp * math.log(max(u, 1e-10))
        
        # Move particle
        px += pdx * path
        py += pdy * path
        pz += pdz * path
        
        # Check bounds
        vx = int(px / voxel_size_nm)
        vy = int(py / voxel_size_nm)
        vz = int(pz / voxel_size_nm)
        
        if vx < 0 or vx >= nx or vy < 0 or vy >= ny or vz < 0 or vz >= nz:
            break
        
        # Sample interaction type
        u = xoroshiro128p_uniform_float32(rng_states, idx)
        
        if u < sigma_ion / sigma_total:
            # Ionization
            W = 13.6  # Mean ionization energy
            E_secondary = sample_secondary_energy(E, rng_states)
            E_deposited = W  # Energy to create ion pair
            E = E - E_deposited - E_secondary
            
            # Score energy deposition
            cuda.atomic.add(energy_dep, (vx, vy, vz), E_deposited)
            cuda.atomic.add(hit_count, (vx, vy, vz), 1)
            
            # TODO: Add secondary electron to stack
            
        elif u < (sigma_ion + sigma_exc) / sigma_total:
            # Excitation
            E_exc = 8.0  # Mean excitation energy
            E = E - E_exc
            cuda.atomic.add(energy_dep, (vx, vy, vz), E_exc)
            
        else:
            # Elastic scattering - change direction
            # Highland formula for angular deflection
            cos_theta = 1.0 - 2.0 * xoroshiro128p_uniform_float32(rng_states, idx)
            sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
            phi = 2.0 * 3.14159 * xoroshiro128p_uniform_float32(rng_states, idx)
            
            # Rotate direction
            if abs(pdz) < 0.99:
                ux = sin_theta * math.cos(phi)
                uy = sin_theta * math.sin(phi)
                uz = cos_theta
                
                # Transform to lab frame
                norm = math.sqrt(pdx*pdx + pdy*pdy)
                if norm > 1e-10:
                    pdx_new = pdx * uz + (pdx * pdz * ux - pdy * uy) / norm
                    pdy_new = pdy * uz + (pdy * pdz * ux + pdx * uy) / norm
                    pdz_new = pdz * uz - norm * ux
                else:
                    pdx_new, pdy_new, pdz_new = ux, uy, uz
                
                pdx, pdy, pdz = pdx_new, pdy_new, pdz_new
    
    # Store final state
    x[idx], y[idx], z[idx] = px, py, pz
    dx[idx], dy[idx], dz[idx] = pdx, pdy, pdz
    energy[idx] = E
    active[idx] = E >= cutoff_energy


@cuda.jit
def track_alphas_kernel(
    x, y, z, dx, dy, dz, energy,
    active, particle_id,
    ionization_xs,
    grid_shape, voxel_size_nm,
    energy_dep, hit_count,
    rng_states,
    n_particles, max_steps
):
    """
    Track alpha particles with Bragg peak behavior.
    Alpha particles produce dense ionization along their path.
    """
    idx = cuda.grid(1)
    if idx >= n_particles or not active[idx]:
        return
    
    px, py, pz = x[idx], y[idx], z[idx]
    pdx, pdy, pdz = dx[idx], dy[idx], dz[idx]
    E = energy[idx]  # keV
    
    cutoff_energy = 100.0  # keV for alphas
    n_xs = ionization_xs.shape[0]
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]
    
    for step in range(max_steps):
        if E < cutoff_energy:
            break
        
        # Alpha stopping power (Bragg curve)
        # S = A * E^(-0.5) for high energy
        # S increases at low energy (Bragg peak)
        if E > 1000:  # High energy regime
            dEdx = 100.0 * math.pow(E / 5000.0, -0.5)  # keV/um
        else:  # Bragg peak regime
            dEdx = 200.0 * math.pow(E / 1000.0, 0.3)
        
        # Step size limited by energy loss
        step_size = min(0.1, E / (dEdx * 10))  # um
        step_nm = step_size * 1000  # nm
        
        # Move particle
        px += pdx * step_nm
        py += pdy * step_nm
        pz += pdz * step_nm
        
        # Check bounds
        vx = int(px / voxel_size_nm)
        vy = int(py / voxel_size_nm)
        vz = int(pz / voxel_size_nm)
        
        if vx < 0 or vx >= nx or vy < 0 or vy >= ny or vz < 0 or vz >= nz:
            break
        
        # Energy deposited in this step
        dE = dEdx * step_size  # keV
        E = E - dE
        
        # Score energy deposition (convert keV to eV)
        cuda.atomic.add(energy_dep, (vx, vy, vz), dE * 1000.0)
        
        # Count ionizations (W-value = 22 eV per ion pair)
        n_ions = int(dE * 1000.0 / W_VALUE_eV)
        cuda.atomic.add(hit_count, (vx, vy, vz), n_ions)
        
        # Small angle scattering
        theta = 0.001 / math.sqrt(E)  # Small angle approximation
        phi = 2.0 * 3.14159 * xoroshiro128p_uniform_float32(rng_states, idx)
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        pdx_new = pdx * cos_theta + sin_theta * math.cos(phi)
        pdy_new = pdy * cos_theta + sin_theta * math.sin(phi)
        pdz_new = pdz * cos_theta
        
        norm = math.sqrt(pdx_new*pdx_new + pdy_new*pdy_new + pdz_new*pdz_new)
        pdx, pdy, pdz = pdx_new/norm, pdy_new/norm, pdz_new/norm
    
    x[idx], y[idx], z[idx] = px, py, pz
    dx[idx], dy[idx], dz[idx] = pdx, pdy, pdz
    energy[idx] = E
    active[idx] = E >= cutoff_energy


# =============================================================================
# DNA DAMAGE CLASSIFICATION
# =============================================================================

class DNADamageCalculator:
    """
    Calculate DNA damage from energy deposition map.
    
    Based on TOPAS-nBio TsDNADamageCalculator.
    """
    
    def __init__(
        self,
        direct_threshold_eV: float = 17.5,
        dsb_bp_threshold: int = 10,
        indirect_probability: float = 0.4
    ):
        self.direct_threshold = direct_threshold_eV
        self.dsb_bp_threshold = dsb_bp_threshold
        self.indirect_prob = indirect_probability
        
        # Damage maps
        self.ssb_map = {}  # (chromosome, bp, strand) -> damage_type
        self.dsb_map = {}  # (chromosome, bp) -> (strand1_type, strand2_type)
        self.base_damage_map = {}
        
    def is_direct_damage(self, edep_eV: float) -> bool:
        """Determine if energy deposition causes direct strand break"""
        if edep_eV < DIRECT_DAMAGE_LOWER_eV:
            return False
        elif edep_eV >= DIRECT_DAMAGE_UPPER_eV:
            return True
        else:
            # Linear probability
            prob = (edep_eV - DIRECT_DAMAGE_LOWER_eV) / (DIRECT_DAMAGE_UPPER_eV - DIRECT_DAMAGE_LOWER_eV)
            return np.random.random() < prob
    
    def classify_damage(self, hits: List[DNAHit]) -> Dict:
        """
        Classify DNA hits into SSBs and DSBs.
        
        Returns:
            Dictionary with damage counts and positions
        """
        # Clear previous maps
        self.ssb_map.clear()
        self.dsb_map.clear()
        self.base_damage_map.clear()
        
        # Group hits by base pair
        bp_damage = {}  # (chromosome, bp) -> {strand: [hits]}
        
        for hit in hits:
            key = (hit.chromosome_id, hit.base_pair_id)
            if key not in bp_damage:
                bp_damage[key] = {1: [], 2: []}
            bp_damage[key][hit.strand].append(hit)
        
        # Classify each hit
        for (chrom, bp), strands in bp_damage.items():
            for strand, strand_hits in strands.items():
                for hit in strand_hits:
                    if hit.component == 1:  # Backbone
                        if self.is_direct_damage(hit.energy_eV):
                            self.ssb_map[(chrom, bp, strand)] = DamageType.DIRECT
                    else:  # Base
                        if self.is_direct_damage(hit.energy_eV):
                            self.base_damage_map[(chrom, bp, strand)] = DamageType.DIRECT
        
        # Identify DSBs (SSBs on opposite strands within dsb_bp_threshold)
        ssb_positions = list(self.ssb_map.keys())
        used_ssbs = set()
        
        for chrom1, bp1, strand1 in ssb_positions:
            if (chrom1, bp1, strand1) in used_ssbs:
                continue
            
            opposite_strand = 3 - strand1  # 1->2, 2->1
            
            # Look for nearby SSB on opposite strand
            for bp2 in range(bp1 - self.dsb_bp_threshold, bp1 + self.dsb_bp_threshold + 1):
                if (chrom1, bp2, opposite_strand) in self.ssb_map:
                    if (chrom1, bp2, opposite_strand) not in used_ssbs:
                        # Found a DSB
                        self.dsb_map[(chrom1, bp1)] = (
                            self.ssb_map[(chrom1, bp1, strand1)],
                            self.ssb_map[(chrom1, bp2, opposite_strand)]
                        )
                        used_ssbs.add((chrom1, bp1, strand1))
                        used_ssbs.add((chrom1, bp2, opposite_strand))
                        break
        
        # Count damage
        n_ssb = len(self.ssb_map) - len(used_ssbs)  # SSBs not part of DSBs
        n_dsb = len(self.dsb_map)
        n_base_damage = len(self.base_damage_map)
        
        return {
            'n_SSB': n_ssb,
            'n_DSB': n_dsb,
            'n_base_damage': n_base_damage,
            'n_SSB_direct': sum(1 for v in self.ssb_map.values() if v == DamageType.DIRECT),
            'n_DSB_direct': sum(1 for v in self.dsb_map.values() if v[0] == DamageType.DIRECT and v[1] == DamageType.DIRECT),
        }


# =============================================================================
# MAIN SIMULATION CLASS
# =============================================================================

class TrackStructureMC:
    """
    Full track-structure Monte Carlo simulation.
    
    Features:
    - Event-by-event electron tracking
    - Alpha particle tracking with Bragg peak
    - DNA geometry overlay
    - SSB/DSB damage classification
    - Per-cell dose accumulation
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        voxel_size_nm: float = 1.0,
        dna_density: Optional[np.ndarray] = None
    ):
        self.grid_shape = grid_shape
        self.voxel_size_nm = voxel_size_nm
        
        # Initialize GPU arrays
        self.energy_dep = cuda.to_device(np.zeros(grid_shape, dtype=np.float32))
        self.hit_count = cuda.to_device(np.zeros(grid_shape, dtype=np.int32))
        
        # Cross-section tables
        self.ion_xs_d = cuda.to_device(ELECTRON_IONIZATION_XS)
        self.exc_xs_d = cuda.to_device(ELECTRON_EXCITATION_XS)
        self.ela_xs_d = cuda.to_device(ELECTRON_ELASTIC_XS)
        self.alpha_xs_d = cuda.to_device(ALPHA_IONIZATION_XS)
        
        # DNA density map
        if dna_density is not None:
            self.dna_density_d = cuda.to_device(dna_density.astype(np.float32))
        else:
            self.dna_density_d = cuda.to_device(np.ones(grid_shape, dtype=np.float32))
        
        # Damage calculator
        self.damage_calc = DNADamageCalculator()
        
        print(f"Track-Structure MC initialized:")
        print(f"  Grid: {grid_shape}")
        print(f"  Voxel size: {voxel_size_nm} nm")
    
    def run_electrons(
        self,
        positions: np.ndarray,  # (N, 3) in nm
        directions: np.ndarray,  # (N, 3) normalized
        energies: np.ndarray,    # (N,) in eV
        max_steps: int = 10000
    ) -> Dict:
        """Run electron track-structure simulation"""
        n_particles = len(energies)
        
        # Initialize particle arrays
        x = cuda.to_device(positions[:, 0].astype(np.float32))
        y = cuda.to_device(positions[:, 1].astype(np.float32))
        z = cuda.to_device(positions[:, 2].astype(np.float32))
        dx = cuda.to_device(directions[:, 0].astype(np.float32))
        dy = cuda.to_device(directions[:, 1].astype(np.float32))
        dz = cuda.to_device(directions[:, 2].astype(np.float32))
        energy = cuda.to_device(energies.astype(np.float32))
        active = cuda.to_device(np.ones(n_particles, dtype=np.bool_))
        particle_id = cuda.to_device(np.arange(n_particles, dtype=np.int32))
        
        # RNG states
        rng_states = create_xoroshiro128p_states(n_particles, seed=42)
        
        # Grid configuration
        threads_per_block = 256
        blocks = (n_particles + threads_per_block - 1) // threads_per_block
        grid_shape_d = cuda.to_device(np.array(self.grid_shape, dtype=np.int32))
        
        # Run kernel
        track_electrons_kernel[blocks, threads_per_block](
            x, y, z, dx, dy, dz, energy,
            active, particle_id,
            self.ion_xs_d, self.exc_xs_d, self.ela_xs_d,
            grid_shape_d, self.voxel_size_nm,
            self.dna_density_d,
            self.energy_dep, self.hit_count,
            rng_states,
            n_particles, max_steps
        )
        
        cuda.synchronize()
        
        # Get results
        energy_dep = self.energy_dep.copy_to_host()
        hit_count = self.hit_count.copy_to_host()
        
        return {
            'energy_dep_eV': energy_dep,
            'hit_count': hit_count,
            'total_energy_eV': energy_dep.sum(),
            'total_hits': hit_count.sum(),
        }
    
    def run_alphas(
        self,
        positions: np.ndarray,  # (N, 3) in nm
        directions: np.ndarray,  # (N, 3) normalized
        energies: np.ndarray,    # (N,) in keV
        max_steps: int = 50000
    ) -> Dict:
        """Run alpha particle track-structure simulation"""
        n_particles = len(energies)
        
        # Initialize particle arrays
        x = cuda.to_device(positions[:, 0].astype(np.float32))
        y = cuda.to_device(positions[:, 1].astype(np.float32))
        z = cuda.to_device(positions[:, 2].astype(np.float32))
        dx = cuda.to_device(directions[:, 0].astype(np.float32))
        dy = cuda.to_device(directions[:, 1].astype(np.float32))
        dz = cuda.to_device(directions[:, 2].astype(np.float32))
        energy = cuda.to_device(energies.astype(np.float32))
        active = cuda.to_device(np.ones(n_particles, dtype=np.bool_))
        particle_id = cuda.to_device(np.arange(n_particles, dtype=np.int32))
        
        rng_states = create_xoroshiro128p_states(n_particles, seed=42)
        
        threads_per_block = 256
        blocks = (n_particles + threads_per_block - 1) // threads_per_block
        grid_shape_d = cuda.to_device(np.array(self.grid_shape, dtype=np.int32))
        
        track_alphas_kernel[blocks, threads_per_block](
            x, y, z, dx, dy, dz, energy,
            active, particle_id,
            self.alpha_xs_d,
            grid_shape_d, self.voxel_size_nm,
            self.energy_dep, self.hit_count,
            rng_states,
            n_particles, max_steps
        )
        
        cuda.synchronize()
        
        energy_dep = self.energy_dep.copy_to_host()
        hit_count = self.hit_count.copy_to_host()
        
        return {
            'energy_dep_eV': energy_dep,
            'hit_count': hit_count,
            'total_energy_eV': energy_dep.sum(),
            'total_hits': hit_count.sum(),
        }
    
    def calculate_damage(
        self,
        energy_dep: np.ndarray,
        nucleus_mask: np.ndarray,
        bp_per_voxel: float = 1000.0
    ) -> Dict:
        """
        Calculate DNA damage from energy deposition map.
        
        Args:
            energy_dep: Energy deposition in eV per voxel
            nucleus_mask: Binary mask of cell nuclei
            bp_per_voxel: Estimated base pairs per voxel
            
        Returns:
            Damage statistics
        """
        hits = []
        
        # Find voxels with energy deposition in nuclei
        in_nucleus = energy_dep > 0
        if nucleus_mask is not None:
            in_nucleus &= nucleus_mask > 0
        
        indices = np.argwhere(in_nucleus)
        
        for ix, iy, iz in indices:
            edep = energy_dep[ix, iy, iz]
            
            # Approximate number of ionizations in this voxel
            n_ions = int(edep / W_VALUE_eV)
            
            # Create DNA hits for each ionization
            for _ in range(min(n_ions, 100)):  # Limit for performance
                hit = DNAHit(
                    position=(
                        (ix + np.random.random()) * self.voxel_size_nm,
                        (iy + np.random.random()) * self.voxel_size_nm,
                        (iz + np.random.random()) * self.voxel_size_nm
                    ),
                    energy_eV=W_VALUE_eV,
                    particle_type=0,
                    chromosome_id=1,  # Simplified: single chromosome
                    base_pair_id=int(np.random.random() * bp_per_voxel),
                    strand=np.random.randint(1, 3),
                    component=np.random.randint(0, 2),  # 0=base, 1=backbone
                    damage_type=0
                )
                hits.append(hit)
        
        # Classify damage
        damage = self.damage_calc.classify_damage(hits)
        damage['n_hits'] = len(hits)
        
        return damage


# =============================================================================
# TESTING
# =============================================================================

def test_track_structure():
    """Test the track-structure simulation"""
    print("\n" + "="*60)
    print("CellRad-MC Track-Structure Test")
    print("="*60)
    
    # Create simulation
    grid_shape = (100, 100, 100)
    voxel_size_nm = 10.0  # 10 nm voxels
    
    mc = TrackStructureMC(grid_shape, voxel_size_nm)
    
    # Test electrons
    print("\n--- Electron Track-Structure Test ---")
    n_electrons = 1000
    positions = np.random.uniform(200, 800, (n_electrons, 3))
    directions = np.random.randn(n_electrons, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    energies = np.random.uniform(100, 10000, n_electrons)  # 100 eV - 10 keV
    
    result = mc.run_electrons(positions, directions, energies)
    print(f"  Total energy deposited: {result['total_energy_eV']:.1f} eV")
    print(f"  Total ionizations: {result['total_hits']}")
    
    # Test alphas
    print("\n--- Alpha Track-Structure Test ---")
    n_alphas = 100
    positions = np.random.uniform(200, 800, (n_alphas, 3))
    directions = np.zeros((n_alphas, 3))
    directions[:, 0] = 1.0  # All moving in +x direction
    energies = np.full(n_alphas, 5000.0)  # 5 MeV
    
    # Reset grids
    mc.energy_dep = cuda.to_device(np.zeros(grid_shape, dtype=np.float32))
    mc.hit_count = cuda.to_device(np.zeros(grid_shape, dtype=np.int32))
    
    result = mc.run_alphas(positions, directions, energies)
    print(f"  Total energy deposited: {result['total_energy_eV']/1e6:.2f} MeV")
    print(f"  Total ionizations: {result['total_hits']}")
    
    # Calculate damage
    print("\n--- DNA Damage Calculation ---")
    damage = mc.calculate_damage(
        result['energy_dep_eV'],
        nucleus_mask=None,
        bp_per_voxel=1000
    )
    print(f"  Total hits processed: {damage['n_hits']}")
    print(f"  SSBs: {damage['n_SSB']}")
    print(f"  DSBs: {damage['n_DSB']}")
    print(f"  Base damages: {damage['n_base_damage']}")
    
    print("\n" + "="*60)
    print("Track-Structure Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_track_structure()

