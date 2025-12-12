#!/usr/bin/env python3
"""
CellRad-MC DNA Geometry Module
==============================

Creates DNA geometry overlays for track-structure simulations.
Integrates with DAPI nuclear masks for cell-specific damage tracking.

Based on TOPAS-nBio DNA geometry models.
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import json

# =============================================================================
# DNA STRUCTURE CONSTANTS (from literature)
# =============================================================================

# DNA double helix structure
DNA_HELIX_DIAMETER_nm = 2.0
DNA_BASE_PAIR_SPACING_nm = 0.34  # 3.4 Å between bp
DNA_HELIX_PITCH_nm = 3.4  # 10 bp per turn
DNA_BACKBONE_RADIUS_nm = 0.12
DNA_BASE_RADIUS_nm = 0.5

# Nucleosome structure
NUCLEOSOME_DIAMETER_nm = 11.0
DNA_PER_NUCLEOSOME_bp = 147  # bp wrapped around histone
LINKER_DNA_bp = 50  # Average linker DNA

# Chromatin fiber
CHROMATIN_FIBER_DIAMETER_nm = 30.0
NUCLEOSOMES_PER_FIBER_LENGTH_nm = 6  # ~6 nucleosomes per 10 nm

# Cell nucleus
TYPICAL_NUCLEUS_DIAMETER_um = 6.0  # Human cells
DNA_CONTENT_bp = 6.4e9  # Human diploid genome
CHROMOSOMES_DIPLOID = 46

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DNAComponent:
    """Single DNA component (backbone or base)"""
    position: Tuple[float, float, float]  # nm
    component_type: int  # 0=backbone, 1=base
    strand: int  # 1 or 2
    base_pair_id: int
    chromosome_id: int

@dataclass
class ChromosomeTerritory:
    """Chromosome territory within nucleus"""
    chromosome_id: int
    center: Tuple[float, float, float]  # nm
    radius: float  # nm
    dna_content_bp: int
    density_bp_per_nm3: float

@dataclass
class CellNucleus:
    """Single cell nucleus with DNA content"""
    cell_id: int
    center: Tuple[float, float, float]  # nm
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # voxel indices
    volume_nm3: float
    dna_content_bp: int
    chromosomes: List[ChromosomeTerritory]
    
# =============================================================================
# DNA GEOMETRY BUILDER
# =============================================================================

class DNAGeometryBuilder:
    """
    Build DNA geometry from cell segmentation masks.
    
    Workflow:
    1. Load DAPI nuclear mask
    2. Identify individual nuclei
    3. Calculate DNA density per voxel
    4. Generate chromosome territories
    5. Calculate base pair positions for damage scoring
    """
    
    def __init__(
        self,
        voxel_size_um: float = 0.455,
        voxel_size_z_um: float = 1.0
    ):
        self.voxel_size_um = voxel_size_um
        self.voxel_size_z_um = voxel_size_z_um
        self.voxel_size_nm = voxel_size_um * 1000
        self.voxel_size_z_nm = voxel_size_z_um * 1000
        
        self.nuclei: List[CellNucleus] = []
        self.dna_density_map: Optional[np.ndarray] = None
        self.chromosome_map: Optional[np.ndarray] = None
        
    def load_nuclear_mask(
        self,
        dapi_mask: np.ndarray,
        min_nucleus_volume_um3: float = 50.0,
        max_nucleus_volume_um3: float = 2000.0
    ) -> int:
        """
        Load DAPI nuclear mask and identify individual nuclei.
        
        Args:
            dapi_mask: Binary or label mask of nuclei
            min_nucleus_volume_um3: Minimum valid nucleus volume
            max_nucleus_volume_um3: Maximum valid nucleus volume
            
        Returns:
            Number of valid nuclei found
        """
        # Label connected components if binary mask
        if dapi_mask.max() == 1:
            labeled_mask, n_labels = ndimage.label(dapi_mask)
        else:
            labeled_mask = dapi_mask
            n_labels = dapi_mask.max()
        
        voxel_vol_um3 = self.voxel_size_um * self.voxel_size_um * self.voxel_size_z_um
        
        self.nuclei = []
        
        for label_id in range(1, n_labels + 1):
            # Find voxels for this nucleus
            nucleus_voxels = np.argwhere(labeled_mask == label_id)
            
            if len(nucleus_voxels) == 0:
                continue
            
            # Calculate volume
            volume_um3 = len(nucleus_voxels) * voxel_vol_um3
            
            # Filter by volume
            if volume_um3 < min_nucleus_volume_um3 or volume_um3 > max_nucleus_volume_um3:
                continue
            
            # Calculate center and bounding box
            min_coords = nucleus_voxels.min(axis=0)
            max_coords = nucleus_voxels.max(axis=0)
            center_voxel = nucleus_voxels.mean(axis=0)
            
            center_nm = (
                center_voxel[0] * self.voxel_size_nm,
                center_voxel[1] * self.voxel_size_nm,
                center_voxel[2] * self.voxel_size_z_nm
            )
            
            volume_nm3 = volume_um3 * 1e9  # um³ to nm³
            
            # Estimate DNA content based on volume
            # Typical human cell has ~6.4 billion bp in ~300 um³
            typical_volume_um3 = 300.0
            dna_content = int(DNA_CONTENT_bp * volume_um3 / typical_volume_um3)
            
            # Create chromosome territories
            chromosomes = self._generate_chromosome_territories(
                center_nm, 
                volume_nm3,
                dna_content
            )
            
            nucleus = CellNucleus(
                cell_id=len(self.nuclei),
                center=center_nm,
                bounding_box=(tuple(min_coords), tuple(max_coords)),
                volume_nm3=volume_nm3,
                dna_content_bp=dna_content,
                chromosomes=chromosomes
            )
            
            self.nuclei.append(nucleus)
        
        print(f"Found {len(self.nuclei)} valid nuclei")
        return len(self.nuclei)
    
    def _generate_chromosome_territories(
        self,
        nucleus_center: Tuple[float, float, float],
        nucleus_volume_nm3: float,
        total_dna_bp: int
    ) -> List[ChromosomeTerritory]:
        """Generate chromosome territories within a nucleus"""
        chromosomes = []
        
        # Calculate approximate nucleus radius
        nucleus_radius = (3.0 * nucleus_volume_nm3 / (4.0 * np.pi)) ** (1.0/3.0)
        
        # Distribute chromosomes
        # Human: 46 chromosomes, varying sizes
        chromosome_sizes = [
            # Chromosome number, relative size
            (1, 0.085), (2, 0.082), (3, 0.067), (4, 0.065), (5, 0.062),
            (6, 0.058), (7, 0.054), (8, 0.050), (9, 0.048), (10, 0.046),
            (11, 0.046), (12, 0.045), (13, 0.039), (14, 0.037), (15, 0.035),
            (16, 0.031), (17, 0.028), (18, 0.027), (19, 0.020), (20, 0.022),
            (21, 0.016), (22, 0.017), (23, 0.053),  # X
        ]
        
        # For diploid, duplicate (simplified)
        all_chromosomes = chromosome_sizes + chromosome_sizes
        
        for i, (chrom_num, rel_size) in enumerate(all_chromosomes):
            # Random position within nucleus
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0.3, 0.8) * nucleus_radius
            
            x = nucleus_center[0] + r * np.sin(phi) * np.cos(theta)
            y = nucleus_center[1] + r * np.sin(phi) * np.sin(theta)
            z = nucleus_center[2] + r * np.cos(phi)
            
            # Territory radius proportional to DNA content
            territory_radius = (rel_size ** 0.33) * nucleus_radius * 0.3
            territory_volume = (4.0/3.0) * np.pi * territory_radius**3
            
            dna_bp = int(total_dna_bp * rel_size)
            density = dna_bp / territory_volume if territory_volume > 0 else 0
            
            chromosomes.append(ChromosomeTerritory(
                chromosome_id=chrom_num + (23 if i >= 23 else 0),
                center=(x, y, z),
                radius=territory_radius,
                dna_content_bp=dna_bp,
                density_bp_per_nm3=density
            ))
        
        return chromosomes
    
    def build_dna_density_map(
        self,
        grid_shape: Tuple[int, int, int],
        dapi_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Build DNA density map from identified nuclei.
        
        Args:
            grid_shape: Output grid dimensions
            dapi_mask: Optional mask to constrain DNA to nuclei
            
        Returns:
            3D array of DNA density (bp/nm³) per voxel
        """
        self.dna_density_map = np.zeros(grid_shape, dtype=np.float32)
        self.chromosome_map = np.zeros(grid_shape, dtype=np.int32)
        
        for nucleus in self.nuclei:
            for chrom in nucleus.chromosomes:
                # Find voxels within chromosome territory
                cx, cy, cz = chrom.center
                r = chrom.radius
                
                # Calculate voxel indices
                x_min = max(0, int((cx - r) / self.voxel_size_nm))
                x_max = min(grid_shape[0], int((cx + r) / self.voxel_size_nm) + 1)
                y_min = max(0, int((cy - r) / self.voxel_size_nm))
                y_max = min(grid_shape[1], int((cy + r) / self.voxel_size_nm) + 1)
                z_min = max(0, int((cz - r) / self.voxel_size_z_nm))
                z_max = min(grid_shape[2], int((cz + r) / self.voxel_size_z_nm) + 1)
                
                for ix in range(x_min, x_max):
                    for iy in range(y_min, y_max):
                        for iz in range(z_min, z_max):
                            # Check if voxel is within territory (ellipsoid)
                            vx = ix * self.voxel_size_nm + self.voxel_size_nm/2
                            vy = iy * self.voxel_size_nm + self.voxel_size_nm/2
                            vz = iz * self.voxel_size_z_nm + self.voxel_size_z_nm/2
                            
                            dist = np.sqrt((vx-cx)**2 + (vy-cy)**2 + (vz-cz)**2)
                            
                            if dist <= r:
                                # Apply mask if provided
                                if dapi_mask is not None:
                                    if ix < dapi_mask.shape[0] and iy < dapi_mask.shape[1] and iz < dapi_mask.shape[2]:
                                        if dapi_mask[ix, iy, iz] == 0:
                                            continue
                                
                                self.dna_density_map[ix, iy, iz] += chrom.density_bp_per_nm3
                                self.chromosome_map[ix, iy, iz] = chrom.chromosome_id
        
        print(f"DNA density map built: {self.dna_density_map.sum():.2e} total bp density")
        return self.dna_density_map
    
    def get_bp_per_voxel(self) -> np.ndarray:
        """Get base pairs per voxel"""
        if self.dna_density_map is None:
            raise ValueError("DNA density map not built")
        
        voxel_vol_nm3 = (self.voxel_size_nm ** 2) * self.voxel_size_z_nm
        return self.dna_density_map * voxel_vol_nm3
    
    def calculate_damage_per_cell(
        self,
        energy_dep_eV: np.ndarray,
        dapi_mask: np.ndarray
    ) -> List[Dict]:
        """
        Calculate DNA damage for each cell.
        
        Args:
            energy_dep_eV: Energy deposition per voxel in eV
            dapi_mask: Labeled mask of cell nuclei
            
        Returns:
            List of damage dictionaries per cell
        """
        from track_structure_mc import DNADamageCalculator, DNAHit
        
        damage_calc = DNADamageCalculator()
        results = []
        
        for nucleus in self.nuclei:
            # Get energy deposition in this nucleus
            bb_min, bb_max = nucleus.bounding_box
            
            # Extract region
            x_slice = slice(bb_min[0], bb_max[0]+1)
            y_slice = slice(bb_min[1], bb_max[1]+1)
            z_slice = slice(bb_min[2], bb_max[2]+1)
            
            region_energy = energy_dep_eV[x_slice, y_slice, z_slice]
            region_mask = dapi_mask[x_slice, y_slice, z_slice] == (nucleus.cell_id + 1)
            
            # Mask energy to this nucleus only
            nucleus_energy = region_energy * region_mask
            
            total_energy_eV = nucleus_energy.sum()
            
            # Estimate damage from energy deposition
            # ~22 eV per ionization, ~30-40 bp per ionization leads to damage
            n_ionizations = total_energy_eV / 22.0
            
            # Literature: ~1 DSB per Gy per Gbp
            dose_Gy = total_energy_eV * 1.602e-19 / (nucleus.volume_nm3 * 1e-21 * 1e-3)  # eV to J, nm³ to cm³
            dsb_per_Gy_per_Gbp = 35  # Literature value
            expected_dsb = dose_Gy * (nucleus.dna_content_bp / 1e9) * dsb_per_Gy_per_Gbp
            expected_ssb = expected_dsb * 25  # SSB:DSB ratio ~25:1
            
            results.append({
                'cell_id': nucleus.cell_id,
                'center': nucleus.center,
                'volume_nm3': nucleus.volume_nm3,
                'dna_content_bp': nucleus.dna_content_bp,
                'total_energy_eV': total_energy_eV,
                'dose_Gy': dose_Gy,
                'n_ionizations': int(n_ionizations),
                'expected_SSB': int(expected_ssb),
                'expected_DSB': int(expected_dsb),
                'survival_probability': np.exp(-0.3 * dose_Gy - 0.03 * dose_Gy**2)  # LQ model
            })
        
        return results
    
    def export_geometry(self, filepath: str):
        """Export DNA geometry to JSON file"""
        data = {
            'voxel_size_um': self.voxel_size_um,
            'voxel_size_z_um': self.voxel_size_z_um,
            'n_nuclei': len(self.nuclei),
            'nuclei': [
                {
                    'cell_id': n.cell_id,
                    'center': n.center,
                    'volume_nm3': n.volume_nm3,
                    'dna_content_bp': n.dna_content_bp,
                    'n_chromosomes': len(n.chromosomes)
                }
                for n in self.nuclei
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Geometry exported to {filepath}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_dna_damage_yields():
    """
    Print literature values for DNA damage yields.
    Used for validation of damage calculations.
    """
    print("\n" + "="*60)
    print("DNA Damage Yield Literature Values")
    print("="*60)
    
    yields = {
        "SSB per Gy per Gbp (γ-rays)": "~1000",
        "DSB per Gy per Gbp (γ-rays)": "~35",
        "SSB:DSB ratio (low LET)": "~25:1",
        "SSB:DSB ratio (high LET)": "~10:1",
        "DSB per Gy per cell (human)": "~35-40",
        "Complex DSB fraction (α)": "~70%",
        "Complex DSB fraction (γ)": "~20%",
        "Lethal lesions per Gy": "~1-2",
    }
    
    for key, value in yields.items():
        print(f"  {key}: {value}")
    
    print("\nReferences:")
    print("  - Nikjoo et al., Radiat. Res. 156 (2001) 577-583")
    print("  - Friedland et al., Radiat. Res. 159 (2003) 401-410")
    print("  - Rothkamm & Löbrich, PNAS 100 (2003) 5057-5062")
    print("="*60)


def calculate_let_from_track_density(
    n_ionizations_per_nm: float,
    ion_pair_energy_eV: float = 22.0
) -> float:
    """
    Calculate Linear Energy Transfer (LET) from ionization density.
    
    Args:
        n_ionizations_per_nm: Ionizations per nanometer track length
        ion_pair_energy_eV: Average energy per ion pair (W-value)
        
    Returns:
        LET in keV/μm
    """
    # Energy per nm
    dE_per_nm = n_ionizations_per_nm * ion_pair_energy_eV / 1000  # keV
    
    # Convert to keV/μm
    let_keV_um = dE_per_nm * 1000
    
    return let_keV_um


# =============================================================================
# TESTING
# =============================================================================

def test_dna_geometry():
    """Test DNA geometry builder"""
    print("\n" + "="*60)
    print("DNA Geometry Test")
    print("="*60)
    
    # Create synthetic nuclear mask
    print("\nCreating synthetic nuclear mask...")
    grid_shape = (100, 100, 50)
    mask = np.zeros(grid_shape, dtype=np.uint8)
    
    # Add some spherical nuclei
    nuclei_centers = [
        (25, 25, 25),
        (75, 25, 25),
        (50, 75, 25),
    ]
    
    for i, (cx, cy, cz) in enumerate(nuclei_centers):
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                for z in range(grid_shape[2]):
                    r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                    if r < 10:  # 10 voxel radius
                        mask[x, y, z] = i + 1
    
    # Build geometry
    builder = DNAGeometryBuilder(voxel_size_um=0.455, voxel_size_z_um=1.0)
    n_nuclei = builder.load_nuclear_mask(mask, min_nucleus_volume_um3=10.0)
    
    print(f"\nFound {n_nuclei} nuclei")
    
    for nucleus in builder.nuclei:
        print(f"\nNucleus {nucleus.cell_id}:")
        print(f"  Center: {nucleus.center}")
        print(f"  Volume: {nucleus.volume_nm3:.2e} nm³")
        print(f"  DNA content: {nucleus.dna_content_bp:.2e} bp")
        print(f"  Chromosomes: {len(nucleus.chromosomes)}")
    
    # Build DNA density map
    print("\nBuilding DNA density map...")
    dna_map = builder.build_dna_density_map(grid_shape, mask)
    
    bp_per_voxel = builder.get_bp_per_voxel()
    print(f"Total base pairs in volume: {bp_per_voxel.sum():.2e}")
    
    # Test damage calculation
    print("\nTesting damage calculation...")
    
    # Create synthetic energy deposition
    energy_dep = np.zeros(grid_shape, dtype=np.float32)
    energy_dep[20:30, 20:30, 20:30] = 100.0  # 100 eV per voxel
    
    damage_results = builder.calculate_damage_per_cell(energy_dep, mask)
    
    for result in damage_results:
        print(f"\nCell {result['cell_id']}:")
        print(f"  Total energy: {result['total_energy_eV']:.1f} eV")
        print(f"  Dose: {result['dose_Gy']:.4f} Gy")
        print(f"  Expected SSBs: {result['expected_SSB']}")
        print(f"  Expected DSBs: {result['expected_DSB']}")
        print(f"  Survival prob: {result['survival_probability']:.4f}")
    
    # Print literature values
    estimate_dna_damage_yields()
    
    print("\n" + "="*60)
    print("DNA Geometry Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_dna_geometry()

