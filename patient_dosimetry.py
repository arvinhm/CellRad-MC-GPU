#!/usr/bin/env python3
"""
CellRad-MC Patient Dosimetry Module
====================================

Clinical dosimetry from PET/CT imaging for targeted radionuclide therapy.

This module bridges the gap between:
- Macro-scale imaging (mm resolution PET/CT)
- Micro-scale radiobiology (cellular DNA damage)

Physics Approach:
-----------------
1. Dose Point Kernel (DPK) convolution for absorbed dose calculation
2. CT-based density correction for tissue heterogeneity
3. MIRD-validated S-values for organ-level dosimetry
4. Literature-derived RBE and DNA damage parameters

Usage:
------
    python patient_dosimetry.py --pet pet_scan.nii.gz --ct ct_scan.nii.gz \
        --segmentation seg_folder/ --isotope Lu-177 --output results/

    # With auto-detection of files in patient folder
    python patient_dosimetry.py --patient-folder patient/fdg_xxx/ --isotope Lu-177

Author: CellRad-MC Team
"""

import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, convolve
from scipy.interpolate import interp1d
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Fundamental constants
AVOGADRO = 6.022e23  # mol^-1
KEV_TO_J = 1.602e-16  # J/keV
MEV_TO_J = 1.602e-13  # J/MeV
WATER_DENSITY = 1.0  # g/cm³


def extract_nifti_metadata(nii_path: Path) -> Dict:
    """
    Extract metadata from NIfTI file header and JSON sidecar if available.
    
    Looks for injection dose, patient weight, scan time, etc.
    """
    metadata = {}
    
    # Load NIfTI header
    img = nib.load(str(nii_path))
    header = img.header
    
    # Extract basic info
    metadata['dimensions'] = list(img.shape)
    metadata['voxel_size_mm'] = list(header.get_zooms()[:3])
    metadata['data_type'] = str(header.get_data_dtype())
    
    # Try to get description field (sometimes contains injection info)
    try:
        descrip = header.get('descrip', b'')
        if isinstance(descrip, np.ndarray):
            descrip = descrip.tobytes().decode('utf-8', errors='ignore').strip()
        elif isinstance(descrip, bytes):
            descrip = descrip.decode('utf-8', errors='ignore').strip()
        else:
            descrip = str(descrip).strip()
        if descrip:
            metadata['description'] = descrip
    except Exception:
        pass
    
    # Check for JSON sidecar (BIDS format)
    json_path = nii_path.with_suffix('').with_suffix('.json')
    if not json_path.exists():
        json_path = str(nii_path).replace('.nii.gz', '.json')
        json_path = Path(json_path)
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                sidecar = json.load(f)
                
            # Common PET metadata fields
            if 'RadionuclideTotalDose' in sidecar:
                metadata['injected_dose_Bq'] = sidecar['RadionuclideTotalDose']
                metadata['injected_dose_MBq'] = sidecar['RadionuclideTotalDose'] / 1e6
            if 'RadionuclideHalfLife' in sidecar:
                metadata['half_life_s'] = sidecar['RadionuclideHalfLife']
            if 'PatientWeight' in sidecar:
                metadata['patient_weight_kg'] = sidecar['PatientWeight']
            if 'SeriesDescription' in sidecar:
                metadata['series_description'] = sidecar['SeriesDescription']
            if 'RadiopharmaceuticalStartTime' in sidecar:
                metadata['injection_time'] = sidecar['RadiopharmaceuticalStartTime']
            if 'AcquisitionTime' in sidecar:
                metadata['acquisition_time'] = sidecar['AcquisitionTime']
                
            metadata['json_sidecar'] = json_path.name
        except Exception as e:
            metadata['json_error'] = str(e)
    
    return metadata

# FDG PET constants
FDG_HALF_LIFE_MIN = 109.77  # F-18 half-life in minutes
SUV_REFERENCE_TIME_MIN = 60  # Standard SUV reference time

# =============================================================================
# ISOTOPE DATABASE - Same as cellrad_mc.py
# =============================================================================

THERAPEUTIC_ISOTOPES = {
    'Lu-177': {
        'type': 'beta',
        'half_life_d': 6.647,
        'half_life_s': 6.647 * 24 * 3600,
        'decay_constant': np.log(2) / (6.647 * 24 * 3600),
        'E_mean_beta_keV': 134.2,  # ICRP 107 weighted average
        'E_max_beta_keV': 498.3,
        'beta_range_mm': 1.8,  # 90% range in soft tissue
        'gammas': [(113.0, 0.062), (208.4, 0.104)],
        'RBE': 1.1,
        'alpha_LQ': 0.35,  # Gy^-1
        'beta_LQ': 0.035,  # Gy^-2
        'DSB_per_Gy': 38.5,  # DSB per cell per Gy
        # Biological half-lives for effective half-life calculation
        'bio_half_life_h': {
            'tumor': 72,      # 3 days - slow clearance from PSMA-expressing tissue
            'kidney': 48,     # 2 days
            'liver': 24,      # 1 day
            'spleen': 36,
            'bone': 120,      # 5 days - bone metastases
            'default': 24,    # 1 day for other tissues
        },
    },
    'Y-90': {
        'type': 'beta',
        'half_life_d': 2.668,
        'half_life_s': 2.668 * 24 * 3600,
        'decay_constant': np.log(2) / (2.668 * 24 * 3600),
        'E_mean_beta_keV': 933.7,
        'E_max_beta_keV': 2280.1,
        'beta_range_mm': 5.3,
        'gammas': [],
        'RBE': 1.0,
        'alpha_LQ': 0.30,
        'beta_LQ': 0.030,
        'DSB_per_Gy': 35.0,
    },
    'I-131': {
        'type': 'beta',
        'half_life_d': 8.025,
        'half_life_s': 8.025 * 24 * 3600,
        'decay_constant': np.log(2) / (8.025 * 24 * 3600),
        'E_mean_beta_keV': 182.0,
        'E_max_beta_keV': 606.3,
        'beta_range_mm': 2.3,
        'gammas': [(364.5, 0.815), (637.0, 0.072)],
        'RBE': 1.0,
        'alpha_LQ': 0.32,
        'beta_LQ': 0.032,
        'DSB_per_Gy': 35.0,
    },
    'Ac-225': {
        'type': 'alpha',
        'half_life_d': 9.920,
        'half_life_s': 9.920 * 24 * 3600,
        'decay_constant': np.log(2) / (9.920 * 24 * 3600),
        'E_alpha_total_MeV': 27.6,  # Total from decay chain
        'n_alphas_effective': 4.0,
        'alpha_range_um': 80,  # In tissue
        'gammas': [(440.5, 0.259)],
        'RBE': 5.0,  # Higher for alpha
        'alpha_LQ': 0.80,
        'beta_LQ': 0.010,
        'DSB_per_Gy': 175.0,  # Much higher for alpha
        'bio_half_life_h': {
            'tumor': 48,
            'kidney': 12,     # Faster clearance for alpha
            'liver': 12,
            'default': 12,
        },
    },
    'Ra-223': {
        'type': 'alpha',
        'half_life_d': 11.43,
        'half_life_s': 11.43 * 24 * 3600,
        'decay_constant': np.log(2) / (11.43 * 24 * 3600),
        'E_alpha_total_MeV': 26.4,
        'n_alphas_effective': 4.0,
        'alpha_range_um': 75,
        'gammas': [(269.5, 0.137)],
        'RBE': 5.0,
        'alpha_LQ': 0.75,
        'beta_LQ': 0.010,
        'DSB_per_Gy': 165.0,
    },
}

# =============================================================================
# ORGAN PARAMETERS (from ICRP 89)
# =============================================================================

ORGAN_PARAMETERS = {
    'liver': {'mass_g': 1800, 'density': 1.05, 'radiosensitive': True},
    'kidney': {'mass_g': 310, 'density': 1.05, 'radiosensitive': True},
    'spleen': {'mass_g': 180, 'density': 1.05, 'radiosensitive': True},
    'heart': {'mass_g': 330, 'density': 1.05, 'radiosensitive': False},
    'lung': {'mass_g': 1000, 'density': 0.26, 'radiosensitive': True},
    'bone': {'mass_g': 4000, 'density': 1.85, 'radiosensitive': True},
    'brain': {'mass_g': 1400, 'density': 1.04, 'radiosensitive': False},
    'bladder': {'mass_g': 50, 'density': 1.05, 'radiosensitive': True},
    'tumor': {'mass_g': None, 'density': 1.05, 'radiosensitive': True},  # Variable
    'duodenum': {'mass_g': 40, 'density': 1.04, 'radiosensitive': True},
}

# =============================================================================
# DOSE POINT KERNEL (DPK) - Precomputed for common isotopes
# =============================================================================

def generate_beta_dpk(E_max_keV: float, E_mean_keV: float, voxel_size_mm: float, 
                      kernel_size: int = 21) -> np.ndarray:
    """
    Generate dose point kernel for beta emitters using analytical model.
    
    Based on Berger/MIRD convolution kernel approach.
    
    The DPK describes dose rate per unit activity as function of distance.
    For clinical dosimetry, this is convolved with the activity distribution.
    
    Parameters
    ----------
    E_max_keV : float
        Maximum beta energy
    E_mean_keV : float
        Mean beta energy (from ICRP 107)
    voxel_size_mm : float
        Voxel size in mm
    kernel_size : int
        Kernel size (should be odd)
        
    Returns
    -------
    kernel : ndarray
        3D dose point kernel, normalized to sum=1
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    
    # Beta range in mm (empirical formula for water)
    # R_90 = 0.0546 * E_max^1.37 (mm) for E in keV
    R_90_mm = 0.0546 * (E_max_keV / 1000) ** 1.37  # Convert to MeV
    
    # Create 3D distance grid
    x = np.arange(kernel_size) - center
    y = np.arange(kernel_size) - center
    z = np.arange(kernel_size) - center
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2) * voxel_size_mm  # Distance in mm
    
    # Avoid division by zero at center
    R[center, center, center] = 0.01
    
    # Dose point kernel (scaled point kernel from MIRD Pamphlet 17)
    # D(r) ∝ (1/r²) × exp(-r/λ) × (1 + ar)
    # where λ is related to mean free path, a is fitting parameter
    
    lambda_mm = R_90_mm / 2.3  # Characteristic decay length
    a = 0.5 / R_90_mm
    
    kernel = (1.0 / (R**2 + 0.01)) * np.exp(-R / lambda_mm) * (1 + a * R)
    
    # Zero contribution beyond 3× R_90
    kernel[R > 3 * R_90_mm] = 0
    
    # Self-dose at center (very important for beta)
    # Approximately 50% of energy deposited locally
    kernel[center, center, center] = kernel[kernel > 0].max() * 5
    
    # Normalize to sum = 1 (energy conservation)
    kernel = kernel / kernel.sum()
    
    return kernel


def generate_alpha_dpk(voxel_size_mm: float, kernel_size: int = 5) -> np.ndarray:
    """
    Generate dose point kernel for alpha emitters.
    
    Alpha particles have very short range (~80 μm = 0.08 mm).
    At mm-scale voxels, nearly all energy is deposited locally.
    
    Returns nearly a delta function.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size, kernel_size))
    
    # Alpha range is ~0.08 mm, much smaller than typical PET voxel (4mm)
    # So essentially all energy stays in the source voxel
    kernel[center, center, center] = 0.95
    
    # Tiny crossfire to immediate neighbors (recoil nuclei)
    kernel[center-1:center+2, center-1:center+2, center-1:center+2] += 0.05/26
    kernel[center, center, center] = 0.95
    
    kernel = kernel / kernel.sum()
    return kernel


# =============================================================================
# PATIENT DATA LOADING
# =============================================================================

def find_patient_files(patient_folder: Path) -> Dict[str, Path]:
    """
    Automatically find PET, CT, and segmentation files in patient folder.
    
    Convention (nnU-Net style):
    - CT ends with _0000.nii.gz (first modality)
    - PET ends with _0001.nii.gz (second modality)
    - Segmentations are in subfolder with organ names
    """
    files = {}
    patient_folder = Path(patient_folder)
    
    # Find all .nii.gz files recursively
    nii_files = list(patient_folder.glob('*.nii.gz'))
    
    pet_files = []
    ct_files = []
    
    for f in nii_files:
        name = f.name.lower()
        # nnU-Net convention: _0000 = CT (first modality), _0001 = PET (second modality)
        if '_0001' in name or 'pet' in name or 'suv' in name:
            pet_files.append(f)
        elif '_0000' in name or 'ct' in name:
            ct_files.append(f)
    
    # Select first found PET and CT
    if pet_files:
        files['pet'] = pet_files[0]
    if ct_files:
        files['ct'] = ct_files[0]
    
    # Find segmentation subfolder
    seg_folders = []
    for subdir in patient_folder.iterdir():
        if subdir.is_dir():
            # Check if it contains organ segmentations
            organ_files = list(subdir.glob('*.nii.gz'))
            if any('liver' in f.name.lower() or 'tumor' in f.name.lower() 
                   for f in organ_files):
                seg_folders.append(subdir)
    
    if seg_folders:
        files['seg_folder'] = seg_folders[0]
    
    # Also check for segmentations in same folder and subfolders
    seg_files = {}
    search_folders = [patient_folder]
    if 'seg_folder' in files:
        search_folders.append(files['seg_folder'])
    for subdir in patient_folder.iterdir():
        if subdir.is_dir():
            search_folders.append(subdir)
    
    organ_names = ['liver', 'kidney', 'spleen', 'heart', 'lung', 'bone', 
                   'brain', 'bladder', 'tumor', 'duodenum', 'combined', 'totalseg']
    
    for folder in search_folders:
        for f in folder.glob('*.nii.gz'):
            fname = f.stem.lower().replace('.nii', '')
            for organ in organ_names:
                if organ in fname:
                    seg_files[organ] = f
    
    files['segmentations'] = seg_files
    
    return files


def load_pet_with_auto_ct(pet_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load PET scan and create synthetic CT (uniform soft tissue).
    
    Used when no CT is available.
    """
    print(f"\nLoading PET: {pet_path}")
    pet_img = nib.load(pet_path)
    pet_data = np.asarray(pet_img.dataobj, dtype=np.float32)
    pet_affine = pet_img.affine
    pet_zooms = pet_img.header.get_zooms()[:3]
    
    print(f"  Shape: {pet_data.shape}")
    print(f"  Voxel size: {pet_zooms[0]:.2f} x {pet_zooms[1]:.2f} x {pet_zooms[2]:.2f} mm")
    print(f"  SUV range: [{pet_data.min():.2f}, {pet_data.max():.2f}]")
    print(f"  Mean SUV (non-zero): {pet_data[pet_data > 0].mean():.2f}")
    
    print(f"\nUsing auto-CT mode: uniform soft tissue (HU=20)")
    ct_data = np.full(pet_data.shape, 20.0, dtype=np.float32)
    
    # Apply tissue-like values based on SUV (very approximate)
    # High SUV regions (likely tumor/organ) -> soft tissue
    # Very low SUV -> could be air/lung
    ct_data[pet_data < 0.1] = -500  # Possible air/lung regions
    
    metadata = {
        'pet_shape': pet_data.shape,
        'pet_voxel_mm': pet_zooms,
        'ct_voxel_mm': pet_zooms,  # Same as PET
        'pet_affine': pet_affine,
        'auto_ct': True,
    }
    
    return pet_data, ct_data, metadata


def load_pet_ct(pet_path: Path, ct_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load PET and CT scans with resampling to common grid.
    
    Returns
    -------
    pet_data : ndarray
        PET SUV values (normalized, non-negative)
    ct_data : ndarray  
        CT HU values, resampled to PET grid
    metadata : dict
        Voxel sizes, dimensions, etc.
    """
    print(f"\nLoading PET: {pet_path}")
    pet_img = nib.load(pet_path)
    pet_data = np.asarray(pet_img.dataobj, dtype=np.float32)
    pet_affine = pet_img.affine
    pet_zooms = pet_img.header.get_zooms()[:3]
    
    print(f"  Shape: {pet_data.shape}")
    print(f"  Voxel size: {pet_zooms[0]:.2f} x {pet_zooms[1]:.2f} x {pet_zooms[2]:.2f} mm")
    print(f"  Raw value range: [{pet_data.min():.2f}, {pet_data.max():.2f}]")
    
    # Handle negative values (can occur from scatter correction artifacts)
    if pet_data.min() < 0:
        print(f"  WARNING: Negative PET values detected - setting to 0")
        pet_data = np.maximum(pet_data, 0)
    
    # Normalize SUV if values seem too high (typical SUV max ~20-50)
    if pet_data.max() > 100:
        print(f"  NOTE: High PET values - may not be standard SUV units")
        # Check if this looks like raw counts vs SUV
        if pet_data.max() > 1000:
            # Likely raw counts or Bq/mL - normalize to SUV-like range
            p99 = np.percentile(pet_data[pet_data > 0], 99)
            if p99 > 100:
                print(f"  Normalizing PET values (99th percentile: {p99:.1f})")
                # Normalize so 99th percentile = 20 (typical high SUV)
                pet_data = pet_data * 20.0 / p99
    
    print(f"  Processed SUV range: [{pet_data.min():.2f}, {pet_data.max():.2f}]")
    if np.any(pet_data > 0):
        print(f"  Mean SUV (non-zero): {pet_data[pet_data > 0].mean():.2f}")
    
    print(f"\nLoading CT: {ct_path}")
    ct_img = nib.load(ct_path)
    ct_data = np.asarray(ct_img.dataobj, dtype=np.float32)
    ct_zooms = ct_img.header.get_zooms()[:3]
    
    print(f"  Shape: {ct_data.shape}")
    print(f"  Voxel size: {ct_zooms[0]:.2f} x {ct_zooms[1]:.2f} x {ct_zooms[2]:.2f} mm")
    print(f"  Raw value range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    
    # Check if this is standard HU or attenuation map
    if ct_data.min() >= 0 and ct_data.max() < 100:
        print(f"  NOTE: CT values suggest attenuation map (not HU)")
        print(f"  Converting to approximate HU scale...")
        # Attenuation coefficient to HU: HU = (μ - μ_water) / μ_water * 1000
        # Typical soft tissue μ ≈ 0.02 /mm at 140 keV
        # If max is ~46, this might be scaled attenuation
        # Map: 0 -> -1000 (air), 20 -> 0 (water), 46 -> +300 (soft tissue/bone)
        ct_data = (ct_data - 20) * 50  # Approximate conversion
        print(f"  Converted HU range: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    
    # Resample CT to PET grid if needed
    if ct_data.shape != pet_data.shape:
        print(f"\n  Resampling CT to PET grid...")
        zoom_factors = np.array(pet_data.shape) / np.array(ct_data.shape)
        ct_data = zoom(ct_data, zoom_factors, order=1)
        print(f"  Resampled CT shape: {ct_data.shape}")
    
    metadata = {
        'pet_shape': pet_data.shape,
        'pet_voxel_mm': pet_zooms,
        'ct_voxel_mm': ct_zooms,
        'pet_affine': pet_affine,
    }
    
    return pet_data, ct_data, metadata


def load_segmentations(seg_files: Dict[str, Path], reference_shape: Tuple) -> Dict[str, np.ndarray]:
    """
    Load organ segmentations and resample to reference grid.
    """
    segmentations = {}
    
    print(f"\nLoading segmentations:")
    for organ, path in seg_files.items():
        if organ in ['combined', 'totalseg']:
            continue  # Skip combined masks
            
        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj)
            
            # Handle multi-label masks
            if data.max() > 1 and organ not in ['totalseg_multilabel']:
                data = (data > 0).astype(np.float32)
            else:
                data = data.astype(np.float32)
            
            # Resample if needed
            if data.shape != reference_shape:
                zoom_factors = np.array(reference_shape) / np.array(data.shape)
                data = zoom(data, zoom_factors, order=0)  # Nearest neighbor for masks
            
            segmentations[organ] = data
            n_voxels = np.sum(data > 0)
            print(f"  {organ}: {n_voxels:,} voxels")
            
        except Exception as e:
            print(f"  {organ}: ERROR - {e}")
    
    return segmentations


# =============================================================================
# SUV TO ACTIVITY CONVERSION
# =============================================================================

def suv_to_activity_concentration(suv: np.ndarray, 
                                   injected_activity_MBq: float,
                                   body_weight_kg: float = 70.0,
                                   uptake_time_min: float = 60.0) -> np.ndarray:
    """
    Convert SUV to activity concentration (Bq/mL).
    
    SUV = (Activity concentration in tissue) / (Injected activity / Body weight)
    
    Activity_conc (Bq/mL) = SUV × (Injected_activity / Body_weight)
    
    Note: This assumes decay-corrected SUV.
    """
    # Activity concentration factor
    activity_factor = (injected_activity_MBq * 1e6) / (body_weight_kg * 1000)  # Bq/mL per unit SUV
    
    activity_conc = suv * activity_factor
    
    return activity_conc  # Bq/mL


def estimate_injected_activity_from_suv(suv_map: np.ndarray,
                                         voxel_volume_mL: float,
                                         body_weight_kg: float = 70.0,
                                         tracer_type: str = 'PSMA') -> float:
    """
    Estimate injected activity from total body SUV.
    
    For whole-body PET, sum(SUV × volume) ≈ injected activity (decay-corrected).
    """
    total_suv_volume = np.sum(suv_map) * voxel_volume_mL  # SUV × mL
    
    # Typical injection activities by tracer
    tracer_doses = {
        'FDG': 7.0,      # MBq/kg
        'PSMA': 3.0,     # MBq/kg (Ga-68 PSMA)
        'DOTATATE': 2.0, # MBq/kg
    }
    
    estimated_MBq_per_kg = tracer_doses.get(tracer_type, 5.0)
    estimated_activity_MBq = estimated_MBq_per_kg * body_weight_kg
    
    print(f"\n  Estimated imaging tracer activity: {estimated_activity_MBq:.0f} MBq")
    print(f"  (Based on typical {tracer_type} dose of {estimated_MBq_per_kg} MBq/kg)")
    
    return estimated_activity_MBq


# =============================================================================
# DOSIMETRY CALCULATIONS
# =============================================================================

def calculate_dose_dpk(activity_map: np.ndarray,
                       ct_data: np.ndarray,
                       isotope: str,
                       voxel_size_mm: Tuple[float, float, float],
                       segmentations: Dict[str, np.ndarray] = None,
                       time_integrated: bool = True) -> np.ndarray:
    """
    Calculate absorbed dose using Dose Point Kernel convolution.
    
    This is the standard clinical dosimetry approach for TRT.
    Uses EFFECTIVE half-life (physical + biological clearance).
    
    D(r) = ∫ A(r') × DPK(|r-r'|) × S(r) dr'
    
    Parameters
    ----------
    activity_map : ndarray
        Activity concentration in Bq/mL
    ct_data : ndarray
        CT HU values for density correction
    isotope : str
        Therapeutic isotope name
    voxel_size_mm : tuple
        Voxel dimensions in mm
    segmentations : dict, optional
        Organ segmentations for organ-specific biological half-life
    time_integrated : bool
        If True, integrate using effective half-life
        
    Returns
    -------
    dose_Gy : ndarray
        Absorbed dose in Gray
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    voxel_vol_cm3 = np.prod(voxel_size_mm) / 1000  # mm³ to cm³
    
    print(f"\n  Calculating dose for {isotope}...")
    print(f"  Voxel volume: {voxel_vol_cm3:.4f} cm³")
    
    # Generate dose point kernel
    if iso['type'] == 'beta':
        dpk = generate_beta_dpk(
            iso['E_max_beta_keV'],
            iso['E_mean_beta_keV'],
            voxel_size_mm[0],
            kernel_size=21
        )
        E_per_decay_keV = iso['E_mean_beta_keV']
    else:  # alpha
        dpk = generate_alpha_dpk(voxel_size_mm[0], kernel_size=5)
        E_per_decay_keV = iso['E_alpha_total_MeV'] * 1000
    
    print(f"  DPK kernel size: {dpk.shape}")
    print(f"  Energy per decay: {E_per_decay_keV:.1f} keV")
    
    # Convert CT HU to density
    density = np.clip((ct_data + 1000) / 1000, 0.1, 2.5)
    
    # Physical half-life
    T_phys_h = iso['half_life_d'] * 24  # hours
    
    # Calculate EFFECTIVE half-life for time integration
    # T_eff = (T_phys × T_bio) / (T_phys + T_bio)
    bio_half_lives = iso.get('bio_half_life_h', {'default': 24})
    default_bio_h = bio_half_lives.get('default', 24)
    
    # Create effective half-life map based on organ segmentations
    T_eff_h = np.full(activity_map.shape, 
                      (T_phys_h * default_bio_h) / (T_phys_h + default_bio_h))
    
    if segmentations:
        for organ, mask in segmentations.items():
            if organ in bio_half_lives:
                T_bio_h = bio_half_lives[organ]
            else:
                T_bio_h = default_bio_h
            T_eff_organ = (T_phys_h * T_bio_h) / (T_phys_h + T_bio_h)
            T_eff_h[mask > 0] = T_eff_organ
    
    # Mean effective half-life for reporting
    mean_T_eff = T_eff_h[activity_map > 0].mean() if np.any(activity_map > 0) else default_bio_h
    print(f"  Physical half-life: {T_phys_h:.1f} h ({iso['half_life_d']:.2f} d)")
    print(f"  Mean effective half-life: {mean_T_eff:.1f} h ({mean_T_eff/24:.2f} d)")
    
    # Time-integrated activity (cumulated activity)
    # Ã = A_0 × T_eff / ln(2)
    # Note: activity_map is in Bq/mL, need to convert to decays per voxel
    voxel_vol_mL = voxel_vol_cm3  # 1 cm³ = 1 mL
    
    if time_integrated:
        T_eff_s = T_eff_h * 3600  # Convert to seconds
        # Cumulated activity per voxel (decays/voxel)
        cumulated_activity = activity_map * voxel_vol_mL * T_eff_s / np.log(2)
    else:
        cumulated_activity = activity_map * voxel_vol_mL
    
    # Convolve activity with DPK
    print("  Convolving with dose point kernel...")
    dose_kernel_result = convolve(cumulated_activity, dpk, mode='constant')
    
    # Convert to absorbed dose
    E_per_decay_J = E_per_decay_keV * KEV_TO_J
    
    # Mass per voxel (kg) = volume (cm³) × density (g/cm³) / 1000
    mass_kg = (voxel_vol_cm3 * density) / 1000
    
    # Dose (Gy = J/kg) = (decays × J/decay) / kg
    dose_Gy = dose_kernel_result * E_per_decay_J / mass_kg
    
    # Ensure non-negative
    dose_Gy = np.maximum(dose_Gy, 0)
    
    print(f"  Dose range: [{dose_Gy.min():.4f}, {dose_Gy.max():.2f}] Gy")
    if np.any(dose_Gy > 0):
        print(f"  Mean dose (non-zero): {dose_Gy[dose_Gy > 0].mean():.4f} Gy")
    
    return dose_Gy


def calculate_organ_doses(dose_map: np.ndarray,
                          segmentations: Dict[str, np.ndarray],
                          voxel_size_mm: Tuple) -> pd.DataFrame:
    """
    Calculate mean and max dose per organ.
    """
    voxel_vol_cm3 = np.prod(voxel_size_mm) / 1000
    
    results = []
    
    for organ, mask in segmentations.items():
        if np.sum(mask) == 0:
            continue
            
        organ_mask = mask > 0
        organ_doses = dose_map[organ_mask]
        
        n_voxels = np.sum(organ_mask)
        volume_cm3 = n_voxels * voxel_vol_cm3
        
        # Get organ parameters
        params = ORGAN_PARAMETERS.get(organ, {'density': 1.05})
        mass_g = volume_cm3 * params.get('density', 1.05)
        
        results.append({
            'organ': organ,
            'volume_cm3': volume_cm3,
            'mass_g': mass_g,
            'mean_dose_Gy': float(organ_doses.mean()),
            'max_dose_Gy': float(organ_doses.max()),
            'min_dose_Gy': float(organ_doses.min()),
            'std_dose_Gy': float(organ_doses.std()),
            'D90_Gy': float(np.percentile(organ_doses, 10)),  # Dose covering 90% of volume
            'D50_Gy': float(np.percentile(organ_doses, 50)),  # Median dose
        })
    
    return pd.DataFrame(results)


# =============================================================================
# DNA DAMAGE AND CELL SURVIVAL
# =============================================================================

def calculate_dna_damage_organ(dose_Gy: float, isotope: str, 
                                cell_count: int = 1e9) -> Dict:
    """
    Calculate DNA damage for an organ based on absorbed dose.
    
    Uses literature-derived DSB yields and RBE factors.
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    
    # DSB per Gy per cell (already includes RBE)
    DSB_per_Gy = iso['DSB_per_Gy']
    SSB_per_Gy = DSB_per_Gy * 20  # SSB:DSB ratio ~20:1 for low-LET
    
    if iso['type'] == 'alpha':
        SSB_per_Gy = DSB_per_Gy * 5  # Lower ratio for high-LET
    
    total_DSB = dose_Gy * DSB_per_Gy * cell_count
    total_SSB = dose_Gy * SSB_per_Gy * cell_count
    
    # Complex DSB fraction (clustered damage)
    if iso['type'] == 'alpha':
        complex_fraction = 0.7  # High-LET creates more complex damage
    else:
        complex_fraction = 0.2  # Low-LET
    
    return {
        'total_DSB': total_DSB,
        'total_SSB': total_SSB,
        'DSB_per_cell': dose_Gy * DSB_per_Gy,
        'SSB_per_cell': dose_Gy * SSB_per_Gy,
        'complex_DSB_fraction': complex_fraction,
        'complex_DSB': total_DSB * complex_fraction,
    }


def calculate_survival_LQ(dose_Gy: float, alpha: float, beta: float) -> float:
    """
    Calculate cell survival using Linear-Quadratic model.
    
    S = exp(-αD - βD²)
    """
    return np.exp(-alpha * dose_Gy - beta * dose_Gy**2)


def calculate_organ_radiobiology(organ_doses: pd.DataFrame,
                                  isotope: str) -> pd.DataFrame:
    """
    Add DNA damage and survival calculations to organ dose table.
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    
    # Add columns
    organ_doses = organ_doses.copy()
    
    survival = []
    dsb_per_cell = []
    ssb_per_cell = []
    
    for _, row in organ_doses.iterrows():
        dose = row['mean_dose_Gy']
        
        # Cell survival
        S = calculate_survival_LQ(dose, iso['alpha_LQ'], iso['beta_LQ'])
        survival.append(S)
        
        # DNA damage
        damage = calculate_dna_damage_organ(dose, isotope, cell_count=1)
        dsb_per_cell.append(damage['DSB_per_cell'])
        ssb_per_cell.append(damage['SSB_per_cell'])
    
    organ_doses['survival_fraction'] = survival
    organ_doses['survival_percent'] = np.array(survival) * 100
    organ_doses['DSB_per_cell'] = dsb_per_cell
    organ_doses['SSB_per_cell'] = ssb_per_cell
    organ_doses['lethal_damage'] = organ_doses['survival_fraction'] < 0.01
    
    return organ_doses


# =============================================================================
# TREATMENT PLANNING ANALYSIS
# =============================================================================

def calculate_required_dose_for_kill(kill_fraction: float, alpha: float, beta: float) -> float:
    """
    Calculate dose required to kill a given fraction of cells using LQ model.
    
    From S = exp(-αD - βD²), solving for D:
    -ln(1-kill_fraction) = αD + βD²
    
    Using quadratic formula: D = (-α + sqrt(α² + 4β × target)) / (2β)
    where target = -ln(survival_fraction) = -ln(1 - kill_fraction)
    """
    survival = 1 - kill_fraction
    target = -np.log(survival)
    
    if beta > 0:
        # Quadratic formula
        discriminant = alpha**2 + 4 * beta * target
        D = (-alpha + np.sqrt(discriminant)) / (2 * beta)
    else:
        # Linear case (α dominates, e.g., alpha particles)
        D = target / alpha
    
    return D


def calculate_treatment_planning(organ_doses: pd.DataFrame,
                                  biodist: pd.DataFrame,
                                  isotope: str,
                                  current_activity_MBq: float) -> Dict:
    """
    Calculate optimal treatment planning for tumor eradication.
    
    Returns dose requirements for 50%, 95%, 99% tumor kill and 
    corresponding organ doses for treatment optimization.
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    alpha = iso['alpha_LQ']
    beta = iso['beta_LQ']
    
    # Target kill fractions
    kill_targets = {
        '50%': 0.50,
        '95%': 0.95,
        '99%': 0.99,
        '99.9%': 0.999,
    }
    
    # Calculate required tumor doses
    required_doses = {}
    for name, kill_frac in kill_targets.items():
        required_doses[name] = calculate_required_dose_for_kill(kill_frac, alpha, beta)
    
    # Get current tumor dose and organ doses
    tumor_row = organ_doses[organ_doses['organ'] == 'tumor']
    if len(tumor_row) == 0:
        return {'error': 'No tumor segmentation found'}
    
    current_tumor_dose = tumor_row['mean_dose_Gy'].values[0]
    
    # Calculate dose ratios (dose per MBq for each organ)
    dose_per_MBq = {}
    for _, row in organ_doses.iterrows():
        organ = row['organ']
        dose_per_MBq[organ] = row['mean_dose_Gy'] / current_activity_MBq
    
    # Calculate required activities and resulting organ doses
    treatment_options = {}
    
    for kill_name, required_tumor_dose in required_doses.items():
        # Required activity to achieve tumor dose
        if dose_per_MBq.get('tumor', 0) > 0:
            required_activity = required_tumor_dose / dose_per_MBq['tumor']
        else:
            required_activity = 0
        
        # Calculate organ doses at this activity
        organ_doses_at_activity = {}
        for organ, d_per_MBq in dose_per_MBq.items():
            organ_doses_at_activity[organ] = {
                'dose_Gy': d_per_MBq * required_activity,
                'survival': calculate_survival_LQ(d_per_MBq * required_activity, alpha, beta) * 100,
            }
        
        treatment_options[kill_name] = {
            'kill_fraction': kill_targets[kill_name],
            'required_tumor_dose_Gy': required_tumor_dose,
            'required_activity_MBq': required_activity,
            'required_activity_GBq': required_activity / 1000,
            'organ_doses': organ_doses_at_activity,
        }
    
    return {
        'dose_per_MBq': dose_per_MBq,
        'treatment_options': treatment_options,
        'current_activity_MBq': current_activity_MBq,
        'current_tumor_dose_Gy': current_tumor_dose,
    }


def calculate_fractionation_strategy(treatment_planning: Dict,
                                      isotope: str,
                                      max_kidney_dose_per_cycle: float = 5.0,
                                      max_bone_marrow_dose_per_cycle: float = 2.0) -> Dict:
    """
    Calculate optimal fractionation strategy.
    
    Considers organ-at-risk (OAR) dose limits per cycle to determine
    optimal number of treatment cycles.
    
    Clinical limits (EANM/SNMMI guidelines):
    - Kidney: 23 Gy cumulative, ~5 Gy per cycle
    - Bone marrow: 2 Gy cumulative, ~0.5 Gy per cycle
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    dose_per_MBq = treatment_planning['dose_per_MBq']
    
    # Organ dose limits (cumulative and per cycle)
    organ_limits = {
        'kidney': {'cumulative_Gy': 23.0, 'per_cycle_Gy': max_kidney_dose_per_cycle},
        'liver': {'cumulative_Gy': 30.0, 'per_cycle_Gy': 10.0},
        'bone_marrow': {'cumulative_Gy': 2.0, 'per_cycle_Gy': 0.5},
        'bone': {'cumulative_Gy': 10.0, 'per_cycle_Gy': 3.0},
        'spleen': {'cumulative_Gy': 20.0, 'per_cycle_Gy': 7.0},
    }
    
    # Calculate maximum activity per cycle based on OAR limits
    max_activity_per_cycle = float('inf')
    limiting_organ = None
    
    for organ, limits in organ_limits.items():
        if organ in dose_per_MBq and dose_per_MBq[organ] > 0:
            max_activity = limits['per_cycle_Gy'] / dose_per_MBq[organ]
            if max_activity < max_activity_per_cycle:
                max_activity_per_cycle = max_activity
                limiting_organ = organ
    
    # Calculate tumor dose per cycle at max safe activity
    tumor_dose_per_cycle = dose_per_MBq.get('tumor', 0) * max_activity_per_cycle
    
    # Calculate cycles needed for each kill target
    fractionation = {}
    alpha = iso['alpha_LQ']
    beta = iso['beta_LQ']
    
    for kill_name, options in treatment_planning['treatment_options'].items():
        required_tumor_dose = options['required_tumor_dose_Gy']
        
        # Number of cycles needed
        n_cycles = np.ceil(required_tumor_dose / tumor_dose_per_cycle) if tumor_dose_per_cycle > 0 else float('inf')
        
        # Cumulative organ doses
        cumulative_organ_doses = {}
        exceeds_limit = False
        
        for organ, d_per_MBq in dose_per_MBq.items():
            cum_dose = d_per_MBq * max_activity_per_cycle * n_cycles
            cum_survival = calculate_survival_LQ(cum_dose, alpha, beta) * 100
            
            limit_info = organ_limits.get(organ, {'cumulative_Gy': float('inf')})
            exceeds = cum_dose > limit_info.get('cumulative_Gy', float('inf'))
            
            cumulative_organ_doses[organ] = {
                'cumulative_dose_Gy': cum_dose,
                'survival_percent': cum_survival,
                'exceeds_limit': exceeds,
            }
            if exceeds:
                exceeds_limit = True
        
        # Tumor survival after treatment
        actual_tumor_dose = tumor_dose_per_cycle * n_cycles
        tumor_survival = calculate_survival_LQ(actual_tumor_dose, alpha, beta) * 100
        
        fractionation[kill_name] = {
            'n_cycles': int(n_cycles),
            'activity_per_cycle_MBq': max_activity_per_cycle,
            'activity_per_cycle_GBq': max_activity_per_cycle / 1000,
            'tumor_dose_per_cycle_Gy': tumor_dose_per_cycle,
            'total_tumor_dose_Gy': actual_tumor_dose,
            'tumor_survival_percent': tumor_survival,
            'cumulative_organ_doses': cumulative_organ_doses,
            'exceeds_organ_limits': exceeds_limit,
            'limiting_organ': limiting_organ,
        }
    
    return {
        'max_activity_per_cycle_MBq': max_activity_per_cycle,
        'max_activity_per_cycle_GBq': max_activity_per_cycle / 1000,
        'limiting_organ': limiting_organ,
        'tumor_dose_per_cycle_Gy': tumor_dose_per_cycle,
        'fractionation_options': fractionation,
        'organ_limits_used': organ_limits,
    }


def generate_treatment_recommendation(fractionation: Dict, isotope: str) -> str:
    """
    Generate clinical treatment recommendations.
    """
    iso_name = isotope.replace('-', '')
    limiting_organ = fractionation['limiting_organ']
    max_activity = fractionation['max_activity_per_cycle_MBq']
    tumor_dose_per_cycle = fractionation['tumor_dose_per_cycle_Gy']
    
    rec = []
    rec.append("=" * 70)
    rec.append("TREATMENT PLANNING RECOMMENDATIONS")
    rec.append("=" * 70)
    rec.append(f"\nIsotope: {isotope}")
    rec.append(f"Limiting organ: {limiting_organ}")
    rec.append(f"Maximum safe activity per cycle: {max_activity:.0f} MBq ({max_activity/1000:.2f} GBq)")
    rec.append(f"Tumor dose per cycle: {tumor_dose_per_cycle:.2f} Gy")
    
    rec.append("\n--- Treatment Options ---\n")
    
    for target, data in fractionation['fractionation_options'].items():
        status = "RECOMMENDED" if not data['exceeds_organ_limits'] else "EXCEEDS LIMITS"
        rec.append(f"Target: {target} tumor kill")
        rec.append(f"  Cycles required: {data['n_cycles']}")
        rec.append(f"  Activity per cycle: {data['activity_per_cycle_GBq']:.2f} GBq")
        rec.append(f"  Total tumor dose: {data['total_tumor_dose_Gy']:.1f} Gy")
        rec.append(f"  Final tumor survival: {data['tumor_survival_percent']:.2f}%")
        rec.append(f"  Status: {status}")
        
        # Key organ doses
        critical_organs = ['kidney', 'liver', 'bone']
        for organ in critical_organs:
            if organ in data['cumulative_organ_doses']:
                od = data['cumulative_organ_doses'][organ]
                limit_status = " (EXCEEDS LIMIT)" if od['exceeds_limit'] else ""
                rec.append(f"    {organ}: {od['cumulative_dose_Gy']:.1f} Gy, {od['survival_percent']:.1f}% survival{limit_status}")
        rec.append("")
    
    # Clinical recommendation
    rec.append("--- Clinical Recommendation ---\n")
    
    # Find safest effective option
    safe_options = [(k, v) for k, v in fractionation['fractionation_options'].items() 
                    if not v['exceeds_organ_limits']]
    
    if safe_options:
        best = max(safe_options, key=lambda x: float(x[0].replace('%', '').replace('.', '')))
        rec.append(f"RECOMMENDED: {best[0]} tumor eradication")
        rec.append(f"  - {best[1]['n_cycles']} cycles of {best[1]['activity_per_cycle_GBq']:.2f} GBq {isotope}")
        rec.append(f"  - Total tumor dose: {best[1]['total_tumor_dose_Gy']:.1f} Gy")
        rec.append(f"  - Expected tumor survival: {best[1]['tumor_survival_percent']:.3f}%")
    else:
        rec.append("WARNING: All treatment options exceed organ dose limits.")
        rec.append("Consider dose reduction or alternative therapies.")
    
    return "\n".join(rec)


# =============================================================================
# BIODISTRIBUTION ANALYSIS
# =============================================================================

def calculate_biodistribution(pet_data: np.ndarray,
                               segmentations: Dict[str, np.ndarray],
                               voxel_size_mm: Tuple) -> pd.DataFrame:
    """
    Calculate SUV-based biodistribution.
    """
    voxel_vol_cm3 = np.prod(voxel_size_mm) / 1000
    
    results = []
    
    # Total body SUV
    total_suv = np.sum(pet_data)
    
    for organ, mask in segmentations.items():
        if np.sum(mask) == 0:
            continue
            
        organ_mask = mask > 0
        organ_suv = pet_data[organ_mask]
        
        n_voxels = np.sum(organ_mask)
        volume_cm3 = n_voxels * voxel_vol_cm3
        
        results.append({
            'organ': organ,
            'volume_cm3': volume_cm3,
            'mean_SUV': float(organ_suv.mean()),
            'max_SUV': float(organ_suv.max()),
            'total_SUV': float(organ_suv.sum()),
            'percent_uptake': float(organ_suv.sum() / total_suv * 100),
            'SUV_CV': float(organ_suv.std() / organ_suv.mean() * 100) if organ_suv.mean() > 0 else 0,
        })
    
    df = pd.DataFrame(results)
    
    # Calculate ratios
    if 'tumor' in df['organ'].values and 'liver' in df['organ'].values:
        tumor_suv = df[df['organ'] == 'tumor']['mean_SUV'].values[0]
        liver_suv = df[df['organ'] == 'liver']['mean_SUV'].values[0]
        df['tumor_to_liver'] = df['mean_SUV'] / liver_suv if liver_suv > 0 else 0
        df.loc[df['organ'] == 'tumor', 'T_L_ratio'] = tumor_suv / liver_suv if liver_suv > 0 else 0
    
    if 'tumor' in df['organ'].values and 'heart' in df['organ'].values:
        tumor_suv = df[df['organ'] == 'tumor']['mean_SUV'].values[0]
        heart_suv = df[df['organ'] == 'heart']['mean_SUV'].values[0]
        df.loc[df['organ'] == 'tumor', 'T_H_ratio'] = tumor_suv / heart_suv if heart_suv > 0 else 0
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_dosimetry_figures(pet_data: np.ndarray,
                              dose_map: np.ndarray,
                              segmentations: Dict[str, np.ndarray],
                              organ_doses: pd.DataFrame,
                              biodist: pd.DataFrame,
                              output_dir: Path,
                              isotope: str):
    """
    Create publication-quality dosimetry figures.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Style settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })
    
    # Color scheme
    organ_colors = {
        'tumor': '#E63946',
        'liver': '#457B9D',
        'kidney': '#2A9D8F',
        'heart': '#E9C46A',
        'lung': '#A8DADC',
        'spleen': '#F4A261',
        'bone': '#264653',
        'brain': '#9B2335',
        'bladder': '#1D3557',
    }
    
    # =========================================================================
    # Figure 1: Overview - PET, CT, Dose MIPs
    # =========================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # PET MIPs
    mid_z = pet_data.shape[2] // 2
    mid_y = pet_data.shape[1] // 2
    mid_x = pet_data.shape[0] // 2
    
    ax = axes[0, 0]
    im = ax.imshow(np.max(pet_data, axis=2).T, cmap='hot', origin='lower')
    ax.set_title('PET SUV - Axial MIP', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='SUV')
    
    ax = axes[0, 1]
    im = ax.imshow(np.max(pet_data, axis=0).T, cmap='hot', origin='lower')
    ax.set_title('PET SUV - Coronal MIP', fontweight='bold')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='SUV')
    
    ax = axes[0, 2]
    im = ax.imshow(np.max(pet_data, axis=1).T, cmap='hot', origin='lower')
    ax.set_title('PET SUV - Sagittal MIP', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='SUV')
    
    # Dose MIPs
    dose_display = np.log10(dose_map + 1e-6)
    
    ax = axes[1, 0]
    im = ax.imshow(np.max(dose_display, axis=2).T, cmap='inferno', origin='lower')
    ax.set_title(f'{isotope} Dose - Axial MIP', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='log₁₀(Dose [Gy])')
    
    ax = axes[1, 1]
    im = ax.imshow(np.max(dose_display, axis=0).T, cmap='inferno', origin='lower')
    ax.set_title(f'{isotope} Dose - Coronal MIP', fontweight='bold')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='log₁₀(Dose [Gy])')
    
    ax = axes[1, 2]
    im = ax.imshow(np.max(dose_display, axis=1).T, cmap='inferno', origin='lower')
    ax.set_title(f'{isotope} Dose - Sagittal MIP', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='log₁₀(Dose [Gy])')
    
    plt.suptitle(f'CellRad-MC Patient Dosimetry: {isotope}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'Figure1_PET_Dose_Overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Figure 2: Organ Dose Bar Chart
    # =========================================================================
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by dose
    df = organ_doses.sort_values('mean_dose_Gy', ascending=True)
    
    colors = [organ_colors.get(o, '#888888') for o in df['organ']]
    
    ax = axes[0]
    bars = ax.barh(df['organ'], df['mean_dose_Gy'], color=colors, edgecolor='black')
    ax.set_xlabel('Mean Absorbed Dose (Gy)', fontweight='bold')
    ax.set_title('Organ Absorbed Doses', fontweight='bold')
    ax.set_xlim(0, df['mean_dose_Gy'].max() * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, df['mean_dose_Gy']):
        ax.text(bar.get_width() + 0.01 * df['mean_dose_Gy'].max(), 
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Kidney tolerance (2 Gy)')
    ax.legend()
    
    # Survival bar chart
    ax = axes[1]
    bars = ax.barh(df['organ'], df['survival_percent'], color=colors, edgecolor='black')
    ax.set_xlabel('Cell Survival (%)', fontweight='bold')
    ax.set_title('Predicted Cell Survival (LQ Model)', fontweight='bold')
    ax.set_xlim(0, 105)
    
    for bar, val in zip(bars, df['survival_percent']):
        ax.text(bar.get_width() + 1, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    
    ax.axvline(x=37, color='orange', linestyle='--', alpha=0.7, label='D₃₇ (1/e survival)')
    ax.legend()
    
    plt.suptitle(f'{isotope} Patient Dosimetry Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'Figure2_Organ_Doses.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Figure 3: Biodistribution
    # =========================================================================
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = biodist.sort_values('percent_uptake', ascending=True)
    colors = [organ_colors.get(o, '#888888') for o in df['organ']]
    
    ax = axes[0]
    bars = ax.barh(df['organ'], df['percent_uptake'], color=colors, edgecolor='black')
    ax.set_xlabel('% Injected Dose', fontweight='bold')
    ax.set_title('Biodistribution (% ID)', fontweight='bold')
    
    for bar, val in zip(bars, df['percent_uptake']):
        ax.text(bar.get_width() + 0.1, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    
    ax = axes[1]
    bars = ax.barh(df['organ'], df['mean_SUV'], color=colors, edgecolor='black')
    ax.set_xlabel('Mean SUV', fontweight='bold')
    ax.set_title('Mean SUV by Organ', fontweight='bold')
    
    for bar, val in zip(bars, df['mean_SUV']):
        ax.text(bar.get_width() + 0.05, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)
    
    plt.suptitle('Biodistribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'Figure3_Biodistribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Figure 4: DNA Damage Analysis
    # =========================================================================
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df = organ_doses.sort_values('DSB_per_cell', ascending=True)
    colors = [organ_colors.get(o, '#888888') for o in df['organ']]
    
    ax = axes[0, 0]
    bars = ax.barh(df['organ'], df['DSB_per_cell'], color=colors, edgecolor='black')
    ax.set_xlabel('DSBs per Cell', fontweight='bold')
    ax.set_title('Double Strand Breaks per Cell', fontweight='bold')
    ax.axvline(x=35, color='red', linestyle='--', alpha=0.7, label='Lethal threshold (~35 DSB)')
    ax.legend()
    
    ax = axes[0, 1]
    bars = ax.barh(df['organ'], df['SSB_per_cell'], color=colors, edgecolor='black')
    ax.set_xlabel('SSBs per Cell', fontweight='bold')
    ax.set_title('Single Strand Breaks per Cell', fontweight='bold')
    
    # Dose-response curve
    ax = axes[1, 0]
    doses = np.linspace(0, organ_doses['max_dose_Gy'].max() * 1.5, 100)
    iso_data = THERAPEUTIC_ISOTOPES[isotope]
    survival = calculate_survival_LQ(doses, iso_data['alpha_LQ'], iso_data['beta_LQ'])
    
    ax.plot(doses, survival * 100, 'b-', linewidth=2, label=f'{isotope} LQ Model')
    ax.axhline(y=37, color='orange', linestyle='--', label='D₃₇')
    ax.axhline(y=10, color='red', linestyle='--', label='10% survival')
    
    # Plot organ points
    for _, row in organ_doses.iterrows():
        color = organ_colors.get(row['organ'], '#888888')
        ax.scatter(row['mean_dose_Gy'], row['survival_percent'], 
                   s=100, c=[color], edgecolors='black', zorder=5,
                   label=row['organ'])
    
    ax.set_xlabel('Dose (Gy)', fontweight='bold')
    ax.set_ylabel('Survival (%)', fontweight='bold')
    ax.set_title('Dose-Response Curve', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Tumor vs Normal tissue comparison
    ax = axes[1, 1]
    if 'tumor' in organ_doses['organ'].values:
        tumor_dose = organ_doses[organ_doses['organ'] == 'tumor']['mean_dose_Gy'].values[0]
        normal_organs = organ_doses[organ_doses['organ'] != 'tumor']
        max_normal_dose = normal_organs['mean_dose_Gy'].max()
        
        therapeutic_ratio = tumor_dose / max_normal_dose if max_normal_dose > 0 else 0
        
        ax.bar(['Tumor', 'Max Normal\nTissue'], [tumor_dose, max_normal_dose],
               color=['#E63946', '#457B9D'], edgecolor='black')
        ax.set_ylabel('Absorbed Dose (Gy)', fontweight='bold')
        ax.set_title(f'Therapeutic Ratio: {therapeutic_ratio:.2f}', fontweight='bold')
        
        ax.text(0, tumor_dose + 0.05, f'{tumor_dose:.3f} Gy', ha='center', fontweight='bold')
        ax.text(1, max_normal_dose + 0.05, f'{max_normal_dose:.3f} Gy', ha='center', fontweight='bold')
    
    plt.suptitle(f'{isotope} Radiobiology Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'Figure4_DNA_Damage.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved figures to {output_dir}")


def create_treatment_planning_figure(treatment_planning: Dict,
                                      fractionation: Dict,
                                      isotope: str,
                                      output_dir: Path):
    """
    Create treatment planning visualization.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color scheme
    colors = {
        '50%': '#2ECC71',
        '95%': '#F39C12', 
        '99%': '#E74C3C',
        '99.9%': '#8E44AD',
    }
    
    organ_colors = {
        'tumor': '#E74C3C',
        'kidney': '#3498DB',
        'liver': '#27AE60',
        'spleen': '#F39C12',
        'bone': '#9B59B6',
        'lung': '#1ABC9C',
    }
    
    # Panel A: Required doses for each kill target
    ax = axes[0, 0]
    targets = list(treatment_planning['treatment_options'].keys())
    tumor_doses = [treatment_planning['treatment_options'][t]['required_tumor_dose_Gy'] for t in targets]
    activities = [treatment_planning['treatment_options'][t]['required_activity_GBq'] for t in targets]
    
    x = np.arange(len(targets))
    bars = ax.bar(x, tumor_doses, color=[colors.get(t, '#666') for t in targets], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel('Required Tumor Dose (Gy)', fontweight='bold')
    ax.set_xlabel('Tumor Kill Target', fontweight='bold')
    ax.set_title('A. Required Doses for Tumor Eradication', fontweight='bold', fontsize=12)
    
    # Add activity labels
    for bar, act in zip(bars, activities):
        height = bar.get_height()
        ax.annotate(f'{act:.1f} GBq',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Organ doses at different treatment levels
    ax = axes[0, 1]
    
    organs = ['tumor', 'kidney', 'liver', 'spleen']
    targets_to_show = ['50%', '95%', '99%']
    
    x = np.arange(len(organs))
    width = 0.25
    
    for i, target in enumerate(targets_to_show):
        doses = []
        for organ in organs:
            if organ in treatment_planning['treatment_options'][target]['organ_doses']:
                doses.append(treatment_planning['treatment_options'][target]['organ_doses'][organ]['dose_Gy'])
            else:
                doses.append(0)
        ax.bar(x + i*width - width, doses, width, label=f'{target} kill', 
               color=colors.get(target, '#666'), alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in organs])
    ax.set_ylabel('Absorbed Dose (Gy)', fontweight='bold')
    ax.set_title('B. Organ Doses at Treatment Levels', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add kidney limit line
    ax.axhline(y=23, color='red', linestyle='--', alpha=0.5, label='Kidney limit (23 Gy)')
    
    # Panel C: Fractionation scheme
    ax = axes[1, 0]
    
    targets = list(fractionation['fractionation_options'].keys())
    n_cycles = [fractionation['fractionation_options'][t]['n_cycles'] for t in targets]
    total_doses = [fractionation['fractionation_options'][t]['total_tumor_dose_Gy'] for t in targets]
    
    x = np.arange(len(targets))
    bars = ax.bar(x, n_cycles, color=[colors.get(t, '#666') for t in targets], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel('Number of Treatment Cycles', fontweight='bold')
    ax.set_xlabel('Tumor Kill Target', fontweight='bold')
    ax.set_title(f'C. Treatment Cycles Required\n(Max {fractionation["max_activity_per_cycle_GBq"]:.2f} GBq/cycle)', 
                 fontweight='bold', fontsize=12)
    
    # Add total dose labels
    for bar, dose in zip(bars, total_doses):
        height = bar.get_height()
        ax.annotate(f'{dose:.0f} Gy total',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Tumor survival curve with treatment points
    ax = axes[1, 1]
    
    iso = THERAPEUTIC_ISOTOPES[isotope]
    alpha, beta = iso['alpha_LQ'], iso['beta_LQ']
    
    doses = np.linspace(0, 20, 100)
    survival = calculate_survival_LQ(doses, alpha, beta) * 100
    
    ax.semilogy(doses, survival, 'b-', linewidth=2, label='LQ Model')
    
    # Mark treatment points
    for target in targets:
        req_dose = treatment_planning['treatment_options'][target]['required_tumor_dose_Gy']
        kill_frac = treatment_planning['treatment_options'][target]['kill_fraction']
        surv = (1 - kill_frac) * 100
        ax.plot(req_dose, surv, 'o', markersize=12, color=colors.get(target, '#666'),
                label=f'{target} kill', zorder=5)
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='-.', alpha=0.5)
    
    ax.set_xlabel('Absorbed Dose (Gy)', fontweight='bold')
    ax.set_ylabel('Tumor Cell Survival (%)', fontweight='bold')
    ax.set_title('D. Dose-Response Curve with Treatment Targets', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_ylim(0.01, 110)
    ax.set_xlim(0, max(20, max(tumor_doses) * 1.1))
    ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'{isotope} Treatment Planning Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / 'Figure5_Treatment_Planning.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure5_Treatment_Planning.png")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(organ_doses: pd.DataFrame,
                    biodist: pd.DataFrame,
                    isotope: str,
                    metadata: Dict,
                    output_dir: Path):
    """
    Generate comprehensive Markdown report.
    """
    iso = THERAPEUTIC_ISOTOPES[isotope]
    
    # Calculate key metrics
    if 'tumor' in organ_doses['organ'].values:
        tumor_row = organ_doses[organ_doses['organ'] == 'tumor'].iloc[0]
        tumor_dose = tumor_row['mean_dose_Gy']
        tumor_survival = tumor_row['survival_percent']
        tumor_dsb = tumor_row['DSB_per_cell']
    else:
        tumor_dose = tumor_survival = tumor_dsb = None
    
    # Find critical organ (highest dose excluding tumor)
    normal_organs = organ_doses[organ_doses['organ'] != 'tumor']
    if len(normal_organs) > 0:
        critical_organ = normal_organs.loc[normal_organs['mean_dose_Gy'].idxmax()]
    else:
        critical_organ = None
    
    report = f"""# CellRad-MC Patient Dosimetry Report

## Isotope: {isotope}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Isotope Properties

| Parameter | Value |
|-----------|-------|
| Radiation Type | {iso['type'].upper()} |
| Half-life | {iso['half_life_d']:.2f} days |
| Mean Energy | {iso.get('E_mean_beta_keV', iso.get('E_alpha_total_MeV', 0)*1000):.1f} keV |
| RBE | {iso['RBE']:.1f} |
| α (LQ) | {iso['alpha_LQ']:.3f} Gy⁻¹ |
| β (LQ) | {iso['beta_LQ']:.4f} Gy⁻² |

---

## 2. Biodistribution Summary

| Organ | Volume (cm³) | Mean SUV | % ID |
|-------|-------------|----------|------|
"""
    
    for _, row in biodist.sort_values('percent_uptake', ascending=False).iterrows():
        report += f"| {row['organ']} | {row['volume_cm3']:.1f} | {row['mean_SUV']:.2f} | {row['percent_uptake']:.1f}% |\n"
    
    report += f"""
---

## 3. Absorbed Dose by Organ

| Organ | Mean Dose (Gy) | Max Dose (Gy) | D₅₀ (Gy) | D₉₀ (Gy) |
|-------|---------------|---------------|----------|----------|
"""
    
    for _, row in organ_doses.sort_values('mean_dose_Gy', ascending=False).iterrows():
        report += f"| {row['organ']} | {row['mean_dose_Gy']:.4f} | {row['max_dose_Gy']:.4f} | {row['D50_Gy']:.4f} | {row['D90_Gy']:.4f} |\n"
    
    report += f"""
---

## 4. Radiobiological Analysis

### 4.1 Cell Survival Prediction (LQ Model)

| Organ | Mean Dose (Gy) | Survival (%) | DSB/Cell | SSB/Cell |
|-------|---------------|--------------|----------|----------|
"""
    
    for _, row in organ_doses.sort_values('survival_percent', ascending=True).iterrows():
        report += f"| {row['organ']} | {row['mean_dose_Gy']:.4f} | {row['survival_percent']:.1f}% | {row['DSB_per_cell']:.1f} | {row['SSB_per_cell']:.0f} |\n"
    
    if tumor_dose is not None and critical_organ is not None:
        therapeutic_ratio = tumor_dose / critical_organ['mean_dose_Gy'] if critical_organ['mean_dose_Gy'] > 0 else float('inf')
        
        report += f"""
### 4.2 Therapeutic Window Analysis

| Metric | Value |
|--------|-------|
| Tumor Dose | {tumor_dose:.4f} Gy |
| Tumor Survival | {tumor_survival:.1f}% |
| Tumor DSB/Cell | {tumor_dsb:.1f} |
| Critical Organ | {critical_organ['organ']} |
| Critical Organ Dose | {critical_organ['mean_dose_Gy']:.4f} Gy |
| Therapeutic Ratio (T/N) | {therapeutic_ratio:.2f} |
"""
    
    report += f"""
---

## 5. DNA Damage Summary

### Interpretation:

- **DSB per cell > 35**: Likely cell death (beyond repair capacity)
- **DSB per cell 10-35**: Significant damage, reduced survival
- **DSB per cell < 10**: Sublethal, likely repair

### Clinical Implications:

"""
    
    for _, row in organ_doses.iterrows():
        if row['DSB_per_cell'] > 35:
            status = "⚠️ HIGH - Significant cell killing expected"
        elif row['DSB_per_cell'] > 10:
            status = "⚡ MODERATE - Some cell death expected"
        else:
            status = "✓ LOW - Sublethal damage"
        report += f"- **{row['organ']}**: {row['DSB_per_cell']:.1f} DSB/cell - {status}\n"
    
    report += f"""
---

## 6. Methodology

### Dosimetry Approach:
- **Dose Point Kernel (DPK) convolution** for beta emitters
- **Local energy deposition** for alpha emitters (range << voxel size)
- **CT-based density correction** for tissue heterogeneity

### Physics Validation:
- ICRP 107 isotope data
- MIRD Pamphlet 17 DPK methodology
- Linear-Quadratic (LQ) model for cell survival (van Leeuwen et al. 2018)

### DNA Damage Model:
- DSB yield: {iso['DSB_per_Gy']:.1f} per Gy per cell (RBE-corrected)
- Complex damage fraction: {"70%" if iso['type'] == 'alpha' else "20%"}

---

## 7. Limitations

1. **PET Resolution**: Activity distribution limited by PET spatial resolution (~4mm)
2. **SUV Calibration**: Assumes standard SUV normalization and decay correction
3. **Tissue Heterogeneity**: Simplified CT-to-density conversion
4. **Biological Variability**: LQ parameters are population averages

---

## 8. References

1. ICRP Publication 107 (2008). Nuclear Decay Data for Dosimetric Calculations.
2. MIRD Pamphlet 17 (1999). The Dosimetry of Nonuniform Activity Distributions.
3. van Leeuwen CM et al. (2018). The alfa and beta of tumours. Eur J Cancer 98:34-48.
4. Sgouros G et al. (2010). MIRD Pamphlet 22: Radiobiology and dosimetry of α-particle emitters.

---

*Generated by CellRad-MC Patient Dosimetry Module*
"""
    
    with open(output_dir / 'DOSIMETRY_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n  Saved report to {output_dir / 'DOSIMETRY_REPORT.md'}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_patient_dosimetry(args):
    """
    Run complete patient dosimetry analysis.
    """
    print("\n" + "=" * 70)
    print("CellRad-MC Patient Dosimetry")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Isotope: {args.isotope}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find patient files
    if args.patient_folder:
        patient_folder = Path(args.patient_folder)
        files = find_patient_files(patient_folder)
        
        if 'pet' not in files:
            # Try subdirectories
            for subdir in patient_folder.iterdir():
                if subdir.is_dir():
                    sub_files = find_patient_files(subdir)
                    if 'pet' in sub_files:
                        files = sub_files
                        break
        
        pet_path = files.get('pet')
        ct_path = files.get('ct')
        seg_files = files.get('segmentations', {})
        
        print(f"\nFound files:")
        print(f"  PET: {pet_path}")
        print(f"  CT: {ct_path}")
        print(f"  Segmentations: {list(seg_files.keys())}")
        
    else:
        pet_path = Path(args.pet)
        ct_path = Path(args.ct)
        seg_files = {}
        
        if args.segmentation:
            seg_folder = Path(args.segmentation)
            for f in seg_folder.glob('*.nii.gz'):
                organ = f.stem.lower().replace('.nii', '')
                seg_files[organ] = f
    
    if pet_path is None:
        raise ValueError("Could not find PET scan")
    
    # Extract metadata from NIfTI files
    print("\n--- Checking for embedded metadata ---")
    pet_metadata = extract_nifti_metadata(pet_path)
    
    if 'injected_dose_MBq' in pet_metadata:
        print(f"  Found injected dose in PET metadata: {pet_metadata['injected_dose_MBq']:.1f} MBq")
        if not hasattr(args, 'activity_MBq') or args.activity_MBq is None:
            args.activity_MBq = pet_metadata['injected_dose_MBq']
            print(f"  Using metadata dose for calculations")
    
    if 'patient_weight_kg' in pet_metadata:
        print(f"  Found patient weight: {pet_metadata['patient_weight_kg']:.1f} kg")
        if not hasattr(args, 'body_weight') or args.body_weight == 70:
            args.body_weight = pet_metadata['patient_weight_kg']
    
    if 'half_life_s' in pet_metadata:
        print(f"  Found radionuclide half-life: {pet_metadata['half_life_s']/3600:.1f} hours")
    
    if 'description' in pet_metadata:
        print(f"  File description: {pet_metadata['description']}")
    
    # Handle missing CT
    use_auto_ct = args.auto_ct or ct_path is None
    
    if ct_path is None and not args.auto_ct:
        print("\n  WARNING: CT scan not found. Using --auto-ct mode (uniform soft tissue).")
        use_auto_ct = True
    
    # Load imaging data
    print("\n" + "=" * 60)
    print("Loading Imaging Data")
    print("=" * 60)
    
    if use_auto_ct:
        pet_data, ct_data, metadata = load_pet_with_auto_ct(pet_path)
    else:
        pet_data, ct_data, metadata = load_pet_ct(pet_path, ct_path)
    
    # Load segmentations
    segmentations = load_segmentations(seg_files, pet_data.shape)
    
    if len(segmentations) == 0:
        print("\n  WARNING: No segmentations found - creating whole-body mask")
        segmentations['whole_body'] = (pet_data > 0).astype(np.float32)
    
    # Calculate biodistribution
    print("\n" + "=" * 60)
    print("Biodistribution Analysis")
    print("=" * 60)
    
    voxel_size_mm = metadata['pet_voxel_mm']
    biodist = calculate_biodistribution(pet_data, segmentations, voxel_size_mm)
    print(biodist.to_string())
    
    # Determine tracer type from filename
    pet_name = str(pet_path).lower()
    if 'psma' in pet_name:
        tracer_type = 'PSMA'
    elif 'dotatate' in pet_name or 'dota' in pet_name:
        tracer_type = 'DOTATATE'
    else:
        tracer_type = 'FDG'
    
    print(f"\n  Detected tracer type: {tracer_type}")
    
    # Estimate activity from SUV
    voxel_vol_mL = np.prod(voxel_size_mm) / 1000
    injected_MBq = estimate_injected_activity_from_suv(
        pet_data, voxel_vol_mL, args.body_weight, tracer_type
    )
    
    # Scale SUV to therapeutic isotope activity
    print(f"\n  Scaling to {args.isotope} therapy...")
    print(f"  Therapeutic activity: {args.activity_MBq:.0f} MBq")
    
    # Standard SUV to activity concentration formula:
    # A(r) [Bq/mL] = SUV(r) × (A_injected [Bq] / body_weight [g])
    # 
    # Where body_weight is in grams (since 1 mL water ≈ 1 g)
    # This gives activity concentration assuming SUV=1 corresponds to
    # uniform distribution of activity throughout the body
    
    body_weight_g = args.body_weight * 1000  # kg to g
    
    # Activity factor: Bq per gram of body weight
    activity_factor = (args.activity_MBq * 1e6) / body_weight_g  # Bq/g = Bq/mL
    
    # Convert SUV to activity concentration
    activity_conc = pet_data * activity_factor  # Bq/mL
    
    # Sanity check
    total_activity_MBq = activity_conc.sum() * voxel_vol_mL / 1e6
    
    print(f"  Activity factor: {activity_factor/1000:.2f} kBq/mL per SUV unit")
    print(f"  Max activity concentration: {activity_conc.max()/1000:.1f} kBq/mL")
    print(f"  (At SUV={pet_data.max():.1f})")
    
    # Note: Total distributed activity may exceed injected activity due to 
    # SUV definition (if mean whole-body SUV > 1). This is expected.
    print(f"  Estimated total body activity: {total_activity_MBq:.1f} MBq")
    
    # Calculate dose
    print("\n" + "=" * 60)
    print("Dosimetry Calculation")
    print("=" * 60)
    
    dose_map = calculate_dose_dpk(
        activity_conc, ct_data, args.isotope, voxel_size_mm,
        segmentations=segmentations
    )
    
    # Calculate organ doses
    organ_doses = calculate_organ_doses(dose_map, segmentations, voxel_size_mm)
    
    # Add radiobiology
    organ_doses = calculate_organ_radiobiology(organ_doses, args.isotope)
    
    print("\n  Organ Doses:")
    print(organ_doses[['organ', 'mean_dose_Gy', 'survival_percent', 'DSB_per_cell']].to_string())
    
    # Save dose map
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    iso_name = args.isotope.replace('-', '')
    
    # Save NIfTI files
    affine = metadata['pet_affine']
    
    dose_img = nib.Nifti1Image(dose_map.astype(np.float32), affine)
    nib.save(dose_img, str(output_dir / f'dose_Gy_{iso_name}.nii.gz'))
    print(f"  dose_Gy_{iso_name}.nii.gz")
    
    activity_img = nib.Nifti1Image(activity_conc.astype(np.float32), affine)
    nib.save(activity_img, str(output_dir / f'activity_Bq_mL_{iso_name}.nii.gz'))
    print(f"  activity_Bq_mL_{iso_name}.nii.gz")
    
    # Save CSV files
    organ_doses.to_csv(output_dir / 'organ_doses.csv', index=False)
    print("  organ_doses.csv")
    
    biodist.to_csv(output_dir / 'biodistribution.csv', index=False)
    print("  biodistribution.csv")
    
    # =========================================================================
    # TREATMENT PLANNING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("Treatment Planning Analysis")
    print("=" * 60)
    
    # Calculate treatment requirements
    treatment_planning = calculate_treatment_planning(
        organ_doses, biodist, args.isotope, args.activity_MBq
    )
    
    print("\n--- Required Doses for Tumor Kill ---")
    print(f"  Current tumor dose: {treatment_planning['current_tumor_dose_Gy']:.2f} Gy")
    print(f"  Dose per MBq (tumor): {treatment_planning['dose_per_MBq'].get('tumor', 0)*1000:.4f} mGy/MBq")
    print("")
    
    for target, options in treatment_planning['treatment_options'].items():
        print(f"  {target} tumor kill:")
        print(f"    Required tumor dose: {options['required_tumor_dose_Gy']:.2f} Gy")
        print(f"    Required activity: {options['required_activity_GBq']:.2f} GBq ({options['required_activity_MBq']:.0f} MBq)")
        
        # Key organ doses
        kidney_dose = options['organ_doses'].get('kidney', {}).get('dose_Gy', 0)
        liver_dose = options['organ_doses'].get('liver', {}).get('dose_Gy', 0)
        print(f"    Kidney dose: {kidney_dose:.2f} Gy")
        print(f"    Liver dose: {liver_dose:.2f} Gy")
        print("")
    
    # Calculate fractionation strategy
    fractionation = calculate_fractionation_strategy(treatment_planning, args.isotope)
    
    print("\n--- Fractionation Strategy ---")
    print(f"  Limiting organ: {fractionation['limiting_organ']}")
    print(f"  Max safe activity per cycle: {fractionation['max_activity_per_cycle_GBq']:.2f} GBq")
    print(f"  Tumor dose per cycle: {fractionation['tumor_dose_per_cycle_Gy']:.2f} Gy")
    print("")
    
    for target, options in fractionation['fractionation_options'].items():
        status = "OK" if not options['exceeds_organ_limits'] else "EXCEEDS LIMITS"
        print(f"  {target} kill: {options['n_cycles']} cycles, {options['total_tumor_dose_Gy']:.1f} Gy total [{status}]")
    
    # Generate recommendation
    recommendation = generate_treatment_recommendation(fractionation, args.isotope)
    print("\n" + recommendation)
    
    # Save treatment planning
    treatment_plan_file = output_dir / 'treatment_planning.json'
    with open(treatment_plan_file, 'w') as f:
        json.dump({
            'treatment_planning': treatment_planning,
            'fractionation': fractionation,
        }, f, indent=2, default=str)
    print(f"\n  Saved treatment planning to: {treatment_plan_file.name}")
    
    # Save recommendation as text
    rec_file = output_dir / 'TREATMENT_RECOMMENDATION.md'
    with open(rec_file, 'w', encoding='utf-8') as f:
        f.write(f"# {args.isotope} Treatment Recommendation\n\n")
        f.write(f"Patient: {pet_path.parent.name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n" + recommendation + "\n```\n\n")
        
        f.write("## Treatment Options Summary\n\n")
        f.write("| Target | Cycles | Activity/Cycle | Total Tumor Dose | Status |\n")
        f.write("|--------|--------|----------------|------------------|--------|\n")
        for target, opts in fractionation['fractionation_options'].items():
            status = "Safe" if not opts['exceeds_organ_limits'] else "Risk"
            f.write(f"| {target} | {opts['n_cycles']} | {opts['activity_per_cycle_GBq']:.2f} GBq | {opts['total_tumor_dose_Gy']:.1f} Gy | {status} |\n")
        
        f.write("\n## Organ Doses at Each Treatment Level\n\n")
        f.write("| Organ | 50% Kill | 95% Kill | 99% Kill |\n")
        f.write("|-------|----------|----------|----------|\n")
        
        organs_to_show = ['tumor', 'kidney', 'liver', 'spleen', 'bone']
        for organ in organs_to_show:
            if organ in treatment_planning['dose_per_MBq']:
                doses = []
                for target in ['50%', '95%', '99%']:
                    if target in treatment_planning['treatment_options']:
                        od = treatment_planning['treatment_options'][target]['organ_doses']
                        if organ in od:
                            doses.append(f"{od[organ]['dose_Gy']:.1f} Gy")
                        else:
                            doses.append("N/A")
                    else:
                        doses.append("N/A")
                f.write(f"| {organ} | {' | '.join(doses)} |\n")
        
        f.write("\n## Clinical Notes\n\n")
        f.write("- Doses calculated using Linear-Quadratic (LQ) model\n")
        f.write(f"- LQ parameters: alpha={THERAPEUTIC_ISOTOPES[args.isotope]['alpha_LQ']}, beta={THERAPEUTIC_ISOTOPES[args.isotope]['beta_LQ']}\n")
        f.write("- Organ limits: Kidney 23 Gy cumulative, Bone marrow 2 Gy cumulative\n")
        f.write("- Treatment schedule: 8-12 week intervals recommended\n")
    
    print(f"  Saved recommendation to: {rec_file.name}")
    
    # Create figures
    print("\n" + "=" * 60)
    print("Creating Figures")
    print("=" * 60)
    
    create_dosimetry_figures(
        pet_data, dose_map, segmentations,
        organ_doses, biodist, output_dir, args.isotope
    )
    
    # Create treatment planning figure
    create_treatment_planning_figure(
        treatment_planning, fractionation, args.isotope, output_dir
    )
    
    # Generate report
    generate_report(organ_doses, biodist, args.isotope, metadata, output_dir)
    
    # Save summary JSON
    summary = {
        'isotope': args.isotope,
        'therapeutic_activity_MBq': args.activity_MBq,
        'body_weight_kg': args.body_weight,
        'pet_voxel_mm': list(voxel_size_mm),
        'organs_analyzed': list(segmentations.keys()),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add organ summary
    summary['organ_doses'] = organ_doses.to_dict(orient='records')
    summary['biodistribution'] = biodist.to_dict(orient='records')
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("  summary.json")
    
    print("\n" + "=" * 70)
    print("DOSIMETRY COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    # Print key results
    print("\n--- Key Results ---")
    if 'tumor' in organ_doses['organ'].values:
        tumor = organ_doses[organ_doses['organ'] == 'tumor'].iloc[0]
        print(f"  Tumor dose: {tumor['mean_dose_Gy']:.4f} Gy")
        print(f"  Tumor survival: {tumor['survival_percent']:.1f}%")
        print(f"  Tumor DSB/cell: {tumor['DSB_per_cell']:.1f}")
    
    print(f"\n  Critical organs:")
    for _, row in organ_doses[organ_doses['organ'] != 'tumor'].nlargest(3, 'mean_dose_Gy').iterrows():
        print(f"    {row['organ']}: {row['mean_dose_Gy']:.4f} Gy ({row['survival_percent']:.1f}% survival)")
    
    return organ_doses, biodist


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CellRad-MC Patient Dosimetry from PET/CT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using patient folder (auto-detection)
  python patient_dosimetry.py --patient-folder patient/fdg_xxx/ --isotope Lu-177 --output results/
  
  # Explicit file paths
  python patient_dosimetry.py --pet pet.nii.gz --ct ct.nii.gz --segmentation seg/ --isotope Ac-225
  
  # With custom activity
  python patient_dosimetry.py --patient-folder patient/ --isotope Lu-177 --activity 7400

Supported Isotopes:
  Beta:  Lu-177, Y-90, I-131
  Alpha: Ac-225, Ra-223
        """
    )
    
    # Input options
    parser.add_argument('--patient-folder', '-p', 
                        help='Patient folder with PET/CT and segmentations')
    parser.add_argument('--pet', help='PET scan (NIfTI)')
    parser.add_argument('--ct', help='CT scan (NIfTI)')
    parser.add_argument('--segmentation', '-s', help='Segmentation folder')
    
    # Isotope
    parser.add_argument('--isotope', '-i', required=True,
                        choices=list(THERAPEUTIC_ISOTOPES.keys()),
                        help='Therapeutic radioisotope')
    
    # Output
    parser.add_argument('--output', '-o', default='dosimetry_results',
                        help='Output directory')
    
    # Clinical parameters
    parser.add_argument('--activity', type=float, default=7400,
                        dest='activity_MBq',
                        help='Therapeutic activity in MBq (default: 7400 for Lu-177)')
    parser.add_argument('--body-weight', type=float, default=70.0,
                        help='Patient body weight in kg (default: 70)')
    
    # CT options
    parser.add_argument('--auto-ct', action='store_true',
                        help='Use uniform soft tissue HU if no CT available')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.patient_folder and (not args.pet or not args.ct):
        parser.error("Either --patient-folder or both --pet and --ct are required")
    
    try:
        run_patient_dosimetry(args)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

