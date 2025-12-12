# CellRad-MC Complete User Guide

## 🎯 What is CellRad-MC?

**CellRad-MC** is a GPU-accelerated Monte Carlo simulation platform for **cellular-level radiation dosimetry** with **DNA damage prediction**. It's designed for **targeted radionuclide therapy (TRT)** research, enabling scientists to:

1. **Calculate absorbed dose** at the cellular level from radioactive isotopes
2. **Predict DNA damage** (SSB, DSB) based on track-structure physics
3. **Estimate cell survival** using the linear-quadratic (LQ) model
4. **Track dose accumulation** over time during therapy

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CellRad-MC Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT                    SIMULATION                    OUTPUT       │
│  ──────                   ──────────                    ──────       │
│                                                                      │
│  ┌─────────────┐         ┌──────────────┐         ┌──────────────┐ │
│  │ Protein Map │ ──────► │  GPU Monte   │ ──────► │  Dose (Gy)   │ │
│  │ (NIfTI)     │         │  Carlo       │         │              │ │
│  └─────────────┘         │  Engine      │         └──────────────┘ │
│                          │              │                           │
│  ┌─────────────┐         │  100M        │         ┌──────────────┐ │
│  │ CT Scan     │ ──────► │  particles   │ ──────► │  Energy (eV) │ │
│  │ (optional)  │         │  in minutes  │         │              │ │
│  └─────────────┘         │              │         └──────────────┘ │
│                          │              │                           │
│  ┌─────────────┐         │  Track-      │         ┌──────────────┐ │
│  │ DAPI Mask   │ ──────► │  Structure   │ ──────► │  DNA Damage  │ │
│  │ (optional)  │         │  Physics     │         │  (SSB/DSB)   │ │
│  └─────────────┘         └──────────────┘         └──────────────┘ │
│                                                                      │
│  ┌─────────────┐                                  ┌──────────────┐ │
│  │ Isotope     │ ──────────────────────────────► │  Cell        │ │
│  │ Selection   │                                 │  Survival    │ │
│  └─────────────┘                                 └──────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Software Components

### 1. **`cellrad_mc.py`** - Standard Monte Carlo Simulation
Main CLI for condensed-history Monte Carlo simulation.

```bash
python cellrad_mc.py --protein protein.nii.gz --isotope Lu-177 --out-dir output
```

**Features:**
- Electron transport (Bethe-Bloch stopping power)
- Alpha transport with Bragg peak
- Gamma/photon transport (Compton, photoelectric)
- Auger cascade modeling

### 2. **`cellrad_track_structure.py`** - Full Track-Structure Simulation
Advanced CLI for event-by-event track-structure simulation.

```bash
python cellrad_track_structure.py --protein protein.nii.gz --dapi dapi.nii.gz \
    --isotope Lu-177 --out-dir output --n-particles 10000000
```

**Features:**
- Event-by-event electron tracking
- DNA geometry overlay
- SSB/DSB damage classification
- Per-cell dose and damage reporting

### 3. **`track_structure_mc.py`** - Track-Structure Engine
Core GPU kernels for nanometer-scale particle tracking.

### 4. **`dna_geometry.py`** - DNA Geometry Builder
Creates DNA density maps from DAPI nuclear masks.

### 5. **`enhanced_physics.py`** - Physics Database
Contains isotope data, cross-sections, and physical constants.

### 6. **`gpu_monte_carlo.py`** - GPU Engine
Core CUDA kernels for particle transport.

---

## 🎮 Command Line Arguments

### **`cellrad_mc.py`** - Standard Simulation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--protein` | path | ✅ | - | Protein/activity map (NIfTI format) |
| `--ct` | path | ❌ | - | CT scan for tissue heterogeneity |
| `--auto-ct` | flag | ❌ | - | Use uniform soft tissue (HU=19) |
| `--isotope` | string | ✅ | - | Radioisotope (Lu-177, Ac-225, etc.) |
| `--out-dir` | path | ✅ | - | Output directory |
| `--n-particles` | int | ❌ | 1000000 | Number of particles to simulate |
| `--voxel-size` | float | ❌ | from file | XY voxel size in microns |
| `--voxel-size-z` | float | ❌ | same as XY | Z voxel size in microns |
| `--threshold` | float | ❌ | 50 | Activity threshold percentile |
| `--gammas` | flag | ❌ | - | Enable gamma transport |
| `--seed` | int | ❌ | 42 | Random seed |
| `--validate` | flag | ❌ | - | Run physics validation |

### **`cellrad_track_structure.py`** - Track-Structure Simulation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--protein` | path | ✅ | - | Protein/activity map (NIfTI format) |
| `--dapi` | path | ❌ | - | DAPI nuclear mask for DNA damage |
| `--ct` | path | ❌ | - | CT scan for tissue heterogeneity |
| `--auto-ct` | flag | ❌ | - | Use uniform soft tissue |
| `--isotope` | string | ✅ | - | Radioisotope |
| `--out-dir` | path | ✅ | - | Output directory |
| `--n-particles` | int | ❌ | 1000000 | Number of particles |
| `--voxel-size` | float | ❌ | from file | XY voxel size (μm) |
| `--voxel-size-z` | float | ❌ | same as XY | Z voxel size (μm) |
| `--threshold` | float | ❌ | 50 | Activity threshold percentile |
| `--seed` | int | ❌ | 42 | Random seed |
| `--validate` | flag | ❌ | - | Run physics validation |

---

## 🧬 Supported Isotopes

| Isotope | Type | Half-life | Main Emissions | Clinical Use |
|---------|------|-----------|----------------|--------------|
| **Lu-177** | β⁻ | 6.65 d | β⁻ (498 keV max) | PRRT, PSMA therapy |
| **Y-90** | β⁻ | 2.67 d | β⁻ (2.28 MeV max) | Radioembolization |
| **I-131** | β⁻/γ | 8.02 d | β⁻ (606 keV), γ | Thyroid therapy |
| **Ac-225** | α | 10.0 d | α (5.8 MeV) chain | TAT |
| **At-211** | α | 7.2 h | α (5.9, 7.5 MeV) | TAT |
| **Tb-161** | β⁻/Auger | 6.9 d | β⁻, Auger | Low LET therapy |
| **I-125** | Auger | 59.4 d | Auger cascade | Brachytherapy |
| **In-111** | Auger | 2.8 d | Auger, γ | Imaging/therapy |
| **Ga-67** | Auger | 3.3 d | Auger, γ | Imaging |
| **Tc-99m** | γ | 6.0 h | γ (140 keV) | Imaging |

---

## 📊 Output Files

### Standard Outputs

| File | Format | Description |
|------|--------|-------------|
| `dose_Gy_*.nii.gz` | NIfTI | Absorbed dose in Gray |
| `energy_eV_*.nii.gz` | NIfTI | Energy deposition in eV |
| `hits_*.nii.gz` | NIfTI | Number of particle hits |
| `uncertainty_*.nii.gz` | NIfTI | Statistical uncertainty |
| `stats_*.json` | JSON | Simulation statistics |

### Track-Structure Outputs (with `--dapi`)

| File | Format | Description |
|------|--------|-------------|
| All standard outputs | - | - |
| `dna_damage_*.json` | JSON | SSB/DSB counts per cell |
| `cell_survival_*.json` | JSON | Survival fractions per cell |

---

## 🔬 Physics Models

### 1. **Electron Transport**
- **Stopping Power**: Bethe-Bloch formula, calibrated to NIST ESTAR
- **Scattering**: Highland multiple Coulomb scattering
- **Energy Range**: 10 eV - 1 MeV
- **Accuracy**: <5% vs NIST for 10-500 keV

### 2. **Alpha Particle Transport**
- **Stopping Power**: NIST ASTAR calibrated
- **Bragg Peak**: Full Bragg curve modeling
- **Energy Range**: 100 keV - 10 MeV
- **Accuracy**: <5% vs NIST

### 3. **Gamma/Photon Transport**
- **Compton Scattering**: Klein-Nishina cross-section
- **Photoelectric Effect**: NIST XCOM data
- **Energy Range**: 1 keV - 10 MeV

### 4. **Auger Cascade**
- **Database**: ICRP 107 calibrated
- **Isotopes**: I-125, In-111, Ga-67, Tc-99m
- **Accuracy**: <5% vs ICRP 107

---

## 🧬 DNA Damage Model

### Damage Classification

| Damage Type | Definition | Threshold |
|-------------|------------|-----------|
| **SSB** | Single Strand Break | 17.5 eV in backbone |
| **DSB** | Double Strand Break | 2 SSBs within 10 bp on opposite strands |
| **Base Damage** | Base oxidation | Variable |
| **Complex DSB** | DSB + additional lesions | Multiple |

### Literature Values

| Parameter | Value | Reference |
|-----------|-------|-----------|
| DSB/Gy/Gbp (γ) | 35 | Rothkamm 2003 |
| SSB/Gy/Gbp (γ) | 1000 | Nikjoo 2001 |
| SSB:DSB ratio (low LET) | 25:1 | Consensus |
| SSB:DSB ratio (high LET) | 5:1 | Friedland 2003 |
| Complex DSB (α) | 70% | Nikjoo 2001 |

---

## 📈 Cell Survival Model

### Linear-Quadratic (LQ) Model

```
S = exp(-αD - βD²)
```

Where:
- `S` = Surviving fraction
- `D` = Dose in Gy
- `α` = Linear coefficient (Gy⁻¹)
- `β` = Quadratic coefficient (Gy⁻²)

### Default Parameters

| Cell Type | α (Gy⁻¹) | β (Gy⁻²) | α/β (Gy) |
|-----------|----------|----------|----------|
| Tumor (γ-rays) | 0.30 | 0.030 | 10.0 |
| Tumor (α-particles) | 0.80 | 0.015 | 53.3 |
| Normal tissue | 0.15 | 0.050 | 3.0 |
| Hypoxic tumor | 0.10 | 0.025 | 4.0 |

---

## 🚀 Example Workflows

### Workflow 1: Basic Dosimetry

```bash
# Simulate Lu-177 therapy
python cellrad_mc.py \
    --protein EGFR_expression.nii.gz \
    --auto-ct \
    --isotope Lu-177 \
    --out-dir output_lu177 \
    --n-particles 10000000 \
    --voxel-size 0.455 \
    --voxel-size-z 1.0
```

### Workflow 2: Alpha Particle Therapy

```bash
# Simulate Ac-225 targeted alpha therapy
python cellrad_mc.py \
    --protein HER2_expression.nii.gz \
    --auto-ct \
    --isotope Ac-225 \
    --out-dir output_ac225 \
    --n-particles 5000000
```

### Workflow 3: Full DNA Damage Analysis

```bash
# Track-structure simulation with DNA damage
python cellrad_track_structure.py \
    --protein PSMA_expression.nii.gz \
    --dapi DAPI_nuclei.nii.gz \
    --isotope Lu-177 \
    --out-dir output_dna_damage \
    --n-particles 20000000
```

### Workflow 4: Convert TIF to NIfTI

```bash
# Convert microscopy TIF to NIfTI
python convert_tif.py input.tif --output-dir converted/
```

### Workflow 5: Run Validation

```bash
# Validate physics
python cellrad_mc.py --validate

# Validate DNA damage model
python dna_damage_validation.py
```

---

## ⚡ Performance

### Typical Performance (RTX 4090)

| Particles | Time | Rate |
|-----------|------|------|
| 1 million | ~2 s | 500k/s |
| 10 million | ~15 s | 670k/s |
| 100 million | ~90 s | 1.1M/s |

### Memory Requirements

| Grid Size | Memory |
|-----------|--------|
| 256³ | ~500 MB |
| 512³ | ~2 GB |
| 1024³ | ~8 GB |

---

## 📚 Validation Status

### Physics Validation

| Test | Status | Error |
|------|--------|-------|
| Electron CSDA Range | ✅ PASS | <5% |
| Alpha CSDA Range | ✅ PASS | <5% |
| Photon MFP | ✅ PASS | <2% |
| Auger Energies | ✅ PASS | <5% |
| Beta Spectra | ✅ PASS | <5% |

### Biological Validation

| Test | Status | Reference |
|------|--------|-----------|
| DSB Yields | ✅ PASS | Rothkamm 2003 |
| SSB:DSB Ratio | ✅ PASS | Literature |
| Cell Survival (LQ) | ✅ PASS | Standard |

---

## 🔧 Troubleshooting

### Common Issues

1. **"CUDA GPU not available"**
   - Ensure NVIDIA GPU is installed
   - Install CUDA toolkit
   - Install compatible numba version

2. **"Out of memory"**
   - Reduce `--n-particles`
   - Use smaller grid (lower resolution input)

3. **"Activity map is all zeros"**
   - Check input file format
   - Adjust `--threshold` parameter

4. **"Voxel size mismatch"**
   - Explicitly set `--voxel-size` and `--voxel-size-z`
   - Check NIfTI header

---

## 📖 References

1. **Geant4-DNA**: Med. Phys. 37 (2010) 4692-4708
2. **TOPAS-nBio**: Phys. Med. 31 (2015) 861-874
3. **NIST ESTAR/ASTAR**: physics.nist.gov
4. **ICRP 107**: Nuclear Decay Data (2008)
5. **Nikjoo et al.**: Radiat. Res. 156 (2001) 577-583
6. **Rothkamm & Löbrich**: PNAS 100 (2003) 5057-5062

---

## 📧 Support

For questions, issues, or contributions, please contact the CellRad-MC development team.

---

*Last updated: December 2025*

