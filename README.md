# CellRad-MC

## GPU-Accelerated Monte Carlo Simulation for Cellular Dosimetry in Targeted Radionuclide Therapy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/cellrad_banner.png" alt="CellRad-MC" width="800"/>
</p>

---

## 🎯 Overview

**CellRad-MC** is a cutting-edge, GPU-accelerated Monte Carlo simulation platform designed for cellular-level dosimetry in targeted radionuclide therapy (TRT). It simulates the transport of beta particles, alpha particles, Auger electrons, and photons through tissue at microscopic resolution.

### Key Features

- ⚡ **Ultra-Fast**: 500,000+ particles/second on consumer GPUs
- 🎯 **Cellular Resolution**: Micron-scale dose calculations
- 🧬 **DNA Damage Modeling**: Predicts DSB, SSB, and cell survival
- 📊 **Publication-Ready**: Nature-level figures and analysis
- ✅ **Validated**: NIST ESTAR/ASTAR and ICRP 107 compliant
- 🔬 **Multi-Isotope**: Supports 15+ clinical radioisotopes

---

## 📂 Repository Structure

```
CellRad-MC/
├── cellrad_mc.py                 # Main Monte Carlo simulation CLI
├── gpu_monte_carlo.py            # GPU transport engine (CUDA kernels)
├── run_comprehensive_analysis.py # Full cellular analysis pipeline
├── enhanced_physics.py           # Physics data and validation
├── dna_geometry.py               # DNA damage calculations
├── convert_tif.py                # TIF to NIfTI converter
├── publication_validation.py     # Physics validation suite
├── isotope_validation.py         # Isotope database validation
├── dna_damage_validation.py      # DNA damage validation
├── track_structure_mc.py         # Track-structure module
├── track_structure_validation.py # Track validation
├── cellrad_track_structure.py    # Track-structure CLI
├── validate.sh / validate.bat    # Validation scripts
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── docs/
│   ├── CELLRAD_MC_COMPLETE_GUIDE.md    # Detailed user guide
│   └── NATURE_METHODS_AND_MATERIALS.md # Publication methods
├── examples/
│   ├── nuclei_EGFR.nii.gz        # Example nuclei mask
│   ├── protein_EGFR.nii.gz       # Example protein expression
│   ├── CT_19.nii.gz              # Example CT scan
│   ├── cell_analysis_EGFR_v5_Lu177/  # Lu-177 analysis example
│   └── cell_analysis_EGFR_v5_Ac225/  # Ac-225 analysis example
└── validation_results/
    ├── Figure1-6_*.png           # Validation figures
    └── *.csv / *.json            # Validation data
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (8+ GB VRAM recommended)
- CUDA Toolkit 11.0 or higher

### Quick Install

```bash
# Clone the repository
git clone https://github.com/cellrad-mc/cellrad-mc.git
cd cellrad-mc

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.20.0
cupy-cuda11x>=10.0.0
numba>=0.55.0
nibabel>=3.2.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
tifffile>=2021.0.0
```

### Verify Installation

```bash
python -c "import cupy; print(f'CuPy OK, GPU: {cupy.cuda.Device(0)}')"
python -c "from gpu_monte_carlo import GPUMonteCarlo; print('CellRad-MC OK')"
```

---

## 🚀 Quick Start

### 1. Basic Simulation (30 seconds)

```bash
python cellrad_mc.py \
    --protein your_protein_map.nii.gz \
    --auto-ct \
    --isotope Lu-177 \
    --n-particles 100000 \
    --out-dir quick_test/
```

### 2. View Results

Open the output folder to find:
- `dose_Gy_Lu177.nii.gz` - Absorbed dose map
- `summary_Lu177.png` - Visual summary
- `stats_Lu177.json` - Statistics

---

## 📖 Tutorials

### Tutorial 1: Your First Simulation

**Goal**: Simulate Lu-177 beta therapy on a protein expression map.

```bash
# Step 1: Prepare your data
# You need a NIfTI file (.nii.gz) with protein expression values

# Step 2: Run simulation
python cellrad_mc.py \
    --protein protein_expression.nii.gz \
    --auto-ct \
    --isotope Lu-177 \
    --n-particles 1000000 \
    --voxel-size 1.0 \
    --out-dir results_lu177/

# Step 3: Check outputs
ls results_lu177/
# dose_keV_Lu177.nii.gz
# dose_Gy_Lu177.nii.gz
# particle_count_Lu177.nii.gz
# uncertainty_rel_Lu177.nii.gz
# summary_Lu177.png
# stats_Lu177.json
```

### Tutorial 2: Alpha Therapy with Ac-225

**Goal**: Simulate Ac-225 alpha particle therapy with full decay chain.

```bash
python cellrad_mc.py \
    --protein tumor_PSMA.nii.gz \
    --auto-ct \
    --isotope Ac-225 \
    --n-particles 1000000 \
    --voxel-size 1.0 \
    --out-dir results_ac225/
```

**What happens:**
1. Ac-225 decays → Fr-221 (5.83 MeV alpha)
2. Fr-221 decays → At-217 (6.34 MeV alpha)
3. At-217 decays → Bi-213 (7.07 MeV alpha)
4. Bi-213 → 2.09% alpha (5.87 MeV) or 97.91% beta → Po-213
5. Po-213 decays → Pb-209 (8.38 MeV alpha)

**Total**: ~4 alphas per decay, ~27.6 MeV deposited locally!

### Tutorial 3: Comprehensive Cellular Analysis

**Goal**: Full analysis with DNA damage and cell survival.

```bash
# Using example data from examples/ directory
python run_comprehensive_analysis.py \
    --nuclei examples/nuclei_EGFR.nii.gz \
    --protein examples/protein_EGFR.nii.gz \
    --output my_analysis/ \
    --isotope Ac-225 \
    --n-particles 10000000 \
    --voxel-size 1.0 \
    --auto-ct
```

**Outputs include:**
- 8 publication-quality figures
- Per-cell dose, DNA damage, survival data
- Spatial heterogeneity analysis
- Comprehensive analysis report (Markdown)

### Tutorial 4: Using Real CT Data

**Goal**: Use patient CT scan for accurate material mapping.

```bash
python cellrad_mc.py \
    --protein SPECT_activity.nii.gz \
    --ct patient_CT.nii.gz \
    --isotope Lu-177 \
    --n-particles 5000000 \
    --out-dir patient_dosimetry/
```

The CT provides:
- Tissue density variations
- Bone/lung/soft tissue differentiation
- Accurate dose calculation near interfaces

### Tutorial 5: Anisotropic Voxels (Microscopy Data)

**Goal**: Handle data with different XY and Z resolutions.

```bash
# Common for confocal microscopy: 0.45 μm XY, 1.0 μm Z
python cellrad_mc.py \
    --protein confocal_Ki67.nii.gz \
    --auto-ct \
    --isotope I-125 \
    --n-particles 1000000 \
    --voxel-size 0.45 \
    --voxel-size-z 1.0 \
    --out-dir confocal_results/
```

### Tutorial 6: Convert TIF Microscopy to NIfTI

**Goal**: Convert multi-channel TIF files for use in CellRad-MC.

```bash
python convert_tif.py \
    --input "DAPI_EGFR_CD31.lif - Series012.tif" \
    --output converted/ \
    --channels DAPI EGFR CD31
```

### Tutorial 7: Run Physics Validation

**Goal**: Verify physics accuracy against NIST/ICRP data.

```bash
# Full validation suite
python publication_validation.py

# Isotope-specific validation
python isotope_validation.py

# Or run both
./validate.sh  # Linux/Mac
# or: bash validate.sh  # Windows with Git Bash
```

---

## ⚙️ Command Line Arguments

### `cellrad_mc.py` - Main Simulation

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--protein` | `-p` | PATH | **Required** | Protein/activity map (NIfTI format) |
| `--ct` | `-c` | PATH | - | CT scan for material mapping (NIfTI) |
| `--auto-ct` | - | FLAG | - | Use uniform soft tissue (HU=19) |
| `--isotope` | `-i` | STR | **Required** | Radioisotope to simulate |
| `--out-dir` | `-o` | PATH | `output/` | Output directory |
| `--n-particles` | `-n` | INT | 100000 | Number of primary decays |
| `--voxel-size` | `-v` | FLOAT | From header | XY voxel size in microns |
| `--voxel-size-z` | - | FLOAT | Same as XY | Z voxel size in microns |
| `--gammas` | - | FLAG | False | Also transport gamma photons |
| `--validate` | - | FLAG | False | Run physics validation |
| `--threshold` | `-t` | FLOAT | 1.0 | Activity threshold (% of max) |
| `--seed` | - | INT | Random | Random seed for reproducibility |

**Note**: Either `--ct` or `--auto-ct` must be specified.

### `run_comprehensive_analysis.py` - Full Analysis

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--nuclei` | - | PATH | **Required** | Nuclei mask (NIfTI, unique IDs per cell) |
| `--protein` | - | PATH | **Required** | Protein expression map (NIfTI) |
| `--output` | `-o` | PATH | `cell_analysis/` | Output directory |
| `--isotope` | `-i` | STR | `Lu-177` | Radioisotope |
| `--n-particles` | `-n` | INT | 10000000 | Number of particles |
| `--voxel-size` | `-v` | FLOAT | 1.0 | Voxel size in microns |
| `--ct` | - | PATH | - | CT scan (NIfTI) |
| `--auto-ct` | - | FLAG | False | Use uniform soft tissue (HU=19) |
| `--skip-simulation` | - | FLAG | False | Skip MC, load existing results |
| `--force` | `-f` | FLAG | False | Force re-run simulation |

**Auto-Resume Feature**: If simulation results already exist in the output directory, the script will automatically skip the Monte Carlo simulation and load existing results. Use `--force` to override this behavior.

### `convert_tif.py` - TIF to NIfTI Converter

| Argument | Type | Description |
|----------|------|-------------|
| `--file` | PATH | Input TIF file (multi-channel microscopy) |
| `--channels` | INT | Number of channels in the TIF |
| `--channel-names` | STR | Comma-separated channel names |
| `--out-dir` | PATH | Output directory |
| `--remove` | STR | Remove string from filename |

---

## ⚛️ Physics Engine

CellRad-MC implements a full GPU Monte Carlo transport code with validated physics.

### Particle Transport

| Particle | Transport Method | Energy Range | Physics |
|----------|------------------|--------------|---------|
| **Electrons (β)** | Continuous slowing-down | 1 keV - 3 MeV | NIST ESTAR stopping powers |
| **Alphas (α)** | Continuous slowing-down | 1 keV - 10 MeV | NIST ASTAR stopping powers |
| **Photons (γ)** | Discrete interactions | 1 keV - 1 MeV | NIST XCOM cross-sections |
| **Auger (e⁻)** | Local deposition | < 30 keV | ICRP 107 spectra |

### Energy Deposition

- **Step-by-step tracking**: 0.1 μm steps for electrons, continuous for alphas
- **Angular scattering**: Modified Highland formula
- **Range straggling**: Gaussian energy loss fluctuations
- **Delta rays**: Secondary electron production (optional)

### Material Handling

7 tissue types supported via CT Hounsfield Units:
- Air (HU < -900)
- Lung (-900 to -500)
- Adipose (-100 to -50)
- Soft tissue (-50 to 100)
- Muscle (100 to 300)
- Cartilage (300 to 700)
- Bone (> 700)

---

## 💊 Supported Isotopes

### Beta Emitters

| Isotope | Half-life | E_max (keV) | E_mean (keV) | Range (mm) | Use Case |
|---------|-----------|-------------|--------------|------------|----------|
| **Lu-177** | 6.64 d | 498 | 134 | ~2 | PSMA, somatostatin |
| **Y-90** | 2.67 d | 2280 | 934 | ~11 | Liver, lymphoma |
| **I-131** | 8.03 d | 606 | 192 | ~2.4 | Thyroid |
| **P-32** | 14.3 d | 1711 | 695 | ~8 | Polycythemia vera |
| **Sr-89** | 50.6 d | 1495 | 583 | ~6 | Bone metastases |
| **Tb-161** | 6.89 d | 593 | 154 | ~2 | Emerging theranostic |
| **Cu-67** | 2.58 d | 577 | 141 | ~1.8 | Antibody therapy |

### Alpha Emitters

| Isotope | Half-life | # Alphas | Total E (MeV) | Range (μm) | Use Case |
|---------|-----------|----------|---------------|------------|----------|
| **Ac-225** | 9.92 d | 4 | 27.6 | 50-80 | PSMA, emerging |
| **Ra-223** | 11.4 d | 4 | 26.4 | 50-75 | Bone metastases |
| **At-211** | 7.21 h | 1 | 6.8 | 50-60 | Brain tumors |
| **Th-227** | 18.7 d | 5 | 32.5 | 50-80 | CD22 targeting |
| **Pb-212** | 10.6 h | 1.6 | 7.8 | 50-70 | HER2 targeting |

### Auger Emitters

| Isotope | Half-life | Auger/decay | Total E (keV) | Range (nm) | Use Case |
|---------|-----------|-------------|---------------|------------|----------|
| **I-125** | 59.4 d | 24.9 | 12.2 | 1-10 | DNA targeting |
| **In-111** | 2.80 d | 14.7 | 6.7 | 1-10 | Antibodies |
| **Ga-67** | 3.26 d | 8.5 | 6.3 | 1-10 | Imaging/therapy |
| **Tc-99m** | 6.01 h | 4.4 | 0.9 | <1 | Diagnostic |

---

## 🧬 DNA Damage & Cell Survival Modeling

CellRad-MC includes peer-reviewed DNA damage and cell survival models based on published literature.

### DNA Damage Parameters (per Gy per Gbp)

| Particle Type | DSB/Gy/Gbp | SSB:DSB Ratio | Complex DSB% | RBE | Reference |
|---------------|------------|---------------|--------------|-----|-----------|
| **Beta (Lu-177)** | 35 | 25:1 | 15% | 1.0 | Nikjoo et al. 1999 |
| **Alpha (Ac-225)** | 40 | 10:1 | 50% | 5.0 | Ritter & Durante 2010 |
| **Auger (I-125)** | 55 | 8:1 | 60% | 8.0 | Howell 1992 |

### Cell Survival: Linear-Quadratic (LQ) Model

```
S = exp(-α·D - β·D²)
```

| Particle Type | α (Gy⁻¹) | β (Gy⁻²) | α/β (Gy) | Source |
|---------------|----------|----------|----------|--------|
| **Beta** | 0.15 | 0.05 | 3.0 | Standard LQ |
| **Alpha** | 0.80 | 0.05 | 16.0 | High-LET |
| **Auger** | 1.20 | 0.02 | 60.0 | Ultra-high-LET |

### DNA Damage Calculation

For each cell, CellRad-MC calculates:
- **Nuclear dose**: Mean dose to the nucleus
- **DSB count**: Double-strand breaks = DSB_rate × Dose × DNA_content
- **SSB count**: Single-strand breaks = DSB × SSB:DSB_ratio
- **Complex DSB**: Clustered damage = DSB × complex_fraction
- **Survival probability**: S = exp(-α·D - β·D²)

---

## 🎯 Spatial Heterogeneity Analysis

The comprehensive analysis includes advanced spatial analysis of tumor dose distribution.

### Region Classification

Cells are automatically classified into three regions based on distance from tumor center:
- **Core** (inner 33%): Tumor center, typically highest expression
- **Intermediate** (33-67%): Transition zone
- **Periphery** (outer 33%): Tumor edge, often lower uptake

### Heterogeneity Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **CV (Coefficient of Variation)** | σ/μ × 100% | Dose uniformity (lower = more uniform) |
| **Gini Coefficient** | 0-1 scale | Inequality (0 = equal, 1 = one cell gets all) |
| **D90** | 90th percentile dose | Dose covering 90% of cells |
| **D10** | 10th percentile dose | Minimum dose to hottest 10% |

### Radial Profile Analysis

Dose is analyzed as a function of distance from tumor center:
- Mean dose vs. radial distance
- ±1 SD error bands
- Identifies dose gradients and cold spots

---

## 📈 Generated Figures (8 Publication-Quality)

| Figure | Content |
|--------|---------|
| **Figure 1** | Data Overview - Nuclei mask, protein expression, dose MIPs |
| **Figure 2** | Dose-Response - Dose distributions, histograms, survival curves |
| **Figure 3** | DNA Damage - DSB/SSB counts, damage vs. dose, survival |
| **Figure 4** | Spatial Analysis - Core vs. periphery, regional comparisons |
| **Figure 5** | Summary - Key metrics overview |
| **Figure 6** | Spatial Heterogeneity - 9-panel regional analysis |
| **Figure 7** | Advanced Dosimetry - LET maps, uncertainty analysis |
| **Figure 8** | Advanced Analysis - Hexbin, contour, radar charts |

---

## 📁 Output Files

### Simulation Outputs

| File | Format | Description |
|------|--------|-------------|
| `dose_keV_{iso}.nii.gz` | NIfTI | Energy deposition (keV per voxel) |
| `dose_Gy_{iso}.nii.gz` | NIfTI | Absorbed dose (Gray) |
| `particle_count_{iso}.nii.gz` | NIfTI | Number of particle hits |
| `uncertainty_rel_{iso}.nii.gz` | NIfTI | Relative uncertainty (%) |
| `LET_keV_um_{iso}.nii.gz` | NIfTI | Linear Energy Transfer |
| `activity_normalized_{iso}.nii.gz` | NIfTI | Preprocessed activity (0-100) |
| `summary_{iso}.png` | PNG | MIP visualization |
| `stats_{iso}.json` | JSON | Simulation statistics |

### Comprehensive Analysis Outputs

| File | Format | Description |
|------|--------|-------------|
| `cell_analysis_complete.csv` | CSV | Per-cell metrics |
| `dose_survival_data.csv` | CSV | Dose-response data |
| `dna_damage_summary.csv` | CSV | DNA damage statistics |
| `cell_spatial_regions.csv` | CSV | Core/intermediate/periphery |
| `region_summary.csv` | CSV | Regional statistics |
| `radial_profile.csv` | CSV | Dose vs. distance |
| `heterogeneity_indices.csv` | CSV | CV, Gini coefficient |
| `Figure1-8_*.png` | PNG | Publication figures |
| `ANALYSIS_REPORT.md` | Markdown | Full analysis report |

---

## 📊 Example Results

### Dose Distribution (Lu-177)

```
============================================================
RESULTS
============================================================
  Total time: 4.23 s
  Total particles: 795,000
  Overall rate: 187,943 particles/sec

  Energy deposition:
    Total: 1,247.83 MeV
    Max voxel: 15.67 keV

  Dose:
    Mean: 2.45e-02 Gy
    Max: 8.91e+00 Gy

  Statistics:
    Voxels hit: 12,456,789
    Mean uncertainty: 2.3%
```

### Ac-225 Alpha Chain

```
============================================================
Running Simulation (1,000,000 decays)
============================================================
  Alpha chain (5 alphas, 4.00 effective per decay)...
    Alpha 1/5 (Ac-225): E = 5.83 MeV, yield = 100.0%, n = 1,000,000
    Alpha 2/5 (Fr-221): E = 6.34 MeV, yield = 100.0%, n = 1,000,000
    Alpha 3/5 (At-217): E = 7.07 MeV, yield = 99.9%, n = 999,016
    Alpha 4/5 (Bi-213): E = 5.87 MeV, yield = 2.1%, n = 21,145
    Alpha 5/5 (Po-213): E = 8.38 MeV, yield = 97.9%, n = 979,056
  Total alphas transported: 3,999,217
  Total energy deposited: 10375.8 MeV
  Time: 7.08 s
  Rate: 564,919 particles/sec
```

---

## ✅ Validation Status

### Physics Validation (NIST)

| Test | Status | Error |
|------|--------|-------|
| Electron CSDA Range (10-500 keV) | ✅ PASS | < 1% |
| Alpha CSDA Range (4-8 MeV) | ✅ PASS | < 1% |
| Photon MFP (30-500 keV) | ✅ PASS | < 2% |

### Isotope Validation (ICRP 107)

| Test | Status | Error |
|------|--------|-------|
| Beta mean energies | ✅ PASS | < 2% |
| Alpha chain energies | ✅ PASS | < 1% |
| Alpha branching ratios | ✅ PASS | 0% |
| Auger total energies | ✅ PASS | < 5% |
| Half-lives | ✅ PASS | 0% |

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
cupy._core._memorypool.OutOfMemoryError
```
**Solution**: Reduce `--n-particles` or process in batches.

**2. No GPU Found**
```
numba.cuda.cudadrv.error.CudaSupportError
```
**Solution**: Install CUDA toolkit and verify with `nvidia-smi`.

**3. NIfTI Loading Error**
```
nibabel.filebasedimages.ImageFileError
```
**Solution**: Check file exists and is valid NIfTI format.

**4. JSON Decode Error (Auto-Resume)**
```
json.decoder.JSONDecodeError
```
**Solution**: Use `--force` flag to regenerate corrupted files.

### Performance Tips

1. **Optimal particle count**: 10⁶ for testing, 10⁷-10⁸ for publication
2. **Memory usage**: ~50 MB per 100³ voxels
3. **Batch processing**: For >10⁸ particles, split into batches
4. **GPU selection**: Use `CUDA_VISIBLE_DEVICES=0` to select GPU

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - quick start and overview |
| [docs/CELLRAD_MC_COMPLETE_GUIDE.md](docs/CELLRAD_MC_COMPLETE_GUIDE.md) | Detailed user guide |
| [docs/NATURE_METHODS_AND_MATERIALS.md](docs/NATURE_METHODS_AND_MATERIALS.md) | Publication-ready methods & materials |
| [validation_results/](validation_results/) | Validation figures and data |
| [examples/](examples/) | Example data and analysis outputs |

---

## 🧪 Running Tests

```bash
# Physics validation
python publication_validation.py

# Isotope database validation
python isotope_validation.py

# DNA damage validation
python dna_damage_validation.py

# Track structure validation
python track_structure_validation.py

# Run all validations
./validate.sh
```

---

## 📝 Citation

If you use CellRad-MC in your research, please cite:

```bibtex
@software{cellrad_mc,
  title = {CellRad-MC: GPU-Accelerated Monte Carlo for Cellular Dosimetry},
  author = {Arvin Haj-Mirzaian},
  year = {2025},
  url = {https://github.com/arvinhm/CellRad-MC}
}
```


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NIST for ESTAR, ASTAR, and XCOM databases
- ICRP for Publication 107 nuclear decay data
- The Geant4 and TOPAS-nBio communities
- NVIDIA for CUDA and CuPy teams

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/cellrad-mc/cellrad-mc/issues)
- **Email**: arvin.mirzaian@gmail.com

---

## 📋 Quick Reference Cheatsheet

### Common Commands

```bash
# Quick test (100k particles, ~30 sec)
python cellrad_mc.py -p protein.nii.gz --auto-ct -i Lu-177 -n 100000 -o test/

# Production run (10M particles)
python cellrad_mc.py -p protein.nii.gz --auto-ct -i Lu-177 -n 10000000 -o results/

# Alpha therapy with Ac-225
python cellrad_mc.py -p PSMA.nii.gz --auto-ct -i Ac-225 -n 1000000 -o ac225/

# Full cellular analysis with DNA damage
python run_comprehensive_analysis.py --nuclei nuclei.nii.gz --protein protein.nii.gz \
    --output analysis/ --isotope Lu-177 --n-particles 10000000 --auto-ct

# Skip simulation, reload existing
python run_comprehensive_analysis.py --nuclei nuclei.nii.gz --protein protein.nii.gz \
    --output analysis/ --skip-simulation

# Force re-run
python run_comprehensive_analysis.py --nuclei nuclei.nii.gz --protein protein.nii.gz \
    --output analysis/ --force

# With real CT data
python cellrad_mc.py -p SPECT.nii.gz -c CT.nii.gz -i Lu-177 -n 5000000 -o patient/

# Validate physics
python publication_validation.py && python isotope_validation.py
```

### Isotope Quick Reference

| Isotope | Type | Best For | Typical Particles |
|---------|------|----------|-------------------|
| `Lu-177` | β | PSMA, somatostatin | 10⁶-10⁷ |
| `Y-90` | β | SIRT, lymphoma | 10⁶-10⁷ |
| `I-131` | β | Thyroid | 10⁶-10⁷ |
| `Ac-225` | α | PSMA (emerging) | 10⁵-10⁶ |
| `Ra-223` | α | Bone mets | 10⁵-10⁶ |
| `At-211` | α | Brain tumors | 10⁵-10⁶ |
| `I-125` | Auger | DNA targeting | 10⁵-10⁶ |

### Output File Quick Reference

| Need | File |
|------|------|
| Dose map | `dose_Gy_*.nii.gz` |
| Energy map | `dose_keV_*.nii.gz` |
| Hit count | `particle_count_*.nii.gz` |
| Uncertainty | `uncertainty_rel_*.nii.gz` |
| Statistics | `stats_*.json` |
| Per-cell data | `cell_analysis_complete.csv` |
| DNA damage | `dna_damage_summary.csv` |
| Survival | `dose_survival_data.csv` |

---

<p align="center">
  Made with ❤️ for the nuclear medicine community
</p>
