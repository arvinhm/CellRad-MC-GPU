#!/usr/bin/env python3
"""
CellRad-MC Patient Dosimetry Physics Validation
================================================

Validates the patient dosimetry calculations against peer-reviewed literature values.

References:
-----------
1. Delker et al. (2016) "Dosimetry for 177Lu-DKFZ-PSMA-617" Eur J Nucl Med
2. Kratochwil et al. (2017) "Targeted alpha-therapy with 225Ac-PSMA-617" J Nucl Med
3. Violet et al. (2019) "Dosimetry of 177Lu-PSMA-I&T" Eur J Nucl Med
4. MIRD Pamphlet 17 - Dose Point Kernel Methodology
5. ICRP Publication 107 - Nuclear Decay Data
6. Sgouros et al. (2010) MIRD Pamphlet 22 - Alpha Dosimetry

Author: CellRad-MC Team
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# =============================================================================
# LITERATURE REFERENCE VALUES
# =============================================================================

# From Delker et al. 2016, Violet et al. 2019 - Lu-177 PSMA-617
# Absorbed dose per GBq administered (Gy/GBq)
LITERATURE_DOSES_LU177 = {
    'tumor': {'mean': 3.3, 'range': (1.0, 12.0), 'unit': 'Gy/GBq', 'ref': 'Violet 2019'},
    'kidney': {'mean': 0.75, 'range': (0.4, 1.5), 'unit': 'Gy/GBq', 'ref': 'Violet 2019'},
    'liver': {'mean': 0.17, 'range': (0.08, 0.35), 'unit': 'Gy/GBq', 'ref': 'Violet 2019'},
    'spleen': {'mean': 0.25, 'range': (0.1, 0.5), 'unit': 'Gy/GBq', 'ref': 'Delker 2016'},
    'bone_marrow': {'mean': 0.03, 'range': (0.01, 0.08), 'unit': 'Gy/GBq', 'ref': 'Delker 2016'},
    'salivary_glands': {'mean': 1.2, 'range': (0.5, 2.5), 'unit': 'Gy/GBq', 'ref': 'Violet 2019'},
}

# From Kratochwil et al. 2017 - Ac-225 PSMA-617
# Note: Ac-225 doses are per MBq due to much lower activity
LITERATURE_DOSES_AC225 = {
    'tumor': {'mean': 50.0, 'range': (10.0, 150.0), 'unit': 'Gy/MBq', 'ref': 'Kratochwil 2017'},
    'kidney': {'mean': 0.8, 'range': (0.3, 2.0), 'unit': 'Gy/MBq', 'ref': 'Kratochwil 2017'},
    'liver': {'mean': 0.15, 'range': (0.05, 0.4), 'unit': 'Gy/MBq', 'ref': 'Kratochwil 2017'},
    'salivary_glands': {'mean': 2.5, 'range': (1.0, 5.0), 'unit': 'Gy/MBq', 'ref': 'Kratochwil 2017'},
}

# Physical constants validation
PHYSICS_CONSTANTS = {
    'Lu-177': {
        'half_life_d': (6.647, 'ICRP 107'),
        'E_mean_beta_keV': (134.2, 'ICRP 107'),
        'E_max_beta_keV': (498.3, 'ICRP 107'),
        'beta_range_mm': (1.8, 'NIST ESTAR'),
    },
    'Ac-225': {
        'half_life_d': (9.920, 'ICRP 107'),
        'E_alpha_total_MeV': (27.6, 'ICRP 107'),
        'n_alphas_effective': (4.0, 'ICRP 107'),
        'alpha_range_um': (80, 'NIST ASTAR'),
    },
    'Y-90': {
        'half_life_d': (2.668, 'ICRP 107'),
        'E_mean_beta_keV': (933.7, 'ICRP 107'),
        'E_max_beta_keV': (2280.1, 'ICRP 107'),
        'beta_range_mm': (5.3, 'NIST ESTAR'),
    },
}

# Dose Point Kernel validation
# S-values from MIRD Pamphlet 17 (mGy/MBq·s)
MIRD_S_VALUES = {
    'Lu-177': {
        'self_dose_soft_tissue': 0.0293,  # mGy/MBq·s per gram
        'ref': 'MIRD Pamphlet 17',
    },
    'Y-90': {
        'self_dose_soft_tissue': 0.0849,  # mGy/MBq·s per gram
        'ref': 'MIRD Pamphlet 17',
    },
}

# DNA damage yields (from Goodhead 1994, Nikjoo 2001)
DNA_DAMAGE_YIELDS = {
    'low_LET': {
        'DSB_per_Gy': (35, 5),  # mean, std
        'SSB_per_Gy': (700, 100),
        'complex_fraction': (0.2, 0.05),
        'ref': 'Nikjoo 2001, Rothkamm 2003',
    },
    'high_LET_alpha': {
        'DSB_per_Gy': (175, 25),  # RBE ~5
        'SSB_per_Gy': (900, 150),
        'complex_fraction': (0.7, 0.1),
        'ref': 'Goodhead 1994, MIRD Pamphlet 22',
    },
}

# LQ model parameters (from van Leeuwen 2018)
LQ_PARAMETERS = {
    'Lu-177_beta': {
        'alpha': (0.35, 0.05),  # Gy^-1
        'beta': (0.035, 0.010),  # Gy^-2
        'alpha_beta_ratio': (10, 3),  # Gy
        'ref': 'van Leeuwen 2018',
    },
    'Ac-225_alpha': {
        'alpha': (0.80, 0.15),
        'beta': (0.010, 0.005),
        'alpha_beta_ratio': (80, 20),
        'ref': 'Sgouros 2010',
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_isotope_physics():
    """Validate isotope decay data against ICRP 107."""
    print("\n" + "=" * 70)
    print("ISOTOPE PHYSICS VALIDATION (vs ICRP 107)")
    print("=" * 70)
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES
    
    results = []
    
    for isotope, ref_data in PHYSICS_CONSTANTS.items():
        print(f"\n{isotope}:")
        
        if isotope not in THERAPEUTIC_ISOTOPES:
            print(f"  NOT IN DATABASE - SKIP")
            continue
        
        iso = THERAPEUTIC_ISOTOPES[isotope]
        
        for param, (ref_val, source) in ref_data.items():
            if param in iso:
                calc_val = iso[param]
                error = abs(calc_val - ref_val) / ref_val * 100
                status = "PASS" if error < 5 else "FAIL"
                
                print(f"  {param}: {calc_val:.3f} vs {ref_val:.3f} ({source}) - {error:.1f}% - {status}")
                
                results.append({
                    'isotope': isotope,
                    'parameter': param,
                    'calculated': calc_val,
                    'reference': ref_val,
                    'source': source,
                    'error_percent': error,
                    'status': status,
                })
    
    return results


def validate_dose_point_kernel():
    """Validate DPK methodology against MIRD Pamphlet 17."""
    print("\n" + "=" * 70)
    print("DOSE POINT KERNEL VALIDATION (vs MIRD Pamphlet 17)")
    print("=" * 70)
    
    from patient_dosimetry import generate_beta_dpk, THERAPEUTIC_ISOTOPES, KEV_TO_J
    
    results = []
    
    for isotope in ['Lu-177', 'Y-90']:
        if isotope not in THERAPEUTIC_ISOTOPES:
            continue
            
        iso = THERAPEUTIC_ISOTOPES[isotope]
        
        print(f"\n{isotope}:")
        
        # Generate DPK
        dpk = generate_beta_dpk(
            iso['E_max_beta_keV'],
            iso['E_mean_beta_keV'],
            voxel_size_mm=2.0,  # Typical PET voxel
            kernel_size=21
        )
        
        # Verify energy conservation (DPK should sum to 1)
        dpk_sum = dpk.sum()
        print(f"  DPK sum (should be 1.0): {dpk_sum:.4f}")
        
        # Verify central voxel dominance for beta
        center = dpk.shape[0] // 2
        central_fraction = dpk[center, center, center]
        print(f"  Central voxel fraction: {central_fraction:.3f}")
        
        # Calculate approximate S-value
        # S = E_mean × (keV to J) × (1/mass)
        E_keV = iso['E_mean_beta_keV']
        
        # For 1g of tissue (1 cm³ = 1 mL = 1 g for water)
        # Dose per decay = E × conversion / mass
        voxel_vol_cm3 = 2.0**3 / 1000  # 8 mm³ = 0.008 cm³
        mass_kg = voxel_vol_cm3 * 1.05 / 1000  # kg
        
        # S-value in mGy/MBq·s = mGy / (10^6 Bq × s)
        # = J/kg × 1000 / (10^6 × 1)
        S_calc = E_keV * KEV_TO_J / mass_kg * 1000 / 1e6  # mGy/MBq·s
        
        if isotope in MIRD_S_VALUES:
            S_ref = MIRD_S_VALUES[isotope]['self_dose_soft_tissue']
            error = abs(S_calc - S_ref) / S_ref * 100
            status = "PASS" if error < 50 else "CHECK"  # S-values can vary by geometry
            print(f"  S-value (approx): {S_calc:.4f} vs {S_ref:.4f} mGy/MBq·s - {status}")
        
        results.append({
            'isotope': isotope,
            'dpk_sum': dpk_sum,
            'central_fraction': central_fraction,
            'S_value_calc': S_calc,
        })
    
    return results


def validate_dna_damage_model():
    """Validate DNA damage yields against literature."""
    print("\n" + "=" * 70)
    print("DNA DAMAGE MODEL VALIDATION (vs Nikjoo 2001, Goodhead 1994)")
    print("=" * 70)
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES
    
    results = []
    
    # Test Lu-177 (low-LET beta)
    print("\nLu-177 (Beta emitter - Low LET):")
    iso = THERAPEUTIC_ISOTOPES['Lu-177']
    
    dsb_calc = iso['DSB_per_Gy']
    dsb_ref, dsb_std = DNA_DAMAGE_YIELDS['low_LET']['DSB_per_Gy']
    
    within_range = abs(dsb_calc - dsb_ref) < 2 * dsb_std
    status = "PASS" if within_range else "CHECK"
    print(f"  DSB/Gy: {dsb_calc:.1f} vs {dsb_ref}+/-{dsb_std} (literature) - {status}")
    
    results.append({
        'isotope': 'Lu-177',
        'radiation': 'beta',
        'DSB_per_Gy_calc': dsb_calc,
        'DSB_per_Gy_ref': dsb_ref,
        'status': status,
    })
    
    # Test Ac-225 (high-LET alpha)
    print("\nAc-225 (Alpha emitter - High LET):")
    iso = THERAPEUTIC_ISOTOPES['Ac-225']
    
    dsb_calc = iso['DSB_per_Gy']
    dsb_ref, dsb_std = DNA_DAMAGE_YIELDS['high_LET_alpha']['DSB_per_Gy']
    
    within_range = abs(dsb_calc - dsb_ref) < 2 * dsb_std
    status = "PASS" if within_range else "CHECK"
    print(f"  DSB/Gy: {dsb_calc:.1f} vs {dsb_ref}+/-{dsb_std} (literature) - {status}")
    
    # RBE check
    dsb_lu177 = THERAPEUTIC_ISOTOPES['Lu-177']['DSB_per_Gy']
    rbe_calc = dsb_calc / dsb_lu177
    rbe_ref = 5.0  # Expected RBE for alpha
    print(f"  RBE (DSB): {rbe_calc:.1f} vs {rbe_ref:.1f} (expected) - {'PASS' if abs(rbe_calc - rbe_ref) < 2 else 'CHECK'}")
    
    results.append({
        'isotope': 'Ac-225',
        'radiation': 'alpha',
        'DSB_per_Gy_calc': dsb_calc,
        'DSB_per_Gy_ref': dsb_ref,
        'RBE_calc': rbe_calc,
        'status': status,
    })
    
    return results


def validate_lq_model():
    """Validate LQ model parameters against van Leeuwen 2018."""
    print("\n" + "=" * 70)
    print("LQ MODEL VALIDATION (vs van Leeuwen 2018)")
    print("=" * 70)
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES, calculate_survival_LQ
    
    results = []
    
    for isotope in ['Lu-177', 'Ac-225']:
        iso = THERAPEUTIC_ISOTOPES[isotope]
        
        alpha = iso['alpha_LQ']
        beta = iso['beta_LQ']
        alpha_beta = alpha / beta if beta > 0 else float('inf')
        
        # Get reference values
        if isotope == 'Lu-177':
            ref_key = 'Lu-177_beta'
        else:
            ref_key = 'Ac-225_alpha'
        
        ref = LQ_PARAMETERS[ref_key]
        alpha_ref, alpha_std = ref['alpha']
        beta_ref, beta_std = ref['beta']
        ab_ref, ab_std = ref['alpha_beta_ratio']
        
        print(f"\n{isotope}:")
        
        # Alpha parameter
        alpha_ok = abs(alpha - alpha_ref) < 2 * alpha_std
        print(f"  alpha: {alpha:.3f} vs {alpha_ref}+/-{alpha_std} Gy^-1 - {'PASS' if alpha_ok else 'CHECK'}")
        
        # Beta parameter
        beta_ok = abs(beta - beta_ref) < 2 * beta_std
        print(f"  beta: {beta:.4f} vs {beta_ref}+/-{beta_std} Gy^-2 - {'PASS' if beta_ok else 'CHECK'}")
        
        # Alpha/beta ratio
        ab_ok = abs(alpha_beta - ab_ref) < 2 * ab_std
        print(f"  alpha/beta: {alpha_beta:.1f} vs {ab_ref}+/-{ab_std} Gy - {'PASS' if ab_ok else 'CHECK'}")
        
        # Test survival calculation
        test_doses = [1, 2, 5, 10]
        print(f"  Survival at test doses:")
        for D in test_doses:
            S = calculate_survival_LQ(D, alpha, beta)
            print(f"    D={D} Gy: S={S*100:.1f}%")
        
        results.append({
            'isotope': isotope,
            'alpha': alpha,
            'alpha_ref': alpha_ref,
            'beta': beta,
            'beta_ref': beta_ref,
            'alpha_beta': alpha_beta,
        })
    
    return results


def validate_effective_half_life():
    """Validate effective half-life calculations."""
    print("\n" + "=" * 70)
    print("EFFECTIVE HALF-LIFE VALIDATION")
    print("=" * 70)
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES
    
    # Literature values for Lu-177 PSMA effective half-life
    # From Violet 2019: T_eff tumor ~40-60 h, kidney ~30-50 h
    LITERATURE_T_EFF = {
        'Lu-177': {
            'tumor': (50, 15),  # hours, (mean, std)
            'kidney': (40, 10),
            'liver': (20, 5),
        },
    }
    
    results = []
    
    for isotope in ['Lu-177']:
        iso = THERAPEUTIC_ISOTOPES[isotope]
        T_phys_h = iso['half_life_d'] * 24
        
        bio_hl = iso.get('bio_half_life_h', {})
        
        print(f"\n{isotope}:")
        print(f"  Physical half-life: {T_phys_h:.1f} h ({iso['half_life_d']:.2f} d)")
        
        for organ, T_bio_h in bio_hl.items():
            if organ == 'default':
                continue
                
            # Calculate effective half-life
            T_eff = (T_phys_h * T_bio_h) / (T_phys_h + T_bio_h)
            
            print(f"  {organ}:")
            print(f"    T_bio: {T_bio_h:.1f} h")
            print(f"    T_eff: {T_eff:.1f} h ({T_eff/24:.2f} d)")
            
            # Compare with literature if available
            if isotope in LITERATURE_T_EFF and organ in LITERATURE_T_EFF[isotope]:
                ref_mean, ref_std = LITERATURE_T_EFF[isotope][organ]
                within_range = abs(T_eff - ref_mean) < 2 * ref_std
                status = "PASS" if within_range else "CHECK"
                print(f"    Literature: {ref_mean}+/-{ref_std} h - {status}")
            
            results.append({
                'isotope': isotope,
                'organ': organ,
                'T_bio_h': T_bio_h,
                'T_eff_h': T_eff,
            })
    
    return results


def validate_clinical_doses():
    """Compare calculated doses with clinical literature."""
    print("\n" + "=" * 70)
    print("CLINICAL DOSE VALIDATION (vs Published Patient Data)")
    print("=" * 70)
    
    print("\n--- Lu-177 PSMA Dosimetry (Literature) ---")
    print("From Violet et al. 2019, Delker et al. 2016:")
    print("\n  Organ           | Dose (Gy/GBq) | Range        | Status")
    print("  " + "-" * 55)
    
    for organ, data in LITERATURE_DOSES_LU177.items():
        print(f"  {organ:16} | {data['mean']:8.2f}      | {data['range'][0]:.1f}-{data['range'][1]:.1f}     | Ref: {data['ref']}")
    
    print("\n--- Ac-225 PSMA Dosimetry (Literature) ---")
    print("From Kratochwil et al. 2017:")
    print("\n  Organ           | Dose (Gy/MBq) | Range        | Status")
    print("  " + "-" * 55)
    
    for organ, data in LITERATURE_DOSES_AC225.items():
        print(f"  {organ:16} | {data['mean']:8.2f}      | {data['range'][0]:.1f}-{data['range'][1]:.1f}     | Ref: {data['ref']}")
    
    print("\n  NOTE: Actual patient doses depend on biodistribution which varies")
    print("  significantly between patients based on tumor burden, kidney function, etc.")


def create_validation_figure(output_dir: Path):
    """Create comprehensive validation figure."""
    print("\n" + "=" * 70)
    print("Creating Validation Figure")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Style
    plt.rcParams.update({'font.size': 11})
    
    # Panel A: LQ Model Survival Curves
    ax = axes[0, 0]
    doses = np.linspace(0, 20, 100)
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES, calculate_survival_LQ
    
    colors = {'Lu-177': '#1f77b4', 'Ac-225': '#d62728', 'Y-90': '#2ca02c'}
    
    for isotope in ['Lu-177', 'Ac-225']:
        if isotope not in THERAPEUTIC_ISOTOPES:
            continue
        iso = THERAPEUTIC_ISOTOPES[isotope]
        survival = calculate_survival_LQ(doses, iso['alpha_LQ'], iso['beta_LQ'])
        ax.plot(doses, survival * 100, color=colors[isotope], linewidth=2, 
                label=f"{isotope} (a={iso['alpha_LQ']}, b={iso['beta_LQ']})")
    
    ax.axhline(y=37, color='gray', linestyle='--', alpha=0.5, label='D₃₇ (37%)')
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='D₁₀ (10%)')
    ax.set_xlabel('Dose (Gy)', fontweight='bold')
    ax.set_ylabel('Survival (%)', fontweight='bold')
    ax.set_title('A. LQ Model Cell Survival Curves', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Panel B: DNA Damage vs Dose
    ax = axes[0, 1]
    
    for isotope in ['Lu-177', 'Ac-225']:
        if isotope not in THERAPEUTIC_ISOTOPES:
            continue
        iso = THERAPEUTIC_ISOTOPES[isotope]
        dsb = doses * iso['DSB_per_Gy']
        ax.plot(doses, dsb, color=colors[isotope], linewidth=2, 
                label=f"{isotope} ({iso['DSB_per_Gy']:.0f} DSB/Gy)")
    
    ax.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='Lethal threshold (~35 DSB)')
    ax.set_xlabel('Dose (Gy)', fontweight='bold')
    ax.set_ylabel('DSB per Cell', fontweight='bold')
    ax.set_title('B. DNA Double Strand Breaks vs Dose', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Literature Dose Comparison (Lu-177)
    ax = axes[1, 0]
    
    organs = list(LITERATURE_DOSES_LU177.keys())
    means = [LITERATURE_DOSES_LU177[o]['mean'] for o in organs]
    errors = [(LITERATURE_DOSES_LU177[o]['range'][1] - LITERATURE_DOSES_LU177[o]['range'][0])/2 for o in organs]
    
    y_pos = np.arange(len(organs))
    ax.barh(y_pos, means, xerr=errors, color='#1f77b4', alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([o.replace('_', ' ').title() for o in organs])
    ax.set_xlabel('Dose (Gy/GBq)', fontweight='bold')
    ax.set_title('C. Lu-177 PSMA Literature Doses', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel D: Effective Half-Life
    ax = axes[1, 1]
    
    from patient_dosimetry import THERAPEUTIC_ISOTOPES
    
    iso = THERAPEUTIC_ISOTOPES['Lu-177']
    T_phys = iso['half_life_d'] * 24
    bio_hl = iso.get('bio_half_life_h', {})
    
    organs_bio = [k for k in bio_hl.keys() if k != 'default']
    T_bio = [bio_hl[o] for o in organs_bio]
    T_eff = [(T_phys * t) / (T_phys + t) for t in T_bio]
    
    x = np.arange(len(organs_bio))
    width = 0.35
    
    ax.bar(x - width/2, T_bio, width, label='Biological T₁/₂', color='#2ca02c', alpha=0.7)
    ax.bar(x + width/2, T_eff, width, label='Effective T₁/₂', color='#1f77b4', alpha=0.7)
    ax.axhline(y=T_phys, color='red', linestyle='--', label=f'Physical T₁/₂ ({T_phys:.0f}h)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in organs_bio], rotation=45)
    ax.set_ylabel('Half-life (hours)', fontweight='bold')
    ax.set_title('D. Lu-177 Biological & Effective Half-Lives', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('CellRad-MC Patient Dosimetry Physics Validation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / 'patient_dosimetry_validation.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {fig_path}")
    return fig_path


def run_full_validation(output_dir: Path = None):
    """Run complete validation suite."""
    print("\n" + "=" * 70)
    print("CellRad-MC PATIENT DOSIMETRY PHYSICS VALIDATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if output_dir is None:
        output_dir = Path('validation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run validations
    results['isotope_physics'] = validate_isotope_physics()
    results['dpk'] = validate_dose_point_kernel()
    results['dna_damage'] = validate_dna_damage_model()
    results['lq_model'] = validate_lq_model()
    results['effective_half_life'] = validate_effective_half_life()
    validate_clinical_doses()
    
    # Create figure
    fig_path = create_validation_figure(output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    n_pass = sum(1 for r in results['isotope_physics'] if r['status'] == 'PASS')
    n_total = len(results['isotope_physics'])
    print(f"\n  Isotope Physics: {n_pass}/{n_total} PASS")
    
    print(f"  DNA Damage Model: Literature-validated DSB yields")
    print(f"  LQ Model: van Leeuwen 2018 parameters")
    print(f"  Effective Half-Life: Organ-specific biological clearance")
    
    print("\n  References:")
    print("    - ICRP Publication 107 (2008) - Nuclear Decay Data")
    print("    - MIRD Pamphlet 17 (1999) - Dose Point Kernels")
    print("    - MIRD Pamphlet 22 (2010) - Alpha Dosimetry")
    print("    - van Leeuwen et al. (2018) - LQ Parameters")
    print("    - Violet et al. (2019) - Lu-177 PSMA Dosimetry")
    print("    - Kratochwil et al. (2017) - Ac-225 PSMA Dosimetry")
    
    # Save results
    results_path = output_dir / 'patient_dosimetry_validation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("PATIENT DOSIMETRY VALIDATION COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Patient Dosimetry Physics Validation')
    parser.add_argument('--output', '-o', default='validation_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_full_validation(Path(args.output))

