#!/bin/bash
# ============================================================================
# CellRad-MC Validation Suite
# ============================================================================
# Run all physics and isotope validations
# Outputs saved to validation_results/

set -e  # Exit on error

echo "=============================================="
echo "CellRad-MC Validation Suite"
echo "=============================================="
echo ""

# Create output directory
mkdir -p validation_results

# Run publication validation (electron, alpha, photon physics)
echo "[1/4] Running Publication Validation..."
python publication_validation.py 2>&1 | tee validation_results/publication_validation.log
echo "      Done!"
echo ""

# Run isotope-specific validation (ICRP 107)
echo "[2/4] Running Isotope Validation..."
python isotope_validation.py 2>&1 | tee validation_results/isotope_validation.log
echo "      Done!"
echo ""

# Run DNA damage validation (if available)
if [ -f "dna_damage_validation.py" ]; then
    echo "[3/4] Running DNA Damage Validation..."
    python dna_damage_validation.py 2>&1 | tee validation_results/dna_damage_validation.log
    echo "      Done!"
    echo ""
else
    echo "[3/4] Skipping DNA Damage Validation (file not found)"
    echo ""
fi

# Run track structure validation (if available)
if [ -f "track_structure_validation.py" ]; then
    echo "[4/4] Running Track Structure Validation..."
    python track_structure_validation.py 2>&1 | tee validation_results/track_structure_validation.log
    echo "      Done!"
    echo ""
else
    echo "[4/4] Skipping Track Structure Validation (file not found)"
    echo ""
fi

echo "=============================================="
echo "Validation Complete!"
echo "=============================================="
echo ""
echo "Results saved to: validation_results/"
echo ""
echo "Generated files:"
ls -la validation_results/*.png 2>/dev/null || echo "  (no PNG files)"
ls -la validation_results/*.csv 2>/dev/null || echo "  (no CSV files)"
ls -la validation_results/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""
echo "Check log files for details:"
ls -la validation_results/*.log 2>/dev/null || echo "  (no log files)"
