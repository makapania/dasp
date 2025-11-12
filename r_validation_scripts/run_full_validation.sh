#!/bin/bash
# =============================================================================
# Full R Validation Workflow
# =============================================================================
#
# This script runs the complete validation workflow:
# 1. Generate sample data
# 2. Run Python tests
# 3. Run R comparisons
# 4. Compare results
#
# Usage:
#   bash r_validation_scripts/run_full_validation.sh
# =============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "R VALIDATION TEST SUITE - FULL WORKFLOW"
echo "================================================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Generate sample data
echo -e "\n${BLUE}Step 1: Generating sample spectral data...${NC}"
python r_validation_scripts/generate_sample_data.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sample data generated successfully${NC}"
else
    echo -e "${YELLOW}⚠ Sample data generation failed - may already exist${NC}"
fi

# Step 2: Run Python tests
echo -e "\n${BLUE}Step 2: Running Python validation tests...${NC}"
pytest tests/test_r_validation.py::test_generate_all_python_results -v -s
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python tests completed successfully${NC}"
else
    echo "✗ Python tests failed"
    exit 1
fi

# Step 3: Run R comparison scripts
echo -e "\n${BLUE}Step 3: Running R validation scripts...${NC}"

echo "  3a. PLS comparison..."
Rscript r_validation_scripts/pls_comparison.R > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "      ${GREEN}✓ PLS${NC}"
else
    echo "      ✗ PLS failed"
fi

echo "  3b. Random Forest comparison..."
Rscript r_validation_scripts/random_forest_comparison.R > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "      ${GREEN}✓ Random Forest${NC}"
else
    echo "      ✗ Random Forest failed"
fi

echo "  3c. XGBoost comparison..."
Rscript r_validation_scripts/xgboost_comparison.R > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "      ${GREEN}✓ XGBoost${NC}"
else
    echo "      ✗ XGBoost failed"
fi

echo "  3d. glmnet (Ridge/Lasso/ElasticNet) comparison..."
Rscript r_validation_scripts/glmnet_comparison.R > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "      ${GREEN}✓ glmnet${NC}"
else
    echo "      ✗ glmnet failed"
fi

# Step 4: Compare results
echo -e "\n${BLUE}Step 4: Comparing Python vs R results...${NC}"
python r_validation_scripts/compare_results.py --model all

echo ""
echo "================================================================================"
echo "VALIDATION WORKFLOW COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: r_validation_scripts/results/"
echo ""
echo "Review the comparison output above to verify that:"
echo "  ✓ PLS predictions match exactly (< 1e-10)"
echo "  ✓ XGBoost predictions match closely (< 1e-4)"
echo "  ✓ Random Forest predictions correlate highly (> 0.95)"
echo "  ✓ Ridge/Lasso/ElasticNet predictions match closely (< 1e-6)"
echo ""
