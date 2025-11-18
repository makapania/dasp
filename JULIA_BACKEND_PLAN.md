# Julia Backend Migration Plan: Multi-Agent Strategy for Exacting Reproducibility

**Date**: 2025-11-18
**Branch**: `claude/julia-backend-setup-01LPirVmjEYpWsDwn5ScAW7s`
**Status**: Phase 0 - Foundation & Assessment

---

## Executive Summary

**Challenge**: Accelerate slow Python analyses while maintaining bit-exact R² reproducibility (±0.001 tolerance)

**Risk Level**: **EXTREME** - Previous Julia attempt failed after extensive work

**Strategy**: Multi-agent team with continuous validation, incremental development, and immediate deviation correction

**Success Criteria**: All 5 reproducibility tests pass with R² < 0.001 difference

**Required Reading**: `R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md` - Documents 26+ hours of debugging that identified 4 critical bugs

---

## I. Multi-Agent Team Architecture

### Agent 1: Chief Architect
**Role**: Strategic oversight and top-down system design

**Core Responsibilities**:
1. **Architectural Decisions**:
   - Design Julia equivalent of Python implementation
   - Decide on parallelization strategies (multi-threading, distributed computing)
   - Select Julia packages (MLJ.jl, StatsBase.jl, DSP.jl for derivatives)
   - Define module boundaries and interfaces

2. **Performance Strategy**:
   - Identify computational bottlenecks in Python (needs profiling data)
   - Design optimized algorithms without breaking reproducibility
   - Plan memory-efficient data structures
   - Decide on SIMD/GPU opportunities

3. **Consistency Mapping**:
   - Maintain Python ↔ Julia correspondence document
   - Track every design decision with rationale
   - Monitor that PATH A and PATH B patterns are preserved
   - Ensure numerical precision matches (Float64 throughout)

4. **Gate-Keeping**:
   - Review all major implementation decisions
   - Approve/reject architectural changes
   - Escalate when team disagrees
   - Make abort/continue decisions at milestones

**Deliverables**:
- High-level architecture document
- Module interface specifications
- Performance optimization plan
- Python-Julia correspondence map

---

### Agent 2: Master Debugger
**Role**: Deviation detection and correctness enforcement

**Core Responsibilities**:
1. **Operational Principle Enforcement**:
   - Monitor for violations of the 4 critical patterns:
     - ❌ **NEVER** filter before preprocessing (Issue #1)
     - ❌ **NEVER** validate before restoration (Issue #2)
     - ❌ **NEVER** forget to transfer training_config (Issue #3)
     - ❌ **NEVER** sort wavelengths (Issue #4)

2. **Deviation Detection**:
   - Watch for the 6 mistake patterns from handoff:
     - Confusing correlation with causation
     - Assuming conventional wisdom
     - Fixing symptoms not root causes
     - Dismissing small discrepancies
     - Hidden assumptions
     - Order-of-operations blind spots

3. **Numerical Precision Tracking**:
   - Monitor for:
     - 0-based (Python) vs 1-based (Julia) indexing errors
     - Broadcasting semantics differences
     - Floating point precision issues
     - Random number generation differences

4. **Red Flag System**:
   - Immediately halt work when:
     - R² difference > 0.001 appears
     - Array mutation detected in loops
     - Wavelengths appear sorted
     - Preprocessing order is wrong

5. **Edge Case Testing**:
   - Create adversarial test cases:
     - Very small validation sets (1 sample)
     - All samples excluded except minimum
     - Single wavelength ranges
     - Extreme preprocessing parameters

**Deliverables**:
- Deviation log (tracks every discrepancy)
- Edge case test suite
- Numerical precision report
- Bug pattern detection rules

---

### Agent 3: Testing & Validation
**Role**: Continuous verification and quality assurance

**Core Responsibilities**:
1. **Reference Output Generation**:
   - Run Python implementation on canonical datasets
   - Record exact outputs (R², coefficients, predictions)
   - Store intermediate results (preprocessing, variable selection)
   - Create golden dataset for comparison

2. **Differential Testing**:
   - **Level 1**: Function-level (SNV output, derivative output, etc.)
   - **Level 2**: Module-level (preprocessing pipeline, model training)
   - **Level 3**: System-level (full Results → Model Dev workflow)
   - **Level 4**: Statistical properties (R² distribution, residuals)

3. **The 5 Critical Tests** (from handoff document):
   - Test 1: Derivative-only models
   - Test 2: Derivative+SNV models
   - Test 3: Wavelength restriction
   - Test 4: Validation samples
   - Test 5: Excluded samples

4. **Continuous Integration**:
   - Run tests after EVERY change
   - No exceptions - even "trivial" changes get tested
   - Maintain test dashboard (all green or STOP)

5. **Performance Benchmarking**:
   - Track speed improvements at each milestone
   - Ensure memory usage is reasonable
   - Profile Julia code to find bottlenecks
   - Compare against Python baseline

6. **Validation Gates**:
   - **Gate 1**: Numerical agreement (R² < 0.001)
   - **Gate 2**: Performance improvement documented
   - **Gate 3**: All edge cases pass
   - **Gate 4**: Code reviewed by all agents
   - **Gate 5**: Documentation complete

**Deliverables**:
- Golden reference dataset
- Automated test harness
- Test status dashboard
- Performance benchmark report
- Gate passage certificates

---

### Agent 4+: Implementation Agents
**Role**: Code translation and optimization

**Core Responsibilities**:
1. **Incremental Translation**:
   - Work in small, testable units
   - Never big-bang rewrites
   - Document every assumption
   - Flag uncertainties immediately

2. **Pattern Preservation**:
   - **PATH A** for derivatives (preprocess → select → train)
   - **PATH B** for other methods (select → preprocess → train)
   - Array mutation prevention (always copy)
   - Deep copies for cached data

3. **Code Documentation**:
   - Inline comments referencing Python line numbers
   - Docstrings explaining deviations (if any)
   - Link to handoff document sections
   - Explain Julia-specific patterns

4. **Collaboration**:
   - Submit work to Master Debugger for review
   - Respond to Chief Architect's design guidance
   - Fix issues flagged by Testing Agent
   - Never proceed if tests fail

**Deliverables**:
- Julia source code
- Code documentation
- Translation notes (Python → Julia)
- Performance optimization notes

---

## II. Development Phases

### Phase 0: Foundation (Week 1) - CURRENT PHASE

**Objective**: Understand the problem and prepare infrastructure

**Tasks**:
1. **Profile Python Implementation** (Critical!)
   - Run Python profiler on real analyses
   - Identify top 10 slowest functions
   - Measure memory usage
   - Document current runtime (baseline)
   - **Output**: "Here's what's slow and by how much"

2. **Julia Ecosystem Assessment**:
   - Evaluate MLJ.jl for model training
   - Find equivalent of scikit-learn preprocessing
   - Test DSP.jl for Savitzky-Golay derivatives
   - Verify numerical precision matches Python
   - Check package stability/maintenance
   - **Decision Point**: Is Julia ecosystem mature enough?

3. **Test Infrastructure Setup**:
   - Create golden reference dataset (Python outputs)
   - Set up automated testing framework
   - Build diff tool (Python vs Julia outputs)
   - Configure CI/CD pipeline
   - **Output**: "We can now test automatically"

4. **Alternative Evaluation** (Parallel Track):
   - Prototype Python optimization with Numba
   - Try JAX for automatic differentiation
   - Evaluate Cython for bottlenecks
   - Test PyO3/Rust for critical functions
   - **Decision Point**: Are alternatives faster/safer than Julia?

**Gates**:
- ✅ Profiling complete, bottlenecks identified
- ✅ Julia packages evaluated and approved
- ✅ Test infrastructure working
- ✅ Alternative approaches assessed
- ✅ GO/NO-GO decision made

**Abort Criteria**:
- Julia packages missing critical features
- Ecosystem appears unstable
- Alternatives (Numba/JAX) achieve adequate speedup with less risk

---

### Phase 1: Proof of Concept (Week 2)

**Objective**: Prove Julia can achieve bit-exact reproducibility on ONE function

**Strategy**: Pick the simplest, most critical function - **SNV preprocessing**

**Why SNV?**:
- Stateless, deterministic
- Critical to Issue #1 (preprocessing order)
- Easy to test (input array → output array)
- Mentioned in 3 of 9 stumbling points

**Implementation Steps**:
1. **Chief Architect**: Design SNV function signature
2. **Implementation Agent**: Translate Python SNV to Julia
3. **Testing Agent**: Create test cases (100 samples, 1000 wavelengths)
4. **Master Debugger**: Compare outputs element-wise

**Success Criteria**:
- Julia SNV output matches Python exactly (Float64 precision)
- Speed improvement measured (expect 2-10x for simple function)
- Edge cases pass (zero std dev, single sample, etc.)

**Gates**:
- ✅ SNV outputs match bit-exactly
- ✅ All edge cases pass
- ✅ Performance improvement documented
- ✅ All agents approve

**Abort Criteria**:
- Cannot achieve numerical match after 1 week
- Performance is slower than Python
- Julia numerical behavior is inconsistent

**If successful**: Proceed to Phase 2
**If failed**: Reassess Julia viability, consider alternatives

---

### Phase 2: Core Preprocessing (Week 3-4)

**Objective**: Implement full preprocessing pipeline with PATH A/B patterns

**Components**:
1. **Savitzky-Golay Derivatives** (DSP.jl)
2. **SNV** (from Phase 1)
3. **deriv_snv pipeline** (derivative → SNV order)
4. **Wavelength filtering** (AFTER preprocessing)

**Critical Patterns to Enforce**:
```julia
# PATH A: Derivative preprocessing
X_transformed = apply_derivative(X_full, wavelengths, deriv_params)
X_filtered = X_transformed[:, wavelength_mask]  # Filter AFTER

# PATH B: Other preprocessing
X_subset = X_full[:, selected_indices]  # Subset FIRST
pipeline = Pipeline(preprocessor, model)
```

**Testing**:
- Run on canonical dataset
- Compare intermediate outputs (derivative, SNV, filtered)
- Test with/without wavelength restriction
- Verify Issue #1 fix (preprocess → filter order)

**Gates**:
- ✅ All preprocessing functions match Python outputs
- ✅ PATH A and PATH B patterns implemented correctly
- ✅ Wavelength filtering happens AFTER preprocessing
- ✅ Edge cases pass (full spectrum, restricted, single wavelength)

---

### Phase 3: Model Training (Week 5-6)

**Objective**: Implement model training for deterministic models

**Models** (in order of complexity):
1. **Ridge** (simplest regularized model)
2. **PLS** (most common in spectroscopy)
3. **Lasso** (for comparison)
4. **ElasticNet** (combination of Ridge + Lasso)

**Start with Ridge**:
- Use MLJ.jl or GLM.jl
- Match hyperparameter tuning
- Ensure CV splitting matches Python exactly
- Verify coefficients match

**Critical Checks**:
- Same CV folds (use identical random seed)
- Same regularization path
- Same convergence criteria
- Coefficients match within numerical precision

**Testing**:
- Train on canonical dataset
- Compare: coefficients, predictions, R²
- Test with Issue #4 fix (wavelength order preserved)
- Run Test 1 (derivative-only models)

**Gates**:
- ✅ Ridge model training matches Python exactly
- ✅ R² difference < 0.001
- ✅ Coefficients match within Float64 precision
- ✅ Test 1 passes

---

### Phase 4: State Management (Week 7)

**Objective**: Implement validation/exclusion logic with correct restoration order

**Components**:
1. **training_config cache** (Issue #3)
2. **validation_indices restoration** (Issue #2)
3. **excluded_spectra restoration** (Issue #2)
4. **Validation checks** (run AFTER restoration)

**Critical Order**:
```julia
# CORRECT order (from handoff)
restore_validation_indices!(config)    # 1. Restore state
restore_excluded_spectra!(config)      # 2. Restore more state
validate_configuration(config)         # 3. THEN validate
filter_data_for_testing(data)         # 4. THEN test
```

**Testing**:
- Run Test 4 (validation samples)
- Run Test 5 (excluded samples)
- Verify correct sample counts
- Check that Issue #2 fix works

**Gates**:
- ✅ State restoration order is correct
- ✅ Test 4 and Test 5 pass
- ✅ Sample counts match exactly
- ✅ No false mismatch warnings

---

### Phase 5: Full Integration (Week 8-9)

**Objective**: End-to-end workflow from data import to model testing

**Workflow**:
```
Data Import → Preprocessing (PATH A/B) → Variable Selection →
Model Training → Config Caching → Model Transfer →
State Restoration → Validation → Model Dev Testing
```

**Testing**:
- Run all 5 critical tests
- Test with real datasets (not just canonical)
- Vary: models, preprocessing, wavelength ranges, validation sets
- Stress test with large datasets

**Gates**:
- ✅ All 5 tests pass with R² < 0.001
- ✅ Full validation checklist complete
- ✅ No deviations from operational principles
- ✅ Performance improvement demonstrated

---

### Phase 6: Performance Optimization (Week 10-11)

**Objective**: Make it fast (only AFTER correctness is proven)

**Optimization Strategies**:
1. **Multi-threading**: Parallelize model grid search
2. **SIMD**: Vectorize preprocessing operations
3. **Memory**: Reduce allocations, use views instead of copies where safe
4. **GPU**: Offload matrix operations if dataset is large enough
5. **Caching**: Memoize expensive computations

**Critical Rule**: **NEVER sacrifice correctness for speed**
- Re-run all 5 tests after EVERY optimization
- If R² breaks, revert immediately
- Document performance vs correctness tradeoffs

**Target Performance**:
- Minimum 2x speedup over Python (else not worth migration)
- Ideal: 5-10x speedup
- Stretch goal: 20-50x with GPU

**Gates**:
- ✅ Speedup ≥ 2x demonstrated
- ✅ All tests still pass after optimization
- ✅ Memory usage is reasonable
- ✅ Scalability verified

---

### Phase 7: Production Hardening (Week 12)

**Objective**: Make it production-ready

**Components**:
1. **Error Handling**: Graceful failures, informative messages
2. **Logging**: Diagnostic output matching Python verbosity
3. **Documentation**: User guide, API reference, migration guide
4. **Deployment**: Installation instructions, dependency management
5. **Monitoring**: Performance metrics, error tracking

**Gates**:
- ✅ Error handling comprehensive
- ✅ Documentation complete
- ✅ Deployment tested on fresh system
- ✅ User acceptance testing passed

---

## III. Communication & Coordination Protocol

### Daily Standup (15 minutes)
- **Chief Architect**: Strategic status, blockers, decisions needed
- **Master Debugger**: Deviations detected, issues flagged
- **Testing Agent**: Test status (green/red), gate passage
- **Implementation Agents**: Progress, questions, help needed

**Output**: Shared status document updated

---

### Weekly Milestone Review (1 hour)
- Review phase completion
- Assess against timeline
- Update risk register
- **Decision**: Continue / Pivot / Abort
- Adjust plan if needed

**Output**: Milestone report with GO/NO-GO decision

---

### Continuous Documentation
- **Deviation Log**: Every discrepancy, how it was resolved
- **Decision Log**: Every architectural decision, with rationale
- **Performance Log**: Benchmarks at each phase
- **Lessons Log**: Real-time capture of insights

**Purpose**: If we have to abort again, next team knows exactly why

---

## IV. Risk Management

### Critical Red Flags (Immediate Pause)
1. R² difference > 0.001 that can't be explained
2. Numerical instability in Julia packages
3. Array mutation bugs appearing repeatedly
4. Tests failing after "trivial" changes
5. Agents disagree on correctness
6. Performance degradation vs Python

**Response**:
- Halt all implementation work
- Root cause analysis by all agents
- Chief Architect decides: fix or abort

---

### Abort Criteria (Stop the project)
1. **Cannot achieve R² < 0.001** after 2 weeks of focused debugging
2. **Julia packages are unstable** or lack critical features
3. **Performance improvement < 2x** (not worth migration cost)
4. **Critical bugs** that can't be resolved without rewriting packages
5. **Timeline slips by > 4 weeks** (diminishing returns)

**Fallback Options** (in order of preference):
1. **Hybrid approach**: Keep Python, accelerate bottlenecks with Julia via PyCall/PythonCall
2. **Python optimization**: Numba for hot loops, Cython for extensions
3. **Partial migration**: Only migrate proven components (e.g., just preprocessing)
4. **Different language**: Rust + PyO3 (safer than Julia, harder to write)
5. **Stay with Python**: Focus on algorithmic improvements instead

---

## V. Key Technical Constraints from Handoff

### The 4 Inviolable Rules
1. **Preprocessing Order**: `full_spectrum → preprocess → filter` (NEVER filter first)
2. **State Restoration Order**: `restore → validate → test` (NEVER validate first)
3. **Metadata Flow**: `training_config` must transfer with models (NEVER skip)
4. **Feature Order**: Preserve wavelength order from DataFrame (NEVER sort)

### The 9 Stumbling Points to Avoid
1. ❌ Don't assume wavelength compression is the issue
2. ❌ Don't assume SNV is buggy when derivative+SNV fails
3. ❌ Don't assume it's a random seed issue if difference is consistent
4. ❌ Don't assume feature order doesn't matter
5. ❌ Don't validate before restoring state
6. ❌ Don't fix symptoms (Model Dev) instead of root cause (Results)
7. ❌ Don't try to fix design flaws with more parameters
8. ❌ Don't assume logic is correct just because whole spectrum works
9. ❌ Don't dismiss small R² differences

### PATH A vs PATH B
```julia
# PATH A: For derivatives (need full spectral context)
X_transformed = preprocess_full_spectrum(X)
X_subset = select_variables(X_transformed)
model = train(X_subset)

# PATH B: For other methods (can work on subsets)
X_subset = select_variables(X)
model = train_with_pipeline(X_subset, preprocessor)
```

### Array Mutation Prevention
```julia
# WRONG
for model in models
    for preprocess in methods
        X_local = X[:, mask]  # Shared reference!
    end
end

# CORRECT
for model in models
    for preprocess in methods
        X_local = copy(X[:, mask])  # New array each time
    end
end
```

---

## VI. Success Metrics

### Must-Have (Required for deployment)
- ✅ All 5 reproducibility tests pass (R² < 0.001)
- ✅ Validation checklist 100% complete
- ✅ Performance improvement ≥ 2x
- ✅ All 4 agents approve deployment
- ✅ User acceptance testing passed

### Should-Have (Desired outcomes)
- ✅ Performance improvement ≥ 5x
- ✅ Memory usage < Python
- ✅ Code is maintainable (well-documented)
- ✅ Installation is easy (one command)

### Nice-to-Have (Stretch goals)
- ✅ Performance improvement ≥ 20x (with GPU)
- ✅ Comprehensive test coverage (>90%)
- ✅ Automated deployment pipeline
- ✅ Performance monitoring dashboard

---

## VII. Phase 0 Execution Plan (Current)

### Step 1: Codebase Exploration
**Goal**: Locate all files mentioned in handoff document

**Key Files to Find**:
- `spectral_predict_gui_optimized.py` (GUI, validation logic)
- `src/spectral_predict/search.py` (model training, preprocessing)
- `src/spectral_predict/preprocess.py` (SNV, derivatives)

**Key Line Numbers to Review** (from handoff):
- spectral_predict_gui_optimized.py:
  - Lines 10244-10304: Wavelength restriction parsing
  - Lines 11143-11156: Training data cache storage
  - Lines 11187-11207: Training config attachment
  - Lines 12216-12252: Validation/excluded indices restoration
  - Lines 13537-13558: Model Dev data exclusion
  - Lines 14347-14352: Wavelength ordering preservation

- src/spectral_predict/search.py:
  - Lines 529-557: Post-preprocessing wavelength filtering
  - Lines 502-524: PATH A preprocessing
  - Lines 565-608: Variable selection
  - Lines 648-689: Model training

- src/spectral_predict/preprocess.py:
  - Lines 8-40: SNV implementation
  - Lines 145-147: deriv_snv pipeline

### Step 2: Architecture Analysis
**Goal**: Understand the current system

**Questions to Answer**:
- What's the overall architecture?
- How do Results tab and Model Dev tab interact?
- What are the data structures?
- What models are supported?
- What preprocessing methods are available?

### Step 3: Performance Profiling
**Goal**: Identify what's slow

**Method**:
- Use cProfile or line_profiler on Python code
- Measure runtime of key functions
- Identify top 10 bottlenecks
- Estimate potential speedup

### Step 4: Julia Ecosystem Research
**Goal**: Determine if Julia is viable

**Packages to Evaluate**:
- MLJ.jl - Machine learning framework
- GLM.jl - Linear models (Ridge, Lasso)
- MultivariateStats.jl - PLS regression
- DSP.jl - Signal processing (Savitzky-Golay)
- DataFrames.jl - Data manipulation
- CSV.jl - Data I/O

**Tests to Run**:
- Install packages
- Test numerical precision vs Python
- Verify Savitzky-Golay matches scipy
- Check if SNV can be implemented identically

### Step 5: Create Assessment Report
**Goal**: GO/NO-GO decision for Julia

**Report Contents**:
- Bottleneck analysis
- Julia ecosystem readiness
- Risk assessment
- Recommendation (proceed / pivot / abort)

---

## VIII. Next Steps

Once Phase 0 is complete, we will:
1. Make GO/NO-GO decision
2. If GO: Proceed to Phase 1 (SNV proof of concept)
3. If NO-GO: Evaluate alternatives (Numba, Cython, Rust)
4. If uncertain: Run more experiments

---

## IX. References

- **Handoff Document**: `R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md`
- **Key Commits**:
  - 8a48ec2: Fixed SNV/derivative preprocessing order
  - 277333e: Added training configuration transfer
  - af7da75: Fixed validation sample restoration order
  - a033439: Added training data cache

- **Related Branches**:
  - `archive/julia-backend`: Previous failed attempt (lessons learned)
  - `main`: Current stable Python implementation

---

## X. Decision Log

| Date | Decision | Rationale | Who |
|------|----------|-----------|-----|
| 2025-11-18 | Use multi-agent architecture | Previous attempt failed; need systematic approach | Chief Architect |
| 2025-11-18 | Start with Phase 0 assessment | Reduce risk; validate viability before coding | Chief Architect |
| 2025-11-18 | Target R² < 0.001 tolerance | Handoff document shows this is critical | Master Debugger |
| 2025-11-18 | Branch name: claude/julia-backend-setup-01LPirVmjEYpWsDwn5ScAW7s | Match session requirements | Implementation |

---

**Status**: Phase 0 in progress
**Next Milestone**: Phase 0 assessment complete
**Target Date**: End of Week 1
