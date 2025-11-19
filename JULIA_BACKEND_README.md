# Julia Backend Migration - Project Overview

**Branch**: `claude/julia-backend-setup-01LPirVmjEYpWsDwn5ScAW7s`
**Date Started**: 2025-11-18
**Current Phase**: Phase 0 - Foundation & Assessment
**Status**: 🟡 In Progress

---

## 📋 Quick Navigation

| Document | Purpose | Status |
|----------|---------|--------|
| [JULIA_BACKEND_PLAN.md](./JULIA_BACKEND_PLAN.md) | 🎯 **Master plan** - 12-week migration strategy | ✅ Complete |
| [R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md](./R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md) | 📚 **Critical reading** - Lessons from 26+ hours debugging | ✅ Existing |
| [PHASE_0_CODEBASE_ANALYSIS.md](./PHASE_0_CODEBASE_ANALYSIS.md) | 🔍 **Codebase analysis** - Python architecture & bottlenecks | ✅ Complete |
| [JULIA_ECOSYSTEM_RESEARCH.md](./JULIA_ECOSYSTEM_RESEARCH.md) | 🧪 **Package evaluation** - Julia ecosystem assessment | 🟡 In Progress |
| [PROFILING_SCRIPT.py](./PROFILING_SCRIPT.py) | ⚡ **Performance profiling** - Identify bottlenecks | 📝 Template ready |

---

## 🎯 Mission

**Accelerate slow Python analyses while maintaining exacting R² reproducibility (±0.001 tolerance)**

**Challenge**: Previous Julia attempt failed - we must learn from those mistakes

**Approach**: Multi-agent team, incremental development, continuous validation

---

## ⚠️ Critical Requirements (Non-Negotiable)

### The 4 Inviolable Rules
1. **Preprocessing Order**: `full_spectrum → preprocess → filter` (NEVER filter first)
2. **State Restoration Order**: `restore → validate → test` (NEVER validate first)
3. **Metadata Flow**: `training_config` must transfer with models (NEVER skip)
4. **Feature Order**: Preserve wavelength order from DataFrame (NEVER sort)

### Success Criteria
- ✅ Derivative-only models: R² difference < 0.001
- ✅ Derivative+SNV models: R² difference < 0.001
- ✅ With wavelength restriction: R² difference < 0.001
- ✅ Without wavelength restriction: R² difference < 0.001
- ✅ With validation samples: R² difference < 0.001

**If any test fails by > 0.001, we STOP and debug!**

---

## 🏗️ Architecture Overview

### Multi-Agent Team

```
┌─────────────────────┐
│  Chief Architect    │  Top-down design, strategic decisions
│   (Agent 1)         │
└──────────┬──────────┘
           │
    ┌──────┴───────┬──────────────┬─────────────┐
    │              │              │             │
┌───▼────┐  ┌──────▼─────┐  ┌────▼──────┐  ┌──▼────────┐
│ Master │  │  Testing   │  │Implementation│  │Implementation│
│Debugger│  │ & Validation│  │  Agent(s)  │  │  Agent(s)  │
│(Agent 2)│  │  (Agent 3)  │  │ (Agent 4+) │  │ (Agent 4+) │
└────────┘  └────────────┘  └────────────┘  └────────────┘
```

**Key Principle**: Every change is validated by all agents before proceeding

---

## 📅 Timeline (12 Weeks)

| Phase | Duration | Goal | Status |
|-------|----------|------|--------|
| **Phase 0** | Week 1 | Foundation & Assessment | 🟡 **Current** |
| **Phase 1** | Week 2 | Proof of Concept (SNV) | ⏸️ Pending |
| **Phase 2** | Week 3-4 | Core Preprocessing | ⏸️ Pending |
| **Phase 3** | Week 5-6 | Model Training | ⏸️ Pending |
| **Phase 4** | Week 7 | State Management | ⏸️ Pending |
| **Phase 5** | Week 8-9 | Full Integration | ⏸️ Pending |
| **Phase 6** | Week 10-11 | Performance Optimization | ⏸️ Pending |
| **Phase 7** | Week 12 | Production Hardening | ⏸️ Pending |

---

## 🔍 Phase 0 Tasks (Current)

### ✅ Completed
- [x] Create branch (`claude/julia-backend-setup-01LPirVmjEYpWsDwn5ScAW7s`)
- [x] Read and digest handoff document (R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md)
- [x] Create master plan (JULIA_BACKEND_PLAN.md)
- [x] Analyze codebase structure (PHASE_0_CODEBASE_ANALYSIS.md)
- [x] Locate key files mentioned in handoff:
  - ✅ src/spectral_predict/preprocess.py (SNV, derivatives)
  - ✅ src/spectral_predict/search.py (wavelength filtering)
  - ✅ spectral_predict_gui_optimized.py (validation restoration)

### 🟡 In Progress
- [ ] Profile Python implementation (identify bottlenecks)
- [ ] Research Julia ecosystem (MLJ.jl, DSP.jl, etc.)
- [ ] Run numerical validation experiments:
  - [ ] Experiment 1: SNV (Julia vs Python)
  - [ ] Experiment 2: Savitzky-Golay (DSP.jl vs scipy)
  - [ ] Experiment 3: Ridge regression (GLM.jl vs scikit-learn)
  - [ ] Experiment 4: PLS regression (MultivariateStats.jl vs scikit-learn) ⚠️ HIGH RISK

### 📝 Pending
- [ ] Create Phase 0 assessment report
- [ ] Make GO/NO-GO decision
- [ ] Present findings to stakeholders

---

## 🚨 Risk Register

| Risk | Level | Mitigation |
|------|-------|-----------|
| **PLS numerical mismatch** | 🔴 **HIGH** | Use ScikitLearn.jl bridge if needed |
| **Julia packages unstable** | 🟡 Medium | Hybrid approach (Python + Julia) |
| **Indexing errors (0-based vs 1-based)** | 🟡 Medium | Extensive testing, code review |
| **Performance < 2x** | 🟡 Medium | Consider Numba/Cython instead |
| **Savitzky-Golay doesn't match** | 🟡 Medium | Use SciPy.jl bridge |

---

## 🎓 Lessons from Previous Attempt

From the handoff document, the previous Julia attempt failed due to:

1. **Preprocessing order was wrong** (filtered before preprocessing)
2. **Validation restored in wrong order** (validated before restoring)
3. **Wavelengths were sorted** (broke feature order reproducibility)
4. **Training config not transferred** (couldn't validate consistency)

**Result**: 1-3% R² discrepancies that couldn't be debugged

**Our Strategy**: Multi-agent validation prevents these mistakes!

---

## 🧪 Key Experiments (Phase 0)

### Experiment 1: SNV Numerical Precision ✅ Low Risk
**Goal**: Verify Julia can match Python SNV exactly
**Success**: Max difference < 1e-14 (Float64 precision)

### Experiment 2: Savitzky-Golay Match 🟡 Medium Risk
**Goal**: Verify DSP.jl matches scipy.signal.savgol_filter
**Success**: Max difference < 1e-12

### Experiment 3: Ridge Regression 🟡 Medium Risk
**Goal**: Verify GLM.jl matches scikit-learn Ridge
**Success**: R² difference < 0.001

### Experiment 4: PLS Regression 🔴 HIGH RISK
**Goal**: Verify MultivariateStats.jl matches scikit-learn PLS
**Success**: R² difference < 0.001
**Fallback**: Use ScikitLearn.jl if doesn't match

---

## 📊 Performance Targets

| Metric | Baseline (Python) | Minimum Target | Ideal Target | Stretch Goal |
|--------|-------------------|----------------|--------------|--------------|
| **Speedup** | 1x | 2x | 5-10x | 20-50x (GPU) |
| **Memory** | Baseline | < 2x | < 1x | < 0.5x |
| **R² Match** | N/A | < 0.001 | < 0.0001 | Bit-exact |

**Critical**: Correctness first, speed second!

---

## 🛠️ Development Setup

### Prerequisites
1. Julia (≥ 1.9)
2. Python (≥ 3.10) - for comparison testing
3. Git

### Installation (Julia packages)
```julia
using Pkg

# Core ML
Pkg.add("MLJ")
Pkg.add("GLM")
Pkg.add("MultivariateStats")

# Signal processing
Pkg.add("DSP")

# Data handling
Pkg.add("DataFrames")
Pkg.add("CSV")

# Testing & benchmarking
Pkg.add("Test")
Pkg.add("BenchmarkTools")

# Python interop (fallback)
Pkg.add("PyCall")
```

### Directory Structure
```
julia_experiments/     # Julia validation experiments
├── test_snv.jl
├── test_savgol.jl
├── test_ridge.jl
└── test_pls.jl

julia_backend/         # Julia implementation (Phase 1+)
├── src/
│   ├── preprocessing.jl
│   ├── models.jl
│   └── search.jl
└── test/
    └── runtests.jl
```

---

## 🚦 Decision Points

### ✅ PROCEED to Phase 1 if:
- All 4 experiments pass (R² < 0.001)
- Julia packages actively maintained
- Estimated speedup ≥ 2x
- No critical ecosystem gaps

### ⚠️ HYBRID APPROACH if:
- PLS doesn't match → Use ScikitLearn.jl bridge
- Some functionality missing → Python for missing parts
- Moderate performance gains (1.5x-2x)

### ❌ ABORT Julia migration if:
- Cannot achieve R² < 0.001 after 2 weeks
- Julia ecosystem too immature
- Performance worse than Python
- Critical packages abandoned

**Fallback Options**:
1. Numba (JIT compile Python hotspots)
2. Cython (C extensions for bottlenecks)
3. Rust + PyO3 (safer than Julia, harder to write)
4. Python algorithmic improvements

---

## 📚 Required Reading

**Before writing any Julia code**, read these in order:

1. **R2_REPRODUCIBILITY_HANDOFF_FOR_JULIA.md** (1 hour)
   - Understand the 4 critical bugs
   - Learn from the 9 stumbling points
   - Internalize: small differences matter!

2. **JULIA_BACKEND_PLAN.md** (30 min)
   - Understand the 7-phase strategy
   - Know the agent roles
   - Memorize abort criteria

3. **PHASE_0_CODEBASE_ANALYSIS.md** (30 min)
   - Understand Python architecture
   - Know PATH A vs PATH B
   - Identify critical sections

---

## 🔗 Key Code Sections (Python)

From handoff document:

### Issue #1 Fix: Post-Preprocessing Wavelength Filtering
**File**: `src/spectral_predict/search.py`
**Lines**: 529-557
**Critical**: Filter AFTER preprocessing, create local copies

### Issue #2 Fix: Validation Restoration Order
**File**: `spectral_predict_gui_optimized.py`
**Lines**: 12741+
**Critical**: Restore BEFORE validation check

### Issue #3 Fix: Training Config Transfer
**File**: `spectral_predict_gui_optimized.py`
**Lines**: 11143-11156 (cache), 11187-11207 (transfer)
**Critical**: Metadata flows with models

### Issue #4 Fix: Wavelength Order Preservation
**File**: `spectral_predict_gui_optimized.py`
**Lines**: 14347-14352
**Critical**: Never sort wavelengths!

---

## 📞 Communication Protocol

### Daily Standup (15 min)
- Chief Architect: Strategy, blockers
- Master Debugger: Deviations detected
- Testing Agent: Test status (red/green)
- Implementation Agents: Progress

### Weekly Review (1 hour)
- Phase completion assessment
- Risk register update
- GO/CONTINUE/ABORT decision

### Documentation
- **Deviation Log**: Track every discrepancy
- **Decision Log**: Record all choices with rationale
- **Performance Log**: Benchmark each phase
- **Lessons Log**: Capture insights in real-time

---

## 🎯 Current Priority

**Phase 0 Assessment** - Due: End of Week 1

**Next Actions**:
1. Run profiling script on Python code
2. Execute Julia experiments (SNV, Savitzky-Golay, Ridge, PLS)
3. Research Julia package maturity
4. Create assessment report with GO/NO-GO recommendation

---

## 📊 Success Metrics

### Must-Have (Required)
- ✅ All 5 reproducibility tests pass (R² < 0.001)
- ✅ Performance improvement ≥ 2x
- ✅ All 4 agents approve
- ✅ Validation checklist 100% complete

### Should-Have (Desired)
- ✅ Performance improvement ≥ 5x
- ✅ Code is maintainable
- ✅ Easy installation

### Nice-to-Have (Stretch)
- ✅ Performance improvement ≥ 20x (GPU)
- ✅ Test coverage > 90%
- ✅ Automated CI/CD

---

## 🆘 Help & Resources

**Julia Documentation**:
- https://docs.julialang.org
- https://alan-turing-institute.github.io/MLJ.jl/dev/ (MLJ)
- https://juliadsp.github.io/DSP.jl/stable/ (DSP)

**Community**:
- https://discourse.julialang.org (forum)
- https://julialang.slack.com (Slack)

**This Project**:
- Questions? Check the plan documents first
- Stuck? Consult handoff document
- Need decision? Escalate to Chief Architect

---

## 📜 License

Same as main project (MIT)

---

**Last Updated**: 2025-11-18
**Branch Status**: 🟢 Active Development
**Phase**: 0 - Foundation & Assessment
**Next Milestone**: Phase 0 completion & GO/NO-GO decision
