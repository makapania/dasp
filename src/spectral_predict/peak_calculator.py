"""FTIR Peak Ratio Calculator — backend module.

Provides dataclasses for peak definitions and presets, built-in preset packs
(Bone FTIR, Mineralogical, Collagen & Tissue, DNA & Nucleic Acids,
Edible Oils & Fats, Food Matrices, Plant Cell Wall, Plant Composition),
calculation functions, local baseline correction for published indices,
and user-preset I/O.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaselineRegion:
    """Two-point local baseline defined by search windows for trough positions.

    Each window is a (low, high) wavenumber range within which the algorithm
    searches for a local minimum to use as a baseline anchor point.  A linear
    baseline is interpolated between the two detected troughs and subtracted
    before measuring peak height.
    """
    left_min: float = 0.0   # low end of left trough search window (cm-1)
    left_max: float = 0.0   # high end of left trough search window
    right_min: float = 0.0  # low end of right trough search window
    right_max: float = 0.0  # high end of right trough search window


@dataclass
class PeakDefinition:
    """A single peak position in the expression.

    Modes:
        "point"  — intensity at the nearest wavelength to *wavenumber*.
        "range"  — maximum intensity within [wavenumber ± half_width].
        "search_max" — find actual local maximum nearest *wavenumber*
                       within ± half_width (handles peak shifts).
        "search_min" — find actual local minimum nearest *wavenumber*
                       within ± half_width (for trough measurements).
    """
    wavenumber: float = 0.0
    mode: str = "point"        # "point", "range", "search_max", "search_min"
    half_width: float = 10.0   # spectral units (cm-1 or nm)
    label: str = ""
    baseline: Optional[BaselineRegion] = None  # per-peak local baseline


@dataclass
class PeakPreset:
    """A saved peak-ratio expression (built-in or user-defined)."""
    name: str = ""
    peak_a: PeakDefinition = field(default_factory=PeakDefinition)
    peak_b: PeakDefinition = field(default_factory=PeakDefinition)
    operator1: str = "/"       # operator between A and B (or grouped pair and C)
    peak_c: Optional[PeakDefinition] = None
    operator2: str = "/"       # operator between the grouped pair and C
    grouping: str = "left"     # "left" = (A op1 B) op2 C, "right" = A op1 (B op2 C)
    description: str = ""
    category: str = "User"
    x_unit: str = "cm-1"      # "cm-1" or "nm"


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Common baseline regions for published bone/enamel indices
# ---------------------------------------------------------------------------

# v4 PO4 region: troughs at ~490-510 and ~690-750 cm-1
# (Weiner & Bar-Yosef 1990; Surovell & Stiner 2001 use ~500-700;
#  Grunenwald et al. 2018 confirm ~495-750 as standard)
_BL_V4_PO4 = BaselineRegion(490, 510, 690, 750)

# v3 CO3 region: troughs at ~1290-1340 and ~1530-1590 cm-1
# (Wright & Schwarcz 1996)
_BL_V3_CO3 = BaselineRegion(1290, 1340, 1530, 1590)

# v3 PO4 region: troughs at ~880-900 and ~1150-1180 cm-1
# (Wright & Schwarcz 1996)
_BL_V3_PO4 = BaselineRegion(880, 900, 1150, 1180)

# Amide I region: troughs at ~1590 and ~1700-1720 cm-1
# (Trueman et al.; Snoeck & Pellegrini 2015)
_BL_AMIDE_I = BaselineRegion(1590, 1600, 1700, 1720)

# Collagen maturity sub-region within Amide I
_BL_COLLAGEN_MATURITY = BaselineRegion(1590, 1600, 1700, 1720)


BUILT_IN_PRESETS: list[PeakPreset] = [
    # --- Bone FTIR ---
    PeakPreset(
        name="Mineral:Matrix",
        peak_a=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        peak_b=PeakDefinition(1660, "search_max", 15, "Amide I", baseline=_BL_AMIDE_I),
        operator1="/",
        description="Phosphate v3 / Amide I — mineral-to-matrix ratio (auto peak search, local baselines)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carbonate:Phosphate",
        peak_a=PeakDefinition(1415, "search_max", 15, "v3 CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        operator1="/",
        description=(
            "Carbonate v3 / Phosphate v3 — C/P ratio "
            "(Wright & Schwarcz 1996, auto peak search, local baselines)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Crystallinity Index",
        peak_a=PeakDefinition(604, "search_max", 10, "604", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(564, "search_max", 10, "564", baseline=_BL_V4_PO4),
        operator1="+",
        peak_c=PeakDefinition(590, "search_min", 10, "590 trough", baseline=_BL_V4_PO4),
        operator2="/",
        grouping="left",
        description=(
            "(604 + 564) / 590 — IRSF apatite crystallinity "
            "(Weiner & Bar-Yosef 1990, auto peak search, local baseline 490-750 cm⁻¹)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Collagen Maturity",
        peak_a=PeakDefinition(1660, "search_max", 10, "1660 mature", baseline=_BL_COLLAGEN_MATURITY),
        peak_b=PeakDefinition(1690, "search_max", 10, "1690 immature", baseline=_BL_COLLAGEN_MATURITY),
        operator1="/",
        description="1660 / 1690 — mature / immature cross-links (auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Acid Phosphate",
        peak_a=PeakDefinition(1128, "search_max", 15, "HPO4", baseline=_BL_V3_PO4),
        peak_b=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        operator1="/",
        description="1128 / 1035 — acid phosphate content (auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Mineralogical ---
    PeakPreset(
        name="Si-O / OH stretch",
        peak_a=PeakDefinition(1030, "range", 30, "Si-O stretch"),
        peak_b=PeakDefinition(3620, "range", 30, "OH stretch"),
        operator1="/",
        description="Silicate Si-O stretch / OH stretch",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carbonate / Silicate",
        peak_a=PeakDefinition(1430, "range", 30, "CO3 v3"),
        peak_b=PeakDefinition(1030, "range", 30, "Si-O stretch"),
        operator1="/",
        description="Carbonate v3 / Silicate Si-O — carbonate vs silicate content",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Quartz / Feldspar",
        peak_a=PeakDefinition(798, "point", 10, "Quartz doublet"),
        peak_b=PeakDefinition(727, "point", 10, "Feldspar"),
        operator1="/",
        description="Quartz / Feldspar relative abundance",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Kaolinite / Illite",
        peak_a=PeakDefinition(3620, "point", 10, "Kaolinite inner OH"),
        peak_b=PeakDefinition(3432, "point", 10, "Illite OH"),
        operator1="/",
        description="Clay weathering index — 1:1 vs 2:1 layer clays",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Calcite / Dolomite",
        peak_a=PeakDefinition(713, "point", 10, "Calcite v4"),
        peak_b=PeakDefinition(728, "point", 10, "Dolomite v4"),
        operator1="/",
        description="v4 bending mode discrimination — calcite vs dolomite",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Sulfate / Carbonate",
        peak_a=PeakDefinition(1120, "point", 10, "SO4 v3"),
        peak_b=PeakDefinition(1430, "point", 10, "CO3 v3"),
        operator1="/",
        description="Evaporite vs carbonate index",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Goethite / Hematite",
        peak_a=PeakDefinition(892, "point", 10, "Goethite OH bend"),
        peak_b=PeakDefinition(470, "point", 10, "Hematite Fe-O"),
        operator1="/",
        description="Iron oxide hydration state (FeOOH vs Fe2O3)",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Kerogen Maturity",
        peak_a=PeakDefinition(2920, "point", 10, "CH2 asym stretch"),
        peak_b=PeakDefinition(1600, "point", 10, "Aromatic C=C"),
        operator1="/",
        description="Aliphatic/aromatic — decreases with thermal maturity",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Kerogen Chain Length",
        peak_a=PeakDefinition(2920, "point", 10, "CH2 asym stretch"),
        peak_b=PeakDefinition(2850, "point", 10, "CH2 sym stretch"),
        operator1="/",
        description="CH2 asym/sym — reflects aliphatic chain character",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Volcanic Glass Hydration",
        peak_a=PeakDefinition(3620, "point", 10, "OH stretch"),
        peak_b=PeakDefinition(1030, "point", 10, "Si-O stretch"),
        operator1="/",
        description="Palagonitization — OH / Si-O ratio",
        category="Mineralogical",
        x_unit="cm-1",
    ),
    # --- Collagen & Tissue ---
    PeakPreset(
        name="Collagen Denaturation",
        peak_a=PeakDefinition(1633, "point", 10, "Random coil"),
        peak_b=PeakDefinition(1660, "point", 10, "Triple helix"),
        operator1="/",
        description="Triple helix to gelatin conversion (FTIR)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Proteoglycan:Collagen",
        peak_a=PeakDefinition(1240, "point", 10, "SO3 stretch"),
        peak_b=PeakDefinition(1338, "point", 10, "Collagen CH2 wag"),
        operator1="/",
        description="Cartilage matrix composition (FTIR)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Collagen Network",
        peak_a=PeakDefinition(1338, "point", 10, "Proline CH2"),
        peak_b=PeakDefinition(1660, "point", 10, "Amide I"),
        operator1="/",
        description="Type II collagen structural integrity (FTIR)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Alpha-Helix / Beta-Sheet",
        peak_a=PeakDefinition(1270, "point", 10, "Alpha-helix"),
        peak_b=PeakDefinition(1240, "point", 10, "Beta-sheet"),
        operator1="/",
        description="Secondary structure in collagen (Raman)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Proline / Hydroxyproline",
        peak_a=PeakDefinition(855, "point", 10, "Proline"),
        peak_b=PeakDefinition(938, "point", 10, "Hydroxyproline"),
        operator1="/",
        description="Post-translational modification status (Raman)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Collagen Helical Integrity",
        peak_a=PeakDefinition(1245, "point", 10, "Less-ordered"),
        peak_b=PeakDefinition(1270, "point", 10, "Ordered helix"),
        operator1="/",
        description="Post-yield damage biomarker (Raman)",
        category="Collagen & Tissue",
        x_unit="cm-1",
    ),
    # --- DNA & Nucleic Acids ---
    PeakPreset(
        name="Phosphodiester B-DNA",
        peak_a=PeakDefinition(1090, "point", 10, "PO2- sym stretch"),
        peak_b=PeakDefinition(1240, "point", 10, "PO2- asym stretch"),
        operator1="/",
        description="B-form dominance indicator (FTIR)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="DNA:Protein Ratio",
        peak_a=PeakDefinition(1080, "point", 10, "Nucleic acid PO4"),
        peak_b=PeakDefinition(1658, "point", 10, "Amide I"),
        operator1="/",
        description="Cell proliferation status (FTIR)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="RNA / DNA Marker",
        peak_a=PeakDefinition(1238, "point", 10, "RNA ribose"),
        peak_b=PeakDefinition(1080, "point", 10, "DNA deoxyribose"),
        operator1="/",
        description="RNA vs DNA discrimination (FTIR)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="DNA Conformation B/Z",
        peak_a=PeakDefinition(785, "point", 10, "B-form marker"),
        peak_b=PeakDefinition(729, "point", 10, "Z-form marker"),
        operator1="/",
        description="Helical conformation discrimination (Raman)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="DNA Damage Index",
        peak_a=PeakDefinition(1650, "point", 10, "Thymine C=O post-UV"),
        peak_b=PeakDefinition(1580, "point", 10, "Baseline thymine"),
        operator1="/",
        description="Cyclobutane pyrimidine dimer detection (Raman)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    # --- Edible Oils & Fats ---
    PeakPreset(
        name="Iodine Value (C=C/CH2)",
        peak_a=PeakDefinition(1654, "point", 10, "C=C stretch"),
        peak_b=PeakDefinition(1465, "point", 10, "CH2 scissor"),
        operator1="/",
        description="C=C / CH2 scissor — unsaturation proxy (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Iodine Value (cis CH)",
        peak_a=PeakDefinition(3010, "point", 10, "=C-H cis stretch"),
        peak_b=PeakDefinition(2924, "point", 10, "CH2 asym stretch"),
        operator1="/",
        description="=C-H cis / CH2 stretch — palm oil IV (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Total Unsaturation",
        peak_a=PeakDefinition(1654, "point", 10, "C=C stretch"),
        peak_b=PeakDefinition(2900, "range", 100, "CH2+CH3 region"),
        operator1="/",
        description="C=C / sum CH region — double bond content (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carbonyl Index",
        peak_a=PeakDefinition(1743, "point", 10, "C=O ester stretch"),
        peak_b=PeakDefinition(1465, "point", 10, "CH2 scissor"),
        operator1="/",
        description="Primary oxidation products in oils (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="OH/CH Oxidation",
        peak_a=PeakDefinition(3400, "point", 10, "OH stretch"),
        peak_b=PeakDefinition(2889, "range", 35, "CH2+CH3 region"),
        operator1="/",
        description="Secondary oxidation — hydroperoxides (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Trans / Cis",
        peak_a=PeakDefinition(966, "point", 10, "Trans =C-H bend"),
        peak_b=PeakDefinition(3010, "point", 10, "Cis =C-H stretch"),
        operator1="/",
        description="Trans =C-H / cis =C-H content ratio (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Adulteration Marker",
        peak_a=PeakDefinition(3009, "point", 10, "=C-H cis stretch"),
        peak_b=PeakDefinition(2924, "point", 10, "CH2 asym stretch"),
        operator1="/",
        description="Olive oil adulteration screening (FTIR)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Unsaturation (Raman)",
        peak_a=PeakDefinition(1655, "point", 10, "C=C stretch"),
        peak_b=PeakDefinition(1440, "point", 10, "CH2 scissor"),
        operator1="/",
        description="C=C / CH2 scissor — Raman unsaturation proxy",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Trans Fat (Raman)",
        peak_a=PeakDefinition(970, "point", 10, "Trans =C-H bend"),
        peak_b=PeakDefinition(2885, "range", 35, "CH2 stretch region"),
        operator1="/",
        description="Trans =C-H / CH2 stretch — margarine/shortening",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Solid Fat Crystallinity",
        peak_a=PeakDefinition(715, "range", 5, "Lattice mode"),
        peak_b=PeakDefinition(1440, "point", 10, "CH2 scissor"),
        operator1="/",
        description="Lattice / CH2 — polymorph/crystallinity (Raman)",
        category="Edible Oils & Fats",
        x_unit="cm-1",
    ),
    # --- Food Matrices ---
    PeakPreset(
        name="Starch Order/Disorder",
        peak_a=PeakDefinition(1047, "point", 10, "Ordered starch"),
        peak_b=PeakDefinition(1022, "point", 10, "Amorphous starch"),
        operator1="/",
        description="Ordered/amorphous starch index (FTIR)",
        category="Food Matrices",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Starch Crystallinity",
        peak_a=PeakDefinition(480, "point", 10, "Skeletal mode"),
        peak_b=PeakDefinition(2900, "point", 10, "CH stretch"),
        operator1="/",
        description="Order/disorder in starch (Raman)",
        category="Food Matrices",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Lipid:Carbohydrate (dairy)",
        peak_a=PeakDefinition(1740, "point", 10, "C=O ester"),
        peak_b=PeakDefinition(1080, "range", 40, "C-O stretch region"),
        operator1="/",
        description="Fat vs carbohydrate in milk powders (Raman)",
        category="Food Matrices",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Lipid:Protein (meat)",
        peak_a=PeakDefinition(2885, "range", 35, "CH2 stretch region"),
        peak_b=PeakDefinition(1650, "point", 10, "Amide I"),
        operator1="/",
        description="Species discrimination / quality (FTIR)",
        category="Food Matrices",
        x_unit="cm-1",
    ),
    # --- Plant Cell Wall ---
    PeakPreset(
        name="Lignin:Cellulose",
        peak_a=PeakDefinition(1515, "range", 5, "Lignin aromatic"),
        peak_b=PeakDefinition(1040, "range", 10, "Cellulose C-O-C"),
        operator1="/",
        description="Relative lignin vs cellulose in biomass (FTIR)",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Hemicellulose:Cellulose",
        peak_a=PeakDefinition(1740, "point", 10, "Hemicellulose C=O"),
        peak_b=PeakDefinition(1040, "range", 10, "Cellulose C-O-C"),
        operator1="/",
        description="Hemicellulose C=O / cellulose C-O-C (FTIR)",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="S/G Lignin (FTIR)",
        peak_a=PeakDefinition(1327, "point", 10, "Syringyl ring"),
        peak_b=PeakDefinition(1269, "point", 10, "Guaiacyl ring"),
        operator1="/",
        description="Syringyl:guaiacyl ratio — hardwood vs softwood",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Polysaccharide:Lignin",
        peak_a=PeakDefinition(1162, "point", 10, "C-O-C glycosidic"),
        peak_b=PeakDefinition(1269, "point", 10, "Guaiacyl ring"),
        operator1="/",
        description="Nano-FTIR component mapping",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Lignin:Cellulose (Raman)",
        peak_a=PeakDefinition(1600, "point", 10, "Lignin aromatic"),
        peak_b=PeakDefinition(1095, "point", 10, "Cellulose C-O-C"),
        operator1="/",
        description="Lignification mapping in stems (Raman)",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="S/G Lignin (Raman)",
        peak_a=PeakDefinition(1330, "point", 10, "Syringyl ring"),
        peak_b=PeakDefinition(1270, "point", 10, "Guaiacyl ring"),
        operator1="/",
        description="Syringyl:guaiacyl via Raman",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Cellulose Crystallinity",
        peak_a=PeakDefinition(1481, "point", 10, "Crystalline CH2 bend"),
        peak_b=PeakDefinition(1462, "point", 10, "Amorphous CH2 bend"),
        operator1="/",
        description="Crystalline/amorphous CH2 bend (Raman)",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Cellulose I / II",
        peak_a=PeakDefinition(577, "point", 10, "Cellulose II"),
        peak_b=PeakDefinition(380, "point", 10, "Cellulose I"),
        operator1="/",
        description="Cellulose II (mercerized) / cellulose I (Raman)",
        category="Plant Cell Wall",
        x_unit="cm-1",
    ),
    # --- Plant Composition ---
    PeakPreset(
        name="Protein:Carbohydrate",
        peak_a=PeakDefinition(1650, "point", 10, "Amide I"),
        peak_b=PeakDefinition(1075, "range", 75, "C-O stretch region"),
        operator1="/",
        description="Metabolic shifts under stress (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Lipid:Carbohydrate",
        peak_a=PeakDefinition(2920, "point", 10, "CH2 asym stretch"),
        peak_b=PeakDefinition(1040, "range", 10, "Cellulose C-O-C"),
        operator1="/",
        description="Lipid vs carbohydrate storage (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Plant Water Status",
        peak_a=PeakDefinition(3300, "range", 100, "OH stretch region"),
        peak_b=PeakDefinition(2920, "point", 10, "CH2 asym stretch"),
        operator1="/",
        description="Tissue hydration vs structural biomass (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Plant Wax / Cuticle",
        peak_a=PeakDefinition(2916, "point", 10, "Aliphatic CH2"),
        peak_b=PeakDefinition(1736, "point", 10, "Ester C=O"),
        operator1="/",
        description="Cuticular wax — aliphatic / ester (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Pectin Esterification",
        peak_a=PeakDefinition(1740, "point", 10, "Ester COOCH3"),
        peak_b=PeakDefinition(1740, "point", 10, "Ester COOCH3"),
        operator1="/",
        peak_c=PeakDefinition(1630, "point", 10, "Free COO-"),
        operator2="+",
        grouping="right",
        description="A / (B+C) — degree of methyl-esterification in pectin (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Starch Organization",
        peak_a=PeakDefinition(1022, "point", 10, "Amorphous starch"),
        peak_b=PeakDefinition(995, "point", 10, "Ordered starch"),
        operator1="/",
        description="Structural organization state (FTIR)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Pectin Content",
        peak_a=PeakDefinition(856, "point", 10, "Pectin-specific"),
        peak_b=PeakDefinition(1100, "point", 10, "Total polysaccharide"),
        operator1="/",
        description="Pectin-specific / total polysaccharide (Raman)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carotenoid:Chlorophyll",
        peak_a=PeakDefinition(1525, "point", 10, "Carotenoid C=C"),
        peak_b=PeakDefinition(1604, "point", 10, "Chlorophyll"),
        operator1="/",
        description="Senescence/stress monitoring (Raman)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carotenoid Structure",
        peak_a=PeakDefinition(1525, "point", 10, "Carotenoid C=C"),
        peak_b=PeakDefinition(1156, "point", 10, "Carotenoid C-C"),
        operator1="/",
        description="Conjugation length fingerprint (Raman)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phenolic:Carotenoid",
        peak_a=PeakDefinition(1602, "point", 10, "Flavonoid ring"),
        peak_b=PeakDefinition(1525, "point", 10, "Carotenoid C=C"),
        operator1="/",
        description="Flavonoid vs carotenoid content (Raman)",
        category="Plant Composition",
        x_unit="cm-1",
    ),
    # --- Bone FTIR (additional indices from Colmenares-Prado et al. 2026) ---
    PeakPreset(
        name="B-Type Carbonate (BPI)",
        peak_a=PeakDefinition(1415, "search_max", 15, "v3 B-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description=(
            "B-type CO3 / v4 PO4 — BPI "
            "(Sponheimer & Lee-Thorp 1999, auto peak search, local baselines)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="A-Type Carbonate (API)",
        peak_a=PeakDefinition(1540, "search_max", 15, "A-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description="A-type CO3 / v4 PO4 — A-carbonate substitution (auto peak search, local baselines)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="A/B Carbonate (C/C)",
        peak_a=PeakDefinition(1445, "search_max", 15, "A-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(1415, "search_max", 15, "B-CO3", baseline=_BL_V3_CO3),
        operator1="/",
        description="Type A / Type B carbonate substitution ratio (Snoeck et al. 2014, auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Protein:Carbonate",
        peak_a=PeakDefinition(1650, "search_max", 15, "Amide I", baseline=_BL_AMIDE_I),
        peak_b=PeakDefinition(1415, "search_max", 15, "v3 B-CO3", baseline=_BL_V3_CO3),
        operator1="/",
        description="Protein relative to carbonate content (Thompson et al. 2013, auto peak search, local baselines)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="OH:Phosphate",
        peak_a=PeakDefinition(630, "search_max", 10, "OH libration", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description="Hydroxyl content relative to v4 PO4 (Snoeck et al. 2014, auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Cyanamide:Phosphate",
        peak_a=PeakDefinition(2010, "search_max", 15, "Cyanamide"),
        peak_b=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        operator1="/",
        description="Cyanamide formation from burning / v3 PO4 (Zazzo et al. 2013, auto peak search)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phosphate High Temp (PHT)",
        peak_a=PeakDefinition(625, "search_max", 10, "PO4 HT", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(610, "search_max", 10, "PO4 ref", baseline=_BL_V4_PO4),
        operator1="/",
        description="High-temperature phosphate indicator (Thompson et al. 2013, auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phosphate Valley (TPV)",
        peak_a=PeakDefinition(1088, "search_max", 10, "v3 PO4 shoulder", baseline=_BL_V3_PO4),
        peak_b=PeakDefinition(1077, "search_min", 10, "v3 PO4 valley", baseline=_BL_V3_PO4),
        operator1="/",
        description="v3 PO4 sharpening at ~1088 — thermal alteration (Colmenares-Prado et al. 2026, auto peak search, local baseline)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Bone FTIR (Amide/Phosphate) ---
    PeakPreset(
        name="Amide:Phosphate",
        peak_a=PeakDefinition(1640, "search_max", 15, "v1 Amide", baseline=_BL_AMIDE_I),
        peak_b=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        operator1="/",
        description=(
            "Amide I / Phosphate v3 — Am/P ratio "
            "(Trueman et al., auto peak search, local baselines 1590-1720 and 880-1180 cm⁻¹)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Bone FTIR (WAMPI — collagen screening) ---
    PeakPreset(
        name="WAMPI",
        peak_a=PeakDefinition(1640, "search_max", 15, "Water+Amide", baseline=_BL_AMIDE_I),
        peak_b=PeakDefinition(604, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description=(
            "Water-Amide on Phosphate Index — collagen preservation screening "
            "(Snoeck & Pellegrini 2015, local baselines)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Bone FTIR (Amide I / Amide II) ---
    PeakPreset(
        name="AmI/AmII",
        peak_a=PeakDefinition(1660, "search_max", 15, "Amide I", baseline=_BL_AMIDE_I),
        peak_b=PeakDefinition(1540, "search_max", 15, "Amide II", baseline=_BL_V3_CO3),
        operator1="/",
        description=(
            "Amide I / Amide II — collagen structural integrity "
            "(France et al. 2020, local baselines)"
        ),
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Enamel FTIR ---
    PeakPreset(
        name="Enamel IRSF",
        peak_a=PeakDefinition(560, "search_max", 10, "v4 PO4 560", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4 600", baseline=_BL_V4_PO4),
        operator1="+",
        peak_c=PeakDefinition(595, "search_min", 10, "v4 PO4 valley", baseline=_BL_V4_PO4),
        operator2="/",
        grouping="left",
        description=(
            "(560 + 600) / 595 — enamel IRSF crystallinity "
            "(Weiner & Bar-Yosef 1990, auto peak search, local baseline 490-750 cm⁻¹)"
        ),
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel C/P",
        peak_a=PeakDefinition(1415, "search_max", 15, "v3 CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(1035, "search_max", 25, "v3 PO4", baseline=_BL_V3_PO4),
        operator1="/",
        description="Carbonate / phosphate — enamel diagenesis indicator (auto peak search, local baselines)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel BPI",
        peak_a=PeakDefinition(1415, "search_max", 15, "v3 B-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description="B-type carbonate / v4 PO4 — enamel B-carbonate content (auto peak search, local baselines)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel API",
        peak_a=PeakDefinition(1540, "search_max", 15, "A-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description="A-type carbonate / v4 PO4 — enamel A-carbonate content (auto peak search, local baselines)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel A/B Carbonate",
        peak_a=PeakDefinition(1445, "search_max", 15, "A-CO3", baseline=_BL_V3_CO3),
        peak_b=PeakDefinition(1415, "search_max", 15, "B-CO3", baseline=_BL_V3_CO3),
        operator1="/",
        description="Type A / Type B carbonate substitution in enamel (auto peak search, local baseline)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel OH/P",
        peak_a=PeakDefinition(630, "search_max", 10, "OH libration", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(600, "search_max", 10, "v4 PO4", baseline=_BL_V4_PO4),
        operator1="/",
        description="Hydroxyl content — fluoride substitution indicator in enamel (auto peak search, local baseline)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel PHT",
        peak_a=PeakDefinition(625, "search_max", 10, "PO4 HT", baseline=_BL_V4_PO4),
        peak_b=PeakDefinition(610, "search_max", 10, "PO4 ref", baseline=_BL_V4_PO4),
        operator1="/",
        description="High-temperature phosphate indicator in enamel (auto peak search, local baseline)",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    # --- Silicate Glass (FTIR — network polymerization and water speciation) ---
    PeakPreset(
        name="NBO/BO Ratio",
        peak_a=PeakDefinition(960, "range", 10, "NBO Si-O"),
        peak_b=PeakDefinition(1075, "range", 25, "BO Si-O-Si"),
        operator1="/",
        description="Non-bridging / bridging oxygen — higher = less polymerized network (FTIR)",
        category="Silicate Glass",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Polymerization Index",
        peak_a=PeakDefinition(925, "range", 25, "Q2/Q3"),
        peak_b=PeakDefinition(1100, "range", 50, "Q4"),
        operator1="/",
        description="Q2-3 / Q4 silicate species — calibrated against NBO/T from NMR (FTIR)",
        category="Silicate Glass",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="H-bonded/Free OH",
        peak_a=PeakDefinition(3295, "range", 115, "Strongly H-bonded OH"),
        peak_b=PeakDefinition(3550, "range", 50, "Free/weakly bound OH"),
        operator1="/",
        description="Strongly H-bonded vs free OH groups in silicate glass (FTIR)",
        category="Silicate Glass",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="SiOH/H2O Speciation",
        peak_a=PeakDefinition(3560, "range", 40, "(Si)O-H stretch"),
        peak_b=PeakDefinition(3300, "range", 100, "Molecular H2O"),
        operator1="/",
        description="(Si)O-H vs molecular water — water speciation in glass (FTIR)",
        category="Silicate Glass",
        x_unit="cm-1",
    ),
    # --- Clay Minerals (FTIR / Raman) ---
    PeakPreset(
        name="Dioctahedral/Trioctahedral",
        peak_a=PeakDefinition(875, "range", 75, "Dioctahedral OH bend"),
        peak_b=PeakDefinition(650, "range", 50, "Trioctahedral OH bend"),
        operator1="/",
        description="Dioctahedral vs trioctahedral phyllosilicate contribution (FTIR)",
        category="Clay Minerals",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Kaolinite Inner/Surface OH",
        peak_a=PeakDefinition(914, "point", 10, "Inner Al2OH bend"),
        peak_b=PeakDefinition(936, "point", 10, "Surface OH"),
        operator1="/",
        description="Inner vs surface OH — structural disorder and intercalation (FTIR)",
        category="Clay Minerals",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Al2OH/AlMgOH",
        peak_a=PeakDefinition(915, "point", 10, "Al2OH bend"),
        peak_b=PeakDefinition(842, "point", 10, "AlMgOH bend"),
        operator1="/",
        description="Octahedral cation substitution — Al vs Mg (FTIR)",
        category="Clay Minerals",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="OH/Si-O Framework",
        peak_a=PeakDefinition(875, "range", 75, "OH bending"),
        peak_b=PeakDefinition(1045, "range", 15, "Si-O stretch"),
        operator1="/",
        description="Structural OH vs Si-O framework — dehydroxylation tracking (FTIR)",
        category="Clay Minerals",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Clay/Quartz",
        peak_a=PeakDefinition(150, "range", 50, "Clay low-freq"),
        peak_b=PeakDefinition(465, "point", 10, "Quartz main band"),
        operator1="/",
        description="Clay vs quartz contribution in sediments (Raman)",
        category="Clay Minerals",
        x_unit="cm-1",
    ),
    # --- DNA & Nucleic Acids (additional) ---
    PeakPreset(
        name="DNA:Protein Amide II",
        peak_a=PeakDefinition(1080, "point", 10, "Sym PO2-"),
        peak_b=PeakDefinition(1545, "point", 10, "Amide II"),
        operator1="/",
        description="DNA phosphate / amide II — cell DNA content marker (FTIR)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="DNA:Protein (Raman)",
        peak_a=PeakDefinition(788, "point", 10, "DNA phosphate"),
        peak_b=PeakDefinition(1003, "point", 10, "Phenylalanine"),
        operator1="/",
        description="DNA backbone / phenylalanine — single-cell DNA marker (Raman)",
        category="DNA & Nucleic Acids",
        x_unit="cm-1",
    ),
    # --- Tissue & Cell Biology ---
    PeakPreset(
        name="Protein:Lipid (FTIR)",
        peak_a=PeakDefinition(1655, "point", 10, "Amide I"),
        peak_b=PeakDefinition(2885, "range", 35, "CH2 stretch"),
        operator1="/",
        description="Protein vs lipid content in serum or tissue (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Protein:Lipid (Raman)",
        peak_a=PeakDefinition(1660, "point", 10, "Amide I"),
        peak_b=PeakDefinition(1445, "point", 10, "CH2 scissoring"),
        operator1="/",
        description="Protein vs lipid in cells or tissue sections (Raman)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Protein:CH Stretch",
        peak_a=PeakDefinition(1650, "point", 10, "Amide I"),
        peak_b=PeakDefinition(2900, "range", 100, "CH stretch envelope"),
        operator1="/",
        description="Amide I / total CH stretch — global protein normalization (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Beta-Sheet/Alpha-Helix",
        peak_a=PeakDefinition(1625, "range", 5, "Beta-sheet amide I"),
        peak_b=PeakDefinition(1654, "range", 4, "Alpha-helix amide I"),
        operator1="/",
        description="Beta-sheet vs alpha-helix — protein misfolding and aggregation (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Lipid Unsaturation (FTIR)",
        peak_a=PeakDefinition(3010, "point", 10, "=C-H stretch"),
        peak_b=PeakDefinition(2885, "range", 35, "CH2 stretch"),
        operator1="/",
        description="Olefinic =C-H / CH2 — unsaturation in intact cells and tissues (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Membrane Packing",
        peak_a=PeakDefinition(2850, "point", 10, "Sym CH2"),
        peak_b=PeakDefinition(2880, "point", 10, "Sym CH3"),
        operator1="/",
        description="CH2/CH3 symmetric stretch — membrane order / microviscosity proxy",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Glycogen:Protein (FTIR)",
        peak_a=PeakDefinition(1055, "range", 25, "Glycogen C-O-C"),
        peak_b=PeakDefinition(1650, "point", 10, "Amide I"),
        operator1="/",
        description="Glycogen storage / metabolism marker in cells (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Glycogen:Lipid (Raman)",
        peak_a=PeakDefinition(480, "point", 10, "Glycogen 480"),
        peak_b=PeakDefinition(940, "point", 10, "Glycogen 940"),
        operator1="+",
        peak_c=PeakDefinition(1445, "point", 10, "CH2 scissoring"),
        operator2="/",
        grouping="left",
        description="(480 + 940) / 1445 — glycogen vs lipid in single cells (Raman)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phosphate:Lipid",
        peak_a=PeakDefinition(1080, "point", 10, "PO2- sym stretch"),
        peak_b=PeakDefinition(2885, "range", 35, "CH2 stretch"),
        operator1="/",
        description="DNA phosphate / lipid CH2 — diagnostic DNA/lipid marker (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="CH2:CH3 Balance",
        peak_a=PeakDefinition(2850, "point", 10, "CH2 stretch"),
        peak_b=PeakDefinition(2950, "point", 10, "CH3 stretch"),
        operator1="/",
        description="Lipid chain length / packing — changes in tumor vs normal tissue (FTIR)",
        category="Tissue & Cell Biology",
        x_unit="cm-1",
    ),
]


# ---------------------------------------------------------------------------
# Local baseline correction functions
# ---------------------------------------------------------------------------

def find_peak_in_window(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    target_wn: float,
    half_width: float = 10.0,
    smooth_window: int = 9,
    find_max: bool = True,
) -> tuple[float, float]:
    """Find the actual peak (max) or trough (min) nearest a target wavenumber.

    Searches within [target_wn - half_width, target_wn + half_width] for the
    local extremum.  A light Savitzky-Golay smooth is applied before finding
    the extremum to reduce noise, but the *raw* intensity is returned.
    Works with both ascending and descending wavenumber arrays.

    Args:
        wavelengths: Wavenumber axis (any order).
        spectrum: Intensity values.
        target_wn: Expected peak/trough position.
        half_width: Search radius in spectral units.
        smooth_window: SG smoothing window for robust extremum detection.
        find_max: True to find maximum (peak), False to find minimum (trough).

    Returns:
        (wavenumber_at_extremum, raw_intensity_at_extremum)
    """
    lo = target_wn - half_width
    hi = target_wn + half_width
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if not mask.any():
        idx = int(np.argmin(np.abs(wavelengths - target_wn)))
        logger.warning(
            "No data in search window [%.1f, %.1f]; using nearest point at %.1f",
            lo, hi, float(wavelengths[idx]),
        )
        return float(wavelengths[idx]), float(spectrum[idx])

    subset_idx = np.where(mask)[0]
    subset_vals = spectrum[subset_idx]

    # SG smoothing: apply if we have at least 5 points (the absolute minimum)
    if len(subset_vals) >= 5:
        sw = min(smooth_window, len(subset_vals))
        if sw % 2 == 0:
            sw -= 1
        sw = max(sw, 5)
        polyorder = min(2, sw - 1)
        smoothed = savgol_filter(subset_vals, sw, polyorder)
    else:
        smoothed = subset_vals

    if find_max:
        local_idx = int(np.argmax(smoothed))
    else:
        local_idx = int(np.argmin(smoothed))

    abs_idx = subset_idx[local_idx]
    return float(wavelengths[abs_idx]), float(spectrum[abs_idx])


def find_trough_in_window(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    window_lo: float,
    window_hi: float,
    smooth_window: int = 9,
) -> tuple[float, float]:
    """Find the minimum intensity position within a wavenumber window.

    Convenience wrapper around :func:`find_peak_in_window` with
    ``find_max=False``.
    """
    centre = (window_lo + window_hi) / 2.0
    half = (window_hi - window_lo) / 2.0
    return find_peak_in_window(
        wavelengths, spectrum, centre, half, smooth_window, find_max=False,
    )


def baseline_at_wavenumber(
    peak_wn: float,
    left_wn: float,
    left_val: float,
    right_wn: float,
    right_val: float,
) -> float:
    """Linearly interpolate the two-point baseline at a given wavenumber."""
    if right_wn == left_wn:
        return (left_val + right_val) / 2.0
    frac = (peak_wn - left_wn) / (right_wn - left_wn)
    return left_val + frac * (right_val - left_val)


def get_baseline_corrected_intensity(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peak_def: PeakDefinition,
    smooth_window: int = 9,
) -> tuple[float, dict | None]:
    """Return baseline-corrected peak intensity and trough diagnostics.

    If the peak has no baseline region, falls back to raw measurement.

    Returns:
        (corrected_intensity, diagnostics_dict_or_None)
        diagnostics keys: left_wn, left_val, right_wn, right_val,
                          baseline_at_peak, raw_intensity
    """
    if peak_def.baseline is None:
        return get_peak_intensity(wavelengths, spectrum, peak_def), None

    bl = peak_def.baseline

    # Find baseline trough anchors, or reuse pre-computed ones
    left_wn, left_val = find_trough_in_window(
        wavelengths, spectrum, bl.left_min, bl.left_max, smooth_window,
    )
    right_wn, right_val = find_trough_in_window(
        wavelengths, spectrum, bl.right_min, bl.right_max, smooth_window,
    )

    # For search_max/search_min modes, get the *found* wavenumber so the
    # baseline is interpolated at the actual peak position, not the nominal.
    if peak_def.mode in ("search_max", "search_min"):
        found_wn, raw_val = find_peak_in_window(
            wavelengths, spectrum, peak_def.wavenumber, peak_def.half_width,
            smooth_window, find_max=(peak_def.mode == "search_max"),
        )
    else:
        found_wn = peak_def.wavenumber
        raw_val = get_peak_intensity(wavelengths, spectrum, peak_def)

    bl_val = baseline_at_wavenumber(found_wn, left_wn, left_val, right_wn, right_val)
    corrected = raw_val - bl_val

    diag = {
        "left_wn": left_wn,
        "left_val": left_val,
        "right_wn": right_wn,
        "right_val": right_val,
        "baseline_at_peak": bl_val,
        "raw_intensity": raw_val,
        "found_wn": found_wn,
    }
    return corrected, diag


# ---------------------------------------------------------------------------
# Calculation functions
# ---------------------------------------------------------------------------

def get_peak_intensity(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peak_def: PeakDefinition,
) -> float:
    """Return the intensity for a single peak definition.

    Point mode: intensity at the nearest wavelength.
    Range mode: maximum intensity within [center ± half_width].
    search_max mode: find actual local maximum near target within ± half_width.
    search_min mode: find actual local minimum near target within ± half_width.
    """
    _, val = get_peak_intensity_with_position(wavelengths, spectrum, peak_def)
    return val


def get_peak_intensity_with_position(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peak_def: PeakDefinition,
) -> tuple[float, float]:
    """Return (found_wavenumber, intensity) for a single peak definition.

    Like :func:`get_peak_intensity` but also returns the actual wavenumber
    where the measurement was taken, which may differ from the nominal
    ``peak_def.wavenumber`` depending on mode:

    - point: nearest grid point wavenumber
    - range: wavenumber of the maximum within [center ± half_width]
    - search_max/search_min: actual extremum position from peak search
    """
    wn = peak_def.wavenumber
    if peak_def.mode == "range":
        lo = wn - peak_def.half_width
        hi = wn + peak_def.half_width
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        if not mask.any():
            idx = int(np.argmin(np.abs(wavelengths - wn)))
            return float(wavelengths[idx]), float(spectrum[idx])
        subset_idx = np.where(mask)[0]
        best = int(np.argmax(spectrum[subset_idx]))
        abs_idx = subset_idx[best]
        return float(wavelengths[abs_idx]), float(spectrum[abs_idx])
    elif peak_def.mode == "search_max":
        return find_peak_in_window(
            wavelengths, spectrum, wn, peak_def.half_width, find_max=True,
        )
    elif peak_def.mode == "search_min":
        return find_peak_in_window(
            wavelengths, spectrum, wn, peak_def.half_width, find_max=False,
        )
    else:
        idx = int(np.argmin(np.abs(wavelengths - wn)))
        return float(wavelengths[idx]), float(spectrum[idx])


def _apply_op(a: float, b: float, op: str) -> float:
    """Apply an arithmetic operator."""
    if op == "/":
        return a / b if b != 0 else float("nan")
    if op == "*":
        return a * b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    raise ValueError(f"Unknown operator: {op}")


def calculate_expression(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    preset: PeakPreset,
    use_local_baseline: bool = True,
) -> float:
    """Evaluate the peak expression for a single spectrum.

    Two-peak: ``A op1 B``
    Three-peak, grouping="left":  ``(A op1 B) op2 C``
    Three-peak, grouping="right": ``A op1 (B op2 C)``

    When *use_local_baseline* is True (default), peaks that carry a
    ``BaselineRegion`` will be baseline-corrected before the arithmetic.
    """
    if use_local_baseline:
        val_a, _ = get_baseline_corrected_intensity(wavelengths, spectrum, preset.peak_a)
        val_b, _ = get_baseline_corrected_intensity(wavelengths, spectrum, preset.peak_b)
    else:
        val_a = get_peak_intensity(wavelengths, spectrum, preset.peak_a)
        val_b = get_peak_intensity(wavelengths, spectrum, preset.peak_b)

    if preset.peak_c is None:
        return _apply_op(val_a, val_b, preset.operator1)

    if use_local_baseline:
        val_c, _ = get_baseline_corrected_intensity(wavelengths, spectrum, preset.peak_c)
    else:
        val_c = get_peak_intensity(wavelengths, spectrum, preset.peak_c)

    if preset.grouping == "left":
        grouped = _apply_op(val_a, val_b, preset.operator1)
        return _apply_op(grouped, val_c, preset.operator2)
    else:
        grouped = _apply_op(val_b, val_c, preset.operator2)
        return _apply_op(val_a, grouped, preset.operator1)


def _precompute_troughs(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peaks: list[PeakDefinition],
    smooth_window: int = 9,
) -> dict[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Pre-compute baseline troughs so shared regions are only searched once.

    Returns a cache mapping (left_min, left_max, right_min, right_max) to
    (left_wn, left_val, right_wn, right_val).
    """
    cache: dict[tuple, tuple] = {}
    for p in peaks:
        if p is None or p.baseline is None:
            continue
        bl = p.baseline
        key = (bl.left_min, bl.left_max, bl.right_min, bl.right_max)
        if key not in cache:
            left_wn, left_val = find_trough_in_window(
                wavelengths, spectrum, bl.left_min, bl.left_max, smooth_window,
            )
            right_wn, right_val = find_trough_in_window(
                wavelengths, spectrum, bl.right_min, bl.right_max, smooth_window,
            )
            cache[key] = (left_wn, left_val, right_wn, right_val)
    return cache


def _get_corrected_with_cache(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peak_def: PeakDefinition,
    trough_cache: dict,
    smooth_window: int = 9,
) -> tuple[float, dict]:
    """Baseline-corrected intensity using pre-computed trough cache."""
    if peak_def.baseline is None:
        found_wn, val = get_peak_intensity_with_position(wavelengths, spectrum, peak_def)
        return val, {"found_wn": found_wn, "raw_intensity": val}

    bl = peak_def.baseline
    key = (bl.left_min, bl.left_max, bl.right_min, bl.right_max)
    left_wn, left_val, right_wn, right_val = trough_cache[key]

    found_wn, raw_val = get_peak_intensity_with_position(
        wavelengths, spectrum, peak_def,
    )

    bl_val = baseline_at_wavenumber(found_wn, left_wn, left_val, right_wn, right_val)
    corrected = raw_val - bl_val

    diag = {
        "left_wn": left_wn, "left_val": left_val,
        "right_wn": right_wn, "right_val": right_val,
        "baseline_at_peak": bl_val, "raw_intensity": raw_val,
        "found_wn": found_wn,
    }
    return corrected, diag


def calculate_expression_detailed(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    preset: PeakPreset,
) -> dict:
    """Like calculate_expression but returns full diagnostics.

    Peaks sharing the same BaselineRegion reuse the same trough positions,
    ensuring consistent baseline anchors.

    Returns dict with keys: value, peak_a_diag, peak_b_diag, peak_c_diag.
    Each diag is a dict containing at least ``found_wn`` and ``raw_intensity``.
    Peaks with baselines also include trough positions and baseline values.
    ``peak_c_diag`` is ``None`` when the preset has no third peak.
    """
    peaks = [preset.peak_a, preset.peak_b]
    if preset.peak_c is not None:
        peaks.append(preset.peak_c)
    trough_cache = _precompute_troughs(wavelengths, spectrum, peaks)

    val_a, diag_a = _get_corrected_with_cache(
        wavelengths, spectrum, preset.peak_a, trough_cache,
    )
    val_b, diag_b = _get_corrected_with_cache(
        wavelengths, spectrum, preset.peak_b, trough_cache,
    )

    if preset.peak_c is None:
        result = _apply_op(val_a, val_b, preset.operator1)
        return {"value": result, "peak_a_diag": diag_a, "peak_b_diag": diag_b, "peak_c_diag": None}

    val_c, diag_c = _get_corrected_with_cache(
        wavelengths, spectrum, preset.peak_c, trough_cache,
    )
    if preset.grouping == "left":
        grouped = _apply_op(val_a, val_b, preset.operator1)
        result = _apply_op(grouped, val_c, preset.operator2)
    else:
        grouped = _apply_op(val_b, val_c, preset.operator2)
        result = _apply_op(val_a, grouped, preset.operator1)

    return {"value": result, "peak_a_diag": diag_a, "peak_b_diag": diag_b, "peak_c_diag": diag_c}


def calculate_all_samples(
    wavelengths: np.ndarray,
    data_matrix: np.ndarray,
    preset: PeakPreset,
    sample_names: list | np.ndarray | None = None,
    use_local_baseline: bool = True,
) -> pd.DataFrame:
    """Calculate the expression for every sample row.

    Returns a DataFrame with columns ``["Sample", "Value"]`` plus
    ``found_{A/B/C}_wn`` columns reporting the actual wavenumber where
    each peak was measured.  When baselines are active, additional
    ``bl_*`` columns report trough positions for quality control.
    """
    n = data_matrix.shape[0]
    values = np.empty(n, dtype=float)

    has_baseline = use_local_baseline and any(
        getattr(p, "baseline", None) is not None
        for p in [preset.peak_a, preset.peak_b]
        + ([preset.peak_c] if preset.peak_c else [])
    )

    diag_rows: list[dict] = []
    peaks_list: list[tuple[PeakDefinition | None, str]] = [
        (preset.peak_a, "A"), (preset.peak_b, "B"), (preset.peak_c, "C"),
    ]

    for i in range(n):
        row: dict = {}
        if has_baseline:
            result = calculate_expression_detailed(wavelengths, data_matrix[i, :], preset)
            values[i] = result["value"]
            for key, label in [
                ("peak_a_diag", "A"), ("peak_b_diag", "B"), ("peak_c_diag", "C"),
            ]:
                d = result[key]
                if d is not None:
                    row[f"found_{label}_wn"] = d["found_wn"]
                    if "left_wn" in d:
                        row[f"bl_{label}_left_wn"] = d["left_wn"]
                        row[f"bl_{label}_right_wn"] = d["right_wn"]
                        row[f"bl_{label}_raw"] = d["raw_intensity"]
                        row[f"bl_{label}_corrected"] = (
                            d["raw_intensity"] - d["baseline_at_peak"]
                        )
        else:
            values[i] = calculate_expression(
                wavelengths, data_matrix[i, :], preset, use_local_baseline=False,
            )
            for peak_def, label in peaks_list:
                if peak_def is not None:
                    found_wn, _ = get_peak_intensity_with_position(
                        wavelengths, data_matrix[i, :], peak_def,
                    )
                    row[f"found_{label}_wn"] = found_wn
        diag_rows.append(row)

    names = sample_names if sample_names is not None else np.arange(n)
    df = pd.DataFrame({"Sample": names, "Value": values})

    if diag_rows:
        diag_df = pd.DataFrame(diag_rows)
        df = pd.concat([df, diag_df], axis=1)

    return df


# ---------------------------------------------------------------------------
# User preset I/O
# ---------------------------------------------------------------------------

_USER_DIR = Path.home() / ".spectral_predict"
_USER_PRESETS_FILE = _USER_DIR / "peak_presets.json"


def _baseline_region_to_dict(bl: BaselineRegion) -> dict:
    return {"left_min": bl.left_min, "left_max": bl.left_max,
            "right_min": bl.right_min, "right_max": bl.right_max}


def _baseline_region_from_dict(d: dict) -> BaselineRegion:
    return BaselineRegion(
        left_min=d.get("left_min", 0.0),
        left_max=d.get("left_max", 0.0),
        right_min=d.get("right_min", 0.0),
        right_max=d.get("right_max", 0.0),
    )


def _peak_def_to_dict(p: PeakDefinition) -> dict:
    d = {"wavenumber": p.wavenumber, "mode": p.mode,
         "half_width": p.half_width, "label": p.label}
    if p.baseline is not None:
        d["baseline"] = _baseline_region_to_dict(p.baseline)
    return d


def _peak_def_from_dict(d: dict) -> PeakDefinition:
    bl = None
    if "baseline" in d and d["baseline"] is not None:
        bl = _baseline_region_from_dict(d["baseline"])
    return PeakDefinition(
        wavenumber=d.get("wavenumber", 0.0),
        mode=d.get("mode", "point"),
        half_width=d.get("half_width", 10.0),
        label=d.get("label", ""),
        baseline=bl,
    )


def _preset_to_dict(p: PeakPreset) -> dict:
    d = {
        "name": p.name,
        "peak_a": _peak_def_to_dict(p.peak_a),
        "peak_b": _peak_def_to_dict(p.peak_b),
        "operator1": p.operator1,
        "peak_c": _peak_def_to_dict(p.peak_c) if p.peak_c else None,
        "operator2": p.operator2,
        "grouping": p.grouping,
        "description": p.description,
        "category": p.category,
        "x_unit": p.x_unit,
    }
    return d


def _preset_from_dict(d: dict) -> PeakPreset:
    return PeakPreset(
        name=d["name"],
        peak_a=_peak_def_from_dict(d["peak_a"]),
        peak_b=_peak_def_from_dict(d["peak_b"]),
        operator1=d.get("operator1", "/"),
        peak_c=_peak_def_from_dict(d["peak_c"]) if d.get("peak_c") else None,
        operator2=d.get("operator2", "/"),
        grouping=d.get("grouping", "left"),
        description=d.get("description", ""),
        category=d.get("category", "User"),
        x_unit=d.get("x_unit", "cm-1"),
    )


def load_user_presets() -> list[PeakPreset]:
    """Load user-saved presets from disk."""
    if not _USER_PRESETS_FILE.exists():
        return []
    try:
        data = json.loads(_USER_PRESETS_FILE.read_text(encoding="utf-8"))
        return [_preset_from_dict(d) for d in data]
    except Exception:
        return []


def save_user_presets(presets: list[PeakPreset]) -> None:
    """Save user presets to disk."""
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    data = [_preset_to_dict(p) for p in presets]
    _USER_PRESETS_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
