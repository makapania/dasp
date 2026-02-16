"""FTIR Peak Ratio Calculator — backend module.

Provides dataclasses for peak definitions and presets, built-in preset packs
(Bone FTIR, Mineralogical, Collagen & Tissue, DNA & Nucleic Acids,
Edible Oils & Fats, Food Matrices, Plant Cell Wall, Plant Composition),
calculation functions, and user-preset I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PeakDefinition:
    """A single peak position in the expression."""
    wavenumber: float = 0.0
    mode: str = "point"        # "point" or "range"
    half_width: float = 10.0   # spectral units (cm-1 or nm)
    label: str = ""


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

BUILT_IN_PRESETS: list[PeakPreset] = [
    # --- Bone FTIR ---
    PeakPreset(
        name="Mineral:Matrix",
        peak_a=PeakDefinition(1020, "point", 10, "v3 PO4"),
        peak_b=PeakDefinition(1660, "point", 10, "Amide I"),
        operator1="/",
        description="Phosphate v3 / Amide I — mineral-to-matrix ratio",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Carbonate:Phosphate",
        peak_a=PeakDefinition(1415, "point", 10, "v3 CO3"),
        peak_b=PeakDefinition(1020, "point", 10, "v3 PO4"),
        operator1="/",
        description="Carbonate v3 / Phosphate v3 — carbonate substitution",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Crystallinity Index",
        peak_a=PeakDefinition(604, "point", 10, "604"),
        peak_b=PeakDefinition(564, "point", 10, "564"),
        operator1="+",
        peak_c=PeakDefinition(590, "point", 10, "590"),
        operator2="/",
        grouping="left",
        description="(604 + 564) / 590 — apatite crystallinity (splitting factor)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Collagen Maturity",
        peak_a=PeakDefinition(1660, "point", 10, "1660 mature"),
        peak_b=PeakDefinition(1690, "point", 10, "1690 immature"),
        operator1="/",
        description="1660 / 1690 — mature / immature cross-links",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Acid Phosphate",
        peak_a=PeakDefinition(1128, "point", 10, "HPO4"),
        peak_b=PeakDefinition(1020, "point", 10, "v3 PO4"),
        operator1="/",
        description="1128 / 1020 — acid phosphate content",
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
        peak_a=PeakDefinition(1415, "point", 10, "v3 B-CO3"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="B-type CO3 / v4 PO4 — B-carbonate substitution (LeGeros & LeGeros 1983)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="A-Type Carbonate (API)",
        peak_a=PeakDefinition(1540, "point", 10, "A-CO3"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="A-type CO3 / v4 PO4 — A-carbonate substitution (Sponheimer & Lee-Thorp 1999)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="A/B Carbonate (C/C)",
        peak_a=PeakDefinition(1445, "point", 10, "A-CO3"),
        peak_b=PeakDefinition(1415, "point", 10, "B-CO3"),
        operator1="/",
        description="Type A / Type B carbonate substitution ratio (Snoeck et al. 2014)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Protein:Carbonate",
        peak_a=PeakDefinition(1650, "point", 10, "Amide I"),
        peak_b=PeakDefinition(1415, "point", 10, "v3 B-CO3"),
        operator1="/",
        description="Protein relative to carbonate content (Thompson et al. 2013)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="OH:Phosphate",
        peak_a=PeakDefinition(630, "point", 10, "OH libration"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="Hydroxyl content relative to v4 PO4 (Snoeck et al. 2014)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Cyanamide:Phosphate",
        peak_a=PeakDefinition(2010, "point", 10, "Cyanamide"),
        peak_b=PeakDefinition(1015, "point", 10, "v3 PO4"),
        operator1="/",
        description="Cyanamide formation from burning / v3 PO4 (Zazzo et al. 2013)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phosphate High Temp (PHT)",
        peak_a=PeakDefinition(625, "point", 10, "PO4 HT"),
        peak_b=PeakDefinition(610, "point", 10, "PO4 ref"),
        operator1="/",
        description="High-temperature phosphate indicator (Thompson et al. 2013)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Phosphate Valley (TPV)",
        peak_a=PeakDefinition(1088, "point", 10, "v3 PO4 shoulder"),
        peak_b=PeakDefinition(1077, "point", 10, "v3 PO4 valley"),
        operator1="/",
        description="v3 PO4 sharpening at ~1088 — thermal alteration (Colmenares-Prado et al. 2026)",
        category="Bone FTIR",
        x_unit="cm-1",
    ),
    # --- Enamel FTIR ---
    PeakPreset(
        name="Enamel IRSF",
        peak_a=PeakDefinition(560, "point", 10, "v4 PO4 560"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4 600"),
        operator1="+",
        peak_c=PeakDefinition(595, "point", 10, "v4 PO4 valley"),
        operator2="/",
        grouping="left",
        description="(560 + 600) / 595 — enamel crystallinity splitting factor",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel C/P",
        peak_a=PeakDefinition(1415, "point", 10, "v3 CO3"),
        peak_b=PeakDefinition(1015, "point", 10, "v3 PO4"),
        operator1="/",
        description="Carbonate / phosphate — enamel diagenesis indicator",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel BPI",
        peak_a=PeakDefinition(1415, "point", 10, "v3 B-CO3"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="B-type carbonate / v4 PO4 — enamel B-carbonate content",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel API",
        peak_a=PeakDefinition(1540, "point", 10, "A-CO3"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="A-type carbonate / v4 PO4 — enamel A-carbonate content",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel A/B Carbonate",
        peak_a=PeakDefinition(1445, "point", 10, "A-CO3"),
        peak_b=PeakDefinition(1415, "point", 10, "B-CO3"),
        operator1="/",
        description="Type A / Type B carbonate substitution in enamel",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel OH/P",
        peak_a=PeakDefinition(630, "point", 10, "OH libration"),
        peak_b=PeakDefinition(600, "point", 10, "v4 PO4"),
        operator1="/",
        description="Hydroxyl content — fluoride substitution indicator in enamel",
        category="Enamel FTIR",
        x_unit="cm-1",
    ),
    PeakPreset(
        name="Enamel PHT",
        peak_a=PeakDefinition(625, "point", 10, "PO4 HT"),
        peak_b=PeakDefinition(610, "point", 10, "PO4 ref"),
        operator1="/",
        description="High-temperature phosphate indicator in enamel",
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
# Calculation functions
# ---------------------------------------------------------------------------

def get_peak_intensity(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    peak_def: PeakDefinition,
) -> float:
    """Return the intensity for a single peak definition.

    Point mode: intensity at the nearest wavelength.
    Range mode: maximum intensity within [center +- half_width].
    """
    wn = peak_def.wavenumber
    if peak_def.mode == "range":
        lo = wn - peak_def.half_width
        hi = wn + peak_def.half_width
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        if not mask.any():
            idx = np.argmin(np.abs(wavelengths - wn))
            return float(spectrum[idx])
        return float(np.max(spectrum[mask]))
    else:
        idx = np.argmin(np.abs(wavelengths - wn))
        return float(spectrum[idx])


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
) -> float:
    """Evaluate the peak expression for a single spectrum.

    Two-peak: ``A op1 B``
    Three-peak, grouping="left":  ``(A op1 B) op2 C``
    Three-peak, grouping="right": ``A op1 (B op2 C)``
    """
    val_a = get_peak_intensity(wavelengths, spectrum, preset.peak_a)
    val_b = get_peak_intensity(wavelengths, spectrum, preset.peak_b)

    if preset.peak_c is None:
        return _apply_op(val_a, val_b, preset.operator1)

    val_c = get_peak_intensity(wavelengths, spectrum, preset.peak_c)
    if preset.grouping == "left":
        grouped = _apply_op(val_a, val_b, preset.operator1)
        return _apply_op(grouped, val_c, preset.operator2)
    else:
        grouped = _apply_op(val_b, val_c, preset.operator2)
        return _apply_op(val_a, grouped, preset.operator1)


def calculate_all_samples(
    wavelengths: np.ndarray,
    data_matrix: np.ndarray,
    preset: PeakPreset,
    sample_names: list | np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate the expression for every sample row.

    Returns a DataFrame with columns ``["Sample", "Value"]``.
    """
    n = data_matrix.shape[0]
    values = np.empty(n, dtype=float)
    for i in range(n):
        values[i] = calculate_expression(wavelengths, data_matrix[i, :], preset)

    names = sample_names if sample_names is not None else np.arange(n)
    return pd.DataFrame({"Sample": names, "Value": values})


# ---------------------------------------------------------------------------
# User preset I/O
# ---------------------------------------------------------------------------

_USER_DIR = Path.home() / ".spectral_predict"
_USER_PRESETS_FILE = _USER_DIR / "peak_presets.json"


def _peak_def_to_dict(p: PeakDefinition) -> dict:
    return asdict(p)


def _peak_def_from_dict(d: dict) -> PeakDefinition:
    return PeakDefinition(**d)


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
