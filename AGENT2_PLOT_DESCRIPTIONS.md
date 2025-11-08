# Tab 9 Calibration Transfer - Plot Descriptions

## Visual Guide to All 6 Diagnostic Plots

---

## Section C: Build Transfer Mapping

### Plot 1: Transfer Quality Plot (3 Subplots)
```
+------------------+------------------+------------------+
|   Master         |   Slave Before   |   Slave After    |
|   Spectra        |   Transfer       |   Transfer       |
|                  |                  |                  |
|     ___          |        ___       |     ___          |
|    /   \         |       /   \      |    /   \         |
|   /     \___     |      /     \___  |   /     \___     |
|  /               |     /            |  /               |
| BLUE (Mean±Std)  | RED (Mean±Std)   | GREEN(Mean±Std)  |
+------------------+------------------+------------------+
    Wavelength (nm)    Wavelength (nm)    Wavelength (nm)

Purpose: Visual comparison showing if transfer brings slave closer to master
Good Result: Plot 3 (green) should resemble Plot 1 (blue)
```

### Plot 2: Transfer Scatter Plot
```
    Transferred Slave Values
         ^
         |           . . .
         |         .   .  .
         |       .  . .  .    <- Ideal: Points cluster
         |     .  . . .  .       along 1:1 line
         |   .  . . . .  .
         | .  . . . .  .
         |. . . . . . .
         +----------------->
           Master Spectra Values

         Red Dashed Line = 1:1 reference
         Title shows: R² = 0.XXXX

Purpose: Quantitative assessment of transfer quality
Good Result: R² close to 1.0, points near diagonal
```

---

## Section D: Export Equalized Spectra

### Plot 1: Multi-Instrument Overlay (2 Subplots)
```
+---------------------------+---------------------------+
|  Before Equalization      |  After Equalization       |
|  (Different Grids)        |  (Common Grid)            |
|                           |                           |
|  Instrument 1 ----        |  Instrument 1 ----        |
|  Instrument 2 ----        |  Instrument 2 ----        |
|  Instrument 3 ----        |  Instrument 3 ----        |
|                           |                           |
|  Different wavelength     |  Same wavelength range    |
|  ranges shown             |  for all instruments      |
+---------------------------+---------------------------+
      Wavelength (nm)             Wavelength (nm)

Purpose: Show alignment of multiple instruments onto common grid
Good Result: Right plot shows all instruments on same wavelength range
```

### Plot 2: Wavelength Grid Comparison
```
    Wavelength (nm)
    400         1000        1600        2200
     |-----------|-----------|-----------|
Inst 1 |█████████████████████|
Inst 2      |███████████████████|
Inst 3   |████████████████████████|
Common     |████████| <- Highlighted in RED
     |-----------|-----------|-----------|

     Numbers show: "400-2200 nm" inside each bar

Purpose: Visualize wavelength coverage and common grid selection
Good Result: Common grid falls within all instrument ranges
```

---

## Section E: Predict with Transfer Model

### Plot 1: Prediction Distribution Histogram
```
    Frequency
         ^
      20 |     ___
         |    |   |
      15 |  __|   |__
         | |  |   |  |
      10 ||  |   |  ||
         ||  |   |  ||
       5 ||__|___|__||
         +------------------>
           Predicted Value

    Red Line = Mean
    Orange Lines = Mean ± Std

    Title: "Mean=X.XXX, Std=X.XXX"

Purpose: Show statistical distribution of predictions
Good Result: Normal-looking distribution, no extreme outliers
```

### Plot 2: Prediction Results Plot
```
    Predicted Value
         ^
         |        .
         |      .   .
      Mean|-----.-----.----  <- Red dashed mean line
         |    .       .
         |  .           .
         | .             .
         +----------------->
           Sample Index

    Blue dots = Individual predictions
    Light blue line = Connecting line (shows trends)

Purpose: Sequential view of all predictions
Good Result: Predictions cluster around mean, no sudden jumps
```

---

## Plot Sizing and Layout

### Section C
- Plot 1: 12 inches wide × 4 inches tall (3 subplots)
- Plot 2: 7 inches wide × 6 inches tall (1 plot)

### Section D
- Plot 1: 12 inches wide × 5 inches tall (2 subplots)
- Plot 2: 10 inches wide × 4 inches tall (1 plot)

### Section E
- Plot 1: 8 inches wide × 5 inches tall (histogram)
- Plot 2: 10 inches wide × 5 inches tall (scatter)

---

## Color Coding

### Section C
- Master: Blue
- Slave Before: Red
- Slave After: Green
- 1:1 Line: Red dashed
- Shaded regions: 30% alpha

### Section D
- Instruments: Tab10 colormap (distinct colors)
- Common Grid: Red with 70% alpha
- Grid lines: 30% alpha

### Section E
- Histogram bars: Steel blue
- Mean line: Red dashed
- Std lines: Orange dotted
- Scatter points: Steel blue with black edges
- Connecting line: Blue, 30% alpha

---

## Interpretation Guide

### Section C: Transfer Quality

**Plot 1 - Spectral Comparison:**
- Look for alignment between green (after) and blue (master)
- Large differences suggest poor transfer
- Check if red (before) differs significantly from blue

**Plot 2 - Scatter Plot:**
- R² > 0.95: Excellent transfer
- R² > 0.90: Good transfer
- R² > 0.80: Acceptable transfer
- R² < 0.80: Poor transfer, investigate

### Section D: Equalization

**Plot 1 - Overlay:**
- Left: May show different wavelength ranges
- Right: All instruments should align
- Spectral shapes should be preserved

**Plot 2 - Grid Comparison:**
- Common grid (red) shows usable range
- Ensure common grid is wide enough for analysis
- Check no instruments are severely truncated

### Section E: Predictions

**Plot 1 - Distribution:**
- Normal distribution suggests good model
- Bimodal distribution may indicate subgroups
- Long tails suggest outliers
- Check mean/std are reasonable

**Plot 2 - Sequential:**
- Look for patterns or trends
- Sudden jumps may indicate issues
- Outliers appear as points far from mean line
- Connecting line helps spot gradual changes

---

## Export Capabilities

Each plot can be exported in:
1. **PNG** - Default, high quality (300 DPI)
2. **PDF** - Vector format, publication-ready
3. **SVG** - Editable vector format
4. **JPEG** - Compressed format

Export button appears at bottom of each plot frame.

---

## Error States

### No Plots Appear
- Check if matplotlib is installed
- Verify data was loaded successfully
- Check console for error messages

### Plots Don't Update
- Frames are cleared before new plots
- Try running the operation again
- Check if data actually changed

### Export Fails
- Ensure write permissions in target directory
- Verify disk space available
- Check filename doesn't have invalid characters

---

## Performance Notes

### Large Datasets
- Section C Plot 2 (scatter) may be slow with >1000 samples
- All plots are optimized with appropriate alpha values
- Histogram automatically bins data for clarity

### Multiple Instruments (Section D)
- Tab10 colormap supports up to 10 instruments
- Colors repeat if >10 instruments used
- Legend helps identify instruments

### Memory Usage
- Plots are created on-demand
- Old plots are cleared before new ones
- Export saves to disk, doesn't keep in memory
