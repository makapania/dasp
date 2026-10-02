# Font-tuple sweep and remaining pixel geometry (QW7 follow-up)

*Prepared 2026-10-02 on branch `feat/dpi-fonts`. Do this after the concurrent GUI branches
(data loading, ensembles, analysis worker, Tab 7, CT, contaminant) have merged, because it
touches lines across the whole of `spectral_predict_gui_optimized.py`.*

## What `feat/dpi-fonts` already did

- **DPI awareness.** `main()` calls `_enable_windows_dpi_awareness()` before `tk.Tk()`. The
  PyInstaller spec embeds a manifest with `dpiAwareness=system`.
- **One scale factor.** `_apply_ui_scale(root)` sets `_UI_SCALE = winfo_fpixels('1i')/96`.
  The factor is at least 1.0, and it is 1.0 on macOS. The same call rescales `SPACING` and
  `SIDEBAR_CONFIG` in place.
- **Pixel helpers.** `_px(n)` scales one pixel length. `_px_geometry("WxH")` scales a
  Toplevel size.
- **Named fonts.** `_init_named_fonts(root)` creates the fonts and stores them in
  `self.fonts`:

  | Key | Tk name | Size and weight |
  |---|---|---|
  | `body` | DaspBody | 10 |
  | `small` | DaspSmall | 9 |
  | `strong` | DaspStrong | 11 bold |
  | `heading` | DaspHeading | 12 bold |
  | `title` | DaspTitle | 16 bold |
  | `mono` | DaspMono | 9 |

  The family is resolved from what is installed: Segoe UI and Consolas on Windows.
- **Styles and main window.** All ttk styles in `_apply_theme` use the named fonts. So do the
  top bar, `_create_accent_button` and the theme-change toast.
- **Dialogs.** The 11 fixed `Toplevel.geometry("WxH")` calls go through `_px_geometry`.

## Rules for the sweep

1. **Replace each literal tuple with `self.fonts[...]`.** The proposed key is in the table
   below. Classes that do not hold the app (for example `SidebarNavigation` and the tooltip
   helpers) can use `tkfont.nametofont('DaspBody')` and the other `Dasp*` names.
2. **Some tuples render in Arial today.** `('TkDefaultFont', N, ...)` and `('Arial', N)`
   are among them: in a tuple, `'TkDefaultFont'` is read as a family name, not as the
   named font, so Windows substitutes Arial. Replacing them changes the face the user sees,
   which is intended.
3. **Add derived named fonts before sweeping the bold and italic rows.** None of the six
   fonts covers 10 bold, 9 bold, 9 italic or 9 underline, which together account for 31
   rows. Add `body_bold` (10 bold), `small_bold` (9 bold), `small_italic` (9 italic) and
   possibly `small_underline` to `_NAMED_FONT_SPECS`. Do not use `strong` (11 bold) for
   them: that makes the text larger.
4. **Decide whether 8 pt becomes 9 pt.** The scale has no 8 pt step. Mapping 8 pt to `small`
   makes 13 labels about 12% wider. Check legends and filter rows for overflow first.
5. **Move `Consolas 10` and `Courier 10` to `mono` (9 pt) only where column alignment
   survives.** These are mostly report text boxes.
6. **Do not multiply font sizes by the scale factor.** Points already follow `tk scaling`.
7. **Leave the computed sizes alone.** These are the `('Arial', size//3, 'bold')` logo
   fallbacks.

## Literal font tuples (101)

Line numbers are for `feat/dpi-fonts` at commit `fd77c84`, so regenerate them after the other
branches merge. The "Widget" column names the nearest constructor or `.config` call above the
tuple, found heuristically, so check it when you edit. A **NEW x** proposal means "add named
font x first" (rule 3).

To regenerate the census, list every tuple whose first element is a quoted family name:

```
rg -n "\((['\"])(Segoe UI|Arial|Consolas|Courier|Tahoma|tahoma|TkDefaultFont)\1\s*," spectral_predict_gui_optimized.py
```

| Line | Function | Widget | Current | Proposed | Notes |
|---:|---|---|---|---|---|
| 352 | `showtip` | `tk.Label` | `("tahoma", "9", "normal")` | small | Tahoma tooltip label |
| 424 | `_show_tip` | `tk.Label` | `('Tahoma', 9)` | small | Tahoma tooltip label |
| 1791 | `_get_font` | `?` | `('SF Pro Text', 11)` | body | 11 normal: body (10) or keep |
| 1793 | `_get_font` | `?` | `('Segoe UI', 10)` | body |  |
| 1795 | `_get_font` | `?` | `('Ubuntu', 10)` | body |  |
| 5261 | `_create_logo_label` | `tk.Label` | `('Arial', size//3, 'bold')` | leave | computed size |
| 5274 | `_create_logo_label` | `tk.Label` | `('Arial', size//3, 'bold')` | leave | computed size |
| 5298 | `_create_logo_label` | `tk.Label` | `('Arial', size//3, 'bold')` | leave | computed size |
| 5484 | `_create_card` | `tk.Label` | `('Segoe UI', 15, 'bold')` | title | 15->16 pt |
| 5494 | `_create_card` | `tk.Label` | `('Segoe UI', 10)` | body |  |
| 5514 | `_create_section_header` | `tk.Label` | `('Segoe UI', 16, 'bold')` | title |  |
| 5533 | `_create_button_with_gradient` | `tk.Button` | `('Segoe UI', 11, 'bold')` | strong |  |
| 5563 | `_create_info_badge` | `tk.Label` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 5595 | `_create_inline_hint` | `tk.Label` | `('Segoe UI', 8, 'bold')` | NEW small_bold | 8 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 5607 | `_create_inline_hint` | `tk.Label` | `('Segoe UI', 9)` | small |  |
| 5639 | `_create_help_panel` | `tk.Label` | `('Consolas', 10)` | mono | 10->9 pt |
| 5649 | `_create_help_panel` | `tk.Label` | `('Segoe UI', 10, 'bold')` | NEW body_bold | 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 5665 | `_create_help_panel` | `tk.Label` | `('Segoe UI', 9)` | small |  |
| 5708 | `_create_legend_card` | `tk.Label` | `('Segoe UI', 10, 'bold')` | NEW body_bold | 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 5738 | `_create_legend_card` | `tk.Label` | `('Segoe UI', 9)` | small |  |
| 5791 | `_create_collapsible_section` | `tk.Label` | `('Segoe UI', 12)` | body? | 12 normal: heading is bold |
| 5800 | `_create_collapsible_section` | `tk.Label` | `('Segoe UI', 13, 'bold')` | title | 13->16 pt |
| 6149 | `_create_tab0a_import_sources` | `tk.Button` | `('TkDefaultFont', 10, 'bold')` | NEW body_bold | TkDefaultFont family (renders Arial today); 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 6172 | `_create_tab0a_import_sources` | `ttk.LabelFrame` | `('TkDefaultFont', 8)` | small | TkDefaultFont family (renders Arial today); 8->9 pt |
| 6183 | `_create_tab0a_import_sources` | `ttk.Label` | `('TkDefaultFont', 8)` | small | TkDefaultFont family (renders Arial today); 8->9 pt |
| 6196 | `_create_tab0a_import_sources` | `tk.Button` | `('TkDefaultFont', 10, 'bold')` | NEW body_bold | TkDefaultFont family (renders Arial today); 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 6227 | `_create_tab0a_import_sources` | `tk.Button` | `('TkDefaultFont', 11, 'bold')` | strong | TkDefaultFont family (renders Arial today) |
| 6331 | `_create_tab0b_merge_combine` | `tk.Button` | `('TkDefaultFont', 10, 'bold')` | NEW body_bold | TkDefaultFont family (renders Arial today); 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 6429 | `_create_tab0c_data_manipulation` | `ttk.Label` | `('Segoe UI', 9)` | small |  |
| 6924 | `_create_explore_tab` | `tk.Label` | `('TkDefaultFont', 9, 'bold')` | NEW small_bold | TkDefaultFont family (renders Arial today); 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 11099 | `_create_explore_screening_plot` | `ttk.Label` | `('Segoe UI', 10, 'bold')` | NEW body_bold | 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 11110 | `_create_explore_screening_plot` | `tk.Listbox` | `('Consolas', 9)` | mono |  |
| 11147 | `_create_explore_screening_plot` | `ttk.Label` | `('Segoe UI', 9)` | small |  |
| 11401 | `_create_tab3_data_quality_check` | `ttk.Label` | `('Segoe UI', 10, 'bold')` | NEW body_bold | 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 11442 | `_create_tab3_data_quality_check` | `ttk.Label` | `('Segoe UI', 10, 'bold')` | NEW body_bold | 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 11621 | `_create_tab4a_basic_settings` | `tk.Label` | `('TkDefaultFont', 9, 'bold')` | NEW small_bold | TkDefaultFont family (renders Arial today); 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 11644 | `_create_tab4a_basic_settings` | `ttk.Label` | `('Segoe UI', 11, 'bold')` | strong |  |
| 14698 | `_create_tab4d_ensemble_methods` | `ttk.Label` | `('TkDefaultFont', 8, 'italic')` | NEW small_italic | TkDefaultFont family (renders Arial today); style ['italic'] needs a derived named font |
| 14714 | `_create_tab4d_ensemble_methods` | `ttk.Label` | `('TkDefaultFont', 8, 'italic')` | NEW small_italic | TkDefaultFont family (renders Arial today); style ['italic'] needs a derived named font |
| 14936 | `_create_tab5_progress` | `ttk.Button` | `('Consolas', 10)` | mono | 10->9 pt |
| 14973 | `_create_tab6_results` | `ttk.Label` | `('Segoe UI', 8, 'italic')` | NEW small_italic | style ['italic'] needs a derived named font |
| 15139 | `_create_tab6_results` | `ttk.Frame` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 15150 | `_create_tab6_results` | `tk.Canvas` | `('Segoe UI', 8)` | small | 8->9 pt |
| 15154 | `_create_tab6_results` | `ttk.Label` | `('Segoe UI', 8)` | small | 8->9 pt |
| 15285 | `_create_tab7a_model_selection` | `ttk.LabelFrame` | `('Consolas', 10)` | mono | 10->9 pt |
| 15338 | `_create_tab7a_model_selection` | `ttk.Frame` | `('Consolas', 9)` | mono |  |
| 15417 | `_create_tab7b_feature_engineering` | `ttk.Frame` | `('Consolas', 9)` | mono |  |
| 16043 | `_create_tab7d_results_diagnostics` | `ttk.LabelFrame` | `('Consolas', 10)` | mono | 10->9 pt |
| 16074 | `_create_tab7d_results_diagnostics` | `ttk.Label` | `('Segoe UI', 9, 'italic')` | NEW small_italic | style ['italic'] needs a derived named font |
| 16081 | `_create_tab7d_results_diagnostics` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 16101 | `_create_tab7d_results_diagnostics` | `ttk.Label` | `('Segoe UI', 8, 'italic')` | NEW small_italic | style ['italic'] needs a derived named font |
| 16131 | `_create_tab7d_results_diagnostics` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 19710 | `_show_alignment_report` | `ttk.Frame` | `('Courier', 10)` | mono | 10->9 pt |
| 22755 | `_setup_predictor_screening_tab` | `tk.Listbox` | `('Consolas', 9)` | mono |  |
| 32755 | `_populate_results_table_inner` | `.config` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 32762 | `_populate_results_table_inner` | `.config` | `('Segoe UI', 8, 'italic')` | NEW small_italic | style ['italic'] needs a derived named font |
| 32969 | `_create_active_filter_controls` | `ttk.Label` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 33105 | `_create_active_filter_controls` | `ttk.Label` | `('Segoe UI', 8, 'italic')` | NEW small_italic | style ['italic'] needs a derived named font |
| 33293 | `_update_filter_controls` | `ttk.Label` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 33390 | `_update_filter_controls` | `ttk.Checkbutton` | `('Segoe UI', 8, 'bold')` | NEW small_bold | 8 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 33391 | `_update_filter_controls` | `tk.Label` | `('Segoe UI', 8)` | small | 8->9 pt |
| 33406 | `_update_quartile_legend` | `?` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 33425 | `_update_quartile_legend` | `tk.Canvas` | `('Segoe UI', 8)` | small | 8->9 pt |
| 33429 | `_update_quartile_legend` | `ttk.Label` | `('Segoe UI', 8)` | small | 8->9 pt |
| 33451 | `_update_class_legend` | `?` | `('Segoe UI', 9, 'bold')` | NEW small_bold | 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 33481 | `_update_class_legend` | `tk.Canvas` | `('Segoe UI', 8)` | small | 8->9 pt |
| 33505 | `_update_class_legend` | `ttk.Label` | `('Segoe UI', 8)` | small | 8->9 pt |
| 38714 | `_plot_classification_roc_curves` | `ttk.Label` | `('Arial', 10)` | body | Arial family (renders Arial today) |
| 38848 | `_add_residual_assessment` | `tk.Label` | `('TkDefaultFont', 9, 'bold')` | NEW small_bold | TkDefaultFont family (renders Arial today); 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 38913 | `_add_leverage_assessment` | `tk.Label` | `('TkDefaultFont', 9, 'bold')` | NEW small_bold | TkDefaultFont family (renders Arial today); 9 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 39074 | `_plot_classification_confidence` | `ttk.Label` | `('Arial', 10)` | body | Arial family (renders Arial today) |
| 42793 | `_export_for_publication` | `.configure` | `('Arial', 16, 'bold')` | title | Arial family (renders Arial today) |
| 42798 | `_export_for_publication` | `tk.Label` | `('Arial', 10)` | body | Arial family (renders Arial today) |
| 42802 | `_export_for_publication` | `tk.LabelFrame` | `('Arial', 10, 'bold')` | NEW body_bold | Arial family (renders Arial today); 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 42818 | `_export_for_publication` | `tk.Radiobutton` | `('Arial', 10)` | body | Arial family (renders Arial today) |
| 42822 | `_export_for_publication` | `tk.StringVar` | `('Arial', 10)` | body | Arial family (renders Arial today) |
| 43034 | `do_export` | `tk.Button` | `('Arial', 12, 'bold')` | heading | Arial family (renders Arial today) |
| 43038 | `do_export` | `tk.Button` | `('Arial', 11)` | body | Arial family (renders Arial today); 11 normal: body (10) or keep |
| 43397 | `_create_parameter_grid_control` | `ttk.Label` | `('Arial', 10, 'bold')` | NEW body_bold | Arial family (renders Arial today); 10 bold has no named font: add `small_bold`/`body_bold` or use strong |
| 43567 | `_preview_wavelength_selection` | `tk.Toplevel` | `('Arial', 12, 'bold')` | heading | Arial family (renders Arial today) |
| 43604 | `_preview_wavelength_selection` | `ttk.LabelFrame` | `('Consolas', 9)` | mono |  |
| 43785 | `_create_tab8a_setup` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 43951 | `_create_tab8b_results` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 43971 | `_create_tab8b_results` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 52584 | `_create_tab9_multi_model_comparison` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 52616 | `_create_tab9_multi_model_comparison` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 52665 | `_create_tab9_multi_model_comparison` | `tk.Listbox` | `('Consolas', 9)` | mono |  |
| 52876 | `_create_tab9_multi_model_comparison` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 52907 | `_create_tab9_multi_model_comparison` | `tk.Label` | `('Segoe UI', 9)` | small |  |
| 52912 | `_create_tab9_multi_model_comparison` | `tk.Label` | `('Segoe UI', 8)` | small | 8->9 pt |
| 52963 | `_create_tab9_multi_model_comparison` | `tk.Label` | `('Segoe UI', 9)` | small |  |
| 52968 | `_create_tab9_multi_model_comparison` | `tk.Label` | `('Segoe UI', 9, 'underline')` | NEW small_underline | style ['underline'] needs a derived named font |
| 54763 | `_create_tab10_calibration_transfer` | `tk.Label` | `('Segoe UI', 11, 'bold')` | strong |  |
| 54778 | `_create_tab10_calibration_transfer` | `tk.Label` | `('Consolas', 9)` | mono |  |
| 55696 | `_create_tab11a_library_management` | `tk.Listbox` | `('Segoe UI', 10)` | body |  |
| 56582 | `_create_tab12a_library_management` | `tk.Listbox` | `('Segoe UI', 10)` | body |  |
| 57889 | `_create_tab13a_load_groups` | `tk.Listbox` | `('Segoe UI', 10)` | body |  |
| 57912 | `_create_tab13a_load_groups` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 58106 | `_create_tab13b_difference_analysis` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 58344 | `_create_tab13c_automated_detection` | `tk.Text` | `('Consolas', 9)` | mono |  |
| 60710 | `_populate_app_method_settings` | `ttk.Label` | `('Arial', 8)` | small | Arial family (renders Arial today); 8->9 pt |

## Pixel geometry not yet scaled

On a 125% display the following items stay at their 96-dpi pixel size. Text is now 25%
larger in pixels, so they look about 20% tighter than before. None of them clips in the
screenshots taken on 2026-10-02 (Import, Configuration, Results, Development, Cal Transfer,
Contaminant). Wrap the values in `_px()` during the sweep.

- **Literal `padx`, `pady`, `ipadx` and `ipady` on per-tab widgets.** There are hundreds of
  them across all tabs. This is cosmetic: spacing gets tighter but nothing is cut off.
  Values that come from `SPACING[...]` are already scaled.
- **Card and section helpers** (`_create_card` :5465, `_create_section_header` :5502,
  `_create_info_badge` :5556, `_create_inline_hint` :5570, `_create_help_panel` :5616,
  `_create_legend_card` :5691, `_create_collapsible_section` :5780). Their padding is in
  literal pixels. These helpers are shared by every tab, so scaling them in one place
  restores the old proportions everywhere. This pairs naturally with QW8 (flatten cards).
- **29 `wraplength=N` values.** Text wraps after fewer words, but it is not clipped.
- **Legend swatches** `tk.Canvas(width=14, height=14)` at :15148, :33423 and :33479.
- **Dialog sizes and the screen edge.** `_px_geometry` does not clamp to the screen. At 150%
  on a 1920×1080 panel, the 520×720 peak-calculator dialog becomes 780×1080, which is taller
  than the work area. That was already true before this change (720 logical px on a 720 px
  logical screen).

### Checked and needing no change

- **Treeview row height.** Tk 9 derives it from the font: 17 px at 96 dpi, 22 px at 120 dpi.
  No `rowheight` is set anywhere.
- **Embedded matplotlib canvases.** `FigureCanvasTk._update_device_pixel_ratio` multiplies
  the figure dpi by `tk scaling / (96/72)`, which is 1.25 at 125%, so figures keep their
  physical size. Setting the figure dpi by hand would scale twice.
- **Text and Listbox `width`/`height`, and ttk `width=`.** These are in characters or lines,
  so they already follow the font.

### Known limits

- **System-aware only.** On a second monitor with a different scale, Windows bitmap-stretches
  the window again. Per-monitor v2 is deferred (roadmap: "Deferred indefinitely").
- **Frozen manifest untested.** The manifest in `spectral_predict_py312.spec` was checked
  through PyInstaller's `winmanifest.create_application_manifest`: the execution level,
  Common-Controls, compatibility, longPathAware and dpiAware/dpiAwareness entries are all
  present. The installer has not been built and run with it.
- **macOS and Linux.** The font-family pick is untested there. On macOS the candidates are
  SF Pro Text, then Helvetica Neue, then Helvetica, falling back to TkDefaultFont's family. On
  Linux they are Inter, Ubuntu, Noto Sans, DejaVu Sans and Liberation Sans.
