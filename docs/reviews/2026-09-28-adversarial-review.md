# Adversarial correctness review of `main` (2026-09-28)

Run at `main` `449dfb1` (after PR #82). 11 areas were each reviewed by an adversarial finder, and every finding was then
re-checked by an independent skeptic told to refute it, with repro scripts where cheap. The scripts lived in a local
scratch dir and are not in the repo. Result: **134 findings, 129 CONFIRMED, 4 PLAUSIBLE, 1 REFUTED.** Severity below is
the verifier's (it downgraded 18 and upgraded 2). IDs are stable references for fix PRs.

Known open issues listed in PROJECT_STATUS §4 were excluded up front.

| Severity | Count |
|---|---|
| critical | 2 |
| high | 30 |
| medium | 61 |
| low | 40 |

## Index

- **R001** [critical] io-persistence: Saved models trained with a Y-transform lose or double-apply spectral preprocessing, so loaded .dasp predictions are wrong (`spectral_predict_gui_optimized.py:42096`)
- **R002** [critical] models-ensemble: Ensemble R2CV/RMSECV is computed with base models trained on the validation fold (in-sample leakage) (`spectral_predict_gui_optimized.py:25492`)
- **R003** [high] bayesian: Boosting CV uses the held-out test fold as the early-stopping eval_set, so Bayesian RMSEcv/Accuracycv for XGBoost/LightGBM/CatBoost is optimistically biased and TPE optimises that biased score (`src/spectral_predict/unified_bayesian.py:1847`)
- **R004** [high] gui-1: Loading a new dataset keeps the previous dataset's exclusions, validation split and validation snapshot (`spectral_predict_gui_optimized.py:19430`)
- **R005** [high] gui-1: validation_X/validation_y snapshots are not rebuilt after a wavelength-range update, working-data replacement or later exclusion (`spectral_predict_gui_optimized.py:19716`)
- **R006** [high] gui-1: Spectrum-click exclusion int-coerces numeric-string IDs, so it never reaches the analysis (`spectral_predict_gui_optimized.py:21071`)
- **R007** [high] gui-1: Quality Check 'Mark for exclusion' excludes position+1, not the sample label (`spectral_predict_gui_optimized.py:22495`)
- **R008** [high] gui-2: Bayesian 'advanced' baseline does nothing in the search, but results, Tab 7 and validation apply it (`spectral_predict_gui_optimized.py:30403`)
- **R009** [high] gui-3: Tab 7 maps selected wavelengths to columns with ±0.5 tolerance and takes the first hit; predict uses ±0.01. Train/predict features differ on fine grids (`spectral_predict_gui_optimized.py:41351`)
- **R010** [high] gui-3: A stale bias/nonlinear correction from an earlier model is saved into a newly trained model (`spectral_predict_gui_optimized.py:42601`)
- **R011** [high] gui-3: File Equalize names exported transformed spectra after the wrong input files (`spectral_predict_gui_optimized.py:47974`)
- **R012** [high] gui-3: CT prediction and transform paths silently extrapolate spectra beyond the measured range (`spectral_predict_gui_optimized.py:50968`)
- **R013** [high] gui-3: Tab 9, CT Mode A and CT Mode B R/A conversions use (and overwrite) the training data's reflectance scale (`spectral_predict_gui_optimized.py:53457`)
- **R014** [high] io-persistence: With Y-transform plus early stopping, the saved model is trained on untransformed y while metadata records the transform (`spectral_predict_gui_optimized.py:41544`)
- **R015** [high] io-persistence: Export bundle ships already-preprocessed data but its script preprocesses again and indexes full-spectrum columns (`src/spectral_predict/export_bundle.py:263`)
- **R016** [high] io-persistence: Saving a classifier crashes when the label encoder has numeric classes, and the stale Bayesian encoder would mis-decode labels (`src/spectral_predict/model_io.py:183`)
- **R017** [high] io-persistence: OPUS reader returns the background reference single-channel instead of absorbance (`src/spectral_predict/readers/opus_reader.py:89`)
- **R018** [high] models-ensemble: Ensemble CV loop indexes a label-indexed pandas Series positionally: KeyError under the locked pandas 3 (`spectral_predict_gui_optimized.py:25490`)
- **R019** [high] models-ensemble: With early stopping plus a Y-transform, the saved final model is trained without the transform that CV used (`spectral_predict_gui_optimized.py:41944`)
- **R020** [high] models-ensemble: Y-transform (TTR) model is saved with a duplicate preprocessor, so saved-model predictions are preprocessed twice (`spectral_predict_gui_optimized.py:42105`)
- **R021** [high] models-ensemble: With wrapped base models, ensemble weights and the stacking meta-learner are fitted on in-sample predictions (`src/spectral_predict/ensemble.py:382`)
- **R022** [high] nsga-ga: NSGA-II fitness early-stops boosters on the held-out CV fold, then scores that same fold (`src/spectral_predict/nsga2_search.py:1486`)
- **R023** [high] nsga-ga: Stored NSGA-II Params for LightGBM and CatBoost describe a different model from the one that was scored (`src/spectral_predict/nsga2_search.py:2436`)
- **R024** [high] one-class: EstimatedEPO (default 'pca_diff' and 'bootstrap') and MultiGroupEPO remove random noise directions, not the contaminant (`src/spectral_predict/contaminant_analysis.py:677`)
- **R025** [high] one-class: OSC removes the y-predictive direction (first PLS loading) instead of y-orthogonal variation (`src/spectral_predict/interference.py:491`)
- **R026** [high] preproc-varsel: Tab 7 refit maps each selected wavelength to the FIRST column within 0.5 units, which is the lower neighbour on grids finer than 0.5; predict then uses the exact column (`spectral_predict_gui_optimized.py:41351`)
- **R027** [high] preproc-varsel: Bayesian search silently skips the 'advanced' (pybaselines) baseline but records baseline_method='advanced' on the row (`src/spectral_predict/unified_bayesian.py:717`)
- **R028** [high] search-cv: Boosting early stopping uses the CV test fold as eval_set, so XGBoost/LightGBM/CatBoost CV metrics are optimistically biased (`src/spectral_predict/cv_utils.py:634`)
- **R029** [high] search-cv: Binary labels other than {0,1} give inconsistent or NaN F1/Precision/Recall and crash ranking (`src/spectral_predict/search.py:4583`)
- **R030** [high] search-cv: LOO classification metrics are averaged over 1-sample folds, so F1cv/MCCcv/Specificitycv are nonsense and Kappacv/ROC_AUCcv are NaN (`src/spectral_predict/search.py:5084`)
- **R031** [high] search-cv: all_vars is written with %g (6 significant digits), so validation silently scores subset models on the full spectrum (`src/spectral_predict/search.py:5522`)
- **R032** [high] transfer-analysis: write_markdown_report fails on non-ASCII target names or an unwritable working directory, aborting the whole analysis tail (`src/spectral_predict/report.py:153`)
- **R033** [medium] bayesian: region_id is capped at n_top_regions-1, so enabling 'test all individual' removes the combined-region subsets and 'pairwise' adds almost nothing, while both change the study identity (`src/spectral_predict/unified_bayesian.py:1566`)
- **R034** [medium] bayesian: Multiclass Bayesian region (and UVE) selection correlates spectra with LabelEncoder integer codes, so the selected wavelengths depend on alphabetical class names (`src/spectral_predict/unified_bayesian.py:1584`)
- **R035** [medium] bayesian: Re-attached TPE sampler reuses the same seed, so after 'auto' migration (or an 'always' resume) the next startup trials exactly replay trials 0..k-1 as duplicates (`src/spectral_predict/unified_bayesian.py:2481`)
- **R036** [medium] gui-1: Explore colour-by-metadata (spectra and PCA scores) indexes the reference table by position (`spectral_predict_gui_optimized.py:4167`)
- **R037** [medium] gui-1: Data Management 'Use for Analysis' / 'Merge & Use' / filter / trim bypass X_original and metadata (`spectral_predict_gui_optimized.py:18338`)
- **R038** [medium] gui-1: Sub-integer wavelength rejection leaves a half-loaded dataset (new y/X_original, old X) (`spectral_predict_gui_optimized.py:19673`)
- **R039** [medium] gui-1: SPXY (the default splitter) is allowed for classification and uses LabelEncoder codes as Y distances (`spectral_predict_gui_optimized.py:20664`)
- **R040** [medium] gui-1: GUI VIP screening uses var(y) instead of q_a^2, so Y-irrelevant X variance is ranked 'important' (`spectral_predict_gui_optimized.py:22731`)
- **R041** [medium] gui-2: One-class run with an integer-coded target and a typed inlier label is always blocked (`spectral_predict_gui_optimized.py:29724`)
- **R042** [medium] gui-2: Bayesian and NSGA-II searches ignore the analysis wavelength restriction while the log says it is applied (`spectral_predict_gui_optimized.py:30366`)
- **R043** [medium] gui-2: Re-ranking after a penalty change treats one-class and multiclass results as classification (`spectral_predict_gui_optimized.py:33612`)
- **R044** [medium] gui-2: A data-viewer cell edit rebuilds all data from the display strings: values rounded, index stringified, hidden excluded rows dropped (`spectral_predict_gui_optimized.py:35029`)
- **R045** [medium] gui-2: Cancelling the Tab 7 mismatch dialog leaves a half-loaded config, so the next Run refits a hybrid of two rows (`spectral_predict_gui_optimized.py:36016`)
- **R046** [medium] gui-2: Loading a result into Tab 7 overwrites the global Analysis Config preprocessing toggles (`spectral_predict_gui_optimized.py:37146`)
- **R047** [medium] gui-2: Tab 7 one-class refit of Bayesian rows uses the live inlier-class box, not the label the search used (`spectral_predict_gui_optimized.py:40794`)
- **R048** [medium] gui-2: Tab 7 refit crashes whenever a Y-transform is selected (TransformedTargetRegressor has no .steps) (`spectral_predict_gui_optimized.py:41570`)
- **R049** [medium] gui-3: The target property name is never saved, so Tab 8 consensus averages models of unrelated targets (`spectral_predict_gui_optimized.py:42418`)
- **R050** [medium] gui-3: Saved .dasp data_type, x_unit and y_transform come from live widgets at save time, not the training snapshot (`spectral_predict_gui_optimized.py:42432`)
- **R051** [medium] gui-3: Tab 8 Excel input loses sample IDs and is misdetected as reflectance (`spectral_predict_gui_optimized.py:44250`)
- **R052** [medium] gui-3: Tab 8 attaches the training reference table (including training targets) to new samples with overlapping IDs (`spectral_predict_gui_optimized.py:44447`)
- **R053** [medium] gui-3: The Tab 8 uncertainty table assumes every model has the first model's task type (`spectral_predict_gui_optimized.py:45177`)
- **R054** [medium] gui-3: Tab 8 validation run with a one-class or binary text-label classifier crashes in the confusion-matrix plot (`spectral_predict_gui_optimized.py:45749`)
- **R055** [medium] gui-3: Tab 9 CSV-directory loader attaches filenames to spectra in a different sort order (`spectral_predict_gui_optimized.py:48709`)
- **R056** [medium] gui-3: CT Mode A predicts without checking the satellite data type against the prediction or transfer model (`spectral_predict_gui_optimized.py:50971`)
- **R057** [medium] gui-3: A failed transfer in the Tab 9 chain still produces 'Comparison complete', predicted on untransferred data (`spectral_predict_gui_optimized.py:53926`)
- **R058** [medium] io-persistence: Code export mis-maps or omits preprocessing (sg4, snv_sg3/4, sg3 polyorder, baseline, smoothing, y-transform) (`src/spectral_predict/code_generator.py:836`)
- **R059** [medium] io-persistence: Merging data sources never detects cross-source duplicate IDs; 'error', 'keep_first' and 'keep_last' all keep both rows (`src/spectral_predict/data_management.py:358`)
- **R060** [medium] io-persistence: CSV-folder and JCAMP-folder readers round the x-axis to integers, silently dropping and mislabeling sub-integer-spaced data (`src/spectral_predict/io.py:288`)
- **R061** [medium] io-persistence: Combined CSV/Excel import drops spectral columns above 10000 and can auto-pick an absorbance column as the target (`src/spectral_predict/io.py:1232`)
- **R062** [medium] io-persistence: read_ascii_spectra is redefined later in io.py: ASCII folder import is broken and the first data row is eaten as a header (`src/spectral_predict/io.py:3354`)
- **R063** [medium] io-persistence: Loading a .dasp or ensemble unpickles arbitrary objects with no trust boundary or warning (`src/spectral_predict/model_io.py:433`)
- **R064** [medium] models-ensemble: Stale nonlinear bias correction from a previous Model Development run can be saved with a different model (`spectral_predict_gui_optimized.py:42603`)
- **R065** [medium] models-ensemble: ClassificationResampler loses k_neighbors (and every other method param) when cloned, so each CV fold resamples with defaults or skips resampling (`src/spectral_predict/imbalance.py:229`)
- **R066** [medium] models-ensemble: RegressionSampleWeighter 'binning' gives the maximum-y sample its own bin and a huge weight (`src/spectral_predict/imbalance.py:837`)
- **R067** [medium] models-ensemble: SVR and SVM grids ignore the user's kernel, C and gamma selections (`src/spectral_predict/models.py:1340`)
- **R068** [medium] models-ensemble: NeuralBoostedClassifier trains from the class-prior log-odds but predicts from 0, so probabilities and labels are biased (`src/spectral_predict/neural_boosted.py:879`)
- **R069** [medium] models-ensemble: The 'Box-Cox' Y-transform offered in the GUI is rejected by YTransformWrapper.wrap and skipped by validate (`src/spectral_predict/y_transform.py:66`)
- **R070** [medium] nsga-ga: GA-PLS classification fitness thresholds predictions at their median, forcing a 50/50 split (`src/spectral_predict/ga_pls.py:216`)
- **R071** [medium] nsga-ga: SmartMutation almost never mutates preprocessing, model or hyperparameter genes, and uses the wrong model list for active genes (`src/spectral_predict/nsga2_search.py:199`)
- **R072** [medium] nsga-ga: Guided NSGA-II (the GUI default) is not reproducible despite random_state=42 (`src/spectral_predict/nsga2_search.py:396`)
- **R073** [medium] nsga-ga: decode_solution sizes PLS n_components from the wavelength count before edge masking and from the wrong N (`src/spectral_predict/nsga2_search.py:2334`)
- **R074** [medium] nsga-ga: NSGA-II top_vars is always the first 30 selected wavelengths in index order, not importance-ranked (`src/spectral_predict/nsga2_search.py:3430`)
- **R075** [medium] one-class: Interference tab: Wavelength Exclusion, OSC and DOSC always crash (`spectral_predict_gui_optimized.py:60680`)
- **R076** [medium] one-class: PCASIMCA allows n_components = n_train − 1, which leaves no residual space and rejects every new sample (`src/spectral_predict/contamination.py:134`)
- **R077** [medium] one-class: Autoscale is fitted on inliers only in grid search but on inliers plus contaminants in validation, Model Development refit and Bayesian (`src/spectral_predict/contamination.py:1218`)
- **R078** [medium] one-class: One-class validation silently returns NaN for every row when wavelengths have more than 6 significant digits (`src/spectral_predict/contamination.py:1244`)
- **R079** [medium] one-class: Outlier report counts Hotelling T² twice: its 'Mahalanobis' flag is sqrt(T²) on the same PCA scores (`src/spectral_predict/outlier_detection.py:558`)
- **R080** [medium] one-class: Multi-class 'elliptic-envelope' engine gives badly miscalibrated p-values when features are close to or above n (no p>n guard) (`src/spectral_predict/simca.py:480`)
- **R081** [medium] preproc-varsel: BaselineAdvanced keeps algorithm params in **kwargs, so sklearn.clone() silently resets them to the registry defaults (`src/spectral_predict/baseline_advanced.py:391`)
- **R082** [medium] preproc-varsel: Wavelength exclusion inside the preprocessing pipeline drops columns, but search keeps the full wavelength axis, so top_vars and all_vars name the wrong wavelengths (`src/spectral_predict/preprocess.py:367`)
- **R083** [medium] preproc-varsel: One-class smart preprocessing discovery with fewer than 2 outliers ranks every config 1.0 and returns the first candidates in list order (`src/spectral_predict/preprocessing_discovery.py:688`)
- **R084** [medium] preproc-varsel: CARS cannot return fewer than ~9% of the variables, because its decay schedule stops at 0.8*0.8*e^-2 instead of decaying to about 2 variables (`src/spectral_predict/variable_selection.py:1412`)
- **R085** [medium] search-cv: Kennard-Stone picks the wrong starting pair (condensed-index conversion is for the lower triangle) (`src/spectral_predict/sample_selection.py:110`)
- **R086** [medium] search-cv: Validation rebuild drops resampling and regression-weighting imbalance methods, so RMSEP/val_* describe an unbalanced model (`src/spectral_predict/search.py:508`)
- **R087** [medium] search-cv: GA/exhaustive preprocessing + baseline/smoothing toggles produce rows labelled ALS/sg0 that were never baseline-corrected or smoothed (`src/spectral_predict/search.py:2896`)
- **R088** [medium] transfer-analysis: Complexity curve and leverage are computed on raw spectra, not the model's preprocessed spectra (`spectral_predict_gui_optimized.py:42251`)
- **R089** [medium] transfer-analysis: Multi-model comparison transfer chain applies models positionally without resampling to the model grid, then relabels columns (`spectral_predict_gui_optimized.py:53913`)
- **R090** [medium] transfer-analysis: NS-PFCE models built with wavelength selection cannot be applied: output width no longer matches the grid (`src/spectral_predict/calibration_transfer.py:1410`)
- **R091** [medium] transfer-analysis: JYPLS-inv transfer matrix omits PLS centering, so 'transferred' spectra are far worse than untransferred ones (`src/spectral_predict/calibration_transfer.py:1641`)
- **R092** [medium] transfer-analysis: compute_leverage uses the full spectral matrix, so every sample gets leverage 1.0 and nothing is ever flagged (`src/spectral_predict/diagnostics.py:91`)
- **R093** [medium] transfer-analysis: Library search linearly extrapolates spectra outside their measured range, and those values dominate the scores (`src/spectral_predict/library_search.py:322`)
- **R094** [low, PLAUSIBLE] bayesian: Phase-2 multi-seed rescore ranks a candidate on whichever seeds survived; a config that failed 4 of 5 seeds is scored on one seed with std=0 (`src/spectral_predict/phase2_rescore.py:58`)
- **R095** [low] gui-1: Append-mode alignment failure leaves merged X_original with un-merged y and no rollback (`spectral_predict_gui_optimized.py:19448`)
- **R096** [low] gui-1: VIP screening crashes on any missing target; categorical correlation treats NaN as a class; screening ignores exclusions (`spectral_predict_gui_optimized.py:22723`)
- **R097** [low] gui-2: Blank or non-integer Bayesian Trials box passes the launch gate, shows a false SQLite warning, then the worker fails (`spectral_predict_gui_optimized.py:26932`)
- **R098** [low] gui-2: Result checkboxes show stale state after filter + toggle + sort; the ensemble uses different rows than shown (`spectral_predict_gui_optimized.py:31639`)
- **R099** [low] gui-2: The one-class learning curve evaluates a different model: raw estimator, no scaler/PCA, trained on outliers too (`spectral_predict_gui_optimized.py:39566`)
- **R100** [low] gui-2: Tab 7 GA failure paths leave Run buttons disabled and the wait cursor on (`spectral_predict_gui_optimized.py:41149`)
- **R101** [low] gui-2: Tab 7 classification reproducibility check compares refit CV accuracy with the row's calibration accuracy (`spectral_predict_gui_optimized.py:41906`)
- **R102** [low] gui-3: Live monitoring detects changes by file count only, so replaced or partially written spectra are not refreshed (`spectral_predict_gui_optimized.py:53602`)
- **R103** [low] io-persistence: read_spectra on a single ASD file returns every ASD spectrum in its folder; JCAMP/Excel folders go to single-file readers (`src/spectral_predict/io.py:2582`)
- **R104** [low] io-persistence: predict_with_uncertainty(validate_wavelengths=False) ignores the full-spectrum handshake that predict_with_model honours (`src/spectral_predict/model_io.py:1048`)
- **R105** [low] models-ensemble: A base model that fails OOF gets equal weight, NaN predictions or a crash, not the promised exclusion (`src/spectral_predict/ensemble.py:403`)
- **R106** [low, PLAUSIBLE] models-ensemble: create_auto_ensembles' CV falls back to the full-data ensemble and counts failed reconstructions as selected models (`src/spectral_predict/ensemble.py:1913`)
- **R107** [low] nsga-ga: GA-PLS / GA-LightGBM importances are coarse selection frequencies, and ties are broken by highest wavelength index (`src/spectral_predict/ga_pls.py:673`)
- **R108** [low] nsga-ga: Exhaustive-preprocessing configs report polyorder = deriv-1, which is invalid and not what was fitted (`src/spectral_predict/ga_preprocessing.py:1352`)
- **R109** [low] nsga-ga: Failed evaluations (1e10 penalty) are feasible and enter the Pareto front, results table and knee normalisation (`src/spectral_predict/nsga2_search.py:1580`)
- **R110** [low] nsga-ga: knee_solution objectives['n_wavelengths'] reports n^2/N instead of n (`src/spectral_predict/nsga2_search.py:2200`)
- **R111** [low, PLAUSIBLE] nsga-ga: NSGA-II best-from-all row drops imbalance handling and computes calibration/R2cv wrongly (`src/spectral_predict/nsga2_search.py:4014`)
- **R112** [low] one-class: Model Development one-class refit can pick the wrong wavelength column (first match within ±0.5) (`spectral_predict_gui_optimized.py:40780`)
- **R113** [low] one-class: 'Apply correction' overwrites the main dataset with mean-centred (EPO) or autoscaled (OPLS filter) spectra (`spectral_predict_gui_optimized.py:60102`)
- **R114** [low] one-class: 'Export Corrected Spectra' writes no file but shows a green 'exported' status (`spectral_predict_gui_optimized.py:60262`)
- **R115** [low] one-class: Any missing target value turns off all y outlier checks in the outlier report (`src/spectral_predict/outlier_detection.py:445`)
- **R116** [low] preproc-varsel: BaselineALS and BaselinePolynomial keep the input dtype, so integer spectra have their baseline-corrected values truncated (`src/spectral_predict/baseline.py:167`)
- **R117** [low] preproc-varsel: The Advanced baseline's GUI 'Lambda' value is silently ignored for algorithms without a lam parameter, and their own parameters cannot be set (`src/spectral_predict/baseline_advanced.py:441`)
- **R118** [low] preproc-varsel: choose_common_grid can extend past the overlapping range, and the extra points are linearly extrapolated (`src/spectral_predict/equalization.py:48`)
- **R119** [low] preproc-varsel: The Explore-tab MovingAverage preview zero-pads the spectrum ends, pulling edge values toward zero (`src/spectral_predict/preprocess.py:259`)
- **R120** [low, PLAUSIBLE] preproc-varsel: y-supervised interference steps (OSC, DOSC, EPO, GLSW) are fitted on all samples, including validation folds, before CV (`src/spectral_predict/search.py:2932`)
- **R121** [low] preproc-varsel: The variable-selection path edge-masks importances on an axis that was already edge-trimmed, discarding a second window//2 interior wavelengths per side (`src/spectral_predict/search.py:3879`)
- **R122** [low] preproc-varsel: Uniform or tied importance fallbacks turn 'top-N' subsets into the N highest-index (longest-wavelength) columns, still labelled as selector output (`src/spectral_predict/search.py:3950`)
- **R123** [low] preproc-varsel: Subset rows' top_vars edge-mask the importance-ordered subset, removing the selector's most important variables from the reported top wavelengths (`src/spectral_predict/search.py:5555`)
- **R124** [low] search-cv: Wavelength-exclusion interference step drops columns but wavelength labels are not updated, so top_vars/all_vars are wrong and wl filtering crashes (`src/spectral_predict/search.py:2932`)
- **R125** [low] search-cv: Calibration F1/Precision/Recall use 'weighted' averaging while CV uses 'binary'/'macro', so the cal-vs-CV gap is misleading (`src/spectral_predict/search.py:5203`)
- **R126** [low] search-cv: XGBoost+class_weight and any booster+regression weighting are CV'd without early stopping, but the row records early_stopping_rounds=40 (`src/spectral_predict/search.py:5371`)
- **R127** [low] transfer-analysis: Transfer-quality plots fail silently for models built on a region of interest (`spectral_predict_gui_optimized.py:48183`)
- **R128** [low] transfer-analysis: Transfer-quality plot reports resubstitution R², which reads as near-perfect for DS and NS-PFCE (`spectral_predict_gui_optimized.py:48344`)
- **R129** [low] transfer-analysis: Regularisation and SVM validation curves exclude the selected alpha or C and ignore the model's other hyperparameters (`src/spectral_predict/diagnostics.py:737`)
- **R130** [low] transfer-analysis: The common grid can end beyond the overlap, and resample_to_grid extrapolates silently (`src/spectral_predict/equalization.py:48`)
- **R131** [low] transfer-analysis: Library save crashes when one sample ID equals 'wl_' plus another sample ID (`src/spectral_predict/library_search.py:525`)
- **R132** [low] transfer-analysis: Bone FTIR ratios are reported as 0.0 instead of NaN when PO4 extraction fails or is non-positive (`src/spectral_predict/peak_calculator.py:1725`)
- **R133** [low] transfer-analysis: HQI and derivative-correlation metrics use r², so inverted or anti-correlated spectra score as perfect matches (`src/spectral_predict/similarity_metrics.py:53`)

## Details

### R001: Saved models trained with a Y-transform lose or double-apply spectral preprocessing, so loaded .dasp predictions are wrong
`spectral_predict_gui_optimized.py:42096` · area io-persistence · persistence · finder critical → verifier **critical / CONFIRMED**

**Problem.** When the Y-transform wraps the whole pipeline in a TransformedTargetRegressor (TTR), _run_refined_model saves the full TTR as the model. It then builds 'preprocessor' only from the steps inside the TTR, before the model step, and skips the `elif use_full_spectrum_preprocessing` branch that would save prep_pipeline. use_full_spectrum_preprocessing is always True (GUI:40242), so SNV, SG derivative, baseline, smoothing and autoscale are fitted OUTSIDE the TTR in prep_pipeline and are never saved. For PLS, RF or the boosters the inner pipeline is just [model], so preprocessor=None and model_io.predict_with_model feeds RAW subset spectra to a model trained on derivative spectra. For Ridge, SVR and MLP the inner [scaler, model] is saved as the preprocessor and applied to the full-width X, so it crashes, or silently mis-scales when there is no subset. On Path B the preprocessing is applied twice: once by the saved preprocessor, again inside the TTR. The applicability domain is also computed on mismatched data. In-app CV and validation metrics look correct because they use final_pipe directly, so the corruption only shows up after save and load.

**Failure scenario.** Model Development: sg1 preprocessing, wavelength subset, PLS, Y-transform='log'. Save and reload the .dasp, then predict on the training spectra. The replay in ttr_check.py gives expected [1.649 4.43 2.835 1.79 2.562] but the loaded model returns [0, 1.28e8, 0, 0, 0] (exp overflow from raw-spectra input). A Path B equivalent (savgol inside the TTR) returns [2.33 2.33 2.40 2.20 2.43] instead of [1.65 4.40 2.93 1.70 2.53].

**Verifier.** GUI:40242 hard-codes use_full_spectrum_preprocessing=True, so refinement always takes Path A (GUI:41290). There, preprocessing runs outside the pipeline in prep_pipeline, and pipe holds only [model] or [scaler, model]. With a Y-transform (and no booster early stopping), GUI:41549 wraps that pipe in a TTR. At save time, GUI:42096-42113 builds final_preprocessor only from the TTR's inner steps, and GUI:42136 `if isinstance(final_pipe,_TTR): pass` skips the branch that would save prep_pipeline. So the SG/SNV/baseline/autoscale pipeline is never persisted. predict_with_model (model_io.py:664-684) then feeds un-derived subset spectra to the TTR. The Path B double-apply part is unreachable from the refinement GUI because the flag is always True. That does not change the verdict: Path A, the path every user takes, is broken.

### R002: Ensemble R2CV/RMSECV is computed with base models trained on the validation fold (in-sample leakage)
`spectral_predict_gui_optimized.py:25492` · area models-ensemble · leakage · finder critical → verifier **critical / CONFIRMED**

**Problem.** _train_ensembles rebuilds the base models once, fitted on ALL of X_filtered (_reconstruct_models_from_results calls pipeline.fit(X_train, y_train) with the full data). The outer 'CV' loop then only refits the ensemble weights or meta-learner on X_cv_train. The fitted base models themselves (ensemble.models) produce cv_ensemble.predict(X_cv_val), and they were trained on those exact validation samples. The reported R2CV/RMSECV/RPD are therefore close to calibration metrics. They are ranked against the individual models' honest R2cv ('Best Individual Model ... improvement') and shown in the Results tab as R2cv. The quartile ensembles (25673-25682) have the same flaw.

**Failure scenario.** 80 samples, 50 noisy features, two RandomForest base models. Each has a true 5-fold CV R2 of about 0.23. Replaying the GUI loop reports R2CV = 0.875 for Simple Average and Region-Aware and 0.851 for Stacking. The user concludes the ensemble beats every individual model by about 0.6 R2 and deploys it.

**Verifier.** _reconstruct_models_from_results fits every base model on all of X_filtered (pipeline.fit(X_train, y_train), GUI ~25128). In the outer KFold loop, create_ensemble(models=models, X=X_cv_train, ...) receives those same fitted objects. SimpleAverage/RegionAware/Stacking predict() calls self.models[i].predict(X), so the validation fold is predicted by models that were trained on it. refit_base_models=True only refits clones inside the inner OOF weight loop and never replaces self.models. Nothing in SESSION_LOG, PROJECT_STATUS or CHANGELOG says this is intended. The code comment claims the metrics are 'realistic, comparable to individual models'.

### R003: Boosting CV uses the held-out test fold as the early-stopping eval_set, so Bayesian RMSEcv/Accuracycv for XGBoost/LightGBM/CatBoost is optimistically biased and TPE optimises that biased score
`src/spectral_predict/unified_bayesian.py:1847` · area bayesian · leakage · finder high → verifier **high / CONFIRMED**

**Problem.** For XGBoost, LightGBM and CatBoost, `use_early_stopping` is on by default (early_stopping_rounds=40). The objective then calls `cross_val_predict_with_early_stopping`, which for each fold fits the booster with `eval_set=[(X_val, y_val)]`, where X_val and y_val are the fold's own test rows (cv_utils.py:1032-1072), and then predicts those same rows. The number of trees is chosen to minimise loss on the rows being scored. That tunes a hyperparameter on the test fold, which is true leakage and not a per-spectrum chemometrics convention. The same code produces the classification `y_proba` (lines 1928-1933). It also makes Params inconsistent: the CV score comes from per-fold early-stopped models, while the full-data refit at line 2056 (and the reported n_estimators or iterations) uses the full tree count. Under LOO the eval_set is the single scored sample. The 2026-05-04 SESSION_LOG entry treats this only as an in-app vs export parity problem (it copied the same eval_set into the export). It is never discussed as leakage, and none of the 'chemometrics convention' decisions cover it.

**Failure scenario.** Pure-noise target (y ~ N(0,1), X ~ N(0,1), 50x30), 5-fold KFold, LightGBM with n_estimators=300 and lr=0.1, fitted through dasp's `_fit_with_early_stopping` with the test fold as eval_set. Mean RMSEcv over 8 datasets was 0.988, against 1.143 for a plain fit, with std(y)=0.996. In 3 of 8 datasets it gave RMSE below std(y), i.e. a positive R2cv on noise. In a Bayesian run TPE picks trials for this inflated score, and the leaderboard ranks boosters above PLS/Ridge on a biased basis. The user's 2026-05-04 report reproduced it: in-app Accuracycv 1.0 vs 0.976 with a plain fit.

**Verifier.** Early stopping is on by default: `early_stopping_rounds=40` in run_unified_bayesian, gated by `_supports_early_stopping` for XGB/LGBM/CatBoost. The regression and classification objectives call `cross_val_predict_with_early_stopping`. That loops over `cv.split` and calls `_fit_with_early_stopping(..., X_val, y_val, ...)` with the fold's own test rows as `eval_set` (cv_utils.py:639), then predicts those same rows. So the tree count is chosen on the scored data. No doc describes this as intentional. The 2026-05-04 SESSION_LOG_ARCHIVE entry (line 2842ff) and PROJECT_STATUS_ARCHIVE:287 treat it only as an in-app vs export parity bug, and the fix copied the same eval_set into the export. That entry itself shows the bias: Accuracycv 1.0 with ES vs 0.976 with a plain fit on the user's data. The same helper is used by grid search `_run_single_fold`, so the effect reaches beyond the Bayesian path. The final full-data refit uses the full n_estimators, so the reported CV score does not describe the model that is saved.

### R004: Loading a new dataset keeps the previous dataset's exclusions, validation split and validation snapshot
`spectral_predict_gui_optimized.py:19430` · area gui-1 · leakage · finder high → verifier **high / CONFIRMED**

**Problem.** _load_and_plot_data replaces X_original/X/y/ref/metadata but never clears excluded_spectra, validation_indices, validation_X/validation_y or validation_enabled. The only reset is the manual button. The calibration-transfer data-replace path at gui:51635 does reset validation, with the comment 'stale after data replacement', but the main Import path does not. The worker then filters the NEW data by the OLD labels, and it computes RMSEP/R2pred against the OLD validation_X/validation_y snapshot.

**Failure scenario.** Load combined file A (no ID column, so IDs are generated as Sample_1..Sample_N), create a 20% SPXY validation set and exclude a few outliers. Then load combined file B, which also gets Sample_1..Sample_M. Run Analysis: rows of B whose labels match A's validation/excluded labels are silently removed from training, and the reported RMSEP/R2pred are A's spectra and A's targets scored by models trained on B. If B's labels don't collide, nothing is held out, yet validation metrics are still computed on dataset A and shown as B's external validation.

**Verifier.** _load_and_plot_data (gui:18877-19509) replaces X_original/y/ref/metadata and refreshes active groups. It never clears excluded_spectra, validation_indices, validation_X/validation_y or validation_enabled. The only resets are the manual buttons (9273, 20471, 20923) and the calibration-transfer replace path (51636), which explicitly calls validation 'stale after data replacement'. The worker filters the new data by the old label sets. All validation-metric sites (30038, 30523, 30758, 30979) pass self.validation_X.values straight through, with no check that the snapshot belongs to the current dataset. The resume-fingerprint logic only covers resumed Bayesian runs. Not demonstrated in the live GUI, but the code path is unambiguous.

### R005: validation_X/validation_y snapshots are not rebuilt after a wavelength-range update, working-data replacement or later exclusion
`spectral_predict_gui_optimized.py:19716` · area gui-1 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** _create_validation_set freezes validation_X = self.X.loc[...] at creation. The data-type and x-unit conversions re-slice it, but _update_wavelengths/_apply_wavelength_filter, _replace_working_data (rubberband/ALS/airPLS/polyBL/custom) and _mbl_replace all replace self.X without touching validation_X. Validation metrics then feed stale spectra into models trained on the new ones. compute_validation_metrics_for_top_models maps all_vars wavelengths to column positions on the training axis and applies the same positions to X_val. With a wider stale X_val, variable-selection models therefore predict on different wavelengths with no error, because SNV/SG preprocessing is row-wise and accepts any width.

**Failure scenario.** Create a validation set on 350-2500 nm, then narrow Import & Preview to 1100-2500 and click Update. Train with UVE/CARS varsel: each row's column indices (relative to 1100 nm) index into the 350 nm-based validation matrix, so R2pred/RMSEP are computed on the wrong wavelengths and reported as valid. Similarly, after 'Replace working data' with ALS baseline correction, calibration is baseline-corrected while the validation spectra stay raw, which silently inflates RMSEP. A sample excluded after the split still stays in validation_X and is scored.

**Verifier.** _apply_wavelength_filter/_update_wavelengths rebuild self.X from X_original and never touch validation_X. Only the data-type and x-unit conversions re-slice it. compute_validation_metrics_for_top_models has no X_train vs X_val width check. It maps all_vars wavelengths to column positions on the training axis and applies those positions to X_val_preprocessed, so a wider stale snapshot is scored on the wrong wavelengths without any error.

### R006: Spectrum-click exclusion int-coerces numeric-string IDs, so it never reaches the analysis
`spectral_predict_gui_optimized.py:21071` · area gui-1 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** Plot lines store gid=str(label). Both click handlers turn the gid back into an int whenever it is all digits. read_combined_csv always casts specimen IDs to str (io.py:1507/1519), so a combined file with a numeric ID column has labels like '5'. In Tab 1, int 5 is added to excluded_spectra and never matches the str index. pos_idx is also taken as the positional 5, so the annotation shows the next sample's ID, Y and metadata. In Explore, get_loc(5) raises KeyError and the handler returns, so clicking cannot exclude anything there.

**Failure scenario.** Load a combined CSV with a 'sample_id' column of 1..N and click the spectrum for sample '5' in Import & Preview. The line turns dotted, the status says '1 spectrum excluded', and the popup shows sample '6's Y value. On the next replot the line is solid again ('5' not in {5}), and Run Analysis trains on sample '5' while logging that 1 spectrum was excluded. Scratchpad check2.py: index dtype str, clicked '5' -> sample_idx 5, pos 5 (label '6'), rows removed by the analysis mask: 0.

**Verifier.** read_combined_csv casts detected ID columns to str (io.py:1507/1519). Lines set gid=str(label), and _on_spectrum_click int-coerces an all-digit gid. It then treats the int as a positional index when it is in range(len(X)) and adds the int to excluded_spectra, where it never matches the str index in the worker's isin mask. _on_explore_spectrum_pick uses the same coercion and then calls get_loc(int), which raises KeyError, and the handler returns.

### R007: Quality Check 'Mark for exclusion' excludes position+1, not the sample label
`spectral_predict_gui_optimized.py:22495` · area gui-1 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** _run_outlier_detection passes a numpy array (self._apply_transformation(self.X.values)) to generate_outlier_report, so outlier_summary['Sample_Index'] holds positional indices 0..N-1. The table shows Sample_Index+1, and _mark_selected_for_exclusion adds that displayed 1-based integer straight into excluded_spectra. Every other consumer treats that set as DataFrame index labels (the worker uses X.index.isin(excluded)). So the outlier table's exclusions either match nothing (string labels) or remove the wrong sample (integer labels).

**Failure scenario.** Combined CSV or ASD data with string IDs (e.g. 'S001'...). Run outlier detection, tick 'select high confidence', click Mark for Exclusion: excluded_spectra becomes {5, 17}. The status reads '2 spectra excluded' and the worker logs 'Excluding 2 user-selected spectra', but X.index.isin({5,17}) is all False and the flagged outliers stay in calibration. With integer labels 0..N-1 (e.g. Data Management RangeIndex data), the sample after each flagged outlier is excluded and the outlier itself is kept.

**Verifier.** _run_outlier_detection passes self._apply_transformation(self.X.values), which is a numpy array, to generate_outlier_report. outlier_detection.py:579-582 then uses list(range(n_samples)) for Sample_Index (its docstring says so). _populate_outlier_table shows Sample_Index+1, and _get_sample_index_from_tree_item/_mark_selected_for_exclusion add that 1-based int to excluded_spectra. The worker (gui:29308) and the other exclusion routes (_resolve_specimen_label, gui:20503-20536) treat the set as index labels. Nothing in SESSION_LOG, PROJECT_STATUS or CHANGELOG documents this as intended.

### R008: Bayesian 'advanced' baseline does nothing in the search, but results, Tab 7 and validation apply it
`spectral_predict_gui_optimized.py:30403` · area gui-2 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** The Bayesian options combobox offers 'advanced' (line 12660), and the worker passes it to run_unified_bayesian via _get_baseline_params_for_method(_setting('bayes_baseline_method')). unified_bayesian.apply_preprocessing only handles polynomial, als, rubber_band and airpls; 'advanced' falls to `bl = None`, so no baseline is applied. The trial is still labelled 'advanced+<core>' with baseline_method='advanced' in the row. Every consumer that rebuilds the row uses build_preprocessing_pipeline, which does implement 'advanced' (pybaselines): Tab 7 (which sets enable_baseline/baseline_method from the prefix), compute_validation_metrics_for_top_models, and save. The ranked CV metrics therefore describe a model without baseline correction, while the refit, RMSEP and saved model use one with it. TPE also spends trials on a toggle that has no effect.

**Failure scenario.** Tick Bayesian Baseline Correction with method 'advanced' and run. The top row reads 'advanced+snv'. Its RMSEcv comes from plain SNV, but the validation RMSEP and the Tab 7 refit use arPLS+SNV, so the reported and reproduced numbers disagree and the saved model differs from the one that was ranked. A scratch script confirmed that apply_preprocessing(X, {'apply_baseline': True, ...}, baseline_method='advanced') equals the unbaselined X, and that the display name is 'advanced+raw'.

**Verifier.** The GUI combobox offers 'advanced'. _get_baseline_params_for_method returns ('advanced', {algorithm, lam}), and the worker passes that to run_unified_bayesian. The objective's CV uses apply_preprocessing (unified_bayesian.py 1302), which has no 'advanced' branch, so `bl = None` and no baseline is applied. Meanwhile _build_display_preprocess_name labels the row 'advanced+...' and the row stores baseline_method='advanced'. build_preprocessing_pipeline (preprocess.py 505) does implement 'advanced' via pybaselines, so the validation, refit and save paths use a model different from the one that was ranked. This is a ranked-versus-reproduced mismatch.

### R009: Tab 7 maps selected wavelengths to columns with ±0.5 tolerance and takes the first hit; predict uses ±0.01. Train/predict features differ on fine grids
`spectral_predict_gui_optimized.py:41351` · area gui-3 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** Path A (which is always used, since use_full_spectrum_preprocessing is hard-coded True at line 40242) and the one-class refit path (line 40780) find each selected wavelength's column with `np.where(np.abs(original_wavelengths - wl) < 0.5)[0][0]`. When the grid spacing is below 0.5 units, the first index inside that window is the neighbouring channel, not the exact match. So the model is trained, cross-validated and validated on columns shifted by one channel. The saved metadata, however, records the exact selected wavelengths (refined_wavelengths = list(selected_wl)). At prediction, model_io step 3 (model_io.py:677) looks them up with a 0.01 tolerance, so it feeds the correctly aligned columns. The saved model therefore receives different features at prediction than it was trained on. The CV and validation metrics shown in Tab 7 describe a model that the .dasp file does not reproduce. The same file already documents ±0.5 nm matching as a bug elsewhere (lines 2071, 2174, 2343).

**Failure scenario.** MIR/FTIR data at 0.482 cm-1 spacing (or Raman/NIR at 0.25 nm). Selected wavenumbers 3999.036, 3997.59 and 3996.144 are trained on columns 3999.518, 3998.072 and 3996.626 (verified with a scratch script), but predicted on 3999.036, 3997.59 and 3996.144. Every feature is off by one channel. A full-spectrum model maps index k to k-1, so the first column is duplicated. On SG-derivative spectra this gives badly biased predictions from a model whose Tab 7 R² looked fine.

**Verifier.** Lines 40780 and 41351 both use `np.where(abs(original_wavelengths - wl) < 0.5)[0][0]`. use_full_spectrum_preprocessing is hard-coded True at 40242, so Path A always runs. X_work, the CV and the final refit all use these indices. refined_wavelengths stores the exact selected_wl (42171), and model_io step 3 (line 677) matches it at 0.01 tolerance. Whenever grid spacing is below 0.5, training uses the neighbouring channel and prediction uses the exact one. No resampling or grid guard exists upstream. The file's own docstrings (2071/2174/2343) already call ±0.5 matching a bug. The impact is limited to fine grids (spacing under 0.5), which covers high-res FTIR and some Raman or 0.25 nm NIR. The standard 1 nm ASD grid is unaffected.

### R010: A stale bias/nonlinear correction from an earlier model is saved into a newly trained model
`spectral_predict_gui_optimized.py:42601` · area gui-3 · persistence · finder high → verifier **high / CONFIRMED**

**Problem.** _save_refined_model embeds `self.nonlinear_correction_data` (when 'use nonlinear' is ticked) or `self.bias_correction_data`, with no check that either was computed for the model being saved. nonlinear_correction_data is set only when the user presses compute (line 37671) and is never reset when a new model is trained. _update_bias_correction_ui returns early for non-regression tasks without clearing bias_correction_data. predict_with_model then applies apply_correction (bias + slope*y, or a polynomial) to every prediction the saved model makes.

**Failure scenario.** The user trains PLS for protein, computes a cubic nonlinear correction, and leaves 'apply' and 'use nonlinear' ticked. They then train SVR for moisture (or rerun with other wavelengths) and click Save. The .dasp now contains protein's polynomial, and every moisture prediction in Tab 8, Tab 9 and CT is silently remapped through it. For a classification model trained after a regression run with correction on, the stale linear correction is saved too: string labels make apply_correction raise, so the model 'fails' with only a console message, and integer labels get shifted to non-class floats.

**Verifier.** nonlinear_correction_data is assigned only at init (3037) and in _compute_nonlinear_correction (37671). Nothing clears it after training, and _update_bias_correction_ui (called at 42329) recomputes only the linear bias_correction_data. The 'use nonlinear', 'apply' and 'save with model' BooleanVars persist, with save defaulting to True. Save (42601-42606) embeds whichever dict is present without checking which model it came from. For a non-regression task, _update_bias_correction_ui returns before touching bias_correction_data, so the previous regression's linear correction stays and is saved if 'apply' is still ticked. save_model writes it unconditionally (model_io 332-336), and predict_with_model applies it to every prediction (821-824), calling np.asarray(..., dtype=float) on the labels. A saved model therefore silently gets the wrong correction. This requires the user to have left the correction checkboxes ticked.

### R011: File Equalize names exported transformed spectra after the wrong input files
`spectral_predict_gui_optimized.py:47974` · area gui-3 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** The spectra come from _load_spectra_from_directory: read_asd_dir order (all ASD extensions merged and Path-sorted) or, for CSV, `sorted(glob.glob(...))`, an ordinal string sort. The sample IDs come from `input_path.glob(f'*{ext}')`, which is grouped by extension and not sorted, so it comes back in directory (NTFS, case-insensitive) order. The two lists are paired by position whenever their lengths match. Output files are then named `{model}_{stem}` and contain a different sample's spectrum.

**Failure scenario.** A CSV folder holds s_1.csv, s_2.csv, sa.csv and sb.csv. The loader order is [s_1, s_2, sa, sb], but the ID order is [sa, sb, s_1, s_2] (verified on this machine). The file written as '..._sa.csv' contains s_1's transferred spectrum, and so on. An ASD folder mixing .asd and .sig files shows the same misassignment. The exported, equalized dataset is silently scrambled.

**Verifier.** _file_equalize_batch loads spectra with _load_spectra_from_directory, which uses sorted(glob.glob) for CSV and read_asd_dir's order for ASD. It takes IDs from unsorted Path.glob grouped by extension, which returns NTFS directory order, and pairs the two lists by position whenever the lengths match. I ran the real loader on a temp folder of legacy-format CSVs. The output file IDs are attached to other files' spectra. This affects filenames that mix '_' with letters at the same position, mixed case, or mixed ASD extensions (.asd/.sig).

### R012: CT prediction and transform paths silently extrapolate spectra beyond the measured range
`spectral_predict_gui_optimized.py:50968` · area gui-3 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** resample_to_grid uses interp1d(..., fill_value='extrapolate'). _run_prediction_workflow resamples the satellite data onto the transfer grid (50952), then resamples the transferred spectra onto the prediction model's wavelengths (50968), and never checks that the target grid lies inside the source range. The load dialog accepts any partial overlap ('Uses linear interpolation'). _load_and_predict_ct only warns about the satellite-to-transfer step. Its 'Extrapolation Warning' compares target_wl with a model_wl_range that is derived from target_wl itself when metadata lacks 'wavelength_range' (which save_model never writes), so it can never fire. _transform_spectra (51496) and _file_equalize_batch (48007) extrapolate the same way into exported files. The library search also aligns through extrapolation.

**Failure scenario.** The transfer model's common grid is 1100-2400 nm and the prediction model was trained on 1000-2500 nm. The 100 nm at each end fed into PLS are straight-line extrapolations of the edge slopes. Predictions come out biased with no warning. With SG derivatives, the extrapolated edges contaminate neighbouring channels too.

**Verifier.** resample_to_grid uses interp1d with fill_value='extrapolate'. _run_prediction_workflow resamples onto transfer_model.wavelengths_common and then onto the prediction model's wavelengths (50968) with no range check. In _load_and_predict_ct, model_wl_range falls back to target_wl's own min and max because no model metadata ever contains 'wavelength_range' (grep finds it only in io readers and data_management). The 'Extrapolation Warning' therefore can never fire. A small resample of 1100-1300 onto 1000-1400 returned linear extrapolations at both ends. A common grid narrower than the primary model's range is the normal CT situation.

### R013: Tab 9, CT Mode A and CT Mode B R/A conversions use (and overwrite) the training data's reflectance scale
`spectral_predict_gui_optimized.py:53457` · area gui-3 · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** _convert_reflectance_to_absorbance divides by `self.data_value_scale`, which is the main dataset's reflectance scale. It also sets `self.data_value_scale = 100.0` when it auto-detects percent data. The Tab 8, CT Section A and contaminant paths save and restore that attribute around the call. _comparison_convert_data_type (53457/53459, also run automatically by live monitoring at 53633), _ct_pred_convert_data_type (50834/50836) and _ct_convert_to_absorbance/_ct_convert_to_reflectance (51966/51989) do not.

**Failure scenario.** Case 1: the training data was % reflectance (data_value_scale=100) and the model is in absorbance. Tab 9 comparison data (or CT Mode A satellite data) arrives as 0-1 reflectance. Convert computes log10(100/R), so every absorbance is 2.0 too high and every prediction is garbage, with no warning. Case 2: the training data was 0-1 and a percent-scale comparison file is converted. The global data_value_scale is permanently set to 100, and a later absorbance-to-reflectance conversion of the main dataset multiplies it by 100.

**Verifier.** _convert_reflectance_to_absorbance reads self.data_value_scale and overwrites it with 100 when percent data is auto-detected. The Tab 8 (44391-44401), CT Section A (51814/51884) and contaminant (59426) paths save and restore it. _comparison_convert_data_type (53457), which live monitoring auto-calls at 53633, _ct_pred_convert_data_type (50834) and _ct_convert_to_absorbance/_ct_convert_to_reflectance (51966/51989) do not. I called the real method with a stub self: at scale 100, 0-1 input R=0.5 gives absorbance 2.301 instead of 0.301. Converting percent-scale data when the scale is 1.0 leaves data_value_scale=100 permanently.

### R014: With Y-transform plus early stopping, the saved model is trained on untransformed y while metadata records the transform
`spectral_predict_gui_optimized.py:41544` · area io-persistence · persistence · finder medium → verifier **high / CONFIRMED**

**Problem.** For boosters with early_stopping_rounds, the Y-transform is deliberately applied by hand only inside the CV fold loop, and pipe is not wrapped in a TTR. The final fit, `final_pipe = clone(pipe); final_pipe.fit(X_raw, y_array)`, therefore trains on RAW y with no transform. The saved .dasp contains that untransformed model, but metadata['y_transform'] records e.g. 'log'. The reported CV R²/RMSE describe a log-target model that was never saved, and anyone reading the metadata is misled about the model inside the file.

**Failure scenario.** XGBoost with early_stopping_rounds=40 and Y-transform='log' on a right-skewed target. CV metrics come from log-y models, but the saved model is fitted on raw y, so its predictions and error distribution differ from what was validated, while the file claims y_transform='log'.

**Verifier.** When _needs_es is set (a booster with early_stopping_rounds > 0), GUI:41544 skips the TTR wrap. The y-transform is then applied only inside the per-fold ES loop (GUI:41609-41641). No later code references y_transform_active or _y_transformer. The final fit at GUI:41956, `final_pipe.fit(X_raw, y_array)`, trains on raw y, yet GUI:42470 writes metadata y_transform='log'. The .dasp therefore holds a model other than the one whose CV metrics were reported. That matches the 'wrong model reproduced at predict time' class, so I raised severity to high. Boosters loaded from Results normally carry early_stopping_rounds, so every booster with a Y-transform hits this. This verdict rests on code reading; I did not run a GUI repro.

### R015: Export bundle ships already-preprocessed data but its script preprocesses again and indexes full-spectrum columns
`src/spectral_predict/export_bundle.py:263` · area io-persistence · reproducibility · finder high → verifier **high / CONFIRMED**

**Problem.** The GUI passes data_X = refined_X_train to create_export_bundle. On Path A (always used) that array is already preprocessed and subset to the selected wavelengths (X_work). ExportBundle writes it to data/spectra.csv, but generates python/analysis.py (and the R wrapper) with include_data=False. The script therefore re-applies SNV/SG derivative/autoscale to the already-processed matrix, then applies variable_indices, which are indices into the FULL wavelength list (GUI:42728-42748), to a matrix that has only the subset columns.

**Failure scenario.** An sg1 PLS model on a 30-wavelength subset of 100 is exported as 'Complete Bundle'. spectra.csv has 30 columns, but analysis.py runs `apply_savgol_derivative(X, derivative=1, ...)` (a second derivative on derivative data) and then `X_processed[:, [40..69]]`, which raises IndexError. With no subset (all wavelengths) the script runs silently on doubly-derived spectra and reports metrics that do not match the GUI.

**Verifier.** GUI:41568 sets X_raw = X_work, which is preprocessed and subset. GUI:42174 stores it as refined_X_train, and GUI:42818/42831 passes it to create_export_bundle. _add_data_files writes it verbatim to data/spectra.csv. _add_python_files, however, generates the script with include_data=False, so CodeGenerator emits the preprocessing and full-spectrum variable_indices. The embedded exports (python_embedded/colab) skip preprocessing correctly because include_data=True; only the bundle path is broken. With a subset the bundled script raises IndexError. With full spectrum it runs silently on doubly-derived data and reports metrics that do not match the GUI.

### R016: Saving a classifier crashes when the label encoder has numeric classes, and the stale Bayesian encoder would mis-decode labels
`src/spectral_predict/model_io.py:183` · area io-persistence · persistence · finder high → verifier **high / CONFIRMED**

**Problem.** save_model writes `label_mapping = dict(zip(label_encoder.classes_, ...))`. For numeric labels the keys are np.int64, which json.dump rejects ('keys must be str...'), so the save fails. The GUI reaches this path after any Bayesian classification search: GUI:30677 fits a LabelEncoder on y even when labels are numeric. In refinement, local_label_encoder stays None for numeric y, and _save_refined_model falls back to `refined_label_encoder or self.label_encoder`. If the JSON step were fixed, predict_with_model would apply inverse_transform to raw class predictions, shifting the labels.

**Failure scenario.** Classes {1,2,3}, Bayesian search, then refine an RF, then Save Model: 'Failed to save model: keys must be str, int, float, bool or None, not int64', and no .dasp is written. Had it saved, the loaded model maps predictions 1→2 and 2→3, and 3 raises 'y contains previously unseen labels' (le_check.py).

**Verifier.** GUI:30674-30679 fits a LabelEncoder on y_np for every Bayesian classification search, whatever the dtype. Refinement creates local_label_encoder only for non-numeric y (GUI:40713), so for numeric labels refined_label_encoder is None. GUI:42573 then falls back to self.label_encoder. In save_model (model_io.py:183), dict(zip(classes_, ...)) produces np.int64 keys, and json.dump rejects them. The GUI catches the error, so no .dasp is written. Numeric class codes (0/1, 1/2/3) are common, so after any Bayesian search this blocks saving a core classification model. I kept high because the reach is broad, although the failure itself is a loud crash.

### R017: OPUS reader returns the background reference single-channel instead of absorbance
`src/spectral_predict/readers/opus_reader.py:89` · area io-persistence · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** The 'priority' chain is guarded by `if spectrum is None and ...`, but `spectrum` is never assigned inside the chain. Every later block therefore overwrites x_data/y_data, and the LAST block present wins. Standard Bruker OPUS files carry AB plus ScSm plus ScRf, so read_opus_file returns the ScRf reference single-channel with data_type='reference'. That background is usually identical or near-identical across a batch, so X carries no sample information. The io.read_opus_file wrapper also merges `**file_metadata` last, so its mapped data_type is overwritten by the raw 'reference'/'transmittance' string.

**Failure scenario.** A folder of OPUS .0 files each containing AB, ScSm and ScRf, loaded via read_opus_dir. With a mocked brukeropus (a=0.5, sm=111, rf=999) read_opus_file returns data_type 'reference' and value 999.0 instead of absorbance 0.5. Every downstream model trains on background spectra.

**Verifier.** `spectrum` is initialised to None at opus_reader.py:69 and not assigned again until after the chain (line ~153). Every `if spectrum is None and ...` guard is therefore always true, and the last available block overwrites x_data/y_data. The installed brukeropus really does expose blocks as attributes named a/sm/rf/t (brukeropus/file/file.py data_keys; constants.py 'rf': 'Reference Spectrum'), so a standard file with AB+ScSm+ScRf yields the background. io.read_opus_file (io.py:3454) wraps this reader, and `**file_metadata` is merged last, so data_type ends up 'reference'. This is silent training-data corruption for any OPUS batch that carries single-channel blocks.

### R018: Ensemble CV loop indexes a label-indexed pandas Series positionally: KeyError under the locked pandas 3
`spectral_predict_gui_optimized.py:25490` · area models-ensemble · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** y_filtered is a pandas Series that keeps the specimen-ID index. That index is set in io.py (X.index = specimen_ids) and filtered with boolean masks at 29298-29387. The ensemble CV loops do y_filtered[train_idx] with a positional integer array. In pandas 3.x (requirements-lock pins 3.0.5) Series.__getitem__ with an int array is label-based: a string index raises KeyError, and an int index with gaps from exclusions or a validation holdout raises KeyError. An int index in non-sorted order silently picks the wrong targets. Every ensemble method then fails in the per-method except block, which only logs '[X] ... failed'. ensemble.py 1893/1978 (create_auto_ensembles) have the same pattern.

**Failure scenario.** Load spectra with specimen IDs 'S0'...'S79', enable ensembles, run the search. Each ensemble method raises KeyError "None of [Index([1, 2, 3, ...])] are in the [index]". All ensembles are skipped and only a log line says so. With an integer ID index and one excluded spectrum, the same KeyError occurs.

**Verifier.** The locked and installed pandas is 3.0.5. io.py sets X.index = specimen_ids / aligned_ref_ids, and y_filtered is only mask-filtered (29297-29387, 29681) with no reset_index before _train_ensembles, so it keeps a label index. y_filtered[train_idx] with an int ndarray is label-based in pandas 3. It raises KeyError for string IDs and for int IDs with gaps, and silently picks the wrong rows for an unsorted int index. The per-method try/except turns this into a log line, so ensembles fail for the normal specimen-ID case.

### R019: With early stopping plus a Y-transform, the saved final model is trained without the transform that CV used
`spectral_predict_gui_optimized.py:41944` · area models-ensemble · correctness · finder medium → verifier **high / CONFIRMED**

**Problem.** For boosting models loaded with early_stopping_rounds, the Y-transform is applied by hand inside the CV loop and the pipeline is deliberately not TTR-wrapped (41544-41546). The final model is then final_pipe = clone(pipe).fit(X_raw, y_array) on raw y, with no transform and no inverse. The CV metrics, and any bias correction fitted on the CV predictions, describe the log/sqrt/power-transformed model, but the persisted model is an untransformed fit. metadata['y_transform'] records the transform, yet no backend code reads it.

**Failure scenario.** An XGBoost row with early_stopping_rounds=20 and Y-transform 'Log' on skewed y. CV R2 reflects the log-space model. The saved .dasp is a raw-y XGBoost with different predictions, and a linear bias correction computed on the log-model CV predictions is applied to it at predict time.

**Verifier.** When _needs_es is true, the pipeline is deliberately not TTR-wrapped (41544-41546) and the transform is applied by hand inside the CV folds (41610-41642). The final model is final_pipe = clone(pipe).fit(X_raw, y_array) on raw y (41944-41955), and no later code re-wraps it (no other uses of _needs_es/y_transform_active). Outside y_transform.py, nothing in src/spectral_predict reads metadata y_transform. The saved model is therefore not the model the CV metrics and the CV-based bias correction describe. That is a wrong model reproduced at predict time, so I raised it to high, although it only triggers with boosting + early_stopping_rounds + a Y-transform.

### R020: Y-transform (TTR) model is saved with a duplicate preprocessor, so saved-model predictions are preprocessed twice
`spectral_predict_gui_optimized.py:42105` · area models-ensemble · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** When Model Development wraps the whole pipeline in TransformedTargetRegressor, the save path does two things. It sets final_model = final_pipe, the full TTR whose regressor_ still contains the preprocessing and scaler steps. It also sets final_preprocessor = Pipeline(inner_steps[:-1]), the same fitted steps a second time. model_io.predict_with_model applies preprocessor.transform(X) and then model.predict(X_processed), so every step (StandardScaler, SNV, derivatives) runs twice at prediction time. Any regression model saved with Log/Log1p/Sqrt/Yeo-Johnson and at least one pipeline step before the model predicts garbage, while the CV and calibration numbers shown in the GUI (computed on final_pipe directly) look fine.

**Failure scenario.** Ridge with Y-transform 'Log', where the pipeline is [scaler, model] (scale-sensitive model). Direct TTR prediction gives R2 0.962. Prediction through the saved preprocessor followed by the saved TTR (the model_io order) gives R2 -2.206.

**Verifier.** In the TTR branch (GUI 42096-42112), final_preprocessor = Pipeline(inner_steps[:-1]) and final_model = final_pipe (the full TTR whose regressor_ still holds those steps). The later preprocessor block says 'final_preprocessor already set in TTR block' and does not change it. refined_model/refined_preprocessor go straight to save_model, and predict_with_model applies preprocessor.transform and then model.predict, so every step runs twice. Nothing in the save path detects a TTR.

### R021: With wrapped base models, ensemble weights and the stacking meta-learner are fitted on in-sample predictions
`src/spectral_predict/ensemble.py:382` · area models-ensemble · leakage · finder high → verifier **high / CONFIRMED**

**Problem.** With refit_base_models=False, which the GUI sets whenever any base model is a GA/NSGA/Combined/WavelengthSubset wrapper, the 'OOF' loop uses the full-data fitted model to predict each 'validation' fold. Region weights (RegionAwareWeightedEnsemble), expert assignments (MixtureOfExperts) and the stacking meta-model (StackingEnsemble) are therefore learned from training-set predictions, even though the docstrings and comments say 'OOF ... prevents leakage'. Overfit models such as RF/XGBoost look near-perfect in-sample and get the largest weight or meta-coefficient, so the combination is biased toward the most overfit member.

**Failure scenario.** The ensemble mixes a GA-preprocessed PLS (wrapped, so any_wrapped=True) with a RandomForest. RF in-sample RMSE is about 1/3 of its CV RMSE, so the region weights and the stacking Ridge put most of the weight on RF. On new data the ensemble is worse than PLS alone, and the leaked CV metrics (see the ensemble CV finding) hide this.

**Verifier.** With refit_base_models=False, which the GUI sets whenever any base model is a GA/Combined/WavelengthSubset wrapper, the 'OOF' loop uses the full-data fitted model to predict each validation fold (ensemble.py 379-389, 635-645, 882-892). Region weights, MoE experts and the stacking meta-model are therefore learned from in-sample predictions. SESSION_LOG (2026-09-14) documents that wrappers switch the CV mode, but not that the weights become in-sample. The effect is measurable on held-out data.

### R022: NSGA-II fitness early-stops boosters on the held-out CV fold, then scores that same fold
`src/spectral_predict/nsga2_search.py:1486` · area nsga-ga · leakage · finder high → verifier **high / CONFIRMED**

**Problem.** For LightGBM, XGBoost and CatBoost, _compute_prediction_error calls cross_val_score_with_early_stopping. In that helper, _fit_with_early_stopping passes the validation fold as eval_set, and the same fold is then scored. LightGBM and XGBoost predict at best_iteration. CatBoost keeps the best model by default when an eval_set is given, and for classifiers it even sets eval_metric='Accuracy'. So each fold's number of trees is tuned on the test fold, and the objective-1 error for boosters is optimistically biased. PLS, Ridge and the other models get no such benefit, so NSGA-II non-dominated sorting and knee or min-error selection are tilted toward boosters. The displayed metrics disagree with the objective: _compute_display_rmse, _compute_nir_metrics and _compute_classification_cv_metrics fit without early stopping, and classification Accuracycv is copied straight from the leaky objective. The project's own convention treats tuning on held-out data as real leakage. The 2026-05-04 parity fix copied this behaviour into the exports instead of removing it.

**Failure scenario.** Small classification set with models ['PLS','LightGBM']. Each LightGBM fold stops at the iteration with the best test-fold loss. SESSION_LOG_ARCHIVE records the in-app booster number as Accuracycv=1.0 against 0.976 for a plain fit with bit-identical params. NSGA-II ranks the LightGBM chromosome above an honestly equal PLS-DA and picks it as the min-error solution, and its Accuracycv column shows the inflated value.

**Verifier.** This is real. nsga2_search.py:1486-1501 routes boosters to cross_val_score_with_early_stopping, and cv_utils._fit_with_early_stopping passes the scored validation fold as eval_set: LightGBM/XGBoost at L639/647 and CatBoost at L663, where CatBoostClassifier also gets eval_metric='Accuracy'. So the test fold picks the tree count. The display helpers (_compute_display_rmse etc., L2696+) use plain cross_val_score/cross_val_predict. Classification Accuracycv is copied from the objective at L3892. Nothing in the docs marks this as accepted leakage: the 2026-05-04 archive entry only restored export parity with the in-app behaviour. One caveat: the pattern is not specific to NSGA-II. Grid _run_single_fold and the GUI refine path do the same thing, so the fix belongs in cv_utils. On 10 synthetic regression sets (n=40, LightGBM with NSGA's LightGBM params), early stopping on the test fold gave a mean RMSEcv of 1.572 against 1.825 for a plain fit. It was lower in 10 of 10, which is a large optimistic bias that favours boosters in the ranking.

### R023: Stored NSGA-II Params for LightGBM and CatBoost describe a different model from the one that was scored
`src/spectral_predict/nsga2_search.py:2436` · area nsga-ga · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** decode_solution builds the Params string from models.get_model() defaults plus a partial set of NSGA overrides. The fitness model, by contrast, comes from _build_model(), which uses different fixed values. For LightGBM the scored model has num_leaves=31 (regression), max_depth=-1 and reg_alpha=0.1. The stored Params instead carry get_model's num_leaves=15, max_depth=10 (5 for classification) and reg_alpha=0.3 (0.5 for classification), because these keys are not in nsga_overrides. For CatBoost the Params add bootstrap_type='Bayesian'/'Bernoulli' and min_data_in_leaf=1/5, which the scored model never had. Tab 7 refits, saved models and validation metrics built from 'Params' therefore train a model that differs from the one whose RMSEcv/Accuracycv chose it.

**Failure scenario.** Regression NSGA-II picks a LightGBM chromosome scored with 31 leaves, unlimited depth and reg_alpha 0.1. Loading the row into Model Development builds LGBMRegressor(num_leaves=15, max_depth=10, reg_alpha=0.3, ...). R2cv and RMSEcv no longer match the Results table, and the saved model is not the one that was selected. Verified by diffing _build_model params against the decode_solution Params: regression LightGBM {'reg_alpha': (0.1, 0.3), 'max_depth': (-1, 10), 'num_leaves': (31, 15)}; CatBoost {'bootstrap_type': (None,'Bayesian'), 'min_data_in_leaf': (None,1)}.

**Verifier.** decode_solution (L2436-2445) builds Params as get_model() defaults plus nsga_overrides. The LightGBM overrides do not include num_leaves, max_depth or reg_alpha, and the CatBoost overrides do not remove get_model's bootstrap_type or min_data_in_leaf. _build_model, which builds the scored model, uses different fixed values. Params is what _rebuild_model_from_row (validation) and the Tab 7 refit consume, so the model that gets refit or saved is not the one that was scored.

### R024: EstimatedEPO (default 'pca_diff' and 'bootstrap') and MultiGroupEPO remove random noise directions, not the contaminant
`src/spectral_predict/contaminant_analysis.py:677` · area one-class · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** _build_projection_matrix mean-centres the interferent library before the SVD. For 'pca_diff' the library is the mean difference plus small random noise (and for 'bootstrap' it is resampled mean differences). Centring subtracts the mean difference, which is the contaminant signal, and leaves only noise or sampling jitter. So the components that get projected out are unrelated to the contaminant. MultiGroupEPO.fit has the same bug (lines 2368-2374): the direction the groups share is lost. Only 'mean_diff' works, because a one-row library skips centring. The GUI 'EPO Projection' correction uses the default 'pca_diff'. The EPO 'influence' that feeds analyze_contaminant_influence's combined influence and exclusion regions is therefore noise.

**Failure scenario.** Synthetic test: 30 clean and 30 contaminated spectra with a Gaussian contaminant peak. Share of the contaminant direction captured by the removed subspace: mean_diff 0.92, pca_diff 0.08, bootstrap 0.07, MultiGroupEPO 0.07. After transform, 99.7% (pca_diff) and 91% (bootstrap) of the group mean difference is still there. With mean_diff it is 0%. The GUI still reports '✓ EPO Projection applied' and overwrites self.X.

**Verifier.** _build_projection_matrix mean-centres the library before the SVD. For pca_diff and bootstrap every row is roughly the mean difference, so centring removes the contaminant direction and the SVD then finds noise. MultiGroupEPO.fit (lines 2368-2374) does the same. EstimatedEPO's default estimation_method is 'pca_diff'. The GUI 'EPO Projection' path (_apply_epo_projection, line 60168) and contaminant_analysis.py:1772 both use that default. Nothing in SESSION_LOG, PROJECT_STATUS or CHANGELOG documents this as intended.

### R025: OSC removes the y-predictive direction (first PLS loading) instead of y-orthogonal variation
`src/spectral_predict/interference.py:491` · area one-class · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** OSC.fit takes the first PLS x-loading p of X on y, normalises it, and deflates X along it, calling it 'the Y-orthogonal variation we want to remove'. The first PLS loading is the direction most correlated with y. So OSC removes analyte signal and keeps orthogonal interference. This is the opposite of Wold et al. (1998) OSC.

**Failure scenario.** Synthetic test: 80 spectra, y peak at channel 30 and a large y-independent interferent at channel 70. The removed direction has |cos| 0.84 with the analyte band and 0.12 with the interferent. Correlation of the removed score with y is 0.71. PLS(2) RMSECV goes from 0.058 on raw data to 0.098 after OSC(1).

**Verifier.** OSC.fit takes the first PLS x-loading of X on y and deflates X along it. That loading is the y-predictive direction, so OSC removes analyte signal, the opposite of Wold OSC. OSC.transform removes the same direction. In the GUI, the search route is commented out (GUI 30853: interference_settings 'DISABLED'), and the Interference-tab route crashes (finding 2), so GUI users cannot reach it today. It is still reachable through public backend calls: build_preprocessing_pipeline(interference={'osc':...}), run_search(interference_settings=...) and the OSC class, and AGENT_COMPOSITION.md lists MSC/OSC as provided preprocessing. Anyone who uses it gets corrupted results.

### R026: Tab 7 refit maps each selected wavelength to the FIRST column within 0.5 units, which is the lower neighbour on grids finer than 0.5; predict then uses the exact column
`spectral_predict_gui_optimized.py:41351` · area preproc-varsel · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** The Tab 7 refit (regression/classification at L41351, one-class at L40780) turns selected_wl into column indices with np.where(|orig-wl|<0.5)[0][0]. That is a first match, not a nearest match. selected_wl has already been snapped to exact axis values. On any axis with spacing below 0.5 (e.g. ~0.3 nm spectrometers, or 0.24/0.48 cm-1 FTIR), the first index within 0.5 is a lower neighbour. So every selected variable shifts down one or two positions. The saved model still records self.refined_wavelengths = list(selected_wl), the true values. model_io's predict path matches with a 0.01 tolerance and feeds the true columns. The model is therefore trained on one set of columns and predicts on another.

**Failure scenario.** On an axis 350.00, 350.33, 350.66, … the selected wavelengths at indices [100, 101, 500] are fitted on columns [99, 100, 499] in the Tab 7 refit, while the model_io predict rule selects [100, 101, 500]. The refit R² and the saved model describe shifted wavelengths, and predictions on new spectra use different features from training.

**Verifier.** Tab 7 Path A (derivative + subset, L41351, and the one-class copy at L40780) maps each wavelength with np.where(|orig-wl|<0.5)[0][0], which is a first match. selected_wl is already snapped to exact axis values by _parse_wavelength_spec, or comes from _original_wavelength_order. The fitted model is trained on X_full_preprocessed[:, those indices]. self.refined_wavelengths = list(selected_wl) and refined_full_wavelengths are saved, and model_io matches with a 0.01 tolerance, so predict picks the exact columns. On any axis with spacing below 0.5 the training columns are shifted to lower neighbours, and prediction uses different features from training. It only affects instruments with sub-0.5 spacing (grids of 0.5 or more are unaffected), but where it applies the saved model is silently wrong.

### R027: Bayesian search silently skips the 'advanced' (pybaselines) baseline but records baseline_method='advanced' on the row
`src/spectral_predict/unified_bayesian.py:717` · area preproc-varsel · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** unified_bayesian.apply_preprocessing only handles polynomial/als/rubber_band/airpls; any other method gives bl=None and no correction. The GUI passes baseline_method='advanced' (arPLS etc.) straight to run_unified_bayesian. So every trial with apply_baseline=True is scored with no baseline correction. The result row still writes baseline_method='advanced' and its baseline_params (L3657/3663). Tab 7, the validation rebuild and model save rebuild through preprocessing_config_from_row → build_preprocessing_pipeline, which DOES apply BaselineAdvanced. The reported CV metrics therefore belong to a different pipeline from the one rebuilt and saved.

**Failure scenario.** The user picks Baseline = Advanced/arPLS (lam=1e3) and runs Bayesian optimisation. The top row says 'advanced' and reports R²cv from uncorrected spectra. Refitting that row in Tab 7 applies arPLS and gives a different R², and the saved model's preprocessing differs from what the search validated.

**Verifier.** The GUI Bayesian combobox (L12660) offers 'advanced', and _get_baseline_params_for_method passes {'algorithm','lam'} to run_unified_bayesian as baseline_method='advanced'. unified_bayesian.apply_preprocessing has no 'advanced' branch, so bl=None and the trial is scored without baseline correction. The row still records baseline_method='advanced'. build_preprocessing_pipeline does insert BaselineAdvanced, so the rebuilt and saved pipeline differs from the one whose CV metrics were reported. Nothing in the docs marks this as intended.

### R028: Boosting early stopping uses the CV test fold as eval_set, so XGBoost/LightGBM/CatBoost CV metrics are optimistically biased
`src/spectral_predict/cv_utils.py:634` · area search-cv · leakage · finder critical → verifier **high / CONFIRMED**

**Problem.** _run_single_fold (search.py:4522-4529) calls _fit_with_early_stopping(final_model, X_train, y_train, X_test_transformed, y_test, ...). That passes the held-out test fold as eval_set, so the boosting-round count is chosen using the test fold's labels and the same fold is then scored. cross_validate_with_early_stopping (cv_utils.py:885-896) and cross_val_predict_with_early_stopping (cv_utils.py:1061-1077) do the same thing. The Bayesian path (unified_bayesian.py:1847/1913/1928) and NSGA-II (nsga2_search.py:1497/1523) use these helpers. early_stopping_rounds=40 is the run_search default, so every booster row is affected unless the user turns it off. This is hyperparameter tuning on the test fold, a cross-sample use of y, and it is not covered by the documented chemometrics-convention exemptions.

**Failure scenario.** On 20 pure-noise regression datasets (n=60, p=30, LightGBM, 5-fold KFold), the mean R2cv was +0.017 with the in-app early stopping and -0.540 with a plain fit. The noise target looks as good as predicting the mean, while honest CV clearly shows the model overfits. Boosters are ranked against PLS/Ridge on inflated numbers. SESSION_LOG_ARCHIVE 2026-05-04 already shows the effect on real data: accuracy on the same folds was 0.976 with plain fit and 1.0 with eval_set=test fold. That entry framed it as export parity, not leakage.

**Verifier.** _run_single_fold (search.py:4521-4529) passes X_test_transformed/y_test (the fold that is then scored) to _fit_with_early_stopping, which uses it as eval_set (cv_utils.py:637/644). cross_val_predict_with_early_stopping does the same. The only doc mention (SESSION_LOG_ARCHIVE 2026-05-04) treats it as export parity; it never calls it an intended exemption. The default is early_stopping_rounds=40, so every booster row is affected. I lowered the rating from critical to high. The bias is large on noise, but on real signal the documented effect is modest (acc 0.976 vs 1.0), and only XGBoost/LightGBM/CatBoost rows are touched. It is still systematic CV leakage that tilts ranking toward boosters.

### R029: Binary labels other than {0,1} give inconsistent or NaN F1/Precision/Recall and crash ranking
`src/spectral_predict/search.py:4583` · area search-cv · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** Numeric labels are passed through unencoded (search.py:1615 encodes only non-numeric dtypes). For binary tasks, f1/precision/recall use average='binary', so the positive class is always 1. compute_specificity takes cm[0,0] (the lower sorted label) as the negative class. With labels {1,2}, class 1 is both the 'positive' class for recall and the 'negative' class for specificity, so Specificitycv equals Recallcv exactly. With labels {2,3} (or any pair without 1), f1_score raises inside try/except and every fold becomes NaN. F1cv is then NaN, CompositeScore = -Acc - 0.0001*NaN = NaN, and compute_composite_score's `rank(...).astype(int)` raises IntCastingNaNError, aborting the whole search at the end. In the repeated-CV branch, _f1/_ps/_rs (search.py:5057-5059) are not wrapped in try/except and raise immediately. The GUI passes numeric targets such as site codes 1/2 straight through.

**Failure scenario.** The same data with labels (0,1) gives Recallcv 0.35, Specificitycv 0.975. With (1,2) it gives Recallcv 0.975, Specificitycv 0.975 (Recall silently became the specificity of class 1). With (2,3) it gives F1cv/Recallcv/Precisioncv NaN, and compute_composite_score raises IntCastingNaNError 'Cannot convert non-finite values (NA or inf) to integer'.

**Verifier.** run_search encodes labels only when they are non-numeric (search.py:1615). The GUI grid path passes y_filtered straight to run_search. Binary F1/precision/recall use average='binary' (pos_label=1), while compute_specificity treats cm[0,0] (the lowest label) as the negative class. With labels {1,2} the metrics silently change meaning. With labels {2,3} the f1 exception is swallowed into NaN, and compute_composite_score's rank().astype(int) then aborts the whole search.

### R030: LOO classification metrics are averaged over 1-sample folds, so F1cv/MCCcv/Specificitycv are nonsense and Kappacv/ROC_AUCcv are NaN
`src/spectral_predict/search.py:5084` · area search-cv · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** For non-repeated CV, _run_single_config reports classification CV metrics as the mean of per-fold metrics. Under LOO each fold has one sample. Binary F1, precision and recall are 0 whenever that sample is the negative class (zero_division=0). MCC on one sample is 0. compute_specificity on a 1x1 confusion matrix returns 0. Kappa is NaN and AUC is skipped, so the mean of an empty list is NaN. The SESSION_LOG claim that 'concat is per-sample-natural' under LOO describes the regression path only. The classification non-repeated branch never pools. validate_cv_strategy_for_task explicitly allows LOO for classification.

**Failure scenario.** A perfectly separable binary dataset (40 samples, RandomForest) gives Accuracycv=1.0, BalancedAcccv=1.0 under LOO, but F1cv=0.5, Precisioncv=0.5, Recallcv=0.5, MCCcv=0.0, Specificitycv=0.0, Kappacv=NaN, ROC_AUCcv=NaN. The same data with 5-fold gives 1.0 for everything. SESSION_LOG_ARCHIVE:2207 records 'Kappa/MCC/Specificity=0' on a real LOO run and wrongly calls it 'Not a plumbing bug'.

**Verifier.** The non-repeated branch at search.py:5083-5098 averages per-fold F1/Precision/Recall/MCC/Kappa/Specificity. Under LOO each fold has one sample, so these metrics degenerate, and Kappa/AUC become the NaN of an empty mean. I reproduced it through the public run_search(cv_strategy='loo'). SESSION_LOG_ARCHIVE:2207 records 'Kappa/MCC/Specificity=0' and calls it 'Not a plumbing bug', but this repro shows it is a plumbing bug: perfectly separable data still gives MCC 0.

### R031: all_vars is written with %g (6 significant digits), so validation silently scores subset models on the full spectrum
`src/spectral_predict/search.py:5522` · area search-cv · correctness · finder high → verifier **high / CONFIRMED**

**Problem.** _run_single_config serialises the model's wavelengths as f"{w:g}". That rounds to 6 significant digits, so 3999.6419 becomes '3999.64'. compute_validation_metrics_for_top_models looks the parsed floats up by exact equality in {float(wl): idx} built from the real column values (search.py:1116-1121). When nothing matches, col_indices becomes None and the code falls through to the 'Full spectrum model' branch (1147-1150). A top-N or varsel subset row is then refit and scored on every wavelength. Its RMSEP/R2pred belong to a different model. The only sign is a printed warning.

**Failure scenario.** OPUS-style wavenumbers such as 3999.6419 - 1.9285*k, a PLS search with variable_counts=[10] and compute_validation=True: every subset row prints 'Only found 0/10 wavelengths'. Its R2pred equals the full-spectrum row's R2pred exactly (0.793427 in both), while R2cv differs (0.959 vs 0.831). The same %g writer produces all_vars for full models after wavelength filtering or edge masking, and FT-NIR wavenumbers above 9999.99 lose precision even when rounded to 2 decimals.

**Verifier.** all_vars is written with f"{w:g}" (search.py:5522/5526). compute_validation_metrics_for_top_models looks the parsed values up by exact float equality (1116-1121). If nothing matches, col_indices becomes None and the code falls into the full-spectrum branch. The subset row's R2pred/RMSEP are then the full-spectrum model's values. The only sign is a printed warning.

### R032: write_markdown_report fails on non-ASCII target names or an unwritable working directory, aborting the whole analysis tail
`src/spectral_predict/report.py:153` · area transfer-analysis · error-handling · finder high → verifier **high / CONFIRMED**

**Problem.** The report file is opened with open(report_path, 'w') and no encoding, so Windows uses cp1252. The target name goes into the heading, so any character outside cp1252 raises UnicodeEncodeError (Greek letters such as δ13C, δ15N or Δ, and CJK). The GUI also writes to Path('reports') relative to the working directory (spectral_predict_gui_optimized.py:31060). That is Program Files in a per-machine install, which SESSION_LOG 2026-09-14 already records as unwritable (the CatBoost catboost_info bug). write_markdown_report is called inside the main try block before ensembles run, self.results_df is set and _populate_results_table is scheduled. An exception there jumps to 'Analysis failed' after the whole search has finished.

**Failure scenario.** The user runs a multi-hour search with target column 'δ13C'. The search completes and the results CSV is written. write_markdown_report raises UnicodeEncodeError ('charmap' codec can't encode 'δ'). The GUI shows 'Analysis failed', ensembles are skipped, and the Results tab is never populated. The same happens when the app runs from an unwritable working directory, even if output_dir points somewhere writable.

**Verifier.** report.py opens the file with open(report_path, 'w') and no encoding, on both the empty and non-empty paths, and the target name goes into the '# Spectral Predict Report: {target}' heading. On this machine (cp1252, utf8_mode 0) any target outside cp1252 raises UnicodeEncodeError. In the GUI, write_markdown_report runs inside the main try block after the results CSV is written but before ensembles, self.results_df, the training cache and _populate_results_table. The except block then shows 'Analysis failed'. The GUI does not chdir and does not enable UTF-8 mode, and the installer shortcut has no WorkingDir, so writing Path('reports') relative to the cwd is also fragile, as the CatBoost SESSION_LOG entry already notes. The results CSV survives, so data is not lost, but the Results tab and ensembles are. δ13C and δ15N are very plausible target names for this user, so I kept it at high.

### R033: region_id is capped at n_top_regions-1, so enabling 'test all individual' removes the combined-region subsets and 'pairwise' adds almost nothing, while both change the study identity
`src/spectral_predict/unified_bayesian.py:1566` · area bayesian · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** The objective suggests `region_id` in [0, n_top_regions-1], i.e. 0..9 (the GUI never passes n_top_regions, so it is always 10), and indexes `dynamic_regions[min(region_idx, len-1)]`. `create_region_subsets` builds more subsets than that when the Bayesian toggles are on. With test_all_individual=True it returns 10 individual regions followed by the top2/top5/top10 combined subsets at indices 10-12, which no trial can reach. With test_pairwise=True it returns about 52 subsets, of which only the first 10 (5 individual, 3 combos, 2 pairs) are reachable. So turning on 'Test all regions individually' actually removes the multi-region combinations from the search, and 'pairwise' searches 2 of about 45 pairs. The GUI gives no sign of this, and both flags still change the study hash (`region_indiv=`, `region_pair=`).

**Failure scenario.** Synthetic 60x700 spectra with n_top_regions=10. Default: 8 subsets, all reachable. all_individual=True: 13 subsets, and ['top2regions', 'top5regions', 'top10regions'] are unreachable. pairwise=True: 52 subsets, and 42+ pair subsets are unreachable. A user who enables 'test all individual' to widen the region search gets a narrower one: no trial can ever evaluate a combined-region model.

**Verifier.** The objective suggests `region_id` in [0, n_top_regions-1] and clamps with `min(region_idx, len-1)` (1566/1598, and the same in the one-class branch at 1337). The GUI Bayesian call (gui ~30428-30455) passes `region_test_all_individual`/`region_test_pairwise` but never `n_top_regions`, so it is always 10. `create_region_subsets` orders the list as individual regions, then combos, then pairs. With test_all_individual the 10 individuals fill indices 0-9 and the top2/top5/top10 combos become unreachable. With pairwise only 10 of ~52 subsets are reachable. The effect is a search-space defect, not score corruption, so medium stands. Minor extra: in the default case (8 subsets), ids 7-9 all clamp to the last subset, so it gets 3/10 of the prior mass.

### R034: Multiclass Bayesian region (and UVE) selection correlates spectra with LabelEncoder integer codes, so the selected wavelengths depend on alphabetical class names
`src/spectral_predict/unified_bayesian.py:1584` · area bayesian · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** For classification, run_unified_bayesian label-encodes y to 0..K-1 (line 2732). The 'region' subset path passes that integer y to `create_region_subsets`, which ranks regions by mean |Pearson r| with y (regions.py compute_region_correlations). The 'uve' path calls `uve_selection(X, y)`, a PLS regression on the codes, and `compute_importances` does not pass task_type for UVE (line 1073). With 3 or more classes the codes impose an arbitrary order. A region that separates the middle-coded class from the other two has near-zero linear correlation, and renaming a class changes which regions are offered to TPE and which are reported. The resulting CV score is still honest, but the search space and the 'selected wavelengths' shown to the user are artefacts of label spelling.

**Failure scenario.** 90 spectra in 3 classes; class 'banana' has a strong band near 1250-1350 nm, and an ordinal-looking band sits near 2000 nm. With labels apple/banana/cherry the top regions are 2000-2100 nm. Relabel so the banana group sorts first (codes change), with identical spectra and groups: the top regions become 1250-1350 nm. Bayesian region trials, and the reported all_vars wavelengths, change with the class names.

**Verifier.** For classification, run_unified_bayesian label-encodes y to 0..K-1 (2729-2732). 'region' is always in available_methods (line 1254), and it calls `create_region_subsets(X_prep, y, ...)`, which ranks regions by |Pearson r| with the integer codes. `compute_region_correlations` has no task_type handling. The UVE path calls `uve_selection(X, y)`, a PLS regression on the codes; its signature has no task_type. With 3 or more classes the region ranking therefore depends on the alphabetical order of the class names. The CV score stays honest, which is why this is medium and not high. The grid-search path shares `create_region_subsets`, so this is a codebase-wide limitation, not one specific to Bayesian.

### R035: Re-attached TPE sampler reuses the same seed, so after 'auto' migration (or an 'always' resume) the next startup trials exactly replay trials 0..k-1 as duplicates
`src/spectral_predict/unified_bayesian.py:2481` · area bayesian · reproducibility · finder medium → verifier **medium / CONFIRMED**

**Problem.** `_migrate_study_to_sqlite` and the 'always' resume path both attach `_make_tpe_sampler(random_state, n_startup)`: a new TPESampler with the same seed (42 from the GUI). During the startup phase TPE draws from its internal RandomSampler RNG, which restarts from the seed. The migrated or resumed study therefore re-suggests exactly the parameter sets of trials 0..k-1. Every one hits the fingerprint cache, is replayed as a duplicate, counts toward n_trials and is filtered from the leaderboard. TPE's KDE then weights each of those points twice. Under the default 'auto' mode, every model whose median trial exceeds 1 s migrates at trial 10. That covers XGBoost, LightGBM, CatBoost, RF, SVM and IsolationForest, the expensive ones. So 10 of the 20 random-exploration trials are wasted and the run returns 10 fewer distinct configurations than requested. A crash-resume under 'always' before trial 20 replays the same way.

**Failure scenario.** Confirmed in dasp: `run_unified_bayesian(X, y, wl, 'Ridge', n_trials=30, enable_sqlite_persistence='auto')` with `_AUTO_THRESHOLD_S=-1` (forces migration, the same path a >1 s model takes). Trials 10-19 all carry `duplicate_of_trial` = 0..9, so the leaderboard has 20 rows instead of 30. Pure Optuna repro: TPESampler(seed=42) for 10 trials, copy_study, load_study with a new TPESampler(seed=42), 10 more trials. The params of trial i+10 equal those of trial i for all i.

**Verifier.** `_make_tpe_sampler(random_state, n_startup)` builds `TPESampler(seed=random_state)` fresh, and it is reattached after `copy_study` in `_migrate_study_to_sqlite` (line 2478-2482) and on the 'always' resume path (3155-3159). The GUI hardcodes random_state=42. With `_AUTO_WARMUP=10` and `DEFAULT_N_STARTUP_TRIALS=20`, migration happens during the random startup phase, and the restarted RandomSampler RNG re-emits the first k parameter sets. The fingerprint cache replays those as duplicates: they consume n_trials and are filtered from the leaderboard. Under 'auto' this hits every model whose median fit exceeds 1 s. Nothing in SESSION_LOG or PROJECT_STATUS notes it. Impact is wasted budget and double-weighted KDE points, not corrupted scores, so medium is correct.

### R036: Explore colour-by-metadata (spectra and PCA scores) indexes the reference table by position
`spectral_predict_gui_optimized.py:4167` · area gui-1 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _get_explore_color_map takes metadata_source[color_by].values[:n_samples] and assigns colours by row position. For directory-plus-reference loads, metadata_source is self.ref, the whole reference CSV as read by read_reference_csv. It is in reference-file order, includes unmatched rows, and is not reordered by align_xy. The colours therefore belong to other specimens. Explore PCA scores use the same map, so apparent group clusters, or their absence, are fabricated. The Quality Check PCA does this correctly with .loc[specimen_ids] (gui:21754).

**Failure scenario.** ASD directory plus a reference CSV listing samples in lab-ID order with a few extra rows for unscanned samples. Colour the Explore PCA by 'Site'. Spectrum i gets the Site of reference row i, so PC1-vs-PC2 groupings by site are scrambled or spurious, and the user draws conclusions about site separation (or picks an Analysis Subset) from the wrong labels.

**Verifier.** _get_explore_color_map takes metadata_source[color_by].values[:n_samples] positionally. For directory-plus-reference loads, self.ref is the full read_reference_csv output in reference-file order, including unmatched rows, and align_xy does not reorder it. The Quality Check PCA uses .loc by label, which shows the intended behaviour. The effect is limited to exploratory colouring, so medium is appropriate.

### R037: Data Management 'Use for Analysis' / 'Merge & Use' / filter / trim bypass X_original and metadata
`spectral_predict_gui_optimized.py:18338` · area gui-1 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** These actions assign self.X/self.y/self.ref directly. They do not update X_original, combined_metadata_df, exclusions or the validation snapshot, and they do not replot (_update_spectral_plots and _update_data_viewer are pass stubs). Later actions rebuild from the stale state: _update_wavelengths sets self.X = X_original (the Import-tab dataset), a data-type conversion converts both, and _on_target_column_changed takes Y from the old combined_metadata_df and reindexes it to the old X_original.

**Failure scenario.** Load file A in Import & Preview, then in Data Management load B and click 'Use for Analysis'. Changing the wavelength range in Import & Preview silently swaps the analysis spectra back to A while self.y stays B's. Switching the target in the Analysis tab rebuilds y from A's metadata over A's index. The worker then trains on A∩B labels (or on A spectra paired with B targets if IDs collide, as with generated Sample_k IDs), and nothing tells the user the dataset changed.

**Verifier.** The Data Management tab is live (gui:5864, button at 6100). _use_for_analysis assigns only self.X/self.y/self.ref. _update_spectral_plots and _update_data_viewer are 'pass' stubs. X_original, combined_metadata_df, exclusions and the validation snapshot are all left untouched. _apply_wavelength_filter later rebuilds self.X from the stale X_original, and _on_target_column_changed reindexes y to the stale X_original.index (gui:17545). Confirmed by code reading.

### R038: Sub-integer wavelength rejection leaves a half-loaded dataset (new y/X_original, old X)
`spectral_predict_gui_optimized.py:19673` · area gui-1 · error-handling · finder medium → verifier **medium / CONFIRMED**

**Problem.** When rounding collapses columns, _apply_wavelength_filter shows an error and returns without touching self.X. _load_and_plot_data has already assigned the new X_original, y, ref and metadata, so it continues and reports '> Loaded ...' using the previous dataset's self.X, then replots and repopulates from it. Nothing reverts the partial assignment. The worker's X/y guard only intersects labels, so whenever labels collide (generated Sample_k IDs, reused filenames) the old spectra are silently paired with the new targets.

**Failure scenario.** Session has dataset A (1 nm, 100 samples, IDs Sample_1..100) loaded. Load combined file B (FTIR at 0.48 cm^-1, 100 samples, generated IDs). The 'Sub-integer ... not supported' dialog appears, but the status bar then says 'Loaded 100 samples x N wavelengths' and plots show A. Run Analysis: X_run = A's spectra, y_run = B's targets, and the labels line up, so the worker's realignment accepts the pairing and trains a model on mismatched spectra and references.

**Verifier.** _apply_wavelength_filter shows an error and returns when rounding collapses columns, leaving self.X unchanged. The caller has already assigned the new X_original/y/ref/combined_metadata_df, ignores the return value, and goes on to report 'Loaded len(self.X)' and replot. The worker takes X_run/y_run = self.X/self.y. When lengths match and labels collide (for example generated Sample_k IDs), the index check passes and the old spectra are paired with the new targets. Confirmed by code reading, not by GUI repro. An error dialog is shown, which makes this less silent, so medium stands.

### R039: SPXY (the default splitter) is allowed for classification and uses LabelEncoder codes as Y distances
`spectral_predict_gui_optimized.py:20664` · area gui-1 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** For non-numeric targets _validation_spxy label-encodes the classes and adds Euclidean distances between the codes to the X distances. The repo already recorded that SPXY's d_y term is undefined for categorical labels and disabled it for multiclass_simca (SESSION_LOG_ARCHIVE:4020). The same categorical encoding still runs for 'classification' and 'one_class', and SPXY is the default algorithm. With 3 or more classes, the 'distance' between classes depends on alphabetical label order.

**Failure scenario.** Classification with classes {'cow','goat','sheep'} and the default SPXY 20%. Samples of 'cow' and 'sheep' (codes 0 and 2) are treated as twice as far apart as cow/goat, so the maximin selection over-samples the alphabetically extreme classes into validation. Renaming a class changes the holdout and therefore the reported R2pred/accuracy.

**Verifier.** _validation_spxy label-encodes non-numeric y and adds Euclidean code distances to d_X. SPXY is disabled only for multiclass_simca (gui:17279), even though the repo's own note (SESSION_LOG_ARCHIVE:4020) says d_y is undefined for categorical labels. With 2 classes the term is just a constant between-class offset, so the practical impact is mainly multiclass (3 or more classes), where the distances depend on label order. That selection bias affects the holdout but does not leak data, so medium (arguably low-medium).

### R040: GUI VIP screening uses var(y) instead of q_a^2, so Y-irrelevant X variance is ranked 'important'
`spectral_predict_gui_optimized.py:22731` · area gui-1 · numerical · finder medium → verifier **medium / CONFIRMED**

**Problem.** _compute_vip_screening computes ssy_comp = sum(T^2) * var(y). That is a constant factor across components, so each component is weighted only by its X-score variance and not by the Y variance it explains. The repo's own models.compute_vip uses SSY_a = q_a^2 * t_a't_a. The screening plot draws a VIP=1 threshold and calls it 'typically considered important', so nuisance bands with high X variance get flagged.

**Failure scenario.** Synthetic data (scratchpad check1.py): a strong X nuisance factor on cols 0-29 unrelated to y, and a weak y-signal on cols 40-45. The GUI formula gives nuisance wavelengths VIP up to 1.14, above the plotted threshold, and puts nuisance columns 5, 24, 0 and 28 in its top 10. models.compute_vip gives the nuisance at most 0.52 and the signal about 2.85. A user choosing analysis regions from the Explore/Quality Check VIP screen would keep baseline or scatter bands.

**Verifier.** ssy_comp = sum(T^2) * var(y) multiplies by the same constant for every component, so the weighting reduces to X-score variance. models.compute_vip uses q_a^2 * t_a't_a, and its docstring says per-component Y variance enters via y_loadings_. The error only affects the Explore/QC screening display, not trained models, so medium.

### R041: One-class run with an integer-coded target and a typed inlier label is always blocked
`spectral_predict_gui_optimized.py:29724` · area gui-2 · correctness · finder high → verifier **medium / CONFIRMED**

**Problem.** _run_analysis passes the typed label as a str. The worker checks `inlier_label not in y_filtered.values`; for an int64 y this is true for '1', so the label is converted with float() and becomes 1.0. check_one_class_inlier_guard then compares y.astype(str) ('1') with str(1.0) ('1.0'), finds zero inliers, and aborts with 'The active Analysis Subset excludes all samples for inlier class 1.0'. The same mismatch would also zero the inliers in the Bayesian backend's string comparison. Only a float64 y (for example a column containing NaN) or the auto-detect path works.

**Failure scenario.** Load a one-class dataset whose class column is integer (1 = clean, 2 = contaminated), type '1' in the inlier class box and click Run (grid or Bayesian). The analysis stops with a wrong 'Analysis Subset excludes all samples' error, although no subset is active. A scratch replication of lines 29724-29745 gave: int64 -> label 1.0 -> the guard blocks.

**Verifier.** resolved_inlier_label is the stripped Entry string. read_combined_csv returns an int64 y for an integer class column (verified). _normalize_mixed_type_labels only runs for object dtype. The worker converts '1' to 1.0 (`1.0 in int64.values` is True), and check_one_class_inlier_guard then compares astype(str) '1' with '1.0', finds zero inliers, and blocks with a misleading 'Analysis Subset excludes all samples' error. A float64 y works because astype(str) gives '1.0'. The run fails loudly and the blank-box auto-detect path is a workaround, so this is not corruption. I lowered severity to medium.

### R042: Bayesian and NSGA-II searches ignore the analysis wavelength restriction while the log says it is applied
`spectral_predict_gui_optimized.py:30366` · area gui-2 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** The worker parses enable_analysis_wl_restriction and custom regions into analysis_wl_min_value, analysis_wl_max_value and analysis_wl_regions_value. It logs 'Variable selection will be constrained to specified range/regions' for every method. Only run_search and run_one_class_search receive these values. The Bayesian regression/classification branch (X_np = X_filtered.values), the Bayesian one-class branch and the NSGA-II branch pass the full spectrum, and run_unified_bayesian has no restriction parameter. Rows can select wavelengths the user excluded (for example water bands). ensemble_metadata and training_data_cache still record analysis_wl_min, which suggests the restriction was honoured. docs/section2_analysis.md 4.1.4 documents the restriction without method caveats, and SESSION_LOG/PROJECT_STATUS do not mention this behaviour.

**Failure scenario.** Enable 'Restrict analysis range' with regions 1100-1350, 1500-1850 and choose Bayesian. The progress log prints the constrained regions, but top rows' all_vars include 1400-1450 nm and 1900-1950 nm, and the user publishes a model believing the water bands were excluded.

**Verifier.** The code is unambiguous. analysis_wl_min/max/regions reach only run_search (30970-30972), and run_one_class_search receives only min/max (29965-29966) and never the regions. The Bayesian regression/classification branch (X_np = X_filtered.values at 30366), the Bayesian one-class branch (29791) and run_nsga2_search (30706) all receive the full spectrum, and run_unified_bayesian has no restriction parameter. The progress log still claims 'Variable selection will be constrained'. I found no caveat for this in SESSION_LOG, PROJECT_STATUS or CHANGELOG. There is also a smaller gap: a one-class grid run ignores custom regions.

### R043: Re-ranking after a penalty change treats one-class and multiclass results as classification
`spectral_predict_gui_optimized.py:33612` · area gui-2 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _rerank_results is triggered by traces on variable_penalty, gap_penalty and use_rmsep_gap. It uses `task_type = "regression" if "R2cv" in cols else "classification"`. compute_composite_score has dedicated 'one_class' (BalancedAcccv-based) and 'multiclass_simca' (NoveltyAUC) branches, and the searches rank with them (search.py 7171, 8349). One-class results are re-ranked by -Accuracycv - 1e-4*F1cv, which on imbalanced inlier/outlier data promotes models that accept everything, and Rank is renumbered accordingly. Multiclass rows have no Accuracycv, so the call raises KeyError, is swallowed ('Re-ranking failed' printed to console), and the penalty change silently has no effect.

**Failure scenario.** Run a one-class screen with 90 inliers and 10 outliers, then nudge the Variable penalty spinbox. A model with Accuracycv 0.90 and BalancedAcccv 0.50 (flags nothing) jumps above one with Accuracycv 0.88 and BalancedAcccv 0.85, and the table's Rank 1 changes to the useless model.

**Verifier.** _rerank_results chooses 'classification' whenever R2cv is absent. One-class rows carry Accuracycv and F1cv (search.py 6480), so the ranking silently switches to accuracy. Multiclass SIMCA rows lack Accuracycv, so compute_composite_score raises KeyError, which is swallowed with a print.

### R044: A data-viewer cell edit rebuilds all data from the display strings: values rounded, index stringified, hidden excluded rows dropped
`spectral_predict_gui_optimized.py:35029` · area gui-2 · data-loss · finder medium → verifier **medium / CONFIRMED**

**Problem.** _populate_data_viewer formats every spectral value as f'{val:.5f}', targets as f'{val:.4f}', and Sample ID as str(idx). It shows only non-excluded rows when 'show excluded' is off. Any single cell edit calls _apply_data_viewer_edits, which rebuilds self.X, self.X_original and self.y from all sheet cells. The effects: (1) every spectrum is rounded to 5 dp and every target to 4 dp (small-magnitude derivative spectra or trace concentrations are destroyed); (2) the index becomes strings, so an integer-indexed dataset no longer matches self.excluded_spectra or self.validation_indices, and the next analysis silently trains on the excluded samples and on the validation holdout, while RMSEP is still computed on the stale validation_X; (3) with show-excluded off, the excluded rows are removed from X/y/X_original entirely; (4) X_original is replaced by the wavelength-filtered X, so wavelengths outside the import range are lost.

**Failure scenario.** Load a CSV with numeric sample IDs, create a 20% validation set, and fix a typo in one metadata cell in the Data Viewer. The next analysis's calibration set includes all 20% validation samples (`X_filtered.index.isin(_validation_rows)` matches nothing, int vs str), and the reported RMSEP is computed on samples the model trained on.

**Verifier.** The core claims hold. The sheet holds 5-dp and 4-dp strings, and _apply_data_viewer_edits rebuilds self.X, self.X_original and self.y from all sheet rows. Excluded rows are not in the sheet when show-excluded is off, so they are dropped. X_original is replaced by the filtered X. Rounding is severe for small values: 4.2e-05 becomes 4e-05. The specific failure scenario is weaker than stated. read_combined_csv already returns a str index for numeric sample IDs, so str(idx) round-trips and validation_indices still match. The int-versus-str mismatch is only possible for loaders that leave an int index, such as a wide CSV with numeric IDs via set_index, which I did not trace through validation.

### R045: Cancelling the Tab 7 mismatch dialog leaves a half-loaded config, so the next Run refits a hybrid of two rows
`spectral_predict_gui_optimized.py:36016` · area gui-2 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _on_result_double_click sets self.selected_model_config to the new row before calling _load_model_for_refinement. That function sets loaded_model_config and overwrites validation state and refine_cv_strategy/folds/repeats, then calls _validate_training_configuration. If the user answers 'No', that raises ValueError, which propagates out of the Tk callback. The wavelength spec, _original_wavelength_order, refine_model_type, refine_preprocess, window and hyperparameter widgets still hold the previously loaded row, and the Run buttons stay enabled. The refit reads Params/n_components, Deriv, Task, imbalance and early_stopping from the new selected_model_config but wavelengths, model type and preprocessing from the old widgets, then compares against the new row's R2cv.

**Failure scenario.** Load row A (PLS, sg2, 40 wavelengths) into Tab 7. Double-click row B (PLS, snv, 5 LVs), answer No to the mismatch warning, go to Tab 7 and click Run. The refit uses A's 40 wavelengths with deriv=0 and n_components=5 from B, and prints 'COMPARISON TO LOADED MODEL' against B's R2cv, with no indication the model is a mix.

**Verifier.** _on_result_double_click assigns self.selected_model_config before calling _load_model_for_refinement, and there is no try block around that call. _load_model_for_refinement sets loaded_model_config, validation_indices, excluded_spectra and refine_cv_*, then raises ValueError when the user answers No. The widgets it would set later (preprocess, model, window, wavelengths, at about 37000 onward) keep the old row's values, and the Run buttons stay enabled. The refit takes Params/n_components (40321-40350) and other fields from selected_model_config, so a hybrid refit is reachable. Answering No also leaves the rejected row's validation_indices and excluded_spectra applied globally. I verified this by reading the code, not by driving the GUI.

### R046: Loading a result into Tab 7 overwrites the global Analysis Config preprocessing toggles
`spectral_predict_gui_optimized.py:37146` · area gui-2 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _load_model_for_refinement writes the loaded row's affixes into the shared analysis Tk variables that the grid search, one-class grid, multiclass and capture_gui_settings all read: enable_baseline, baseline_method, enable_smoothing and use_autoscale. Merely inspecting a result in Tab 7 therefore changes the preprocessing of the next grid run without any visible notice on the Analysis Config tab. These keys are in CAPTURABLE_SETTINGS, so it also creates a spurious 'Settings differ from the interrupted run' prompt on a pending Bayesian resume.

**Failure scenario.** The user configures grid search with ALS baseline and smoothing, runs a Bayesian search, and double-clicks a Bayesian row without baseline or smoothing to inspect it. enable_baseline and enable_smoothing become False. The next grid run silently runs without baseline correction or smoothing, and its results are compared against earlier runs that had them.

**Verifier.** Lines 37146-37166 set self.enable_baseline, baseline_method, enable_smoothing and use_autoscale. These are the same Tk vars bound to the Analysis Config checkbuttons (11864, 11917, 11999) and read by _get_baseline_params (23689) for the grid worker. Tab 7 has no separate baseline or smoothing vars. The checkboxes do change on screen, but nothing alerts the user, and nothing in the docs marks this as intended.

### R047: Tab 7 one-class refit of Bayesian rows uses the live inlier-class box, not the label the search used
`spectral_predict_gui_optimized.py:40794` · area gui-2 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** Grid one-class rows carry 'inlier_class_label' (search.py 6486), but unified_bayesian.convert_study_to_dataframe writes no such column. _detect_refine_task_type therefore returns no label and _load_model_for_refinement leaves self.inlier_class_label unchanged. The refit takes `inlier_label = self.inlier_class_label.get()` and falls back to the row's (absent) column. If the Bayesian run used the auto-detected label (box blank), the refit is blocked by the guard with a misleading 'Analysis Subset excludes all samples for inlier class ''' message. If the user has since typed another class, the refit silently models a different inlier class, and the saved model's refined_config records that class.

**Failure scenario.** Run Bayesian one-class with the inlier box blank (auto-detect 'clean' confirmed), then double-click the top row and click Run in Tab 7. The run is blocked with an 'Analysis Subset' error. Alternatively, after typing 'clean_B' for another run, double-clicking an old row refits and saves a model with clean_B as the inlier class.

**Verifier.** Only grid rows (search.py 6486, 7089) write 'inlier_class_label'. convert_study_to_dataframe's row dict has no such column, and the GUI never adds it to Bayesian one-class results. The only setter of the inlier box is line 37015, which runs only when _detect_refine_task_type finds a label, so the box is left as it was. The auto-detected label from _run_analysis is local (resolved_inlier_label) and never written back to the box. With a blank box the refit reaches check_one_class_inlier_guard with '' and is blocked with the misleading subset message. With a stale typed value the refit silently models the wrong inlier class.

### R048: Tab 7 refit crashes whenever a Y-transform is selected (TransformedTargetRegressor has no .steps)
`spectral_predict_gui_optimized.py:41570` · area gui-2 · crash · finder high → verifier **medium / CONFIRMED**

**Problem.** When refine_y_transform is not 'None' and the model has no early stopping, the pipeline is replaced by YTransformWrapper.wrap(pipe, ...), which returns a sklearn TransformedTargetRegressor. The next statement does `final_model = pipe.steps[-1][1]`, and the fold loop later uses `pipe_fold.named_steps`. A TransformedTargetRegressor has neither attribute, so every Y-transform refit raises AttributeError and the Y-transform feature cannot be used for PLS, Ridge, SVR, RF and others. The TTR-specific save code at 42095-42113 is never reached. No test covers refine_y_transform.

**Failure scenario.** Load any PLS regression row into Tab 7, set Y-Transform to 'Log1p' and click Run. The refit fails with "'TransformedTargetRegressor' object has no attribute 'steps'". A scratch script confirmed that YTransformWrapper.wrap(Pipeline, m) has no `steps` for Log, Log1p, Sqrt and Yeo-Johnson.

**Verifier.** Between line 41549 (`pipe = YTransformWrapper.wrap(pipe, y_transform)`) and line 41570 (`final_model = pipe.steps[-1][1]`) nothing reassigns `pipe`, so every non-early-stopping regression refit with a Y-transform raises AttributeError. The thread's outer except catches it and reports an error, so this is a loud failure that makes the feature unusable, not silent corruption. I lowered severity to medium for that reason. A second bug on the same path: the combobox value 'Box-Cox' lowercases to 'box-cox', which wrap() does not accept (it expects 'boxcox'), so Box-Cox fails with ValueError even when no early stopping is involved.

### R049: The target property name is never saved, so Tab 8 consensus averages models of unrelated targets
`spectral_predict_gui_optimized.py:42418` · area gui-3 · persistence · finder medium → verifier **medium / CONFIRMED**

**Problem.** The metadata dict built in _save_refined_model has no target/property name (the only trace is a 15-character slug in the default filename). refined_config has none either, so _export_for_publication writes 'target' and the Tab 9 'Model Info' sheet always shows 'N/A' for Target Variable. Because nothing identifies what a model predicts, _add_consensus_predictions averages every numeric regression column it finds, weighted by R².

**Failure scenario.** The user loads a protein model (R² 0.92) and a moisture model (R² 0.90) together in Tab 8, a normal multi-analyte workflow. 'Consensus_Quality_Weighted' is emitted as a ~50/50 blend of protein % and moisture % and exported as a consensus prediction. Validation-source metrics likewise compare every model against whatever target is currently loaded.

**Verifier.** The save metadata dict (42418-42471) has no target key, and 'target_name' at 42754 falls back to the default 'target' because refined_config never sets it. _add_consensus_predictions skips only classification and one-class models. It weights all numeric regression columns by R² with no check that the targets match, so mixing a protein model and a moisture model yields a blended 'Consensus_Quality_Weighted'. The docs say nothing about consensus requiring a single target.

### R050: Saved .dasp data_type, x_unit and y_transform come from live widgets at save time, not the training snapshot
`spectral_predict_gui_optimized.py:42432` · area gui-3 · persistence · finder medium → verifier **medium / CONFIRMED**

**Problem.** The metadata takes 'data_type' from `self.current_data_type.get()`, 'x_unit' from `self.current_x_unit.get()` and 'y_transform' from `self.refine_y_transform.get()` at the moment Save is clicked. The filename's _abs/_ref suffix does the same. refined_config does not record any of them. Tab 8 and Tab 9 use metadata data_type to decide whether to warn and in which direction to offer conversion.

**Failure scenario.** The user trains in absorbance, converts the dataset to reflectance in the Import tab to inspect it, then saves the refined model. The file is stamped data_type='reflectance' and named *_ref.dasp. Later, Tab 8 flags correct absorbance prediction data as a mismatch and offers 'Convert to Reflectance'. Following that advice feeds 10^-A to an absorbance model and silently produces garbage predictions.

**Verifier.** Metadata takes data_type and x_unit from the live widgets (42432-42433) and y_transform from refine_y_transform (42470). refined_config (42176-42190) records none of them. current_data_type changes on the Import-tab conversion (20085) and on every new data load (20118), and neither clears refined_model. So training on absorbance, then converting or loading reflectance data before saving, stamps the model 'reflectance'.

### R051: Tab 8 Excel input loses sample IDs and is misdetected as reflectance
`spectral_predict_gui_optimized.py:44250` · area gui-3 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** CSV input goes through read_csv_spectra, which indexes by sample ID. Excel input uses bare `pd.read_excel`: the ID column stays a data column and the index is 0..N-1. detect_spectral_data_type then fails astype(float) on the string column and returns its fallback ('reflectance', 50%). _convert_prediction_data_type calls the conversion on `.values`, which include the string column, so it raises.

**Failure scenario.** An .xlsx of absorbance spectra with a 'SampleID' column is predicted with an absorbance model. Tab 8 shows 'Prediction data: REFLECTANCE' and a Mismatch, and every run pops 'Model trained on ABSORBANCE, but prediction data is REFLECTANCE'. The Convert button crashes. The exported predictions have Sample = 0,1,2... instead of the real IDs, so results cannot be matched back to samples.

**Verifier.** Tab 8 reads Excel with bare pd.read_excel (44250) and CSV with read_csv_spectra. With a SampleID column, the Excel path keeps a 0..N-1 index, and detect_spectral_data_type hits its non_numeric fallback of ('reflectance', 50.0). The CSV path gets real IDs. Prediction itself probably still works through wavelength-column selection.

### R052: Tab 8 attaches the training reference table (including training targets) to new samples with overlapping IDs
`spectral_predict_gui_optimized.py:44447` · area gui-3 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** For any prediction source (not only the validation set), every column of self.ref, the training reference file with the target property and metadata, is copied into the results whenever all prediction-sample IDs appear in the training ref index. Instrument auto-naming (e.g. ASD 'Spectrum00001' restarting each session) makes such overlaps common. The copied columns are exported next to the predictions.

**Failure scenario.** New ASD files spectrum00001-00020 are predicted, and the training set also contained spectrum00001-00020. The exported predictions CSV gains a '%Protein' column holding the *training* samples' lab values, which looks like reference data for the new samples. A user comparing against it gets fabricated agreement or disagreement.

**Verifier.** At lines 44442-44451, every self.ref column, including the target, is copied into the results whenever all prediction IDs exist in self.ref.index. Nothing checks pred_data_source, so any source is affected. I found nothing in the docs marking this as intended. The impact depends on ID collisions between new and training samples, which are plausible with instrument auto-naming.

### R053: The Tab 8 uncertainty table assumes every model has the first model's task type
`spectral_predict_gui_optimized.py:45177` · area gui-3 · error-handling · finder medium → verifier **medium / CONFIRMED**

**Problem.** _display_uncertainty chooses a single layout from the first model that has uncertainty data, then applies it to all models. If the first is regression and a later model is a classifier, `f"{predictions[i]:.4f}"` raises on a string label. That surfaces as a 'Prediction Error' dialog after the predictions have been computed. If the first is one-class, regression models' numeric predictions are rendered as 'Outlier' (`raw == 1` is False). If the first is a classifier, regression models are silently skipped.

**Failure scenario.** The user loads a protein PLS model and a 'grade' PLS-DA classifier in Tab 8 and runs predictions. The run ends with 'An error occurred during predictions: Unknown format code 'f' for object of type 'str'', and the uncertainty and consensus panels stay empty. In the reverse load order, the uncertainty table lists every protein prediction as 'Outlier'.

**Verifier.** _display_uncertainty chooses its branch from the first model's uncertainty keys (44947). The regression branch loops over every model and formats f"{predictions[i]:.4f}" (45177) with no type guard, so a classifier's string labels raise ValueError. That error reaches _run_predictions' except and shows 'Prediction Error'. In the one-class branch, numeric regression predictions go through the `raw == 1` check and are labelled 'Outlier'. The classification branch skips models without probabilities. All three behaviours match the finding. The result is a crash or a mislabelled display, not corrupted predictions.

### R054: Tab 8 validation run with a one-class or binary text-label classifier crashes in the confusion-matrix plot
`spectral_predict_gui_optimized.py:45749` · area gui-3 · error-handling · finder high → verifier **medium / CONFIRMED**

**Problem.** _plot_classification_validation_results calls precision/recall/f1 with `average='binary'` whenever there are two classes. With string labels ('Inlier (X)'/'Outlier', or 'Clean'/'Contaminated') sklearn raises `pos_label=1 is not a valid label`. Nothing catches this inside the plot function. It propagates through _display_predictions to the _run_predictions handler, which shows 'Prediction Error' and sets the status to 'Error occurred'. _display_consensus_info and _display_uncertainty never run. The stats panel has the same default-binary call (line 45340); there the error is caught, so the validation metrics are silently replaced by '[!] Could not calculate validation metrics'.

**Failure scenario.** The user saves a one-class SIMCA model, selects 'validation set' as the Tab 8 source, and runs predictions. The validation set contains both inliers and outliers (the normal case), so the run ends in an error dialog 'pos_label=1 is not a valid label. It should be one of ['Inlier (A)' 'Outlier']' (reproduced with sklearn in the project venv). No accuracy or F1 is ever shown for binary text classifiers.

**Verifier.** _plot_classification_validation_results uses average='binary' when there are two classes and has no try/except. The chain _display_predictions -> _plot_prediction_results -> this function raises into _run_predictions' outer except, which shows 'Prediction Error'. _display_consensus_info and _display_uncertainty are then skipped. One-class labels are the strings 'Inlier (X)'/'Outlier' (44557-44561). In the project venv, sklearn raises 'pos_label=1 is not a valid label' for such labels. The predictions table has already been filled before the plot, so this is a crash and lost metrics, not corrupted results. I downgraded it to medium.

### R055: Tab 9 CSV-directory loader attaches filenames to spectra in a different sort order
`spectral_predict_gui_optimized.py:48709` · area gui-3 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _load_spectra_from_directory_as_df loads the CSV spectra through _load_spectra_from_directory, which iterates `sorted(glob.glob(...))` (ordinal, case-sensitive). It then labels the rows with `sorted(dir_path.glob('*.csv'))`. On Windows, Path objects sort case-insensitively (casefold). When the filenames differ only by case ordering, the index labels are assigned to the wrong spectra.

**Failure scenario.** The comparison or live-monitoring folder holds apple.csv, Banana.csv, cherry.csv and Date.csv. The spectra are read in the order [Banana, Date, apple, cherry] but labelled [apple, Banana, cherry, Date] (verified on this machine). Every prediction, flag, reliability score and exported row in Tab 9 is attributed to the wrong sample.

**Verifier.** _load_spectra_from_directory_as_df labels rows with sorted(dir_path.glob('*.csv')). That sort is case-insensitive for WindowsPath, while the spectra were loaded with case-sensitive sorted(glob.glob). I ran the real method on a temp folder: every index label points to the wrong spectrum. This only triggers with mixed-case filenames.

### R056: CT Mode A predicts without checking the satellite data type against the prediction or transfer model
`spectral_predict_gui_optimized.py:50971` · area gui-3 · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _run_prediction_workflow applies the transfer and calls predict_with_model directly. It never compares ct_pred_data_type with the prediction model's metadata['data_type'] or the transfer model's data type. The Mode A type UI only shows the detected type with no 'model expects' check. Tab 8 (predict_with_uncertainty's data_type_warning) and Tab 9 (the mismatch prompt) both guard against this.

**Failure scenario.** Satellite spectra are exported as reflectance, while the transfer and PLS models were built on absorbance. Mode A applies the absorbance DS matrix to reflectance values and outputs confident, meaningless predictions with no warning.

**Verifier.** _run_prediction_workflow (50922-50975) never reads ct_pred_data_type or metadata['data_type']. _update_ct_pred_data_type_ui shows only the detected type and confidence, with no comparison to what the model expects. This is a missing guard, not an active corruption. It only matters when the user supplies data of the wrong type.

### R057: A failed transfer in the Tab 9 chain still produces 'Comparison complete', predicted on untransferred data
`spectral_predict_gui_optimized.py:53926` · area gui-3 · error-handling · finder medium → verifier **medium / CONFIRMED**

**Problem.** If any transfer model in the chain raises, _apply_transfer_chain shows an error box and returns the original X_data. _run_comparison then predicts on the raw satellite spectra and reports 'Comparison complete!' with nothing marking the results as untransferred. The chain also never resamples the input onto transfer_model.wavelengths_common: it multiplies whatever columns the data has, then relabels the output columns as wavelengths_common. A same-length but offset grid is therefore transferred and relabelled without any error.

**Failure scenario.** In live monitoring, a DS transfer is enabled but the satellite files have 2151 channels while the DS model was built on 2001. apply_ds raises and a dialog appears (on every scan). The comparison then runs on uncorrected satellite spectra, and the green status and exported Excel file look like normal transferred results.

**Verifier.** _apply_transfer_chain catches any exception, shows an error box and returns the original X_data (53923-53926). _run_comparison (53995) uses the returned frame without checking it and completes normally. In the whole Tab 9 region, the only reference to wavelengths_common is the column relabel at 53933, so the input is never resampled onto the transfer grid. A same-length offset grid is transferred and relabelled silently. I confirmed this by reading the code, not by a GUI repro.

### R058: Code export mis-maps or omits preprocessing (sg4, snv_sg3/4, sg3 polyorder, baseline, smoothing, y-transform)
`src/spectral_predict/code_generator.py:836` · area io-persistence · reproducibility · finder high → verifier **medium / CONFIRMED**

**Problem.** The non-embedded Python, notebook and bundle exports rebuild preprocessing from refined_config['preprocessing'], which holds Tab 7 combobox names: raw, snv, sg1-sg4, snv_sg1-snv_sg4, deriv_snv. Only sg1-3, snv_sg1-2, snv_deriv and deriv_snv have explicit branches. Every other name falls through to get_preprocessing_template, which recognises only tokens starting with 'deriv'. The consequences: sg4 becomes a raw copy, and snv_sg3/snv_sg4 become SNV only with no derivative. The template also hard-codes polyorder 2 for deriv1 and 3 for everything else, so sg3 exports polyorder 3 where the GUI uses 4 (get_polyorder_from_deriv). The 'polyorder' key passed by the GUI is ignored. Baseline correction and SG smoothing, which Path A applies (GUI:41293-41304), are not in model_config at all. y_transform is not exported; an earlier review deferred this. The exported script reports different CV metrics and produces a different model.

**Failure scenario.** Export a 'Python Script - Basic' for an snv_sg4 PLS model: the script applies `X_processed = apply_snv(X)` only. For sg4 it applies `X_processed = X.copy()`. For sg3 it uses savgol polyorder=3 instead of 4. For a model with ALS baseline, or Y-transform=log, the export silently drops that step (confirmed by codegen_check.py).

**Verifier.** The Tab 7 combobox (GUI:15325) offers sg4, snv_sg3 and snv_sg4. code_generator.py:807-839 has no branches for those names, and the fallback get_preprocessing_template only recognises tokens starting with 'deriv'. So sg4 exports as X.copy() and snv_sg3/snv_sg4 as SNV only. The emitted apply_savgol_derivative def hard-codes polyorder=3, so sg3 runs with polyorder 3 where the GUI uses 4 (get_polyorder_from_deriv, GUI:40011-40020). The model_config built at GUI:42751-42815 has no baseline, smoothing or y_transform keys. I lowered severity to medium: this affects exported scripts only, not the saved or in-app model, and the affected options (sg4, snv_sg3/4, baseline, smoothing) are less common.

### R059: Merging data sources never detects cross-source duplicate IDs; 'error', 'keep_first' and 'keep_last' all keep both rows
`src/spectral_predict/data_management.py:358` · area io-persistence · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** In all three merge strategies the duplicate tracker appends `X_subset.index[0]` (the source's FIRST ID) on every iteration instead of `idx`. sample_ids therefore holds only each source's first ID, and a duplicate of any other sample is never seen. Even when a duplicate is detected, 'keep_first' only `continue`s past the append and never removes the row, and 'keep_last' has no code at all. The merged X/y then carries duplicate index labels: the same specimen can land in train and test folds, and label-based .loc lookups return multiple rows.

**Failure scenario.** Source A = [S1,S2,S3], source B = [S4,S2], merged with handle_duplicates='error' (the default). No error is raised and the merged index is [S1,S2,S3,S4,S2]. 'keep_first' and 'keep_last' give the identical result (merge_check.py).

**Verifier.** data_management.py:358 appends X_subset.index[0] on every iteration instead of idx, so only each source's first ID is tracked and duplicates of any other sample are never detected. 'keep_first' only skips the append, and 'keep_last' has no code at all. The GUI merge (GUI:18463) passes the user's duplicate_handling_var straight through. The merged frame keeps duplicate index labels, which can put the same specimen in train and test folds. I kept medium because whether cross-source duplicate IDs are common depends on the data.

### R060: CSV-folder and JCAMP-folder readers round the x-axis to integers, silently dropping and mislabeling sub-integer-spaced data
`src/spectral_predict/io.py:288` · area io-persistence · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** read_csv_dir and read_jcamp_dir round every spectrum's x-axis to int to align grids, then drop duplicates with keep='first'. For data sampled finer than 1 unit (e.g. 0.5 nm NIR, or FTIR at ~0.48 or 0.96 cm-1) half or more of the points are discarded. numpy's round-half-to-even also relabels values: the value stored at 1002 nm actually belongs to 1001.5 nm. The single-file read_csv_spectra does not round, so the same data read as a folder and as a file gives different axes. A model trained on one form cannot predict on the other, because the 0.01 match tolerance fails. The rounding itself was deliberate (commit 1d43006, to fix NaN grids); the data loss and relabeling are not addressed.

**Failure scenario.** Folder of long-format CSVs at 0.5 nm spacing (300 points from 1000-1149.5 nm): read_csv_dir returns 151 columns, and column 1002 holds the 1001.5 nm reading (1.0015 instead of 1.002) (ascii_check.py).

**Verifier.** io.py:288-289/296 (read_csv_dir) and 1736-1737 (read_jcamp_dir) round the x-axis to int and drop duplicates with keep='first'. Commit 1d43006 made the rounding deliberate, to fix NaN grids; no doc records a decision to accept losing sub-integer data. At 0.5 nm spacing half the points are lost, and the survivors are relabeled (1001.5 is stored under 1002).

### R061: Combined CSV/Excel import drops spectral columns above 10000 and can auto-pick an absorbance column as the target
`src/spectral_predict/io.py:1232` · area io-persistence · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** identify_wavelength_columns accepts only headers with 100 <= value <= 10000. FT-NIR data in wavenumbers commonly spans 12500-4000 cm-1, so every column above 10000 is treated as non-spectral: it goes into metadata_cols and becomes a y candidate. When the target name contains none of the priority keywords (e.g. 'N', 'C', 'Moisture'), auto_detect_y_column's 'first numeric column' fallback picks the 12500 cm-1 absorbance column as y. The GUI calls read_combined_csv/read_combined_excel with auto-detection (GUI:16565, 18082, 50596).

**Failure scenario.** Combined CSV: sample_id, 1063 wavenumber columns 12500-4004 cm-1, target 'N'. read_combined_csv keeps 750 spectral columns (max 9996), moves 313 spectral columns into metadata, and reports y_col='12500.0' (combined_check.py).

**Verifier.** identify_wavelength_columns (io.py:1231-1233) accepts only headers with 100 <= value <= 10000. FT-NIR wavenumber columns above 10000 therefore become metadata. For a target like 'N', none of the priority keywords matches, so auto_detect_y_column falls back to the first numeric column, which is the 12500 cm-1 absorbance.

### R062: read_ascii_spectra is redefined later in io.py: ASCII folder import is broken and the first data row is eaten as a header
`src/spectral_predict/io.py:3354` · area io-persistence · correctness · finder high → verifier **medium / CONFIRMED**

**Problem.** io.py defines read_ascii_spectra twice. The later definition (line 3354) silently replaces the directory-aware one at line 1888. The replacement calls pd.read_csv on the path with the default header=0 and never handles directories. The GUI calls read_ascii_spectra with the spectral-data FOLDER when .dpt/.dat/.asc files are detected (GUI:16450, 19159, 44235, 50550, 51215), and every delimiter attempt fails and is swallowed by `except Exception: continue`. For headerless two-column files such as Bruker .dpt, the first (x, y) pair becomes the column header and is lost.

**Failure scenario.** Folder of three headerless .dpt files: read_ascii_spectra(folder) raises 'Could not parse ASCII file: <folder>'. A single 200-point headerless file yields 199 points; the 4000.0 cm-1 point is dropped (ascii_check.py).

**Verifier.** io.py defines read_ascii_spectra twice (lines 1888 and 3354), and inspect shows the line-3354 version is the effective one. It calls pd.read_csv(path) with the default header and has no directory handling. The GUI passes the folder path at GUI:19159 (Import tab when detected_type=='ascii'), GUI:44235 and GUI:48694, so ASCII folder import always fails with 'Could not parse ASCII file'. For a headerless single file, the first data row becomes the header. I lowered severity to medium: folder import fails loudly rather than corrupting data, and losing one edge point from a single file is minor.

### R063: Loading a .dasp or ensemble unpickles arbitrary objects with no trust boundary or warning
`src/spectral_predict/model_io.py:433` · area io-persistence · security · finder medium → verifier **medium / CONFIRMED**

**Problem.** load_model runs joblib.load on up to 7 pickles from the archive (model, preprocessor, label_encoder, scaler, pca_reducer, pca_model). load_ensemble also unpickles ensemble_state.pkl and loads base-model paths taken verbatim from ensemble_config.json (`tmpdir_path / base_file`, not validated to stay inside the temp dir). .dasp files are designed to be shared and loaded in the Prediction tab, so opening a received model file gives code execution. An earlier audit noted this (SESSION_LOG_ARCHIVE:1739), but no warning, signature or restricted unpickler was added.

**Failure scenario.** A colleague emails model.dasp whose model.pkl carries a __reduce__ payload. Loading it in the Prediction tab (GUI:43953 load_model) executes the payload with the user's privileges before any metadata validation.

**Verifier.** model_io.py runs joblib.load on archive members (model.pkl, preprocessor, label_encoder, scaler, PCA and others). load_ensemble also joins config-supplied base_file names onto tmpdir without validating them, and loads ensemble_state.pkl. This is inherent to pickle, and SESSION_LOG_ARCHIVE:1739/1567 already records it as a known risk with no mitigation (no warning, signature or restricted unpickler). It is real, but it is the standard risk of the sklearn/joblib model format, and exploiting it requires the user to open an attacker-supplied file. Medium is appropriate.

### R064: Stale nonlinear bias correction from a previous Model Development run can be saved with a different model
`spectral_predict_gui_optimized.py:42603` · area models-ensemble · persistence · finder medium → verifier **medium / CONFIRMED**

**Problem.** self.nonlinear_correction_data is only set when the user clicks compute (37671) and is never reset when a new Model Development run finishes. _update_bias_correction_ui recomputes only the linear correction. When 'apply correction' and 'use nonlinear' remain ticked, saving a newly refined model stores the polynomial fitted to the previous model's CV predictions. apply_correction then applies it to every future prediction.

**Failure scenario.** Refine PLS with 8 LVs, compute Polynomial(3) correction, tick apply and use-nonlinear. Re-run Model Development with 3 LVs or different preprocessing, then save. The .dasp file carries the cubic fitted to the 8-LV model's predictions, and every prediction from the 3-LV model is distorted by it.

**Verifier.** grep finds only three writes and reads of nonlinear_correction_data: the init (3037), the compute button (37671) and the save (42603). _update_bias_correction_ui recomputes only the linear bias_correction_data, and apply_bias_correction/use_nonlinear_correction are never reset between runs. A stale polynomial is therefore saved with a new model whenever the boxes stay ticked. I confirmed this by reading the deterministic code path; I did not click through it in the GUI.

### R065: ClassificationResampler loses k_neighbors (and every other method param) when cloned, so each CV fold resamples with defaults or skips resampling
`src/spectral_predict/imbalance.py:229` · area models-ensemble · hyperparameter-mapping · finder high → verifier **medium / CONFIRMED**

**Problem.** ClassificationResampler.__init__(self, method, random_state, **params) keeps the extra params in self.params. sklearn's get_params ignores VAR_KEYWORD arguments, so clone() rebuilds the object with params={}. Every CV path clones the pipeline per fold (search._run_single_fold 4377, the Model Development loop 41581, cross_val_predict), so the GUI's k_neighbors never reaches SMOTE, ADASYN, BorderlineSMOTE, SMOTETomek or SMOTEENN. The minimum-size guard also reads self.params.get('k_neighbors', 5), so after cloning it uses 5. A user who lowered k to handle a small minority class gets resampling silently skipped in every fold ('Skipping resampling for this fold'). The reported CV metrics are then for an unresampled model, while the final full-data fit, if it is not cloned, does resample.

**Failure scenario.** 40 samples with 35/5 classes, SMOTE with k_neighbors=3 chosen in the GUI. The original object resamples to [35, 35]. clone(resampler).fit_resample gives [35, 5] with the warning 'Some classes have <=5 samples ... Skipping resampling'.

**Verifier.** __init__ keeps the **params extras in self.params, but sklearn get_params only reports method and random_state, so clone() drops k_neighbors. The GUI passes k_neighbors through build_imbalance_transformer (GUI 23576/23579 -> imbalance.py 970), and search clones per fold (search.py 4377). After cloning, both SMOTE and the size guard use k=5. It only matters when the user picks k != 5, and it silently changes the resampling that the CV metrics describe, so I rated it medium rather than high.

### R066: RegressionSampleWeighter 'binning' gives the maximum-y sample its own bin and a huge weight
`src/spectral_predict/imbalance.py:837` · area models-ensemble · numerical · finder medium → verifier **medium / CONFIRMED**

**Problem.** np.digitize(y, bins=linspace(min, max, n_bins+1)) uses all n_bins+1 edges, so y == y.max() lands in bin n_bins+1 on its own, along with any ties at the max. Its weight is N/(n_bins*1), tens of times the typical weight, so a single extreme sample dominates the weighted fit for every model that accepts sample_weight (applied in search._run_single_fold 4394-4410). RegressionUndersampler and RegressionResampler use edges[:-1] correctly.

**Failure scenario.** 200 normally distributed y values with n_bins=5. The max-y sample gets weight 33.3 against a median weight of 0.56 (about 59x) and carries 16.7% of the total training weight in every fold. Ridge/XGBoost fits are pulled toward that one point.

**Verifier.** np.digitize is given all n_bins+1 edges, so y == max lands in its own bin n_bins+1 with count 1 and weight N/n_bins. build_imbalance_transformer('binning', regression) returns RegressionSampleWeighter, and search._run_single_fold feeds sample_weight_ into the model fit (4390-4410).

### R067: SVR and SVM grids ignore the user's kernel, C and gamma selections
`src/spectral_predict/models.py:1340` · area models-ensemble · hyperparameter-mapping · finder high → verifier **medium / CONFIRMED**

**Problem.** get_model_grids takes svr_kernels, svr_C_list and svr_gamma_list, and the GUI builds them from checkboxes and custom entry fields (GUI 29051-29091, passed at 30907-30909). Inside the 'SVR' block, svr_kernels is overwritten from the tier config, and the loops iterate svr_Cs/svr_gammas, also taken from the tier config. The user's lists are never used. The classification SVM block does the same (1768-1771). epsilon, shrinking, degree and coef0 are honoured, so the dropping is partial and easy to miss. Results rows still report C=1/10 correctly, so the user just sees their custom values 'never win'.

**Failure scenario.** The user unticks C=1 and C=10, enters custom C=100 and gamma='auto', and runs Comprehensive. get_model_grids('regression', ..., svr_kernels=['rbf'], svr_C_list=[100.0], svr_gamma_list=['auto', 0.01]) returns rbf and linear with C in {1, 10} and gamma 'scale' only. The requested models are never evaluated.

**Verifier.** get_model_grids resolves svr_kernels/svr_C_list/svr_gamma_list at 1021-1029. The SVR block at 1340-1343 then overwrites svr_kernels and loops over svr_Cs/svr_gammas from the tier config. The SVM block (1768-1771) does the same. The GUI passes the user lists (30907-30909). I lowered severity to medium: user settings are silently ignored, but result rows label the actual params correctly, so no scientific result is corrupted.

### R068: NeuralBoostedClassifier trains from the class-prior log-odds but predicts from 0, so probabilities and labels are biased
`src/spectral_predict/neural_boosted.py:879` · area models-ensemble · correctness · finder high → verifier **medium / CONFIRMED**

**Problem.** _fit_binary starts boosting at F_init = log(p/(1-p)) of the training prevalence, so the weak learners model the residual relative to the prior. predict_proba then starts every sample at F = 0 (p = 0.5) for binary, and at 0 for every one-vs-rest class, which discards the learned intercept. On imbalanced data this shifts every log-odds by -F_init, and predictions collapse toward 50/50. In multiclass the shift differs per class and favours minority classes. The CV metrics from search (which call predict) and the saved models are both affected.

**Failure scenario.** Binary data with 15% positives and uninformative features. The fitted model's mean P(class 1) on test data is 0.504 and it predicts class 1 for 55% of samples. Test accuracy is 0.462 versus 0.85 for the majority baseline, with or without early stopping.

**Verifier.** _fit_binary starts boosting at F_init = logit(prevalence), and the weak learners fit residuals relative to that. predict_proba starts at logit(0.5)=0 and multiclass starts at 0.0. The learned intercept is not stored and is dropped. The classifier is reachable through get_model/get_model_grids (models.py 490, 719, 1742). I downgraded severity: CV and deploy use the same biased predict, so the reported metrics honestly show a poor model rather than a leaked or mismatched one.

### R069: The 'Box-Cox' Y-transform offered in the GUI is rejected by YTransformWrapper.wrap and skipped by validate
`src/spectral_predict/y_transform.py:66` · area models-ensemble · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** The GUI combobox offers 'Box-Cox' (GUI 15403). wrap() lowercases it to 'box-cox' but only matches 'boxcox', so it raises ValueError 'Unknown Y-transform method'. validate() also only checks 'boxcox', so the y>0 requirement is never checked. _get_transformer (early-stopping path) accepts 'box-cox', so with boosters the transform gets past validate and then fails inside PowerTransformer for y<=0. The option cannot work in the normal path.

**Failure scenario.** Model Development, regression, Y-transform = Box-Cox, PLS model. Line 41549 raises ValueError: Unknown Y-transform method: 'Box-Cox' and the refinement aborts. YTransformWrapper.validate([-1.0, 2.0], 'Box-Cox') returns None (passes).

**Verifier.** The GUI combobox offers 'Box-Cox', and GUI 41517 passes the raw string. wrap() lowercases it to 'box-cox' but only matches 'boxcox', so it raises ValueError. validate() also checks only 'boxcox', so negative y is never flagged. _get_transformer accepts both spellings, so the early-stopping path gets past validate.

### R070: GA-PLS classification fitness thresholds predictions at their median, forcing a 50/50 split
`src/spectral_predict/ga_pls.py:216` · area nsga-ga · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _fitness_function classifies PLS-regression predictions by comparing them with the median prediction. That forces half the samples into class 1 whatever the class balance, and it only ever predicts {0,1}, so any class >= 2 is always wrong. GA-PLS varsel for classification (search.py 3758/3796, with y label-encoded) therefore optimises a distorted objective and its selection frequencies do not reflect discriminative wavelengths. ga_preprocessing._evaluate_pls:664 has the same defect; there it is used as the exhaustive-preprocessing proxy and as a fallback.

**Failure scenario.** A perfectly separable 80/20 binary set scores accuracy 0.70 for every chromosome that separates perfectly. A perfectly separable 3-class set (34/33/33) scores 0.51. Both were verified with _fitness_function and evaluate_fitness(fitness_model='pls'). The GA cannot tell perfect wavelength subsets apart, and in the multiclass case it rewards subsets that merge classes 1 and 2.

**Verifier.** ga_pls._fitness_function (L209-217) thresholds PLS predictions at their median and can only predict {0,1}. I checked that 'ga' is an implemented varsel method (search.py L1693) and that it calls ga_pls_selection with task_type=classification for linear models (L3758). ga_preprocessing._evaluate_pls:664 has the same code. Perfectly separable subsets get an accuracy capped well below 1.

### R071: SmartMutation almost never mutates preprocessing, model or hyperparameter genes, and uses the wrong model list for active genes
`src/spectral_predict/nsga2_search.py:199` · area nsga-ga · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** SmartMutation sets self.prob=0.1 after super().__init__(), which overwrites pymoo's per-individual mutation probability, so only about 10% of offspring are mutated at all. Within those, PM._do uses prob_var=1/n_var, which is about 0.001 per gene for 1000 wavelengths. The documented 'always mutate' control genes 0-3 and the hyperparameter genes are therefore effectively frozen, and exploration of model, preprocessing and hyperparameters relies only on SBX between existing parents. In addition, active hyperparameter genes are looked up in the global MODEL_TYPES by index instead of problem.model_types. For a user subset such as ['PLS','LightGBM'], LightGBM (index 1) is treated as 'Ridge' (no active genes), so its lr, reg, subsample and colsample genes are always restored.

**Failure scenario.** SmartMutation.do on 2000 offspring with models ['PLS','LightGBM'] and 1000 wavelengths: 0 of 2000 offspring had any of genes 0-12 changed. Only 9.5% had any wavelength change. A model type or window absent from the first generation can never appear, and LightGBM hyperparameters stay at their seeded or crossover values.

**Verifier.** pymoo 0.6.2 Mutation.do applies Xp only where random() <= self.prob. SmartMutation sets self.prob=0.1 after super().__init__(), and PM._do uses prob_var=min(0.5, 1/n_var). The active-gene lookup uses the global MODEL_TYPES[model_idx], so index 1 in ['PLS','LightGBM'] resolves to 'Ridge' and all hyperparameter genes are restored. The unguided path's PM(prob=0.1) has the same low per-individual rate.

### R072: Guided NSGA-II (the GUI default) is not reproducible despite random_state=42
`src/spectral_predict/nsga2_search.py:396` · area nsga-ga · reproducibility · finder high → verifier **medium / CONFIRMED**

**Problem.** SeededWavelengthSampling draws the initial population from global np.random.randint and from an unseeded np.random.default_rng(). SmartMutation also uses an unseeded default_rng() for every wavelength drop/add. pymoo 0.6.2 seeds only its own algorithm.random_state Generator; it does not seed the global state or these private generators. The GUI's default mode is 'NSGA-II w/Guidance' with a hard-coded 'Fixed seed' of 42, yet two identical runs return different Pareto fronts, knee or min-error solutions, wavelength sets and reported errors.

**Failure scenario.** Running run_nsga2_search twice on the same X/y with random_state=42 and use_guidance=True gives different results. In the scratch run, run 1 selected 16 wavelengths with error 1.7362, run 2 selected 14 with error 1.6004, and the two selections were not identical. With use_guidance=False, both runs were identical.

**Verifier.** pymoo 0.6.2 Algorithm.setup only does self.random_state = np.random.default_rng(self.seed). It does not seed the global RNG. SeededWavelengthSampling calls global np.random.randint and an unseeded default_rng(), and SmartMutation also uses an unseeded default_rng(). The GUI defaults to 'NSGA-II w/Guidance' with random_state=42. Two identical guided runs gave different results, while unguided runs were identical. I rated it medium rather than high: results are not corrupted, but the fixed seed does not make runs reproducible.

### R073: decode_solution sizes PLS n_components from the wavelength count before edge masking and from the wrong N
`src/spectral_predict/nsga2_search.py:2334` · area nsga-ga · correctness · finder high → verifier **medium / CONFIRMED**

**Problem.** decode_solution computes n_selected_features from the raw chromosome mask before the derivative edge-zone masking, which only happens at line 2457. It also caps by n_samples-1, and convert_nsga2_to_v1_format passes total_samples_original, which includes excluded and validation samples. The fitness function instead used min(model_param+1, 15, n_post_mask-1, min_train_fold-1). So the Params n_components can exceed the number of fitted features and differ from the model that was scored. The same row's LVs column uses the post-mask count, so the row contradicts itself.

**Failure scenario.** A deriv2 window-51 chromosome selects 52 wavelengths, 40 of them inside the 25-point edge zones, with model_param=14. Fitness fits PLS with 11 LVs on 12 wavelengths. Params store n_components=15 (verified: 'post-mask n_wl: 12 stored params: {... n_components: 15 ...}') while LVs shows 11. Refitting from Params either raises (15 components > 12 features) or uses a different LV count from the one evaluated.

**Verifier.** This is real. n_selected_features is counted at L2334, before edge masking at L2457. The GUI passes total_samples_original as n_samples, which is larger than the fitted N. The fitness function caps at post-mask features-1 and fold-train-1. The decoded n_components is therefore never smaller than the fitted value, and whenever the two differ it is above the GUI refit guard's limit (GUI L41479-41491). In practice the refit shows an 'Invalid PLS Configuration' error, and compute_validation_metrics_for_top_models skips the row when n_components > n_features (search.py L1158). So the result is an error or a skipped row, not a silently different model, and I lowered the severity to medium. It does hit parsimonious derivative solutions near min_wavelengths, which are exactly the ones the Pareto front favours.

### R074: NSGA-II top_vars is always the first 30 selected wavelengths in index order, not importance-ranked
`src/spectral_predict/nsga2_search.py:3430` · area nsga-ga · error-handling · finder medium → verifier **medium / CONFIRMED**

**Problem.** _compute_top_variables calls get_feature_importances(model, model_type). That function takes four arguments (model, model_name, X, y), so every call raises TypeError. The bare except silently falls back to selected_indices[:30]. Every NSGA-II row's top_vars column, which is documented as 'ordered by importance (most important first)', is really just the lowest-index selected wavelengths.

**Failure scenario.** Any NSGA-II run: for a solution selecting indices 0,2,4,..., top_vars is '0,2,4,6,8' for top_n=5, identical to the first selected indices. Calling get_feature_importances with two arguments directly raises 'missing 2 required positional arguments: X and y'. Users reading top_vars as the most informative bands are misled.

**Verifier.** get_feature_importances requires (model, model_name, X, y), and L3430 calls it with two arguments. The broad except then falls back to selected_indices[:30]. There is a second failure path as well: decoded has no 'model_param' key, so _build_model gets {} and PLS would already fail at {}+1. top_vars is therefore always index-ordered.

### R075: Interference tab: Wavelength Exclusion, OSC and DOSC always crash
`spectral_predict_gui_optimized.py:60680` · area one-class · error-handling · finder medium → verifier **medium / CONFIRMED**

**Problem.** _app_apply_correction reads excluder.wavelengths_kept_, which does not exist; WavelengthExcluder sets wavelengths_out_. It also calls OSC(...).fit_transform(X) and DOSC(...).fit_transform(X) without y, but both fit() methods require y. Three of the six methods in the tab can never succeed.

**Failure scenario.** Choose 'Wavelength Exclusion' with range '10-20': AttributeError: 'WavelengthExcluder' object has no attribute 'wavelengths_kept_'. Choose 'OSC' or 'DOSC': TypeError: OSC.fit() missing 1 required positional argument: 'y'. The user gets only an error dialog.

**Verifier.** The Interference tab combobox offers all six methods (GUI 56025-56032). _app_apply_correction reads excluder.wavelengths_kept_, but WavelengthExcluder only sets wavelengths_out_. It calls OSC(...).fit_transform(X) and DOSC(...).fit_transform(X) without y, and both fit(X, y) signatures require y. Neither class overrides fit_transform. The exceptions are caught and shown only as an error dialog.

### R076: PCASIMCA allows n_components = n_train − 1, which leaves no residual space and rejects every new sample
`src/spectral_predict/contamination.py:134` · area one-class · numerical · finder medium → verifier **medium / CONFIRMED**

**Problem.** The clamp max_components = min(n_samples - 1, n_features) is justified in the code as 'PCA needs at least one residual dimension'. But mean-centred data of n rows has rank n−1, so n−1 components reconstruct the training set exactly. Training Q is then 0 and the zero-guard sets q_scale=1e-10, so any new sample gets p_Q≈1e-300 and is rejected. The fit floor was relaxed to 3 rows so small CV folds could fit SIMCA. Those folds now clamp grid values (5, 7) to n−1 and reject all held-out inliers, while resubstitution (cal) specificity looks perfect. This is a deterministic failure, separate from the documented MoM small-n over-rejection.

**Failure scenario.** PCASIMCA(n_components=5) fitted on 6 clean rows: q_threshold_method_='zero_guard', false rejection of fresh inliers 100%. The same happens at 4 rows. run_one_class_cv on 8 inliers + 5 outliers with 5 folds: specificity 0.0 for n_components=5 and 7. The cal model accepts its own training rows.

**Verifier.** The max_components=min(n-1, p) clamp lets n-1 components reproduce mean-centred training data exactly. Q_train is then 0, the zero_guard sets q_scale=1e-10, and every new sample is rejected. Caveat: at these n the documented MoM small-n over-rejection is already severe (nc=2 on 4-6 rows rejects 89-95% of fresh inliers), so the extra harm is 'certain' versus 'nearly certain'. The zero-guard is still a distinct, deterministic failure and fold results disagree with the cal model, so medium stands.

### R077: Autoscale is fitted on inliers only in grid search but on inliers plus contaminants in validation, Model Development refit and Bayesian
`src/spectral_predict/contamination.py:1218` · area one-class · reproducibility · finder medium → verifier **medium / CONFIRMED**

**Problem.** run_one_class_search fits the preprocessing pipeline, including the StandardScaler autoscale step, on inlier rows only (search.py:6344). Three other paths fit it on all training rows, contaminants included: the one-class validation rebuild (pipe.fit_transform(X_train)), the Model Development one-class refit (GUI 40772, prep_pipeline_oc.fit_transform(X_full)), and the Bayesian one-class objective (unified_bayesian.py:776). For a grid row with Autoscale=True, the val_* columns and the refit model are built in a different feature space from the model that was ranked. Contaminant variance also changes the column scaling that the one-class model sees.

**Failure scenario.** Grid one-class search with autoscale=True, where contaminants add strong variance in a band. The CV row uses per-column std from inliers only. Validation and Model Development divide that band by a much larger std, so SIMCA/OCSVM distances change. val_* and the refit metrics disagree with BalancedAcccv for reasons unrelated to generalisation.

**Verifier.** Confirmed by code reading, no repro. search.py:6344 fits the prep pipeline on X_np[inlier_indices]. The validation rebuild (contamination.py:1218) fits pipe.fit_transform(X_train) on all training rows, contaminants included. The Model Development refit (GUI ~40772) fits on X_full, and the Bayesian objective (unified_bayesian.py:776) runs StandardScaler().fit_transform(X) on all rows. The other steps are per-spectrum and stateless, so only the autoscale step differs. When Autoscale=True, the column std differs whenever contaminants add variance. No doc entry treats the inlier-only or all-rows split as intended. The rows are self-consistent within each path, so this is a reproducibility mismatch rather than leakage. Medium is right.

### R078: One-class validation silently returns NaN for every row when wavelengths have more than 6 significant digits
`src/spectral_predict/contamination.py:1244` · area one-class · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** The search paths write all_vars with f'{w:g}', which rounds to 6 significant digits (search.py:6494/7104; unified_bayesian.py:1549-1554). compute_validation_metrics_for_top_one_class_models then looks up float(w) exactly in a map built from the full-precision wavelengths. It skips any row with a miss, and only logs a warning. Wavelengths with more than 6 significant digits are common: the GUI nm↔cm-1 conversion gives values like 1e7/1350 = 7407.407…, and SPC wavenumbers rounded to 2 dp above 10000 (e.g. 10000.25 → '10000.2'). The val_* columns are then all NaN with no user-facing message.

**Failure scenario.** Reproduced: wavelengths = 1e7/np.arange(1000,1100). run_one_class_search, then compute_validation_metrics_for_top_one_class_models, logs '99/100 model wavelengths not found in wavelength map, skipping' for every row. val_BalancedAcc and val_Sensitivity are NaN for all models.

**Verifier.** The grid path (search.py 6494 and 7104) and the Bayesian path (unified_bayesian.py 1549-1554) serialise all_vars with :g, which keeps 6 significant digits. compute_validation_metrics_for_top_one_class_models builds its map from full-precision floats, does exact lookups, and skips any row with a miss after only a logger warning. The GUI (30060-30064) passes full-precision float(col) wavelengths. The GUI nm-to-cm-1 conversion (convert_x_axis, 1e7/x) does not round the column headers.

### R079: Outlier report counts Hotelling T² twice: its 'Mahalanobis' flag is sqrt(T²) on the same PCA scores
`src/spectral_predict/outlier_detection.py:558` · area one-class · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** compute_mahalanobis_distance runs on the PCA scores returned by run_pca_outlier_detection. Those scores are mean-zero with covariance diag(eigenvalues), which is the same covariance T² uses, so the distance is exactly sqrt(T²). The report treats T² and Mahalanobis as independent methods and marks samples flagged by 2+ as combined/'moderate confidence' and 3+ as 'high confidence'. One extreme T² sample therefore gets two votes. Separately, the Q-residual threshold is the 95th percentile of the data itself, so about 5% of samples are always flagged, even in clean data.

**Failure scenario.** Numerical check: max|maha − sqrt(T2)| = 4.4e-16. A sample with a large score-space distance but normal Q and y is flagged by both T2 (F-limit) and Maha (median+3·MAD), so Total_Flags=2 and it counts as a combined outlier on one statistic. On clean data Q always flags ceil(0.05·n) samples; 5 of 100 in the test.

**Verifier.** compute_mahalanobis_distance runs on the PCA scores with np.cov, which is the same covariance used for T2, so the distance is exactly sqrt(T2). The two flags differ only in threshold (F-limit versus median+3MAD), so one statistic can supply 2 of the Total_Flags votes. The Q threshold is np.percentile(q, 95), so about 5% of samples are always flagged. This only affects a diagnostic report, but it inflates the 'combined/moderate confidence' classes that users may act on.

### R080: Multi-class 'elliptic-envelope' engine gives badly miscalibrated p-values when features are close to or above n (no p>n guard)
`src/spectral_predict/simca.py:480` · area one-class · numerical · finder medium → verifier **medium / CONFIRMED**

**Problem.** contamination.run_one_class_cv reduces EllipticEnvelope input with PCA when n_features > n_samples. MultiClassClassModel does not. Its cross-fit null is scored by fold models trained on about 0.8n rows, while test samples are scored by a final model trained on n rows. When p is close to or above n the MCD covariance is near-singular, so the two sets of Mahalanobis scores are on very different scales and the empirical p-value is not level-alpha. The code comment assumes EE fails every fold when p > n and is then marked unmodelable. In practice it usually fits without error.

**Failure scenario.** Two classes, alpha=0.05, fresh class-'a' samples (1000). False rejection: n=30,p=30 → 51%; n=40,p=100 → 12%; n=40,p=300 → 0.3%. Null sizes are complete, so nothing is flagged unmodelable. Decision matrix, 'novel' labels and evaluate_novelty rates are all wrong.

**Verifier.** MultiClassClassModel fits the final non-SIMCA engine on all n class rows and fits the cross-fit null on fold models trained on about 0.8n rows. The EllipticEnvelope branch has no PCA reduction. EE fits without error near and above p=n, so the null is complete and the class is not marked unmodelable, contrary to the comment at 1025. The empirical p-value is badly miscalibrated. It is well calibrated when p is much smaller than n.

### R081: BaselineAdvanced keeps algorithm params in **kwargs, so sklearn.clone() silently resets them to the registry defaults
`src/spectral_predict/baseline_advanced.py:391` · area preproc-varsel · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** __init__ stores extra params (lam, eta, half_window, poly_order, …) with setattr from **params. get_params() only reports the named signature arguments (method, wavenumbers). clone() therefore builds an instance without lam, and transform then falls back to info['default_params'] (lam=1e5). Any code path that clones a pipeline containing this step silently changes the baseline. The ensemble OOF refit does this: _clone_for_refit → clone(model) on GUI-rebuilt Pipelines that contain shared_prep from build_preprocessing_pipeline.

**Failure scenario.** An ensemble built from rows that used Advanced/arPLS with lam=1e2 or 1e3. Every per-fold refit uses lam=1e5, so OOF predictions and the resulting ensemble weights come from different preprocessing than the base models. Scratch check: clone(BaselineAdvanced('arpls', lam=1e2)) has no lam attribute, and its output differs by up to 6.6 units.

**Verifier.** get_params() exposes only method and wavenumbers, so clone() drops lam and transform falls back to the registry defaults. The GUI ensemble builds pipelines with build_preprocessing_pipeline, including baseline from the row (L24989), and EnsembleModel.fit calls _clone_for_refit -> clone(model) when refit_base_models=True. Tab 7 Path B does not pass baseline at all, so the fold and final clones there are not affected by this particular bug.

### R082: Wavelength exclusion inside the preprocessing pipeline drops columns, but search keeps the full wavelength axis, so top_vars and all_vars name the wrong wavelengths
`src/spectral_predict/preprocess.py:367` · area preproc-varsel · index-misalignment · finder medium → verifier **medium / CONFIRMED**

**Problem.** build_preprocessing_pipeline adds WavelengthExcluder as a pipeline step, and that step removes columns. search.py takes wavelengths = X.columns (full axis) before preprocessing. It then uses that array unchanged as wavelengths_for_models, for edge-trimming and for mapping importance indices to wavelengths. After the excluded band, every index points to a wavelength shifted by the band's width. all_vars lists all original wavelengths although n_vars is smaller. Reachable through run_search(interference_settings=...) because _has_enabled_interference accepts wavelength_exclusion. The GUI currently does not pass interference to the search.

**Failure scenario.** Synthetic data with a single informative peak at 1250 nm, 2 nm spacing, exclusion '1100-1200'. run_search reports n_vars=99 and full_vars=150, top_vars '1148,1146,1150,…' (inside the excluded band, 102 nm from the real peak), and all_vars listing all 150 wavelengths.

**Verifier.** WavelengthExcluder drops columns inside the prep pipeline, but search keeps wavelengths = X.columns (the full axis) as wavelengths_for_models, so importance indices map to shifted wavelengths and all_vars lists the full axis. It is reachable through the public run_search(interference_settings=...). The GUI hand-off is commented out (GUI L30853), so medium is appropriate.

### R083: One-class smart preprocessing discovery with fewer than 2 outliers ranks every config 1.0 and returns the first candidates in list order
`src/spectral_predict/preprocessing_discovery.py:688` · area preproc-varsel · misleading-output · finder medium → verifier **medium / CONFIRMED**

**Problem.** For task_type='one_class', _quick_evaluate returns a constant 1.0 whenever fewer than 2 samples are labelled -1. That is the normal case for a pure-inlier one-class training set. Every configuration ties, score_config normalises them all to 0, and select_diverse_configs keeps the stable sort. The 'top N' is therefore the first N entries of PREPROCESSING_CANDIDATES at window 7 (raw, snv, deriv1 w7, snv_deriv1 w7, …). These go to run_one_class_search as 'discovered' configurations and are printed as a ranking with Acc=1.0000, with no warning that nothing was evaluated.

**Failure scenario.** discover_preprocessing(X, y=np.ones(40), task_type='one_class', n_top=6) returns [('raw',None,1.0), ('snv',None,1.0), ('deriv1',7,1.0), ('snv_deriv1',7,1.0), ('deriv1_snv',7,1.0), ('deriv2',7,1.0)]. The larger windows (11-37) and the higher derivatives are never offered, whatever the data.

**Verifier.** _quick_evaluate returns a constant 1.0 for one_class when there are fewer than 2 outliers. Every candidate ties, and discover_preprocessing returns the first entries in list order with no warning. run_one_class_search passes y_oc from the inlier label, and the GUI one-class path passes smart_preprocess (L29986), so a pure-inlier training set hits this.

### R084: CARS cannot return fewer than ~9% of the variables, because its decay schedule stops at 0.8*0.8*e^-2 instead of decaying to about 2 variables
`src/spectral_predict/variable_selection.py:1412` · area preproc-varsel · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** r = 0.8*exp(-2*it/n_iterations) and n_sample = p*(monte_carlo_samples/100)*r. At the last iteration this keeps about 0.8*0.8*e^-2 ≈ 9% of the variables. In Li et al. 2009 CARS the decay runs from all variables down to 2 by the final run. RMSECV usually keeps falling as uninformative variables are removed, so the best iteration lands on or near the last one and the result is pinned to the schedule floor rather than the RMSECV optimum. This affects cars, cars-aware, cars-tree, uve_cars, fipls_cars, uve_cars_spa and the smart-discovery cars_tree importance. It also sets the 'method-optimal' count search tests (n_method_optimal = count_nonzero). METHOD_DESIGN_vip_cars.md calls this path 'Canonical CARS (Li 2009)', and the docstring promises 20-50 variable sets.

**Failure scenario.** X random (100×500), y = x50 + x150 + x250 + noise. Over 4 seeds cars_selection returns 45-50 non-zero variables, with the best iteration at 47-50 of 50. The minimum reachable is 45 (9% of 500), so the 3-variable optimum can never be reached.

**Verifier.** r = 0.8*exp(-2*it/N) and n_sample = p*0.8*r, so the last iteration keeps about 0.0902*p variables. The best iteration usually lands at or near the last one, so the output is pinned at the schedule floor, not decayed toward 2 as in Li 2009. SESSION_LOG's BoneCollagen figure (193 of 2151 kept with 50 iterations = 8.97%) matches this floor exactly. The true variables are still included, so this is over-retention rather than a wrong answer; medium stands.

### R085: Kennard-Stone picks the wrong starting pair (condensed-index conversion is for the lower triangle)
`src/spectral_predict/sample_selection.py:110` · area search-cv · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** kennard_stone converts the argmax of scipy's pdist condensed vector to (i, j) with `i = floor(0.5*(1+sqrt(1+8k))); j = k - i*(i-1)//2`. That formula inverts a lower-triangular ordering, but pdist uses upper-triangular row-major order ((0,1),(0,2),...,(0,n-1),(1,2),...). The seed pair is therefore generally not the most distant pair, so every calibration-transfer transfer set and model_io representative set differs from true KS. spxy in the same file does it correctly via squareform/unravel_index.

**Failure scenario.** For X = [[0],[1],[2],[10]], kennard_stone(X, 2) returns [2, 1] (distance 1) instead of the farthest pair (0, 3) (distance 10). This feeds GUI calibration transfer (spectral_predict_gui_optimized.py:47108) and model_io.py:267.

**Verifier.** sample_selection.py:110-111 inverts a lower-triangular condensed index, but scipy pdist uses upper-triangular row-major order. The result is always a valid pair with i>j, so nothing crashes, but it is usually not the farthest pair. The later max-min steps still spread the selection, which keeps this at medium (it changes calibration-transfer and representative subsets). This is outside search-cv proper.

### R086: Validation rebuild drops resampling and regression-weighting imbalance methods, so RMSEP/val_* describe an unbalanced model
`src/spectral_predict/search.py:508` · area search-cv · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** compute_validation_metrics_for_top_models rebuilds each row with _rebuild_model_from_row and fits only after _apply_class_weight_discriminator_for_rebuilt_model. That helper returns {} for any imbalance_method other than 'class_weight'/'auto'. SMOTE/ADASYN/undersampling for classification and RegressionSampleWeighter/RegressionUndersampler/SMOGN for regression were applied inside every CV fold (via the 'imbalance' pipeline step), but they are never applied when fitting the validation model. The row's R2cv/Accuracycv come from a resampled or weighted fit, while its RMSEP/R2pred/val_Accuracy come from a plain fit. The gap penalty in 'use_rmsep_gap' mode then compares mismatched models.

**Failure scenario.** A classification search with imbalance_method='smote' and a held-out validation set: Accuracycv/Recallcv reflect SMOTE-balanced training, but val_Recall for the minority class reflects an unweighted model trained on the imbalanced calibration set. It is typically much lower, and it looks like overfitting (or not) for reasons unrelated to the ranked model.

**Verifier.** _apply_class_weight_discriminator_for_rebuilt_model returns {} for any method other than class_weight/auto (search.py:505-509). compute_validation_metrics_for_top_models never builds the imbalance transformer. The docs mention only the class_weight fix (PR for validation rebuild); resamplers were never addressed or scoped out. Repro: with SMOTE the CV metrics change, but every val_* column is bit-identical to the no-imbalance run, so validation scores a model different from the one CV scored.

### R087: GA/exhaustive preprocessing + baseline/smoothing toggles produce rows labelled ALS/sg0 that were never baseline-corrected or smoothed
`src/spectral_predict/search.py:2896` · area search-cv · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** When ga_preprocess=True, the baseline and smoothing doubling blocks (2731-2766) duplicate each GA config, renaming it 'als+...'/'sg0+...' and setting baseline_method/smoothing. The main loop's GA branch then only runs preprocess_cfg['ga_transform'] (plus autoscale) and ignores baseline_method and smoothing. Result rows claim Preprocess='als+sg0+snv', baseline_method='als', smoothing=True, but the fit is identical to the plain row. Model Development or export code that trusts these columns will rebuild a different pipeline. The GUI passes ga_preprocess, baseline_method and smoothing together (spectral_predict_gui_optimized.py:30846-30948).

**Failure scenario.** With optimize_preprocessing patched to return one SNV chromosome and run_search(ga_preprocess=True, baseline_method='als', smoothing=True), the four rows 'snv', 'als+snv', 'sg0+snv', 'als+sg0+snv' all have exactly the same R2cv (-0.143694 at 1 LV, -0.273508 at 2 LV).

**Verifier.** The GA configs carry baseline_method/smoothing, and the baseline and smoothing doubling blocks (2731-2766) rename them 'als+...'/'sg0+...'. The main loop's ga_transform branch (2896-2905) then applies only the GA closure plus autoscale and never builds baseline/smoothing. The GUI passes ga_preprocess, baseline_method and smoothing together (gui ~30846/30848/30947). The repro patches optimize_preprocessing to return one SNV chromosome, but the branch logic being exercised is the real one.

### R088: Complexity curve and leverage are computed on raw spectra, not the model's preprocessed spectra
`spectral_predict_gui_optimized.py:42251` · area transfer-analysis · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** In the refit path, X_raw = X_work is raw spectra unless the derivative-plus-subset route is used (comment at 41568), and preprocessing such as SNV or SG derivatives lives inside the pipeline. _compute_validation_curve(X=X_raw, ...) calls compute_pls_complexity_curve, a bare PLSRegression with no preprocessing, and the ensemble and regularisation curves. refined_X_cv = X_raw is also what compute_leverage receives. The curve and its optimal_idx therefore describe a different model from the one whose CV metrics are shown next to them.

**Failure scenario.** The selected model is PLS with SNV plus a 2nd-derivative SG filter and 8 LVs. The complexity plot is RMSECV against LVs for PLS on raw reflectance, which typically bottoms out at a different LV count. The user reads the 'optimal' marker as evidence that 8 LVs over- or under-fits the preprocessed model when the curve never saw that model.

**Verifier.** In the default refit path (PATH B, 41418-41470), X_work = X_base_df[selected_cols].values holds raw spectra, and preprocessing is built into the pipe by build_preprocessing_pipeline. X_raw = X_work (41568) is passed both to _compute_validation_curve (42251) and to self.refined_X_cv for leverage (42213). compute_pls_complexity_curve fits a bare PLSRegression on X (diagnostics.py:322), and the Ridge/Lasso/SVM curves use bare estimators. The curve and its optimal marker therefore describe an unpreprocessed model, not the one whose CV metrics are shown. Only the derivative-plus-subset and GA paths pass preprocessed X.

### R089: Multi-model comparison transfer chain applies models positionally without resampling to the model grid, then relabels columns
`spectral_predict_gui_optimized.py:53913` · area transfer-analysis · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** _apply_transfer_chain passes X_current.values straight to apply_transfer_dispatch, or slices ROI indices from it, without resampling the comparison data onto transfer_model.wavelengths_common. It then relabels the output columns as wavelengths_common. Every other apply path resamples first (47728, 48007, 50948, 51495). If the comparison file happens to have the same number of columns on a different grid, the DS/PDS/TSR coefficients are applied to the wrong wavelengths and the columns are silently mislabelled. If the column count differs, the error is shown but the function returns the original, untransferred data, and _run_comparison carries on predicting as if the transfer had been applied.

**Failure scenario.** The transfer model's common grid is 1000-2498 nm at 2 nm (750 points). The comparison CSV covers 1100-2598 nm at 2 nm (also 750 points). A (750x750) is applied position by position, so the 1100 nm column is treated as 1000 nm. The output is labelled 1000-2498 and the models predict on shifted, mis-transferred spectra with no warning. With a 751-point file, the user gets an error box and then comparison results computed on untransferred satellite data.

**Verifier.** _apply_transfer_chain (53875-53938) sends X_current.values straight to apply_transfer_dispatch, or slices ROI indices from it, without resampling to transfer_model.wavelengths_common. It then relabels the output columns as wavelengths_common. Neither the add-model step (53783-53810) nor _run_comparison checks the grid. On exception it shows an error box and returns X_data unchanged, and _run_comparison (53993-53996) goes on to predict with the untransferred data. The more common case is a comparison file on the satellite's native grid, whose column count differs from the common grid. That gives an error box followed by results computed on untransferred data. The same-count, different-grid case gives silent positional misapplication. I traced this through the code and did not run the GUI.

### R090: NS-PFCE models built with wavelength selection cannot be applied: output width no longer matches the grid
`src/spectral_predict/calibration_transfer.py:1410` · area transfer-analysis · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** With use_wavelength_selection=True, apply_nspfce returns only the selected columns (n_selected wide). Every GUI consumer assumes full-width output on wavelengths_common. The prediction path calls resample_to_grid(X_transferred, common_wl, target_wl); the ROI splice assigns X_out[:, roi_indices] = X_region_transferred; the equalize and export paths store the result against model_wavelengths. Each of these raises a shape or length error. The option is exposed in the GUI (the 'NS-PFCE: Use Wavelength Selection' checkbox) and model building reports success, but any use of the model fails. The quality plot also fails, and that error is only printed to the console.

**Failure scenario.** The user ticks 'Use Wavelength Selection' with CARS and builds an NS-PFCE model (success dialog shown). In Section D prediction, resample_to_grid(X_transferred with 40 columns, common_wl with 300 points) raises 'x and y arrays must be equal in length along interpolation axis'. If an ROI was set, the splice raises a broadcast error. The model is unusable.

**Verifier.** With wavelength selection on and return_full_spectrum=False (the default), apply_nspfce returns only n_selected columns. The GUI stores wavelengths_common=self.ct_wavelengths_common, which is full width (47263-47270), and its checkbox exposes the option. The prediction path passes the output to resample_to_grid(X_transferred, common_wl, target_wl), and the ROI splice assigns it into full-width columns. Both fail.

### R091: JYPLS-inv transfer matrix omits PLS centering, so 'transferred' spectra are far worse than untransferred ones
`src/spectral_predict/calibration_transfer.py:1641` · area transfer-analysis · correctness · finder critical → verifier **medium / CONFIRMED**

**Problem.** estimate_jypls_inv builds B = W @ M_scores @ P.T from sklearn PLS weights and loadings. The PLS was fitted on mean-centred X_aug (sklearn centres internally), but apply_jypls_inv computes X_satellite_new @ B with no centring and no mean or offset added back. It also uses x_weights_ rather than x_rotations_ to form scores. The output is therefore a rank-k projection that has lost the mean spectrum, not an estimate of the primary-domain spectrum. The GUI offers the method (method 'jypls-inv', lines 47164-47222) and the output feeds the primary model for predictions. The test suite was relaxed to match ('JYPLS-inv does not guarantee spectral RMSE improvement', tests/test_jypls_inv.py:257).

**Failure scenario.** Synthetic paired NIR set: 40 samples, 200 wavelengths, Xs = 0.9*Xp + 0.05 + noise, 12 transfer samples. Spectral RMSE to primary is 0.027 before transfer and 0.44 after JYPLS-inv, for every n_components tried (1, 2, 3, 5). With a primary PLS model for a 0.2-1.0 target, prediction RMSE is 0.012 on primary spectra, 0.059 on raw satellite spectra, and 1.98 on JYPLS-transferred spectra: roughly 33 times worse than applying no transfer. The error is silent; the GUI reports 'built successfully' with an 'Explained Variance' figure.

**Verifier.** The math bug is real. apply_jypls_inv computes X @ (W M P^T) with no x_mean_ centring or offset, and uses x_weights_ rather than x_rotations_, so the output is a rank-k projection that has lost the mean spectrum. My repro reproduced it. The 'GUI offers the method' claim is wrong, though. The JYPLS-inv radio button is created with state='disabled' (GUI 54887-54888) and has been since commit b177c91 (Nov 2025). ct_method_var is only ever initialised to 'nspfce' and is never set to 'jypls-inv' anywhere else. So the GUI cannot build a JYPLS-inv model. The bug can only be reached through the public backend functions estimate_jypls_inv/apply_jypls_inv (and apply_transfer_dispatch), or through a previously saved jypls-inv model file. These are not documented in AGENT_COMPOSITION.md. The corruption is silent, but it is effectively dormant for GUI users, so I lowered critical to medium.

### R092: compute_leverage uses the full spectral matrix, so every sample gets leverage 1.0 and nothing is ever flagged
`src/spectral_predict/diagnostics.py:91` · area transfer-analysis · numerical · finder high → verifier **medium / CONFIRMED**

**Problem.** Hat values are computed on [1, X] with X equal to the full spectra, where p is hundreds or thousands of wavelengths. When n <= p+1 the SVD branch gives U as an n-by-n orthonormal matrix, so every hat value is exactly 1. The 2p/n and 3p/n thresholds, with p = n_wavelengths + 1, exceed 1 whenever 2(p+1) > n, so no sample can ever cross them. The Model Development leverage plot therefore always reports 'High leverage: 0 samples' for spectral data, including gross outliers. Hat values for PLS should come from the latent-variable scores.

**Failure scenario.** 60 samples by 200 wavelengths, with sample 0 multiplied by 20 as a gross outlier: every leverage is 1.0 (outlier 0.99999), threshold_2p = 6.7, and 0 samples are flagged. 80 samples by 50 variables with an outlier: outlier leverage 0.998, threshold 1.275, still not flagged. The plot tells the user the calibration set has no influential samples.

**Verifier.** compute_leverage builds hat values on [1, X] with X as the full spectral matrix. When n <= p+1, U from the thin SVD is square orthonormal, so every leverage is 1. The GUI thresholds 2(p+1)/n and 3(p+1)/n then exceed 1 and can never be crossed. The GUI caller (38827-38833) passes self.refined_X_cv, which is the full wavelength matrix. The plot therefore always reports zero high-leverage samples on spectral data. This is a misleading diagnostic, but it does not change any fitted model or prediction, so I rated it medium rather than high.

### R093: Library search linearly extrapolates spectra outside their measured range, and those values dominate the scores
`src/spectral_predict/library_search.py:322` · area transfer-analysis · correctness · finder medium → verifier **medium / CONFIRMED**

**Problem.** The library grid is fixed to the first spectrum ever added (line 207). Every later query or entry is aligned with interp1d(..., fill_value='extrapolate'), so any part of the grid outside a spectrum's own range is filled by extending the line through its last two points. There is no overlap check or masking, and the similarity metrics use the whole vector. Searches across instruments with different ranges therefore compare mostly fabricated values.

**Failure scenario.** The library was seeded with an ASD spectrum (350-2500 nm). The user searches with a spectrum measured at 1000-2500 nm. The query is extrapolated over 350-1000 nm from its slope at 1000-1002 nm, which can run to large positive or negative values. HQI, SAM or Euclidean distance over 350-2500 nm is then driven by the 650 nm of invented data, and match rankings are arbitrary. Adding such spectra also runs the near-duplicate check on extrapolated data.

**Verifier.** The library grid is set from the first spectrum ever added (library_search.py:207). _align_to_grid uses interp1d with fill_value='extrapolate', and no overlap check or masking happens anywhere: search(), _check_spectral_duplicate and export_to_csv all use it. A query or entry covering a narrower range than the seed grid is therefore linearly extrapolated across the missing region, and every metric scores the whole vector. I confirmed this from the code and wrote no repro.

### R094: Phase-2 multi-seed rescore ranks a candidate on whichever seeds survived; a config that failed 4 of 5 seeds is scored on one seed with std=0
`src/spectral_predict/phase2_rescore.py:58` · area bayesian · numerical · finder low → verifier **low / PLAUSIBLE**

**Problem.** `_aggregate_scores` drops non-finite per-seed scores and returns the mean and std of the survivors. `n_valid` is computed but not used in `_rank_key` or `_select_diverse`, and not exported in `winner_scores`. The TPE multistart wrapper then overwrites cfg['score'] with that mean and reports '±std'. A candidate that raises or returns -inf on most seeds (tree-proxy CV crash, NaN after preprocessing) is ranked on a single lucky seed. Its std is 0.0, which also wins the std tie-break. It can take a top-N slot over configs that were stable on all 5 seeds, and it is presented as '±0.0000'.

**Failure scenario.** n_seeds=5; candidate A scores -0.50 on 5/5 seeds (mean -0.50, std 0.02). Candidate B raises on 4 seeds and scores -0.45 once, giving mean -0.45 and std 0.0. B outranks A and is printed as 'RMSE=0.4500 ±0.0000', which looks like the most stable, best config.

**Verifier.** The mechanism is real. `phase2_adaptive_rescore` converts exceptions to -inf, `_aggregate_scores` drops non-finite values and ranks on the survivors, and n_valid is discarded at line 260 and never exported, so a 1-of-5-seed survivor wins with std 0.0. But the only real caller, tpe_preprocessing_discovery `_eval_fn`, applies seed-independent preprocessing. A non-finite X_prep or a preprocessing exception therefore fails all seeds alike and hits the all-invalid sentinel, which is already filtered. Seed-dependent failure could only come from the seeded CV evaluators (shuffled KFold/StratifiedKFold with LGBM, PLS or LogReg). There, fold-fit errors become NaN via cross_val_score's error_score, and a training fold missing a class would need a class with 1 member, which fails every seed. I could not build a realistic real-path input that fails on only some seeds. Impact is limited to ranking or display of preprocessing candidates.

### R095: Append-mode alignment failure leaves merged X_original with un-merged y and no rollback
`spectral_predict_gui_optimized.py:19448` · area gui-1 · error-handling · finder low → verifier **low / CONFIRMED**

**Problem.** When one side of an append has targets and the other does not (e.g. appending a spectra-only file), _merge_spectral_data returns y_merged = y_existing, which is shorter than X_merged. The post-load index check then shows 'Data Alignment Error' and returns. By that point X_original, source_group_names and data_sources have already been mutated and are not rolled back, unlike the try/except rollback just above. self.X also stays at the pre-append filtered data, so the tabs disagree.

**Failure scenario.** Dataset with targets loaded; tick Append and load a folder of unknown spectra without a reference file. After the error dialog, the data-sources label reports both sources and X_original holds N+M rows while y holds N. The next 'Update wavelengths' sets self.X to N+M rows, and every later analysis fails with the worker's length-mismatch error until the user reloads from scratch.

**Verifier.** When y_new is None, _merge_spectral_data sets y_merged = y_existing, so X_merged has N+M rows and y has N. The index-equality check at gui:19443 then shows an error and returns. By then source_group_names/data_sources have been appended and _existing_* cleared, and nothing restores X_original. Only the try/except above rolls back. The worker's length check later raises a clear error, so this is an error-handling annoyance rather than silent corruption: low.

### R096: VIP screening crashes on any missing target; categorical correlation treats NaN as a class; screening ignores exclusions
`spectral_predict_gui_optimized.py:22723` · area gui-1 · error-handling · finder low → verifier **low / CONFIRMED**

**Problem.** Data is loaded with drop_na_y=False, so self.y routinely contains NaN. _compute_vip_screening passes y_numeric straight to PLSRegression.fit, which raises 'Input y contains NaN', so VIP screening fails on every such dataset. The RF path drops NaN, but correlation and VIP do not. Correlation on a categorical target converts to str first, so NaN becomes a 'nan' class, and it correlates spectra with arbitrary LabelEncoder codes. None of the three screens applies excluded_spectra or the analysis subset, so outliers the user already removed still drive the importance ranking.

**Failure scenario.** A combined CSV where 3 of 120 samples lack the protein value. In Explore -> Predictor Screening, choose VIP and the error dialog 'VIP screening failed: Input y contains NaN' appears. With a categorical target that has blanks, correlation screening ranks wavelengths by their correlation with codes that include a spurious 'nan' category.

**Verifier.** Data is loaded with drop_na_y=False. _compute_vip_screening passes y straight to PLSRegression.fit with no NaN filter, and the Explore caller (gui:10872) does not filter either. Correlation screening label-encodes categorical y, which also encodes NaN as a class. No screening function references excluded_spectra. Nothing is corrupted beyond the screening display, so low.

### R097: Blank or non-integer Bayesian Trials box passes the launch gate, shows a false SQLite warning, then the worker fails
`spectral_predict_gui_optimized.py:26932` · area gui-2 · error-handling · finder low → verifier **low / CONFIRMED**

**Problem.** n_unified_trials (a ttk.Entry over an IntVar) is missing from BAYESIAN_REQUIRED_SETTINGS, so an unreadable value does not block launch. In _register_fresh_run, `int(self.n_unified_trials.get())` raises TclError inside the generic `except Exception` meant for run-state failures. That path logs 'Run-state init failed' and, under the default 'auto' persistence, shows 'Crash-resume disabled ... SQLite store could not be initialized', which is false. The launch proceeds with _pending_bayesian_n_trials None. The worker's _bayes_n_trials() then falls back to a live, off-thread `self.n_unified_trials.get()`, which raises at line 30339 and ends the run with 'Analysis failed: expected floating-point number'.

**Failure scenario.** Clear the Trials box (or type 300.5), choose Bayesian and click Run. The user sees a misleading SQLite/crash-resume warning, then an 'Analysis failed' dialog, instead of the 'Invalid settings' message the gate gives for other blank fields.

**Verifier.** n_unified_trials is not in BAYESIAN_REQUIRED_SETTINGS, so the launch gate (26852-26870) does not block a blank value. The cost estimate at 24440 swallows the error. In _register_fresh_run, int(self.n_unified_trials.get()) sits inside the try, so a TclError goes to the generic `except Exception`. That logs 'Run-state init failed' and, under auto/always persistence, shows the false 'SQLite store could not be initialized' warning. _pending_bayesian_n_trials stays None, and the worker's _bayes_n_trials() then calls the live Tk var off-thread, which raises again. The user gets a misleading error sequence instead of a clean block.

### R098: Result checkboxes show stale state after filter + toggle + sort; the ensemble uses different rows than shown
`spectral_predict_gui_optimized.py:31639` · area gui-2 · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** _apply_filters_inner stores results_filtered_df as a copy of results_df. _on_result_click toggles 'Select' only in self.results_df. _sort_results_by_column then re-sorts and redisplays the stale results_filtered_df, so the checkbox column reverts to the state at filter time. _on_train_ensemble_click reads self.results_df['Select'], so the trained ensemble differs from what the table shows. Clicking a stale ☐ to re-select actually deselects the row.

**Failure scenario.** Filter Model=PLS, tick rows 3 and 5, then click the RMSEcv header. Both rows show ☐ again. The user ticks them 'again', which toggles them off in results_df while the display shows ☑, and Train Ensemble builds without them.

**Verifier.** _apply_filters_inner stores a copy of results_df in results_filtered_df. _on_result_click toggles 'Select' only in self.results_df (32805). The sort handler repopulates from results_filtered_df (31639), and _populate_results_table_inner renders the checkbox from that frame's row['Select'] (32651). The display therefore reverts to the state at filter time, while the ensemble reads results_df['Select'].

### R099: The one-class learning curve evaluates a different model: raw estimator, no scaler/PCA, trained on outliers too
`spectral_predict_gui_optimized.py:39566` · area gui-2 · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** After a one-class refit, refined_model is cv_result['cal_model'] (fit on scaled/PCA-reduced inliers), refined_X_train is the unscaled X_work, and refined_y_train is ±1 labels. The learning-curve button is enabled for this case. _run_learning_curve_thread clones the bare estimator and calls sklearn learning_curve with scoring='accuracy' on unscaled X, fitting every training slice on inliers and outliers together (one-class fit ignores y) without the scaler or PCA reducer. The one-class CV protocol trains on inliers only. The plotted '1-Accuracy' curve and its interpretation text describe a model that was never built.

**Failure scenario.** Refit an OCSVM one-class row in Tab 7 and click Learning Curve. The curve shows fits on contaminated, unscaled data, for example flat high error or misleading 'more data will help' advice, unrelated to the reported BalancedAcccv of the actual model.

**Verifier.** The one-class refit stores refined_model = cv_result['cal_model']. That is the bare estimator: the scaler and PCA are kept separately as cal_scaler and cal_pca_reducer (contamination.py 830-894). refined_X_train is the unscaled X_work and refined_y_train is ±1. _run_learning_curve_thread clones the bare estimator and calls sklearn learning_curve with scoring='accuracy'. That fits on inliers and outliers alike, with no scaler or PCA. The learning-curve button is enabled after any successful refit. This is a diagnostic plot only.

### R100: Tab 7 GA failure paths leave Run buttons disabled and the wait cursor on
`spectral_predict_gui_optimized.py:41149` · area gui-2 · error-handling · finder low → verifier **low / CONFIRMED**

**Problem.** _run_refined_model disables the three Run buttons and sets cursor='wait'. Every other early exit goes through _update_refined_results, which re-enables them. The GA ImportError branch and the GA optimisation exception branch only set refine_status and show a dialog before `return`, so all Tab 7 Run buttons stay disabled and the busy cursor remains until another model is loaded from Results.

**Failure scenario.** Choose preprocessing 'ga_optimized' on a fresh config (no saved genes) and let optimize_ga_preprocessing raise (for example too few samples for cv_folds). After the error dialog, the Run Refined Model buttons are greyed out and the cursor stays busy, so the user cannot rerun with other settings.

**Verifier.** _run_refined_model disables all three Run buttons and sets cursor='wait' (39675-39679). The GA ImportError branch (41131-41149) and the GA-optimisation exception branch (41218-41237) schedule only a status update and a messagebox, then return from inside the outer try. They never reach _update_refined_results (42282-42294), the only path in the thread that restores the cursor and buttons. Loading a model (37220) re-enables the buttons, as the finding says.

### R101: Tab 7 classification reproducibility check compares refit CV accuracy with the row's calibration accuracy
`spectral_predict_gui_optimized.py:41906` · area gui-2 · correctness · finder medium → verifier **low / CONFIRMED**

**Problem.** For classification the 'COMPARISON TO LOADED MODEL' block uses selected_model_config['Accuracy'], labelled 'Original Accuracy (from Results tab)', against results['accuracy_mean'], which is the refit's CV accuracy. In both grid (search.py 5484) and Bayesian rows, 'Accuracy' is the calibration (training-set) accuracy; the comparable column is 'Accuracycv'. The regression branch correctly uses R2cv. A faithful refit therefore reports a large negative 'Difference', and a wrong refit can appear to match.

**Failure scenario.** A PLS-DA row has Accuracy 1.000 (calibration) and Accuracycv 0.850. The Tab 7 refit reproduces 0.850 exactly but shows 'Original Accuracy 1.0000 / Refined 0.8500 / Difference -0.1500', so the user concludes the refit failed to reproduce the ranked model.

**Verifier.** The classification branch (41905-41910) compares results['accuracy_mean'] (refit CV) with selected_model_config['Accuracy']. search.py 5484 sets 'Accuracy' to the calibration accuracy and 5496 sets 'Accuracycv' to the CV accuracy. The regression branch correctly uses R2cv. This only affects the diagnostic comparison text, with no effect on models, so I lowered severity to low. It can still mislead a reproducibility check.

### R102: Live monitoring detects changes by file count only, so replaced or partially written spectra are not refreshed
`spectral_predict_gui_optimized.py:53602` · area gui-3 · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** _perform_live_scan reloads and re-runs the comparison only when the ASD+CSV+SPC file count changes. A spectrum caught mid-write is skipped by read_asd_dir (logged 'skipping ...') or read truncated. Once the write completes, the count is unchanged, so it is not re-read until another file arrives. An overwritten or replaced file (same count) is never picked up. The count also ignores JCAMP, ASCII and Omnic files, which the loader does read.

**Failure scenario.** The last sample of a shift is saved as the scan fires. It is skipped as unreadable, the status says 'Monitoring active - 25 files', and the displayed and exported results contain 24 samples with no indication that one is missing. A re-measured sample saved over its old file keeps showing the old prediction.

**Verifier.** _perform_live_scan reloads only when _count_spectral_files (ASD+CSV+SPC only) changes; otherwise it just updates the timestamp. read_asd_dir prints 'Warning: skipping' for unreadable files (io.py 738/746), so a file caught mid-write is dropped and is not re-read until the count changes. An overwritten file (same count) is never reloaded, and JCAMP, ASCII and Omnic files are not counted.

### R103: read_spectra on a single ASD file returns every ASD spectrum in its folder; JCAMP/Excel folders go to single-file readers
`src/spectral_predict/io.py:2582` · area io-persistence · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** read_spectra is on the declared stable API (AGENT_COMPOSITION.md §io). For a single .asd/.sig/.sco file it calls read_asd_dir(path.parent), which silently returns every ASD file in that directory. For a directory, _detect_directory_format may return 'jcamp' or 'excel', which dispatch to read_jcamp_file/read_excel_spectra with the directory path and crash, even though read_jcamp_dir exists.

**Failure scenario.** An agent script calls read_spectra('field/plot7.asd') expecting one spectrum and receives 250 rows (all of field/). Downstream it pairs them with a single reference value or predicts on the wrong samples.

**Verifier.** io.py:2576-2582: for a single ASD file, read_spectra returns read_asd_dir(path.parent), which reads every ASD file in the folder. For a directory detected as 'jcamp' or 'excel', it dispatches to read_jcamp_file/read_excel_spectra with the directory path, even though read_jcamp_dir exists. read_spectra is on the declared agent surface (AGENT_COMPOSITION.md:532). The GUI does not use this dispatcher for these cases, so severity is low. Confirmed by code reading; the dispatch is deterministic.

### R104: predict_with_uncertainty(validate_wavelengths=False) ignores the full-spectrum handshake that predict_with_model honours
`src/spectral_predict/model_io.py:1048` · area io-persistence · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** predict_with_model was fixed so that the DataFrame plus validate_wavelengths=False path preprocesses the full spectrum and then subsets (model_io.py:700-710). predict_with_uncertainty recomputes X_processed for AD and tree variance with the old logic: `preprocessor.transform(X_new.values)` with no subset. Every GUI-saved model now has use_full_spectrum_preprocessing=True, so the second pass hands a full-width matrix to pca_model.transform (fitted on the subset width) or to the RF trees. The call raises after the predictions have already been computed.

**Failure scenario.** predict_with_uncertainty(loaded, X_df_full_width, validate_wavelengths=False) on a subset derivative model: predictions are computed, then 'X has 1000 features, but PCA is expecting 120', so no result is returned.

**Verifier.** With a DataFrame and validate_wavelengths=False, predict_with_uncertainty (model_io.py:1048-1053) transforms the full-width matrix and never applies the full-spectrum subset that predict_with_model honours (model_io.py:700-710). The AD PCA then gets the wrong feature count. The GUI calls it with validate_wavelengths=True (GUI:44504) or the default, so only public-API callers who pass False are affected.

### R105: A base model that fails OOF gets equal weight, NaN predictions or a crash, not the promised exclusion
`src/spectral_predict/ensemble.py:403` · area models-ensemble · error-handling · finder medium → verifier **low / CONFIRMED**

**Problem.** When a base model's per-fold refit raises, the warning says 'This model will be excluded from weight calculation' and its OOF row is set to NaN. In RegionAwareWeightedEnsemble the NaN spreads through the per-region normalisation and nan_to_num then sets every weight in every region to 1/n_models, so the broken model gets full equal weight and region weighting is disabled. In MixtureOfExperts, avg_pred becomes NaN, the region boundaries become NaN and the expert weights contain NaN, so predictions are NaN. StackingEnsemble passes the NaN column to Ridge.fit, which raises.

**Failure scenario.** The ensemble holds a good Ridge and a model whose refit raises (e.g. a clone-incompatible wrapper) but which predicts noise when used as fitted. region_weighted gives weights [[0.5,0.5,0.5],[0.5,0.5,0.5]], so the noise model gets half the weight. mixture_experts gives expert_weights [[nan,1,1],[nan,0,0]]. stacking fails with 'Input X contains NaN'.

**Verifier.** This is partly overstated. When refit raises, the OOF row becomes NaN. RegionAware then falls back to equal weights 1/n in every region, so the failed model gets half the weight despite the warning saying it is excluded. Stacking raises 'Input X contains NaN' (the GUI's per-method except catches it). MoE gets NaN region boundaries and NaN expert weights, but its predictions were not NaN in my run, so the 'predictions are NaN' claim did not reproduce. The trigger, a model whose clone/refit raises but whose pre-fitted predict works, is uncommon, so I rated it low.

### R106: create_auto_ensembles' CV falls back to the full-data ensemble and counts failed reconstructions as selected models
`src/spectral_predict/ensemble.py:1913` · area models-ensemble · leakage · finder low → verifier **low / PLAUSIBLE**

**Problem.** If a fold's reselection yields fewer than 2 unique models, cv_predictions[val_idx] comes from the ensemble fitted on all of X_train, which is in-sample. unique_model_count counts indices before reconstruction succeeds (all_selected_indices.add precedes the try), so a fold where every reconstruction fails still passes the >=2 check. Its RegionSpecialistEnsemble then has no models and predict() returns zeros. The function is currently only exercised by tests, not by the GUI, so impact is limited.

**Failure scenario.** reconstruct_func raises for the selected rows on a fold's training subset (e.g. n_components greater than that fold's samples). cv_selection['unique_model_count']=2 with empty model lists, so predict returns np.zeros and R2 is strongly negative. Alternatively the count is <2 and the main ensemble, trained on the validation rows, predicts them.

**Verifier.** The code matches the description: all_selected_indices.add runs before the reconstruct try, and empty region lists make predict return zeros. However, create_auto_ensembles has no production caller (grep finds only its definition outside tests). Selection comes from results_df rankings and is the same in every fold, and the main ensemble already requires unique_model_count >= 2. The '<2 -> full-data ensemble fallback' branch is therefore effectively unreachable. Only the fold-specific reconstruction-failure case remains, and I did not demonstrate it.

### R107: GA-PLS / GA-LightGBM importances are coarse selection frequencies, and ties are broken by highest wavelength index
`src/spectral_predict/ga_pls.py:673` · area nsga-ga · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** ga_pls_selection and ga_lightgbm_selection return selection_counts/n_runs, which takes only n_runs+1 distinct values (for example {0, 0.5, 1} in quick mode with 2 runs). Each run's best chromosome keeps about 30% of the wavelengths. search.py turns these into top-N subsets with np.argsort(importances, kind='stable')[-n:], so within a tied frequency group the highest-index wavelengths win. The 'top-N GA' subsets are therefore largely an index-order artifact, not a ranking.

**Failure scenario.** Quick-mode GA-PLS (2 runs) on 1000 wavelengths: about 100 wavelengths are selected in both runs (frequency 1.0). 'top-50 ga' then takes the 50 of them with the largest column indices, systematically favouring the long-wavelength end rather than the most informative bands.

**Verifier.** ga_pls_selection returns selection_counts/n_runs (L669-696), which has only n_runs+1 distinct levels: 2 runs in quick mode, 5 by default in the GUI. search.py L3950 takes np.argsort(importances, kind='stable')[-n_fit:][::-1], and among ties a stable ascending sort puts the highest indices last. Within a frequency level, top-N therefore picks the highest column indices. I verified this from the code and numpy's stable-sort semantics.

### R108: Exhaustive-preprocessing configs report polyorder = deriv-1, which is invalid and not what was fitted
`src/spectral_predict/ga_preprocessing.py:1352` · area nsga-ga · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** exhaustive_search sets polyorder = max(deriv_order - 1, 0). The transform actually applied uses SavgolDerivative's defaults of 2/3 for deriv1/deriv2 and pinned 4/5 for deriv3/deriv4. search.py copies this value into the results 'Poly' column, and the GUI copies it into saved-model metadata 'polyorder'. A polyorder below the derivative order is mathematically invalid (scipy returns all-zero coefficients). smart_exhaustive_search uses max(deriv+1, 2) for the same field. Search-time and validation rebuilds use the chromosome, so fitted numbers are unaffected; the displayed and persisted preprocessing description is wrong.

**Failure scenario.** An exhaustive 2nd-derivative row shows Poly=1 (3rd derivative: 2; 4th: 3), while the model used polyorder 3 (4/5). Anyone rebuilding the preprocessing from the Poly column or from saved metadata gets a zero-valued derivative.

**Verifier.** exhaustive_search L1352 sets polyorder = deriv-1, but the transform actually applied (_spectrum_steps L249) uses the SavgolDerivative defaults for deriv1/2 and pinned values of 4/5 for deriv3/4. smart_exhaustive_search uses deriv+1, which does match. The value flows to search.py L2417 and then to the 'Poly' column (L5352) and the GUI saved-model 'polyorder' metadata (L42442). The fitted numbers are unaffected. Only the displayed and persisted description is wrong.

### R109: Failed evaluations (1e10 penalty) are feasible and enter the Pareto front, results table and knee normalisation
`src/spectral_predict/nsga2_search.py:1580` · area nsga-ga · correctness · finder medium → verifier **low / CONFIRMED**

**Problem.** Any exception in _compute_prediction_error returns error=1e10, but G stays feasible and objectives 2 and 3 are real. A failed chromosome with few wavelengths or low complexity is therefore non-dominated and survives into res.F. convert_nsga2_to_v1_format then emits it as a ranked row. find_knee_point min-max normalises objective 1 across the front, so a single 1e10 row squashes every real error to about 0. Knee and 'Balanced' selection then effectively ignore prediction error. Besides the known SVM mapping, deriv3 with window 5 and deriv4 with windows 5 and 7 always raise, because SavgolDerivative needs window >= polyorder+2 with default polyorders 4 and 5. Swallowed CatBoost failures are another source.

**Failure scenario.** In a scratch classification run with models ['PLS','RandomForest','SVM'] and seed 1, an SVM penalty chromosome landed on the front and in the results table with Accuracycv=-1e10. In a find_knee_point check on the front [[.50,.30,.20],[.55,.10,.15],[.80,.02,.05],[1.50,.01,.03]], the knee is index 2 (error 0.80). Adding one row [1e10, .0028, 0] moves the knee to index 3 (error 1.50).

**Verifier.** The mechanism is real. A failure returns 1e10 while G stays feasible, and find_knee_point's min-max normalisation gets squashed: the knee moves from 2 to 3 with bias 2 but is unchanged with bias 1. deriv3/deriv4 with window 5 always raise (SavgolDerivative window < polyorder+2). The claim that deriv4 with window 7 also raises is wrong; it runs. Penalty rows did reach the front: in 6 seeds with models ['PLS','RandomForest','SVM'], 5 had 1-2 penalty rows after 2 generations. After 6 generations none remained in any seed. With the GUI default of 120 generations they are normally dominated out. The GUI default selection 'Min Error' (bias 0) does not use find_knee_point either. Real, but transient in realistic runs, so low.

### R110: knee_solution objectives['n_wavelengths'] reports n^2/N instead of n
`src/spectral_predict/nsga2_search.py:2200` · area nsga-ga · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** Objective 2 is stored as (n_selected/N)^2, but both result paths convert it back by multiplying by N. The reported wavelength count in knee_solution['objectives'] is therefore n^2/N.

**Failure scenario.** A 16-wavelength solution on a 60-wavelength spectrum reports objectives['n_wavelengths'] = 4.27 (verified), while decode gives 16. Anything reading the objectives dict gets a wrong parsimony value.

**Verifier.** F[:,1] = (n/N)^2 (L1302), and L2200 and L2224 multiply it back by N. Nothing reads knee_solution['objectives']['n_wavelengths']: grep finds only 'error' and 'complexity' being consumed (L2243-2244, L3942, L3991, GUI L30824). It is a wrong value in the public return dict with no current effect.

### R111: NSGA-II best-from-all row drops imbalance handling and computes calibration/R2cv wrongly
`src/spectral_predict/nsga2_search.py:4014` · area nsga-ga · correctness · finder low → verifier **low / PLAUSIBLE**

**Problem.** When the min-error solution is not on the Pareto front, a best_row is added. It hard-codes 'Imbalance': 'none' and omits the imbalance_method/imbalance_params keys, so compute_validation_metrics_for_top_models and refits of that row run without the user's SMOTE or class_weight setting. Its calibration RMSE/R2 and R2cv call get_model(name, task, {}), which puts {} in the n_components slot and raises TypeError, so they silently become NaN/None. They also apply the SG/SNV transform to X[:, selected_indices] (non-contiguous columns) rather than to the full spectrum. The trigger is uncommon: selection_bias=0 and a min-error solution missing from the final front.

**Failure scenario.** Classification run with imbalance_method='smote' where the min-error chromosome was dropped from res.F. The added 'nsga2_best' row gets validation metrics computed without SMOTE, and its R2cv is None even though it is ranked first.

**Verifier.** The code defects are real. best_row hard-codes 'Imbalance': 'none' and has no imbalance_method key, and the GUI refit reads selected_model_config.get('imbalance_method') (L41020), so a refit of this row drops SMOTE or class_weight. get_model(name, task, {}) raises TypeError for every model ('<' between int and dict, confirmed), so RMSE/R2/R2cv silently become NaN/None. The transform is also applied to non-contiguous columns. Part of the finding is wrong, though: compute_validation_metrics_for_top_models takes imbalance_method as a function argument and never reads it from the row, so validation metrics are not affected. I also could not show the trigger happening. Only feasible evaluations are recorded, the minimum-error point is an extreme of objective 1 that NSGA-II's crowding keeps, and the row is added only when knee_error < pareto_min_error strictly. The failure scenario also cites R2cv, which exists only on the regression branch.

### R112: Model Development one-class refit can pick the wrong wavelength column (first match within ±0.5)
`spectral_predict_gui_optimized.py:40780` · area one-class · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** When refitting a one-class model, the stored wavelengths are mapped to columns with np.where(np.abs(original_wavelengths - wl) < 0.5)[0][0], i.e. the first column within 0.5 units, not the nearest. When the axis spacing is under 1 unit (e.g. FTIR/Raman at ~0.3-0.5 cm-1), a lower neighbour comes first, so the model is refitted on shifted channels. Duplicate indices can also appear.

**Failure scenario.** Axis 1000.0, 1000.3, 1000.6, 1000.9 and a stored wavelength of 1000.6: the matches are [1000.3, 1000.6, 1000.9] and idx[0] selects 1000.3. Every selected variable moves one channel down and the reported refined metrics describe a different feature set.

**Verifier.** GUI 40778-40781 takes idx[0] of all columns within 0.5 units, not the nearest. On axes spaced under 0.5 units the lower neighbour wins, so the refit and saved model use shifted channels. Axes spaced 0.5 or more are unaffected, because no neighbour falls strictly inside 0.5. Impact is limited to high-resolution axes, so low is acceptable, though for those users it silently changes the saved model's features.

### R113: 'Apply correction' overwrites the main dataset with mean-centred (EPO) or autoscaled (OPLS filter) spectra
`spectral_predict_gui_optimized.py:60102` · area one-class · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** EstimatedEPO.transform returns X − X_mean_ projected, and ContaminantOPLSDA.transform returns StandardScaler-scaled, centred data. _contam_apply_correction writes that result into self.X under the original wavelength columns. Later searches then run SNV, baseline, MSC and derivatives on centred residuals rather than spectra. SNV of a centred spectrum divides by the residual's own std, which amplifies noise, and baseline fits lose their meaning.

**Failure scenario.** The user applies 'OPLS-DA Filter' to the Main Dataset, then runs a search with SNV or ALS baseline. Every spectrum is now zero-mean per column and unit-variance, so row-wise SNV and baseline correction produce non-physical features, and the results are not comparable to runs on the original spectra.

**Verifier.** EstimatedEPO.transform returns (X - X_mean_) @ P_orth_. ContaminantOPLSDA.transform (scale=True by default) returns StandardScaler-scaled, centred data. _contam_apply_correction writes the result into self.X under the original columns. A backup, X_before_contam_correction, is stored but nothing restores it. Later SNV, MSC and baseline steps then act on centred residuals. The user starts this action, so low is fair.

### R114: 'Export Corrected Spectra' writes no file but shows a green 'exported' status
`spectral_predict_gui_optimized.py:60262` · area one-class · persistence · finder low → verifier **low / CONFIRMED**

**Problem.** _contam_export_corrected_spectra is a placeholder. After the user picks a path it shows 'Export functionality would save corrected spectra here' and then sets the status label to '✓ Corrected spectra exported to <file>'. Nothing is written.

**Failure scenario.** The user exports corrected spectra to corrected.csv. The panel shows a success tick with the filename and no file exists. A user who dismisses the info box keeps a persistent false success message.

**Verifier.** _contam_export_corrected_spectra, wired to a button at GUI 58345, only shows 'Export functionality would save corrected spectra here' and then sets a green '✓ Corrected spectra exported to <file>' status. No file is written.

### R115: Any missing target value turns off all y outlier checks in the outlier report
`src/spectral_predict/outlier_detection.py:445` · area one-class · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** check_y_data_consistency uses np.mean/np.std/np.median, which return NaN if any y is NaN. The zero-std guard (std < 1e-10) is False for NaN, so every z-score is NaN and |z|>3 is never true. The GUI deliberately keeps rows with missing targets (self.y reindexed with NaN, 'analysis code handles NaN') and passes self.y.values straight to generate_outlier_report.

**Failure scenario.** y has one NaN and one mistyped value of 50 among N(0,1) values. check_y_data_consistency returns n_outliers=0 and Y_ZScore is NaN for all rows, so the gross y error is never flagged.

**Verifier.** check_y_data_consistency uses np.mean and np.std without removing NaN. A NaN std fails the <1e-10 guard, every z-score becomes NaN, and |z|>3 is never True. The GUI keeps rows with NaN targets (17547) and passes self.y.values to generate_outlier_report (21654). Range bounds still work, but the z-score check is silently disabled.

### R116: BaselineALS and BaselinePolynomial keep the input dtype, so integer spectra have their baseline-corrected values truncated
`src/spectral_predict/baseline.py:167` · area preproc-varsel · numerical · finder low → verifier **low / CONFIRMED**

**Problem.** Both transforms use X = np.asarray(X) and X_corrected = np.zeros_like(X) without casting to float. BaselineRubberBand, BaselineAirPLS and BaselineAdvanced all cast. read_csv_spectra and read_combined_csv keep integer columns (pd.to_numeric), and run_search passes X.values straight into the pipeline. So integer detector counts give an int64 output, and y - z is truncated toward zero.

**Failure scenario.** A Raman or counts CSV with integer intensities and baseline 'als': the corrected spectrum is quantised to whole counts, with fractional parts dropped by truncation rather than rounding. Low-intensity regions near the baseline turn into steps, which SNV and derivatives then amplify.

**Verifier.** BaselineALS and BaselinePolynomial use zeros_like on the input without casting, so int64 input gives int64 output. read_combined_csv keeps integer CSV columns as int64, and run_search passes X.values straight through. The error is under one count, which matters only for low-count data.

### R117: The Advanced baseline's GUI 'Lambda' value is silently ignored for algorithms without a lam parameter, and their own parameters cannot be set
`src/spectral_predict/baseline_advanced.py:441` · area preproc-varsel · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** The GUI always sends {'algorithm': key, 'lam': value} for the advanced baseline. transform only forwards keys listed in the algorithm's default_params. For snip, imor, modpoly, imodpoly, dietrich and std_distribution, 'lam' is dropped, and their real parameters (max_half_window, half_window, poly_order) always stay at the registry defaults. The user's value has no effect and no warning is given.

**Failure scenario.** The user picks Advanced → ModPoly and sets Lambda to 1e3, expecting to tune it. The pipeline runs ModPoly with poly_order=5 regardless, and every search row labels the run with the Lambda value that had no effect.

**Verifier.** transform forwards only keys listed in the algorithm's default_params. For snip, imor, modpoly, imodpoly, dietrich and std_distribution, the GUI's 'lam' is dropped, and those algorithms' own parameters cannot be set from the GUI.

### R118: choose_common_grid can extend past the overlapping range, and the extra points are linearly extrapolated
`src/spectral_predict/equalization.py:48` · area preproc-varsel · numerical · finder low → verifier **low / CONFIRMED**

**Problem.** np.arange(min_wl, max_wl + spacing, spacing) can yield a last point greater than max_wl, the smallest instrument maximum, whenever the range is not an exact multiple of the spacing. resample_to_grid uses interp1d(..., fill_value='extrapolate'). So that point is a linear extrapolation beyond at least one instrument's measured range, and it is stacked into the equalized dataset as if measured.

**Failure scenario.** Instruments covering 400-2500 and 350-2600 with a coarsest spacing of 1.7 nm: the grid ends at 2501.2 nm, and the first instrument's value there is extrapolated from its last two (often noisy) edge points. Any SG derivative near that end then carries the artefact.

**Verifier.** np.arange(min_wl, max_wl + spacing, spacing) overshoots max_wl whenever the range is not an exact multiple of the spacing, and resample_to_grid extrapolates linearly. It is used by the GUI equalization path (L46745). The artefact is limited to one edge point.

### R119: The Explore-tab MovingAverage preview zero-pads the spectrum ends, pulling edge values toward zero
`src/spectral_predict/preprocess.py:259` · area preproc-varsel · numerical · finder low → verifier **low / CONFIRMED**

**Problem.** np.convolve(x, ones(w)/w, mode='same') treats values past the ends as zeros. The first and last w//2 points are averaged with zeros, so an absorbance of 1.0 shows as about 0.6 at the edge for w=5. SavgolSmooth (interp mode) and GaussianSmooth (reflect) handle edges properly; this one does not. It is used by the GUI custom preprocessing preview (_compute_custom_spectra).

**Failure scenario.** Explore tab, Smoothing = Moving Average, w=11 on flat 1.0-absorbance spectra: the first and last 5 points plot at 0.55-0.91, showing edge features that are not in the data.

**Verifier.** np.convolve with mode='same' zero-pads the ends. The only caller is the GUI Explore custom preview (L8428), so this is display only.

### R120: y-supervised interference steps (OSC, DOSC, EPO, GLSW) are fitted on all samples, including validation folds, before CV
`src/spectral_predict/search.py:2932` · area preproc-varsel · leakage · finder low → verifier **low / PLAUSIBLE**

**Problem.** The grid path runs prep_pipeline.fit_transform(X_np, y_np) once on the whole calibration set and then cross-validates with skip_spectral_preprocessing=True. For OSC and DOSC, fit uses y to remove X-variance orthogonal to the target, so held-out samples' y shape the features the CV scores. This goes beyond the documented 'per-spectrum preprocessing and autoscale pre-CV are fine' convention, and OSC's own docstring says it is only CV-safe when fitted on training folds. It is reachable only through run_search(interference_settings=...), since the GUI hand-off is disabled.

**Failure scenario.** run_search(..., interference_settings={'osc': {'enabled': True, 'n_components': 2}}) on a small, noisy calibration set. R²cv is inflated because OSC components were computed with the test folds' y. The same pipeline refitted on training data only and scored on a true holdout performs noticeably worse.

**Verifier.** The code does fit y-supervised OSC/DOSC once on all samples before CV (L2932), so test-fold y informs the features. The claimed impact (inflated R2cv) was not shown for OSC, where the leaky fit made R2cv much worse on noise. DOSC showed only a small optimistic bias (about +0.1 R2). This is a methodology issue reachable only through run_search(interference_settings=...), since the GUI hand-off is disabled.

### R121: The variable-selection path edge-masks importances on an axis that was already edge-trimmed, discarding a second window//2 interior wavelengths per side
`src/spectral_predict/search.py:3879` · area preproc-varsel · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** When deriv and window are set and no wavelength restriction is active, X_for_models has already had window//2 columns removed from each end (L3046, _apply_edge_mask_to_data). X_transformed_varsel is that trimmed matrix. L3879 then runs _apply_edge_mask on its importances, zeroing another window//2 columns at each end. Those columns are valid interior wavelengths. Dense selectors can never pick them within top-N, sparse selectors (CARS/SPA/UVE-SPA/VCPA) lose any variables they chose there, and n_method_optimal shrinks accordingly. The one-class path (L6915) does the same.

**Failure scenario.** deriv1, window 31, a CARS subset that includes wavelengths 16-30 positions from either spectral end: those variables get importance 0 and are removed from the top-N subsets and from the method-optimal count, although they are outside the SG boundary zone.

**Verifier.** X_transformed_varsel = X_for_models, which was already trimmed by _apply_edge_mask_to_data (L3046), and L3879 then applies _apply_edge_mask again. Positions window//2 to 2*(window//2) from each original end can never enter varsel subsets, even when the real signal is there. The effect is lost variables, not corrupted results, so low stands.

### R122: Uniform or tied importance fallbacks turn 'top-N' subsets into the N highest-index (longest-wavelength) columns, still labelled as selector output
`src/spectral_predict/search.py:3950` · area preproc-varsel · misleading-output · finder low → verifier **low / CONFIRMED**

**Problem.** top_indices = argsort(importances, kind='stable')[-n_fit:]. With ties, a stable ascending sort puts the highest indices last. Several selectors return np.ones on failure: ipls_selection when every interval has CV R² ≤ 0, spa_selection when all seeds fail, CARS/UVE/iPLS below their minimum feature count, and uve_selection when all scores are 0. search only sets used_uniform_fallback for all-zero arrays, so these fallbacks go ahead as 'top{N}_{method}' subsets, which are just the last N wavelengths. The same tie-break favours long wavelengths among VCPA-IRIV's three score levels (1/0.5/0.25).

**Failure scenario.** A weak target where every iPLS interval's CV R² ≤ 0: ipls_selection returns np.ones, and the rows 'top50_ipls', 'top100_ipls' are the 50/100 longest wavelengths, shown as if iPLS had chosen them.

**Verifier.** ipls_selection clips R2 at 0 and returns np.ones when every interval fails. search only flags an all-zero array, and the stable argsort then takes the highest indices. The subsets go out labelled topN_ipls as if iPLS had chosen them. The other np.ones fallbacks follow the same code path.

### R123: Subset rows' top_vars edge-mask the importance-ordered subset, removing the selector's most important variables from the reported top wavelengths
`src/spectral_predict/search.py:5555` · area preproc-varsel · misleading-output · finder medium → verifier **low / CONFIRMED**

**Problem.** In _run_single_config, X is reduced to X[:, subset_indices]. subset_indices = argsort(importances)[-n:][::-1], so the columns are in descending selector importance. The display-importance block then calls _apply_edge_mask(importances, preprocess_cfg), which zeroes the first and last window//2 positions of that subset array. Those positions are not spectral edges: they are the most and least important selected variables. The same mask also runs on full-model rows whose axis was already trimmed at L3046, so the full row is masked twice. top_vars feeds the GUI best-model summary ('first 5') and the tooltip.

**Failure scenario.** sg1, window 17, 'importance' selector, top-40 subset: the subset's all_vars start 1314, 1286, 1312, 1288 …. Its top_vars start 1318, 1282, 1308 …, and none of the selector's top-8 wavelengths appear, although the full model ranks 1314 first.

**Verifier.** _run_single_config reduces X to X[:, subset_indices], with the indices in descending importance, then applies _apply_edge_mask to the subset importances. The mask zeroes the first and last window//2 positions of the importance-ordered array rather than spectral edges. For subsets with more than 30 vars the selector's top variables vanish from top_vars. This affects display and summary only; the fitted model and all_vars are correct, so I lowered severity from medium to low.

### R124: Wavelength-exclusion interference step drops columns but wavelength labels are not updated, so top_vars/all_vars are wrong and wl filtering crashes
`src/spectral_predict/search.py:2932` · area search-cv · correctness · finder medium → verifier **low / CONFIRMED**

**Problem.** WavelengthExcluder removes columns inside the global prep pipeline, but run_search keeps wavelengths = X.columns (full length) as wavelengths_for_models. n_vars reflects the reduced matrix, while all_vars lists every original wavelength, including excluded ones. top_vars maps importance indices into the unreduced wavelength array, shifting every reported wavelength after the excluded band. With analysis_wl_min/max the full-length wl_mask is applied to the reduced matrix and raises IndexError. The validation rebuild never applies interference at all, because interference is not persisted in the row. Reachable through run_search(interference_settings=...). The GUI currently comments this argument out (gui:30853).

**Failure scenario.** 60 channels at 1000-1590 nm, signal at 1500 nm, exclude '1100-1290': the row has n_vars=40 but 60 all_vars entries, and top_vars[:3]=['1300','1370','1110'] (1110 is an excluded wavelength; the true signal is at 1500). Adding analysis_wl_min=1000, analysis_wl_max=1400 raises 'IndexError: boolean index did not match ... size of axis is 40 but ... 60'.

**Verifier.** The claim reproduces through public run_search(interference_settings=...). n_vars is 40, but all_vars has 60 entries and top_vars include excluded wavelengths. Adding analysis_wl_min/max raises IndexError. I lowered the severity because the GUI passes no interference_settings (gui:30853 is commented out: 'DISABLED: Code stashed (broke R² reproducibility)'), and AGENT_COMPOSITION.md does not mention the argument. Only direct backend callers can reach it.

### R125: Calibration F1/Precision/Recall use 'weighted' averaging while CV uses 'binary'/'macro', so the cal-vs-CV gap is misleading
`src/spectral_predict/search.py:5203` · area search-cv · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** The calibration metrics in _run_single_config use average='weighted' for F1, Precision and Recall. The CV counterparts in _run_single_fold and the repeated-CV pooling use 'binary' for binary tasks and 'macro' otherwise. The F1 vs F1cv (etc.) columns shown side by side therefore measure different quantities. On imbalanced binary data the weighted calibration F1 is dominated by the majority class, while F1cv covers only the positive class. The apparent overfitting gap is largely an artefact of the averaging choice.

**Failure scenario.** Binary data with an 80/20 split, where the model predicts the majority class well and the minority class poorly even in-sample: F1 (cal, weighted) is about 0.8, while F1cv (binary, minority=1) is about 0.3. That reads as a large overfit even if calibration binary F1 is also about 0.3.

**Verifier.** search.py:5203-5205 computes calibration F1/Precision/Recall with average='weighted', while CV uses 'binary'/'macro' (4579; 5055 in the repeated branch). The columns shown side by side measure different quantities. The composite score uses only Accuracycv and F1cv (scoring.py:74), so this affects display and interpretation only, not ranking. Low severity is appropriate.

### R126: XGBoost+class_weight and any booster+regression weighting are CV'd without early stopping, but the row records early_stopping_rounds=40
`src/spectral_predict/search.py:5371` · area search-cv · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** In _run_single_fold, the regression-weighting branch (4392-4413) and the classification sample_weight branch (4419-4469, taken for XGBClassifier because it has no class_weight attribute) both fit the final model without early stopping. The early-stopping branch runs only when sample_weight_train is None. _run_single_config still writes early_stopping_rounds=40 into the row for every XGBoost/LightGBM/CatBoost model. That field exists so 'Model Development can reproduce boosted results', so a refit of such a row applies early stopping the search never used and gets different CV numbers.

**Failure scenario.** Classification with imbalance_method='class_weight' and XGBoost: the search CV fits full n_estimators trees with balanced sample_weight, the row says early_stopping_rounds=40, and the Model Development refit early-stops on each fold. Accuracycv does not reproduce.

**Verifier.** XGBClassifier has no class_weight attribute (verified: hasattr False, while LGBMClassifier's is True). With imbalance_method='class_weight', _run_single_config therefore sets use_sample_weight_for_classification=True (search.py:4842). _run_single_fold sets sample_weight_train='applied', so the early-stopping branch guarded by `if sample_weight_train is None` (4472) never runs. The row still records early_stopping_rounds=40 (5371-5373). GUI refine (gui:41630-41636) calls _fit_with_early_stopping with sample_weight=_es_sample_weight, so a refit early-stops where the search did not, and the CV numbers diverge. Shown by code read plus the attribute check; I did not run the numeric divergence end to end. Low severity is right: this is reproducibility drift, and the search path is actually the less-leaky one.

### R127: Transfer-quality plots fail silently for models built on a region of interest
`spectral_predict_gui_optimized.py:48183` · area transfer-analysis · error-handling · finder low → verifier **low / CONFIRMED**

**Problem.** When ROI is enabled, the model is estimated on the clipped columns (A is roi-by-roi; B, slope and M are ROI-sized). _plot_transfer_quality applies it to the full-width ct_X_satellite_common, which raises a shape mismatch. The exception is caught by 'except Exception: print(...)', so no plots appear. The user still gets the 'built successfully' dialog and no indication that the quality check never ran.

**Failure scenario.** The user enables ROI 1100-1800 nm (350 of 750 points) and builds DS. X @ A with shapes (n, 750) and (350, 350) raises ValueError, the console prints 'Error creating transfer quality plots', and the Section C plot area stays empty.

**Verifier.** With ROI enabled, the build path estimates on clipped arrays (46996-47008), so A, B and T are ROI-sized. _plot_transfer_quality (48181-48200) applies them to the full-width ct_X_satellite_common with no ROI handling, unlike _apply_transfer_with_roi. The resulting shape error is swallowed by 'except Exception: print(...)' at 48371, so the plots silently do not appear. I confirmed this from the code.

### R128: Transfer-quality plot reports resubstitution R², which reads as near-perfect for DS and NS-PFCE
`spectral_predict_gui_optimized.py:48344` · area transfer-analysis · leakage · finder medium → verifier **low / CONFIRMED**

**Problem.** _plot_transfer_quality applies the model to the same paired standardisation spectra it was fitted on (ct_X_satellite_common) and reports R² against ct_X_primary_common. DS is a p-by-p regression fitted on n << p samples, and NS-PFCE is essentially the same with ridge 1e-6. Both interpolate the training pairs, so the plot shows R² close to 1.0000 whatever the true transfer error on new samples. No held-out or cross-validated transfer error is reported anywhere.

**Failure scenario.** 12 transfer standards, 200 wavelengths, DS with lambda=0.01: in-sample RMSE is 0.0019 but RMSE on other samples is 0.0105 (for comparison, NS-PFCE in-sample RMSE is 0.00057). With lambda=1e-6, in-sample RMSE is 2.5e-7 against 0.0105 out of sample. The scatter plot shows R² of about 1.0000, and the user concludes the transfer is essentially perfect.

**Verifier.** The DS build estimates A from all of ct_X_primary_common and ct_X_satellite_common (47028). _plot_transfer_quality then applies it to the same arrays and reports r2_score on the flattened values. That is resubstitution, and in-sample error is about 10x smaller than out-of-sample. However, R² on flattened spectra is dominated by between-wavelength variance, and even raw untransferred data scores about 0.98. This is a display diagnostic that never feeds a model or prediction, so I lowered it to low. There is no leakage into results.

### R129: Regularisation and SVM validation curves exclude the selected alpha or C and ignore the model's other hyperparameters
`src/spectral_predict/diagnostics.py:737` · area transfer-analysis · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** The sweep is np.logspace(log10(base)-2, log10(base)+2, 8). With 8 points over 4 decades the step is 4/7 decade, so base_alpha falls between grid points, and selected_idx marks a point about 1.93 times away from the chosen value. The estimator is built as model_class() with defaults, which drops ElasticNet l1_ratio and max_iter and SVR/SVC kernel, gamma and epsilon (the GUI caller at 39362-39365 does the same for SVM). The curve's 'selected' point is therefore neither the selected hyperparameter value nor the selected model.

**Failure scenario.** The selected model is ElasticNet(alpha=0.01, l1_ratio=0.9). The curve evaluates ElasticNet(l1_ratio=0.5) at alphas such as 0.0052 and 0.0193, and the marker sits at 0.0193 or 0.0052, so the plotted CV error at the 'selected' point does not match the model's reported CV error.

**Verifier.** compute_regularization_validation_curve uses np.logspace(log10(base)-2, log10(base)+2, 8). The exponent offsets are ±0.286, ±0.857, and so on, so base_alpha is never evaluated and the nearest point is 10^0.286 ≈ 1.93x away. The estimator is model_class() with default settings, which drops l1_ratio, max_iter and the kernel/gamma/epsilon settings. The GUI SVM branch (39360-39366) does the same. Only the diagnostic display is affected.

### R130: The common grid can end beyond the overlap, and resample_to_grid extrapolates silently
`src/spectral_predict/equalization.py:48` · area transfer-analysis · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** choose_common_grid uses np.arange(min_wl, max_wl + spacing, spacing), so the last point can lie up to one spacing beyond the overlap maximum. The grid also comes from the registered profiles' wavelengths, not the loaded paired data. resample_to_grid then fills anything outside a spectrum's range with linear extrapolation (fill_value='extrapolate') without warning. Transfer matrices are fitted on, and applied to, these invented edge channels.

**Failure scenario.** Overlap 1000-2500 nm with coarsest spacing 3.5 nm: the grid's last point is 2501.5 nm, extrapolated for both instruments. If the loaded spectra are narrower than the registered profile (for example trimmed files), whole bands at the edges are extrapolated and enter DS/PDS estimation and prediction.

**Verifier.** np.arange(min_wl, max_wl + spacing, spacing) can emit a final point up to one spacing beyond the overlap maximum. resample_to_grid then extrapolates it without warning. The GUI (46745-46752) builds the grid from registered profiles and resamples the loaded data onto it. The overshoot is at most one coarse step, so the practical impact is small unless the loaded data are narrower than the profile.

### R131: Library save crashes when one sample ID equals 'wl_' plus another sample ID
`src/spectral_predict/library_search.py:525` · area transfer-analysis · persistence · finder low → verifier **low / CONFIRMED**

**Problem.** Spectra and wavelength arrays are passed together as keyword arguments: **spectra_dict and **{f'wl_{k}': ...}. A sample named 'wl_X' alongside a sample 'X' (or a sample named '__grid__') produces a duplicate keyword, and the call raises TypeError. By then add_spectrum has already put the entry in memory, so memory and disk diverge and every later auto-save fails until the process ends. All entries added since the last successful save are lost.

**Failure scenario.** The user adds a CSV whose sample IDs include 'A' and 'wl_A'. The first add succeeds. The second raises 'savez_compressed() got multiple values for keyword argument wl_A' from _save, and after that every add_spectrum or remove_spectrum auto-save fails, so on restart the library is missing everything added since the last good save.

**Verifier.** np.savez_compressed(**spectra_dict, **{'wl_'+k: ...}) collides when sample IDs 'X' and 'wl_X' coexist. The entry is added to memory before _save, and every later auto-save raises. It is worse than reported: the JSON metadata is written before the npz. On reload, 'wl_A' is restored with A's wavelength array as its spectrum, a silent corruption, and entries added after the failure (B) are dropped. This requires unusual sample naming, so I kept it at low.

### R132: Bone FTIR ratios are reported as 0.0 instead of NaN when PO4 extraction fails or is non-positive
`src/spectral_predict/peak_calculator.py:1725` · area transfer-analysis · correctness · finder low → verifier **low / CONFIRMED**

**Problem.** When the PO4 v3 block fails, PO4_intensity is NaN. 'result.get(...) or 0' keeps NaN because NaN is truthy, 'po4 > 0' is then False, and Am_P and C_P are set to 0.0. OrgInorg is also set to 0.0 when po4 + co3 is NaN or non-positive. A real-looking ratio of 0.0 (for example Am/P = 0, meaning no collagen) is written in place of 'not measurable', which contradicts the docstring ('Values are np.nan when a feature could not be extracted').

**Failure scenario.** A spectrum whose baseline-corrected PO4 peak comes out non-positive (a poor trough fit on a strongly sloped ATR baseline), or whose PO4 extraction raises, gets PO4_intensity NaN or ≤ 0 but Am_P = 0.0, C_P = 0.0 and OrgInorg = 0.0 in the exported table. Those values then enter diagenesis statistics as genuine zero-collagen, zero-carbonate samples.

**Verifier.** The 'or 0' trick keeps NaN, and the checks 'po4 > 0' and '(po4+co3) > 0' then fall back to 0.0 instead of NaN, which contradicts the docstring. In my repro, with no PO4 region, PO4_intensity came out as 0.0 rather than NaN. Am_P and C_P were 0.0, and OrgInorg became am/co3 = 1.67, also misleading. So the 'not measurable' case is exported as real-looking zeros.

### R133: HQI and derivative-correlation metrics use r², so inverted or anti-correlated spectra score as perfect matches
`src/spectral_predict/similarity_metrics.py:53` · area transfer-analysis · correctness · finder medium → verifier **low / CONFIRMED**

**Problem.** hit_quality_index returns r**2. A spectrum that is perfectly anti-correlated with the query (r = -1) scores 1.0, identical to a true match. deriv1_corr and deriv2_corr reuse it, so derivative features of opposite sign (a band against a trough) also score 1.0. Library search ranks these as top hits, and _check_spectral_duplicate (library_search.py:298) uses the same HQI, so an anti-correlated spectrum is rejected as a duplicate at the 0.9999 threshold.

**Failure scenario.** A library entry holds a reflectance spectrum s, and the query is a transmission-like or inverted spectrum 1 - s, or a spectrum whose derivative is the negative of the reference's. compute_similarity(s, 1-s, 'hqi') = 1.0 and compute_similarity(s, 1-s, 'deriv1_corr') = 1.0, so the search reports a perfect match. Adding 1 - s to the library is refused as 'very similar (HQI=1.0000)'.

**Verifier.** hit_quality_index returns r**2 and deriv1_corr/deriv2_corr reuse it, so inverted spectra score 1.0 and the duplicate check would treat them as duplicates. However, the docstring states 'HQI = r²' deliberately, and a squared-correlation HQI is the standard, sign-insensitive library-search convention. It only matters for a library that mixes inverted or transmission-like and reflectance/absorbance spectra, which is an edge case. I lowered it to low.

## Refuted

- 'Import transfer model to registry' always fails: it passes the .json path where a path prefix is expected (`spectral_predict_gui_optimized.py:46517`): The path bug is real inside _import_model_to_registry: it passes the .json path to load_transfer_model, which appends '.json'. But the method is dead code. A grep of the whole repo finds only its definition at GUI line 46499, with no button command, menu entry, binding or other reference. Users cannot reach it.

## Coverage notes (what each finder did and did not read)

- **search-cv**: Read in full: cv_utils.py, scoring.py, analysis_subset.py, sample_selection.py, search_controller.py. In search.py I read run_search from its setup through config building and the main grid loop (lines 1358-3530), the ranking and validation tail (4150-4333), _run_single_fold, _run_single_config, compute_validation_metrics_for_top_models and _apply_class_weight_discriminator_for_rebuilt_model, plus the one-class preprocessing, CV and result-row block (6300-6550). I skimmed the varsel method branches (3530-4150) and did not review them in detail. Not reviewed: the rest of the one-class varsel loop, the multiclass SIMCA functions (7218 to end), _rebuild_model_from_row, and _multiclass_holdout_metrics. I left out the areas the project records as deliberate: varsel, region and importance selection on full calibration data, and per-spectrum or autoscale preprocessing before CV. I checked OSC fitted with y on full data before CV on a noise target and saw no inflation, so I dropped it. Eight findings were confirmed with scratch scripts in scratchpad/review/search-cv: es2.py, loo_cls.py, wlfmt.py, labels.py, ga_bl.py, ks.py, wlx.py, and the parity check behind the early-stopping claim. The validation-rebuild imbalance gap and the calibration-vs-CV averaging mismatch come from reading the code only, with no script run. A slower background early-stopping script (es.py, XGBoost) may still be running and can be ignored. No findings were dropped for the 15-item cap.
- **bayesian**: Read in full: unified_bayesian.py (objective, fingerprinting, persistence/resume/migration, convert_study_to_dataframe), search_spaces.py, phase2_rescore.py, tpe_preprocessing_discovery.py, bayesian_utils.py, bayesian_config.py (not imported anywhere, dead code), extra_axes_advisory.py, run_gui_settings.py, most of run_state.py (lines 100-660). Followed call chains into cv_utils.cross_val_predict_with_early_stopping, regions.create_region_subsets, variable_selection._cap_top_n/uve_selection, models.estimator_params_from_row, and the two GUI run_unified_bayesian call sites. Checked docs for deliberate decisions: full-data varsel and autoscale before CV are deliberate chemometrics conventions and are not reported; TPE discovery exploring every baseline method with default params is by design. Reproduced findings 1-4 with scripts under scratchpad/review/bayesian/. Finding 1 lives in shared cv_utils and also affects grid search; it is reported here because the Bayesian objective optimises on it. Things I looked at and found sound: study-name hash components, data-fingerprint resume gating, fit fingerprint and rehydration, extra-axes resolution and application (PLS-DA lr_C is routed correctly), Pipeline-prefixed Params stripping. Minor things not reported: training_config hardcodes random_state 42 (harmless, since the GUI also uses 42); UVE returns raw reliability without applying its noise threshold, so it is ranking-only; GUI n_top_regions is not passed to the Bayesian path. Not covered in depth: run_state.py below line 660 (discard/cleanup), and one-class run_one_class_cv internals. No findings were dropped to fit the 15 cap.
- **nsga-ga**: I read all of nsga2_search.py (4156 lines) and ga_pls.py, and nearly all of ga_preprocessing.py and ga_lightgbm.py. The only parts of ga_lightgbm.py I skipped were its public-wrapper tail and docstrings. I followed the call chains into cv_utils (the early-stopping CV loop and _fit_with_early_stopping), models.get_model / build_model / get_feature_importances / PLSTransformer, preprocess.SavgolDerivative and preprocessing_config_from_row, and the pymoo 0.6.2 Mutation/PM/Algorithm seeding. I also read the relevant parts of search.py: the GA varsel call sites, the exhaustive-config consumer and the stable-argsort top-N selection. In the GUI I read the NSGA-II invocation (about line 30680) and the Tab 7 deriv/polyorder resolution.

I checked the following with scratch scripts under scratchpad/review/nsga-ga:
- decode_solution stores n_components=15 for a 12-wavelength subset.
- top_vars comes back in index order.
- Guided NSGA-II is not reproducible with random_state=42, while unguided is.
- objectives['n_wavelengths'] reports 4.27 for 16 wavelengths.
- A penalty row reached the front and the results table as Accuracycv=-1e10.
- find_knee_point moves the knee when a 1e10 row is present.
- The stored Params differ from the fitness model for LightGBM and CatBoost.
- The GA-PLS median-threshold accuracies are 0.70 and 0.51.
- SmartMutation changed zero control genes in 2000 offspring.

I followed the project's documented decisions and did not report: variable selection on the full calibration set (deliberate chemometrics convention), per-spectrum preprocessing outside the folds, or any of the known issues listed in the brief. I did not report NaN-target handling, because I could not confirm that NaN y reaches NSGA-II. I also dropped a classification-only case where Lasso/ElasticNet/SVR rows store parameters for an estimator that build_model cannot rebuild, since it overlaps the known SVM/SVR issue.
- **models-ensemble**: I read these files in full: models.py, ensemble.py, neural_boosted.py, imbalance.py (lines 1-1179), y_transform.py, bias_correction.py, fit_overlay.py, model_config.py and model_registry.py. I only grep-skimmed ensemble_viz.py; it is plotting code and I checked it for metric computation only. I followed these call chains into the GUI: _train_ensembles (24632-25770), _reconstruct_models_from_results, the Model Development CV loop, final fit and save (41500-42210), the bias-correction UI (37597-37700) and the imbalance params builder. In the backend I followed search._run_single_fold, search._needs_resampling_pipeline, the unified_bayesian suggest_model_params / build_model usage, and model_io.predict_with_model (lines 700-826). I confirmed these findings with scratch scripts run on .venv314 (pandas 3.0.5): ensemble CV leakage (true CV R2 0.23 vs reported R2CV 0.875), the string-index KeyError, the NeuralBoosted prior drop (test accuracy 0.46 vs 0.85 for the majority baseline), the TTR save path (R2 0.962 direct vs -2.206 through preprocessor plus model), the SVR/SVM grid ignoring user lists, the resampler clone dropping k_neighbors, the binning weight outlier and the failed-base-model weights. Scripts are in the scratchpad review/models-ensemble folder. I did not deeply verify NeuralBoosted early-stopping truncation (all trailing non-improving learners are kept, which looks minor). I did not audit the nsga2_search._build_model hyperparameter mapping or the Tab 7 refit paths (several are already known issues). create_auto_ensembles/ClassSpecialistEnsemble are only used by tests, so I rated their issue low. No findings were dropped for the cap.
- **preproc-varsel**: Read in full: preprocess.py, preprocessing_wrapper.py, preprocessing_discovery.py, variable_selection.py (all except ipls_backward, mc_sipls, mwpls, ga_pls/ga_lightgbm bodies, which I skimmed), wavelength_selection.py, baseline.py, baseline_advanced.py, regions.py and equalization.py. Followed call chains into search.py (the grid preprocess/trim/varsel loop at L2880-4030, _run_single_config L4678-4905 and the importance/top_vars block ~L5530-5580, the smart/TPE config builders, and the varsel cache key), unified_bayesian.apply_preprocessing, interference.py (WavelengthExcluder, OSC), ensemble.py cloning, model_io wavelength matching, and the GUI Tab 7 refit (L40200-41480), the ensemble rebuild (~L24940-25080) and the Explore preview (~L8410).

Checked empirically with scratch scripts (in the scratchpad review/preproc-varsel folder): clone() drops BaselineAdvanced params; the wavelength-exclusion label shift through run_search; subset top_vars losing the selector's top variables; the CARS subset-size floor; one-class discovery with no outliers; Bayesian skipping 'advanced'; the 0.5 vs 0.01 index rule.

Tested and dropped: UVE with unit-variance noise against derivative-scale X (the selection did not change), and CARS weight underflow (did not happen).

Left out on purpose: varsel/autoscale/SNV run on the full calibration set before CV (a documented chemometrics-convention decision), SNV near-zero std (T-26 WONT_FIX), and the Basic-discovery LightGBM proxy (documented as out of scope).

Low-value items not listed: PreprocessorConfig always applies SNV before the derivative, and defaults polyorder to 2 even for deriv 2 (only tests use it). The discovery importance_method is effectively a no-op (selected_wavelengths is metadata only, and cars_tree ignores task_type). _quick_evaluate's PLS-RMSE fallback would invert the higher-is-better ranking for classification and one-class if LightGBM throws. EPOWithLibrary is a local class, so a pipeline using it cannot be pickled (API-only).
- **one-class**: I read all of contamination.py and simca.py (including the metrics, Wold selection and novelty code), outlier_detection.py, contaminant_analysis.py and interference.py. I followed the call chains into search.run_one_class_search (preprocessing, variable selection, how each result row is written), the one-class Bayesian objective in unified_bayesian.py (lines 690-778 and 1440-1560), cv_utils.build_cv_splitter, io.py (wavelength rounding and unit conversion), and the GUI ranges that call these modules: contaminant correction and export (~58870, 60025-60270), the interference-removal tab (60648-60757), outlier detection (21612-21660), the Model Development one-class refit (40740-40995), and target loading (17545).

I ran scratch checks in the review/one-class scratch folder for findings 1, 2, 3, 4, 6, 7, 8 and 9, and each one reproduced. I also checked and dropped these:
- scipy's chi2 'mm' fit matches the manual method-of-moments fit.
- Pooling AUC across folds under kfold/LOO differs from the mean of per-fold AUCs by less than 0.015.
- The non-SIMCA engines OCSVM, IsolationForest and LOF calibrate close to the nominal 5%.
- IsolationForest score_samples has the right sign.
- The multi-class and novelty metric formulas are correct.

I left these out as deliberate or already known:
- Autoscale and variable selection before CV (chemometrics convention).
- Supervised variable selection on y_oc in one-class mode.
- Outliers counted k times under plain K-fold (TODO in the code).
- DD-SIMCA over-rejecting at small n in general (SESSION_LOG_ARCHIVE). Finding 7 is only the deterministic n-1 clamp case.
- Everything on the known-issues list.

The %g wavelength-format problem (finding 4) also affects the regression/classification validation path (search.py:1116). I checked it only in the one-class path. No findings were dropped for the 15 cap.
- **io-persistence**: Read in full: model_io.py (save/load/predict/predict_with_uncertainty/ensemble), export_bundle.py, r_code_generator.py, templates/preprocessing.py, readers/opus_reader.py, and the merge paths of data_management.py. I read these sections of io.py: the CSV and CSV-directory readers, reference-file reading, align_xy, the ASD directory and single-file ASCII readers, the SPC directory reader, the combined-CSV path (identify_wavelength_columns and the ID/y auto-detection), detect_format/read_spectra dispatch, the Excel reader, both read_ascii_spectra definitions, read_jcamp_file and the OPUS wrappers. Of code_generator.py I read the constructor, preprocessing, variable-selection and model-rendering sections. I skimmed omnic_reader.py. I followed the save and export call chains into the GUI: _run_refined_model Path A/B/GA/TTR/one-class construction (about lines 40011-42240), _save_refined_model, _export_for_publication, the prediction-tab loader and predictor, and the calibration-transfer predict call.

Not reviewed: agilent_reader.py, perkinelmer_reader.py, asd_native.py, asd_r_bridge.py, io.py write_* functions, read_combined_excel beyond its shared helpers, templates/models.py, templates/validation.py, templates/variable_selection.py, templates/visualization.py, the code_generator CV/final-fit renderers, and the imbalance/one-class export templates.

Dynamic checks: I confirmed findings 1, 2, 3, 4, 5, 6, 7, 8 and 9 with scripts under scratchpad\review\io-persistence\ (ttr_check.py, opus_check.py with a mocked brukeropus, bundle_check.py, codegen_check.py, le_check.py, ascii_check.py, merge_check.py, combined_check.py). Findings 10 to 13 come from reading the code only.

Dropped: a suspected applicability-domain mismatch in the Path B raw X_train. use_full_spectrum_preprocessing is hard-coded True at GUI:40242, so Path B cannot run.

Deliberate or known, not reported: early-stopping models' final fit without early stopping (SESSION_LOG_ARCHIVE:2881); the export's y_transform gap, previously deferred in the archive, is folded into finding 4; the OPUS keep-last-duplicate-stem behaviour is documented. I saw no need to exceed 15 findings.
- **transfer-analysis**: I read all of calibration_transfer.py, instrument_profiles.py, similarity_metrics.py, library_search.py, diagnostics.py, run_logging.py, progress_monitor.py, report.py and resource_paths.py. In peak_calculator.py I read the calculation, baseline and bone-FTIR functions (lines 1030 to 1762) and skimmed the preset tables. interactive.py and interactive_gui.py are imported nowhere in the repo (grep over *.py, spec and toml files), and ProgressMonitor is never instantiated. I treated those three as dead code and did not audit them.

I followed the GUI call chains for building, plotting, applying, saving and loading transfer models (spectral_predict_gui_optimized.py lines 46500 to 48372, 50300 to 50354, 50920 to 50960, 51480 to 51510, 53771 to 53935), the leverage and complexity-curve callers (38790 to 38900, 39310 to 39370, 41568, 42213 to 42251), and the report call site (31040 to 31172). I also read equalization.choose_common_grid.

I checked these numerically with scratch scripts in the review folder:
- the JYPLS-inv degradation (spectral RMSE and downstream prediction RMSE);
- DS overfitting on transfer standards;
- save/load round trips for all methods;
- leverage saturation when p is at least n;
- HQI and derivative-HQI returning 1.0 for inverted spectra;
- report.py failing on a 'δ13C' target (UnicodeEncodeError under cp1252, Python 3.14.7, utf8_mode 0).

A second report.py failure (AttributeError for one-class results) is real but unreachable, because the one-class GUI path returns before the report is written, so I dropped it. I also dropped these as too minor or intended:
- reversed baseline regions in np.interp (small effect in testing);
- CTAI/NS-PFCE losing scalar params such as n_components on save (display only);
- NS-PFCE showing MSE labelled 'Final RMSE';
- PDS having no intercept (it still worked out of sample);
- one per-process run log shared across GUI runs;
- jackknife intervals (the function is unused).

No findings were dropped because of the 15 cap.
- **gui-1**: What I read closely: _load_and_plot_data (every reader branch, append/merge, the alignment guard), _apply_wavelength_filter/_update_wavelengths, the x-unit and data-type conversions, _on_target_column_changed, the validation-set creators (KS/SPXY/Random/Stratified/Manual/reset), every exclusion entry point (Tab 1 click, Explore pick/toggle/restore, sample-set exclude/keep-only, Quality Check PCA right-click, outlier table mark/unmark), the outlier detection flow plus outlier_detection.generate_outlier_report, both predictor-screening paths (correlation/VIP/RF, main and Explore), the Explore colour map and PCA, the baseline and manual-baseline "replace working data" paths, the peak-calculator scope and data helpers, Data Management use-for-analysis/merge/filter/trim, _merge_spectral_data/_merge_metadata_frames, _check_for_incomplete_run and _export_preprocessed_csv. I followed call chains into io.align_xy, io.read_combined_csv, search.compute_validation_metrics_for_top_models (positional col_indices), models.compute_vip, and the worker's row filtering and X/y guard (gui ~29287-29690), because correctness depends on them. Checks run under scratchpad/review/gui-1: check1.py shows that on a synthetic nuisance+signal dataset the GUI VIP gives nuisance wavelengths VIP 1.14, above the plotted VIP=1 threshold, where models.compute_vip gives 0.52. It also shows that generate_outlier_report given a numpy array returns positional Sample_Index. check2.py shows read_combined_csv with a numeric ID column gives a str index, and that the click-handler coercion then excludes 0 rows and resolves the wrong sample. Skimmed only: theme/style/sidebar code, the widget builders for tabs 4A-7D (about 11024-16300), peak-calculator dialog internals, the custom-preprocessing subtab controls, imbalance UI helpers and tooltips. _run_analysis (24305+) is outside my range; I read it only as call-chain context, and its one-class early-return UI leak is left to that area's reviewer. No findings were dropped for the 15 cap. I left out a few low-value items: sample_sets are not reset on reload or row delete (positional 'Set' colouring and the peak-calc scope mask can crash); the manual-baseline CubicSpline raises on duplicate anchors; the outlier report goes stale after the data changes.
- **gui-2**: Read in full: _run_analysis (24305-24631), the resume launch gate and its helpers (26373-27417, including _confirm_resume_before_launch, the validation/model/trial reconciliation and _complete_run_state_after_search), the _run_analysis_thread setup, row filtering, one-class, multiclass, Bayesian and NSGA-II branches and the post-search tail (27425-27760, 29285-30000, 30000-30830, 30997-31188). Also read the progress callbacks, results table population/sort/filter/click/double-click, rerank and overfit tagging, the multiclass run-selected and save code, results export, data-viewer edit/revert, _load_model_for_refinement, _validate_training_configuration, the learning-curve and complexity-curve code, and all of _run_refined_model_thread (39659-42290). I followed calls into unified_bayesian.py (apply_preprocessing, convert_study_to_dataframe, suggest_model_params), scoring.compute_composite_score, run_gui_settings, analysis_subset.check_one_class_inlier_guard, diagnostics.compute_learning_curve, y_transform and cv_utils.build_cv_splitter. Three findings were confirmed with scratch scripts: the Y-transform AttributeError, the no-op 'advanced' baseline, and the integer-label inlier guard. Skimmed only: the grid-path hyperparameter collection (27760-28900; it reads live Tk state, which is a known issue), _reconstruct_models_from_results, _train_ensembles and the ensemble tab (24632-25775), the one-class override collectors, the filter-control builders, _compute_expert_choices, SHAP, and the plotting and diagnostics helpers (37272-39300). Ensemble reconstruction of Bayesian rows was not verified. Not reported: known issues from the brief, and live-state reads on post-search paths (last_training_config, validation metrics, validation indices attached at double-click), since those fall under the known 'post-search paths read live Tk state' item. Nothing was dropped for the 15-finding cap.
- **gui-3**: I read these parts of spectral_predict_gui_optimized.py in full: the Tab 7 refinement worker from about line 40225 to 42300 (the one-class path, Path A, the GA path, the final pipeline split and refined_* state), _save_refined_model, _export_for_publication, the wavelength spec format/parse helpers, all of Tab 8 (model and data loading, _run_predictions, consensus, the display, uncertainty, stats and plot functions, export), the CT model loaders, _load_and_predict_ct, _file_equalize_batch, the directory loaders, CT Mode A (load, convert, _run_prediction_workflow, export), CT Mode B (load, convert, transform, use-as-working-data), all of Tab 9 (model and data loading, data-type handling, live monitoring, the transfer chain, _run_comparison, display, export), and the Tab 12 library search and compare functions. I followed these calls into other modules: model_io (save_model, predict_with_model, predict_with_uncertainty, _select_wavelengths_from_dataframe, save_ensemble/load_ensemble), calibration_transfer.resample_to_grid, library_search._align_to_grid and search, bias_correction.apply_correction, io.list_asd_files, io.read_asd_dir and io.detect_spectral_data_type, and code_generator's embedded-data handling (that one checked out as deliberate). I confirmed four things with scratch scripts: the 0.5 vs 0.01 tolerance mismatch, the sklearn pos_label failure on string labels, and the two filename sort-order mismatches (glob.glob vs Path.glob on Windows). I only skimmed or skipped: _build_ct_transfer_model, _equalize_and_export, the CT plotting helpers, Tab 11 interference removal (57141+) and all of Tab 13 contaminant analysis (57593 to end). I did not re-check ensemble prediction internals in model_io beyond the GUI boundary. The line-count tool reported 53066 lines, but real line numbers run to about 61700, so the tabs past 57000 were not reviewed. I dropped three lower-severity findings to stay within the 15 limit: (a) _format_wavelengths_as_spec rounds range endpoints to .1f, and _parse_wavelength_spec then matches ranges exactly, so a full-spectrum model reloaded into Tab 7 on a grid that is not a multiple of 0.1 silently loses its first and last wavelengths; (b) _load_and_predict_ct exports its predictions as Sample_1..N with no filenames; (c) the Tab 9 Excel summary picks the primary column by substring (`primary_filename in col`), so an auxiliary model whose filename contains the primary's name also lands in the Summary sheet. I found no deliberate-decision notes for any of the reported items in PROJECT_STATUS.md or SESSION_LOG.md. The ±0.5 nm matching in finding 1 is flagged as a bug elsewhere in the same file (lines 2071, 2174, 2343), but these two sites were never fixed.
