"""
Tab 9 Plotting Functions for Calibration Transfer
Agent 2 Implementation
"""

def _plot_transfer_quality(self, method):
    """Plot transfer quality diagnostics for Section C.

    Shows:
    1. Transfer Quality Plot (3 subplots): Master, Slave before, Slave after
    2. Transfer Scatter Plot: Master vs Transferred with R²
    """
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_transfer_plot_frame.winfo_children():
            widget.destroy()

        # Apply transfer to get transferred spectra
        if method == 'ds':
            A = self.ct_transfer_model.params['A']
            X_transferred = apply_ds(self.ct_X_slave_common, A)
        elif method == 'pds':
            B = self.ct_transfer_model.params['B']
            window = self.ct_transfer_model.params['window']
            X_transferred = apply_pds(self.ct_X_slave_common, B, window)

        # === Plot 1: Transfer Quality Plot (3 subplots) ===
        fig1 = Figure(figsize=(12, 4))

        # Subplot 1: Master spectra
        ax1 = fig1.add_subplot(131)
        master_mean = np.mean(self.ct_X_master_common, axis=0)
        master_std = np.std(self.ct_X_master_common, axis=0)
        ax1.plot(self.ct_wavelengths_common, master_mean, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(self.ct_wavelengths_common,
                       master_mean - master_std,
                       master_mean + master_std,
                       alpha=0.3, color='b', label='±1 Std')
        ax1.set_xlabel('Wavelength (nm)', fontsize=10)
        ax1.set_ylabel('Reflectance', fontsize=10)
        ax1.set_title('Master Spectra', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Slave before transfer
        ax2 = fig1.add_subplot(132)
        slave_mean = np.mean(self.ct_X_slave_common, axis=0)
        slave_std = np.std(self.ct_X_slave_common, axis=0)
        ax2.plot(self.ct_wavelengths_common, slave_mean, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(self.ct_wavelengths_common,
                       slave_mean - slave_std,
                       slave_mean + slave_std,
                       alpha=0.3, color='r', label='±1 Std')
        ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        ax2.set_ylabel('Reflectance', fontsize=10)
        ax2.set_title('Slave Before Transfer', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Slave after transfer
        ax3 = fig1.add_subplot(133)
        trans_mean = np.mean(X_transferred, axis=0)
        trans_std = np.std(X_transferred, axis=0)
        ax3.plot(self.ct_wavelengths_common, trans_mean, 'g-', linewidth=2, label='Mean')
        ax3.fill_between(self.ct_wavelengths_common,
                       trans_mean - trans_std,
                       trans_mean + trans_std,
                       alpha=0.3, color='g', label='±1 Std')
        ax3.set_xlabel('Wavelength (nm)', fontsize=10)
        ax3.set_ylabel('Reflectance', fontsize=10)
        ax3.set_title('Slave After Transfer', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        fig1.tight_layout()

        # Embed plot 1
        canvas1 = FigureCanvasTkAgg(fig1, self.ct_transfer_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Add export button for plot 1
        self._add_plot_export_button(self.ct_transfer_plot_frame, fig1, "transfer_quality")

        # === Plot 2: Transfer Scatter Plot ===
        fig2 = Figure(figsize=(7, 6))
        ax = fig2.add_subplot(111)

        # Flatten arrays for scatter plot
        master_flat = self.ct_X_master_common.ravel()
        transferred_flat = X_transferred.ravel()

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(master_flat, transferred_flat)

        # Scatter plot with alpha for density
        ax.scatter(master_flat, transferred_flat, alpha=0.3, s=10, edgecolors='none')

        # Add 1:1 line
        min_val = min(master_flat.min(), transferred_flat.min())
        max_val = max(master_flat.max(), transferred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

        ax.set_xlabel('Master Spectra Values', fontsize=11)
        ax.set_ylabel('Transferred Slave Values', fontsize=11)
        ax.set_title(f'Transfer Quality Scatter Plot (R² = {r2:.4f})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig2.tight_layout()

        # Embed plot 2
        canvas2 = FigureCanvasTkAgg(fig2, self.ct_transfer_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Add export button for plot 2
        self._add_plot_export_button(self.ct_transfer_plot_frame, fig2, "transfer_scatter")

    except Exception as e:
        print(f"Error creating transfer quality plots: {str(e)}")


def _plot_equalization_quality(self, instruments_data, equalized_data, common_grid):
    """Plot equalization quality diagnostics for Section D.

    Shows:
    1. Multi-Instrument Overlay Plot (2 subplots): Before and After
    2. Wavelength Grid Comparison

    Parameters
    ----------
    instruments_data : dict
        Dictionary of {instrument_id: (wavelengths, X)} before equalization
    equalized_data : dict
        Dictionary of {instrument_id: X_equalized} after equalization
    common_grid : array
        Common wavelength grid
    """
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_equalize_plot_frame.winfo_children():
            widget.destroy()

        # === Plot 1: Multi-Instrument Overlay (2 subplots) ===
        fig1 = Figure(figsize=(12, 5))

        # Subplot 1: Before equalization
        ax1 = fig1.add_subplot(121)
        colors = plt.cm.tab10(np.linspace(0, 1, len(instruments_data)))
        for idx, (inst_id, (wl, X)) in enumerate(instruments_data.items()):
            # Plot mean spectrum for each instrument
            mean_spectrum = np.mean(X, axis=0)
            ax1.plot(wl, mean_spectrum, linewidth=2, label=inst_id, color=colors[idx])

        ax1.set_xlabel('Wavelength (nm)', fontsize=10)
        ax1.set_ylabel('Reflectance', fontsize=10)
        ax1.set_title('Before Equalization\n(Different Wavelength Grids)',
                     fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: After equalization
        ax2 = fig1.add_subplot(122)
        for idx, (inst_id, X_eq) in enumerate(equalized_data.items()):
            mean_spectrum = np.mean(X_eq, axis=0)
            ax2.plot(common_grid, mean_spectrum, linewidth=2, label=inst_id, color=colors[idx])

        ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        ax2.set_ylabel('Reflectance', fontsize=10)
        ax2.set_title('After Equalization\n(Common Wavelength Grid)',
                     fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)

        fig1.tight_layout()

        # Embed plot 1
        canvas1 = FigureCanvasTkAgg(fig1, self.ct_equalize_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Add export button
        self._add_plot_export_button(self.ct_equalize_plot_frame, fig1, "equalization_overlay")

        # === Plot 2: Wavelength Grid Comparison ===
        fig2 = Figure(figsize=(10, 4))
        ax = fig2.add_subplot(111)

        # Prepare data for bar chart
        labels = []
        min_wls = []
        max_wls = []

        for inst_id, (wl, _) in instruments_data.items():
            labels.append(inst_id)
            min_wls.append(wl.min())
            max_wls.append(wl.max())

        # Add common grid
        labels.append('Common Grid')
        min_wls.append(common_grid.min())
        max_wls.append(common_grid.max())

        # Create bar chart
        y_pos = np.arange(len(labels))
        widths = np.array(max_wls) - np.array(min_wls)

        bars = ax.barh(y_pos, widths, left=min_wls, height=0.6)

        # Highlight common grid
        bars[-1].set_color('red')
        bars[-1].set_alpha(0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_title('Wavelength Range Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add wavelength range annotations
        for i, (label, min_wl, max_wl) in enumerate(zip(labels, min_wls, max_wls)):
            ax.text(min_wl + (max_wl - min_wl)/2, i,
                   f'{min_wl:.0f}-{max_wl:.0f} nm',
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        fig2.tight_layout()

        # Embed plot 2
        canvas2 = FigureCanvasTkAgg(fig2, self.ct_equalize_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Add export button
        self._add_plot_export_button(self.ct_equalize_plot_frame, fig2, "wavelength_grid_comparison")

    except Exception as e:
        print(f"Error creating equalization plots: {str(e)}")


def _plot_ct_predictions(self, y_pred):
    """Plot prediction results for Section E.

    Shows:
    1. Prediction Distribution Histogram
    2. Prediction Results Plot (scatter with line)

    Parameters
    ----------
    y_pred : array
        Predicted values
    """
    if not HAS_MATPLOTLIB:
        return

    try:
        # Clear previous plots
        for widget in self.ct_prediction_plot_frame.winfo_children():
            widget.destroy()

        # Calculate statistics
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)

        # === Plot 1: Prediction Distribution Histogram ===
        fig1 = Figure(figsize=(8, 5))
        ax = fig1.add_subplot(111)

        # Histogram
        n, bins, patches = ax.hist(y_pred, bins=20, alpha=0.7, color='steelblue',
                                  edgecolor='black', linewidth=1.2)

        # Add mean line
        ax.axvline(mean_pred, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_pred:.3f}')

        # Add ±1 std lines
        ax.axvline(mean_pred - std_pred, color='orange', linestyle=':', linewidth=2,
                  label=f'Mean ± Std')
        ax.axvline(mean_pred + std_pred, color='orange', linestyle=':', linewidth=2)

        ax.set_xlabel('Predicted Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Prediction Distribution\n(Mean={mean_pred:.3f}, Std={std_pred:.3f})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        fig1.tight_layout()

        # Embed plot 1
        canvas1 = FigureCanvasTkAgg(fig1, self.ct_prediction_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Add export button
        self._add_plot_export_button(self.ct_prediction_plot_frame, fig1, "prediction_distribution")

        # === Plot 2: Prediction Results Plot ===
        fig2 = Figure(figsize=(10, 5))
        ax = fig2.add_subplot(111)

        # Sample indices
        sample_indices = np.arange(len(y_pred))

        # Scatter plot
        ax.scatter(sample_indices, y_pred, alpha=0.6, s=50,
                  edgecolors='black', linewidths=0.5, c='steelblue', label='Predictions')

        # Connecting line
        ax.plot(sample_indices, y_pred, 'b-', alpha=0.3, linewidth=1)

        # Mean line
        ax.axhline(mean_pred, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_pred:.3f}')

        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Predicted Value', fontsize=11)
        ax.set_title('Calibration Transfer Predictions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig2.tight_layout()

        # Embed plot 2
        canvas2 = FigureCanvasTkAgg(fig2, self.ct_prediction_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Add export button
        self._add_plot_export_button(self.ct_prediction_plot_frame, fig2, "prediction_results")

    except Exception as e:
        print(f"Error creating prediction plots: {str(e)}")
