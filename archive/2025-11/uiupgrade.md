# UI Upgrade Plan: Add Theme System to Combined-Format Branch

## Overview

**Target Branch:** `claude/combined-format-011CUzTnzrJQP498mXKLe4vt`
**Source Branch:** `claude/modernize-ui-design-011CUys922UbwzScZzbVVTc7`
**File to Modify:** `spectral_predict_gui_optimized.py`

**Objective:** Add the 5-theme visual system from the UI branch to the combined-format branch WITHOUT changing any functional/scientific code.

## What Gets Added

The UI branch provides a beautiful theme system with:
- **5 Japanese-inspired themes:** Sakura (🌸), Matcha (🍵), Sumi-e (🖌️), Yuhi (🌅), Ocean (🌊)
- **Top navigation bar** with theme switcher buttons
- **Theme management methods** for switching themes dynamically
- **Modern layout helpers** for future UI enhancements (currently not used in existing tabs)

## Prerequisites

1. Ensure you're on the `claude/combined-format-011CUzTnzrJQP498mXKLe4vt` branch
2. Back up current work (create a backup branch)
3. Have both branches available locally

---

## Step-by-Step Integration Guide

### Phase 1: Backup and Preparation (5 minutes)

**Step 1.1:** Create backup branch
```bash
git checkout claude/combined-format-011CUzTnzrJQP498mXKLe4vt
git branch backup-combined-format-before-ui
```

**Step 1.2:** Verify you're on the correct branch
```bash
git branch --show-current
# Should output: claude/combined-format-011CUzTnzrJQP498mXKLe4vt
```

---

### Phase 2: Code Extraction (10 minutes)

**Step 2.1:** Checkout the UI branch to extract code
```bash
git checkout claude/modernize-ui-design-011CUys922UbwzScZzbVVTc7
```

**Step 2.2:** Extract the following sections from `spectral_predict_gui_optimized.py`:

#### Extract Section A: Theme System in __init__() (Lines ~407-408)
```python
# Set default theme (can be changed by user)
self.current_theme_name = tk.StringVar(value='ocean')
self._apply_theme('ocean')
```

#### Extract Section B: Complete _configure_style() Method (Lines ~293-526)
This is a COMPLETE REPLACEMENT. Copy the entire method from:
```python
def _configure_style(self):
    """Configure modern Wabi-Sabi aesthetic with multiple theme support."""
    # ... all the way to the end of the method
```

#### Extract Section C: _create_top_bar() Method (Lines ~550-616)
```python
def _create_top_bar(self):
    """Create a beautiful top bar with app title and theme switcher."""
    # ... entire method
```

#### Extract Section D: _switch_theme() Method (Lines ~617-632)
```python
def _switch_theme(self, theme_name):
    """Switch to a new theme with smooth transition effect."""
    # ... entire method
```

#### Extract Section E: _update_widget_colors() Method (Lines ~634-659)
```python
def _update_widget_colors(self, widget):
    """Recursively update all widget colors to match current theme."""
    # ... entire method
```

#### Extract Section F: _show_theme_notification() Method (Lines ~661-680)
```python
def _show_theme_notification(self, theme_name):
    """Show a beautiful notification when theme changes."""
    # ... entire method
```

#### Extract Section G: Layout Helper Methods (Lines ~682-866)
```python
# ========== Modern Layout Helper Methods ==========

def _create_card(self, parent, title=None, subtitle=None):
    # ... entire method

def _create_section_header(self, parent, text, row, column=0, columnspan=3):
    # ... entire method

def _create_button_with_gradient(self, parent, text, command, style='accent'):
    # ... entire method

def _create_info_badge(self, parent, text, bg_color=None):
    # ... entire method

def _create_grid_layout(self, parent, num_columns=2):
    # ... entire method

def _create_checkbox_group(self, parent, title, variables_dict, columns=3):
    # ... entire method

def _create_collapsible_section(self, parent, title, expanded=True):
    # ... entire method

# ========== End of Modern Layout Helpers ==========
```

#### Extract Section H: Modified _create_ui() Method (Lines ~527-548)
Note the addition of `self._create_top_bar()` call:
```python
def _create_ui(self):
    """Create modern 9-tab user interface with theme switching."""
    # Create top bar with theme switcher and title
    self._create_top_bar()

    # Create main content area with notebook
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=20, pady=(0, 20))

    # Create tabs
    self._create_tab1_import_preview()
    # ... etc
```

**Step 2.3:** Return to the combined-format branch
```bash
git checkout claude/combined-format-011CUzTnzrJQP498mXKLe4vt
```

---

### Phase 3: Apply Changes to Combined-Format Branch (45 minutes)

#### Change 1: Read the current file
```bash
# Read spectral_predict_gui_optimized.py to understand current state
```

**Current State Summary:**
- Line 338-367: Simple `_configure_style()` with single theme
- Line 368-386: `_create_ui()` without top bar
- NO theme switching methods
- NO layout helper methods

#### Change 2: Locate the __init__() method
Find the end of the `__init__()` method (around line ~336, before `self._create_ui()`)

**Action:** ADD the theme initialization code AFTER the last variable initialization and BEFORE `self._create_ui()`:

```python
# ... existing code in __init__ ...

# Configure event debouncing (prevent slowdown on tab switches)
self._configure_timers = {}

# ADD THIS SECTION:
# Theme system initialization (added from UI modernization)
# NOTE: Theme is initialized in _configure_style() which is called earlier
# This variable tracks the current theme selection
self.theme_buttons = {}  # Will be populated by _create_top_bar()

self._create_ui()
```

**Note:** The actual theme initialization happens in the new `_configure_style()` method, so we just need to add the `self.theme_buttons` variable here.

#### Change 3: Replace _configure_style() Method (Lines 338-367)

**Current code to REPLACE:**
```python
def _configure_style(self):
    """Configure modern art gallery aesthetic."""
    style = ttk.Style()

    # Art gallery color palette
    self.colors = {
        'bg': '#F5F5F5',
        'panel': '#FFFFFF',
        'text': '#2C3E50',
        'text_light': '#7F8C8D',
        'accent': '#3498DB',
        'accent_dark': '#2980B9',
        'success': '#27AE60',
        'border': '#E8E8E8',
        'shadow': '#D0D0D0'
    }

    self.root.configure(bg=self.colors['bg'])

    style.configure('Modern.TButton', font=('Segoe UI', 10), padding=(15, 8))
    style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'), padding=(20, 12))
    style.configure('TFrame', background=self.colors['bg'])
    style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
    style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.colors['text'])
    style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.colors['text'])
    style.configure('Subheading.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.colors['accent'])
    style.configure('Caption.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_light'])
    style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
    style.configure('TNotebook.Tab', font=('Segoe UI', 11), padding=(20, 10))
```

**New code (from UI branch):**
```python
def _configure_style(self):
    """Configure modern Wabi-Sabi aesthetic with multiple theme support."""
    style = ttk.Style()

    # Define 5 beautiful theme skins inspired by Japanese aesthetics and modern design
    self.themes = {
        'sakura': {  # Cherry Blossom - Soft, elegant, feminine
            'name': '🌸 Sakura',
            'bg': '#FFF8F8',
            'bg_secondary': '#FFE8E8',
            'panel': '#FFFFFF',
            'sidebar': '#FFD1D1',
            'sidebar_hover': '#FFC1C1',
            'text': '#5D4157',
            'text_light': '#9B8A96',
            'text_inverse': '#FFFFFF',
            'accent': '#FF6B9D',
            'accent_dark': '#E85A8A',
            'accent_gradient': ['#FF6B9D', '#FFB6D9'],
            'success': '#82C785',
            'warning': '#F4A261',
            'border': '#FFD1D1',
            'shadow': '#FFE8E8',
            'tab_bg': '#FFFFFF',
            'tab_active': '#FF6B9D',
            'card_bg': '#FFFFFF',
        },
        'matcha': {  # Green Tea - Calm, natural, balanced
            'name': '🍵 Matcha',
            'bg': '#F8FBF6',
            'bg_secondary': '#E8F5E0',
            'panel': '#FFFFFF',
            'sidebar': '#B8D4A8',
            'sidebar_hover': '#A8C498',
            'text': '#2D4A2B',
            'text_light': '#6B8268',
            'text_inverse': '#FFFFFF',
            'accent': '#88CC77',
            'accent_dark': '#6BB85C',
            'accent_gradient': ['#88CC77', '#B8E6A8'],
            'success': '#6BBD6C',
            'warning': '#E8A547',
            'border': '#D0E5C8',
            'shadow': '#E8F5E0',
            'tab_bg': '#FFFFFF',
            'tab_active': '#88CC77',
            'card_bg': '#FFFFFF',
        },
        'sumie': {  # Ink Painting - Minimalist, monochromatic, zen
            'name': '🖌️ Sumi-e',
            'bg': '#F5F5F5',
            'bg_secondary': '#E8E8E8',
            'panel': '#FFFFFF',
            'sidebar': '#4A4A4A',
            'sidebar_hover': '#5A5A5A',
            'text': '#2C2C2C',
            'text_light': '#7A7A7A',
            'text_inverse': '#FFFFFF',
            'accent': '#5A5A5A',
            'accent_dark': '#3A3A3A',
            'accent_gradient': ['#5A5A5A', '#8A8A8A'],
            'success': '#6B9B6C',
            'warning': '#D89A5A',
            'border': '#D8D8D8',
            'shadow': '#C8C8C8',
            'tab_bg': '#FFFFFF',
            'tab_active': '#5A5A5A',
            'card_bg': '#FFFFFF',
        },
        'yuhi': {  # Sunset - Warm, vibrant, energetic
            'name': '🌅 Yuhi',
            'bg': '#FFF9F5',
            'bg_secondary': '#FFE8D8',
            'panel': '#FFFFFF',
            'sidebar': '#FF9A6C',
            'sidebar_hover': '#FF8A5C',
            'text': '#4A3A2F',
            'text_light': '#8A7A6F',
            'text_inverse': '#FFFFFF',
            'accent': '#FF6B4A',
            'accent_dark': '#E85A3A',
            'accent_gradient': ['#FF6B4A', '#FFB494'],
            'success': '#7FC77F',
            'warning': '#FFA726',
            'border': '#FFD1B8',
            'shadow': '#FFE8D8',
            'tab_bg': '#FFFFFF',
            'tab_active': '#FF6B4A',
            'card_bg': '#FFFFFF',
        },
        'ocean': {  # Ocean Wave - Deep, sophisticated, modern
            'name': '🌊 Ocean',
            'bg': '#F5F9FB',
            'bg_secondary': '#E0EEF5',
            'panel': '#FFFFFF',
            'sidebar': '#5BA3C4',
            'sidebar_hover': '#4B93B4',
            'text': '#1E3A52',
            'text_light': '#5E7A92',
            'text_inverse': '#FFFFFF',
            'accent': '#3D8AB8',
            'accent_dark': '#2D7AA8',
            'accent_gradient': ['#3D8AB8', '#7DC4E8'],
            'success': '#5CB85C',
            'warning': '#F0AD4E',
            'border': '#A8D4E8',
            'shadow': '#D0E4F0',
            'tab_bg': '#FFFFFF',
            'tab_active': '#3D8AB8',
            'card_bg': '#FFFFFF',
        }
    }

    # Set default theme (can be changed by user)
    self.current_theme_name = tk.StringVar(value='ocean')
    self._apply_theme('ocean')

def _apply_theme(self, theme_name):
    """Apply a specific theme to the application."""
    if theme_name not in self.themes:
        theme_name = 'ocean'

    self.colors = self.themes[theme_name]
    self.current_theme_name.set(theme_name)

    # Configure root window
    self.root.configure(bg=self.colors['bg'])

    # Get modern font stack (try Inter, SF Pro, fallback to system fonts)
    import platform
    system = platform.system()
    if system == 'Darwin':  # macOS
        heading_font = ('SF Pro Display', 'Helvetica Neue', 'Arial')
        body_font = ('SF Pro Text', 'Helvetica Neue', 'Arial')
    elif system == 'Windows':
        heading_font = ('Segoe UI', 'Arial')
        body_font = ('Segoe UI', 'Arial')
    else:  # Linux
        heading_font = ('Inter', 'Ubuntu', 'DejaVu Sans', 'Arial')
        body_font = ('Inter', 'Ubuntu', 'DejaVu Sans', 'Arial')

    style = ttk.Style()

    # Modern button styles with gradients (simulated with colors)
    style.configure('Modern.TButton',
                   font=(body_font, 10),
                   padding=(15, 8),
                   borderwidth=0,
                   relief='flat')
    style.map('Modern.TButton',
             background=[('active', self.colors['accent']),
                       ('!disabled', self.colors['panel'])])

    style.configure('Accent.TButton',
                   font=(body_font, 11, 'bold'),
                   padding=(20, 12),
                   background=self.colors['accent'],
                   foreground=self.colors['text_inverse'],
                   borderwidth=0,
                   relief='flat')
    style.map('Accent.TButton',
             background=[('active', self.colors['accent_dark']),
                       ('!disabled', self.colors['accent'])])

    # Frame styling
    style.configure('TFrame', background=self.colors['bg'])
    style.configure('Card.TFrame', background=self.colors['card_bg'], relief='flat')
    style.configure('Sidebar.TFrame', background=self.colors['sidebar'])

    # Label hierarchy with modern typography
    style.configure('TLabel',
                   background=self.colors['bg'],
                   foreground=self.colors['text'],
                   font=(body_font, 10))
    style.configure('Title.TLabel',
                   font=(heading_font, 28, 'bold'),
                   foreground=self.colors['text'],
                   background=self.colors['bg'])
    style.configure('Heading.TLabel',
                   font=(heading_font, 16, 'bold'),
                   foreground=self.colors['text'],
                   background=self.colors['bg'])
    style.configure('Subheading.TLabel',
                   font=(heading_font, 12, 'bold'),
                   foreground=self.colors['accent'],
                   background=self.colors['bg'])
    style.configure('Caption.TLabel',
                   font=(body_font, 9),
                   foreground=self.colors['text_light'],
                   background=self.colors['bg'])
    style.configure('SidebarLabel.TLabel',
                   font=(body_font, 11),
                   foreground=self.colors['text_inverse'],
                   background=self.colors['sidebar'],
                   padding=(15, 10))
    style.configure('CardLabel.TLabel',
                   background=self.colors['card_bg'],
                   foreground=self.colors['text'],
                   font=(body_font, 10))

    # Notebook styling - will be replaced with sidebar navigation
    style.configure('TNotebook',
                   background=self.colors['bg'],
                   borderwidth=0,
                   tabmargins=[0, 0, 0, 0])
    style.configure('TNotebook.Tab',
                   font=(body_font, 11),
                   padding=(20, 10),
                   borderwidth=0)
    style.map('TNotebook.Tab',
             background=[('selected', self.colors['tab_active']),
                       ('!selected', self.colors['tab_bg'])],
             foreground=[('selected', self.colors['text_inverse']),
                       ('!selected', self.colors['text'])])

    # Entry and input styling
    style.configure('TEntry',
                   fieldbackground=self.colors['panel'],
                   foreground=self.colors['text'],
                   borderwidth=1,
                   relief='solid')

    # Checkbutton styling
    style.configure('TCheckbutton',
                   background=self.colors['bg'],
                   foreground=self.colors['text'],
                   font=(body_font, 10))

    # Radiobutton styling
    style.configure('TRadiobutton',
                   background=self.colors['bg'],
                   foreground=self.colors['text'],
                   font=(body_font, 10))
```

#### Change 4: Update _create_ui() Method (Lines 368-386)

**Current code:**
```python
def _create_ui(self):
    """Create 7-tab user interface."""
    # Create notebook
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Create tabs
    self._create_tab1_import_preview()
    # ... rest of tabs
```

**Updated code:**
```python
def _create_ui(self):
    """Create modern 9-tab user interface with theme switching."""
    # Create top bar with theme switcher and title
    self._create_top_bar()

    # Create main content area with notebook
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill='both', expand=True, padx=20, pady=(0, 20))

    # Create tabs
    self._create_tab1_import_preview()
    # ... rest of tabs (no changes to these lines)
```

**Note:** The only changes are:
1. Add `self._create_top_bar()` call at the beginning
2. Update notebook padding from `padx=10, pady=10` to `padx=20, pady=(0, 20)` (to accommodate top bar)
3. Update docstring

#### Change 5: Add Theme Management Methods

**Location:** Add AFTER the `_configure_style()` method and BEFORE `_create_ui()`

**Insert these methods:**

```python
def _create_top_bar(self):
    """Create a beautiful top bar with app title and theme switcher."""
    top_bar = tk.Frame(self.root, bg=self.colors['bg'], height=80)
    top_bar.pack(fill='x', padx=20, pady=(20, 10))
    top_bar.pack_propagate(False)

    # Left side: App title with gradient effect (simulated)
    title_frame = tk.Frame(top_bar, bg=self.colors['bg'])
    title_frame.pack(side='left', fill='y')

    app_title = tk.Label(title_frame,
                        text="Spectral Predict",
                        font=('SF Pro Display', 'Segoe UI', 'Arial', 32, 'bold'),
                        fg=self.colors['text'],
                        bg=self.colors['bg'])
    app_title.pack(side='left', pady=5)

    subtitle = tk.Label(title_frame,
                       text="  Automated Spectral Analysis",
                       font=('SF Pro Text', 'Segoe UI', 'Arial', 12),
                       fg=self.colors['text_light'],
                       bg=self.colors['bg'])
    subtitle.pack(side='left', pady=8)

    # Right side: Theme switcher with beautiful buttons
    theme_frame = tk.Frame(top_bar, bg=self.colors['bg'])
    theme_frame.pack(side='right', fill='y', padx=10)

    tk.Label(theme_frame,
            text="Theme:",
            font=('SF Pro Text', 'Segoe UI', 'Arial', 11),
            fg=self.colors['text_light'],
            bg=self.colors['bg']).pack(side='left', padx=(0, 10))

    # Create theme buttons with hover effects
    self.theme_buttons = {}
    for theme_name, theme_data in self.themes.items():
        btn = tk.Button(theme_frame,
                      text=theme_data['name'],
                      font=('SF Pro Text', 'Segoe UI', 'Arial', 10, 'bold'),
                      fg='white',
                      bg=theme_data['accent'],
                      activebackground=theme_data['accent_dark'],
                      relief='flat',
                      borderwidth=0,
                      padx=15,
                      pady=8,
                      cursor='hand2',
                      command=lambda tn=theme_name: self._switch_theme(tn))
        btn.pack(side='left', padx=3)

        # Add hover effect
        def on_enter(e, b=btn, td=theme_data):
            b['bg'] = td['accent_dark']

        def on_leave(e, b=btn, td=theme_data):
            b['bg'] = td['accent']

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)

        self.theme_buttons[theme_name] = btn

    # Add a subtle gradient line separator
    separator = tk.Frame(self.root, bg=self.colors['border'], height=2)
    separator.pack(fill='x', padx=20)

def _switch_theme(self, theme_name):
    """Switch to a new theme with smooth transition effect."""
    # Store current tab selection
    current_tab = self.notebook.index(self.notebook.select())

    # Apply new theme
    self._apply_theme(theme_name)

    # Update all widgets to reflect new theme
    self._update_widget_colors(self.root)

    # Restore tab selection
    self.notebook.select(current_tab)

    # Show a subtle notification
    self._show_theme_notification(self.themes[theme_name]['name'])

def _update_widget_colors(self, widget):
    """Recursively update all widget colors to match current theme."""
    try:
        # Update widget background if it has one
        if hasattr(widget, 'configure'):
            widget_type = widget.winfo_class()

            # Update different widget types appropriately
            if widget_type == 'Frame':
                widget.configure(bg=self.colors['bg'])
            elif widget_type == 'Label':
                widget.configure(bg=self.colors['bg'], fg=self.colors['text'])
            elif widget_type == 'Canvas':
                widget.configure(bg=self.colors['bg'])
            elif widget_type == 'Button':
                # Skip theme buttons as they have custom colors
                if widget not in self.theme_buttons.values():
                    widget.configure(bg=self.colors['panel'], fg=self.colors['text'])

        # Recursively update children
        for child in widget.winfo_children():
            self._update_widget_colors(child)

    except Exception:
        # Some widgets might not support color changes
        pass

def _show_theme_notification(self, theme_name):
    """Show a beautiful notification when theme changes."""
    # Create a temporary notification label
    notif = tk.Label(self.root,
                    text=f"✨ Theme changed to {theme_name}",
                    font=('SF Pro Text', 'Segoe UI', 'Arial', 11),
                    fg=self.colors['text_inverse'],
                    bg=self.colors['accent'],
                    padx=20,
                    pady=10)
    notif.place(relx=0.5, rely=0.95, anchor='center')

    # Fade out after 2 seconds
    def fade_out():
        try:
            notif.destroy()
        except:
            pass

    self.root.after(2000, fade_out)
```

#### Change 6: Add Modern Layout Helper Methods

**Location:** Add AFTER the theme management methods and BEFORE `_create_tab1_import_preview()`

**Insert these methods:**

```python
# ========== Modern Layout Helper Methods ==========

def _create_card(self, parent, title=None, subtitle=None):
    """Create a modern card container with optional title."""
    # Card frame with subtle shadow effect
    card_outer = tk.Frame(parent, bg=self.colors['shadow'], padx=2, pady=2)

    card = tk.Frame(card_outer, bg=self.colors['card_bg'], padx=20, pady=20)
    card.pack(fill='both', expand=True)

    if title:
        # Card title with accent color
        title_label = tk.Label(card,
                              text=title,
                              font=('SF Pro Display', 'Segoe UI', 'Arial', 14, 'bold'),
                              fg=self.colors['text'],
                              bg=self.colors['card_bg'],
                              anchor='w')
        title_label.pack(fill='x', pady=(0, 5))

    if subtitle:
        # Card subtitle
        subtitle_label = tk.Label(card,
                                 text=subtitle,
                                 font=('SF Pro Text', 'Segoe UI', 'Arial', 10),
                                 fg=self.colors['text_light'],
                                 bg=self.colors['card_bg'],
                                 anchor='w')
        subtitle_label.pack(fill='x', pady=(0, 15))

    return card_outer, card

def _create_section_header(self, parent, text, row, column=0, columnspan=3):
    """Create a modern section header with accent line."""
    header_frame = tk.Frame(parent, bg=self.colors['bg'])
    header_frame.grid(row=row, column=column, columnspan=columnspan, sticky='ew', pady=(20, 10))

    # Accent line on the left
    accent_line = tk.Frame(header_frame, bg=self.colors['accent'], width=4)
    accent_line.pack(side='left', fill='y', padx=(0, 10))

    # Header text
    header_label = tk.Label(header_frame,
                           text=text,
                           font=('SF Pro Display', 'Segoe UI', 'Arial', 16, 'bold'),
                           fg=self.colors['text'],
                           bg=self.colors['bg'],
                           anchor='w')
    header_label.pack(side='left', fill='x')

    return header_frame

def _create_button_with_gradient(self, parent, text, command, style='accent'):
    """Create a modern button with gradient-like effect."""
    if style == 'accent':
        bg_color = self.colors['accent']
        hover_color = self.colors['accent_dark']
    else:
        bg_color = self.colors['panel']
        hover_color = self.colors['bg_secondary']

    btn = tk.Button(parent,
                   text=text,
                   font=('SF Pro Text', 'Segoe UI', 'Arial', 11, 'bold'),
                   fg=self.colors['text_inverse'] if style == 'accent' else self.colors['text'],
                   bg=bg_color,
                   activebackground=hover_color,
                   relief='flat',
                   borderwidth=0,
                   padx=25,
                   pady=12,
                   cursor='hand2',
                   command=command)

    # Add hover effect
    def on_enter(e):
        btn['bg'] = hover_color

    def on_leave(e):
        btn['bg'] = bg_color

    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)

    return btn

def _create_info_badge(self, parent, text, bg_color=None):
    """Create a small info badge/pill."""
    if bg_color is None:
        bg_color = self.colors['bg_secondary']

    badge = tk.Label(parent,
                    text=text,
                    font=('SF Pro Text', 'Segoe UI', 'Arial', 9, 'bold'),
                    fg=self.colors['text'],
                    bg=bg_color,
                    padx=10,
                    pady=4)
    return badge

def _create_grid_layout(self, parent, num_columns=2):
    """Create a responsive grid layout container."""
    grid_frame = tk.Frame(parent, bg=self.colors['bg'])

    # Configure columns to distribute space evenly
    for i in range(num_columns):
        grid_frame.columnconfigure(i, weight=1, uniform='col')

    return grid_frame

def _create_checkbox_group(self, parent, title, variables_dict, columns=3):
    """Create a modern checkbox group with grid layout."""
    # Create card for checkbox group
    card_outer, card = self._create_card(parent, title=title)

    # Create grid for checkboxes
    checkbox_frame = tk.Frame(card, bg=self.colors['card_bg'])
    checkbox_frame.pack(fill='both', expand=True)

    row, col = 0, 0
    for label, var in variables_dict.items():
        cb = ttk.Checkbutton(checkbox_frame,
                            text=label,
                            variable=var,
                            style='TCheckbutton')
        cb.grid(row=row, column=col, sticky='w', padx=10, pady=5)

        col += 1
        if col >= columns:
            col = 0
            row += 1

    return card_outer

def _create_collapsible_section(self, parent, title, expanded=True):
    """Create a collapsible section with expand/collapse animation."""
    section_frame = tk.Frame(parent, bg=self.colors['bg'])

    # Header with toggle button
    header = tk.Frame(section_frame, bg=self.colors['bg_secondary'], cursor='hand2')
    header.pack(fill='x', pady=(5, 0))

    # Expand/collapse indicator
    indicator = tk.Label(header,
                        text='▼' if expanded else '▶',
                        font=('SF Pro Text', 'Segoe UI', 'Arial', 12),
                        fg=self.colors['text'],
                        bg=self.colors['bg_secondary'],
                        padx=10)
    indicator.pack(side='left')

    # Section title
    title_label = tk.Label(header,
                          text=title,
                          font=('SF Pro Display', 'Segoe UI', 'Arial', 13, 'bold'),
                          fg=self.colors['text'],
                          bg=self.colors['bg_secondary'],
                          anchor='w')
    title_label.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=8)

    # Content frame (initially visible if expanded)
    content = tk.Frame(section_frame, bg=self.colors['bg'])
    if expanded:
        content.pack(fill='both', expand=True, pady=(0, 5))

    # Toggle function
    is_expanded = [expanded]  # Use list to allow modification in nested function

    def toggle():
        if is_expanded[0]:
            content.pack_forget()
            indicator.config(text='▶')
            is_expanded[0] = False
        else:
            content.pack(fill='both', expand=True, pady=(0, 5))
            indicator.config(text='▼')
            is_expanded[0] = True

    header.bind('<Button-1>', lambda e: toggle())
    indicator.bind('<Button-1>', lambda e: toggle())
    title_label.bind('<Button-1>', lambda e: toggle())

    return section_frame, content

# ========== End of Modern Layout Helpers ==========
```

---

### Phase 4: Testing (30 minutes)

#### Test 1: Syntax Check
```bash
python spectral_predict_gui_optimized.py
# Should launch without errors
```

#### Test 2: Theme Switching
1. Launch the GUI
2. Verify top bar appears with 5 theme buttons
3. Click each theme button and verify:
   - Background colors change
   - Text colors update
   - Accent colors change on tabs
   - Notification appears at bottom
4. Themes to test:
   - 🌸 Sakura (soft pink)
   - 🍵 Matcha (green)
   - 🖌️ Sumi-e (monochrome)
   - 🌅 Yuhi (orange/sunset)
   - 🌊 Ocean (blue - default)

#### Test 3: Functional Testing
1. Load data (Tab 1)
2. Switch themes while data is loaded
3. Run analysis (verify theme doesn't affect analysis)
4. Check results (Tab 5) with different themes
5. Verify no console errors

#### Test 4: Visual Verification
For each theme, verify:
- Top bar displays correctly
- Theme buttons have hover effects
- Tabs change colors appropriately
- All text is readable (good contrast)
- Notification toast appears and disappears

---

### Phase 5: Commit (5 minutes)

**Step 5.1:** Stage changes
```bash
git add spectral_predict_gui_optimized.py
```

**Step 5.2:** Commit with descriptive message
```bash
git commit -m "ui: Add 5-theme visual system to combined-format branch

Integrates beautiful theme system from UI modernization branch:
- 5 Japanese-inspired themes: Sakura, Matcha, Sumi-e, Yuhi, Ocean
- Top navigation bar with theme switcher buttons
- Dynamic theme switching with notification system
- Modern layout helper methods for future enhancements

Changes:
- Replace _configure_style() with multi-theme version
- Add theme management methods (_switch_theme, _update_widget_colors, etc.)
- Add _create_top_bar() for theme selection UI
- Add 7 layout helper methods (not yet applied to existing tabs)

IMPORTANT: Zero changes to functional/scientific code
IMPORTANT: All analysis logic remains unchanged

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

Before considering the upgrade complete, verify:

- [ ] GUI launches without errors
- [ ] All 5 themes are selectable
- [ ] Theme buttons have hover effects (color changes on hover)
- [ ] Clicking a theme changes the app colors
- [ ] Toast notification appears when switching themes
- [ ] All tabs are still accessible
- [ ] Data loading works with different themes
- [ ] Analysis runs successfully (no functional changes)
- [ ] Results display correctly in all themes
- [ ] No console errors or warnings
- [ ] Top bar displays properly on window resize
- [ ] Text is readable in all themes (contrast is good)

---

## Troubleshooting

### Issue 1: GUI doesn't launch
**Symptom:** Error on startup
**Solution:** Check Python syntax - likely missing comma or parenthesis in theme dictionaries

### Issue 2: Theme buttons don't appear
**Symptom:** Top bar missing or no buttons
**Solution:** Verify `_create_top_bar()` is called in `_create_ui()` BEFORE notebook creation

### Issue 3: Theme switching doesn't work
**Symptom:** Clicking buttons does nothing
**Solution:**
- Check that `self.theme_buttons` is initialized in `__init__()`
- Verify `_switch_theme()` method exists
- Check lambda in button command is correct

### Issue 4: Colors look wrong
**Symptom:** Some widgets don't change color
**Solution:**
- Some ttk widgets resist color changes
- This is expected - focus on main containers and labels
- Check `_update_widget_colors()` is being called

### Issue 5: Font rendering issues
**Symptom:** Fonts look different on different platforms
**Solution:** This is expected - the code uses platform-specific fonts (SF Pro on Mac, Segoe UI on Windows)

---

## What NOT to Change

**CRITICAL: Do NOT modify these areas:**

❌ Any analysis logic or methods
❌ Data loading/processing code
❌ Model training/validation code
❌ Any methods in `src/spectral_predict/` modules
❌ Tab content creation (except for applying layout helpers later)
❌ Functional variables or data structures
❌ Outlier detection logic
❌ Preprocessing methods
❌ File I/O operations

**The ONLY file to change:** `spectral_predict_gui_optimized.py`

**The ONLY sections to change:**
1. `__init__()` - add theme button variable
2. `_configure_style()` - complete replacement
3. `_create_ui()` - add top bar call
4. New methods - add theme methods and helpers

---

## Future Enhancements (Not Part of This Upgrade)

The layout helper methods are defined but NOT yet applied to existing tabs. Future work could:
- Replace existing tab layouts with modern card-based design
- Add collapsible sections for better organization
- Use gradient buttons instead of plain ttk buttons
- Add info badges for status indicators
- Create responsive grid layouts for better spacing

**These are infrastructure for future work - do NOT apply them now unless specifically requested.**

---

## Summary

This upgrade adds a beautiful, modern theme system to the combined-format branch with:
- **Zero impact** on scientific functionality
- **5 professional themes** inspired by Japanese aesthetics
- **Smooth theme switching** with visual feedback
- **Modern UI infrastructure** ready for future enhancements
- **Platform-optimized fonts** for macOS, Windows, and Linux

**Total changes:** ~600 lines added to `spectral_predict_gui_optimized.py`
**Files modified:** 1
**Risk level:** Low (purely cosmetic changes)
**Testing time:** 30 minutes
**Implementation time:** 60 minutes

---

## Questions or Issues

If you encounter any problems during implementation:
1. Check the Troubleshooting section above
2. Verify you're making changes to the correct branch
3. Compare your code with the UI branch version line-by-line
4. Test incrementally (don't make all changes at once)
5. Keep the backup branch available for rollback

Good luck with the upgrade! The end result will be a beautiful, theme-able GUI with zero impact on the underlying analysis code.
