"""Test Dear PyGui file dialog behavior."""

import dearpygui.dearpygui as dpg
from pathlib import Path

def get_desktop_path() -> str:
    """Get the user's Desktop path."""
    desktop = Path.home() / "Desktop"
    if desktop.exists():
        return str(desktop)
    onedrive_desktop = Path.home() / "OneDrive" / "Desktop"
    if onedrive_desktop.exists():
        return str(onedrive_desktop)
    return str(Path.home())

def file_callback(sender, app_data):
    print(f"Callback triggered!")
    print(f"Sender: {sender}")
    print(f"App data: {app_data}")
    if app_data:
        print(f"Keys: {app_data.keys()}")
        if 'file_path_name' in app_data:
            print(f"Selected: {app_data['file_path_name']}")
        if 'selections' in app_data:
            print(f"Selections: {app_data['selections']}")

def open_file_dialog():
    print(f"Opening file dialog, default path: {get_desktop_path()}")

    # Try simpler approach without file extensions
    dpg.add_file_dialog(
        directory_selector=False,
        show=True,
        callback=file_callback,
        tag="file_dlg",
        width=800,
        height=500,
        default_path=get_desktop_path(),
    )

def open_folder_dialog():
    print(f"Opening folder dialog, default path: {get_desktop_path()}")

    dpg.add_file_dialog(
        directory_selector=True,
        show=True,
        callback=file_callback,
        tag="folder_dlg",
        width=800,
        height=500,
        default_path=get_desktop_path(),
    )

dpg.create_context()
dpg.create_viewport(title="File Dialog Test", width=600, height=400)

with dpg.window(label="Test", width=600, height=400):
    dpg.add_text("Click buttons to test file dialogs:")
    dpg.add_button(label="Open File Dialog", callback=open_file_dialog)
    dpg.add_button(label="Open Folder Dialog", callback=open_folder_dialog)
    dpg.add_text("Check console for output")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
