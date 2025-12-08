"""
Set up Windows file association for .sproject files.

This script registers the .sproject extension with Windows so that
double-clicking a .sproject file will open it in Spectral Predict v3.

Usage:
    python -m spectral_predict_v3.setup_file_association

Note: This script modifies the Windows registry for the current user.
No admin privileges are required.
"""

import sys
import os
from pathlib import Path


def setup_windows_file_association():
    """Register .sproject file type with Windows."""
    if sys.platform != 'win32':
        print("This script is for Windows only.")
        sys.exit(1)

    import winreg

    # Get Python executable path
    python_exe = sys.executable

    # Get the module directory
    module_dir = Path(__file__).parent

    # Command to run - use module execution
    # This is more reliable than pointing to a specific .py file
    command = f'"{python_exe}" -m spectral_predict_v3 "%1"'

    try:
        # Create file extension key
        # HKEY_CURRENT_USER\Software\Classes\.sproject
        ext_key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\.sproject"
        )
        winreg.SetValue(ext_key, "", winreg.REG_SZ, "SpectralPredict.Project")
        winreg.CloseKey(ext_key)
        print("Created .sproject extension registry key")

        # Create file type key
        # HKEY_CURRENT_USER\Software\Classes\SpectralPredict.Project
        type_key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\SpectralPredict.Project"
        )
        winreg.SetValue(type_key, "", winreg.REG_SZ, "Spectral Predict Project")
        winreg.CloseKey(type_key)
        print("Created SpectralPredict.Project type key")

        # Set default icon (optional - would need an .ico file)
        # icon_path = module_dir / "resources" / "icon.ico"
        # if icon_path.exists():
        #     icon_key = winreg.CreateKey(
        #         winreg.HKEY_CURRENT_USER,
        #         r"Software\Classes\SpectralPredict.Project\DefaultIcon"
        #     )
        #     winreg.SetValue(icon_key, "", winreg.REG_SZ, str(icon_path))
        #     winreg.CloseKey(icon_key)

        # Set open command
        cmd_key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\SpectralPredict.Project\shell\open\command"
        )
        winreg.SetValue(cmd_key, "", winreg.REG_SZ, command)
        winreg.CloseKey(cmd_key)
        print("Created open command registry key")

        print()
        print("File association set up successfully!")
        print()
        print("Configuration:")
        print(f"  Extension: .sproject")
        print(f"  Type: SpectralPredict.Project")
        print(f"  Command: {command}")
        print()
        print("You can now double-click .sproject files to open them in Spectral Predict v3.")
        print()
        print("Note: You may need to log out and log back in, or restart Explorer,")
        print("for the association to take effect.")

    except PermissionError:
        print("Error: Permission denied.")
        print("Try running this script with administrator privileges.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up file association: {e}")
        sys.exit(1)


def remove_file_association():
    """Remove the .sproject file association."""
    if sys.platform != 'win32':
        print("This script is for Windows only.")
        sys.exit(1)

    import winreg

    try:
        # Delete extension key
        try:
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\.sproject"
            )
            print("Removed .sproject extension key")
        except FileNotFoundError:
            pass

        # Delete type keys (need to delete children first)
        try:
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\SpectralPredict.Project\shell\open\command"
            )
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\SpectralPredict.Project\shell\open"
            )
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\SpectralPredict.Project\shell"
            )
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Classes\SpectralPredict.Project"
            )
            print("Removed SpectralPredict.Project type keys")
        except FileNotFoundError:
            pass

        print("File association removed successfully.")

    except Exception as e:
        print(f"Error removing file association: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up or remove Windows file association for .sproject files"
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help="Remove the file association instead of setting it up"
    )

    args = parser.parse_args()

    if args.remove:
        remove_file_association()
    else:
        setup_windows_file_association()


if __name__ == "__main__":
    main()
