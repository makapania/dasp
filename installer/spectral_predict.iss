; Spectral Predict Inno Setup Script
; Creates Windows installer with Start Menu shortcuts and file associations
; Built with Nuitka standalone output

#define MyAppName "Spectral Predict"
#define MyAppVersion "0.4.0"
#define MyAppPublisher "Spectral Predict"
#define MyAppURL "https://github.com/spectral-predict"
#define MyAppExeName "SpectralPredict.exe"
#define MyAppAssocName "Spectral Predict Model File"
#define MyAppAssocExt ".dasp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{B8E7F2A1-4C3D-4E5F-9A1B-2C3D4E5F6A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output location and filename
OutputDir=..\dist\installer
OutputBaseFilename=SpectralPredict_Setup_{#MyAppVersion}
; Use modern compression
Compression=lzma2/ultra64
SolidCompression=yes
; Require Windows 10 or later
MinVersion=10.0
; Modern installer appearance
WizardStyle=modern
; Icon for installer
SetupIconFile=..\asp_logo.ico
; Uninstall icon
UninstallDisplayIcon={app}\{#MyAppExeName}
; Privileges - allow user or admin install
PrivilegesRequiredOverridesAllowed=dialog
; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "fileassoc"; Description: "Associate .dasp files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: checkedonce

[Files]
; Main application files (Nuitka standalone output folder)
Source: "..\dist\SpectralPredict\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Example data (also included by Nuitka, but ensure it's present)
Source: "..\example\BoneCollagen.csv"; DestDir: "{app}\example"; Flags: ignoreversion

[Icons]
; Start Menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; File association for .dasp files
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc

[Run]
; Option to run after install
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Custom code for initialization
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
