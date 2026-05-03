; Spectral Predict Inno Setup Script — Python 3.12 build (experimental)
;
; PARALLEL to spectral_predict.iss (the production 3.11 installer) — this file
; must never be touched during 3.11 build path changes. The two installers
; produce distinct output filenames so they can coexist on one machine for
; A/B testing, but they share the same AppId so installing one upgrades the
; other. That's intentional: users should have exactly one Spectral Predict
; installed at a time, regardless of which Python runtime it bundles.

#define MyAppName "Spectral Predict"
#define MyAppVersion "0.5.0b2"
#define MyAppPublisher "Spectral Predict"
#define MyAppURL "https://github.com/makapania/dasp"
#define MyAppExeName "SpectralPredict-py312.exe"
#define MyAppBundleDir "SpectralPredict-py312"
#define MyAppAssocName "Spectral Predict Model File"
#define MyAppAssocExt ".dasp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; Same AppId as the 3.11 installer — installing one replaces the other.
; Prevents the "two Spectral Predicts in Add/Remove Programs" confusion.
AppId={{B8E7F2A1-4C3D-4E5F-9A1B-2C3D4E5F6A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion} (Python 3.12)
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output location — distinct filename so it doesn't collide with the 3.11
; installer. User can spot-check which one they have by the filename.
OutputDir=..\dist\installer
OutputBaseFilename=SpectralPredict_Setup_py312_{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
MinVersion=10.0
WizardStyle=modern
SetupIconFile=..\asp_logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "fileassoc"; Description: "Associate .dasp files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: checkedonce

[Files]
; Main application files (PyInstaller 3.12 output folder)
Source: "..\dist\{#MyAppBundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Example data (also included in the bundle, but ensured here)
Source: "..\example\BoneCollagen.csv"; DestDir: "{app}\example"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; File association for .dasp files
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: fileassoc

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
