; Inno Setup script: builds TechnoForesightAssistant-Setup-<version>.exe
; Per-user install (no admin / UAC prompt needed).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6B8E9A52-3C1F-4D7B-9E2A-7F4C1D0B5E91}
AppName=TechnoForesight Assistant
AppVersion={#AppVersion}
AppPublisher=TechnoForesight
AppPublisherURL=https://technoforesight.in/assistant
AppSupportURL=https://technoforesight.in/contact
DefaultDirName={localappdata}\Programs\TechnoForesight Assistant
DefaultGroupName=TechnoForesight
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=TechnoForesightAssistant-Setup-{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\TechnoForesightAssistant.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "startup"; Description: "Start automatically when I sign in (so Alt+Space always works)"; GroupDescription: "Options:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Options:"; Flags: unchecked

[Files]
Source: "..\dist\TechnoForesightAssistant\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\TechnoForesight Assistant"; Filename: "{app}\TechnoForesightAssistant.exe"
Name: "{userdesktop}\TechnoForesight Assistant"; Filename: "{app}\TechnoForesightAssistant.exe"; Tasks: desktopicon
Name: "{userstartup}\TechnoForesight Assistant"; Filename: "{app}\TechnoForesightAssistant.exe"; Tasks: startup

[Run]
Filename: "{app}\TechnoForesightAssistant.exe"; Description: "Launch TechnoForesight Assistant now"; Flags: nowait postinstall skipifsilent
