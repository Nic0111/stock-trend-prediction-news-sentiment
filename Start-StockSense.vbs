Option Explicit

Dim shell, fso, scriptDirectory, powershellPath, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
powershellPath = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"
command = """" & powershellPath & """ -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & scriptDirectory & "\start_stocksense.ps1"""
shell.Run command, 0, False
