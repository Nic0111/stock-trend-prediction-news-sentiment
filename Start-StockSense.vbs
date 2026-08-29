Option Explicit

Dim shell, fso, scriptDirectory, powershellPath, command, exitCode, logDirectory
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
powershellPath = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"
command = """" & powershellPath & """ -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & scriptDirectory & "\start_stocksense.ps1"""
logDirectory = scriptDirectory & "\.cache\launcher-logs"
exitCode = shell.Run(command, 0, True)

If exitCode <> 0 Then
    MsgBox "StockSense could not start. Diagnostic logs are available in:" & vbCrLf & logDirectory, _
        vbExclamation, "StockSense startup"
End If
