# ReSearch (Windows)

ReSearch is a Windows taskbar app that lets you “Ask your files.” It opens in an overlay (Win+Shift+Space), answers locally using Ollama + FAISS, and cites the top 3 most relevant files. Single-click a citation to open the file.

## Run (dev)
1) PowerShell (Admin), project root: `.\scripts\windows-setup.ps1`
2) Put docs in `C:\Users\<you>\Documents\test`
3) `.\scripts\dev.ps1` then press **Win+Shift+Space**

## Notes
- Update the path in `app/src/App.tsx` to your actual username.
- Index and metadata are stored in `server\data\` at runtime.
