# YOLO Demo Image Pack (Downloader)

This pack includes a **manifest** and a **script** to fetch representative images for common YOLO tasks (detection, segmentation, pose, oriented boxes) plus video analytics and industry examples.

## How to use
```bash
# macOS/Linux
bash download.sh ./assets
# Windows (with Git Bash) do the same; with PowerShell, see below
```

**PowerShell (Windows)**:
```powershell
$Root = "assets"
Get-Content manifest.csv | Select-Object -Skip 1 | ForEach-Object {
  $cols = $_.Split(',')
  $group = $cols[0]; $filename = $cols[1]; $url = $cols[3]
  $folder = Join-Path $Root $group
  New-Item -ItemType Directory -Force -Path $folder | Out-Null
  Invoke-WebRequest -Uri $url -OutFile (Join-Path $folder $filename)
}
```

> Note: Unsplash images use the official force-download endpoint, which redirects to the original file.

## Licensing
- **Unsplash** items: Unsplash License (free to use, attribution appreciated but not required).
- **Wikimedia Commons** items: Each file page specifies its license (mostly CC BY/CC BY-SA or Public Domain). See the `page_url` in the manifest for attribution details.

If you plan to publish or redistribute, include attribution for the CC-licensed items (author + license).

## File List
See `manifest.csv` for the complete list. You can also prune or add rows for your project.
