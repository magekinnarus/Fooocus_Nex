# Fooocus_Nex GIMP Plugins

This directory contains plugins for GIMP to enable a seamless round-trip workflow with the Fooocus_Nex Staging Area.

## Features
- **Send to Staging**: Uploads the active layer from GIMP directly to the Fooocus_Nex Staging Area.
- **Receive from Staging**: Fetches the image targeted in the Fooocus_Nex Staging Palette (by clicking the 'G' button) and loads it as a new layer in GIMP.

## Installation

### GIMP 2.10 (Local PC)
1. Copy `fooocus_nex_gimp2.py` to your GIMP plugins folder:
   - **Windows**: `%APPDATA%\GIMP\2.10\plug-ins\`
   - **Linux**: `~/.config/GIMP/2.10/plug-ins/`
   - **macOS**: `~/Library/Application Support/GIMP/2.10/plug-ins/`
2. Restart GIMP.
3. The options will appear under `Filters > Fooocus_Nex`.

### GIMP 3.0 (Local PC)
1. Create a folder named `fooocus_nex_gimp3` in your GIMP 3.0 plugins directory.
2. Copy `fooocus_nex_gimp3.py` into that folder.
3. Make sure the file is executable (`chmod +x` on Linux/macOS).
4. Restart GIMP.

## Usage with Colab
If you are using Fooocus_Nex on Google Colab:
1. Copy your tunnel URL (e.g., from zrok, cloudflare, or localtunnel).
2. When running the plugin in GIMP, paste this URL into the **Fooocus API URL** field in the dialog.
3. Ensure the URL does not end with a slash.

## How it Works
1. **Send**: GIMP exports the layer to a temporary PNG and POSTs it to `/staging_api/upload`.
2. **Target**: In the Fooocus WebUI, open the **Staging Palette** and click the **G** button on any image. This flags it as the target.
## Troubleshooting

### "Unable to run GimpPdbProgress callback"
If you see this warning, it usually means a PDB call was interrupted. The latest version of these plugins (v1.1) adds explicit progress handling and undo groups to prevent this. 
- **Check Error Console**: Go to `Windows > Dockable Dialogs > Error Console` in GIMP to see detailed Python stack traces.
- **Restart GIMP**: Sometimes a "dying" plugin leaves GIMP's PDB in a stalled state. A restart usually clears it.

### Plugin not appearing in menu
- Ensure the file has `.py` extension.
- **Linux/macOS**: Ensure files are executable (`chmod +x`).
- **GIMP 3.0**: Ensure the script is inside a folder of the same name.
