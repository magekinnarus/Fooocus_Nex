# Frontend UI Design: 4-Zone Layout Specification

**Date:** 2026-03-05
**Source:** Director specification during P4-M02 planning
**Status:** Reference document — implementation via P4-M02 work orders

## 1. Layout Overview

The frontend is a **4-zone professional creative tool** layout with a top menu bar,
three vertical panels, and a horizontal split in the center panel.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Menu Bar  (File, Edit, View... + Connection Status)                │
├──────────┬──────────────────────────────────┬────────────────────────┤
│          │                                  │                        │
│  LEFT    │  CENTER - TOP                    │  RIGHT                 │
│  PANEL   │  Preview / Image Gallery         │  PANEL                 │
│          │                                  │  (Working Space)       │
│ Parameter│                                  │                        │
│   Tabs   ├──────────────────────────────────┤  Tabs:                 │
│          │  CENTER - BOTTOM                 │  Images (staging)      │
│          │  Workflow Tabs                   │  Models (browser)      │
│          │  (Upscale/Inpaint/Outpaint/      │  Prompts (library)     │
│          │   ControlNet/Metadata)           │                        │
└──────────┴──────────────────────────────────┴────────────────────────┘
```

All panels are **resizable** via drag handles. Design is **dark themed** (professional
creative tool aesthetic).

## 2. Left Panel — Parameter Tabs

### Default Tab
- **Positive prompt** — multi-line textarea, expandable downward as content grows.
  `Enter` is the chunk separator for prompt concatenation (inherited from Fooocus).
- **Negative prompt** — multi-line textarea, expandable.
- **Clip skip** — moved here from Advanced tab.
- **Steps** — default: 20, moved from Advanced.
- **Sampler** — dropdown, moved from Advanced.
- **Scheduler** — dropdown, moved from Advanced.
- **Guidance scale** — moved from Advanced.
- **Generate button** — primary action.

> **Removed from Default tab:** Quality, Speed, Extreme Speed, Hyper-SD, Lightning presets.
> Extreme Speed (LCM) and Lightning go to Debug Tools. Hyper-SD removed entirely.

### Prompt Presets Tab (renamed from "Styles")
- Displays presets from `sdxl_styles_fooocus.json` and `sdxl_styles_dj.json`.
- Each preset shown as **thumbnail + name underneath**.
- Click to toggle selection (multi-select).
- Selected presets applied via `style_selections` in generation request.

### Models Tab
- **Checkpoint/GGUF** dropdown — same as current Fooocus.
- **CLIP 1** dropdown — for clip model selection.
- **CLIP 2** dropdown — for clip model selection.
- **VAE** dropdown — for VAE model selection.
- **LoRA** — add/remove with weight sliders.

### Advanced / Debug Tools Tab
- **Extreme Speed** (LCM setup)
- **Lightning** setup
- **Sharpness** parameter
- Other remaining Fooocus advanced settings (preserved as-is for now).

## 3. Right Panel — Working Space Tabs

### Images Tab (Staging Area)
- Place images from the preview panel or from external sources.
- External sources include: Gimp import plugin, file system / directory browser.
- Can send images back to Gimp via plugin.
- Functions as a staging area for workflow inputs.

### Models Tab (Visual Browser)
- Browse **checkpoints**, **GGUF unets**, and **LoRAs** with thumbnails.
- **Drag** or **right-click** to place into left panel Models tab dropdowns.
- Distinct from left panel Models tab: right = discovery browser, left = active selection.

### Prompts Tab (Library)
- Stored prompts for use and reuse.
- **Drag** current prompt from left panel to save here.
- **Drag** from library or **right-click** to add to prompt boxes in left panel.

## 4. Center Panel — Preview + Workflows

### Preview Tab (default)
- Main image output display with full-resolution viewing.
- Shows generation progress during inference.
- Images can be dragged to the right panel staging area or to workflow tabs below.

### Image Gallery Tab
- Browse generated images from outputs directory.
- Drag images into:
  - Right panel staging area
  - Workflow tabs below

### Workflow Panel (bottom section)
- Tab-based layout similar to Fooocus_Nex.
- Tabs: **Upscale**, **Inpaint**, **Outpaint**, **ControlNet**, **Metadata**, etc.
- Core interaction: generate image in preview → drag down into workflow tab for further processing.
- To be revised as needed during implementation.

## 5. Menu Bar
- **File**, **Edit**, **View**, etc. — details to be added later.
- **Connection status indicator** — green/red/yellow, links to API server.
- Application title "Nex Engine".

## 6. Key Interaction Patterns

| Action | Description |
|--------|-------------|
| **Generate** | Left panel Default tab → click Generate or `Ctrl+Enter` |
| **Apply preset** | Left panel Prompt Presets tab → click to toggle selection |
| **Stage image** | Preview panel → drag or button to right panel Images tab |
| **Feed workflow** | Preview or Gallery → drag to workflow tab (bottom center) |
| **Browse models** | Right panel Models tab → drag/right-click to left panel Models tab |
| **Reuse prompt** | Right panel Prompts tab → drag/right-click to left panel prompt box |
| **Import image** | Right panel Images tab → Gimp plugin or file system import |
| **Export to Gimp** | Right panel Images tab → send to Gimp via plugin |

## 7. Persistence & Data

| Data | M02 Approach | Future Approach |
|------|-------------|-----------------|
| Saved prompts | `localStorage` | JSON Server, lowDB, or Firebase |
| Panel sizes | `localStorage` | `localStorage` (sufficient) |
| API URL | `localStorage` | `localStorage` (sufficient) |
| Generation history | In-memory state | JSON Server / lowDB |
| Model thumbnails | Type-based icons | Generated preview images |
| Prompt preset thumbs | Placeholder icons | Generated or user-provided images |

> **Note:** MongoDB is not required. JSON Server or lowDB can meet all persistence needs.
> The Director has prior experience with JSON Server → MongoDB → Firebase migration path.

## 8. Technology Stack

| Choice | Technology |
|--------|-----------|
| Framework | React 18+ (Vite) |
| Styling | Vanilla CSS (dark theme, CSS custom properties) |
| Typography | Inter (Google Fonts) |
| Icons | Lucide React |
| HTTP | Fetch API |
| WebSocket | Native WebSocket API |
| State | React hooks |
| Canvas (future) | Fabric.js or Konva |
