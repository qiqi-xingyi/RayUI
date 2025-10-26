# RayUI — Ray Tracing Visualization Tool

## 📘 Overview
**RayUI** is a Python-based ray tracing visualization tool with a graphical user interface built using Tkinter.  
It supports two primary rendering modes — **Shaded (Phong lighting)** and **Labels (Semantic Color Rendering)** — allowing users to easily visualize both photorealistic illumination and semantic color segmentation of objects within a Cornell-box–style 3D scene.

This program is designed for educational and demonstrative purposes in computer graphics and computational visualization, showcasing a simplified ray tracing pipeline implemented entirely in Python.

---

## 🧩 Environment Setup

### 1. Python Version
- Requires **Python ≥ 3.9** (tested with 3.10, 3.11, 3.12)
- Works on **macOS**, **Windows**, and **Linux**

### 2. Required Libraries
Install the dependencies:
```bash
pip install numpy pillow
```

*(Tkinter is included by default in most Python distributions; no need to install separately.)*

---

## 🚀 Run the Program
In the same directory as `ray_ui_app_final.py`, open a terminal and execute:
```bash
python ray_ui_app_final.py
```

### macOS Users
If Tkinter windows or file dialogs fail to open, ensure you are using the **official python.org build** of Python (not Homebrew’s).  
No additional dependencies are needed for macOS.

---

## 🖼️ User Interface and Usage Guide

### Main Window
When the program launches, a GUI window appears with **controls on the left** and a **preview area on the right**.

#### Control Panel Fields

| Field | Description | Recommended |
|--------|--------------|-------------|
| **Width / Height** | Output image resolution | 600 × 600 |
| **Use reference size** | If checked, automatically uses the size of an image opened via “Open Image (Ref)” | ✓ Recommended |
| **Render Mode** | `"labels"` (semantic rendering) or `"shaded"` (Phong lighting) | **labels** |
| **Projection** | `"perspective"` (3D projection) or `"orthographic"` (flat view) | **perspective** |
| **FOV (deg)** | Field of view (applies to perspective projection) | 25–35° |
| **Reflect depth** | Reflection recursion depth (for shaded mode only) | 0 |

#### Buttons

| Button | Function |
|---------|-----------|
| **Render** | Starts rendering based on current settings |
| **Save Image…** | Exports the rendered result (PNG, JPG, BMP) |
| **Open Image (Ref)…** | Opens a reference image for visual comparison; can be used with “Use reference size” |
| **Status Bar** | Displays progress, status, and save path messages |

---

## 🎨 Rendering Modes

### 1. Labels Mode (Semantic Rendering)
Generates a **color-coded segmentation** of the scene objects, useful for creating ground-truth–like mask images.

| Object | Color (RGB) | Description |
|---------|-------------|-------------|
| Back wall | (0, 255, 0) | Green |
| Left wall | (255, 0, 0) | Red |
| Right wall | (0, 0, 255) | Blue |
| Floor | (0, 0, 255) | Blue |
| Ceiling | (0, 0, 255) | Blue |
| Small sphere | (255, 165, 0) | Orange |
| Large sphere | (0, 128, 128) | Teal |
| Background | (0, 0, 0) | Black |

This mode matches the simplified visualization required for semantic segmentation or “mask” output.  
Each object’s class is rendered in a unique color without lighting or shading.

---

### 2. Shaded Mode (Phong Lighting)
Renders the same geometry using **Phong local illumination**.  
Includes ambient, diffuse, and specular components and optionally supports **single-bounce reflections** (controlled by “Reflect depth”).

This mode illustrates how surface normals and material properties affect light interaction within the scene.

---

## 📐 Scene and Geometry Overview

- **Scene Type:** Cornell-box–style environment  
- **Camera Position:** (0, 1, 2.5)  
- **View Direction:** −z axis  
- **Coordinate Ranges:**  
  - x ∈ [−1, 1]  
  - y ∈ [0, 2]  
  - z ∈ [−3, 0]

### Objects
| Object | Type | Position / Plane | Notes |
|---------|------|------------------|-------|
| Left wall | Plane | x = −1 | Red |
| Right wall | Plane | x = +1 | Blue |
| Floor | Plane | y = 0 | Blue |
| Ceiling | Plane | y = 2 | Blue |
| Back wall | Plane | z = −3 | Green |
| Small sphere | Sphere | center (−0.5, 0.3, −2.2), radius 0.3 | Orange |
| Large sphere | Sphere | center (0.4, 0.55, −1.2), radius 0.55 | Teal |

### Lighting
Two white point lights placed near the top of the scene illuminate the objects:
- Light 1: position (0.0, 1.8, −1.0)
- Light 2: position (0.5, 1.0, −2.5)

---

## 📏 Output Image Example

### **Labels Mode**
A 2D segmentation-like image showing each object in a distinct solid color.

### **Shaded Mode**
A soft-shaded 3D rendering of the same geometry with realistic lighting.

---

## 🧱 Notes & Features
- No OpenGL or GPU dependency — fully CPU-based using NumPy.
- Cross-platform (macOS / Windows / Linux).
- Multithreaded rendering keeps UI responsive.
- Safe file dialogs on macOS (no Tk `NSInvalidArgumentException` crash).
- “Use reference size” ensures perfect pixel alignment with an input reference image.

---

## 💾 Optional Packaging (if needed)
You can optionally package the script into a standalone application.

### macOS `.app`
```bash
pyinstaller --windowed --onefile ray_ui_app_final.py
```

### Windows `.exe`
```bash
pyinstaller --noconsole --onefile ray_ui_app_final.py
```

Both commands produce a distributable file in the `dist/` folder.

---

## 📄 Example Workflow

1. Launch the program:
   ```bash
   python ray_ui_app_final.py
   ```
2. (Optional) Open a reference image via **Open Image (Ref)**.
3. Check **Use reference size** to match your input’s resolution.
4. Choose:
   - **Render Mode:** `labels`
   - **Projection:** `perspective`
   - **FOV:** around 30°
5. Click **Render**.
6. When finished, preview the result and **Save Image…** as `.png`.

---

## 🔧 Troubleshooting

| Issue | Cause / Fix |
|--------|--------------|
| Tk window won’t open on macOS | Use the official python.org build, not Homebrew |
| File dialog crash (macOS) | Already fixed in this version |
| Image too large / cropped | Adjust FOV or projection to perspective |
| Side walls missing | Use perspective projection (orthographic can’t hit side walls) |
| Slow rendering | Decrease resolution (e.g., 400×400) |
| Wrong output colors | Check Render Mode = “labels” |

---

## 🧰 Dependencies Summary
| Library | Version | Purpose |
|----------|----------|----------|
| `numpy` | ≥1.20 | Vector math & ray intersection |
| `Pillow` | ≥9.0 | Image output & GUI integration |
| `tkinter` | Built-in | GUI toolkit (for controls and file dialogs) |

---

## 🏫 Author and Acknowledgments
**Author:** Yuqi Zhang
**Year:** 2025

This tool was developed as part of a course assignment demonstrating ray tracing and semantic rendering principles.

---
