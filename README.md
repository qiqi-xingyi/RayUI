、RayUI — Recursive Ray Tracer (Project 3)
==================================================================

Overview
--------
RayUI is a Python-based recursive ray tracer with a graphical user interface.
It builds upon previous versions by implementing a full recursive ray tracing algorithm
to simulate global illumination effects.

Key features added in this version (Project 3):
1. Recursive Ray Tracing (Reflection & Refraction)
2. Transparent materials with Index of Refraction (Snell's Law)
3. Shadow casting (Shadow Rays) for realistic lighting occlusion
4. Triangle mesh support (carried over from Project 2)

Environment Setup
-----------------
Python version: Python 3.9 or higher
Libraries required:
- numpy
- pillow

Install dependencies:
    pip install numpy pillow

Running the Program
-------------------
In the directory containing main.py, run:
    python main.py

The GUI window will open, showing a control panel on the left and a preview window on the right.

Project 3 Features & Requirements
---------------------------------
This updated program fulfills the requirements for Advanced Computer Graphics Project 3:

1. [cite_start]**Recursive Ray Tracer**: Implements a recursive `trace_shaded` function to handle multiple bounces of light[cite: 5].
2. **Reflection & Refraction**:
   - Renders smooth reflection on the mirror sphere.
   - [cite_start]Renders refraction on the glass sphere[cite: 6].
3. [cite_start]**Transparency**: The glass sphere is rendered with transparency and an Index of Refraction (IOR = 1.5)[cite: 7].
4. [cite_start]**Shadows (Extra Credit)**: Implements shadow rays in the Phong shading model to cast hard shadows from objects[cite: 11, 20].

Scene Description
-----------------
The scene is a Cornell-box–style environment modified to demonstrate optical effects:

- **Left Object**: Glass Sphere (Transparent, Refractive, IOR 1.5)
- **Right Object**: Mirror Sphere (Highly Reflective)
- **Center Object**: Triangle Mesh (loaded from `model.obj`)
- **Walls**: Left (Red), Right (Mirror-like Blue), Floor (Grey), Ceiling, Back.

Lighting:
- Main Light: (0.0, 1.8, -1.5)
- Fill Light: (0.0, 0.5, 0.0)

User Interface Guide
--------------------
Control panel fields:

**Width / Height**
- Output image resolution.
- Recommended: 400x400 for quick testing, 600x600 for final quality.

**Render Mode**
- "shaded": Full recursive rendering (Phong + Shadows + Reflect/Refract).
- "labels": Semantic color segmentation.

**Trace depth** (Previously "Reflect depth")
- Controls the maximum recursion depth for reflection and refraction.
- **Important**: Must be set to at least 1 (ideally 3-5) to see transparency and glass effects.
- Setting to 0 will result in opaque, black appearance for transparent objects.

**Buttons**
- **Render**: Begins rendering.
- **Save Image…**: Saves the result.
- **Open Image**: Loads a reference image.

Triangle Mesh Object
--------------------
The program attempts to load `model.obj`. If the file is missing, it will generate a fallback tetrahedron geometry to prevent crashing.
To generate the standard cube mesh, run:
    python make_model.py

Example Workflow
----------------
1. Ensure `model.obj` exists (or run `make_model.py`).
2. Run the UI:
       python main.py
3. Set **Trace depth** to **3** or higher.
4. Render in **shaded** mode.
   - Observe the glass sphere (left) distorting the background.
   - Observe the mirror sphere (right) reflecting the scene.
   - Observe shadows cast by spheres and the mesh onto the floor/walls.
5. Save the image.

Troubleshooting
----------------
**Glass sphere looks black:**
- Increase the "Trace depth" value. If depth is 0, light cannot pass through the object.

**Shadows look too dark:**
- This is expected behavior for point lights (hard shadows). The ambient light setting ensures areas are not pitch black.

**Render is slow:**
- Recursive ray tracing is computationally expensive. Reduce the image resolution (e.g., 300x300) or reduce Trace depth.

Credits
-------
Author: Yuqi Zhang
Updated: 2025
Designed for educational use in Advanced Computer Graphics.