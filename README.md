RayUI — Ray Tracing Visualization Tool (with Triangle Mesh Support)
==================================================================

Overview
--------
RayUI is a Python-based ray tracer with a graphical user interface built using Tkinter.
It renders a Cornell-box–style 3D scene and supports two rendering modes:

1. Shaded Mode (Phong lighting)
2. Labels Mode (semantic color rendering)

This version adds support for loading triangle mesh objects (.obj) and rendering them alongside
built-in spheres and planes. The mesh is transformed, shaded, and accelerated using an AABB bounding
box. This update satisfies the requirements of Advanced Computer Graphics Project 2.


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


Triangle Mesh Object
--------------------
The ray tracer now supports loading a triangle-mesh OBJ file.
The default file is:
    model.obj

You may download an OBJ model or generate one using the provided script:
    python make_model.py

This creates a cube made of 12 triangles.

The mesh is automatically loaded in the scene through:
    TriangleMesh(obj_path="model.obj", scale=0.6, translation=(0.0, 0.3, -1.8))


Scene Description
-----------------
The scene is a Cornell-box–style environment containing:

- Left wall
- Right wall
- Floor
- Ceiling
- Back wall
- Small sphere
- Large sphere
- One triangle mesh object (loaded from model.obj)

Lighting:
- Light 1: (0.0, 1.8, -1.0)
- Light 2: (0.5, 1.0, -2.5)

Camera:
- Position: (0, 1, 2.5)
- Looks toward: negative z direction


User Interface Guide
--------------------
Control panel fields:

Width / Height
- Output image resolution
- Recommended: 600 × 600

Use reference size
- If checked, uses the resolution of a loaded reference image

Render Mode
- "labels": semantic color rendering
- "shaded": Phong illumination

Projection
- "perspective": camera-style view
- "orthographic": flat projection

FOV (deg)
- Field of view for perspective projection
- Recommended: 25–35 degrees

Reflect depth
- Reflection recursion depth for shaded mode
- Recommended: 0

Buttons:

Render
- Begins rendering with current settings

Save Image…
- Saves the rendered output as PNG/JPG/BMP

Open Image (Ref)…
- Loads an image for comparison (does not affect rendering)

Status Bar
- Shows progress and save messages


Rendering Modes
---------------
Labels Mode:
- Produces color-coded segmentation of scene objects
- Each object class has a fixed color
- Useful for mask-like output

Shaded Mode:
- Uses Phong lighting (ambient, diffuse, specular)
- Mesh, spheres, and walls are shaded consistently


Project 2 Requirements
----------------------
This updated program completes Project 2 requirements:

1. Loads and displays a triangle mesh object from an OBJ file
2. Applies geometric transformations (scale, translation)
3. Performs ray–triangle intersection using the Möller–Trumbore method
4. Applies AABB bounding box acceleration
5. Renders the mesh using local shading (Phong)
6. Mesh participates normally in label rendering

All requirements are fulfilled.


Example Workflow
----------------
1. Generate triangle mesh:
       python make_model.py

2. Run the UI:
       python main.py

3. Render in shaded mode → Save as shaded.png
4. Render in labels mode → Save as labels.png
5. Submit:
   - main.py
   - model.obj
   - make_model.py
   - shaded.png
   - labels.png
   - README.md


Troubleshooting
---------------
Mesh does not appear:
- Ensure model.obj exists in the same folder as main.py
- Adjust scale or translation values

Rendering is slow:
- Reduce image resolution

Colors appear incorrect:
- Check selected rendering mode (labels vs shaded)


Credits
-------
Author: Yuqi Zhang
Updated: 2025
Designed for educational use in Advanced Computer Graphics.
