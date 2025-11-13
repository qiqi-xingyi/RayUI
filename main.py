# --*-- conding:utf-8 --*--
# @time:10/26/22 15:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

# --*-- conding:utf-8 --*--
# @time:10/26/22 15:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py


# Notes:
#   - "Open Image..." is for viewing your input reference; it does not affect label rendering.
#   - For macOS Tk's file dialog, filetypes use tuple forms (no semicolon-separated patterns).

import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

# ------------------------------
# Math utilities
# ------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def reflect(I: np.ndarray, N: np.ndarray) -> np.ndarray:
    return I - 2.0 * np.dot(I, N) * N

# ------------------------------
# Materials and primitives
# ------------------------------
@dataclass
class Material:
    color: np.ndarray
    ka: float = 0.1
    kd: float = 0.8
    ks: float = 0.5
    shininess: float = 32.0
    reflectivity: float = 0.0

class Primitive:
    class_name: str = "object"
    material: Material

    def intersect(self, ro: np.ndarray, rd: np.ndarray):
        raise NotImplementedError

class Sphere(Primitive):
    def __init__(self, center, radius, material: Material, class_name="sphere"):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material
        self.class_name = class_name

    def intersect(self, ro: np.ndarray, rd: np.ndarray):
        oc = ro - self.center
        b = 2.0 * np.dot(rd, oc)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) * 0.5
        t2 = (-b + sqrt_disc) * 0.5
        t = None
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
        if t is None:
            return None
        p = ro + t * rd
        n = normalize(p - self.center)
        return (t, n, self.material, p, self)

class Plane(Primitive):
    def __init__(self, normal, d, material: Material, class_name="plane"):
        self.normal = normalize(np.array(normal, dtype=np.float32))
        self.d = float(d)
        self.material = material
        self.class_name = class_name

    def intersect(self, ro: np.ndarray, rd: np.ndarray):
        denom = np.dot(rd, self.normal)
        if abs(denom) < 1e-6:
            return None
        t = (self.d - np.dot(ro, self.normal)) / denom
        if t <= 1e-4:
            return None
        p = ro + t * rd
        return (t, self.normal, self.material, p, self)

# ------------------------------
# Triangle mesh primitive
# ------------------------------
class TriangleMesh(Primitive):
    def __init__(
        self,
        obj_path: str,
        material: Material,
        scale=1.0,
        translation=(0.0, 0.0, 0.0),
        class_name: str = "mesh_object",
    ):
        self.material = material
        self.class_name = class_name

        positions = []
        faces = []

        with open(obj_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v" and len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    positions.append([x, y, z])
                elif parts[0] == "f" and len(parts) >= 4:
                    idx = []
                    for token in parts[1:4]:
                        items = token.split("/")
                        vi = int(items[0]) - 1
                        idx.append(vi)
                    faces.append(idx)

        if not positions or not faces:
            raise ValueError(f"OBJ load failed or empty: {obj_path}")

        self.vertices = np.array(positions, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)

        if np.isscalar(scale):
            s = np.array([scale, scale, scale], dtype=np.float32)
        else:
            s = np.array(scale, dtype=np.float32)

        t = np.array(translation, dtype=np.float32)
        self.vertices = self.vertices * s + t

        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)

    def _ray_aabb_intersect(self, ro: np.ndarray, rd: np.ndarray):
        tmin = -1e20
        tmax = 1e20
        for i in range(3):
            if abs(rd[i]) < 1e-8:
                if ro[i] < self.bbox_min[i] or ro[i] > self.bbox_max[i]:
                    return None
            else:
                invD = 1.0 / rd[i]
                t0 = (self.bbox_min[i] - ro[i]) * invD
                t1 = (self.bbox_max[i] - ro[i]) * invD
                if invD < 0.0:
                    t0, t1 = t1, t0
                if t0 > tmin:
                    tmin = t0
                if t1 < tmax:
                    tmax = t1
                if tmax < tmin:
                    return None
        if tmax < 1e-4:
            return None
        if tmin > 1e-4:
            return tmin
        return tmax

    def intersect(self, ro: np.ndarray, rd: np.ndarray):
        if self._ray_aabb_intersect(ro, rd) is None:
            return None

        best_t = 1e20
        best_p = None
        best_n = None

        v = self.vertices
        eps = 1e-6

        for face in self.faces:
            i0, i1, i2 = face
            v0 = v[i0]
            v1 = v[i1]
            v2 = v[i2]

            e1 = v1 - v0
            e2 = v2 - v0
            pvec = np.cross(rd, e2)
            det = np.dot(e1, pvec)

            if abs(det) < eps:
                continue
            inv_det = 1.0 / det

            tvec = ro - v0
            u = np.dot(tvec, pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue

            qvec = np.cross(tvec, e1)
            v_param = np.dot(rd, qvec) * inv_det
            if v_param < 0.0 or u + v_param > 1.0:
                continue

            t = np.dot(e2, qvec) * inv_det
            if t <= 1e-4 or t >= best_t:
                continue

            best_t = t
            p = ro + t * rd
            n = normalize(np.cross(e1, e2))
            if np.dot(n, -rd) < 0.0:
                n = -n
            best_p = p
            best_n = n

        if best_p is None:
            return None
        return (best_t, best_n, self.material, best_p, self)

# ------------------------------
# Light & shaders (for shaded mode)
# ------------------------------
@dataclass
class PointLight:
    position: np.ndarray
    color: np.ndarray
    Ia: float = 0.12
    Id: float = 1.20
    Is: float = 1.20

def phong_shade(pos: np.ndarray, normal: np.ndarray, view_dir: np.ndarray,
                material: Material, lights: list[PointLight]) -> np.ndarray:
    base_color = material.color
    col = np.zeros(3, dtype=np.float32)
    for L in lights:
        col += L.Ia * material.ka * base_color * L.color
    for L in lights:
        Ldir = normalize(L.position - pos)
        diff = max(np.dot(Ldir, normal), 0.0)
        R = reflect(-Ldir, normal)
        spec = 0.0
        if diff > 0.0:
            spec = max(np.dot(R, view_dir), 0.0) ** material.shininess
        col += L.Id * material.kd * diff * base_color * L.color
        col += L.Is * material.ks * spec * L.color
    return np.clip(col, 0.0, 1.0)

# ------------------------------
# Scene (Cornell-box-like)
# ------------------------------
@dataclass
class Scene:
    objects: list[Primitive]
    lights: list[PointLight]

def build_scene() -> Scene:
    mat_left   = Material(color=np.array([1.00, 0.20, 0.20], dtype=np.float32))
    mat_right  = Material(color=np.array([0.15, 0.35, 1.00], dtype=np.float32))
    mat_floor  = Material(color=np.array([0.15, 0.35, 1.00], dtype=np.float32))
    mat_ceil   = Material(color=np.array([0.15, 0.35, 1.00], dtype=np.float32))
    mat_back   = Material(color=np.array([0.15, 1.00, 0.15], dtype=np.float32))
    mat_small  = Material(color=np.array([1.00, 0.65, 0.00], dtype=np.float32), ks=0.2, shininess=16)
    mat_big    = Material(color=np.array([0.00, 0.50, 0.50], dtype=np.float32), ks=0.6, shininess=64, reflectivity=0.0)
    mat_mesh   = Material(color=np.array([0.80, 0.80, 0.80], dtype=np.float32), ks=0.6, shininess=64)

    left   = Plane(normal=(-1, 0, 0), d= 1.0, material=mat_left,  class_name="left")
    right  = Plane(normal=( 1, 0, 0), d= 1.0, material=mat_right, class_name="right")
    floor  = Plane(normal=(0,  1, 0), d= 0.0, material=mat_floor, class_name="floor")
    ceil   = Plane(normal=(0, -1, 0), d= 2.0, material=mat_ceil,  class_name="ceiling")
    back   = Plane(normal=(0,  0, 1), d=-3.0, material=mat_back,  class_name="back")

    smallS = Sphere(center=(-0.5, 0.3, -2.2), radius=0.3, material=mat_small, class_name="small_sphere")
    bigS   = Sphere(center=( 0.4, 0.55, -1.2), radius=0.55, material=mat_big, class_name="big_sphere")

    objects = [left, right, floor, ceil, back, smallS, bigS]

    try:
        mesh = TriangleMesh(
            obj_path="model.obj",
            material=mat_mesh,
            scale=0.6,
            translation=(0.0, 0.3, -1.8),
            class_name="mesh_object",
        )
        objects.append(mesh)
    except FileNotFoundError:
        pass

    lights = [
        PointLight(position=np.array([ 0.0, 1.8, -1.0], dtype=np.float32),
                   color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                   Ia=0.10, Id=1.2, Is=1.2),
        PointLight(position=np.array([ 0.5, 1.0, -2.5], dtype=np.float32),
                   color=np.array([1.0, 0.95, 0.95], dtype=np.float32),
                   Ia=0.08, Id=0.9, Is=0.8),
    ]
    return Scene(objects, lights)

# ------------------------------
# Label/ID color map
# ------------------------------
CLASS_COLORS = {
    "back":          (0, 255, 0),
    "left":          (255, 0, 0),
    "right":         (0, 0, 255),
    "floor":         (0, 0, 255),
    "ceiling":       (0, 0, 255),
    "small_sphere":  (255, 165, 0),
    "big_sphere":    (0, 128, 128),
    "mesh_object":   (255, 0, 255),
    "object":        (255, 255, 255),
    "background":    (0, 0, 0),
}

def class_color_rgb01(name: str) -> np.ndarray:
    r, g, b = CLASS_COLORS.get(name, CLASS_COLORS["object"])
    return np.array([r, g, b], dtype=np.float32) / 255.0

# ------------------------------
# Ray tracing core
# ------------------------------
def intersect_scene(ro: np.ndarray, rd: np.ndarray, scene: Scene):
    tmin = 1e20
    hit = None
    for obj in scene.objects:
        h = obj.intersect(ro, rd)
        if h is None:
            continue
        t, n, m, p, who = h
        if t < tmin:
            tmin, hit = t, (p, n, m, who)
    return hit

def trace_shaded(ro: np.ndarray, rd: np.ndarray, scene: Scene, depth=0, max_depth=0) -> np.ndarray:
    hit = intersect_scene(ro, rd, scene)
    if not hit:
        return np.array([0.75, 0.85, 1.0], dtype=np.float32)
    p, n, m, who = hit
    view_dir = normalize(ro - p)
    local = phong_shade(p, n, view_dir, m, scene.lights)

    r = m.reflectivity if hasattr(m, "reflectivity") else 0.0
    if r > 0.0 and depth < max_depth:
        refl_dir = normalize(reflect(-view_dir, n))
        refl_col = trace_shaded(p + 1e-4 * refl_dir, refl_dir, scene, depth + 1, max_depth)
        return (1 - r) * local + r * refl_col
    return local

def trace_labels(ro: np.ndarray, rd: np.ndarray, scene: Scene) -> np.ndarray:
    hit = intersect_scene(ro, rd, scene)
    if not hit:
        return class_color_rgb01("background")
    _, _, _, who = hit
    return class_color_rgb01(getattr(who, "class_name", "object"))

# ------------------------------
# Projections
# ------------------------------
def render(width=600, height=600, mode="labels", projection="orthographic",
           fov_deg=20.0, max_reflect_depth=0) -> Image.Image:
    scene = build_scene()
    img = np.zeros((height, width, 3), dtype=np.float32)

    cam_pos = np.array([0.0, 1.0, 2.5], dtype=np.float32)
    aspect = width / float(height)
    fov = math.radians(max(1.0, min(150.0, fov_deg)))

    for y in range(height):
        ndc_y = 1 - 2 * ((y + 0.5) / height)
        py = math.tan(fov * 0.5) * ndc_y
        for x in range(width):
            ndc_x = 2 * ((x + 0.5) / width) - 1
            px = math.tan(fov * 0.5) * ndc_x * aspect

            if projection == "orthographic":
                ox = -1.0 + (ndc_x + 1.0) * 0.5 * 2.0
                oy =  2.0 - (ndc_y + 1.0) * 0.5 * 2.0
                ro = cam_pos + np.array([ox, oy - 1.0, 0.0], dtype=np.float32)
                rd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            else:
                ro = cam_pos
                rd = normalize(np.array([px, py, -1.0], dtype=np.float32))

            if mode == "labels":
                col = trace_labels(ro, rd, scene)
            else:
                col = trace_shaded(ro, rd, scene, depth=0, max_reflect_depth=max_reflect_depth)
            img[y, x] = col

    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img8)

# ------------------------------
# Tkinter UI
# ------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ray Tracing — Shaded / Labels")
        self.geometry("980x760")
        self.minsize(820, 600)

        self._pil_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        side = tk.Frame(self, padx=10, pady=10)
        side.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(side, text="Width:").grid(row=0, column=0, sticky="w")
        tk.Label(side, text="Height:").grid(row=1, column=0, sticky="w")
        self.width_var = tk.StringVar(value="600")
        self.height_var = tk.StringVar(value="600")
        tk.Entry(side, textvariable=self.width_var, width=8).grid(row=0, column=1, sticky="w")
        tk.Entry(side, textvariable=self.height_var, width=8).grid(row=1, column=1, sticky="w")

        self.use_ref_size = tk.BooleanVar(value=False)
        tk.Checkbutton(side, text="Use reference size", variable=self.use_ref_size).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 4))

        tk.Label(side, text="Render Mode:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.mode_var = tk.StringVar(value="labels")
        ttk.Combobox(side, textvariable=self.mode_var, values=["labels", "shaded"], width=12, state="readonly").grid(row=3, column=1, sticky="w")

        tk.Label(side, text="Projection:").grid(row=4, column=0, sticky="w")
        self.proj_var = tk.StringVar(value="perspective")
        ttk.Combobox(side, textvariable=self.proj_var, values=["orthographic", "perspective"], width=12, state="readonly").grid(row=4, column=1, sticky="w")

        tk.Label(side, text="FOV (deg):").grid(row=5, column=0, sticky="w")
        self.fov_var = tk.StringVar(value="30")
        tk.Entry(side, textvariable=self.fov_var, width=8).grid(row=5, column=1, sticky="w")

        tk.Label(side, text="Reflect depth:").grid(row=6, column=0, sticky="w")
        self.rdepth_var = tk.StringVar(value="0")
        tk.Entry(side, textvariable=self.rdepth_var, width=8).grid(row=6, column=1, sticky="w")

        self.render_btn = tk.Button(side, text="Render", command=self.on_render_click)
        self.render_btn.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 4))

        self.save_btn = tk.Button(side, text="Save Image...", command=self.on_save_click, state=tk.DISABLED)
        self.save_btn.grid(row=8, column=0, columnspan=2, sticky="we", pady=(4, 10))

        self.open_btn = tk.Button(side, text="Open Image (Ref)...", command=self.on_open_click)
        self.open_btn.grid(row=9, column=0, columnspan=2, sticky="we")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(side, textvariable=self.status_var, anchor="w", justify="left", wraplength=200).grid(row=10, column=0, columnspan=2, sticky="we", pady=(10, 0))

        self.preview = tk.Label(self, bg="#cccccc")
        self.preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Render", command=self.on_render_click)
        filemenu.add_command(label="Open Image (Ref)...", command=self.on_open_click)
        filemenu.add_command(label="Save Image...", command=self.on_save_click)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.bind("<Configure>", lambda e: self.update_preview())

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def on_render_click(self):
        try:
            if self.use_ref_size.get() and self._pil_image is not None:
                w, h = self._pil_image.size
                self.width_var.set(str(w))
                self.height_var.set(str(h))
            else:
                w = int(self.width_var.get())
                h = int(self.height_var.get())
            if w <= 0 or h <= 0:
                raise ValueError("Invalid image size.")
            mode = self.mode_var.get()
            proj = self.proj_var.get()
            fov = float(self.fov_var.get())
            rdepth = int(self.rdepth_var.get())
            rdepth = max(0, min(3, rdepth))
        except Exception as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return

        self.render_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.set_status(f"Rendering {mode} | {proj} | {w}x{h} ...")

        def _work():
            try:
                img = render(width=w, height=h, mode=mode, projection=proj, fov_deg=fov, max_reflect_depth=rdepth)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Render Error", str(e)))
                self.after(0, lambda: self.set_status("Render failed."))
                self.after(0, lambda: self.render_btn.config(state=tk.NORMAL))
                return
            self._pil_image = img
            self.after(0, self.update_preview)
            self.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.render_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.set_status("Done."))

        threading.Thread(target=_work, daemon=True).start()

    def on_save_click(self):
        if self._pil_image is None:
            messagebox.showinfo("No Image", "Please render or open an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("JPEG Image", ("*.jpg", "*.jpeg")),
                       ("BMP Image", "*.bmp")]
        )
        if not path:
            return
        try:
            self._pil_image.save(path)
            self.set_status(f"Saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def on_open_click(self):
        path = filedialog.askopenfilename(
            title="Open Image (Reference)",
            filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp")),
                       ("All Files", ("*.*",))]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open Error", str(e))
            return
        self._pil_image = img
        self.update_preview()
        self.save_btn.config(state=tk.NORMAL)
        self.set_status(f"Opened: {path}")

    def update_preview(self):
        if self._pil_image is None:
            return
        max_w = max(200, self.preview.winfo_width() or 800)
        max_h = max(200, self.preview.winfo_height() or 600)
        img = self._pil_image.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(img)
        self.preview.config(image=self._tk_image)

if __name__ == "__main__":
    app = App()
    app.mainloop()
