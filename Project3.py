# --*-- conding:utf-8 --*--
# @time:12/8/25 19:50
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Project3.py

# --*-- conding:utf-8 --*--
# Project 3 Implementation
# Includes: Recursive Ray Tracing, Reflection, Refraction (Transparency), Shadows

import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Optional, List

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


def refract(I: np.ndarray, N: np.ndarray, ior: float) -> Optional[np.ndarray]:
    """
    计算折射向量 (Snell's Law).
    I: 入射光线方向 (Normalized)
    N: 法线 (Normalized)
    ior: 目标介质的折射率 (Index of Refraction)
    返回: 折射光线方向，如果发生全内反射(TIR)则返回 None
    """
    cosi = np.clip(np.dot(I, N), -1.0, 1.0)
    etai = 1.0
    etat = ior
    n = N

    # 判断是从空气进入物体，还是从物体射出
    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -N

    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)

    if k < 0.0:
        return None  # 全内反射 Total Internal Reflection

    return eta * I + (eta * cosi - math.sqrt(k)) * n


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
    transparency: float = 0.0  # 新增: 透明度 (0.0 - 1.0)
    ior: float = 1.0  # 新增: 折射率 (例如玻璃 ~1.5)


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

        try:
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
        except Exception:
            # 如果文件没找到，创建一个简单的四面体作为默认 fallback，避免程序崩溃
            positions = [[0, 1, 0], [-1, -1, 1], [1, -1, 1], [0, -1, -1]]
            faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
            print(f"Warning: {obj_path} not found. Using fallback geometry.")

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
# Scene & Lights
# ------------------------------
@dataclass
class PointLight:
    position: np.ndarray
    color: np.ndarray
    Ia: float = 0.12
    Id: float = 1.20
    Is: float = 1.20


@dataclass
class Scene:
    objects: List[Primitive]
    lights: List[PointLight]


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


# ------------------------------
# Shading & Tracing
# ------------------------------
def phong_shade(pos: np.ndarray, normal: np.ndarray, view_dir: np.ndarray,
                material: Material, scene: Scene) -> np.ndarray:
    """
    计算 Phong 光照模型，包含阴影检测。
    """
    base_color = material.color
    col = np.zeros(3, dtype=np.float32)

    # 1. 环境光 (Ambient) - 总是存在
    for L in scene.lights:
        col += L.Ia * material.ka * base_color * L.color

    # 2. 漫反射 (Diffuse) 和 高光 (Specular) - 受阴影影响
    for L in scene.lights:
        Ldir_raw = L.position - pos
        dist = np.linalg.norm(Ldir_raw)
        Ldir = Ldir_raw / dist

        # --- 阴影检测 (Shadow Ray) ---
        # 向光源发射射线，起点稍微偏离表面 (bias) 以防自我遮挡
        shadow_bias = 1e-4
        shadow_orig = pos + normal * shadow_bias

        # 检查是否有什么东西挡在当前点和光源之间
        hit = intersect_scene(shadow_orig, Ldir, scene)
        in_shadow = False
        if hit:
            hit_p, _, _, _ = hit
            # 如果撞击点的距离小于到光源的距离，说明被遮挡
            if np.linalg.norm(hit_p - shadow_orig) < dist:
                in_shadow = True

        if not in_shadow:
            # Diffuse
            diff = max(np.dot(Ldir, normal), 0.0)
            col += L.Id * material.kd * diff * base_color * L.color

            # Specular
            if diff > 0.0:
                R = reflect(-Ldir, normal)
                spec = max(np.dot(R, view_dir), 0.0) ** material.shininess
                col += L.Is * material.ks * spec * L.color

    return np.clip(col, 0.0, 1.0)


def trace_shaded(ro: np.ndarray, rd: np.ndarray, scene: Scene, depth=0, max_depth=0) -> np.ndarray:
    hit = intersect_scene(ro, rd, scene)
    if not hit:
        return np.array([0.05, 0.05, 0.05], dtype=np.float32)  # 背景色变暗一点

    p, n, m, who = hit
    view_dir = normalize(ro - p)

    # 1. Local Color (Phong with Shadows)
    local = phong_shade(p, n, view_dir, m, scene)

    # 递归终止条件
    if depth >= max_depth:
        return local

    final_color = local

    # 2. Reflection (递归)
    reflect_color = np.zeros(3, dtype=np.float32)
    if m.reflectivity > 0.0:
        refl_dir = normalize(reflect(-view_dir, n))
        # 偏移一点避免自相交
        reflect_color = trace_shaded(p + 1e-4 * n, refl_dir, scene, depth + 1, max_depth)

    # 3. Refraction (递归)
    refract_color = np.zeros(3, dtype=np.float32)
    if m.transparency > 0.0:
        refr_dir = refract(rd, n, m.ior)
        if refr_dir is not None:
            refr_dir = normalize(refr_dir)
            # 注意：折射光线是穿过物体的，所以偏移方向要小心
            # 这里简单处理，假设物体是薄壁或实体，进入内部
            refract_color = trace_shaded(p - 1e-4 * n, refr_dir, scene, depth + 1, max_depth)
        else:
            # 全内反射，能量通常转为反射，这里简单处理，保持黑色或加到反射中
            pass

            # 简单的混合模型 (Fresnel 效果是加分项，这里用线性混合满足基本要求)
    # 权重归一化: (1 - refl - trans) * local + refl * Refl + trans * Refr
    k_refl = m.reflectivity
    k_trans = m.transparency
    k_local = max(0.0, 1.0 - k_refl - k_trans)

    final_color = k_local * local + k_refl * reflect_color + k_trans * refract_color

    return np.clip(final_color, 0.0, 1.0)


def trace_labels(ro: np.ndarray, rd: np.ndarray, scene: Scene) -> np.ndarray:
    hit = intersect_scene(ro, rd, scene)
    if not hit:
        return class_color_rgb01("background")
    _, _, _, who = hit
    return class_color_rgb01(getattr(who, "class_name", "object"))


# ------------------------------
# Scene Setup
# ------------------------------
def build_scene() -> Scene:
    # 颜色定义
    red = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    blue = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    grey = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    green = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    orange = np.array([1.0, 0.6, 0.1], dtype=np.float32)
    white = np.array([0.9, 0.9, 0.9], dtype=np.float32)

    # 材质定义
    # 左墙: 漫反射红
    mat_left = Material(color=red, ka=0.1, kd=0.8, ks=0.1, shininess=10)
    # 右墙: 镜面反射 (Mirror-like) - 展示反射
    mat_right = Material(color=blue, ka=0.1, kd=0.5, ks=0.8, shininess=64, reflectivity=0.5)
    # 地板: 稍微有点反射
    mat_floor = Material(color=grey, ka=0.1, kd=0.8, ks=0.2, reflectivity=0.2)
    # 天花板
    mat_ceil = Material(color=grey, ka=0.1, kd=0.8)
    # 背景墙
    mat_back = Material(color=green, ka=0.1, kd=0.8)

    # 玻璃球 (Transparency & Refraction)
    # ior=1.5 (玻璃), transparency=0.9, reflectivity=0.1 (Fresnel效应简单模拟)
    mat_glass = Material(color=np.array([1.0, 1.0, 1.0]), ka=0.0, kd=0.0, ks=0.8,
                         shininess=128, reflectivity=0.1, transparency=0.9, ior=1.5)

    # 镜面球 (Reflection)
    mat_mirror = Material(color=np.array([1.0, 1.0, 1.0]), ka=0.0, kd=0.1, ks=0.9,
                          shininess=128, reflectivity=0.85)

    mat_mesh = Material(color=orange, ks=0.4, shininess=32)

    # 几何体
    left = Plane(normal=(-1, 0, 0), d=1.0, material=mat_left, class_name="left")
    right = Plane(normal=(1, 0, 0), d=1.0, material=mat_right, class_name="right")
    floor = Plane(normal=(0, 1, 0), d=0.0, material=mat_floor, class_name="floor")
    ceil = Plane(normal=(0, -1, 0), d=2.0, material=mat_ceil, class_name="ceiling")
    back = Plane(normal=(0, 0, 1), d=-3.0, material=mat_back, class_name="back")

    # 场景中的物体
    # 左前方的玻璃球
    glassS = Sphere(center=(-0.4, 0.4, -1.5), radius=0.4, material=mat_glass, class_name="small_sphere")
    # 右后方的镜面球
    mirrorS = Sphere(center=(0.5, 0.5, -2.2), radius=0.5, material=mat_mirror, class_name="big_sphere")

    objects = [left, right, floor, ceil, back, glassS, mirrorS]

    try:
        mesh = TriangleMesh(
            obj_path="model.obj",
            material=mat_mesh,
            scale=0.5,
            translation=(0.0, 0.25, -1.8),  # 放在两个球中间
            class_name="mesh_object",
        )
        objects.append(mesh)
    except Exception:
        pass

    lights = [
        # 主光源
        PointLight(position=np.array([0.0, 1.8, -1.5], dtype=np.float32),
                   color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                   Ia=0.1, Id=1.0, Is=1.0),
        # 辅助光
        PointLight(position=np.array([0.0, 0.5, 0.0], dtype=np.float32),
                   color=np.array([0.3, 0.3, 0.3], dtype=np.float32),
                   Ia=0.0, Id=0.2, Is=0.2),
    ]
    return Scene(objects, lights)


# ------------------------------
# Label/ID color map
# ------------------------------
CLASS_COLORS = {
    "back": (0, 255, 0),
    "left": (255, 0, 0),
    "right": (0, 0, 255),
    "floor": (0, 0, 255),
    "ceiling": (0, 0, 255),
    "small_sphere": (255, 165, 0),
    "big_sphere": (0, 128, 128),
    "mesh_object": (255, 0, 255),
    "object": (255, 255, 255),
    "background": (0, 0, 0),
}


def class_color_rgb01(name: str) -> np.ndarray:
    r, g, b = CLASS_COLORS.get(name, CLASS_COLORS["object"])
    return np.array([r, g, b], dtype=np.float32) / 255.0


# ------------------------------
# Projections & Render
# ------------------------------
def render(width=600, height=600, mode="labels", projection="orthographic",
           fov_deg=20.0, max_reflect_depth=3) -> Image.Image:  # 默认递归深度改为3
    scene = build_scene()
    img = np.zeros((height, width, 3), dtype=np.float32)

    cam_pos = np.array([0.0, 1.0, 2.5], dtype=np.float32)
    aspect = width / float(height)
    fov = math.radians(max(1.0, min(150.0, fov_deg)))

    for y in range(height):
        # 简单的进度打印，方便知道还没卡死
        if y % 50 == 0:
            print(f"Rendering line {y}/{height}...")

        ndc_y = 1 - 2 * ((y + 0.5) / height)
        py = math.tan(fov * 0.5) * ndc_y
        for x in range(width):
            ndc_x = 2 * ((x + 0.5) / width) - 1
            px = math.tan(fov * 0.5) * ndc_x * aspect

            if projection == "orthographic":
                ox = -1.0 + (ndc_x + 1.0) * 0.5 * 2.0
                oy = 2.0 - (ndc_y + 1.0) * 0.5 * 2.0
                ro = cam_pos + np.array([ox, oy - 1.0, 0.0], dtype=np.float32)
                rd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            else:
                ro = cam_pos
                rd = normalize(np.array([px, py, -1.0], dtype=np.float32))

            if mode == "labels":
                col = trace_labels(ro, rd, scene)
            else:
                # 开启递归，深度由界面控制
                col = trace_shaded(ro, rd, scene, depth=0, max_depth=max_reflect_depth)
            img[y, x] = col

    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img8)


# ------------------------------
# Tkinter UI
# ------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ray Tracing Project 3 - Shadows & Refraction")
        self.geometry("980x760")
        self.minsize(820, 600)

        self._pil_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        side = tk.Frame(self, padx=10, pady=10)
        side.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(side, text="Width:").grid(row=0, column=0, sticky="w")
        tk.Label(side, text="Height:").grid(row=1, column=0, sticky="w")
        self.width_var = tk.StringVar(value="400")  # 默认小一点以便快速测试
        self.height_var = tk.StringVar(value="400")
        tk.Entry(side, textvariable=self.width_var, width=8).grid(row=0, column=1, sticky="w")
        tk.Entry(side, textvariable=self.height_var, width=8).grid(row=1, column=1, sticky="w")

        self.use_ref_size = tk.BooleanVar(value=False)
        tk.Checkbutton(side, text="Use reference size", variable=self.use_ref_size).grid(row=2, column=0, columnspan=2,
                                                                                         sticky="w", pady=(4, 4))

        tk.Label(side, text="Render Mode:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.mode_var = tk.StringVar(value="shaded")
        ttk.Combobox(side, textvariable=self.mode_var, values=["labels", "shaded"], width=12, state="readonly").grid(
            row=3, column=1, sticky="w")

        tk.Label(side, text="Projection:").grid(row=4, column=0, sticky="w")
        self.proj_var = tk.StringVar(value="perspective")
        ttk.Combobox(side, textvariable=self.proj_var, values=["orthographic", "perspective"], width=12,
                     state="readonly").grid(row=4, column=1, sticky="w")

        tk.Label(side, text="FOV (deg):").grid(row=5, column=0, sticky="w")
        self.fov_var = tk.StringVar(value="45")
        tk.Entry(side, textvariable=self.fov_var, width=8).grid(row=5, column=1, sticky="w")

        tk.Label(side, text="Trace depth:").grid(row=6, column=0, sticky="w")
        self.rdepth_var = tk.StringVar(value="3")  # 默认3层递归
        tk.Entry(side, textvariable=self.rdepth_var, width=8).grid(row=6, column=1, sticky="w")

        self.render_btn = tk.Button(side, text="Render", command=self.on_render_click)
        self.render_btn.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 4))

        self.save_btn = tk.Button(side, text="Save Image...", command=self.on_save_click, state=tk.DISABLED)
        self.save_btn.grid(row=8, column=0, columnspan=2, sticky="we", pady=(4, 10))

        self.open_btn = tk.Button(side, text="Open Image (Ref)...", command=self.on_open_click)
        self.open_btn.grid(row=9, column=0, columnspan=2, sticky="we")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(side, textvariable=self.status_var, anchor="w", justify="left", wraplength=200).grid(row=10, column=0,
                                                                                                      columnspan=2,
                                                                                                      sticky="we",
                                                                                                      pady=(10, 0))

        self.preview = tk.Label(self, bg="#cccccc")
        self.preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
            rdepth = max(0, min(10, rdepth))
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
                print(e)
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