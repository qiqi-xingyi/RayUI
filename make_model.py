# --*-- conding:utf-8 --*--
# @time:11/12/25 19:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:make_model.py

# Generate a simple cube mesh as triangles and save to model.obj

verts = [
    (-0.5, -0.5, -0.5),  # 1
    ( 0.5, -0.5, -0.5),  # 2
    ( 0.5,  0.5, -0.5),  # 3
    (-0.5,  0.5, -0.5),  # 4
    (-0.5, -0.5,  0.5),  # 5
    ( 0.5, -0.5,  0.5),  # 6
    ( 0.5,  0.5,  0.5),  # 7
    (-0.5,  0.5,  0.5),  # 8
]

# Each face is split into 2 triangles, indices are 1-based for OBJ
faces = [
    # front (z = -0.5)
    (1, 2, 3),
    (1, 3, 4),
    # back (z = 0.5)
    (5, 7, 6),
    (5, 8, 7),
    # left (x = -0.5)
    (1, 4, 8),
    (1, 8, 5),
    # right (x = 0.5)
    (2, 6, 7),
    (2, 7, 3),
    # bottom (y = -0.5)
    (1, 5, 6),
    (1, 6, 2),
    # top (y = 0.5)
    (4, 3, 7),
    (4, 7, 8),
]

if __name__ == '__main__':


    with open("model.obj", "w") as f:
        f.write("# Simple cube made of triangles\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print("model.obj written.")
