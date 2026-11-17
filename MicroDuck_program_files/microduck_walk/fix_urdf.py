"""Make a physics-ready copy of microduck.urdf (the original is not touched).

The exported URDF has no <collision> tags and every link has mass 0.01 kg and
inertia 1e-5. This script:
  * computes each link's volume / centre of mass / inertia from the convex hull
    of its visual STL meshes (uniform density),
  * scales the density so the whole robot weighs --total_mass kg,
  * adds <collision> tags that reuse the visual meshes,
  * writes absolute mesh paths so the file can live anywhere.

  python fix_urdf.py --total_mass 0.5
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = r"C:\Users\punn_\OneDrive\Desktop\IsaacSim_PK\microduck_description-main\urdf\microduck.urdf"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=DEFAULT_SRC)
ap.add_argument("--dst", default=os.path.join(HERE, "assets", "microduck_fixed.urdf"))
ap.add_argument("--total_mass", type=float, default=0.5, help="real robot weight in kg")
ap.add_argument("--min_mass", type=float, default=0.002)
args = ap.parse_args()

src_dir = os.path.dirname(os.path.abspath(args.src))
tree = ET.parse(args.src)
root = tree.getroot()


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def origin_tf(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T[:3, :3] = rpy_matrix(*rpy)
        T[:3, 3] = xyz
    return T


def abs_mesh(fn):
    if fn.startswith("package://"):
        fn = fn.split("/", 3)[-1]
    if not os.path.isabs(fn):
        fn = os.path.normpath(os.path.join(src_dir, fn))
    return fn


hulls = {}  # per link: one trimesh whose mass properties stand for the link's material
n_closed = n_hull = 0
for link in root.findall("link"):
    parts = []
    for vis in link.findall("visual"):
        mesh_el = vis.find("geometry/mesh")
        if mesh_el is None:
            continue
        path = abs_mesh(mesh_el.get("filename"))
        mesh_el.set("filename", path.replace("\\", "/"))
        m = trimesh.load(path, force="mesh")
        scale = [float(v) for v in mesh_el.get("scale", "1 1 1").split()]
        m.apply_scale(scale)
        m.apply_transform(origin_tf(vis))
        # closed mesh -> its real material volume (thin printed shells stay light,
        # solid servos stay heavy); open mesh -> fall back to its convex hull
        if m.is_volume and m.volume > 0:
            parts.append(m)
            n_closed += 1
        else:
            parts.append(m.convex_hull)
            n_hull += 1
        # collision = same geometry as the visual
        col = ET.SubElement(link, "collision")
        for child in vis:
            if child.tag in ("origin", "geometry"):
                col.append(ET.fromstring(ET.tostring(child)))
    if parts:
        # volume integrals are additive, so concatenating closed meshes gives the summed mass properties
        hulls[link.get("name")] = trimesh.util.concatenate(parts)

total_vol = sum(h.volume for h in hulls.values())
rho = args.total_mass / total_vol
print(f"links with meshes: {len(hulls)}, meshes closed {n_closed} / hull fallback {n_hull}")
print(f"material volume {total_vol * 1e6:.1f} cm^3, density {rho:.0f} kg/m^3")

for link in root.findall("link"):
    name = link.get("name")
    for old in link.findall("inertial"):
        link.remove(old)
    h = hulls.get(name)
    if h is None:
        mass, com, I = args.min_mass, np.zeros(3), np.eye(3) * 1e-7
    else:
        mass = max(rho * h.volume, args.min_mass)
        com = h.center_mass
        I = h.moment_inertia * (mass / h.volume)  # trimesh uses density 1 -> mass == volume
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.6f}" for v in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{I[0, 0]:.3e}", ixy=f"{I[0, 1]:.3e}", ixz=f"{I[0, 2]:.3e}",
        iyy=f"{I[1, 1]:.3e}", iyz=f"{I[1, 2]:.3e}", izz=f"{I[2, 2]:.3e}",
    )
    print(f"  {name:22s} mass {mass * 1000:7.1f} g  com {np.round(com, 4)}")

os.makedirs(os.path.dirname(args.dst), exist_ok=True)
tree.write(args.dst, encoding="utf-8", xml_declaration=True)
print("wrote", args.dst)
