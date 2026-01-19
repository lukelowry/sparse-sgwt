import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    @lru_cache(maxsize=8)
    def from_ply(cls, filepath: str) -> "Mesh":
        with open(filepath, 'rb') as f:
            fmt, vertex_count, face_count, vertex_props, current_element = "ascii", 0, 0, [], None
            while (line := f.readline().strip()):
                line_str = line.decode('ascii', errors='ignore')
                if line_str == "end_header": break
                parts = line_str.split()
                if not parts: continue
                if parts[0] == "format": fmt = parts[1]
                elif parts[0] == "element":
                    current_element = parts[1]
                    if current_element == "vertex": vertex_count = int(parts[2])
                    elif current_element == "face": face_count = int(parts[2])
                elif parts[0] == "property" and current_element == "vertex":
                    vertex_props.append((parts[2], parts[1]))

            if fmt == "ascii":
                lines = f.readlines()
                vertices = np.array([list(map(float, lines[i].split()[:3])) for i in range(vertex_count)], dtype=np.float32)
                faces = np.array([list(map(int, lines[vertex_count + i].split()[1:4])) for i in range(face_count)], dtype=np.int32)
            elif fmt == "binary_little_endian":
                np_type_map = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8'}
                dtype = np.dtype([(name, np_type_map.get(t, 'f4')) for name, t in vertex_props])
                vertex_data = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype)
                names = vertex_data.dtype.names
                keys = ('x', 'y', 'z') if {'x', 'y', 'z'}.issubset(names) else names[:3]
                vertices = np.column_stack([vertex_data[k] for k in keys]).astype(np.float32)
                face_data = np.frombuffer(f.read(face_count * 13), dtype=np.dtype([('n', 'u1'), ('v', '3i4')]))
                faces = face_data['v']
            else:
                raise ValueError(f"Unsupported PLY format: {fmt}")
        return cls(vertices=vertices, faces=faces)


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def get_luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def is_dark_center_cmap(cmap) -> bool:
    center_color = cmap(0.5)[:3]
    return get_luminance(np.array(center_color)) < 0.4


def apply_shading_multiplicative(base_colors: np.ndarray, normals: np.ndarray, 
                                  light_dir: np.ndarray, ambient: float = 0.4, diffuse: float = 0.6) -> np.ndarray:
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.abs(normals @ light_dir)
    shade = ambient + diffuse * intensity
    shaded = base_colors.copy()
    shaded[:, :3] *= shade[:, np.newaxis]
    return np.clip(shaded, 0, 1)


def apply_shading_additive(base_colors: np.ndarray, normals: np.ndarray, 
                           light_dir: np.ndarray, strength: float = 0.35) -> np.ndarray:
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.abs(normals @ light_dir)
    shading = (intensity - 0.5) * 2 * strength
    shaded = base_colors.copy()
    shaded[:, :3] = base_colors[:, :3] + shading[:, np.newaxis]
    return np.clip(shaded, 0, 1)


def plot_mesh_wavelet(signal: np.ndarray, ply_path: str, title: str, output_filename: Path,
                      cmap: str = 'RdBu_r', elev: int = 20, azims: list = [-120, 0, 120],
                      mesh_rotation: tuple = (0, 0, 0), light_dir: np.ndarray = None, 
                      zoom: float = 1.5):
    print(f"Loading mesh from {ply_path}...")
    mesh = Mesh.from_ply(ply_path)
    vertices, faces = mesh.vertices.copy(), mesh.faces
    
    signal = np.asarray(signal).flatten()
    if len(signal) != len(vertices):
        raise ValueError(f"Signal length ({len(signal)}) != Vertices ({len(vertices)})")
    
    if any(mesh_rotation):
        R = rotation_matrix(*mesh_rotation)
        vertices = vertices @ R.T
    
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    
    print(f"Generating 3D surface for '{title}'...")
    
    face_verts = vertices[faces]
    face_normals = compute_face_normals(vertices, faces)
    face_values = signal[faces].mean(axis=1)
    
    vmax = np.abs(signal).max() or 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    base_colors = colormap(norm(face_values))
    
    dark_cmap = is_dark_center_cmap(colormap)
    
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    max_range = (maxs - mins).max() / 2 / zoom
    mid = (maxs + mins) / 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=-0.1)
    
    for ax, azim in zip(np.atleast_1d(axes), azims):
        azim_rad, elev_rad = np.radians(azim), np.radians(elev)
        view_light = light_dir if light_dir is not None else np.array([
            np.cos(elev_rad) * np.sin(azim_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad)
        ])
        
        if dark_cmap:
            shaded_colors = apply_shading_additive(base_colors, face_normals, view_light)
        else:
            shaded_colors = apply_shading_multiplicative(base_colors, face_normals, view_light)
        
        # Edge colors match face colors exactly for seamless blending
        poly = Poly3DCollection(face_verts, facecolors=shaded_colors, 
                                edgecolors=shaded_colors, linewidths=0, antialiased=False)
        ax.add_collection3d(poly)
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    
    fig.suptitle(title, fontsize=18, y=0.98)
    
    out_path = Path(output_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=400)
    plt.close(fig)
    print(f"Plot saved to {out_path}")