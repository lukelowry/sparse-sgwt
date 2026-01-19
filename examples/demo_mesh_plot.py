import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from sgwt.util import _parse_ply, _load_resource


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


def get_bundled_ply_path(mesh_name: str) -> str:
    return _load_resource(f"library/MESH/{mesh_name}.ply", lambda p: p)


def load_and_prepare_mesh(mesh: str, signal: np.ndarray, mesh_rotation: Tuple[float, float, float]):
    if mesh.upper() in ('BUNNY', 'HORSE', 'LBRAIN', 'ENGINE'):
        ply_path = get_bundled_ply_path(mesh.upper())
    else:
        ply_path = mesh
    
    print(f"Loading mesh from {ply_path}...")
    verts_list, faces_list, _ = _parse_ply(ply_path)
    vertices = np.array(verts_list, dtype=np.float32)
    faces = np.array([f[:3] for f in faces_list], dtype=np.int32)
    signal = np.asarray(signal).flatten()
    
    if len(signal) != len(vertices):
        raise ValueError(f"Signal length ({len(signal)}) != Vertices ({len(vertices)})")
    
    if any(mesh_rotation):
        R = rotation_matrix(*mesh_rotation)
        vertices = vertices @ R.T
    
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    return vertices, faces, signal


def compute_view_light(azim: float, elev: float) -> np.ndarray:
    azim_rad, elev_rad = np.radians(azim), np.radians(elev)
    return np.array([
        np.cos(elev_rad) * np.sin(azim_rad),
        np.cos(elev_rad) * np.cos(azim_rad),
        np.sin(elev_rad)
    ])


def compute_shaded_face_colors(vertices: np.ndarray, faces: np.ndarray, signal: np.ndarray,
                                cmap: str, azim: float, elev: float, 
                                light_dir: Optional[np.ndarray] = None) -> np.ndarray:
    face_normals = compute_face_normals(vertices, faces)
    face_values = signal[faces].mean(axis=1)
    
    vmax = np.abs(signal).max() or 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    base_colors = colormap(norm(face_values))
    dark_cmap = is_dark_center_cmap(colormap)
    
    view_light = light_dir if light_dir is not None else compute_view_light(azim, elev)
    
    if dark_cmap:
        shaded_colors = apply_shading_additive(base_colors, face_normals, view_light)
    else:
        shaded_colors = apply_shading_multiplicative(base_colors, face_normals, view_light)
    
    return shaded_colors


# =============================================================================
# MATPLOTLIB BACKEND
# =============================================================================

def plot_mesh_wavelet_matplotlib(signal: np.ndarray, mesh: str, title: str, output_filename: Path,
                                  cmap: str = 'RdBu_r', elev: int = 20, azims: List[int] = [-120, 0, 120],
                                  mesh_rotation: Tuple[float, float, float] = (0, 0, 0), 
                                  light_dir: Optional[np.ndarray] = None, zoom: float = 1.5):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    vertices, faces, signal = load_and_prepare_mesh(mesh, signal, mesh_rotation)
    
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
        view_light = light_dir if light_dir is not None else compute_view_light(azim, elev)
        
        if dark_cmap:
            shaded_colors = apply_shading_additive(base_colors, face_normals, view_light)
        else:
            shaded_colors = apply_shading_multiplicative(base_colors, face_normals, view_light)
        
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


# =============================================================================
# PYVISTA BACKEND - render each view separately and combine
# =============================================================================
def plot_mesh_wavelet_pyvista(signal: np.ndarray, mesh: str, title: str, output_filename: Path,
                               cmap: str = 'RdBu_r', elev: int = 20, azims: List[int] = [-120, 0, 120],
                               mesh_rotation: Tuple[float, float, float] = (0, 0, 0),
                               light_dir: Optional[np.ndarray] = None, zoom: float = 1.5,
                               window_size: Tuple[int, int] = (800, 800)):
    if not HAS_PYVISTA:
        raise ImportError("PyVista required: pip install pyvista")
    
    from PIL import Image
    
    vertices, faces, signal = load_and_prepare_mesh(mesh, signal, mesh_rotation)
    
    print(f"Generating 3D surface for '{title}'...")
    
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    center = (maxs + mins) / 2
    
    # Use the diagonal of the bounding box to ensure full visibility from any angle
    diagonal = np.linalg.norm(maxs - mins)
    
    # parallel_scale controls the visible half-height in world units
    # Divide by zoom to match matplotlib behavior (higher zoom = smaller scale = more zoomed in)
    parallel_scale = diagonal / 2 / zoom
    
    distance = diagonal * 2
    
    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces]).ravel()
    
    images = []
    
    for azim in azims:
        shaded_colors = compute_shaded_face_colors(vertices, faces, signal, cmap, azim, elev, light_dir)
        
        pv_mesh = pv.PolyData(vertices.copy(), pv_faces.copy())
        pv_mesh.cell_data['colors'] = (shaded_colors[:, :3] * 255).astype(np.uint8)
        
        plotter = pv.Plotter(off_screen=True, window_size=window_size, border=False)
        plotter.add_mesh(pv_mesh, scalars='colors', rgb=True, lighting=False, show_scalar_bar=False)
        
        azim_rad = np.radians(-azim)
        elev_rad = np.radians(elev)
        
        cam_x = center[0] + distance * np.cos(elev_rad) * np.sin(azim_rad)
        cam_y = center[1] + distance * np.cos(elev_rad) * np.cos(azim_rad)
        cam_z = center[2] + distance * np.sin(elev_rad)
        
        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.up = (0, 0, 1)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        plotter.camera.clipping_range = (0.01, distance * 100)
        plotter.set_background('white')
        
        img = plotter.screenshot(return_img=True)
        images.append(img)
        plotter.close()
    
    combined = np.concatenate(images, axis=1)
    
    if title:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (pil_img.width - text_width) // 2
        draw.text((x, 10), title, fill='black', font=font)
        combined = np.array(pil_img)
    
    out_path = Path(output_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(out_path, dpi=(400, 400))
    print(f"Plot saved to {out_path}")


# =============================================================================
# UNIFIED API
# =============================================================================

def plot_mesh_wavelet(signal: np.ndarray, mesh: str, title: str, output_filename: Path,
                      cmap: str = 'RdBu_r', elev: int = 20, azims: List[int] = [-120, 0, 120],
                      mesh_rotation: Tuple[float, float, float] = (0, 0, 0),
                      light_dir: Optional[np.ndarray] = None, zoom: float = 1.5,
                      backend: str = 'auto', **kwargs):
    """
    Plot a mesh wavelet visualization.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal values at each vertex.
    mesh : str
        Either a bundled mesh name ('BUNNY', 'HORSE', 'LBRAIN') or a path to a .ply file.
    title : str
        Title for the plot.
    output_filename : Path
        Output path for the saved image.
    cmap : str
        Colormap name.
    elev : int
        Elevation angle for viewing.
    azims : list
        List of azimuth angles for multiple views.
    mesh_rotation : tuple
        Rotation angles (rx, ry, rz) in degrees.
    light_dir : np.ndarray
        Light direction vector (None = auto from view angle).
    zoom : float
        Zoom factor.
    backend : str
        'auto' (use pyvista if available), 'pyvista', or 'matplotlib'.
    """
    if backend == 'auto':
        backend = 'pyvista' if HAS_PYVISTA else 'matplotlib'
    
    if backend == 'pyvista':
        plot_mesh_wavelet_pyvista(signal, mesh, title, output_filename, cmap=cmap,
                                   elev=elev, azims=azims, mesh_rotation=mesh_rotation,
                                   light_dir=light_dir, zoom=zoom, **kwargs)
    else:
        plot_mesh_wavelet_matplotlib(signal, mesh, title, output_filename, cmap=cmap,
                                      elev=elev, azims=azims, mesh_rotation=mesh_rotation,
                                      light_dir=light_dir, zoom=zoom)