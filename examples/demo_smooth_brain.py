import numpy as np
from pathlib import Path
import pyvista as pv

import sgwt
from sgwt import Convolve
from demo_mesh_plot import plot_mesh_on_plotter

def smooth_grid(outpng,
                scales=(0.1, 1.0, 10.0, 100.0, 500.0),
                tile=420, cmap="viridis", background="white"):

    rows = [
        dict(name="BUNNY",  L=sgwt.MESH_BUNNY,  V=sgwt.BUNNY_XYZ,  order=3,
             azim=-120, elev=20, zoom=1.3, rot=(90, 0, 90)),
        dict(name="HORSE",  L=sgwt.MESH_HORSE,  V=sgwt.HORSE_XYZ,  order=1,
             azim=-120, elev=10, zoom=1.3, rot=(0, 0, 180)),
        dict(name="LBRAIN", L=sgwt.MESH_LBRAIN, V=sgwt.LBRAIN_XYZ, order=1,
             azim=-120, elev=20, zoom=1.3, rot=(0, 0, 180)),
    ]

    nrows, ncols = len(rows), len(scales)
    plotter = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                         window_size=(ncols * tile, nrows * tile),
                         border=False)

    light_dir = np.array([0.3, 0.3, 1.0], dtype=float)

    for r, cfg in enumerate(rows):
        V0 = np.asarray(cfg["V"], dtype=np.float64)
        B = V0.copy(order="F")  # (n_vertices, n_timesteps) Fortran order recommended [1]

        with Convolve(cfg["L"]) as conv:
            V_list = conv.lowpass(B, scales=list(scales), order=cfg["order"])  # list per scale [1]

        V0f = V0.astype(np.float32)
        for c, Vlp in enumerate(V_list):
            Vlp = np.asarray(Vlp, dtype=np.float32)
            disp = np.linalg.norm(Vlp - V0f, axis=1)

            plotter.subplot(r, c)
            plot_mesh_on_plotter(
                plotter,
                vertices=Vlp,
                signal=disp,
                mesh=cfg["name"],
                cmap=cmap,
                azim=cfg["azim"],
                elev=cfg["elev"],
                mesh_rotation=cfg["rot"],
                light_dir=light_dir,
                zoom=cfg["zoom"],
                background=background,
            )

    out = Path(outpng)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()

smooth_grid("out/mesh_smoothing_grid.png")