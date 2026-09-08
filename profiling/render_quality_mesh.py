"""Diagnostic views of unchanged exported meshes, outside benchmark timing."""

import argparse

import numpy as np
import pyvista as pv
import trimesh

p = argparse.ArgumentParser()
p.add_argument("mesh")
p.add_argument("output")
a = p.parse_args()
m = trimesh.load_mesh(a.mesh)
print("bounds", m.bounds, "watertight", m.is_watertight)
poly = pv.PolyData(
    m.vertices, np.column_stack((np.full(len(m.faces), 3), m.faces)).ravel()
)
plot = pv.Plotter(off_screen=True, window_size=(1000, 1000))
plot.set_background("white")
plot.add_mesh(
    poly,
    color="lightgray",
    smooth_shading=True,
    ambient=0.2,
    diffuse=0.75,
    specular=0.15,
)
plot.view_xy()
plot.camera.zoom(1.1)
plot.show(screenshot=a.output)
