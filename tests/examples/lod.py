from fpy2 import *

from math import fabs, sqrt, log2

fmax = max

@fpy
def lod_anisotropic(dx_u, dx_v, dy_u, dy_v, max_aniso):
    dx2 = dx_u ** 2 + dx_v ** 2
    dy2 = dy_u ** 2 + dy_v ** 2
    det = fabs(dx_u * dy_v - dx_v * dy_u)
    x_major = dx2 > dy2
    major2 = dx2 if x_major else dy2
    major = sqrt(major2)
    norm_major = 1.0 / major

    aniso_dir_u = (dx_u if x_major else dy_u) * norm_major
    aniso_dir_v = (dx_v if x_major else dy_v) * norm_major
    aniso_ratio = major2 / det

    # clamp anisotropy ratio and compute LOD
    if aniso_ratio > max_aniso:
        aniso_ratio = max_aniso
        minor = major / aniso_ratio
    else:
        minor = det / major
    
    # clamp LOD
    if minor < 1.0:
        aniso_ratio = fmax(1.0, aniso_ratio * minor)

    lod = log2(minor)
    return lod, aniso_ratio, aniso_dir_u, aniso_dir_v


# from CMU's Scotty3D renderer
w = 256
h = 256
dxdu = 0.00173199
dxdv = -5.96046e-08
dydu = -1.19209e-07
dydv = 0.00173205

lod_anisotropic(w * dxdu, h * dxdv, w * dydu, h * dydv, 16.0)
