"""Mapa de densidad sobre el mapa de calles de Esri llevado a grises.

Solo necesita el GeoJSON de hexágonos: lo genera GitHub Actions (`.github/workflows/mapa.yml`),
que sí tiene acceso a las teselas del mapa base.

python scripts/mapa_densidad.py --densidad resultados/densidad_la_mariposa_v2.geojson --salida docs/img/densidad.png
"""
from __future__ import annotations

import argparse

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TINTA = "#1d1d1b"
NARANJA = ["#fde3d3", "#f6a173", "#eb6834", "#b8401a", "#6e2208"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densidad", required=True)
    ap.add_argument("--salida", default="docs/img/densidad.png")
    ap.add_argument("--margen", type=float, default=60.0, help="metros alrededor de los hexágonos")
    a = ap.parse_args()

    hx = gpd.read_file(a.densidad)
    hx = hx[hx["puestos"] > 0].to_crs(3857)
    xmin, ymin, xmax, ymax = hx.total_bounds
    m = a.margen

    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    ax.set_xlim(xmin - m, xmax + m)
    ax.set_ylim(ymin - m, ymax + m)
    cmap = LinearSegmentedColormap.from_list("naranja", NARANJA)
    hx.plot(ax=ax, column="puestos", cmap=cmap, alpha=0.82, edgecolor="none", linewidth=0, zorder=2,
            legend=True, legend_kwds={"label": "puestos por hexágono de 585 m²", "shrink": 0.5})
    img, ext = cx.bounds2img(xmin - m, ymin - m, xmax + m, ymax + m, zoom=18,
                             source=cx.providers.Esri.WorldStreetMap)
    rgb = img[..., :3].astype(float)
    if rgb.std() < 8 or (rgb.mean(-1) > 245).mean() > 0.97:
        raise SystemExit("El servidor del mapa base devolvió teselas vacías o de error: no se publica la figura.")
    gris = rgb.mean(-1, keepdims=True)
    base = 0.15 * rgb + 0.85 * gris            # casi sin color: el color es de los datos
    base = 255 - (255 - base) * 0.75           # más claro
    ax.imshow(base.clip(0, 255).astype(np.uint8), extent=ext, zorder=1, interpolation="lanczos")
    ax.set_xlim(xmin - m, xmax + m)
    ax.set_ylim(ymin - m, ymax + m)
    ax.set_axis_off()
    ax.set_title("Dónde se concentran los puestos · La Mariposa, Bogotá", loc="left", fontsize=13,
                 fontweight="bold", color=TINTA)
    ax.text(0.01, 0.006, "Mapa base: Esri, HERE, Garmin, © OpenStreetMap contributors · Puestos: UrbanVendor-Vision v2",
            transform=ax.transAxes, fontsize=6.5, color="#52514e", zorder=4,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
