"""Mapa de densidad de puestos sobre el ortomosaico, con escala gráfica y norte.

El fondo es una copia liviana del ortomosaico (docs/base/la_mariposa_35cm.tif, 35 cm/píxel):
el original de 3,45 cm pesa 1 GB y no cabe en el repositorio.

python scripts/mapa_densidad.py --densidad resultados/densidad_la_mariposa_v2.geojson \
       --fondo docs/base/la_mariposa_35cm.tif --salida docs/img/densidad.png
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, Rectangle

TINTA = "#1d1d1b"
NARANJA = ["#fde3d3", "#f6a173", "#eb6834", "#b8401a", "#6e2208"]


def escala_grafica(ax, largo_m: float, x0: float, y0: float, alto_m: float, tramos: int = 4):
    """Barra blanco/negro en unidades del mapa (metros), con caja de fondo y rótulos."""
    paso = largo_m / tramos
    pad = alto_m * 1.6
    ax.add_patch(Rectangle((x0 - pad, y0 - pad * 1.1), largo_m + 2 * pad, alto_m + pad * 3.4,
                           facecolor="white", alpha=0.85, edgecolor="none", zorder=5))
    for i in range(tramos):
        ax.add_patch(Rectangle((x0 + i * paso, y0), paso, alto_m, facecolor=TINTA if i % 2 == 0 else "white",
                               edgecolor=TINTA, linewidth=0.8, zorder=6))
    for i in (0, tramos // 2, tramos):
        rot = f"{i * paso:.0f}" + (" m" if i == tramos else "")
        ax.text(x0 + i * paso, y0 + alto_m * 1.5, rot, ha="center", va="bottom", fontsize=8, color=TINTA, zorder=6)


def flecha_norte(ax, x: float, y: float, largo: float):
    ax.add_patch(FancyArrow(x, y, 0, largo, width=largo * 0.12, head_width=largo * 0.42,
                            head_length=largo * 0.45, length_includes_head=True, color=TINTA, zorder=6))
    ax.text(x, y + largo * 1.12, "N", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TINTA, zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densidad", required=True)
    ap.add_argument("--fondo", default="docs/base/la_mariposa_35cm.tif")
    ap.add_argument("--salida", default="docs/img/densidad.png")
    ap.add_argument("--color-fondo", type=float, default=0.55, help="0 = gris, 1 = color original")
    a = ap.parse_args()

    with rasterio.open(a.fondo) as src:
        img = src.read().astype(float)
        b = src.bounds
        crs = src.crs
    rgb = img[:3].transpose(1, 2, 0)
    gris = rgb.mean(-1, keepdims=True)
    rgb = a.color_fondo * rgb + (1 - a.color_fondo) * gris   # algo apagado: el color fuerte es de los datos
    rgba = np.dstack([rgb.clip(0, 255) / 255, img[3] / 255])

    hx = gpd.read_file(a.densidad).to_crs(crs)
    hx = hx[hx["puestos"] > 0]

    fig, ax = plt.subplots(figsize=(7.8, 9.0))
    ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), zorder=1, interpolation="lanczos")

    # Opacidad creciente con la densidad: lo poco denso deja ver el terreno, los focos quedan sólidos
    cmap = LinearSegmentedColormap.from_list("naranja", NARANJA)
    norm = plt.Normalize(vmin=1, vmax=float(hx["puestos"].max()))
    t = norm(hx["puestos"].to_numpy())
    colores = cmap(t)
    colores[:, 3] = 0.30 + 0.55 * np.sqrt(t)
    hx.plot(ax=ax, color=colores, edgecolor="none", linewidth=0, zorder=2)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.45, pad=0.02,
                 label="puestos por hexágono de 585 m²")

    ancho = b.right - b.left
    largo = 100 if ancho > 300 else 50
    escala_grafica(ax, largo, b.left + 0.05 * ancho, b.bottom + 0.035 * (b.top - b.bottom), alto_m=ancho * 0.012)
    flecha_norte(ax, b.right - 0.07 * ancho, b.top - 0.11 * (b.top - b.bottom), largo=ancho * 0.06)

    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_axis_off()
    ax.set_title("Dónde se concentran los puestos · La Mariposa, Bogotá", loc="left", fontsize=13,
                 fontweight="bold", color=TINTA)
    ax.text(0.0, -0.012, "Ortomosaico: vuelo DJI Mavic Air 2, 2 de diciembre de 2021 · MAGNA-SIRGAS Bogotá (EPSG:3116)"
            " · Puestos: UrbanVendor-Vision v2", transform=ax.transAxes, fontsize=7, color="#52514e", va="top")
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
