"""Mapa de densidad: mapa de calles + ortomosaico + hexágonos, con grilla de coordenadas, escala y norte.

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
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as Poly, Rectangle

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


def flecha_norte(ax, x: float, y: float, alto: float):
    """Norte de cartografía clásica: punta partida en dos tonos y una N fina encima."""
    w = alto * 0.36
    ax.add_patch(Rectangle((x - w * 1.1, y - alto * 0.18), w * 2.2, alto * 1.75, facecolor="white", alpha=0.85,
                           edgecolor="none", zorder=5))
    ax.add_patch(Poly([(x, y + alto), (x - w / 2, y), (x, y + alto * 0.28)], closed=True, facecolor=TINTA,
                      edgecolor=TINTA, linewidth=0.8, zorder=6))
    ax.add_patch(Poly([(x, y + alto), (x + w / 2, y), (x, y + alto * 0.28)], closed=True, facecolor="white",
                      edgecolor=TINTA, linewidth=0.8, zorder=6))
    ax.text(x, y + alto * 1.08, "N", ha="center", va="bottom", fontsize=10, color=TINTA, zorder=6)


def leyenda(ax, fig, cmap, norm):
    """Barra de color horizontal dentro del mapa, abajo a la derecha, sobre una caja blanca."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0, w, h = 0.60, 0.045, 0.34, 0.022
    ax.add_patch(FancyBboxPatch((x0 - 0.02, y0 - 0.032), w + 0.04, 0.105, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.008", facecolor="white", alpha=0.9,
                                edgecolor="none", zorder=5))
    ax.text(x0, y0 + h + 0.018, "Puestos por hexágono de 585 m²", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", color=TINTA, zorder=6)
    cax = ax.inset_axes([x0, y0, w, h], zorder=6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=TINTA, length=2)
    cb.set_ticks([1, 5, 10, 15, 20, 25][: 1 + sum(v <= norm.vmax for v in (5, 10, 15, 20, 25))])


def grilla(ax, xmin, ymin, xmax, ymax, paso: float):
    """Grilla de coordenadas planas (m) con rótulos en el borde, al estilo de una plancha."""
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(paso))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(paso))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(True, color="#5f5e5a", linewidth=0.45, alpha=0.6, linestyle=(0, (4, 3)), zorder=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=6.5, colors="#52514e", length=3, direction="out", top=True, right=True,
                   labeltop=True, labelright=False)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    for sp in ax.spines.values():
        sp.set_color("#52514e")
        sp.set_linewidth(0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densidad", required=True)
    ap.add_argument("--fondo", default="docs/base/la_mariposa_35cm.tif")
    ap.add_argument("--salida", default="docs/img/densidad.png")
    ap.add_argument("--color-fondo", type=float, default=0.75, help="0 = gris, 1 = color original")
    ap.add_argument("--sin-mapa-base", action="store_true", help="para pruebas sin internet")
    ap.add_argument("--margen", type=float, default=90.0, help="m de mapa de calles alrededor del ortomosaico")
    ap.add_argument("--paso-grilla", type=float, default=100.0)
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

    m = a.margen
    xmin, ymin, xmax, ymax = b.left - m, b.bottom - m, b.right + m, b.top + m
    ancho_fig = 10.0  # el mapa ocupa todo el ancho: la leyenda va dentro
    fig, ax = plt.subplots(figsize=(ancho_fig, ancho_fig * (ymax - ymin) / (xmax - xmin)))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if not a.sin_mapa_base:
        import contextily as cx

        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldStreetMap, zoom=18, attribution=False,
                       zorder=0, reset_extent=True)
        for im in ax.get_images():  # mapa de calles en grises claros: es contexto, no protagonista
            arr = np.asarray(im.get_array()).astype(float)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                if arr.max() <= 1:
                    arr = arr * 255
                rgbm = arr[..., :3]
                g = rgbm.mean(-1, keepdims=True)
                rgbm = 255 - (255 - (0.2 * rgbm + 0.8 * g)) * 0.7
                im.set_data(rgbm.clip(0, 255).astype(np.uint8))
    ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), zorder=1, interpolation="lanczos")

    # Opacidad creciente con la densidad: lo poco denso deja ver el terreno, los focos quedan sólidos
    cmap = LinearSegmentedColormap.from_list("naranja", NARANJA)
    norm = plt.Normalize(vmin=1, vmax=float(hx["puestos"].max()))
    t = norm(hx["puestos"].to_numpy())
    colores = cmap(t)
    colores[:, 3] = 0.30 + 0.55 * np.sqrt(t)
    hx.plot(ax=ax, color=colores, edgecolor="none", linewidth=0, zorder=2)
    leyenda(ax, fig, cmap, norm)

    ancho, alto = xmax - xmin, ymax - ymin
    largo = 100 if ancho > 300 else 50
    escala_grafica(ax, largo, xmin + 0.05 * ancho, ymin + 0.035 * alto, alto_m=ancho * 0.010)
    flecha_norte(ax, xmax - 0.07 * ancho, ymax - 0.12 * alto, alto=ancho * 0.055)
    grilla(ax, xmin, ymin, xmax, ymax, a.paso_grilla)
    ax.set_title("Dónde se concentran los puestos · La Mariposa, Bogotá", loc="left", fontsize=13, pad=22,
                 fontweight="bold", color=TINTA)
    ax.text(0.0, -0.045, "Ortomosaico: DJI Mavic Air 2, 2 dic. 2021 · Mapa base: Esri, HERE, Garmin, © OpenStreetMap"
            " contributors · Coordenadas MAGNA-SIRGAS Bogotá (EPSG:3116), grilla cada "
            f"{a.paso_grilla:.0f} m", transform=ax.transAxes, fontsize=6.5, color="#52514e", va="top")
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
