"""Mapas de puestos: mapa de calles + ortomosaico + datos, con grilla de coordenadas, escala y norte.

Tres mapas con el mismo formato: la segmentación (máscaras) con una vista de detalle, los puestos uno por
uno (nodos) y la densidad por hexágono.

El fondo es una copia liviana del ortomosaico (docs/base/la_mariposa_35cm.tif, 35 cm/píxel):
el original de 3,45 cm pesa 1 GB y no cabe en el repositorio.

python scripts/mapa_densidad.py --densidad resultados/densidad_la_mariposa_v2.geojson \
       --puntos resultados/centroides_la_mariposa_v2.geojson --fondo docs/base/la_mariposa_35cm.tif
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


def leyenda_puntos(ax, n, diametros, tam):
    """Caja blanca abajo a la derecha: título con el total y tres tamaños de referencia."""
    from matplotlib.patches import FancyBboxPatch

    x0, y0 = 0.60, 0.045
    ax.add_patch(FancyBboxPatch((x0 - 0.02, y0 - 0.032), 0.38, 0.105, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.008", facecolor="white", alpha=0.9,
                                edgecolor="none", zorder=5))
    ax.text(x0, y0 + 0.04, f"{n} puestos detectados · tamaño según diámetro", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=TINTA, zorder=6)
    for k, d in enumerate(diametros):
        x = x0 + 0.02 + k * 0.115
        ax.scatter([x], [y0 + 0.008], s=[tam(d)], transform=ax.transAxes, color=NARANJA[2],
                   edgecolor="white", linewidth=0.6, zorder=6)
        ax.text(x + 0.018, y0 + 0.008, f"{d:.1f} m".replace(".", ","), transform=ax.transAxes, fontsize=7.5,
                va="center", color=TINTA, zorder=6)


def mascaras(ax, gdf, lw=0.6):
    """Máscaras de segmentación: relleno naranja translúcido y contorno oscuro."""
    gdf.plot(ax=ax, facecolor=NARANJA[2], alpha=0.55, edgecolor="none", zorder=3)
    gdf.boundary.plot(ax=ax, color=NARANJA[4], linewidth=lw, zorder=3)


def vista_detalle(ax, ruta, gdf, pos=(0.035, 0.6, 0.36, 0.36)):
    """Recuadro dentro del mapa con el ortomosaico a 5 cm y las máscaras; se une a su zona con líneas."""
    with rasterio.open(ruta) as z:
        img = z.read().transpose(1, 2, 0)
        zb = z.bounds
    ins = ax.inset_axes(list(pos), zorder=7)
    ins.imshow(img, extent=(zb.left, zb.right, zb.bottom, zb.top), interpolation="lanczos")
    sub = gdf.cx[zb.left:zb.right, zb.bottom:zb.top]
    sub.plot(ax=ins, facecolor=NARANJA[2], alpha=0.35, edgecolor="none")
    sub.boundary.plot(ax=ins, color=NARANJA[2], linewidth=1.6)
    ins.set_xlim(zb.left, zb.right)
    ins.set_ylim(zb.bottom, zb.top)
    ins.set_xticks([]), ins.set_yticks([])
    ins.set_xlabel(""), ins.set_ylabel("")
    for sp in ins.spines.values():
        sp.set_color(TINTA)
        sp.set_linewidth(1.2)
    ins.text(0.03, 0.03, f"Detalle · {zb.right - zb.left:.0f} × {zb.top - zb.bottom:.0f} m · {len(sub)} máscaras",
             transform=ins.transAxes, fontsize=7.5, color=TINTA, va="bottom",
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2.5))
    ax.indicate_inset_zoom(ins, edgecolor=TINTA, linewidth=1.0, alpha=0.9)


def leyenda_mascaras(ax, n):
    from matplotlib.patches import FancyBboxPatch

    x0, y0 = 0.60, 0.045
    ax.add_patch(FancyBboxPatch((x0 - 0.02, y0 - 0.032), 0.38, 0.105, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.008", facecolor="white", alpha=0.9,
                                edgecolor="none", zorder=5))
    ax.text(x0, y0 + 0.04, "Segmentación YOLO26-seg (v2)", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", color=TINTA, zorder=6)
    ax.add_patch(Rectangle((x0, y0 - 0.004), 0.03, 0.022, transform=ax.transAxes, facecolor=NARANJA[2], alpha=0.55,
                           edgecolor=NARANJA[4], linewidth=0.8, zorder=6))
    ax.text(x0 + 0.04, y0 + 0.007, f"máscara de sombrilla · {n} puestos", transform=ax.transAxes, fontsize=7.5,
            va="center", color=TINTA, zorder=6)


def lienzo(a, b, rgba, crs, titulo):
    """Mapa de calles + ortomosaico + grilla, escala y norte. Devuelve (fig, ax)."""
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
    ancho, alto = xmax - xmin, ymax - ymin
    largo = 100 if ancho > 300 else 50
    escala_grafica(ax, largo, xmin + 0.05 * ancho, ymin + 0.035 * alto, alto_m=ancho * 0.010)
    flecha_norte(ax, xmax - 0.07 * ancho, ymax - 0.12 * alto, alto=ancho * 0.055)
    grilla(ax, xmin, ymin, xmax, ymax, a.paso_grilla)
    ax.set_title(titulo, loc="left", fontsize=13, pad=22, fontweight="bold", color=TINTA)
    ax.text(0.0, -0.045, "Ortomosaico: DJI Mavic Air 2, 2 dic. 2021 · Mapa base: Esri, HERE, Garmin, © OpenStreetMap"
            " contributors · Coordenadas MAGNA-SIRGAS Bogotá (EPSG:3116), grilla cada "
            f"{a.paso_grilla:.0f} m", transform=ax.transAxes, fontsize=6.5, color="#52514e", va="top")
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densidad", required=True)
    ap.add_argument("--puntos", help="centroides (GeoJSON de puntos con diametro_m)")
    ap.add_argument("--mascaras", help="polígonos de los puestos (GeoJSON)")
    ap.add_argument("--zoom", default="docs/base/zoom_plaza_5cm.tif", help="recorte del ortomosaico a 5 cm")
    ap.add_argument("--salida-segmentacion", default="docs/img/segmentacion.png")
    ap.add_argument("--fondo", default="docs/base/la_mariposa_35cm.tif")
    ap.add_argument("--salida", default="docs/img/densidad.png")
    ap.add_argument("--salida-puntos", default="docs/img/puestos.png")
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

    # 0. Segmentación de todo el ortomosaico, con vista de detalle
    if a.mascaras:
        mk = gpd.read_file(a.mascaras).to_crs(crs)
        fig, ax = lienzo(a, b, rgba, crs, "Segmentación de sombrillas · La Mariposa, Bogotá")
        mascaras(ax, mk)
        vista_detalle(ax, a.zoom, mk)
        leyenda_mascaras(ax, len(mk))
        ax.set_xlabel(""), ax.set_ylabel("")
        fig.savefig(a.salida_segmentacion, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # 1. Puestos uno por uno (nodos), tamaño según el diámetro medido
    if a.puntos:
        pts = gpd.read_file(a.puntos).to_crs(crs)
        fig, ax = lienzo(a, b, rgba, crs, "Dónde están los puestos · La Mariposa, Bogotá")
        def tam(d):
            return 8 + 18 * np.clip(np.asarray(d, float) - 1.2, 0, None)
        ax.scatter(pts.geometry.x, pts.geometry.y, s=tam(pts["diametro_m"]), color=NARANJA[2], alpha=0.9,
                   edgecolor="white", linewidth=0.5, zorder=3)
        leyenda_puntos(ax, len(pts), (1.5, 2.2, 2.9), tam)
        ax.set_xlabel(""), ax.set_ylabel("")
        fig.savefig(a.salida_puntos, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # 2. Densidad por hexágono
    hx = gpd.read_file(a.densidad).to_crs(crs)
    hx = hx[hx["puestos"] > 0]
    fig, ax = lienzo(a, b, rgba, crs, "Dónde se concentran los puestos · La Mariposa, Bogotá")
    # Opacidad creciente con la densidad: lo poco denso deja ver el terreno, los focos quedan sólidos
    cmap = LinearSegmentedColormap.from_list("naranja", NARANJA)
    norm = plt.Normalize(vmin=1, vmax=float(hx["puestos"].max()))
    t = norm(hx["puestos"].to_numpy())
    colores = cmap(t)
    colores[:, 3] = 0.30 + 0.55 * np.sqrt(t)
    hx.plot(ax=ax, color=colores, edgecolor="none", linewidth=0, zorder=2)
    leyenda(ax, fig, cmap, norm)
    ax.set_xlabel(""), ax.set_ylabel("")
    fig.savefig(a.salida, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
