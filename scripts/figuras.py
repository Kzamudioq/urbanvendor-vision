"""Figuras del README a partir de la ortofoto y los GeoJSON (no necesita los pesos).

python scripts/figuras.py --orto la_mariposa.png --v1 puestos_v1.geojson --v2 puestos_v2.geojson \
       --densidad densidad_v2.geojson --salida docs/img
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window

TINTA = "#1d1d1b"
AZUL = "#2a78d6"
NARANJA = "#eb6834"


def leer(src, c, f, w, h, out=None):
    a = src.read((1, 2, 3), window=Window(c, f, w, h), out_shape=None if out is None else (3, *out),
                 resampling=Resampling.average, boundless=True, fill_value=255)
    return np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def a_px(src, geom):
    inv = ~src.transform
    xs, ys = np.asarray(geom.exterior.coords).T
    return np.c_[inv.c + inv.a * xs + inv.b * ys, inv.f + inv.d * xs + inv.e * ys]


def fuente(n):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def antes_despues(src, v1, v2, c, f, lado, salida):
    paneles = []
    for gdf, color, titulo in ((v1, (255, 214, 0), "v1 · objetos de 0,55 m"),
                               (v2, (255, 59, 48), "v2 · sombrillas de 2,0 m")):
        im = Image.fromarray(leer(src, c, f, lado, lado))
        d = ImageDraw.Draw(im)
        x0, y0 = src.transform * (c, f)
        x1, y1 = src.transform * (c + lado, f + lado)
        for g in gdf.cx[min(x0, x1):max(x0, x1), min(y0, y1):max(y0, y1)].geometry:
            p = a_px(src, g) - [c, f]
            d.line([tuple(q) for q in p] + [tuple(p[0])], fill=color, width=4)
        d.rectangle([0, 0, lado, 64], fill=(20, 20, 20))
        d.text((22, 14), titulo, fill=(255, 255, 255), font=fuente(34))
        paneles.append(im)
    out = Image.new("RGB", (2 * lado + 16, lado), (255, 255, 255))
    out.paste(paneles[0], (0, 0))
    out.paste(paneles[1], (lado + 16, 0))
    out.reduce(2).save(salida, quality=88)


def escala(src, v2, salida, gsd_raster, gsd_modelo, lienzo=400):
    """Mismo lienzo de 400 píxeles de entrada al modelo: cuánto terreno cabe y qué tamaño tiene una sombrilla."""
    g = v2.sort_values("confianza", ascending=False).iloc[5].geometry
    cx, cy = (~src.transform) * (g.centroid.x, g.centroid.y)
    casos = [("entrenamiento", gsd_modelo), ("ortofoto en v1", gsd_raster * 1024 / 1504), ("ortofoto en v2", gsd_modelo)]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 4.2), gridspec_kw={"wspace": 0.06})
    for a, (t, gsd) in zip(ax, casos):
        n = int(round(lienzo * gsd / gsd_raster))  # píxeles nativos que caben en el lienzo
        img = Image.fromarray(leer(src, int(cx - n / 2), int(cy - n / 2), n, n)).resize((lienzo, lienzo), Image.BILINEAR)
        a.imshow(img)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#d6d4cc")
        a.set_title(f"{t}\n{lienzo * gsd:.0f} m de lado · sombrilla ≈ {2.0 / gsd:.0f} px", fontsize=10.5,
                    color=TINTA, loc="left")
    fig.suptitle("Lo que cabe en 400 píxeles de entrada del modelo", x=0.125, ha="left",
                 fontsize=13, fontweight="bold", color=TINTA, y=1.05)
    fig.savefig(salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mapa_densidad(src, hexes, salida, red=14):
    w, h = src.width // red, src.height // red
    base = leer(src, 0, 0, src.width, src.height, out=(h, w)).astype(float)
    gris = base.mean(-1, keepdims=True)
    base = (0.35 * base + 0.65 * gris) * 0.9 + 25
    alfa = src.read(4, out_shape=(h, w), resampling=Resampling.nearest) if src.count >= 4 else np.full((h, w), 255)
    base[alfa == 0] = 255
    b = src.bounds
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.imshow(base.clip(0, 255).astype(np.uint8), extent=(b.left, b.right, b.bottom, b.top))
    cmap = LinearSegmentedColormap.from_list("naranja", ["#fde3d3", "#f6a173", NARANJA, "#b8401a", "#6e2208"])
    hx = hexes[hexes["puestos"] > 0]
    hx.plot(ax=ax, column="puestos", cmap=cmap, alpha=0.85, edgecolor="none", linewidth=0,
            legend=True, legend_kwds={"label": "puestos por hexágono de 585 m²", "shrink": 0.55})
    ax.set_axis_off()
    ax.set_title("Dónde se concentran los puestos · La Mariposa", loc="left", fontsize=13,
                 fontweight="bold", color=TINTA)
    fig.savefig(salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orto", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--densidad", required=True)
    ap.add_argument("--salida", default="docs/img")
    ap.add_argument("--recorte", type=int, nargs=3, default=[5230, 14168, 1200], help="col fila lado")
    ap.add_argument("--gsd-modelo", type=float, default=0.1065)
    a = ap.parse_args()
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.orto) as src:
        v1, v2 = gpd.read_file(a.v1).to_crs(src.crs), gpd.read_file(a.v2).to_crs(src.crs)
        antes_despues(src, v1, v2, *a.recorte, out / "antes_despues.jpg")
        escala(src, v2, out / "escala.png", abs(src.res[0]), a.gsd_modelo)
        mapa_densidad(src, gpd.read_file(a.densidad).to_crs(src.crs), out / "densidad.png")


if __name__ == "__main__":
    main()
