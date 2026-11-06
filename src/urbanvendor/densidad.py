"""De puntos a densidad: cuántos puestos por hexágono (o por tramo de vía).

Contar puestos sueltos sirve poco para gestión del espacio público; lo que interesa es
**dónde se concentran**. La malla hexagonal evita el sesgo de orientación de una cuadrícula
y todos sus vecinos están a la misma distancia.
"""
from __future__ import annotations

import math

import numpy as np


def malla_hexagonal(xmin: float, ymin: float, xmax: float, ymax: float, lado: float):
    """Hexágonos de lado `lado` (m), con punta arriba, que cubren el rectángulo. Lista de shapely Polygon."""
    from shapely.geometry import Polygon

    w = math.sqrt(3) * lado
    h = 2 * lado
    hexes = []
    fila = 0
    y = ymin - h
    while y < ymax + h:
        x = xmin - w + (w / 2 if fila % 2 else 0)
        while x < xmax + w:
            pts = [(x + lado * math.cos(math.radians(a)), y + lado * math.sin(math.radians(a)))
                   for a in range(30, 390, 60)]
            hexes.append(Polygon(pts))
            x += w
        y += 0.75 * h
        fila += 1
    return hexes


def conteo_por_hexagono(puestos, lado: float = 15.0, area_valida=None):
    """GeoDataFrame de hexágonos con `puestos` (conteo) y `por_ha`. Solo hexágonos con área válida."""
    import geopandas as gpd

    cent = puestos.copy()
    cent["geometry"] = puestos.geometry.centroid
    xmin, ymin, xmax, ymax = (area_valida.total_bounds if area_valida is not None else cent.total_bounds)
    hexes = gpd.GeoDataFrame(geometry=malla_hexagonal(xmin, ymin, xmax, ymax, lado), crs=puestos.crs)
    if area_valida is not None:
        hexes = hexes[hexes.intersects(area_valida.union_all())]
    j = gpd.sjoin(cent[["geometry"]], hexes, predicate="within", how="inner")
    hexes["puestos"] = j.groupby("index_right").size().reindex(hexes.index, fill_value=0).astype(int)
    hexes["por_ha"] = (hexes["puestos"] / (hexes.area / 10_000)).round(1)
    return hexes.reset_index(drop=True)


def resumen(puestos, hexes) -> dict:
    occ = hexes[hexes["puestos"] > 0]
    orden = np.sort(hexes["puestos"].to_numpy())[::-1]
    total = int(orden.sum())
    k = int(np.searchsorted(np.cumsum(orden), 0.5 * total) + 1) if total else 0
    return {
        "puestos": int(len(puestos)),
        "diametro_mediano_m": float(np.median(puestos["diametro_m"])) if len(puestos) else float("nan"),
        "hexagonos_con_puestos": int(len(occ)),
        "hexagonos_totales": int(len(hexes)),
        "max_por_hexagono": int(orden[0]) if len(orden) else 0,
        "hexagonos_con_la_mitad": k,
    }
