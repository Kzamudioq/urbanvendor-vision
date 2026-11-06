"""Inferencia sobre una ortofoto completa: ventanas a la escala del modelo, unión y georreferencia.

Pasos:
1. La ortofoto se recorre en ventanas de lado ``ventana`` (en píxeles del ráster) con solape.
   El lado sale de ``escala.ventana_equivalente`` para que, al reducirse a ``imgsz``, los
   objetos lleguen al modelo del mismo tamaño que en entrenamiento.
2. Se descartan detecciones que tocan un borde interior de la ventana: un objeto cortado
   por el borde aparece completo en la ventana vecina (el solape es mayor que el objeto).
3. Los polígonos pasan a coordenadas del mapa con la transformación afín del ráster.
4. Entre ventanas se eliminan duplicados: si dos polígonos se solapan más de ``iou`` (o uno
   queda casi contenido en otro), se conserva el de mayor confianza.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np


@dataclass
class Deteccion:
    poligono: np.ndarray   # (n, 2) en píxeles de la imagen entregada al detector
    confianza: float


class Detector(Protocol):
    def __call__(self, rgb: np.ndarray) -> list[Deteccion]: ...


def ventanas(ancho: int, alto: int, lado: int, solape: float) -> Iterator[tuple[int, int, int, int]]:
    """(col, fila, ancho, alto) que cubren todo el ráster; la última fila/columna se ajusta al borde."""
    paso = max(1, int(lado * (1 - solape)))

    def inicios(n):
        if n <= lado:
            return [0]
        xs = list(range(0, n - lado, paso))
        return xs + [n - lado]

    for f in inicios(alto):
        for c in inicios(ancho):
            yield c, f, min(lado, ancho - c), min(lado, alto - f)


def toca_borde_interior(pol: np.ndarray, win: tuple[int, int, int, int], ancho: int, alto: int,
                        margen: float = 2.0) -> bool:
    c, f, w, h = win
    x0, y0 = pol.min(0)
    x1, y1 = pol.max(0)
    return ((x0 <= margen and c > 0) or (y0 <= margen and f > 0)
            or (x1 >= w - margen and c + w < ancho) or (y1 >= h - margen and f + h < alto))


def suprimir_duplicados(geoms: list, confs: np.ndarray, iou: float = 0.4, contencion: float = 0.7) -> list[int]:
    """NMS sobre polígonos shapely. Devuelve los índices conservados, de mayor a menor confianza."""
    from shapely import STRtree

    orden = np.argsort(-confs)
    arbol = STRtree(geoms)
    vivo = np.ones(len(geoms), bool)
    quedan = []
    for i in orden:
        if not vivo[i]:
            continue
        quedan.append(int(i))
        gi = geoms[i]
        for j in arbol.query(gi):
            if j == i or not vivo[j]:
                continue
            gj = geoms[j]
            inter = gi.intersection(gj).area
            if inter == 0:
                continue
            union = gi.area + gj.area - inter
            if inter / union > iou or inter / min(gi.area, gj.area) > contencion:
                vivo[j] = False
    return quedan


def detectar_ortofoto(ruta: str, detector: Detector, ventana: int, solape: float = 0.2,
                      progreso: Callable[[int, int], None] | None = None,
                      conf_min: float = 0.0, diametro_min: float = 0.0):
    """Corre el detector sobre toda la ortofoto. Devuelve un GeoDataFrame en el CRS del ráster.

    ``conf_min`` y ``diametro_min`` (m) filtran al final. Con el modelo v1 en las 12 fotos
    etiquetadas, 0,4 y 1,2 m subieron la precisión de 0,74 a 0,88 conservando el 96 % de los
    aciertos: casi todos los falsos positivos son objetos rojos o azules de menos de 1 m.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import Polygon

    with rasterio.open(ruta) as src:
        W, H, T, crs = src.width, src.height, src.transform, src.crs
        bandas = (1, 2, 3)
        mascara = src.count >= 4
        wins = list(ventanas(W, H, ventana, solape))
        geoms, confs = [], []
        for k, win in enumerate(wins):
            c, f, w, h = win
            rgb = src.read(bandas, window=Window(c, f, w, h))
            if mascara:
                alfa = src.read(4, window=Window(c, f, w, h))
                if not alfa.any():
                    if progreso:
                        progreso(k + 1, len(wins))
                    continue
            rgb = _a_uint8(rgb).transpose(1, 2, 0)
            for d in detector(np.ascontiguousarray(rgb)):
                if len(d.poligono) < 3 or toca_borde_interior(d.poligono, win, W, H):
                    continue
                px = d.poligono + [c, f]
                xs = T.c + T.a * px[:, 0] + T.b * px[:, 1]
                ys = T.f + T.d * px[:, 0] + T.e * px[:, 1]
                g = Polygon(np.c_[xs, ys]).buffer(0)
                if g.is_empty or g.area == 0:
                    continue
                geoms.append(g)
                confs.append(d.confianza)
            if progreso:
                progreso(k + 1, len(wins))

    confs = np.asarray(confs, float)
    keep = suprimir_duplicados(geoms, confs) if geoms else []
    gdf = gpd.GeoDataFrame({"confianza": confs[keep].round(3)}, geometry=[geoms[i] for i in keep], crs=crs)
    if len(gdf):
        gdf["area_m2"] = gdf.area.round(2)
        gdf["diametro_m"] = (2 * np.sqrt(gdf["area_m2"] / np.pi)).round(2)
        gdf["x"] = gdf.centroid.x.round(2)
        gdf["y"] = gdf.centroid.y.round(2)
        gdf = gdf[(gdf["confianza"] >= conf_min) & (gdf["diametro_m"] >= diametro_min)].reset_index(drop=True)
    gdf.insert(0, "id", np.arange(1, len(gdf) + 1))
    return gdf


def _a_uint8(a: np.ndarray) -> np.ndarray:
    """Ortofotos exportadas a 16 bits con valores 0-255 (caso de la_mariposa.png) o a 8 bits."""
    if a.dtype == np.uint8:
        return a
    mx = float(a.max()) if a.size else 0
    if mx <= 255:
        return a.astype(np.uint8)
    return (a / (65535 if mx > 4095 else 4095) * 255).clip(0, 255).astype(np.uint8)


def detector_ultralytics(pesos: str, imgsz: int, conf: float = 0.25, device: str | None = None) -> Detector:
    """Adaptador para un modelo YOLO de segmentación (o detección: usa la caja como polígono)."""
    from ultralytics import YOLO

    modelo = YOLO(pesos)

    def detectar(rgb: np.ndarray) -> list[Deteccion]:
        r = modelo.predict(rgb[..., ::-1], imgsz=imgsz, conf=conf, verbose=False, device=device,
                           retina_masks=False)[0]
        out = []
        cajas = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        cf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        polis = r.masks.xy if r.masks is not None else [None] * len(cajas)
        for caja, p, c in zip(cajas, polis, cf):
            if p is None or len(p) < 3:
                x0, y0, x1, y1 = caja
                p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            out.append(Deteccion(np.asarray(p, float), float(c)))
        return out

    return detectar
