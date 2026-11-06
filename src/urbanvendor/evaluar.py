"""Evaluación contra etiquetas de Labelme, foto por foto, y control de fugas entre conjuntos.

Una métrica de validación solo vale si las fotos de validación no muestran el mismo suelo
que las de entrenamiento. En un vuelo fotogramétrico las fotos vecinas se solapan un 70-80 %:
la misma sombrilla aparece en varias. Por eso aquí se mide primero qué tan lejos está cada
foto evaluada de la foto de entrenamiento más cercana.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def leer_labelme(ruta: str | Path, etiqueta: str | None = None) -> list[np.ndarray]:
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    out = []
    for s in d["shapes"]:
        if etiqueta and s.get("label") != etiqueta:
            continue
        p = np.asarray(s["points"], float)
        if s.get("shape_type") == "rectangle" and len(p) == 2:
            (x0, y0), (x1, y1) = p
            p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if s.get("shape_type") == "circle" and len(p) == 2:
            (cx, cy), (px, py) = p
            r = math.hypot(px - cx, py - cy)
            t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            p = np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
        if len(p) >= 3:
            out.append(p)
    return out


@dataclass
class Resultado:
    referencia: int
    detectados: int
    aciertos: int

    @property
    def precision(self) -> float:
        return self.aciertos / self.detectados if self.detectados else float("nan")

    @property
    def recall(self) -> float:
        return self.aciertos / self.referencia if self.referencia else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def emparejar(ref: list[np.ndarray], pred: list[np.ndarray], confs: np.ndarray, iou: float = 0.5) -> Resultado:
    """Emparejamiento voraz por confianza: cada referencia se usa una sola vez (criterio COCO)."""
    from shapely.geometry import Polygon

    R = [Polygon(p).buffer(0) for p in ref]
    P = [Polygon(p).buffer(0) for p in pred]
    usado = np.zeros(len(R), bool)
    aciertos = 0
    for i in np.argsort(-np.asarray(confs)):
        mejor, jm = 0.0, -1
        for j, r in enumerate(R):
            if usado[j] or not P[i].intersects(r):
                continue
            inter = P[i].intersection(r).area
            v = inter / (P[i].area + r.area - inter)
            if v > mejor:
                mejor, jm = v, j
        if mejor >= iou:
            usado[jm] = True
            aciertos += 1
    return Resultado(len(R), len(P), aciertos)


def distancia_m(lat1, lon1, lat2, lon2) -> float:
    dy = (lat1 - lat2) * 110_574
    dx = (lon1 - lon2) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def solape_aproximado(d: float, lado_m: float) -> float:
    """Fracción de huella que comparten dos fotos nadir separadas `d` metros a lo largo de un lado `lado_m`."""
    return max(0.0, 1 - d / lado_m)


def zonas_por_distancia(coords: dict[str, tuple[float, float]], umbral_m: float) -> dict[str, int]:
    """Agrupa fotos cuya distancia encadenada es menor al umbral (componentes conexas).

    Para validación cruzada sin fuga: todas las fotos de una zona van al mismo pliegue.
    """
    nombres = list(coords)
    padre = {n: n for n in nombres}

    def raiz(n):
        while padre[n] != n:
            padre[n] = padre[padre[n]]
            n = padre[n]
        return n

    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            if distancia_m(*coords[a], *coords[b]) < umbral_m:
                padre[raiz(a)] = raiz(b)
    raices = {}
    return {n: raices.setdefault(raiz(n), len(raices)) for n in nombres}
