"""Escala: el error que tumbó la primera versión, y cómo no repetirlo.

Un detector aprende objetos de un tamaño en píxeles. Si en inferencia el mismo objeto
llega 4 veces más grande, el modelo no lo reconoce y en cambio "encuentra" cosas que, a
esa escala, tienen el tamaño aprendido. Aquí todo se expresa en **metros por píxel en la
entrada del modelo** (GSD efectivo) para que entrenamiento e inferencia coincidan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Sensores DJI frecuentes: modelo de cámara -> (ancho del sensor mm, alto mm)
SENSORES = {
    "FC3170": (6.4, 4.8),     # Mavic Air 2, 1/2"
    "FC6310": (13.2, 8.8),    # Phantom 4 Pro, 1"
    "FC6310S": (13.2, 8.8),   # Phantom 4 Pro V2
    "FC3411": (13.2, 8.8),    # Air 2S, 1"
    "FC7303": (6.3, 4.7),     # Mini 2
    "L2D-20c": (17.3, 13.0),  # Mavic 3, 4/3"
}


@dataclass(frozen=True)
class Foto:
    ruta: str
    camara: str
    ancho_px: int
    alto_px: int
    focal_mm: float
    altura_m: float
    lat: float | None
    lon: float | None
    rumbo: float | None

    @property
    def gsd(self) -> float:
        """Metros por píxel en el suelo para una foto nadir: altura × sensor / (focal × píxeles)."""
        sensor_w = SENSORES[self.camara][0]
        return self.altura_m * sensor_w / (self.focal_mm * self.ancho_px)


def leer_foto_dji(ruta: str | Path) -> Foto:
    """Lee cámara, focal, tamaño (EXIF) y altura relativa, GPS y rumbo (XMP de DJI)."""
    from PIL import ExifTags, Image

    ruta = str(ruta)
    with Image.open(ruta) as im:
        exif = im.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        sub = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}
        w, h = im.size
    with open(ruta, "rb") as f:
        cab = f.read(256_000)
    m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", cab, re.S)
    xmp = m.group().decode(errors="ignore") if m else ""

    def num(clave):
        r = re.search(clave + r'="([^"]+)"', xmp)
        return float(r.group(1)) if r else None

    return Foto(ruta=ruta, camara=str(tags.get("Model", "")).strip(), ancho_px=w, alto_px=h,
                focal_mm=float(sub.get("FocalLength", 0)), altura_m=num("RelativeAltitude"),
                lat=num("GpsLatitude"), lon=num("GpsLongitude"), rumbo=num("FlightYawDegree"))


def gsd_efectivo(gsd_original: float, *redimensiones: float) -> float:
    """GSD que ve el modelo después de cada reducción (factores < 1 achican la imagen).

    Ejemplo del modelo v1: foto 4000 px a 3,56 cm -> Roboflow 'fit' a 1800×1200 (×0,4) ->
    entrenamiento con imgsz 1500 sobre 1800 px de ancho (×0,833) = 10,7 cm por píxel.
    """
    g = gsd_original
    for f in redimensiones:
        g /= f
    return g


def ventana_equivalente(gsd_raster: float, gsd_modelo: float, imgsz: int) -> int:
    """Lado de la ventana del ráster (en sus píxeles) que, reducida a `imgsz`, queda al GSD del modelo."""
    return int(round(imgsz * gsd_modelo / gsd_raster))


def factor_de_escala(gsd_raster: float, gsd_modelo: float, ventana: int, imgsz: int) -> float:
    """Cuántas veces más grandes llegan los objetos al modelo que en entrenamiento (1 = correcto)."""
    gsd_entrada = gsd_raster * ventana / imgsz
    return gsd_modelo / gsd_entrada
