"""Del Labelme original a un dataset YOLO de segmentación, sin cambiar la escala y sin fugas.

Diferencias con la exportación de Roboflow usada en v1:

* **Escala nativa.** Las fotos se cortan en teselas de ``lado`` píxeles en vez de reducirse
  a 1800×1200. El GSD del modelo queda igual al de la foto (3,6 cm a 100 m con Mavic Air 2)
  y, en la ortofoto, basta con usar ventanas de ``imgsz`` píxeles sin redimensionar.
* **Pliegues por zona.** Las fotos que se solapan en el terreno quedan en el mismo pliegue
  (``evaluar.zonas_por_distancia``). Validar con una foto tomada 20 m al lado de una de
  entrenamiento es validar con las mismas sombrillas.
* **Aumentos en el entrenamiento, no en el dataset.** Rotaciones, volteos y brillo los aplica
  Ultralytics en cada época; no se triplican archivos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .escala import leer_foto_dji
from .evaluar import leer_labelme, zonas_por_distancia


def teselas(ancho: int, alto: int, lado: int, solape: float):
    from .ortofoto import ventanas

    yield from ventanas(ancho, alto, lado, solape)


def recortar_poligonos(polis: list[np.ndarray], win, min_frac: float = 0.35) -> list[np.ndarray]:
    """Recorta cada polígono a la tesela; descarta los que quedan con menos de `min_frac` de su área."""
    from shapely.geometry import Polygon, box

    c, f, w, h = win
    caja = box(c, f, c + w, f + h)
    out = []
    for p in polis:
        g = Polygon(p).buffer(0)
        if not g.intersects(caja):
            continue
        r = g.intersection(caja)
        if r.is_empty or r.area < min_frac * g.area:
            continue
        if r.geom_type != "Polygon":
            r = max(getattr(r, "geoms", [r]), key=lambda x: x.area)
            if r.geom_type != "Polygon":
                continue
        xy = np.asarray(r.exterior.coords)[:-1] - [c, f]
        out.append(xy / [w, h])
    return out


def construir_dataset(carpeta_fotos: str | Path, salida: str | Path, lado: int = 1024, solape: float = 0.25,
                      etiqueta: str = "puesto_sombrilla", umbral_zona_m: float = 140.0,
                      teselas_vacias: float = 0.15, seed: int = 0) -> dict:
    """Escribe images/ y labels/ por zona + un data.yaml por pliegue (deja-una-zona-fuera)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    carpeta_fotos, salida = Path(carpeta_fotos), Path(salida)
    fotos = sorted(p for p in carpeta_fotos.glob("*.json"))
    meta = {}
    for js in fotos:
        jpg = next((js.with_suffix(s) for s in (".JPG", ".jpg", ".jpeg") if js.with_suffix(s).exists()), None)
        if jpg is None:
            continue
        meta[js.stem] = (js, jpg, leer_foto_dji(jpg))
    coords = {k: (m[2].lat, m[2].lon) for k, m in meta.items()}
    zona = zonas_por_distancia(coords, umbral_zona_m)

    conteo = {}
    for nombre, (js, jpg, foto) in meta.items():
        polis = leer_labelme(js, etiqueta)
        im = Image.open(jpg).convert("RGB")
        z = zona[nombre]
        dimg = salida / f"zona_{z}" / "images"
        dlab = salida / f"zona_{z}" / "labels"
        dimg.mkdir(parents=True, exist_ok=True)
        dlab.mkdir(parents=True, exist_ok=True)
        for k, win in enumerate(teselas(im.width, im.height, lado, solape)):
            polys = recortar_poligonos(polis, win)
            if not polys and rng.random() > teselas_vacias:
                continue  # se conservan algunas teselas sin sombrillas como ejemplos negativos
            c, f, w, h = win
            stem = f"{nombre}_t{k:02d}"
            im.crop((c, f, c + w, f + h)).save(dimg / f"{stem}.jpg", quality=95)
            lineas = ["0 " + " ".join(f"{v:.6f}" for v in p.ravel()) for p in polys]
            (dlab / f"{stem}.txt").write_text("\n".join(lineas), encoding="utf-8")
        conteo[nombre] = {"zona": z, "sombrillas": len(polis), "gsd_cm": round(foto.gsd * 100, 2)}

    zonas = sorted(set(zona.values()))
    for z in zonas:
        train = [str((salida / f"zona_{o}" / "images").resolve()) for o in zonas if o != z]
        val = str((salida / f"zona_{z}" / "images").resolve())
        (salida / f"pliegue_{z}.yaml").write_text(
            "train:\n" + "".join(f"  - {t}\n" for t in train) + f"val: {val}\n"
            f"nc: 1\nnames: ['{etiqueta}']\n", encoding="utf-8")
    return {"fotos": conteo, "zonas": len(zonas)}
