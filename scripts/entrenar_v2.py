"""Entrenamiento v2 en un computador local con GPU (lo mismo que notebooks/entrenar_v2_colab.ipynb).

1. Dataset en teselas nativas de 1024 px, agrupado en zonas del vuelo que no se solapan.
2. Validación cruzada por zona: un modelo por pliegue, evaluado en la zona que no vio.
3. Modelo final con todas las fotos.
4. Ortofoto completa -> GeoJSON de puestos y densidad por hexágono.

Uso (PowerShell, desde la carpeta del repositorio):

    python scripts/entrenar_v2.py --fotos C:/Users/ktzam/Downloads/PONENCIA/etiquetas ^
        --orto C:/Users/ktzam/Downloads/PONENCIA/centro/la_mariposa.tif --salida trabajo_v2

Los pesos quedan en trabajo_v2/ (excluido de git por .gitignore: *.pt).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fotos", required=True, help="carpeta con DJI_xxxx.JPG + DJI_xxxx.json de Labelme")
    ap.add_argument("--orto", required=True, help="ortofoto (GeoTIFF o PNG georreferenciado)")
    ap.add_argument("--salida", default="trabajo_v2")
    ap.add_argument("--modelo", default="yolo26s-seg.pt")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8, help="baja a 4 si la GPU se queda sin memoria")
    ap.add_argument("--solo-cv", action="store_true", help="solo la validación cruzada")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from ultralytics import YOLO

    from urbanvendor.datos import construir_dataset

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch no ve la GPU. Instala la versión con CUDA (ver README, 'Entrenar en tu computador').")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    salida = Path(a.salida).resolve()
    ds = salida / "dataset"
    info = construir_dataset(a.fotos, ds, lado=1024, umbral_zona_m=140)
    (salida / "dataset.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(info['fotos'])} fotos en {info['zonas']} zonas")

    hp = dict(imgsz=1024, epochs=a.epocas, patience=40, batch=a.batch, degrees=180, flipud=0.5, fliplr=0.5,
              scale=0.25, mosaic=1.0, close_mosaic=15, cos_lr=True, seed=0, workers=4, plots=True)

    filas = []
    for cfg in sorted(glob.glob(str(ds / "pliegue_*.yaml"))):
        nombre = Path(cfg).stem
        m = YOLO(a.modelo)
        m.train(data=cfg, project=str(salida / "cv"), name=nombre, exist_ok=True, **hp)
        v = m.val(data=cfg, imgsz=1024, split="val", plots=False)
        filas.append({"pliegue": nombre, "mAP50_mascara": round(v.seg.map50, 3), "mAP50_95_mascara": round(v.seg.map, 3),
                      "precision": round(v.seg.mp, 3), "recall": round(v.seg.mr, 3)})
    cv = pd.DataFrame(filas)
    cv.to_csv(salida / "validacion_cruzada.csv", index=False)
    print(cv.to_string(index=False))
    if a.solo_cv:
        return

    zonas = sorted(glob.glob(str(ds / "zona_*" / "images")))
    todo = ds / "todo.yaml"
    todo.write_text("train:\n" + "".join(f"  - {z}\n" for z in zonas) + f"val: {zonas[0]}\nnc: 1\n"
                    "names: [puesto_sombrilla]\n", encoding="utf-8")
    final = YOLO(a.modelo)
    final.train(data=str(todo), project=str(salida), name="final", exist_ok=True, **{**hp, "patience": 0, "epochs": 120})
    pesos = salida / "final" / "weights" / "last.pt"

    from urbanvendor.cli import main as cli

    cli(["ortofoto", "--raster", a.orto, "--pesos", str(pesos), "--imgsz", "1024", "--gsd-modelo", "0.0356",
         "--salida", str(salida / "puestos_v2.geojson")])
    cli(["densidad", "--puestos", str(salida / "puestos_v2.geojson"), "--lado", "15",
         "--salida", str(salida / "densidad_v2.geojson")])


if __name__ == "__main__":  # necesario en Windows: los procesos que cargan datos importan este archivo
    main()
