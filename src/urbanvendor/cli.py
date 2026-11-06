"""python -m urbanvendor <comando>

  escala     GSD de una foto DJI y la ventana correcta para una ortofoto
  dataset    Labelme -> dataset YOLO-seg en teselas nativas, con pliegues por zona
  ortofoto   detectar puestos en una ortofoto completa -> GeoJSON
  densidad   conteo por hexágono -> GeoJSON + resumen
  evaluar    precisión / recall foto por foto contra Labelme, con control de fugas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="urbanvendor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("escala")
    e.add_argument("--foto", required=True)
    e.add_argument("--gsd-raster", type=float, help="m/píxel de la ortofoto")
    e.add_argument("--reduccion", type=float, nargs="*", default=[], help="factores aplicados antes de entrenar")
    e.add_argument("--imgsz", type=int, default=1024)

    d = sub.add_parser("dataset")
    d.add_argument("--fotos", required=True, help="carpeta con DJI_xxxx.JPG + DJI_xxxx.json de Labelme")
    d.add_argument("--salida", required=True)
    d.add_argument("--lado", type=int, default=1024)
    d.add_argument("--zona-m", type=float, default=140.0, help="fotos más cerca que esto (≈ ancho de la huella) van a la misma zona")

    o = sub.add_parser("ortofoto")
    o.add_argument("--raster", required=True)
    o.add_argument("--pesos", required=True)
    o.add_argument("--imgsz", type=int, default=1024)
    o.add_argument("--gsd-modelo", type=float, required=True, help="m/píxel que vio el modelo al entrenar")
    o.add_argument("--conf", type=float, default=0.25, help="umbral del detector")
    o.add_argument("--conf-final", type=float, default=0.4, help="umbral para el mapa final")
    o.add_argument("--diametro-min", type=float, default=1.2, help="m; descarta objetos menores que una sombrilla")
    o.add_argument("--solape", type=float, default=0.2)
    o.add_argument("--salida", default="puestos.geojson")

    h = sub.add_parser("densidad")
    h.add_argument("--puestos", required=True)
    h.add_argument("--lado", type=float, default=15.0, help="lado del hexágono en metros")
    h.add_argument("--salida", default="densidad.geojson")

    v = sub.add_parser("evaluar")
    v.add_argument("--fotos", required=True)
    v.add_argument("--pesos", required=True)
    v.add_argument("--imgsz", type=int, default=1504)
    v.add_argument("--entrenamiento", nargs="*", default=[], help="nombres de fotos usadas para entrenar")
    v.add_argument("--salida", default="evaluacion.csv")

    a = ap.parse_args(argv)

    if a.cmd == "escala":
        from .escala import gsd_efectivo, leer_foto_dji, ventana_equivalente

        f = leer_foto_dji(a.foto)
        g = gsd_efectivo(f.gsd, *a.reduccion)
        out = {"camara": f.camara, "altura_m": f.altura_m, "gsd_foto_cm": round(f.gsd * 100, 2),
               "gsd_modelo_cm": round(g * 100, 2)}
        if a.gsd_raster:
            out["ventana_px"] = ventana_equivalente(a.gsd_raster, g, a.imgsz)
        print(json.dumps(out, ensure_ascii=False, indent=1))

    elif a.cmd == "dataset":
        from .datos import construir_dataset

        print(json.dumps(construir_dataset(a.fotos, a.salida, a.lado, umbral_zona_m=a.zona_m),
                         ensure_ascii=False, indent=1))

    elif a.cmd == "ortofoto":
        import rasterio

        from .escala import ventana_equivalente
        from .ortofoto import detectar_ortofoto, detector_ultralytics

        with rasterio.open(a.raster) as s:
            gsd = abs(s.res[0])
        ventana = ventana_equivalente(gsd, a.gsd_modelo, a.imgsz)
        print(f"GSD ortofoto {gsd * 100:.2f} cm · modelo {a.gsd_modelo * 100:.2f} cm · ventana {ventana} px",
              file=sys.stderr)
        det = detector_ultralytics(a.pesos, a.imgsz, a.conf)
        gdf = detectar_ortofoto(a.raster, det, ventana, a.solape,
                                progreso=lambda k, n: print(f"\r  ventana {k}/{n}", end="", file=sys.stderr),
                                conf_min=a.conf_final, diametro_min=a.diametro_min)
        gdf.to_file(a.salida, driver="GeoJSON")
        print(f"\n{len(gdf)} puestos -> {a.salida}", file=sys.stderr)

    elif a.cmd == "densidad":
        import geopandas as gpd

        from .densidad import conteo_por_hexagono, resumen

        p = gpd.read_file(a.puestos)
        hx = conteo_por_hexagono(p, a.lado)
        hx[hx["puestos"] > 0].to_file(a.salida, driver="GeoJSON")
        print(json.dumps(resumen(p, hx), ensure_ascii=False, indent=1))

    elif a.cmd == "evaluar":
        import numpy as np
        import pandas as pd
        from ultralytics import YOLO

        from .escala import leer_foto_dji
        from .evaluar import distancia_m, emparejar, leer_labelme

        modelo = YOLO(a.pesos)
        filas, meta = [], {}
        for js in sorted(Path(a.fotos).glob("*.json")):
            jpg = js.with_suffix(".JPG")
            if jpg.exists():
                meta[js.stem] = (js, jpg, leer_foto_dji(jpg))
        for nombre, (js, jpg, f) in meta.items():
            r = modelo.predict(str(jpg), imgsz=a.imgsz, conf=0.25, verbose=False)[0]
            pred = [np.asarray(p) for p in (r.masks.xy if r.masks is not None else [])]
            res = emparejar(leer_labelme(js), pred, r.boxes.conf.cpu().numpy())
            cerca = min((distancia_m(f.lat, f.lon, meta[t][2].lat, meta[t][2].lon)
                         for t in a.entrenamiento if t in meta and t != nombre), default=float("nan"))
            filas.append({"foto": nombre, "rol": "entrenamiento" if nombre in a.entrenamiento else "evaluación",
                          "m_a_foto_de_entrenamiento": round(cerca), "sombrillas": res.referencia,
                          "detectadas": res.detectados, "aciertos": res.aciertos,
                          "precision": round(res.precision, 3), "recall": round(res.recall, 3),
                          "f1": round(res.f1, 3)})
        t = pd.DataFrame(filas)
        t.to_csv(a.salida, index=False)
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
