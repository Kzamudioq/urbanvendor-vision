import json

import numpy as np
import pytest

from urbanvendor import densidad, escala, evaluar, ortofoto
from urbanvendor.datos import recortar_poligonos


# ------------------------------------------------------------------ escala
def test_gsd_mavic_air2_a_100m():
    f = escala.Foto("x.jpg", "FC3170", 4000, 3000, 4.5, 100.1, None, None, None)
    assert f.gsd == pytest.approx(0.0356, abs=2e-4)


def test_v1_llegaba_4_veces_mas_grande():
    g_modelo = escala.gsd_efectivo(0.03559, 0.4, 1504 / 1800)  # Roboflow 0,4 + imgsz 1504 sobre 1800 px
    assert g_modelo == pytest.approx(0.1065, abs=5e-4)
    assert escala.factor_de_escala(0.0345453, g_modelo, 1024, 1504) == pytest.approx(4.53, abs=0.05)
    v = escala.ventana_equivalente(0.0345453, g_modelo, 1504)
    assert escala.factor_de_escala(0.0345453, g_modelo, v, 1504) == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------ ortofoto: ventanas, bordes, duplicados, georreferencia
def _ortofoto_sintetica(tmp_path, n=40, lado=1500, r=30, seed=0):
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    img = np.full((4, lado, lado), 120, np.uint8)
    img[3] = 255
    yy, xx = np.mgrid[:lado, :lado]
    centros = []
    while len(centros) < n:
        c = np.array([rng.integers(100 + r + 2, lado - r - 2), rng.integers(r + 2, lado - r - 2)])
        if all(np.hypot(*(c - q)) > 2.5 * r for q in centros):
            centros.append(c)
    for cx, cy in centros:
        disco = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[0][disco], img[1][disco], img[2][disco] = 230, 30, 60
    img[:3, :, :100] = 255
    img[3, :, :100] = 0  # franja sin datos
    T = from_origin(1_000_000, 1_001_000, 0.05, 0.05)
    p = tmp_path / "orto.tif"
    with rasterio.open(p, "w", driver="GTiff", width=lado, height=lado, count=4, dtype="uint8",
                       crs="EPSG:3116", transform=T) as d:
        d.write(img)
    return p, centros, T


def _detector_de_discos(rgb):
    """Detector falso: componentes rojas. Recorta en el borde como haría un modelo real."""
    from scipy import ndimage

    m = (rgb[..., 0] > 200) & (rgb[..., 1] < 80)
    lab, n = ndimage.label(m)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append(ortofoto.Deteccion(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), 0.9))
    return out


def test_ortofoto_completa_sin_duplicados_ni_cortes(tmp_path):
    p, centros, T = _ortofoto_sintetica(tmp_path)
    gdf = ortofoto.detectar_ortofoto(str(p), _detector_de_discos, ventana=400, solape=0.3)
    assert len(gdf) == len(centros)                       # cada disco una vez, ninguno partido
    assert gdf["diametro_m"].between(2.9, 3.6).all()      # disco de 3 m, caja de 61 px: 3,44 m equivalentes
    cx = np.array([c[0] for c in centros]) + 0.5
    cy = np.array([c[1] for c in centros]) + 0.5
    xs, ys = T.c + T.a * cx, T.f + T.e * cy
    d = [np.min(np.hypot(gdf["x"] - x, gdf["y"] - y)) for x, y in zip(xs, ys)]
    assert max(d) < 0.15                                  # georreferencia correcta (< 3 píxeles)
    assert gdf.crs.to_epsg() == 3116
    assert len(ortofoto.detectar_ortofoto(str(p), _detector_de_discos, 400, 0.3, diametro_min=4.0)) == 0


def test_ventanas_cubren_todo():
    ws = list(ortofoto.ventanas(1000, 700, 400, 0.25))
    cubierto = np.zeros((700, 1000), bool)
    for c, f, w, h in ws:
        cubierto[f:f + h, c:c + w] = True
    assert cubierto.all()


def test_nms_conserva_el_de_mayor_confianza():
    from shapely.geometry import box

    g = [box(0, 0, 10, 10), box(1, 1, 11, 11), box(50, 50, 60, 60), box(2, 2, 8, 8)]
    keep = ortofoto.suprimir_duplicados(g, np.array([0.6, 0.9, 0.5, 0.4]))
    assert keep == [1, 2]


def test_ortofoto_16_bits_se_lee_bien():
    a = np.array([[[0, 128, 255]]], np.uint16)
    assert ortofoto._a_uint8(a).tolist() == [[[0, 128, 255]]]


# ------------------------------------------------------------------ evaluación y fugas
def test_emparejamiento_iou():
    cuad = lambda x: np.array([[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]], float)
    res = evaluar.emparejar([cuad(0), cuad(100)], [cuad(1), cuad(2), cuad(300)], np.array([0.9, 0.8, 0.7]))
    assert (res.referencia, res.detectados, res.aciertos) == (2, 3, 1)  # el duplicado cuenta como falso positivo
    assert res.precision == pytest.approx(1 / 3) and res.recall == pytest.approx(0.5)


def test_zonas_agrupan_fotos_que_se_solapan():
    c = {"A1": (4.6000, -74.0786), "A2": (4.6002, -74.0788), "B1": (4.6030, -74.0786), "B2": (4.6032, -74.0784)}
    z = evaluar.zonas_por_distancia(c, 120)
    assert z["A1"] == z["A2"] and z["B1"] == z["B2"] and z["A1"] != z["B1"]


def test_labelme(tmp_path):
    js = tmp_path / "f.json"
    js.write_text(json.dumps({"shapes": [
        {"label": "puesto_sombrilla", "shape_type": "polygon", "points": [[0, 0], [10, 0], [10, 10]]},
        {"label": "puesto_sombrilla", "shape_type": "circle", "points": [[50, 50], [55, 50]]},
        {"label": "otro", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}]}))
    p = evaluar.leer_labelme(js, "puesto_sombrilla")
    assert len(p) == 2 and len(p[1]) == 24


def test_recorte_a_tesela():
    p = np.array([[90, 10], [130, 10], [130, 50], [90, 50]], float)  # 25 % dentro de la tesela 0-100
    assert recortar_poligonos([p], (0, 0, 100, 100)) == []
    out = recortar_poligonos([p], (80, 0, 100, 100))
    assert len(out) == 1 and out[0].min() >= 0 and out[0].max() <= 1


# ------------------------------------------------------------------ densidad
def test_densidad_hexagonal(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    pts = [Point(5 + i * 0.5, 5) .buffer(1) for i in range(10)] + [Point(200, 200).buffer(1)]
    gdf = gpd.GeoDataFrame({"diametro_m": [2.0] * 11}, geometry=pts, crs="EPSG:3116")
    hx = densidad.conteo_por_hexagono(gdf, lado=15)
    assert hx["puestos"].sum() == 11
    r = densidad.resumen(gdf, hx)
    assert r["max_por_hexagono"] >= 5 and r["hexagonos_con_la_mitad"] <= 2
