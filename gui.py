import flet as ft
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.strtree import STRtree
from pathlib import Path
import tempfile, zipfile, io, os, shutil
from tqdm import tqdm
from datetime import datetime
import folium
import webbrowser

# GDAL/pyogrio för snabbare IO (särskilt på Render)
os.environ["GEOPANDAS_IO_ENGINE"] = "pyogrio"

# ---- Server-/webb-kompatibla mappar ----
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "assets"))      # statiska filer som kan laddas i webben
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))     # dit klienten får ladda upp filer
ASSETS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


# ======================== Geometrihjälp ========================
def perim_area_ratio(geom):
    """Beräknar (perimeter^2) / area. Robust för Polygon/MultiPolygon."""
    if geom is None or geom.is_empty:
        return 0.0

    def _poly_pa(poly):
        perim = poly.exterior.length
        if poly.interiors:
            perim += sum(r.length for r in poly.interiors)
        a = poly.area
        return perim, a

    if geom.geom_type == "Polygon":
        perim, a = _poly_pa(geom)
        return 0.0 if a <= 0 else (perim ** 2) / a

    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        total_perim = 0.0
        total_area = 0.0
        for part in getattr(geom, "geoms", []):
            if part.is_empty or part.geom_type != "Polygon":
                continue
            p, a = _poly_pa(part)
            total_perim += p
            total_area += a
        return 0.0 if total_area <= 0 else (total_perim ** 2) / total_area

    a = geom.area
    return 0.0 if a <= 0 else (geom.length ** 2) / a


def count_vertices(geom):
    """Räknar hörn (vertex) för Polygon/MultiPolygon (exkl. duplicerad sista punkt)."""
    if geom is None or geom.is_empty:
        return 0

    def _poly_vertices(poly):
        pts = len(poly.exterior.coords) - 1
        for hole in poly.interiors:
            pts += len(hole.coords) - 1
        return pts

    if geom.geom_type == "Polygon":
        return _poly_vertices(geom)

    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(_poly_vertices(g) for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon")

    return 0


def union_all_safe(geoms):
    """Robust union av en sekvens av geometrier."""
    seq = list(geoms)
    try:
        from shapely import union_all as shp_union_all  # Shapely 2.x
        return shp_union_all(seq)
    except Exception:
        pass
    try:
        from shapely.ops import unary_union  # Shapely 1.x/2.x ops
        return unary_union(seq)
    except Exception:
        pass

    u = None
    for g in seq:
        if u is None:
            u = g
        else:
            try:
                u = u.union(g)
            except Exception:
                continue
    return u


def adjacent_ratio_with_strtree(gdf):
    """
    Andel av varje geometri's YTTRE BOUNDARY som delas med andra objekt i gdf.
    Returnerar np.array med värden 0..1 avrundat till 2 decimaler.
    """
    geoms = np.array(gdf.geometry.values, dtype=object)
    tree = STRtree(list(geoms))
    id2idx = {id(geom): i for i, geom in enumerate(geoms)}
    ratios = np.zeros(len(geoms), dtype="float64")

    for i, geom in enumerate(tqdm(geoms, desc="ANGRÄNSAR (STRtree.query)")):
        total_len = geom.boundary.length
        if not np.isfinite(total_len) or total_len <= 0:
            ratios[i] = 0.0
            continue

        shared_len = 0.0
        candidates = tree.query(geom)

        if len(candidates) > 0 and isinstance(candidates[0], (int, np.integer)):
            for j in candidates:
                if int(j) == i:
                    continue
                inter = geoms[i].boundary.intersection(geoms[int(j)].boundary)
                if not inter.is_empty:
                    shared_len += inter.length
        else:
            for other in candidates:
                j = id2idx.get(id(other), None)
                if j is None or j == i:
                    continue
                inter = geom.boundary.intersection(other.boundary)
                if not inter.is_empty:
                    shared_len += inter.length

        ratios[i] = round(shared_len / total_len, 2)

    return ratios


# ======================== Plots + QML ========================
def plot_with_optional_basemap(gdf, field, title, out_path, include_basemap=True):
    colors = {1: "red", 2: "orange", 3: "green"}
    legend = [
        mpatches.Patch(color="red", label="Klass 1"),
        mpatches.Patch(color="orange", label="Klass 2"),
        mpatches.Patch(color="green", label="Klass 3"),
    ]
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        if include_basemap:
            import contextily as ctx
            gdf_web = gdf.to_crs(3857)
            gdf_web.plot(color=gdf_web[field].map(colors), edgecolor="black", linewidth=0.2, ax=ax)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        else:
            gdf.plot(color=gdf[field].map(colors), edgecolor="black", linewidth=0.2, ax=ax)
    except Exception:
        gdf.plot(color=gdf[field].map(colors), edgecolor="black", linewidth=0.2, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.legend(handles=legend, title="Klass")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def create_qml_str(field_name):
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="{field_name}">
    <categories>
      <category render="true" symbol="0" value="1" label="Klass 1"/>
      <category render="true" symbol="1" value="2" label="Klass 2"/>
      <category render="true" symbol="2" value="3" label="Klass 3"/>
    </categories>
    <symbols>
      <symbol name="0" type="fill"><layer class="SimpleFill"><prop k="color" v="255,0,0,150"/></layer></symbol>
      <symbol name="1" type="fill"><layer class="SimpleFill"><prop k="color" v="255,165,0,150"/></layer></symbol>
      <symbol name="2" type="fill"><layer class="SimpleFill"><prop k="color" v="0,128,0,150"/></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""


# ======================== Folium-export ========================
def export_folium_maps(gdf, out_html, w_stor=33, w_form=33, w_arr=34):
    """
    Skapar en Folium-HTML med lager:
    - Storlek, Form, Arrondering, Total
    Endast "Total" har popup med förklaring. Legend visar aktuella vikter.
    """
    gdf4326 = gdf.to_crs(4326)
    gj_str = gdf4326.to_json()

    # Center: beräkna i proj (3006), transformera till WGS84
    centroids = gdf.to_crs(3006).centroid.to_crs(4326)
    center = [centroids.y.mean(), centroids.x.mean()]

    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    def style_by_field(field):
        def style_function(feature):
            klass = feature["properties"].get(field, 0)
            if klass == 1:
                fill_color = "red"
            elif klass == 2:
                fill_color = "orange"
            elif klass == 3:
                fill_color = "green"
            else:
                fill_color = "gray"
            return {
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.6,
                "fillColor": fill_color,
            }
        return style_function

    def popup_html(props):
        # Förklarande popup för TOTAL-lagret (statisk, utan API-anrop)
        return f"""
        <b>Block-ID:</b> {props.get('blockid', 'okänd')}<br>
        <b>Total klass:</b> {props.get('TOTAL_SLUT', 'NA')}<br>
        <hr>
        <b>Storlek:</b> AREA = {props.get('AREA', 'NA')} m² → klass {props.get('STOR_SLUT', 'NA')}<br>
        <b>Form:</b> Perim²/Area = {props.get('PERIM_KV', 'NA')}, Brytpunkter = {props.get('BRYTPKT', 'NA')} → klass {props.get('FORM_SLUT', 'NA')}<br>
        <b>Arrondering:</b> BL500 = {props.get('BL500', 'NA')}, Angränsar = {props.get('ANGRANSAR', 'NA')} → klass {props.get('ARR_SLUT', 'NA')}
        """

    layers = [
        ("Storlek", "STOR_SLUT"),
        ("Form", "FORM_SLUT"),
        ("Arrondering", "ARR_SLUT"),
        ("Total", "TOTAL_SLUT"),
    ]

    for label, field in layers:
        geojson = folium.GeoJson(
            gj_str,
            name=label,
            style_function=style_by_field(field),
            show=(field == "TOTAL_SLUT"),
        )

        if field == "TOTAL_SLUT":
            # Skriv popup-content i feature properties och bind som popup
            data_dict = geojson.data if isinstance(geojson.data, dict) else None
            if data_dict:
                for feature in data_dict.get("features", []):
                    props = feature.get("properties", {})
                    feature["properties"]["popupContent"] = popup_html(props)
            folium.GeoJsonPopup(fields=["popupContent"]).add_to(geojson)

        geojson.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend med aktuella vikter
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px;
        background-color: white; 
        border:2px solid grey; 
        border-radius:5px;
        z-index:9999;
        font-size:14px; 
        padding: 8px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        max-width: 260px;
        line-height: 1.2;
    ">
      <div style="font-weight:bold; margin-bottom:4px;">📊 Klass-legend</div>
      <div style="display:flex; align-items:center; gap:6px;"><i style="background:red; width:14px; height:14px; opacity:0.7;"></i><span>Klass 1</span></div>
      <div style="display:flex; align-items:center; gap:6px;"><i style="background:orange; width:14px; height:14px; opacity:0.7;"></i><span>Klass 2</span></div>
      <div style="display:flex; align-items:center; gap:6px;"><i style="background:green; width:14px; height:14px; opacity:0.7;"></i><span>Klass 3</span></div>
      <hr style="margin:6px 0;">
      <div style="font-weight:bold; margin-bottom:2px;">Viktning</div>
      <div>• Storlek: {int(round(w_stor))}%</div>
      <div>• Form: {int(round(w_form))}%</div>
      <div>• Arrondering: {int(round(w_arr))}%</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    return out_html


# ======================== TXT-sammanställning ========================
def write_params_txt(
    revision_dir: Path,
    input_path: Path,
    *,
    area_min, area_max,
    bryt_min, bryt_max,
    perim_min, perim_max,
    block_min, block_max,
    angr_min, angr_max,
    weight_stor, weight_form, weight_arr,
    export_gpkg, export_shp, export_plots, include_basemap
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_w = (weight_stor + weight_form + weight_arr) or 1
    ws = weight_stor / total_w
    wf = weight_form / total_w
    wa = weight_arr / total_w

    txt = f"""Jordbruksanalys – parametrar och vikter
Datum/tid: {ts}
Indata: {input_path}

Trösklar (→ klassning 1/2/3):
  AREA: klass 1 om AREA < {area_min}, klass 3 om AREA ≥ {area_max}
  PERIMETER²/AREA: klass 3 om PERIM_KV < {perim_min}, klass 1 om PERIM_KV > {perim_max}
  BRYTPUNKTER: klass 3 om BRYTPKT < {bryt_min}, klass 1 om BRYTPKT > {bryt_max}
  BLOCK_INOM_500M (BL500): klass 1 om BL500 < {block_min}, klass 3 om BL500 > {block_max}
  ANGRÄNSAR: klass 1 om ANGRANSAR < {angr_min}, klass 3 om ANGRANSAR > {angr_max}

Vikter (för TOTAL):
  Procentvärden: storlek {weight_stor} %, form {weight_form} %, arrondering {weight_arr} %
  Normaliserat (summa = 1.000): storlek {ws:.3f}, form {wf:.3f}, arrondering {wa:.3f}

Sammanvägning:
  STOR_BER = (AREA_KL + AREA_FORD) / 2  → STOR_SLUT = avrundning
  FORM_BER = (PERIM_KL + BRYT_KL) / 2   → FORM_SLUT = avrundning
  ARR_BER  = (BL500_KL + ANGR_KL) / 2   → ARR_SLUT  = avrundning
  TOTAL_BER = (w_s*STOR_SLUT + w_f*FORM_SLUT + w_a*ARR_SLUT) / (w_s+w_f+w_a) → TOTAL_SLUT = avrundning

Exportval:
  GeoPackage: {export_gpkg} | Shapefile: {export_shp} | Plots (PNG): {export_plots} | OSM-bakgrund i plottar: {include_basemap}

Resultatmapp:
  {revision_dir}
"""
    (revision_dir / "analys_parametrar.txt").write_text(txt, encoding="utf-8")


# ======================== Hjälp: ZIP ========================
def zip_revision(revision_dir: Path, out_zip: Path):
    """
    Skapar en ZIP med export/, plots/, folium_all.html och analys_parametrar.txt
    """
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # medtag export och plots
        for sub in ["export", "plots"]:
            base = revision_dir / sub
            if base.exists():
                for root, _, files in os.walk(base):
                    for f in files:
                        full = Path(root) / f
                        rel = full.relative_to(revision_dir)
                        z.write(full, rel.as_posix())
        # enstaka filer
        for fname in ["folium_all.html", "analys_parametrar.txt"]:
            f = revision_dir / fname
            if f.exists():
                z.write(f, f.name)


# ======================== Analys + Export ========================
def analyze_and_export(
    input_path: str,
    area_min=20000, area_max=100000,
    bryt_min=20, bryt_max=50,
    perim_min=20, perim_max=50,
    block_min=150000, block_max=350000,
    angr_min=0.2, angr_max=0.5,
    export_gpkg=True, export_shp=True, export_plots=True,
    include_basemap=True,
    weight_stor=1, weight_form=1, weight_arr=1,   # 0..100
    results_root: Path = Path("results")
):
    src_path = Path(input_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Indata saknas: {src_path}")

    # Om .zip → packa upp och hitta .shp
    temp_unzip_dir = None
    read_path = src_path
    if src_path.suffix.lower() == ".zip":
        temp_unzip_dir = Path(tempfile.mkdtemp(prefix="upload_shp_"))
        with zipfile.ZipFile(src_path, "r") as z:
            z.extractall(temp_unzip_dir)
        shps = list(temp_unzip_dir.glob("**/*.shp"))
        if not shps:
            raise FileNotFoundError("Ingen .shp hittades i zip-filen.")
        read_path = shps[0]

    # skapa unik huvudmapp baserat på indatafilnamn + revision
    base_name = read_path.stem
    results_root.mkdir(exist_ok=True)
    rev = 1
    revision_dir = results_root / f"{base_name}_rev{rev}"
    while revision_dir.exists():
        rev += 1
        revision_dir = results_root / f"{base_name}_rev{rev}"
    revision_dir.mkdir()

    # export/ och plots/
    export_dir = revision_dir / "export"; export_dir.mkdir(exist_ok=True)
    plot_dir   = revision_dir / "plots";  plot_dir.mkdir(exist_ok=True)

    # undermappar för export
    full_dir   = export_dir / "full";    full_dir.mkdir(exist_ok=True)
    stor_dir   = export_dir / "storlek"; stor_dir.mkdir(exist_ok=True)
    form_dir   = export_dir / "form";    form_dir.mkdir(exist_ok=True)
    arr_dir    = export_dir / "arr";     arr_dir.mkdir(exist_ok=True)
    total_dir  = export_dir / "total";   total_dir.mkdir(exist_ok=True)

    # undermappar för plots
    plots_full = plot_dir / "full";      plots_full.mkdir(exist_ok=True)
    plots_stor = plot_dir / "storlek";   plots_stor.mkdir(exist_ok=True)
    plots_form = plot_dir / "form";      plots_form.mkdir(exist_ok=True)
    plots_arr  = plot_dir / "arr";       plots_arr.mkdir(exist_ok=True)
    plots_total= plot_dir / "total";     plots_total.mkdir(exist_ok=True)

    # ---------- Läs data ----------
    gdf = gpd.read_file(str(read_path)).to_crs(3006)
    if not gdf.is_valid.all():
        gdf["geometry"] = gdf.buffer(0)

    # ---------- Beräkningar ----------
    # AREA
    gdf["AREA"] = gdf.geometry.area.round(0).astype(int)
    gdf["AREA_KL"] = np.select([gdf["AREA"] >= area_max, gdf["AREA"] < area_min],[3,1],default=2)
    gdf["AREA_FORD"] = pd.qcut(gdf["AREA"], 3, labels=[1,2,3]).astype(int)

    # PERIMETER²/AREA
    gdf["PERIM_KV"] = [round(perim_area_ratio(geom), 0) for geom in tqdm(gdf.geometry, desc="PERIMETER2/AREA")]
    gdf["PERIM_KV"] = gdf["PERIM_KV"].astype(int)
    gdf["PERIM_KL"] = np.select([gdf["PERIM_KV"] < perim_min, gdf["PERIM_KV"] > perim_max],[3,1],default=2)

    # BRYTPUNKTER
    gdf["BRYTPKT"] = [count_vertices(geom) for geom in tqdm(gdf.geometry, desc="BRYTPUNKTER")]
    gdf["BRYT_KL"] = np.select([gdf["BRYTPKT"] < bryt_min, gdf["BRYTPKT"] > bryt_max],[3,1],default=2)

    # BLOCK_INOM_500M
    all_union = union_all_safe(gdf.geometry)
    block_areas = []
    for geom in tqdm(gdf.geometry, desc="BLOCK_INOM_500M"):
        buf = geom.centroid.buffer(500)
        clipped = buf.intersection(all_union)
        block_areas.append(clipped.area)
    gdf["BL500"] = np.round(block_areas, 0).astype(int)
    gdf["BL500_KL"] = np.select([gdf["BL500"] > block_max, gdf["BL500"] < block_min],[3,1],default=2)

    # ANGRÄNSAR_MOT_BLOCK
    gdf["ANGRANSAR"] = adjacent_ratio_with_strtree(gdf)
    gdf["ANGR_KL"] = np.select([gdf["ANGRANSAR"] > angr_max, gdf["ANGRANSAR"] < angr_min],[3,1],default=2)

    # ---------- Sammanvägda ----------
    gdf["STOR_BER"] = ((gdf["AREA_KL"] + gdf["AREA_FORD"]) / 2).round(2)
    gdf["STOR_SLUT"] = gdf["STOR_BER"].round().astype(int)

    gdf["FORM_BER"] = ((gdf["PERIM_KL"] + gdf["BRYT_KL"]) / 2).round(2)
    gdf["FORM_SLUT"] = gdf["FORM_BER"].round().astype(int)

    gdf["ARR_BER"] = ((gdf["BL500_KL"] + gdf["ANGR_KL"]) / 2).round(2)
    gdf["ARR_SLUT"] = gdf["ARR_BER"].round().astype(int)

    # ---------- TOTAL (viktad) ----------
    total_weights = weight_stor + weight_form + weight_arr
    if total_weights <= 0:
        total_weights = 3
        weight_stor = weight_form = weight_arr = 1

    gdf["TOTAL_BER"] = (
        (weight_stor * gdf["STOR_SLUT"] +
         weight_form * gdf["FORM_SLUT"] +
         weight_arr  * gdf["ARR_SLUT"]) / total_weights
    ).round(2)
    gdf["TOTAL_SLUT"] = gdf["TOTAL_BER"].round().astype(int)

    # ---------- TXT-sammanställning ----------
    write_params_txt(
        revision_dir=revision_dir,
        input_path=src_path,
        area_min=area_min, area_max=area_max,
        bryt_min=bryt_min, bryt_max=bryt_max,
        perim_min=perim_min, perim_max=perim_max,
        block_min=block_min, block_max=block_max,
        angr_min=angr_min, angr_max=angr_max,
        weight_stor=weight_stor, weight_form=weight_form, weight_arr=weight_arr,
        export_gpkg=export_gpkg, export_shp=export_shp,
        export_plots=export_plots, include_basemap=include_basemap
    )

    # ---------- Export ----------
    # Full
    if export_shp:
        gdf.to_file(full_dir / "jordbruksblock_full.shp")
    if export_gpkg:
        gdf.to_file(full_dir / "jordbruksblock_full.gpkg", driver="GPKG")

    # Storlek
    gdf_stor = gdf[["geometry","STOR_SLUT"]]
    if export_shp:
        gdf_stor.to_file(stor_dir / "jordbruksblock_storlek.shp")
    if export_gpkg:
        gdf_stor.to_file(stor_dir / "jordbruksblock_storlek.gpkg", driver="GPKG")
    (stor_dir / "jordbruksblock_storlek.qml").write_text(create_qml_str("STOR_SLUT"), encoding="utf-8")
    if export_plots:
        plot_with_optional_basemap(gdf, "STOR_SLUT", "Sammanvägd storlek", plots_stor / "storlek.png", include_basemap)

    # Form
    gdf_form = gdf[["geometry","FORM_SLUT"]]
    if export_shp:
        gdf_form.to_file(form_dir / "jordbruksblock_form.shp")
    if export_gpkg:
        gdf_form.to_file(form_dir / "jordbruksblock_form.gpkg", driver="GPKG")
    (form_dir / "jordbruksblock_form.qml").write_text(create_qml_str("FORM_SLUT"), encoding="utf-8")
    if export_plots:
        plot_with_optional_basemap(gdf, "FORM_SLUT", "Sammanvägd form", plots_form / "form.png", include_basemap)

    # Arr
    gdf_arr = gdf[["geometry","ARR_SLUT"]]
    if export_shp:
        gdf_arr.to_file(arr_dir / "jordbruksblock_arr.shp")
    if export_gpkg:
        gdf_arr.to_file(arr_dir / "jordbruksblock_arr.gpkg", driver="GPKG")
    (arr_dir / "jordbruksblock_arr.qml").write_text(create_qml_str("ARR_SLUT"), encoding="utf-8")
    if export_plots:
        plot_with_optional_basemap(gdf, "ARR_SLUT", "Sammanvägd arrondering", plots_arr / "arr.png", include_basemap)

    # Total
    gdf_total = gdf[["geometry","TOTAL_SLUT"]]
    if export_shp:
        gdf_total.to_file(total_dir / "jordbruksblock_total.shp")
    if export_gpkg:
        gdf_total.to_file(total_dir / "jordbruksblock_total.gpkg", driver="GPKG")
    (total_dir / "jordbruksblock_total.qml").write_text(create_qml_str("TOTAL_SLUT"), encoding="utf-8")
    if export_plots:
        plot_with_optional_basemap(gdf, "TOTAL_SLUT", "Total brukningsklass", plots_total / "total.png", include_basemap)

    # Full overview plot
    if export_plots:
        fields_full = [
            ("AREA_KL","Area – tröskelklass"),
            ("AREA_FORD","Area – tredjedelsfördelning"),
            ("PERIM_KL","Form – perimeter²/area"),
            ("BRYT_KL","Form – brytpunkter"),
            ("BL500_KL","Arrondering – block inom 500 m"),
            ("ANGR_KL","Arrondering – andel angränsande"),
            ("STOR_SLUT","Sammanvägd storlek"),
            ("FORM_SLUT","Sammanvägd form"),
            ("ARR_SLUT","Sammanvägd arrondering"),
            ("TOTAL_SLUT","Total klass"),
        ]
        colors = {1:"red",2:"orange",3:"green"}
        legend = [
            mpatches.Patch(color="red", label="Klass 1"),
            mpatches.Patch(color="orange", label="Klass 2"),
            mpatches.Patch(color="green", label="Klass 3"),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(25, 12))
        for ax, (field, title) in zip(axes.flatten(), fields_full):
            gdf.plot(color=gdf[field].map(colors), edgecolor="black", linewidth=0.2, ax=ax)
            ax.set_title(title, fontsize=10)
            ax.legend(handles=legend, title="Klass")
        plt.tight_layout()
        plt.savefig(plots_full / "overview.png", dpi=300)
        plt.close()

    # Folium-export
    folium_path = revision_dir / "folium_all.html"
    try:
        export_folium_maps(
            gdf,
            folium_path,
            w_stor=weight_stor, w_form=weight_form, w_arr=weight_arr
        )
    except Exception as ex:
        print(f"⚠️ Kunde inte skapa Folium-karta: {ex}")

    # Zip för nedladdning
    zip_path = revision_dir / f"{revision_dir.name}.zip"
    zip_revision(revision_dir, zip_path)

    # Rensa ev. temporär uppackning
    if temp_unzip_dir and temp_unzip_dir.exists():
        shutil.rmtree(temp_unzip_dir, ignore_errors=True)

    return revision_dir.resolve(), folium_path.resolve(), zip_path.resolve()


# ======================== Flet GUI ========================
def app_main(page: ft.Page):
    page.title = "🌾 Jordbruksanalys – Desktop/Web"
    page.scroll = "adaptive"

    # ---- File pickers ----
    # 1) Lokal fil (endast desktop) – shapefile/gpkg
    picker = ft.FilePicker(
        on_result=lambda e: setattr(
            input_path, "value",
            (e.files[0].path if e.files else input_path.value)
        ) or page.update()
    )

    # 2) Upload (web): gpkg eller zippad shapefile
    pending_upload_name = {"name": None}

    def on_upload_select(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        f = e.files[0]  # vi tar en fil
        dest_name = f.name
        pending_upload_name["name"] = dest_name
        upload_url = page.get_upload_url(dest_name, expires=600)
        uploader.upload([ft.FilePickerUploadFile(dest_name, upload_url=upload_url)])
        status.value = "⬆️ Laddar upp fil…"
        page.update()

    def on_upload_done(e: ft.FilePickerUploadEvent):
        # När klar: sätt input_path till uppladdad fil på servern
        if pending_upload_name["name"]:
            up_path = UPLOAD_DIR / pending_upload_name["name"]
            input_path.value = str(up_path)
            status.value = f"✅ Fil uppladdad: {up_path.name}"
            page.update()

    uploader = ft.FilePicker(on_result=on_upload_select, on_upload=on_upload_done)

    page.overlay.append(picker)
    page.overlay.append(uploader)

    # ---- Input path ----
    input_path = ft.TextField(
        label="Indata (.gpkg, .zip med shapefile, eller lokal .shp/.gpkg på desktop)",
        value="shape_in/Jordbruksblock_inom_3km.shp",
        expand=True
    )
    browse_btn = ft.ElevatedButton(
        "Bläddra lokalt…",
        tooltip="Välj lokal fil (endast desktop)",
        on_click=lambda e: picker.pick_files(
            allow_multiple=False, allowed_extensions=["shp", "gpkg"]
        )
    )
    upload_btn = ft.ElevatedButton(
        "Ladda upp (.gpkg/.zip)…",
        tooltip="Ladda upp fil från din dator (webb/Render)",
        on_click=lambda e: uploader.pick_files(
            allow_multiple=False, allowed_extensions=["gpkg", "zip"]
        )
    )
    demo_btn = ft.TextButton(
        "Använd demo-dataset",
        on_click=lambda e: setattr(input_path, "value", "shape_in/Jordbruksblock_inom_3km.shp") or page.update()
    )

    # ---- Thresholds ----
    area_min = ft.TextField(label="Area < (klass 1)", value="20000", width=180)
    area_max = ft.TextField(label="Area ≥ (klass 3)", value="100000", width=180)
    bryt_min = ft.TextField(label="Brytpunkter < (klass 3)", value="20", width=180)
    bryt_max = ft.TextField(label="Brytpunkter > (klass 1)", value="50", width=180)
    perim_min = ft.TextField(label="Perim²/Area < (klass 3)", value="20", width=180)
    perim_max = ft.TextField(label="Perim²/Area > (klass 1)", value="50", width=180)
    block_min = ft.TextField(label="Block inom 500 m < (klass 1)", value="150000", width=200)
    block_max = ft.TextField(label="Block inom 500 m > (klass 3)", value="350000", width=200)
    angr_min = ft.TextField(label="Angränsar < (klass 1)", value="0.2", width=180)
    angr_max = ft.TextField(label="Angränsar > (klass 3)", value="0.5", width=180)

    # ---- Vikt-sliders ----
    weight_stor = ft.Slider(label="Storlek", min=0, max=100, divisions=100, value=33)
    weight_form = ft.Slider(label="Form", min=0, max=100, divisions=100, value=33)
    weight_arr = ft.Slider(label="Arrondering", min=0, max=100, divisions=100, value=34)
    weight_label = ft.Text("Fördelning: Storlek 33 % | Form 33 % | Arrondering 34 %", weight="bold")

    def normalize_weights(e=None):
        total = weight_stor.value + weight_form.value + weight_arr.value
        if total <= 0:
            return
        weight_stor.value = round(100 * weight_stor.value / total)
        weight_form.value = round(100 * weight_form.value / total)
        weight_arr.value = 100 - weight_stor.value - weight_form.value
        weight_label.value = f"Fördelning: Storlek {int(weight_stor.value)}% | Form {int(weight_form.value)}% | Arrondering {int(weight_arr.value)}%"
        page.update()

    weight_stor.on_change = normalize_weights
    weight_form.on_change = normalize_weights
    weight_arr.on_change = normalize_weights

    # ---- Exportval ----
    export_gpkg = ft.Checkbox(label="GeoPackage + QML", value=True)
    export_shp = ft.Checkbox(label="Shapefile + QML", value=True)
    export_plots = ft.Checkbox(label="Plots (PNG)", value=True)
    include_basemap = ft.Checkbox(label="OSM-bakgrund i plottar (långsammare)", value=True)

    progress = ft.ProgressBar(width=400, value=0)
    status = ft.Text("", color="blue")

    # Resultatknappar (visas efter körning)
    open_map_btn = ft.ElevatedButton(
        "Öppna interaktiv karta (ny flik)",
        visible=False,
        on_click=lambda e: page.launch_url(e.control.data)  # e.control.data = URL
    )
    download_btn = ft.ElevatedButton(
        "Ladda ner resultat (ZIP)",
        visible=False,
        on_click=lambda e: page.launch_url(e.control.data)
    )

    # ---- Callback: Kör analys ----
    def run_clicked(e):
        # Läs parametrar
        try:
            params = dict(
                input_path=input_path.value,
                area_min=int(area_min.value), area_max=int(area_max.value),
                bryt_min=int(bryt_min.value), bryt_max=int(bryt_max.value),
                perim_min=int(perim_min.value), perim_max=int(perim_max.value),
                block_min=int(block_min.value), block_max=int(block_max.value),
                angr_min=float(angr_min.value), angr_max=float(angr_max.value),
                export_gpkg=export_gpkg.value, export_shp=export_shp.value,
                export_plots=export_plots.value, include_basemap=include_basemap.value,
                weight_stor=weight_stor.value, weight_form=weight_form.value, weight_arr=weight_arr.value,
                results_root=Path("results")
            )
        except Exception as ex:
            status.value = f"❌ Ogiltig inmatning: {ex}"
            status.color = "red"
            page.update()
            return

        status.value = "🚜 Kör analys…"
        status.color = "blue"
        progress.value = 0.1
        open_map_btn.visible = False
        download_btn.visible = False
        page.update()

        try:
            revision_dir, folium_path, zip_path = analyze_and_export(**params)

            # Kopiera karta + zip till assets/<revision>/ så att webben kan nå dem
            assets_rev = ASSETS_DIR / revision_dir.name
            assets_rev.mkdir(parents=True, exist_ok=True)
            # Karta
            try:
                shutil.copy2(folium_path, assets_rev / "folium_all.html")
            except Exception:
                pass
            # ZIP
            try:
                shutil.copy2(zip_path, assets_rev / zip_path.name)
            except Exception:
                pass

            # URLs för knapparna
            map_url = page.get_asset_url(f"{revision_dir.name}/folium_all.html")
            zip_url = page.get_asset_url(f"{revision_dir.name}/{zip_path.name}")

            open_map_btn.data = map_url
            download_btn.data = zip_url
            open_map_btn.visible = True
            download_btn.visible = True

            status.value = f"✅ Klar! Resultat i: {revision_dir}\nInteraktiv karta & ZIP finns under knapparna."
            status.color = "green"
            progress.value = 1.0

            # Lokalt: öppna även i systemets webbläsare
            if not os.environ.get("RENDER"):
                try:
                    webbrowser.open(folium_path.as_uri())
                except Exception:
                    pass

        except Exception as ex:
            status.value = f"❌ Fel: {ex}"
            status.color = "red"
            progress.value = 0.0

        page.update()

    # ---- Layout ----
    page.add(
        ft.Column([
            ft.Text("Indata", size=16, weight="bold"),
            ft.Row([input_path]),
            ft.Row([browse_btn, upload_btn, demo_btn], spacing=10),
            ft.Divider(),
            ft.Row([area_min, area_max, bryt_min, bryt_max]),
            ft.Row([perim_min, perim_max, block_min, block_max]),
            ft.Row([angr_min, angr_max]),
            ft.Text("Vikter för totalberäkning", size=16, weight="bold"),
            ft.Text("⚖️ Alla lika → standard (33/33/34)"),
            ft.Text("🚜 Mer fokus på storlek → större fält prioriteras"),
            weight_stor,
            ft.Text("🗺️ Mer fokus på form → regelbundna fält premieras"),
            weight_form,
            ft.Text("📍 Mer fokus på arrondering → sammanhängande landskap"),
            weight_arr,
            weight_label,
            ft.Row([export_gpkg, export_shp, export_plots, include_basemap]),
            ft.ElevatedButton("🚜 Kör analys och export", on_click=run_clicked),
            progress,
            status,
            ft.Row([open_map_btn, download_btn])
        ], spacing=10, expand=False)
    )


# ---- Kör appen (lokalt och på Render) ----
if __name__ == "__main__":
    ft.app(
        target=app_main,
        assets_dir=str(ASSETS_DIR),
        upload_dir=str(UPLOAD_DIR),
        port=int(os.environ.get("PORT", 8080)),
    )
