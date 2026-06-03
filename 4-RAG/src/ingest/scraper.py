#!/usr/bin/env python3
"""
Scraper para corpus RAG de seguridad pública en México.
Descarga PDFs de fuentes institucionales, valida contenido, clasifica y registra.
"""

import json, os, re, sys, time, hashlib, urllib.parse
from datetime import datetime
from pathlib import Path

import requests
import fitz

BASE_DIR = Path(__file__).parent.parent.parent / "corpus" / "pdf"
METADATA_FILE = BASE_DIR / "metadata" / "documentos.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

CATEGORIES = [
    "homicidios", "crimen_organizado", "tierra_caliente",
    "desplazamiento_forzado", "extorsion", "militarizacion",
    "prevencion_social", "violencia_urbana", "violencia_rural",
    "registro_victimas", "cifra_negra", "subregistro",
    "impacto_socioeconomico",
]

# Confirmed-working PDF URLs organized by category
DIRECT_SOURCES = [
    # === HOMICIDIOS ===
    {
        "url": "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/edr/DH2024_CP_Ene-dic.pdf",
        "titulo": "Defunciones por Homicidio 2024 - Comunicado de prensa",
        "autor": "INEGI", "institucion": "INEGI", "anio": 2025, "categoria": "homicidios",
    },
    {
        "url": "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/edr/DH2024_RR_Ene-dic.pdf",
        "titulo": "Defunciones por Homicidio 2024 - Resultados completos",
        "autor": "INEGI", "institucion": "INEGI", "anio": 2025, "categoria": "homicidios",
    },
    {
        "url": "https://www.mucd.org.mx/wp-content/uploads/2025/12/Atlas2024.pdf",
        "titulo": "Atlas de Homicidios: México 2024",
        "autor": "Mónica Daniela Osorio Reyes", "institucion": "MUCD", "anio": 2025, "categoria": "homicidios",
    },
    # === CRIMEN ORGANIZADO ===
    {
        "url": "https://www.unodc.org/documents/evaluation/Meta-Analysis/UNODC_Work_in_Mexico_Meta_Synthesis.pdf",
        "titulo": "Meta-síntesis de la labor de UNODC en México 2016-2023",
        "autor": "UNODC", "institucion": "UNODC", "anio": 2024, "categoria": "crimen_organizado",
    },
    # === TIERRA CALIENTE ===
    {
        "url": "https://classic.scielo.org.mx/pdf/rmcps/v69n250/0185-1918-rmcps-69-250-281.pdf",
        "titulo": "Violencia y participación electoral en Tierra Caliente",
        "autor": "Revista Mexicana de Ciencias Políticas y Sociales", "institucion": "UNAM", "anio": 2024, "categoria": "tierra_caliente",
    },
    {
        "url": "https://icg-prod.s3.amazonaws.com/089-mexico-hot-land.pdf",
        "titulo": "Electoral Violence and Illicit Influence in Mexico's Hot Land",
        "autor": "International Crisis Group", "institucion": "International Crisis Group", "anio": 2024, "categoria": "tierra_caliente",
    },
    # === DESPLAZAMIENTO FORZADO ===
    {
        "url": "https://www.scjn.gob.mx/sites/default/files/publicaciones_scjn/publicacion/2023-03/Manual%20sobre%20desplazamiento%20interno.pdf",
        "titulo": "Manual sobre desplazamiento interno",
        "autor": "SCJN / ACNUR / CICR", "institucion": "SCJN", "anio": 2023, "categoria": "desplazamiento_forzado",
    },
    {
        "url": "https://centroprodh.org.mx/wp-content/uploads/2023/11/DesplazamientoForzado.pdf",
        "titulo": "Informe temático sobre desplazamiento forzado interno",
        "autor": "Colectivo EPUMX", "institucion": "Centro Prodh", "anio": 2023, "categoria": "desplazamiento_forzado",
    },
    # === EXTORSIÓN ===
    {
        "url": "https://onc.org.mx/public/onc_site/uploads/extorsion_vf.pdf",
        "titulo": "La Extorsión Bajo el Caleidoscopio: muchas modalidades y pocas políticas públicas",
        "autor": "ONC", "institucion": "Observatorio Nacional Ciudadano", "anio": 2024, "categoria": "extorsion",
    },
    {
        "url": "https://onc.org.mx/public/onc_site/uploads/trimestrales/ReporteDAI4t_2024.pdf",
        "titulo": "Reporte de Delitos de Alto Impacto - 4T 2024",
        "autor": "ONC", "institucion": "Observatorio Nacional Ciudadano", "anio": 2025, "categoria": "extorsion",
    },
    {
        "url": "https://integralia.com.mx/web/wp-content/uploads/2025/02/17FEB25_Reporte-anual_Delitos-2024_VF.pdf",
        "titulo": "Reporte Anual de Delitos 2024",
        "autor": "Integralia", "institucion": "Integralia", "anio": 2025, "categoria": "extorsion",
    },
    # === MILITARIZACION ===
    {
        "url": "https://www.mucd.org.mx/wp-content/uploads/2024/02/Negocio2.0.pdf",
        "titulo": "El negocio de la militarización: Opacidad, poder y dinero",
        "autor": "MUCD", "institucion": "México Unido Contra la Delincuencia", "anio": 2024, "categoria": "militarizacion",
    },
    {
        "url": "https://www.fundacioncarolina.es/wp-content/uploads/2023/06/Militarizacion.pdf",
        "titulo": "Militarización y militarismo en México",
        "autor": "Lisa Sánchez y Gerardo Álvarez", "institucion": "Fundación Carolina", "anio": 2023, "categoria": "militarizacion",
    },
    # === PREVENCION SOCIAL ===
    {
        "url": "https://www.diputados.gob.mx/LeyesBiblio/pdf/LGPSVD_040521.pdf",
        "titulo": "Ley General para la Prevención Social de la Violencia y la Delincuencia",
        "autor": "Cámara de Diputados", "institucion": "Cámara de Diputados", "anio": 2021, "categoria": "prevencion_social",
    },
    # === VIOLENCIA URBANA ===
    {
        "url": "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2024/ENSU/ENSU2024_04.pdf",
        "titulo": "ENSU Primer Trimestre 2024",
        "autor": "INEGI", "institucion": "INEGI", "anio": 2024, "categoria": "violencia_urbana",
    },
    # === VIOLENCIA RURAL ===
    {
        "url": "https://noria-research.com/wp-content/uploads/2023/09/NORIA_MXCA_SEMBRANDO_VIDA_INFORME_SEPT_23_ES.pdf",
        "titulo": "Sembrando Vida en municipios con antecedentes de cultivos ilícitos",
        "autor": "Paul Frissard Martínez", "institucion": "Noria Research", "anio": 2023, "categoria": "violencia_rural",
    },
    # === CIFRA NEGRA ===
    {
        "url": "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2024/ENVIPE/ENVIPE_24.pdf",
        "titulo": "ENVIPE 2024: Encuesta Nacional de Victimización",
        "autor": "INEGI", "institucion": "INEGI", "anio": 2024, "categoria": "cifra_negra",
    },
    # === SUBREGISTRO ===
    {
        "url": "https://www.mexicoevalua.org/wp-content/uploads/2024/10/HALLAZGOS2023.pdf",
        "titulo": "Hallazgos 2023: Seguimiento del sistema de justicia penal",
        "autor": "México Evalúa", "institucion": "México Evalúa", "anio": 2024, "categoria": "subregistro",
    },
    # === REGISTRO VICTIMAS ===
    {
        "url": "https://www.mexicoevalua.org/wp-content/uploads/2023/10/HALLAZGOS2022.pdf",
        "titulo": "Hallazgos 2022: Seguimiento del sistema de justicia penal",
        "autor": "México Evalúa", "institucion": "México Evalúa", "anio": 2023, "categoria": "registro_victimas",
    },
    # === IMPACTO SOCIOECONOMICO ===
    {
        "url": "https://www.mexicoevalua.org/wp-content/uploads/2024/12/balance-seguridad-amlo.pdf",
        "titulo": "(In) Seguridad pública en México, 2018-2024: Un balance de la gestión",
        "autor": "México Evalúa", "institucion": "México Evalúa", "anio": 2024, "categoria": "impacto_socioeconomico",
    },
    {
        "url": "https://reliefweb.int/attachments/58a529cb-f646-4c09-8333-90860794d902/MPI-ENG-2024-web-130524.pdf",
        "titulo": "Mexico Peace Index 2024",
        "autor": "Institute for Economics & Peace", "institucion": "IEP", "anio": 2024, "categoria": "impacto_socioeconomico",
    },
    {
        "url": "https://documents1.worldbank.org/curated/en/099021925181518988/pdf/P179488-c49ff360-e38a-4961-85e2-1ef4e6cc5294.pdf",
        "titulo": "Mexico Poverty and Equity Assessment",
        "autor": "World Bank", "institucion": "World Bank", "anio": 2024, "categoria": "impacto_socioeconomico",
    },
    {
        "url": "https://www.icrc.org/sites/default/files/document_new/file_list/balancehumanitario.ingles_mexico.2024.pdf",
        "titulo": "Humanitarian Report 2024 Mexico - ICRC",
        "autor": "ICRC", "institucion": "ICRC", "anio": 2024, "categoria": "impacto_socioeconomico",
    },
    {
        "url": "https://www.state.gov/wp-content/uploads/2025/08/624521_MEXICO-2024-HUMAN-RIGHTS-REPORT.pdf",
        "titulo": "Mexico 2024 Human Rights Report",
        "autor": "U.S. Department of State", "institucion": "U.S. State Department", "anio": 2025, "categoria": "impacto_socioeconomico",
    },
    # === REGISTRO VICTIMAS ===
]

def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return []

def save_metadata(docs):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

def is_valid_pdf(content):
    return content[:4] == b"%PDF"

def count_pages(content):
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        pages = doc.page_count
        doc.close()
        return pages
    except:
        return 0

def make_filename(categoria, institucion, anio, url):
    name = f"{categoria}_{institucion.lower().replace(' ','_').replace(',','')}_{anio}"
    name = re.sub(r'[^a-z0-9_.\-]', '_', name)
    # Get original extension
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if path.endswith(".pdf"):
        name += ".pdf"
    else:
        name += ".pdf"
    return name

def download_source(source):
    url = source["url"]
    cat = source["categoria"]
    
    dest_dir = BASE_DIR / cat
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    fname = make_filename(cat, source["institucion"], source["anio"], url)
    dest = dest_dir / fname
    
    if dest.exists() and dest.stat().st_size > 1000:
        return f"  ✓ Already exists: {fname}"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
        r.raise_for_status()
        
        if not is_valid_pdf(r.content):
            ct = r.headers.get("Content-Type", "?")
            return f"  ✗ Not PDF [{ct}]: {source['titulo'][:50]}"
        
        pages = count_pages(r.content)
        if pages < 3:
            return f"  ✗ Too short ({pages}p): {source['titulo'][:50]}"
        
        with open(dest, "wb") as f:
            f.write(r.content)
        
        entry = {
            "titulo": source["titulo"],
            "autor": source.get("autor", "Desconocido"),
            "institucion": source["institucion"],
            "anio": source["anio"],
            "url": url,
            "categoria": cat,
            "numero_paginas": pages,
            "fecha_descarga": datetime.now().isoformat(),
            "archivo": f"{cat}/{fname}",
        }
        
        return f"  ✓ {cat}/{fname} ({pages}p)", entry
    
    except Exception as e:
        return f"  ✗ Error: {source['titulo'][:40]} - {str(e)[:60]}"

def run():
    print("=" * 60)
    print("  SCRAPER - Corpus Seguridad Pública México")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    docs = load_metadata()
    existing_urls = {d["url"] for d in docs}
    
    new_count = 0
    skip_count = 0
    
    for source in DIRECT_SOURCES:
        if source["url"] in existing_urls:
            print(f"  - Already in metadata: {source['titulo'][:50]}")
            skip_count += 1
            continue
        
        result = download_source(source)
        print(result)
        
        if isinstance(result, tuple):
            _, entry = result
            docs.append(entry)
            new_count += 1
            existing_urls.add(source["url"])
        
        time.sleep(1)
    
    save_metadata(docs)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"  New: {new_count} | Skipped: {skip_count} | Total: {len(docs)}")
    print("=" * 60)
    
    by_cat = {}
    for d in docs:
        by_cat.setdefault(d["categoria"], []).append(d)
    
    for cat in CATEGORIES:
        count = len(by_cat.get(cat, []))
        bar = "█" * count
        print(f"  {cat:<30} {count:>2}  {bar}")
    
    # Show gaps
    gaps = [c for c in CATEGORIES if len(by_cat.get(c, [])) == 0]
    if gaps:
        print(f"\n  ⚠ GAPS (no documents): {', '.join(gaps)}")

if __name__ == "__main__":
    run()
