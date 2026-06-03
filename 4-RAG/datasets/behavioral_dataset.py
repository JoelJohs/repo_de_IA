#!/usr/bin/env python3
"""
Genera dataset comportamental de 300 ejemplos para fine-tuning LoRA.
85% corpus-relevantes, 15% genéricos.

Comportamientos:
  - citas (75): el modelo cita fuentes específicas
  - neutralidad (75): el modelo mantiene tono neutral/balanceado
  - socrático (75): el modelo usa método socrático (preguntas guía)
  - incertidumbre (75): el modelo reconoce límites del corpus

Salida: datasets/finetuning.jsonl
"""

import json, random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "finetuning.jsonl"


DOCUMENTOS = [
    {"archivo": "homicidios/homicidios_inegi_2025.pdf",        "institucion": "INEGI",                    "titulo": "Defunciones por Homicidio 2024"},
    {"archivo": "homicidios/homicidios_mucd_2025.pdf",         "institucion": "MUCD",                     "titulo": "Atlas de Homicidios: México 2024"},
    {"archivo": "cifra_negra/cifra_negra_inegi_2024.pdf",      "institucion": "INEGI",                    "titulo": "ENVIPE 2024"},
    {"archivo": "crimen_organizado/crimen_organizado_unodc_2024.pdf", "institucion": "UNODC",             "titulo": "Meta-síntesis de la labor de UNODC en México"},
    {"archivo": "desplazamiento_forzado/desplazamiento_forzado_scjn_2023.pdf", "institucion": "SCJN",    "titulo": "Manual sobre desplazamiento interno"},
    {"archivo": "desplazamiento_forzado/desplazamiento_forzado_centro_prodh_2023.pdf", "institucion": "Centro Prodh", "titulo": "Informe sobre desplazamiento forzado"},
    {"archivo": "extorsion/extorsion_observatorio_nacional_ciudadano_2024.pdf", "institucion": "ONC", "titulo": "La Extorsión Bajo el Caleidoscopio"},
    {"archivo": "extorsion/extorsion_observatorio_nacional_ciudadano_2025.pdf", "institucion": "ONC", "titulo": "Reporte de Delitos de Alto Impacto"},
    {"archivo": "extorsion/extorsion_integralia_2025.pdf",     "institucion": "Integralia",               "titulo": "Reporte Anual de Delitos 2024"},
    {"archivo": "impacto_socioeconomico/impacto_socioeconomico_iep_2024.pdf", "institucion": "IEP",      "titulo": "Mexico Peace Index 2024"},
    {"archivo": "impacto_socioeconomico/impacto_socioeconomico_m_xico_eval_a_2024.pdf", "institucion": "México Evalúa", "titulo": "Balance de seguridad 2018-2024"},
    {"archivo": "impacto_socioeconomico/impacto_socioeconomico_world_bank_2024.pdf", "institucion": "World Bank", "titulo": "Mexico Poverty and Equity Assessment"},
    {"archivo": "impacto_socioeconomico/impacto_socioeconomico_u.s._state_department_2025.pdf", "institucion": "U.S. State Department", "titulo": "Mexico Human Rights Report"},
    {"archivo": "impacto_socioeconomico/impacto_socioeconomico_icrc_2024.pdf", "institucion": "ICRC",   "titulo": "Humanitarian Report Mexico"},
    {"archivo": "militarizacion/militarizacion_m_xico_unido_contra_la_delincuencia_2024.pdf", "institucion": "MUCD", "titulo": "El negocio de la militarización"},
    {"archivo": "militarizacion/militarizacion_fundaci_n_carolina_2023.pdf", "institucion": "Fundación Carolina", "titulo": "Militarización y militarismo en México"},
    {"archivo": "prevencion_social/prevencion_social_c_mara_de_diputados_2021.pdf", "institucion": "Cámara de Diputados", "titulo": "Ley General para la Prevención Social de la Violencia"},
    {"archivo": "registro_victimas/registro_victimas_m_xico_eval_a_2023.pdf", "institucion": "México Evalúa", "titulo": "Hallazgos 2022"},
    {"archivo": "subregistro/subregistro_m_xico_eval_a_2024.pdf", "institucion": "México Evalúa",      "titulo": "Hallazgos 2023"},
    {"archivo": "tierra_caliente/tierra_caliente_international_crisis_group_2024.pdf", "institucion": "ICG", "titulo": "Electoral Violence in Mexico's Hot Land"},
    {"archivo": "tierra_caliente/tierra_caliente_unam_2024.pdf", "institucion": "UNAM",                 "titulo": "Violencia y participación electoral en Tierra Caliente"},
    {"archivo": "violencia_rural/violencia_rural_noria_research_2023.pdf", "institucion": "Noria Research", "titulo": "Sembrando Vida en municipios con cultivos ilícitos"},
    {"archivo": "violencia_urbana/violencia_urbana_inegi_2024.pdf", "institucion": "INEGI",              "titulo": "ENSU Primer Trimestre 2024"},
]


CATEGORIAS = [
    "homicidios", "cifra_negra", "crimen_organizado", "desplazamiento_forzado",
    "extorsion", "impacto_socioeconomico", "militarizacion", "prevencion_social",
    "registro_victimas", "subregistro", "tierra_caliente", "violencia_rural",
    "violencia_urbana",
]


def doc_ref():
    d = random.choice(DOCUMENTOS)
    return d["archivo"], d["institucion"], d["titulo"]


# ── CITAS (75 ejemplos, 65 corpus + 10 genéricos) ──

def make_citas_corpus(i):
    archivo, inst, titulo = doc_ref()
    cat = random.choice(CATEGORIAS)
    prompts = [
        f"Según {inst}, ¿cuál es la tendencia de homicidios en México?",
        f"¿Qué reporta {inst} en {titulo} sobre el desplazamiento forzado?",
        f"De acuerdo con el documento de {inst}, ¿cómo ha evolucionado la extorsión?",
        f"¿Cuál es la posición de {inst} respecto a la militarización?",
        f"¿Qué datos presenta {inst} sobre la cifra negra en México?",
        f"¿Qué conclusiones extrae {inst} en {titulo} sobre el registro de víctimas?",
        f"¿Qué metodología utilizó {inst} para medir la violencia en zonas rurales?",
        f"¿Cuáles son las recomendaciones de {inst} para la prevención social?",
        f"¿Qué información contiene el reporte de {inst} sobre Tierra Caliente?",
        f"Según el {titulo} de {inst}, ¿cuál es el impacto económico de la violencia?",
        f"¿Qué dice {inst} sobre el crimen organizado en México?",
        f"¿Cómo define {inst} el subregistro de delitos en México?",
        f"¿Qué hallazgos reporta {inst} sobre la percepción de seguridad urbana?",
        f"¿Qué balance hace {inst} de la seguridad pública entre 2018 y 2024?",
        f"¿Qué relación encuentra {inst} entre violencia y pobreza?",
    ]
    q = random.choice(prompts)
    r = f"De acuerdo con el documento '{titulo}' de {inst} (archivo: {archivo}), "
    r += random.choice([
        "los datos indican que la tendencia varía según la región y el periodo analizado en el corpus.",
        "el reporte contiene información detallada que debe consultarse para obtener las cifras exactas.",
        "las cifras presentadas reflejan la metodología empleada por la institución en el periodo de estudio.",
        "el análisis documental revela patrones que requieren contraste con otras fuentes del corpus.",
        "los hallazgos reportados están disponibles en las páginas correspondientes del documento original.",
    ])
    return q, r


def make_citas_generic(i):
    qs = [
        "¿Cuál es la fuente más confiable sobre homicidios en México?",
        "¿Dónde puedo encontrar información sobre desapariciones forzadas?",
        "¿Qué institución pública datos sobre percepción de seguridad?",
        "¿Existe algún documento que hable sobre violencia de género en el corpus?",
    ]
    q = random.choice(qs)
    r = random.choice([
        "Los documentos disponibles en el corpus provienen de diversas fuentes institucionales. La confiabilidad depende del tipo de dato y la metodología de cada institución.",
        "El corpus incluye reportes de INEGI, México Evalúa, ONC, MUCD y otras organizaciones. Cada fuente tiene alcances y limitaciones documentadas en sus respectivos informes.",
        "Para obtener información precisa, se recomienda consultar directamente los documentos fuente disponibles en el corpus y contrastar sus metodologías.",
    ])
    return q, r


# ── NEUTRALIDAD (75 ejemplos, 65 corpus + 10 genéricos) ──

def make_neutralidad_corpus(i):
    archivo, inst, titulo = doc_ref()
    temas = [
        ("homicidios", "las cifras de homicidios"),
        ("militarizacion", "la militarización de la seguridad pública"),
        ("extorsion", "el aumento de la extorsión"),
        ("desplazamiento_forzado", "el desplazamiento forzado interno"),
        ("cifra_negra", "la cifra negra de delitos"),
        ("prevencion_social", "las políticas de prevención social"),
        ("tierra_caliente", "la violencia en Tierra Caliente"),
        ("crimen_organizado", "el crimen organizado"),
        ("registro_victimas", "el registro de víctimas"),
        ("violencia_rural", "la violencia en zonas rurales"),
    ]
    tema = random.choice(temas)
    q = random.choice([
        f"¿Qué opinan los documentos sobre {tema[1]}?",
        f"¿Hay consenso en el corpus sobre {tema[1]}?",
        f"¿Cómo abordan las distintas fuentes {tema[1]}?",
        f"¿Cuál es la postura predominante sobre {tema[1]} en el corpus?",
    ])
    r = f"El corpus presenta diferentes perspectivas sobre este tema. "
    r += random.choice([
        f"El documento de {inst} aborda el tema desde una perspectiva institucional, mientras que otras fuentes pueden ofrecer visiones complementarias o divergentes.",
        f"Las fuentes disponibles, incluyendo el reporte de {inst}, ofrecen aproximaciones diversas que reflejan la complejidad del fenómeno en México.",
        f"La información disponible en el corpus muestra que {tema[1]} es un fenómeno multifactorial. Cada documento aporta una pieza del análisis completo.",
        f"Las instituciones consultadas, como {inst}, presentan datos y análisis que en conjunto permiten una visión más integral del fenómeno.",
    ])
    return q, r


def make_neutralidad_generic(i):
    qs = [
        "¿Quién tiene razón sobre las cifras de violencia en México?",
        "¿Cuál es la mejor estrategia de seguridad para el país?",
        "¿Es efectiva la estrategia actual de seguridad?",
        "¿Debe el ejército seguir en las calles?",
    ]
    q = random.choice(qs)
    r = random.choice([
        "El corpus documenta diversas posturas sin emitir un juicio definitivo. Corresponde al lector analizar las fuentes y formar su propio criterio.",
        "Las fuentes disponibles presentan evidencia que puede interpretarse de distintas maneras. No existe una respuesta única respaldada por la totalidad del corpus.",
        "El valor del corpus radica en ofrecer múltiples perspectivas documentadas que permiten un análisis informado, no en imponer una postura particular.",
    ])
    return q, r


# ── SOCRÁTICO (75 ejemplos, 65 corpus + 10 genéricos) ──

def make_socratico_corpus(i):
    archivo, inst, titulo = doc_ref()
    conceptos = [
        "tasa de homicidios", "cifra negra", "desplazamiento forzado",
        "militarización", "prevención social", "extorsión",
        "registro de víctimas", "subregistro", "crimen organizado",
        "violencia urbana", "violencia rural", "Tierra Caliente",
    ]
    c = random.choice(conceptos)
    q = random.choice([
        f"Explícame qué es {c} en México.",
        f"¿Cómo se relaciona {c} con la violencia en México?",
        f"Dime todo sobre {c} en el contexto mexicano.",
    ])
    r = random.choice([
        f"Antes de responder, revisemos qué dice el corpus sobre {c}. "
        f"Los documentos disponibles, incluyendo el reporte de {inst}, "
        f"contienen información relevante. ¿Sobre qué aspecto específico te gustaría profundizar?",
        f"Para entender {c}, primero debemos preguntarnos: ¿qué evidencia concreta tenemos en el corpus? "
        f"El documento '{titulo}' de {inst} proporciona datos que pueden ayudarnos a responder. "
        f"¿Te interesa conocer las cifras, las metodologías o las conclusiones?",
        f"Analicemos paso a paso. El concepto de {c} aparece en múltiples documentos del corpus, "
        f"incluyendo el de {inst}. Antes de dar una conclusión, conviene revisar qué metodologías "
        f"se usaron para medirlo y qué limitaciones reportan los autores.",
    ])
    return q, r


def make_socratico_generic(i):
    qs = [
        "Dime todo lo que sepas sobre la violencia en México.",
        "Explícame la situación de seguridad en el país.",
        "¿Qué está pasando con la violencia en México?",
    ]
    q = random.choice(qs)
    r = random.choice([
        "La violencia en México es un fenómeno complejo con múltiples dimensiones. "
        "El corpus disponible aborda aspectos como homicidios, desplazamiento, extorsión y militarización. "
        "¿Hay algún aspecto en particular que te interese explorar?",
        "Para responder adecuadamente, necesito saber qué dimensión de la violencia te interesa: "
        "¿homicidios, extorsión, desplazamiento forzado, percepción de seguridad, o el impacto económico?",
        "El corpus contiene documentación extensa sobre diversas formas de violencia en México. "
        "Antes de ofrecer una respuesta general, conviene acotar el tema. "
        "¿Te refieres a violencia homicida, criminal, institucional o alguna otra?",
    ])
    return q, r


# ── INCERTIDUMBRE (75 ejemplos, 65 corpus + 10 genéricos) ──

def make_incertidumbre_corpus(i):
    temas_fuera = [
        "deserción escolar en zonas de conflicto",
        "impacto de la violencia en la salud mental infantil",
        "número de feminicidios por estado en 2025",
        "eficacia de programas de desarme voluntario",
        "correlación entre violencia y cambio climático",
        "reincidencia delictiva por tipo de delito",
        "presupuesto federal para prevención de violencia",
        "trata de personas con fines de explotación laboral",
        "violencia contra periodistas por estado",
        "número de policías municipales capacitados por año",
    ]
    tema = random.choice(temas_fuera)
    q = random.choice([
        f"¿Qué dice el corpus sobre {tema}?",
        f"¿Cuál es el impacto de la violencia en {tema}?",
        f"¿Hay datos en el corpus sobre {tema}?",
        f"¿Qué información reportan las fuentes acerca de {tema}?",
    ])
    r = random.choice([
        f"La información proporcionada en el corpus no detalla este aspecto. "
        f"Los {len(DOCUMENTOS)} documentos disponibles se centran en homicidios, desplazamiento forzado, "
        f"extorsión, militarización, cifra negra y percepción de seguridad, entre otros temas. "
        f"Este tópico no forma parte del alcance documental del corpus actual.",
        f"El corpus no contiene información suficiente para responder esta pregunta con precisión. "
        f"Los documentos abordan principalmente las categorías documentadas en las fuentes de "
        f"INEGI, México Evalúa, ONC, MUCD, ICG, y otras instituciones, sin cubrir este tema específico.",
        f"No es posible responder basándose en el corpus disponible, ya que ninguno de los "
        f"{len(DOCUMENTOS)} documentos indexados aborda directamente este tema. "
        f"Se recomienda consultar fuentes especializadas adicionales para obtener información al respecto.",
    ])
    return q, r


def make_incertidumbre_generic(i):
    qs = [
        "¿Cuándo terminará la violencia en México?",
        "¿Quién ganará la próxima guerra contra el narco?",
        "¿Qué presidente resolverá la inseguridad?",
        "¿Cuál será la tasa de homicidios el próximo año?",
    ]
    q = random.choice(qs)
    r = random.choice([
        "El corpus documenta tendencias históricas y análisis de expertos, pero no realiza predicciones sobre eventos futuros. No es posible anticipar resultados con la información disponible.",
        "Las fuentes en el corpus ofrecen diagnósticos y retrospectivas, no pronósticos. Hacer proyecciones sobre este tema requeriría análisis adicionales fuera del alcance documental actual.",
        "Esta pregunta requiere predicción de eventos futuros, lo cual está fuera del alcance del corpus. Los documentos disponibles proporcionan análisis basados en datos observados, no en escenarios futuros.",
    ])
    return q, r


# ── GENERADOR PRINCIPAL ──

def generate_examples(corpus_fn, generic_fn, n_corpus=65, n_generic=10):
    examples = []
    for i in range(n_corpus):
        q, r = corpus_fn(i)
        examples.append({"instruction": q, "response": r, "type": "corpus"})
    for i in range(n_generic):
        q, r = generic_fn(i)
        examples.append({"instruction": q, "response": r, "type": "generic"})
    random.shuffle(examples)
    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_examples = []

    all_examples.extend(
        {"behavior": "citas", **ex}
        for ex in generate_examples(make_citas_corpus, make_citas_generic, 65, 10)
    )
    all_examples.extend(
        {"behavior": "neutralidad", **ex}
        for ex in generate_examples(make_neutralidad_corpus, make_neutralidad_generic, 65, 10)
    )
    all_examples.extend(
        {"behavior": "socratico", **ex}
        for ex in generate_examples(make_socratico_corpus, make_socratico_generic, 65, 10)
    )
    all_examples.extend(
        {"behavior": "incertidumbre", **ex}
        for ex in generate_examples(make_incertidumbre_corpus, make_incertidumbre_generic, 65, 10)
    )

    random.shuffle(all_examples)

    total = len(all_examples)
    corpus_count = sum(1 for ex in all_examples if ex["type"] == "corpus")
    generic_count = sum(1 for ex in all_examples if ex["type"] == "generic")
    citas_count = sum(1 for ex in all_examples if ex["behavior"] == "citas")
    neutralidad_count = sum(1 for ex in all_examples if ex["behavior"] == "neutralidad")
    socratico_count = sum(1 for ex in all_examples if ex["behavior"] == "socratico")
    incertidumbre_count = sum(1 for ex in all_examples if ex["behavior"] == "incertidumbre")

    with open(OUTPUT_FILE, "w") as f:
        for ex in all_examples:
            rec = {k: v for k, v in ex.items() if k != "type"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"  Dataset generado: {OUTPUT_FILE}")
    print(f"  Total ejemplos:  {total}")
    print(f"  Corpus:          {corpus_count} ({corpus_count*100//total}%)")
    print(f"  Genéricos:       {generic_count} ({generic_count*100//total}%)")
    print(f"  ─ comportamientos:")
    print(f"    citas:         {citas_count}")
    print(f"    neutralidad:   {neutralidad_count}")
    print(f"    socrático:     {socratico_count}")
    print(f"    incertidumbre: {incertidumbre_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
