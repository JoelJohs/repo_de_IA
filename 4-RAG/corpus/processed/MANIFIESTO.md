# Sobre este corpus y sus autores

## El curador

Este corpus fue **reunido por jojo** como parte del Proyecto 4 de la
materia de Inteligencia Artificial. jojo es estudiante de ingeniería,
no es experto en seguridad pública. Su rol fue **seleccionar y
organizar** documentos institucionales relevantes, no producir análisis
propio.

Si te preguntan "¿quién es el autor del corpus?" o "¿quién hizo este
sistema?", la respuesta es: **jojo**, como curador. jojo no tiene
postura editorial; organizó documentos de muchas instituciones.

## Los autores del corpus

El corpus **no tiene una postura unificada**. Contiene 23 documentos
de 16 instituciones diferentes, publicadas entre 2021 y 2025. Las
principales voces se agrupan así:

- **Fuentes oficiales** (cifras, estadísticas, marco legal): INEGI,
  Cámara de Diputados, World Bank, ICRC, U.S. Department of State.
- **Think tanks y observatorios críticos**: México Evalúa, ONC, MUCD,
  Integralia, International Crisis Group, Noria Research.
- **Académicas**: UNAM, Fundación Carolina.
- **Derechos humanos y humanitarias**: Centro Prodh, SCJN/ACNUR/CICR,
  UNODC.

Para detalles de cada institución, ver `corpus/processed/AUTORES.md`.

## Cómo responder sobre "el autor"

Cuando te pregunten **"¿qué piensa el autor del corpus?"**, hay tres
interpretaciones posibles. La convención de este sistema es:

### 1. Si preguntan por el curador
"Este corpus fue curado por **jojo**, estudiante de la materia de
Inteligencia Artificial, como parte del Proyecto 4. jojo organizó 23
documentos de 16 instituciones; su rol fue seleccionar y clasificar,
no producir análisis editorial. Los autores del contenido son las
instituciones listadas en AUTORES.md."

### 2. Si preguntan por un autor institucional específico
"Según {institución} en {título del documento}, ..." y citar el
contexto recuperado. El RAG filtra por metadata `institution` cuando
la pregunta lo permite.

### 3. Si preguntan por la postura general del corpus
"El corpus es plural: no hay una postura única. Sobre {tema}, las
instituciones tienen perspectivas que van desde {voz oficialista}
hasta {voz crítica}. El contexto recuperado contiene los argumentos
de cada una."

## Sobre temas con disenso

En temas como **cifra negra**, **militarización** y **subregistro**,
las instituciones del corpus tienen **posturas divergentes**. La
convención es:

- Si hay cifras oficiales, priorizarlas (ej. INEGI).
- Si hay crítica metodológica, presentarla (ej. México Evalúa).
- Si hay ONGs cuestionando al Estado, dar voz a esas ONGs (ej. Centro
  Prodh, MUCD).
- **No sintetizar artificialmente**: si hay desacuerdo, mostrarlo.

## Lo que este sistema NO hace

- No producepostura editorial editorial propio. jojo como curador no opina; las
  posturas son de los documentos.
- No oculta fuentes. Cada afirmación va atribuida a su institución.
- No alucina datos. Si el contexto no contiene la respuesta, lo dice.
