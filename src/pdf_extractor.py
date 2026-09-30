"""
pdf_extractor.py
------------------
Extrae variables clinicas (edad, IMC, presion arterial, colesterol,
glucemia) desde el texto de un PDF de estudios medicos, usando expresiones
regulares flexibles pensadas para informes en espanol.

Esto es un extractor "best effort": los informes medicos no tienen un
formato estandar, asi que se buscan varias formas comunes de nombrar cada
variable. Lo que no se pueda extraer, se deja en None para que el usuario
lo complete a mano en el front.
"""

import re
import pdfplumber


PATRONES = {
    "edad": [
        r"edad\s*[:\-]?\s*(\d{1,3})\s*a(?:ñ|n)os",
        r"edad\s*[:\-]?\s*(\d{1,3})\b",
    ],
    "imc": [
        r"imc\s*[:\-]?\s*(\d{1,2}[.,]?\d{0,2})",
        r"(?:indice de masa corporal)\s*[:\-]?\s*(\d{1,2}[.,]?\d{0,2})",
    ],
    "presion": [
        r"presi[oó]n\s*arterial\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        r"(?:ta|pa)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
    ],
    "colesterol_total": [
        r"colesterol\s*total\s*[:\-]?\s*(\d{2,3}[.,]?\d{0,2})",
    ],
    "glucemia_ayunas": [
        r"glucemia(?:\s*en\s*ayunas)?\s*[:\-]?\s*(\d{2,3}[.,]?\d{0,2})",
        r"glucosa(?:\s*en\s*ayunas)?\s*[:\-]?\s*(\d{2,3}[.,]?\d{0,2})",
    ],
}


def _to_float(s):
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def extraer_texto(pdf_file) -> str:
    """pdf_file: ruta o file-like object (p.ej. lo que devuelve st.file_uploader)"""
    texto = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texto.append(t)
    return "\n".join(texto).lower()


def extraer_datos_clinicos(pdf_file) -> dict:
    """
    Devuelve un dict con las claves:
    edad, imc, presion_sistolica, presion_diastolica,
    colesterol_total, glucemia_ayunas
    Los valores no encontrados quedan en None.
    """
    texto = extraer_texto(pdf_file)

    resultado = {
        "edad": None,
        "imc": None,
        "presion_sistolica": None,
        "presion_diastolica": None,
        "colesterol_total": None,
        "glucemia_ayunas": None,
    }

    for patron in PATRONES["edad"]:
        m = re.search(patron, texto)
        if m:
            resultado["edad"] = int(m.group(1))
            break

    for patron in PATRONES["imc"]:
        m = re.search(patron, texto)
        if m:
            resultado["imc"] = _to_float(m.group(1))
            break

    for patron in PATRONES["presion"]:
        m = re.search(patron, texto)
        if m:
            resultado["presion_sistolica"] = _to_float(m.group(1))
            resultado["presion_diastolica"] = _to_float(m.group(2))
            break

    for patron in PATRONES["colesterol_total"]:
        m = re.search(patron, texto)
        if m:
            resultado["colesterol_total"] = _to_float(m.group(1))
            break

    for patron in PATRONES["glucemia_ayunas"]:
        m = re.search(patron, texto)
        if m:
            resultado["glucemia_ayunas"] = _to_float(m.group(1))
            break

    return resultado
