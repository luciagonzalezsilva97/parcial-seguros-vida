"""
app.py
-------
Front del Agente de IA de Evaluacion de Riesgo para Seguro de Vida.

Correr con: streamlit run app/app.py  (desde la raiz del proyecto)
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Permite importar los modulos de src/ aunque streamlit corra desde otra carpeta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from premium import calcular_prima          # noqa: E402
from pdf_extractor import extraer_datos_clinicos  # noqa: E402

MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_riesgo.joblib")

st.set_page_config(page_title="Agente de Riesgo - Seguro de Vida", page_icon="🩺")

st.title("🩺 Agente de IA — Evaluación de Riesgo para Seguro de Vida")
st.caption(
    "Proyecto académico. El dataset de entrenamiento es sintético; los umbrales "
    "clínicos usados están documentados y citados en el código (ver generate_dataset.py)."
)


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "No se encontró el modelo entrenado en models/modelo_riesgo.joblib. "
        "Corré primero: python src/pipeline.py"
    )
    st.stop()

modelo = cargar_modelo()

# ---------------------------------------------------------------
# Estado inicial de los campos clinicos (se completan por PDF o a mano)
# ---------------------------------------------------------------
defaults = {
    "edad": 35,
    "imc": 24.0,
    "presion_sistolica": 120.0,
    "presion_diastolica": 80.0,
    "colesterol_total": 190.0,
    "glucemia_ayunas": 90.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.subheader("1. Cargar estudios médicos (PDF) — opcional")
pdf_file = st.file_uploader("Subí el PDF de estudios médicos", type=["pdf"])

if pdf_file is not None:
    with st.spinner("Extrayendo datos del PDF..."):
        datos = extraer_datos_clinicos(pdf_file)
    encontrados = {k: v for k, v in datos.items() if v is not None}
    if encontrados:
        st.success(f"Se extrajeron {len(encontrados)} de 6 campos del PDF. Revisalos abajo antes de evaluar.")
        for k, v in encontrados.items():
            st.session_state[k] = v
    else:
        st.warning("No se pudo extraer ningún valor automáticamente. Completá los campos a mano.")

st.subheader("2. Revisar / completar datos")

col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad (años)", 18, 90, value=int(st.session_state["edad"]))
    sexo = st.selectbox("Sexo", ["M", "F"])
    imc = st.number_input("IMC (kg/m²)", 12.0, 60.0, value=float(st.session_state["imc"]))
    tabaquismo = st.selectbox("¿Fuma?", ["No", "Sí"]) == "Sí"
with col2:
    presion_sistolica = st.number_input(
        "Presión sistólica (mmHg)", 70.0, 260.0, value=float(st.session_state["presion_sistolica"])
    )
    presion_diastolica = st.number_input(
        "Presión diastólica (mmHg)", 40.0, 150.0, value=float(st.session_state["presion_diastolica"])
    )
    colesterol_total = st.number_input(
        "Colesterol total (mg/dL)", 100.0, 400.0, value=float(st.session_state["colesterol_total"])
    )
    glucemia_ayunas = st.number_input(
        "Glucemia en ayunas (mg/dL)", 50.0, 400.0, value=float(st.session_state["glucemia_ayunas"])
    )

st.subheader("3. Datos de la póliza")
col3, col4 = st.columns(2)
with col3:
    ocupacion_riesgo = st.selectbox("Categoría ocupacional", ["bajo", "medio", "alto"])
with col4:
    suma_asegurada = st.selectbox(
        "Suma asegurada (ARS)", [3_000_000, 5_000_000, 8_000_000, 12_000_000, 20_000_000]
    )

if st.button("Evaluar riesgo y calcular prima", type="primary"):
    X = pd.DataFrame([{
        "edad": edad,
        "imc": imc,
        "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica,
        "colesterol_total": colesterol_total,
        "glucemia_ayunas": glucemia_ayunas,
        "tabaquismo": int(tabaquismo),
        "sexo": sexo,
        "ocupacion_riesgo": ocupacion_riesgo,
    }])

    proba_no_asegurable = modelo.predict_proba(X)[0, 1]
    risk_score = round(proba_no_asegurable * 100, 1)
    decision = "No asegurable" if proba_no_asegurable >= 0.5 else "Asegurable"

    st.divider()
    st.subheader("Resultado")

    c1, c2 = st.columns(2)
    with c1:
        if decision == "Asegurable":
            st.success(f"✅ Decisión: {decision}")
        else:
            st.error(f"⛔ Decisión: {decision}")
        st.metric("Score de riesgo", f"{risk_score} / 100")

    with c2:
        prima = calcular_prima(suma_asegurada, risk_score, ocupacion_riesgo)
        st.metric("Prima mensual estimada", f"$ {prima['prima_mensual_ars']:,.2f} ARS")
        st.metric("Prima anual estimada", f"$ {prima['prima_anual_ars']:,.2f} ARS")

    with st.expander("Ver detalle del cálculo de la prima y su fuente"):
        st.json(prima)

    with st.expander("Ver datos usados para la predicción"):
        st.dataframe(X)
