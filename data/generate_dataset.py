"""
generate_dataset.py
--------------------
Genera un dataset SINTETICO de individuos con parametros clinicos y
demograficos, para el Agente de IA de evaluacion de riesgo de Seguro de Vida.

Por que sintetico:
No existe un dataset publico que combine (a) datos clinicos reales de una
persona (IMC, presion arterial, colesterol, glucemia) con (b) una decision
real de aseguradora (asegurable / no asegurable) y su prima, porque esa
informacion es propiedad de las aseguradoras y no se publica. El propio
enunciado del parcial habilita esta alternativa: "Si no encuentran datos de
seguros+salud, combinar datos clinicos publicos con datos simulados de
polizas" / "Generar dataset sintetico y documentar la funcion que creo la
prima".

Los RANGOS y UMBRALES clinicos usados aqui SI son reales y estan tomados de
fuentes publicas reconocidas (documentadas abajo, y repetidas en el README).
Lo sintetico es la COMBINACION de variables en una misma persona y la regla
de decision de riesgo, no los rangos clinicos en si.

Fuentes de los umbrales clinicos:
- Presion arterial: American Heart Association / ACC 2017 Guideline
  Normal <120/80 | Elevada 120-129/<80 | Hipertension E1 130-139/80-89 |
  Hipertension E2 >=140/90
  https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
- Colesterol total (mg/dL): NCEP ATP III
  Deseable <200 | Limite alto 200-239 | Alto >=240
  https://www.nhlbi.nih.gov/health/cholesterol
- Glucemia en ayunas (mg/dL): American Diabetes Association (ADA)
  Normal <100 | Prediabetes 100-125 | Diabetes >=126
  https://diabetes.org/diabetes/a1c/diagnosis
- IMC (kg/m2): OMS / CDC
  Bajo peso <18.5 | Normal 18.5-24.9 | Sobrepeso 25-29.9 | Obesidad >=30
  https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html
- Prevalencia de tabaquismo en Argentina (~22% adultos): Encuesta Nacional de
  Factores de Riesgo (ENFR), Ministerio de Salud de Argentina.
  https://www.argentina.gob.ar/salud/deteccion-temprana/enfr

La REGLA de riesgo (risk_score) y el UMBRAL de decision (asegurable / no
asegurable) son una construccion propia del equipo, documentada explicitamente
para el informe de fairness/etica (no son un estandar de mercado).
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N = 6000

def generar_dataset(n=N, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # --- Demograficos ---
    edad = rng.integers(18, 76, size=n)
    sexo = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])

    ocupacion_riesgo = rng.choice(
        ["bajo", "medio", "alto"], size=n, p=[0.55, 0.30, 0.15]
    )  # bajo=oficina, medio=comercial/conduccion, alto=construccion/altura/mineria

    # --- Antropometria -> IMC ---
    altura_cm = np.where(
        sexo == "M",
        rng.normal(175, 7, n),
        rng.normal(162, 7, n),
    ).clip(140, 205)

    imc_objetivo = rng.normal(26.5, 4.5, n).clip(15, 48)
    peso_kg = imc_objetivo * (altura_cm / 100) ** 2
    imc = peso_kg / (altura_cm / 100) ** 2

    # --- Presion arterial (correlacionada con edad e IMC) ---
    sistolica = (
        95
        + 0.45 * edad
        + 0.9 * (imc - 22).clip(0, None)
        + rng.normal(0, 8, n)
    ).clip(85, 210)
    diastolica = (
        60
        + 0.20 * edad
        + 0.5 * (imc - 22).clip(0, None)
        + rng.normal(0, 6, n)
    ).clip(55, 130)

    # --- Colesterol total (correlacionado con edad e IMC) ---
    colesterol = (
        150
        + 0.8 * edad
        + 1.5 * (imc - 22).clip(0, None)
        + rng.normal(0, 20, n)
    ).clip(110, 340)

    # --- Glucemia en ayunas (correlacionada con IMC y edad) ---
    glucemia = (
        80
        + 0.35 * edad
        + 1.1 * (imc - 22).clip(0, None)
        + rng.normal(0, 10, n)
    ).clip(60, 300)

    # --- Tabaquismo (prevalencia ~22% Argentina, ENFR) ---
    tabaquismo = rng.choice([0, 1], size=n, p=[0.78, 0.22])

    # --- Suma asegurada solicitada (ARS) ---
    suma_asegurada = rng.choice(
        [3_000_000, 5_000_000, 8_000_000, 12_000_000, 20_000_000], size=n
    )

    df = pd.DataFrame(
        {
            "edad": edad,
            "sexo": sexo,
            "ocupacion_riesgo": ocupacion_riesgo,
            "altura_cm": altura_cm.round(1),
            "peso_kg": peso_kg.round(1),
            "imc": imc.round(1),
            "presion_sistolica": sistolica.round(0),
            "presion_diastolica": diastolica.round(0),
            "colesterol_total": colesterol.round(0),
            "glucemia_ayunas": glucemia.round(0),
            "tabaquismo": tabaquismo,
            "suma_asegurada": suma_asegurada,
        }
    )

    # ------------------------------------------------------------------
    # REGLA PROPIA DE RIESGO (documentada, no es un estandar de mercado)
    # Cada factor clinico aporta puntos si supera el umbral "de alerta"
    # segun las fuentes citadas arriba. Se normaliza a 0-100.
    # ------------------------------------------------------------------
    pts = np.zeros(n)

    # Edad: aporta de forma continua (a mas edad, mas riesgo actuarial)
    pts += (df["edad"] - 18) / (75 - 18) * 25  # hasta 25 pts

    # IMC (OMS/CDC)
    pts += np.select(
        [df["imc"] < 18.5, df["imc"] < 25, df["imc"] < 30, df["imc"] >= 30],
        [5, 0, 8, 18],
    )

    # Presion arterial (AHA/ACC 2017)
    pts += np.select(
        [
            (df["presion_sistolica"] >= 140) | (df["presion_diastolica"] >= 90),
            (df["presion_sistolica"] >= 130) | (df["presion_diastolica"] >= 80),
            (df["presion_sistolica"] >= 120),
        ],
        [20, 12, 5],
        default=0,
    )

    # Colesterol (NCEP ATP III)
    pts += np.select(
        [df["colesterol_total"] >= 240, df["colesterol_total"] >= 200],
        [15, 8],
        default=0,
    )

    # Glucemia (ADA)
    pts += np.select(
        [df["glucemia_ayunas"] >= 126, df["glucemia_ayunas"] >= 100],
        [15, 7],
        default=0,
    )

    # Tabaquismo
    pts += df["tabaquismo"] * 15

    # Ocupacion de riesgo
    pts += df["ocupacion_riesgo"].map({"bajo": 0, "medio": 6, "alto": 14})

    # Ruido pequeño para que el modelo no aprenda una regla perfecta lineal
    pts += rng.normal(0, 4, n)

    risk_score = pts.clip(0, 100)
    df["risk_score"] = risk_score.round(1)

    # Umbral propio de decision (documentar en el informe): >=65 No asegurable
    df["decision"] = np.where(df["risk_score"] >= 65, "No asegurable", "Asegurable")

    # Simular ~3% de missing en variables clinicas, como pasaria con un PDF real
    for col in ["presion_sistolica", "presion_diastolica", "colesterol_total", "glucemia_ayunas"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generar_dataset()
    out_path = "/home/claude/parcial_seguros/data/dataset_used.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generado: {out_path}")
    print(df.shape)
    print(df["decision"].value_counts(normalize=True))
