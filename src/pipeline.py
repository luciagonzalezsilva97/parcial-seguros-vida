"""
pipeline.py
------------
Pipeline IDEM Diabetes: ingestion -> limpieza -> feature engineering ->
modelado -> evaluacion -> export del modelo.

Corre con: python3 src/pipeline.py
Genera en /models: modelo_riesgo.joblib, preprocesador.joblib, metricas.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib

# Rutas relativas a la RAIZ del proyecto (parcial_seguros/), sea cual sea
# la computadora donde se corra. Este archivo vive en parcial_seguros/src/,
# asi que "subimos" un nivel para encontrar data/ y models/.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_used.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUM_COLS = [
    "edad", "imc", "presion_sistolica", "presion_diastolica",
    "colesterol_total", "glucemia_ayunas", "tabaquismo",
]
CAT_COLS = ["sexo", "ocupacion_riesgo"]
TARGET = "decision"  # Asegurable / No asegurable


def cargar_y_limpiar(path=DATA_PATH):
    df = pd.read_csv(path)

    # 1) Ingestion: chequeo de columnas esperadas
    esperado = set(NUM_COLS + CAT_COLS + [TARGET, "risk_score", "suma_asegurada"])
    faltantes = esperado - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {faltantes}")

    # 2) Limpieza basica: outliers fisiologicamente imposibles -> NaN
    #    (para que despues los impute el pipeline, en vez de dejarlos colar)
    df.loc[(df["presion_sistolica"] < 70) | (df["presion_sistolica"] > 260), "presion_sistolica"] = np.nan
    df.loc[(df["imc"] < 12) | (df["imc"] > 60), "imc"] = np.nan

    # 3) Feature engineering: variables categoricas de riesgo clinico
    #    (bucketizacion segun los mismos umbrales documentados en generate_dataset.py)
    df["imc_categoria"] = pd.cut(
        df["imc"], bins=[0, 18.5, 25, 30, 100],
        labels=["bajo_peso", "normal", "sobrepeso", "obesidad"],
    )
    df["hipertension"] = (
        (df["presion_sistolica"] >= 140) | (df["presion_diastolica"] >= 90)
    ).astype("Int64")
    df["glucemia_alta"] = (df["glucemia_ayunas"] >= 126).astype("Int64")
    df["colesterol_alto"] = (df["colesterol_total"] >= 240).astype("Int64")

    return df


def construir_preprocesador():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return pre


def entrenar_y_evaluar():
    df = cargar_y_limpiar()

    X = df[NUM_COLS + CAT_COLS]
    y = (df[TARGET] == "No asegurable").astype(int)  # 1 = No asegurable (clase positiva = riesgo)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = construir_preprocesador()

    modelos = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
    }

    resultados = {}
    pipelines = {}

    for nombre, clf in modelos.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        resultados[nombre] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        pipelines[nombre] = pipe
        print(f"\n=== {nombre} ===")
        print(classification_report(y_test, y_pred, target_names=["Asegurable", "No asegurable"]))
        print("Matriz de confusion:\n", confusion_matrix(y_test, y_pred))

    # Elegimos el mejor por ROC-AUC (metrica robusta a desbalance de clases)
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    mejor_pipe = pipelines[mejor_nombre]
    print(f"\n>>> Mejor modelo: {mejor_nombre} (ROC-AUC={resultados[mejor_nombre]['roc_auc']})")

    # --- Fairness check simple: comparar tasa de "No asegurable" predicha por sexo ---
    X_test_sexo = X_test.copy()
    X_test_sexo["pred"] = mejor_pipe.predict(X_test)
    fairness = X_test_sexo.groupby("sexo")["pred"].mean().to_dict()
    print("\nTasa de 'No asegurable' predicha por sexo (fairness check):", fairness)

    # --- Export ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(mejor_pipe, os.path.join(MODELS_DIR, "modelo_riesgo.joblib"))
    with open(os.path.join(MODELS_DIR, "metricas.json"), "w") as f:
        json.dump(
            {"resultados_por_modelo": resultados, "mejor_modelo": mejor_nombre,
             "fairness_tasa_no_asegurable_por_sexo": fairness},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\nModelo exportado en {os.path.join(MODELS_DIR, 'modelo_riesgo.joblib')}")
    return resultados, mejor_nombre


if __name__ == "__main__":
    entrenar_y_evaluar()
