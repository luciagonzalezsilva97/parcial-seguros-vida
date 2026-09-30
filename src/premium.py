"""
premium.py
-----------
Calculo de la prima estimada de Seguro de Vida, usando como base la logica
de "tasa por mil sobre la suma asegurada" que utiliza la normativa argentina
de seguros de vida colectivos.

FUENTE (citar en el informe):
- Superintendencia de Seguros de la Nacion (SSN), Resolucion 28.979/2002,
  Anexo I, Seguro Colectivo de Vida Obligatorio: fija la prima como una
  tasa fija mensual por cada $1.000 de suma asegurada
  ($0,24 por mil al momento de la norma).
  https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-28979-2002-78852
- SSN, Circular / Resolucion 611, Anexo II: menciona el ajuste de la tasa
  segun una "tabla de categorias ocupacionales" (a mayor riesgo de la
  actividad del asegurado, mayor la tasa aplicada).

ACLARACION PARA EL INFORME:
La tasa base ($0,24 por mil) corresponde a una norma puntual de un seguro
colectivo obligatorio de una epoca especifica, y NO es la tasa comercial
vigente de mercado (las aseguradoras privadas calculan sus propias tasas
con tablas de mortalidad actuariales, que no son publicas). Para este
trabajo (nivel "aprobar"), se usa esa tasa como TASA BASE ILUSTRATIVA y se
la ESCALA con dos factores documentados y propios del equipo:
  1) factor de riesgo (a partir del risk_score 0-100 que devuelve el modelo)
  2) factor ocupacional (siguiendo el criterio de "tabla de categorias
     ocupacionales" que menciona la SSN, con 3 niveles propios: bajo/medio/alto)
Esto se deja explicito para que quede claro que la formula esta basada en
una fuente real pero los factores de ajuste son una construccion del equipo
(igual que se aclaro para el risk_score en generate_dataset.py).
"""

TASA_BASE_POR_MIL = 0.24  # $ mensuales por cada $1000 de suma asegurada (SSN Res. 28.979/2002)

FACTOR_OCUPACIONAL = {
    "bajo": 1.0,   # trabajo de oficina / administrativo
    "medio": 1.3,  # comercial, conduccion, trabajo de campo
    "alto": 1.6,   # construccion, altura, mineria, tareas de riesgo
}


def factor_riesgo(risk_score: float) -> float:
    """
    Convierte el risk_score (0-100, salida del modelo) en un factor
    multiplicativo de 1.0 (riesgo minimo) a 2.0 (riesgo maximo).
    Construccion propia, documentada: cada punto de riesgo aumenta
    la tasa proporcionalmente.
    """
    risk_score = max(0.0, min(100.0, risk_score))
    return 1.0 + (risk_score / 100.0)


def calcular_prima(
    suma_asegurada: float,
    risk_score: float,
    ocupacion_riesgo: str = "bajo",
) -> dict:
    """
    Devuelve el detalle del calculo de la prima mensual y anual estimada.

    Parametros
    ----------
    suma_asegurada : monto asegurado en ARS
    risk_score : 0-100, salida del modelo de riesgo
    ocupacion_riesgo : "bajo" | "medio" | "alto"
    """
    if ocupacion_riesgo not in FACTOR_OCUPACIONAL:
        ocupacion_riesgo = "bajo"

    f_riesgo = factor_riesgo(risk_score)
    f_ocupacional = FACTOR_OCUPACIONAL[ocupacion_riesgo]

    prima_mensual = (suma_asegurada / 1000) * TASA_BASE_POR_MIL * f_riesgo * f_ocupacional
    prima_anual = prima_mensual * 12

    return {
        "tasa_base_por_mil": TASA_BASE_POR_MIL,
        "factor_riesgo": round(f_riesgo, 3),
        "factor_ocupacional": f_ocupacional,
        "prima_mensual_ars": round(prima_mensual, 2),
        "prima_anual_ars": round(prima_anual, 2),
        "fuente": "SSN Resolucion 28.979/2002 (tasa por mil) + factores propios documentados",
    }


if __name__ == "__main__":
    # Ejemplo rapido
    ejemplo = calcular_prima(suma_asegurada=8_000_000, risk_score=42.0, ocupacion_riesgo="medio")
    for k, v in ejemplo.items():
        print(f"{k}: {v}")
