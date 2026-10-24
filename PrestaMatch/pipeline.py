import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Constantes fijas requeridas por la cátedra
RANDOM_STATE = 42
BUDGET = 2_000_000.0  # Presupuesto máximo asignado

print("=== 1. CARGANDO DATOS ===")
# Si los archivos están en la misma carpeta que pipeline.py:
train_raw = pd.read_csv("data/solicitudes_train.csv")
test_raw = pd.read_csv("data/solicitudes_test.csv")
comp = pd.read_csv("data/comportamiento_bancario.csv")

print(f"Train: {train_raw.shape} | Test: {test_raw.shape} | Comportamiento: {comp.shape}")

print("\n=== 2. FEATURE ENGINEERING SOBRE HISTORIAL BANCARIO ===")
comp = comp.sort_values(["ID_Usuario", "Mes_Referencia"])
g = comp.groupby("ID_Usuario")

# Estadísticos globales del usuario
agg = g.agg(
    saldo_mean=("Saldo_Promedio_Mensual", "mean"),
    saldo_std=("Saldo_Promedio_Mensual", "std"),
    saldo_min=("Saldo_Promedio_Mensual", "min"),
    saldo_max=("Saldo_Promedio_Mensual", "max"),
    atraso_mean=("Dias_Atraso_Otras_Deudas", "mean"),
    atraso_max=("Dias_Atraso_Otras_Deudas", "max"),
    atraso_std=("Dias_Atraso_Otras_Deudas", "std"),
    n_meses=("Mes_Referencia", "count"),
).reset_index()

# Proporción de meses con atraso financiero
pct_atraso = g["Dias_Atraso_Otras_Deudas"].apply(lambda s: (s > 0).mean()).reset_index(name="pct_meses_con_atraso")

# Tendencia: comparativa de los últimos 3 meses frente a los primeros 3 meses
def calcular_tendencia(df_u):
    df_u = df_u.sort_values("Mes_Referencia")
    primeros = df_u.head(3)
    ultimos = df_u.tail(3)
    return pd.Series({
        "saldo_reciente_mean": ultimos["Saldo_Promedio_Mensual"].mean(),
        "saldo_inicial_mean": primeros["Saldo_Promedio_Mensual"].mean(),
        "atraso_reciente_mean": ultimos["Dias_Atraso_Otras_Deudas"].mean(),
        "atraso_reciente_max": ultimos["Dias_Atraso_Otras_Deudas"].max(),
    })

tend = g.apply(calcular_tendencia).reset_index()

# Unir features bancarias y rellenar desvíos nulos (usuarios con 1 solo mes)
comp_features = agg.merge(pct_atraso, on="ID_Usuario").merge(tend, on="ID_Usuario")
comp_features["saldo_tendencia"] = comp_features["saldo_reciente_mean"] - comp_features["saldo_inicial_mean"]
comp_features["saldo_std"] = comp_features["saldo_std"].fillna(0)
comp_features["atraso_std"] = comp_features["atraso_std"].fillna(0)

print(f"Features bancarias generadas: {comp_features.shape[1] - 1} por usuario.")

print("\n=== 3. PREPARANDO DATASETS Y RATIOS ===")
def preparar_solicitudes(df_sol, df_comp):
    df = df_sol.merge(df_comp, on="ID_Usuario", how="left")
    
    # Ratios económicos de solvencia
    df["ratio_monto_ingreso"] = df["Monto_Solicitado"] / df["Ingreso_Mensual"]
    df["ratio_saldo_monto"] = df["saldo_mean"] / df["Monto_Solicitado"]
    df["mes_solicitud"] = pd.to_datetime(df["Fecha_Solicitud"]).dt.month
    
    # One-hot encoding de categorías
    df = pd.get_dummies(df, columns=["Motivo_Prestamo"], prefix="motivo")
    return df

train = preparar_solicitudes(train_raw, comp_features)
test = preparar_solicitudes(test_raw, comp_features)

# Asegurar alineación exacta de columnas entre train y test
feature_cols = [c for c in train.columns if c not in ["ID_Solicitud", "ID_Usuario", "Fecha_Solicitud", "Estado_Final"]]
for c in feature_cols:
    if c not in test.columns:
        test[c] = 0
test = test[[c for c in train.columns if c != "Estado_Final"]]

X = train[feature_cols]
y = train["Estado_Final"]

print(f"Cantidad de variables predictoras: {len(feature_cols)}")

print("\n=== 4. ESTRATEGIA DE NEGOCIO Y OPTIMIZACIÓN (KNAPSACK) ===")
def optimizar_presupuesto(p_pago, monto, tasa, presupuesto=BUDGET):
    """
    Optimización Knapsack Greedy:
    Ordena por Densidad de Valor (ROI esperado por dólar):
    ROI_esperado = p_pago * (1 + tasa) - 1
    """
    roi_esperado = p_pago * (1 + tasa) - 1
    orden = np.argsort(-roi_esperado)  # Mayor rentabilidad relativa primero

    aprobado = np.zeros(len(monto), dtype=int)
    acumulado = 0.0
    for i in orden:
        if roi_esperado[i] <= 0:
            break  # Descarta préstamos con pérdida esperada
        if acumulado + monto[i] <= presupuesto:
            aprobado[i] = 1
            acumulado += monto[i]
    return aprobado

def calcular_ganancia_neta_real(aprobado, monto, tasa, estado_final):
    """Calcula la ganancia real en dólares según las reglas de la competencia."""
    return np.where(
        aprobado == 1,
        np.where(estado_final == 1, monto * tasa, -monto),
        0.0
    ).sum()

# Split de validación para simular el impacto en dólares reales
X_tr, X_val, y_tr, y_val, tr_raw, val_raw = train_test_split(
    X, y, train_raw, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

val_montos = val_raw.loc[X_val.index, "Monto_Solicitado"].values
val_tasas = val_raw.loc[X_val.index, "Tasa_Interes"].values
val_estado = y_val.values

# --- Modelo 1: Regresión Logística (Baseline) ---
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
logreg.fit(X_tr, y_tr)
p_val_logreg = logreg.predict_proba(X_val)[:, 1]
aprob_logreg = optimizar_presupuesto(p_val_logreg, val_montos, val_tasas, BUDGET)
ganancia_logreg = calcular_ganancia_neta_real(aprob_logreg, val_montos, val_tasas, val_estado)

# --- Modelo 2: XGBoost ---
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss"
)
xgb.fit(X_tr, y_tr)
p_val_xgb = xgb.predict_proba(X_val)[:, 1]
aprob_xgb = optimizar_presupuesto(p_val_xgb, val_montos, val_tasas, BUDGET)
ganancia_xgb = calcular_ganancia_neta_real(aprob_xgb, val_montos, val_tasas, val_estado)

print(f"Ganancia neta en validación (Regresión Logística): ${ganancia_logreg:,.2f} USD")
print(f"Ganancia neta en validación (XGBoost):              ${ganancia_xgb:,.2f} USD")

# Selección del modelo ganador
if ganancia_logreg >= ganancia_xgb:
    mejor_modelo = logreg
    nombre_ganador = "Regresión Logística"
else:
    mejor_modelo = xgb
    nombre_ganador = "XGBoost"

print(f"\nModelo seleccionado para producción: {nombre_ganador}")

print("\n=== 5. REENTRENAMIENTO FINAL Y PREDICCIÓN EN TEST ===")
mejor_modelo.fit(X, y)
X_test = test[feature_cols]
p_test = mejor_modelo.predict_proba(X_test)[:, 1]

test_montos = test["Monto_Solicitado"].values
test_tasas = test["Tasa_Interes"].values

aprobado_test = optimizar_presupuesto(p_test, test_montos, test_tasas, BUDGET)
monto_total_aprobado = test.loc[aprobado_test == 1, "Monto_Solicitado"].sum()

print(f"Solicitudes aprobadas: {aprobado_test.sum()} de {len(aprobado_test)}")
print(f"Presupuesto consumido: ${monto_total_aprobado:,.2f} USD / ${BUDGET:,.2f} USD")

# Control de restricción dura
assert monto_total_aprobado <= BUDGET, "¡ERROR: Se excedió el presupuesto de $2M!"

print("\n=== 6. GENERANDO SUBMISSION.CSV ===")
submission = pd.DataFrame({
    "ID_Solicitud": test["ID_Solicitud"],
    "Decision_Aprobacion": aprobado_test
})
submission.to_csv("submission.csv", index=False)
print("¡Archivo 'submission.csv' generado correctamente y listo para subir!")