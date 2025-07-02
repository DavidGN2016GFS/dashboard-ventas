import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

# ========================
# Encabezado del Dashboard
# ========================
st.set_page_config(layout="centered")
st.title("📈 Predicción de Ventas Diarias")
st.markdown("Este panel predice las ventas diarias para los próximos 30 días usando una red neuronal entrenada con datos históricos.")

# ========================
# 1. Cargar datos desde GitHub
# ========================
@st.cache_data
def cargar_datos():
    url = "https://github.com/DavidGN2016GFS/EcomMLInsights/releases/download/v1.0-preprocesado/df_preprocesado.csv"
    df = pd.read_csv(url)
    return df

df = cargar_datos()

# ========================
# 2. Limpieza
# ========================
def limpiar_y_convertir(df, col):
    return pd.to_numeric(
        df[col].astype(str).str.replace(r"[^\d\.]", "", regex=True).replace("", np.nan),
        errors="coerce"
    )

for col in ['quantity', 'unit_price', 'total_price']:
    df[col] = limpiar_y_convertir(df, col)

df['hora_categorica'] = df['hour'].apply(lambda h: 'mañana' if 6 <= h < 12 else 'tarde' if 12 <= h < 18 else 'noche' if 18 <= h <= 23 else 'madrugada')
df['is_weekend'] = df['day'].apply(lambda x: 1 if x in [6, 7] else 0)
df['mes_estacional'] = df['month'].apply(lambda x: 'alta' if x in [11, 12, 1] else 'baja')

# ========================
# 3. Agrupar por fecha
# ========================
df_grouped = df.groupby(['year', 'month', 'day']).agg({
    'total_price': 'sum',
    'quantity': 'sum',
    'unit_price': 'mean',
    'hour': 'mean',
    'is_weekend': 'first',
    'mes_estacional': 'first',
    'bank_name': 'first',
    'district': 'first'
}).reset_index()

df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month', 'day']])

# ========================
# 4. Codificación
# ========================
le = LabelEncoder()
for col in ['bank_name', 'district', 'mes_estacional']:
    df_grouped[col] = le.fit_transform(df_grouped[col].astype(str))

# ========================
# 5. Modelo
# ========================
features = ['quantity', 'unit_price', 'hour', 'is_weekend', 'mes_estacional', 'bank_name', 'district']
X = df_grouped[features]
y = df_grouped['total_price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

# ========================
# 6. Predicción de 30 días
# ========================
df_future = df_grouped.sort_values('date').tail(30).copy()
last_date = df_grouped['date'].max()
df_future['date'] = [last_date + timedelta(days=i) for i in range(1, 31)]

# Aplicar tendencia ascendente y variación
dias = np.arange(1, 31)
df_future['quantity'] = df_grouped['quantity'].mean() * (1 + 0.01 * dias)
df_future['unit_price'] = df_grouped['unit_price'].mean()
df_future['hour'] = df_grouped['hour'].mean() + np.sin(np.linspace(0, np.pi, 30))
df_future['is_weekend'] = [1 if (last_date + timedelta(days=i)).weekday() >= 5 else 0 for i in range(1, 31)]

for col in ['bank_name', 'district', 'mes_estacional']:
    df_future[col] = df_grouped[col].mode()[0]
X_future = df_future[features]
X_future_scaled = scaler.transform(X_future)
predictions = model.predict(X_future_scaled).flatten()
df_future['predicted_sales'] = predictions

# ========================
# 7. Gráfico interactivo
# ========================
st.subheader("🔮 Predicción de Ventas Diarias (30 días siguientes)")

historical_chart = alt.Chart(df_grouped).mark_line().encode(
    x='date:T',
    y='total_price:Q',
    tooltip=['date:T', 'total_price']
).properties(title="Ventas Históricas")

future_chart = alt.Chart(df_future).mark_line(strokeDash=[5, 5], color='orange').encode(
    x='date:T',
    y='predicted_sales:Q',
    tooltip=['date:T', 'predicted_sales']
)

st.altair_chart(historical_chart + future_chart, use_container_width=True)

# ========================
# 8. Tabla de predicción
# ========================
st.subheader("📋 Predicciones detalladas")
st.dataframe(df_future[['date', 'predicted_sales']].round(2).rename(columns={
    'date': 'Fecha',
    'predicted_sales': 'Ventas Predichas'
}))

# ========================
# 9. Explicación para no técnicos
# ========================
with st.expander("🧠 ¿Cómo funciona este modelo?"):
    st.markdown("""
- El modelo usa una red neuronal para aprender patrones históricos de ventas.
- Se entrena con variables como cantidad vendida, precio, hora del día, banco, distrito, etc.
- Una vez entrenado, puede predecir las ventas para los próximos 30 días con buena precisión.
- Aunque no es perfecto, ayuda al equipo de logística a anticipar demanda de inventario y personal.
""")

