import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Cargar datos
data = pd.read_csv("depurado.csv")

fig_size = (10, 5)  # Ajusta el tamaño aquí

# Configuración de la app
st.set_page_config(page_title="Análisis de Combustibles en Guatemala", layout="wide")

# Columnas de interés
columns_interes = [
    'Diesel bajo azufre',
    'Diesel ultra bajo azufre',
    'Gas licuado de petróleo',
    'Gasolina regular',
    'Gasolina superior',
    'Diesel alto azufre'
]

# Conversión de la fecha
data['Fecha'] = pd.to_datetime(data['Fecha'])
data['mes'] = data['Fecha'].dt.month
data['año'] = data['Fecha'].dt.year

# Transformación a formato largo
df_combustibles = data[['Fecha', 'año', 'mes'] + columns_interes]
df_long = pd.melt(
    df_combustibles, 
    id_vars=['Fecha', 'año', 'mes'], 
    value_vars=columns_interes, 
    var_name='tipo_combustible', 
    value_name='importaciones'
)

# Suma de importaciones generales
importaciones_generales_mes = df_long.groupby('mes')['importaciones'].sum()
importaciones_generales_año = df_long.groupby('año')['importaciones'].sum()

# Suma de importaciones por mes y año
importaciones_por_mes = data.groupby('mes')[columns_interes].sum()
importaciones_por_año = data.groupby('año')[columns_interes].sum()

# **Filtros de selección con opción "Todos"**
col1, col2 = st.columns(2)

with col1:
    mes_seleccionado = st.selectbox(
        "Selecciona un mes:", 
        ["Todos"] + sorted(importaciones_por_mes.index.tolist())
    )

with col2:
    año_seleccionado = st.selectbox(
        "Selecciona un año:", 
        ["Todos"] + sorted(importaciones_por_año.index.tolist())
    )

# **Mostrar gráficos en columnas**
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Importaciones Totales en el Mes: {mes_seleccionado}")
    fig_mes, ax_mes = plt.subplots(figsize=(8, 5))

    if mes_seleccionado == "Todos":
        importaciones_por_mes.plot(kind='bar', ax=ax_mes)
    else:
        importaciones_por_mes.loc[[mes_seleccionado]].plot(kind='bar', ax=ax_mes)

    ax_mes.set_xlabel('Tipo de Combustible')
    ax_mes.set_ylabel('Importaciones')
    st.pyplot(fig_mes)

with col2:
    st.subheader(f"Importaciones Totales en el Año: {año_seleccionado}")
    fig_año, ax_año = plt.subplots(figsize=(8, 5))

    if año_seleccionado == "Todos":
        importaciones_por_año.plot(kind='bar', ax=ax_año)
    else:
        importaciones_por_año.loc[[año_seleccionado]].plot(kind='bar', ax=ax_año)

    ax_año.set_xlabel('Tipo de Combustible')
    ax_año.set_ylabel('Importaciones')
    st.pyplot(fig_año)

# **Fila inferior: Gráficos agregados por mes y año**
col3, col4 = st.columns(2)

with col3:
    st.subheader("Importaciones Totales Agregadas por Mes")
    fig_mes_total, ax_mes_total = plt.subplots(figsize=(8, 5))
    importaciones_generales_mes.plot(kind='bar', ax=ax_mes_total)
    ax_mes_total.set_xlabel('Mes')
    ax_mes_total.set_ylabel('Importaciones')
    st.pyplot(fig_mes_total)

with col4:
    st.subheader("Importaciones Totales Agregadas por Año")
    fig_año_total, ax_año_total = plt.subplots(figsize=(8, 5))
    importaciones_generales_año.plot(kind='bar', ax=ax_año_total)
    ax_año_total.set_xlabel('Año')
    ax_año_total.set_ylabel('Importaciones')
    st.pyplot(fig_año_total)

# **Filtro de tipo de combustible**
tipo_seleccionado = st.selectbox("Selecciona un tipo de combustible:", columns_interes)

# Filtrar datos según selección del usuario
df_filtrado = data[['Fecha', tipo_seleccionado]].dropna()

# **Histograma y gráfico Q-Q**
st.subheader(f"Análisis del combustible: {tipo_seleccionado}")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(df_filtrado[tipo_seleccionado], kde=True, ax=axes[0])
axes[0].set_title(f'Histograma de {tipo_seleccionado}')
stats.probplot(df_filtrado[tipo_seleccionado], dist="norm", plot=axes[1])
axes[1].set_title(f'Gráfico Q-Q de {tipo_seleccionado}')

st.pyplot(fig)

# **Gráfico interactivo con Plotly**
st.subheader(f"Tendencia de importaciones de {tipo_seleccionado} a lo largo del tiempo")
fig_line = px.line(df_filtrado, x='Fecha', y=tipo_seleccionado, 
                   title=f'Importaciones de {tipo_seleccionado} en el tiempo')

st.plotly_chart(fig_line, use_container_width=True)

# **Modelo de Regresión Lineal**
st.subheader(f"Modelo de Regresión Lineal para {tipo_seleccionado}")

df_filtrado['mes'] = df_filtrado['Fecha'].dt.month
df_filtrado['año'] = df_filtrado['Fecha'].dt.year
X = df_filtrado[['año', 'mes']]
y = df_filtrado[tipo_seleccionado]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = root_mean_squared_error(y_test, y_pred)

st.write(f"**Error cuadrático medio (MSE):** {mse:.2f}")

fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
ax_reg.plot(y_test.values, label='Valores Reales', marker='o')
ax_reg.plot(y_pred, label='Valores Predichos', marker='x')
ax_reg.set_title(f'Valores Reales vs Predichos para {tipo_seleccionado}')
ax_reg.legend()

st.pyplot(fig_reg)
