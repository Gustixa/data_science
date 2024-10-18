import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import plotly.express as px

# Cargar datos
data = pd.read_csv("depurado.csv")

# Configurar Streamlit
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
df_long = pd.melt(df_combustibles, id_vars=['Fecha', 'año', 'mes'], 
                  value_vars=columns_interes, 
                  var_name='tipo_combustible', 
                  value_name='importaciones')

# Título de la aplicación
st.title("Análisis de Importaciones de Combustibles en Guatemala")

# Filtro de selección de tipo de combustible
tipo_seleccionado = st.selectbox("Selecciona un tipo de combustible:", columns_interes)

# Filtrar datos según selección del usuario
df_filtrado = data[['Fecha', tipo_seleccionado]].dropna()

# **Histograma y gráfico Q-Q**
st.subheader(f"Análisis del combustible: {tipo_seleccionado}")

# Crear histogramas y gráficos Q-Q dinámicamente con Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histograma
sns.histplot(df_filtrado[tipo_seleccionado], kde=True, ax=axes[0])
axes[0].set_title(f'Histograma de {tipo_seleccionado}')

# Gráfico Q-Q
stats.probplot(df_filtrado[tipo_seleccionado], dist="norm", plot=axes[1])
axes[1].set_title(f'Gráfico Q-Q de {tipo_seleccionado}')

# Ajustar el layout
plt.tight_layout()

# Mostrar los gráficos en Streamlit
st.pyplot(fig)

# **Gráfico Interactivo con Plotly**
st.subheader(f"Tendencia de importaciones de {tipo_seleccionado} a lo largo del tiempo")

fig_line = px.line(df_filtrado, x='Fecha', y=tipo_seleccionado, 
                   title=f'Importaciones de {tipo_seleccionado} en el tiempo',
                   labels={'Fecha': 'Fecha', tipo_seleccionado: 'Importaciones'})

st.plotly_chart(fig_line, use_container_width=True)
