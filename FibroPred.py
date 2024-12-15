import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg

import warnings

warnings.filterwarnings('ignore')

if 'data' not in st.session_state:
    data=pd.read_csv(cfg.INPUT_DATA, sep=';')
    data.to_csv(cfg.OUTPUT_DATA, index=False, sep=';')
    st.session_state['data'] = True

else:
    data=pd.read_csv(cfg.OUTPUT_DATA, sep=';')

st.set_page_config(
    page_title="FibroPred"
)


st.write("# 🧬Predictor pronòstic en fibrosi pulmonar (FibroPred)")
st.write("## Introducció")


st.markdown(
    """
    La fibrosi pulmonar és una malaltia crònica que resulta de la mala reparació del pulmó 
    després de diferents tipus de lesions. Aquesta cicatriu al teixit pulmonar impedeix el 
    pas normal d'oxigen, causant retracció, enduriment i progressiva insuficiència respiratòria. 
    Les malalties intersticials fibrosants, que inclouen la fibrosi pulmonar idiopàtica i altres tipus, 
    presenten un curs clínic i resposta terapèutica molt variable entre pacients. 
    A mesura que la malaltia avança, la qualitat de vida dels pacients es veu greument limitada, 
    i l'efecte dels tractaments antifibròtics és imprevisible.
"""
)

st.write("## Objectiu")


st.markdown(
    """
    Desenvolupar un model predictiu (FibroPred) al diagnòstic per millorar la presa de decisions 
    terapèutiques des de la primera visita, ajudant a predir la progressió de la malaltia en pacients 
    amb malalties pulmonars intersticials fibrosants.
"""
)


st.write("## Dades")


st.markdown(
    """
    A continuació es mostren el conjunt de dades amb el que s'ha treballat, i algunes
    gràfiques que permeten una comprensió immediata dels factors que influeixen en la 
    progressió de la malaltia i ajudarà en la presa de decisions terapèutiques des del diagnòstic.
"""
)


st.write(data.head())


# Muertes por género

st.write("### Nombre de morts per gènere")

st.markdown(
    """
    Aquí es mostra la distribució de les morts segons el gènere dels pacients. 
    Aquesta anàlisi és essencial per entendre com el sexe pot influir en la mortalitat associada a la malaltia 
    i identificar possibles diferències en l'impacte de la fibrosi pulmonar.
"""

)



explode = (0, 0.1)  # solo "Saque" el 2do pedazo (ejem. 'cerdos')
death_counts = data.groupby('Sex')['Death'].sum()
# Crear la figura y ajustar su tamaño
fig2, ax2 = plt.subplots(figsize=(14, 9))
# Gráfico de pie
ax2.pie(death_counts, labels=death_counts.index, autopct='%1.1f%%', shadow=True, startangle=0, explode=explode, colors=['Orange', None])
ax2.set_title('Distribució de Morts per Gènere')
# Mostrar el gráfico en Streamlit
st.pyplot(fig2)




# MUERTES POR RANGO DE EDADES



st.write("### Nombre de morts per rang d'edats")

st.markdown(
    """
    Altres dades importants a tenir en compte es la distribució de les morts agrupades per rangs d'edat. 
    L'objectiu és visualitzar com varia la mortalitat en funció de l'edat i identificar les edats 
    amb un risc més elevat de defunció a causa de la malaltia.
"""
)



# Filtrar solo las filas donde 'Death' es 1
deaths = data[data['Death'] == 1][cfg.AGE_COLUMN]

# Contar las ocurrencias de cada categoría de edad
death_counts_by_age_range = deaths.value_counts(sort=False).sort_index()

# Crear el gráfico como un histograma manual
fig, ax = plt.subplots(figsize=(8, 5))

# Generar las barras con las posiciones ajustadas
x = range(len(death_counts_by_age_range))  # Posiciones de las barras
ax.bar(x, death_counts_by_age_range.values, edgecolor='black', alpha=0.7, width=0.8)

# Ajustar las etiquetas del eje x con los nombres de las categorías
ax.set_xticks(x)
ax.set_xticklabels(death_counts_by_age_range.index, rotation=45)

# Configurar títulos y etiquetas
ax.set_xlabel("Rang d'edat")
ax.set_ylabel('Quantitat de Morts')
ax.set_title("Distribució de Morts per rangs d'edat")

# Mostrar el gráfico en Streamlit
st.pyplot(fig)







st.write("### Nombre de morts segons tabaquisme")

st.markdown(
    """
    A continuació es mostra un gràfic on s'analitza la relació entre l'hàbit de fumar i la mortalitat, 
    diferenciant entre fumadors, exfumadors i persones que mai han fumat. Aquesta informació ajuda a avaluar 
    l'impacte del tabaquisme sobre la malaltia i la seva progressió.
"""
)



# Configurar el tamaño de las figuras
width = 7  # Ancho de la figura
height = 5  # Alto de la figura


# Reemplazar los valores numericos por etiquetas en 'TOBACCO' antes de graficar
data['TOBACCO'] = data['TOBACCO'].map({0: 'No fumador', 1: 'Fumador', 2: 'Exfumador'})

# Agrupar los datos por 'TOBACCO' para contar las muertes y los casos vivos
alive_counts_by_tobacco = data[data['Death'] == 0].groupby('TOBACCO').size()  # Vivos
death_counts_by_tobacco = data[data['Death'] == 1].groupby('TOBACCO').size()  # Muertes

# Convertir las series a DataFrames para hacer el gráfico apilado
df_alive = alive_counts_by_tobacco.reset_index(name='Vius')
df_deaths = death_counts_by_tobacco.reset_index(name='Morts')

# Ajustar los datos para el gráfico apilado
fig, ax = plt.subplots(figsize=(10, 6))



# Gráfico apilado para muertes y diagnósticos
ax.bar(df_alive['TOBACCO'], df_alive['Vius'], color='blue', label='Vius', width=0.5, alpha=0.2)
ax.bar(df_deaths['TOBACCO'], df_deaths['Morts'], color='blue', label='Morts', width=0.5, bottom=0, alpha=0.7)



# Etiquetas y Títulos
ax.set_xlabel('Hàbit de fumar')
ax.set_ylabel('Número')
ax.set_title('Distribució de Morts i Vius segons l\'hàbit de fumar')
ax.legend()



# Mostrar el gráfico en Streamlit
st.pyplot(fig)







