import streamlit as st
import pandas as pd
import os

import config as cfg

# Configurar título de la página
st.title("Introduir noves dades a la base de dades")

# Ruta del archivo original y copia
ruta_original = cfg.OUTPUT_DATA  # Cambia a tu archivo
ruta_copia = cfg.OUTPUT_DATA

df = pd.read_csv(ruta_copia, sep=";")  # Usar coma como delimitador (por defecto)


# Mostrar los datos actuales
st.subheader("Dades actuals a l'arxiu:")
st.dataframe(df)

# Crear un formulario para introducir nuevos datos
st.subheader("Afegir noves dades")


with st.form(key="formulario_datos"):
    # Crear un campo por cada columna del Excel
    nuevas_entradas = {}
    for columna in df.columns:
        nuevas_entradas[columna] = st.text_input(f"Introduïu {columna}:", key=columna)
    
    # Botón para enviar el formulario
    enviar = st.form_submit_button(label="Afegir Dades")
    
    # Procesar el formulario al enviarlo
    if enviar:
        # Convertir las entradas en un DataFrame de una fila
        nueva_fila = pd.DataFrame([nuevas_entradas])
        # Añadir la nueva fila al DataFrame existente
        df = pd.concat([df, nueva_fila], ignore_index=True)
        # Guardar el DataFrame actualizado en la copia del archivo
        df.to_csv(ruta_copia, index=False, sep=';')
        st.success("S'han afegit nous valors a la taula.")

# Mostrar el DataFrame actualizado
st.subheader("Dades Actualitzades:")
st.dataframe(df)