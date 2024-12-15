import streamlit as st
import pandas as pd

from utils.pred import FibroPred
import config as cfg

ruta_copia = cfg.OUTPUT_DATA

df = pd.read_csv(ruta_copia, sep=";")  # Usar coma como delimitador (por defecto)

# Configurar título de la página
st.title("Realitzar una predicció per inferència de dades")

# Mostrar los datos actuales
st.subheader("Selecciona una fila de la taula")
with st.form(key='Inferencia'):
    fila=st.number_input(f'Introduiu una fila per predir:', step=1, min_value=0)
    st.form_submit_button(label="Escollir fila")
    st.write(df.iloc[[fila]])

    actual_fila = df.iloc[[fila]]

left,center, right = st.columns(3)

model_type = -1

if left.button('Any 0'):
    model_type = 0

if center.button('Any 1'):
    model_type = 1

if right.button('Any 2'):
    model_type = 2

st.session_state['model_type'] = model_type

if st.session_state['model_type'] != -1:

    model = FibroPred(years = st.session_state['model_type'], typem = "patata")

    if 'model' not in st.session_state:
        st.session_state['model'] = model
    
    else:
        model = st.session_state['model']
        
    # st.session_state['model'] = model

    output, images = model.inference(row=actual_fila)
    
    st.session_state['output'] = output
    st.session_state['images'] = images

    st.subheader("Resultat de la predicció")

    # Crear 2 columnas
    left, center, right = st.columns(3)

    left.metric(label="Death Prediction", value=output['Death'], border=True)
    center.metric(label="Progressive disease", value=output['Progressive disease'], border=True)
    right.metric(label="Necessity of Transplantation Prediction", value=output['Necessity of transplantation'], border=True)

    st.session_state['counter'] = 0
    st.session_state['counter_max'] = len(images)

    counter = 0
    counter_max = len(images)

    images_per_list = len(images[counter])
    
    tab1, tab2, tab3 = st.tabs(cfg.TARGET_COLUMNS)
    
    with tab1:
        counter = 0
        left, right = st.columns(2)

        left.image(images[counter][0])

        right.image(images[counter][1])

        st.session_state['counter'] = counter
        
    with tab2:
        counter = 1
        left, right = st.columns(2)

        left.image(images[counter][0])

        right.image(images[counter][1])

        st.session_state['counter'] = counter
        
    with tab3:
        counter = 2
        left, right = st.columns(2)

        left.image(images[counter][0])

        right.image(images[counter][1])

        st.session_state['counter'] = counter
    



    












