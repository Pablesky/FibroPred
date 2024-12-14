import streamlit as st
import pandas as pd

from utils.pred import FibroPred
import config as cfg

def make_inference(model_type, pandas_row):
    dict_output = output = {
            'Progressive disease':1.0, 
            'Death':1.0, 
            'Necessity of transplantation':0.0
        }

    images = ['images/test.jpg'] * 3
    images1 = ['images/test1.jpg'] * 3
    images = [images, images1]

    return dict_output, images


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

if left.button('atDiagnosticTime'):
    model_type = 0

if center.button('oneYearAfter'):
    model_type = 1

if right.button('twoYearsAfter'):
    model_type = 2

st.session_state['model_type'] = model_type

if st.session_state['model_type'] != -1:

    model = FibroPred(years = st.session_state['model_type'], typem = "patata")

    if 'model' not in st.session_state:
        st.session_state['model'] = model
    
    else:
        model = st.session_state['model']

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

    left, right = st.columns(2)

    left.image(images[counter][0])

    right.image(images[counter][1])

    if st.button('Next'):
        counter += 1
        counter = counter % 3

    st.session_state['counter'] = counter

else:
    if 'output' in st.session_state:

        output = st.session_state['output']
        images = st.session_state['images']

        left, center, right = st.columns(3)

        left.metric(label="Death Prediction", value=output['Death'], border=True)
        center.metric(label="Progressive disease", value=output['Progressive disease'], border=True)
        right.metric(label="Necessity of Transplantation Prediction", value=output['Necessity of transplantation'], border=True)

        counter = st.session_state['counter']
        counter_max = st.session_state['counter_max']

        images_per_list = len(images[counter])

        left, right = st.columns(2)

        left.image(images[counter][0])

        right.image(images[counter][1])

        if st.button('Next'):
            counter += 1
            counter = counter % counter_max
            st.session_state['counter'] = counter



    












