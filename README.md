If error scikit-learn -> pip install "scikit-learn<1.6.0"

Tambien he quitado en el codigo de pred, una lista en el crear el nuevo dataframe

new_df = pd.DataFrame(row, columns=df_dropped_columns.columns)

Antes:

new_df = pd.DataFrame(\[row\], columns=df_dropped_columns.columns)

Tambien he agregado a las .drop(..., error='ignore'), porque no funciona bien lo de borrar columnas

Ejecutar la web

```
streamlit run Prueba.py
```