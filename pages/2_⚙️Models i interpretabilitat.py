import streamlit as st
import pandas as pd

import config as cfg

def display_images(index):
    images = []
    captions = []
    
    for feature in cfg.TARGET_COLUMNS:
        images.append(f"images_general/bar_plot{feature}{index}.png")
        images.append(f"images_general/beeswarm_plot{feature}{index}.png")
        
        captions.append(f'Bar plot {feature}')
        captions.append(f'Beeswarm plot {feature}')
    cols = st.columns(2)  # Create 3 columns
    
    for i, image_path in enumerate(images):
        with cols[i % 2]:  # Rotate through the columns
            st.image(image_path, use_container_width =True, caption=captions[i])
            
def display_more_info(index):
    pass

df0 = pd.read_csv('model_results/any0.csv', sep=';')
print(df0)
df0 = df0.set_index(cfg.METRIC_COLUMN)

df1 = pd.read_csv('model_results/any1.csv', sep=';')
df1 = df1.set_index(cfg.METRIC_COLUMN)

df2 = pd.read_csv('model_results/any2.csv', sep=';')
df2 = df2.set_index(cfg.METRIC_COLUMN)

# Sample data for different "sheets"
sheet1 = df0

sheet2 = df1

sheet3 = df2

# Dictionary to store sheets
sheets = {
    "At diagnosis": sheet1,
    "After one year": sheet2,
    "After two years": sheet3,
}

# Tabs for different sheets
tab1, tab2, tab3 = st.tabs(sheets.keys())

with tab1:
    index = 0
    st.write(list(sheets.keys())[0])
    st.dataframe(sheet1)
    display_images(index)

with tab2:
    index = 1
    st.write(list(sheets.keys())[1])
    st.dataframe(sheet2)
    display_images(index)

with tab3:
    index = 2
    st.write(list(sheets.keys())[2])
    st.dataframe(sheet3)
    display_images(index)
