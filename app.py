# app.py
import streamlit as st  # Import Streamlit for creating web app
from src.funciones import carga_limpieza_data, preparar_data, machine_learning

def main():
    import pandas as pd
    import numpy as np

    df=carga_limpieza_data()

    potential_categorical_from_numerical = df.select_dtypes("number").loc[:, df.select_dtypes("number").nunique() < 20]

    df_categorical = pd.concat([df.select_dtypes("object"), potential_categorical_from_numerical], axis=1)

    df_numerical = df.select_dtypes("number").drop(columns=potential_categorical_from_numerical.columns)

    # Creamos un nuevo dataframe para datos artificiales categoricos
    df_cat_to_predict = pd.DataFrame(columns= df_categorical.columns)

    # Eliminamos la columna 'obesity_level' ya que es la que queremos predecir
    df_cat_to_predict.drop(columns=['obesity_level'], inplace=True)

    # Generamos datos aleatorios
    gender_options = df_categorical['gender'].unique()
    temp_gender = np.random.choice(gender_options)
    gender = st.sidebar.selectbox('Gender', options=gender_options, default=temp_gender)
    df_cat_to_predict.loc[1, "gender"] = gender

    calc_options= df_categorical['calc'].unique()
    temp_calc=np.random.choice(calc_options)
    calc = st.sidebar.multiselect('Drink Alcohol', options=calc_options, default=temp_calc)
    df_cat_to_predict.loc[1, "calc"] = calc

    favc_options= df_categorical['favc'].unique()
    temp_favc=np.random.choice(favc_options)
    favc = st.sidebar.multiselect('High caloric food', options=favc_options, default=temp_favc)
    df_cat_to_predict.loc[1, "favc"] = favc

    scc_options= df_categorical['scc'].unique()
    temp_scc=np.random.choice(scc_options)
    scc = st.sidebar.multiselect('Monitor calories', options=scc_options, default=temp_scc)
    df_cat_to_predict.loc[1, "scc"] = scc

    smoke_options= df_categorical['smoke'].unique()
    temp_smoke=np.random.choice(smoke_options)
    smoke = st.sidebar.multiselect('Smoke', options=smoke_options, default=temp_smoke)
    df_cat_to_predict.loc[1, "smoke"] = smoke

    family_history_options= df_categorical['family_history'].unique()
    temp_family_history=np.random.choice(family_history_options)
    family_history = st.sidebar.multiselect('Family history obesity', options=family_history_options, default=temp_family_history)
    df_cat_to_predict.loc[1, "family_history"] = family_history

    caec_options= df_categorical['caec'].unique()
    temp_caec=np.random.choice(caec_options)
    caec = st.sidebar.multiselect('Food between meals', options=caec_options, default=temp_caec)
    df_cat_to_predict.loc[1, "caec"] = caec

    mtrans_options= df_categorical['mtrans'].unique()
    temp_mtrans=np.random.choice(mtrans_options)
    mtrans = st.sidebar.multiselect('Type of transport', options=mtrans_options, default=temp_mtrans)
    df_cat_to_predict.loc[1, "mtrans"] = mtrans
    
    ###

    # Creamos un nuevo dataframe para datos artificiales numericos
    df_num_predict = pd.DataFrame(columns= df_numerical.columns)

    # Generamos datos aleatorios
    age_options= df_numerical['age'].unique()
    temp_age = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "age"] = temp_age

    height_options= df_numerical['height'].unique()
    temp_height = st.sidebar.slider('Minimum Rating', min_value=0, max_value=10, value=2, step=1)
    df_num_predict.loc[1, "height"] = temp_height

    weight_options= df_numerical['weight'].unique()
    temp_weight = st.sidebar.slider('Minimum Rating', min_value=0, max_value=250, value=50, step=1)
    df_num_predict.loc[1, "weight"] = temp_weight

    fcvc_options= df_numerical['fcvc'].unique()
    temp_fcvc = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "fcvc"] = temp_fcvc

    ncp_options= df_numerical['ncp'].unique()
    temp_ncp = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "ncp"] = temp_ncp

    ch2o_options= df_numerical['ch2o'].unique()
    temp_ch2o = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "ch2o"] = temp_ch2o

    faf_options= df_numerical['faf'].unique()
    temp_faf = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "faf"] = temp_faf

    tue_options= df_numerical['tue'].unique()
    temp_tue = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "tue"] = temp_tue

    bmi_options= df_numerical['bmi'].unique()
    temp_bmi = st.sidebar.slider('Minimum Rating', min_value=0, max_value=100, value=25, step=1)
    df_num_predict.loc[1, "bmi"] = temp_bmi

    """-----------------------------------------------------------------"""

    



