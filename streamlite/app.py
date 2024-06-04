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
    df_cat_to_predict.loc[1, "gender"] = temp_gender

    calc_options= df_categorical['calc'].unique()
    temp_calc=np.random.choice(calc_options)
    df_cat_to_predict.loc[1, "calc"] = temp_calc

    favc_options= df_categorical['favc'].unique()
    temp_favc=np.random.choice(favc_options)
    df_cat_to_predict.loc[1, "favc"] = temp_favc

    scc_options= df_categorical['scc'].unique()
    temp_scc=np.random.choice(scc_options)
    df_cat_to_predict.loc[1, "scc"] = temp_scc

    smoke_options= df_categorical['smoke'].unique()
    temp_smoke=np.random.choice(smoke_options)
    df_cat_to_predict.loc[1, "smoke"] = temp_smoke

    family_history_options= df_categorical['family_history'].unique()
    temp_family_history=np.random.choice(family_history_options)
    df_cat_to_predict.loc[1, "family_history"] = temp_family_history

    caec_options= df_categorical['caec'].unique()
    temp_caec=np.random.choice(caec_options)
    df_cat_to_predict.loc[1, "caec"] = temp_caec

    mtrans_options= df_categorical['mtrans'].unique()
    temp_mtrans=np.random.choice(mtrans_options)
    df_cat_to_predict.loc[1, "mtrans"] = temp_mtrans


    ###

    # Creamos un nuevo dataframe para datos artificiales numericos
    df_num_predict = pd.DataFrame(columns= df_numerical.columns)

    # Generamos datos aleatorios
    age_options= df_numerical['age'].unique()
    temp_age=np.random.choice(age_options)
    df_num_predict.loc[1, "age"] = temp_age

    height_options= df_numerical['height'].unique()
    temp_height=np.random.choice(height_options)
    df_num_predict.loc[1, "height"] = temp_height

    weight_options= df_numerical['weight'].unique()
    temp_weight=np.random.choice(weight_options)
    df_num_predict.loc[1, "weight"] = temp_weight

    fcvc_options= df_numerical['fcvc'].unique()
    temp_fcvc=np.random.choice(fcvc_options)
    df_num_predict.loc[1, "fcvc"] = temp_fcvc

    ncp_options= df_numerical['ncp'].unique()
    temp_ncp=np.random.choice(ncp_options)
    df_num_predict.loc[1, "ncp"] = temp_ncp

    ch2o_options= df_numerical['ch2o'].unique()
    temp_ch2o=np.random.choice(ch2o_options)
    df_num_predict.loc[1, "ch2o"] = temp_ch2o

    faf_options= df_numerical['faf'].unique()
    temp_faf=np.random.choice(faf_options)
    df_num_predict.loc[1, "faf"] = temp_faf

    tue_options= df_numerical['tue'].unique()
    temp_tue=np.random.choice(tue_options)
    df_num_predict.loc[1, "tue"] = temp_tue

    bmi_options= df_numerical['bmi'].unique()
    temp_bmi=np.random.choice(bmi_options)
    df_num_predict.loc[1, "bmi"] = temp_bmi

    #-------------------------------------

    questions = st.sidebar.multiselect('questions', options=df_cat_to_predict[
        'gender', 'calc', 'favc', 'scc', 'smoke', 'family_history', 'caec', 'metrans'
        ].unique(), default=df_cat_to_predict['gender', 'calc', 'favc', 'scc', 'smoke', 'family_history', 'caec', 'metrans'].unique())
    