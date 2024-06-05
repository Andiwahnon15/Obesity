# app.py
import streamlit as st  # Import Streamlit for creating web app
from src.funciones import carga_limpieza_data, preparar_data, machine_learning

def main():
    
    st.title('Want to know if you may be overweight?')  # Set the title of the web app
    
    st.info("#### Just fill in the following information and you will get the answer.")
    import pandas as pd
    import numpy as np
    
    df=carga_limpieza_data("config.yaml")

    potential_categorical_from_numerical = df.select_dtypes("number").loc[:, df.select_dtypes("number").nunique() < 20]

    df_categorical = pd.concat([df.select_dtypes("object"), potential_categorical_from_numerical], axis=1)

    df_numerical = df.select_dtypes("number").drop(columns=potential_categorical_from_numerical.columns)
    #Training the model
    df_final = preparar_data(df)
    rf = machine_learning(df_final)

    # Creamos un nuevo dataframe para datos artificiales categoricos
    df_cat_to_predict = pd.DataFrame(columns= df_categorical.columns)

    # Eliminamos la columna 'obesity_level' ya que es la que queremos predecir
    df_cat_to_predict.drop(columns=['obesity_level'], inplace=True)

    # Generamos datos aleatorios
    st.markdown("##### Introduce your gender")
    gender_options = df_categorical['gender'].unique()
    temp_gender = np.random.choice(gender_options)
    gender = st.selectbox('Gender', options=gender_options)
    df_cat_to_predict.loc[1, "gender"] = gender

    st.markdown("<hr>", unsafe_allow_html=True)

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### How often do you drink alcohol?")
        calc_options = df_categorical['calc'].unique()
        temp_calc = np.random.choice(calc_options)
        calc = st.multiselect('Alcohol', options=calc_options)
        df_cat_to_predict.loc[1, "calc"] = calc

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you eat high caloric food frequently?")
        favc_options = df_categorical['favc'].unique()
        temp_favc = np.random.choice(favc_options)
        favc = st.multiselect('High caloric food', options=favc_options)
        df_cat_to_predict.loc[1, "favc"] = favc

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### Do you monitor the calories you eat daily?")
        scc_options= df_categorical['scc'].unique()
        temp_scc=np.random.choice(scc_options)
        scc = st.multiselect('Monitor calories', options=scc_options)
        df_cat_to_predict.loc[1, "scc"] = scc

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you smoke?")
        smoke_options= df_categorical['smoke'].unique()
        temp_smoke=np.random.choice(smoke_options)
        smoke = st.multiselect('Smoke', options=smoke_options)
        df_cat_to_predict.loc[1, "smoke"] = smoke

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### Your family suffers from obesity?")
        family_history_options= df_categorical['family_history'].unique()
        temp_family_history=np.random.choice(family_history_options)
        family_history = st.multiselect('Family history obesity', options=family_history_options)
        df_cat_to_predict.loc[1, "family_history"] = family_history

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you eat any food between meals?")
        caec_options= df_categorical['caec'].unique()
        temp_caec=np.random.choice(caec_options)
        caec = st.multiselect('Food between meals', options=caec_options)
        df_cat_to_predict.loc[1, "caec"] = caec

    st.markdown("##### Which transportation do you usually use?")
    mtrans_options= df_categorical['mtrans'].unique()
    temp_mtrans=np.random.choice(mtrans_options)
    mtrans = st.multiselect('Type of transport', options=mtrans_options)
    df_cat_to_predict.loc[1, "mtrans"] = mtrans
    
    ###

    # Creamos un nuevo dataframe para datos artificiales numericos
    df_num_predict = pd.DataFrame(columns= df_numerical.columns)

    # Generamos datos aleatorios

    st.markdown("##### Introduce your age")
    age_options= df_numerical['age'].unique()
    temp_age = st.slider('Age', min_value=0, max_value=100, step=1)
    df_num_predict.loc[1, "age"] = temp_age

    st.markdown("##### Introduce your height")
    height_options= df_numerical['height'].unique()
    temp_height = st.slider('cm', min_value=1.0, max_value=3.0, step=0.1)
    df_num_predict.loc[1, "height"] = temp_height

    st.markdown("##### Introduce your weight")
    weight_options= df_numerical['weight'].unique()
    temp_weight = st.slider('kg', min_value=0, max_value=250, step=1)
    df_num_predict.loc[1, "weight"] = temp_weight

    st.markdown("##### Do you usually eat vegetables in your meals?")
    fcvc_options= df_numerical['fcvc'].unique()
    temp_fcvc = st.slider('Vegetables', min_value=0, max_value=5, step=1)
    df_num_predict.loc[1, "fcvc"] = temp_fcvc

    st.markdown("##### How many main meals do you have daily?")
    ncp_options= df_numerical['ncp'].unique()
    temp_ncp = st.slider('Meals', min_value=0, max_value=5, step=1)
    df_num_predict.loc[1, "ncp"] = temp_ncp

    st.markdown("##### How much water do you drink daily?")
    ch2o_options= df_numerical['ch2o'].unique()
    temp_ch2o = st.slider('liters', min_value=0, max_value=5, step=1)
    df_num_predict.loc[1, "ch2o"] = temp_ch2o

    st.markdown("##### How often do you have physical activity?")
    faf_options= df_numerical['faf'].unique()
    temp_faf = st.slider('Days', min_value=0, max_value=7, step=1)
    df_num_predict.loc[1, "faf"] = temp_faf

    st.markdown("##### How much time do you use technological devices such as cell phone, videogames, television, computer and others?")
    tue_options= df_numerical['tue'].unique()
    temp_tue = st.slider(' Hours', min_value=0, max_value=10, step=1)
    df_num_predict.loc[1, "tue"] = temp_tue

    st.markdown("## Results")
    temp_bmi = (temp_weight / temp_height/ 100) ** 2
    df_num_predict.loc[1, "bmi"] = temp_bmi

    # Determinar el color del texto en función del resultado
    color = 'green'  # Por defecto, el color es verde
    if temp_bmi < 18.5:
        color = 'yellow'  # Cambiar a amarillo si el IMC es bajo
    elif temp_bmi > 25:
        color = 'red'  # Cambiar a rojo si el IMC es alto
    st.markdown(f'<p style="color:{color}">{temp_bmi}</p>', unsafe_allow_html=True)

# Python script entry point
if __name__ == '__main__':
     main()  # Call the main function when the script is executed

