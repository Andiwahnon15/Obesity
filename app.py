# app.py
import streamlit as st  # Import Streamlit for creating web app
from src.funciones import carga_limpieza_data, preparar_data, machine_learning
from PIL import Image

def main():
    
    #st.title('Want to know if you may be overweight?')  # Set the title of the web app
    st.write('<h1 style="color: orange;">Want to know if you may be overweight?</h1>', unsafe_allow_html=True)
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
    gender = st.selectbox('Gender', options=gender_options, index=0)
    df_cat_to_predict.loc[1, "gender"] = gender

    st.markdown("<hr>", unsafe_allow_html=True)

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### How often do you drink alcohol?")
        calc_options = df_categorical['calc'].unique()
        temp_calc = np.random.choice(calc_options)
        calc = st.selectbox('Alcohol', options=calc_options, index=0)
        df_cat_to_predict.loc[1, "calc"] = calc

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you eat high caloric food frequently?")
        favc_options = df_categorical['favc'].unique()
        temp_favc = np.random.choice(favc_options)
        favc = st.selectbox('High caloric food', options=favc_options, index=0)
        df_cat_to_predict.loc[1, "favc"] = favc

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### Do you monitor the calories you eat daily?")
        scc_options= df_categorical['scc'].unique()
        temp_scc=np.random.choice(scc_options)
        scc = st.selectbox('Monitor calories', options=scc_options, index=0)
        df_cat_to_predict.loc[1, "scc"] = scc

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you smoke?")
        smoke_options= df_categorical['smoke'].unique()
        temp_smoke=np.random.choice(smoke_options)
        smoke = st.selectbox('Smoke', options=smoke_options, index=0)
        df_cat_to_predict.loc[1, "smoke"] = smoke

    #Creamos las columnas
    left_column, right_column = st.columns(2)

    # Pregunta 1: "How often do you drink alcohol?"
    with left_column:
        st.markdown("##### Your family suffers from obesity?")
        family_history_options= df_categorical['family_history'].unique()
        temp_family_history=np.random.choice(family_history_options)
        family_history = st.selectbox('Family history obesity', options=family_history_options, index=0)
        df_cat_to_predict.loc[1, "family_history"] = family_history

    st.markdown("<hr>", unsafe_allow_html=True)

    # Pregunta 2: "Do you eat high caloric food frequently?"
    with right_column:
        st.markdown("##### Do you eat any food between meals?")
        caec_options= df_categorical['caec'].unique()
        temp_caec=np.random.choice(caec_options)
        caec = st.selectbox('Food between meals', options=caec_options, index=0)
        df_cat_to_predict.loc[1, "caec"] = caec

    st.markdown("##### Which transportation do you usually use?")
    mtrans_options= df_categorical['mtrans'].unique()
    temp_mtrans=np.random.choice(mtrans_options)
    mtrans = st.selectbox('Type of transport', options=mtrans_options, index=0)
    df_cat_to_predict.loc[1, "mtrans"] = mtrans
    
    #Creamos las columnas faltantes
    new_columns = ['calc_Always','calc_Frequently',
        'calc_Sometimes', 'calc_no','caec_Always','caec_Frequently', 'caec_Sometimes',
        'caec_no','mtrans_Automobile','mtrans_Bike', 'mtrans_Motorbike',
        'mtrans_Public_Transportation', 'mtrans_Walking']

    # Añadimos las nuevas columnas
    for col in new_columns:
        df_cat_to_predict[col] = np.nan
        
    # Rellenamos las filas de las columnas nuevas con los datos aleatorios que se generen
    df_cat_to_predict.loc[1, "calc_" + df_cat_to_predict['calc'][1]] = 1 
    df_cat_to_predict.loc[1, "caec_" + df_cat_to_predict['caec'][1]] = 1 
    df_cat_to_predict.loc[1, "mtrans_" + df_cat_to_predict['mtrans'][1]] = 1 
        
    # Eliminamos las columnas sobrantes 
    df_cat_to_predict.drop(columns=['calc', 'caec', 'mtrans'], inplace=True)

    # Rellenamos los Na con 0
    df_cat_to_predict.fillna(0, inplace=True)
        
        
        
        ###

    # Creamos un nuevo dataframe para datos artificiales numericos
    df_num_predict = pd.DataFrame(columns= df_numerical.columns)

    # Generamos datos aleatorios

    st.markdown("##### Introduce your age")
    age_options= df_numerical['age'].unique()
    temp_age = st.slider('Age', min_value=0, max_value=100, value=20, step=1)
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
    temp_tue = st.slider(' Hours', min_value=0, max_value=10, value=5, step=1)
    df_num_predict.loc[1, "tue"] = temp_tue

    st.write('<h1 style="color: pink;">Results</h1>', unsafe_allow_html=True)
    temp_bmi = (temp_weight / temp_height/ 100) ** 2
    df_num_predict.loc[1, "bmi"] = temp_bmi

    #Predicción
    df_predict = pd.concat([df_cat_to_predict, df_num_predict], axis=1)
    df_predict[df_predict.select_dtypes(include=["bool"]).columns]=df_predict[df_predict.select_dtypes(include=["bool"]).columns].astype(int)
    

    #Nuevo dataframe para predicción
    df_predict['gender'] = df_predict['gender'].replace({'Male': 0, 'Female':1})
    df_predict['favc'] = df_predict['favc'].replace({'no': 0, 'yes': 1})
    df_predict['scc'] = df_predict['scc'].replace({'no': 0, 'yes': 1})
    df_predict['smoke'] = df_predict['smoke'].replace({'no': 0, 'yes': 1})
    df_predict['family_history'] = df_predict['family_history'].replace({'no': 0, 'yes': 1})

    st.dataframe(df_predict)

    result= rf.predict(df_predict)[0]
    st.text(result)

    if result == 'Insufficient_Weight':
        st.write('<span style="color: #e63946; font-size: 20px;">Insufficient Weight</span>', unsafe_allow_html=True)
    elif result == 'Normal_Weight':
        st.write('<span style="color: #a8dadc; font-size: 20px;">Normal Weight</span>', unsafe_allow_html=True)
    elif result == 'Overweight_Level_I':
        st.write('<span style="color: #a8dadc; font-size: 20px;">Overweight Level I</span>', unsafe_allow_html=True)
    elif result == 'Overweight_Level_II':
        st.write('<span style="color: #a8dadc; font-size: 20px;">Overweight Level II</span>', unsafe_allow_html=True)
    elif result == 'Obesity_Type_I':
        st.write('<span style="color: #a8dadc; font-size: 20px;">Obesity Type I</span>', unsafe_allow_html=True)
    elif result == 'Obesity_Type_II':
        st.write('<span style="color: #a8dadc; font-size: 20px;">Obesity Type II</span>', unsafe_allow_html=True)
    else:
        st.write('<span style="color: #f4a261; font-size: 20px;">Obesity_Type_III</span>', unsafe_allow_html=True)

    # Mostrar imagen en la barra lateral
    #st.sidebar.image('alimentación-saludable.jpg', use_column_width=True)
    
    # Agregar las URLs de las imágenes desde Google Drive
    #image_url_left = "https://drive.google.com/file/d/1Dk83VaD2v3uEhBcq7hu098uAjtNItmnL/view?usp=sharing"
    #image_url_right = "https://drive.google.com/uc?export=download&id=1Dk83VaD2v3uEhBcq7hu098uAjtNItmnL"
    """
    # Mostrar las imágenes en los laterales
    col1, col2 = st.columns(2)
    with col1:
        st.sidebar.image(image_url_left, use_column_width=True)

    with col2:
        st.sidebar.image(image_url_right, use_column_width=True)
    """
    #img = Image.open("alimentacion-saludable.png")
    #st.sidebar.image(img, width=200)

# Python script entry point
if __name__ == '__main__':
     main()  # Call the main function when the script is executed

