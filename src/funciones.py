def carga_limpieza_data(yalm_path):

    import pandas as pd
    import yaml

    #leemos el archivo Yaml en Python
    try:
        with open(yalm_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error leyendo el archivo .yaml:', e)
        return None

    #importamos los dataframes
    try:
        df = pd.read_csv(config['data']['df'], sep=",", header=0, low_memory=False)
    except Exception as e:
        print('Error importando la data', e)
        

    #Quitamos las mayusculas 
    df.columns=df.columns.str.lower()

    #Cambiamos los nombres de dos columnas "family_history_with_overweight" y "nobeyesdad"
    df=df.rename(columns={"family_history_with_overweight": "family_history", "nobeyesdad": "obesity_level"})

    return df

def grafico_genero(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #calculamos la proporción de cada uno de los géneros
    gender=df['gender'].value_counts(normalize=True)

    #creamos gráfico circular para mostrar la proporción de géneros
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1) 
    plt.pie(gender, labels=gender.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
    plt.title('Distribución de Género')
    plt.show()

def grafico_peso_genero(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Distribución del Peso por Género
    sns.boxplot(data=df, x='gender', y='weight')
    plt.title('Distribución del Peso por Género')
    plt.show()

def grafico_edad(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.histplot(data=df, x='age', kde=True)
    plt.title('Distribución de Edad')
    plt.show()

def grafico_edad_peso(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(data=df, x='age', y='weight', hue='gender')
    plt.title('Relación entre Edad y Peso')
    plt.show()

def grafico_nivel_obesidad(df):
    import matplotlib.pyplot as plt
    # Proporción de Niveles de Obesidad
    df['obesity_level'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Proporción de Niveles de Obesidad')
    plt.ylabel('')
    plt.show()

def grafico_imc_nivel_obesidad(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Calculo el índice de masa corporal (IMC) por nivel de obesidad
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    # Distribución IMC por nivel de obesidad
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x='obesity_level', y='bmi')
    plt.title('Distribución del IMC por Nivel de Obesidad')
    plt.show()

def grafico_imc_consumo_agua(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Relación entre IMC y consumo de agua diaria
    plt.figure(figsize=(25, 10))
    sns.scatterplot(data=df, x='ch2o', y='bmi', hue='obesity_level')
    plt.title('Relación entre IMC y Consumo de Agua Diario')
    plt.figure(figsize=(30, 12))
    plt.show()

def grafico_altura_peso(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Relación entre altura y peso
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=df, x='height', y='weight', hue='obesity_level')
    plt.title('Relación entre Altura y Peso')
    plt.show()

def matriz_correlacion_num(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Matriz de correlación entre variables númericas
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[['age', 'height', 'weight', 'ncp', 'ch2o', 'faf', 'tue', 'bmi']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()

def matriz_corr_todas_var(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    #calculamos la matriz de correlación para crear un mapa de calor que muestre la relación entre todas las variables

    #convertimos variables categóricas en variables dummy
    df_dummies = pd.get_dummies(df)

    #calculamos la matriz de correlación
    matriz_de_correlacion_1 = df_dummies.corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(matriz_de_correlacion_1, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación (Variables Dummy)')
    plt.show()

def preparar_data(df):
    import pandas as pd
    #seleccionamos las columnas categoricas y numericas y las separamos en dataframes distintos.
    potential_categorical_from_numerical = df.select_dtypes("number").loc[:, df.select_dtypes("number").nunique() < 20]

    df_categorical = pd.concat([df.select_dtypes("object"), potential_categorical_from_numerical], axis=1)

    df_numerical = df.select_dtypes("number").drop(columns=potential_categorical_from_numerical.columns)

    # Realizar One-Hot Encoding usando pd.get_dummies()
    df_encoded = pd.get_dummies(df_categorical, columns=['calc', 'caec', 'mtrans'], drop_first=True)

    # Convertir 'gender' a formato binario
    df_encoded['gender'] = df_encoded['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Asegurar que las columnas binarias estén en formato correcto (0 y 1)
    binary_columns = ['favc', 'scc', 'smoke', 'family_history']
    for col in binary_columns:
        df_encoded[col] = df_encoded[col].apply(lambda x: 1 if x == 'yes' else 0)

    # Unimos los dos dataframes
    df_final = pd.concat([df_encoded, df_numerical], axis=1)

    return df_final

def machine_learning(df_final):
    from sklearn.ensemble import RandomForestClassifier

    # Eliminamos la columna 'obesity_level' de los features y lo colocamos como target
    features = df_final.drop(columns = "obesity_level")
    target = df_final[["obesity_level"]]

    # Cambiamos el dtype a integer
    features[features.select_dtypes(include=["bool"]).columns]=features[features.select_dtypes(include=["bool"]).columns].astype(int)

    rf = RandomForestClassifier(max_depth=9, n_estimators=105)
    rf.fit(features, target)

    return rf 

def grafico_confusion(features, target):
    from sklearn.metrics import confusion_matrix
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(features)
    class_names = target['obesity_level'].unique()
    import plotly.graph_objects as go
    # Creating a confusion matrix heatmap using Plotly Graph Objects with true class names
    fig = go.Figure(data=go.Heatmap(
                    z=conf_matrix,
                    x=class_names,
                    y=class_names,
                    colorscale='Blues'))

    # Adding annotations
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            fig.add_annotation(
                x=j, y=i,
                text=str(conf_matrix[i][j]),
                showarrow=False,
                font=dict(color="black", size=12)
            )

    # Updating layout for better readability
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=800, height=800
    )

    fig.show()
