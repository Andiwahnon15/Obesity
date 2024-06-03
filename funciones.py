def carga_limpieza_data():
    import pandas as pd
    #Leemos la data
    url=r'C:\Users\andre\Downloads\ObesityDataSet_raw_and_data_sinthetic.csv'
    df=pd.read_csv(url)

    #Hacemos que se vean rodas las columnas
    pd.set_option('display.max_columns', None)

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
    #calculamos la matriz de correlación para crear un mapa de calor que muestre la relación entre todas las variables

    #convertimos variables categóricas en variables dummy
    df_dummies = pd.get_dummies(df)

    #calculamos la matriz de correlación
    matriz_de_correlacion_1 = df_dummies.corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(matriz_de_correlacion_1, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación (Variables Dummy)')
    plt.show()

def 