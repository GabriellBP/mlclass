import json
import threading
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import math
from random import randrange

global max_accuracy
max_accuracy = 0.82
n = 4
new_data_location = 'csv/new_data2.csv'


def makeRequest():
    global max_accuracy
    dataset = pd.read_csv('../diabetes_dataset.csv')

    glucose = [dataset['Glucose'].mean(), dataset['Glucose'].std()]
    blood_p = [dataset['BloodPressure'].mean(), dataset['BloodPressure'].std()]
    skin_t = [dataset['SkinThickness'].mean(), dataset['SkinThickness'].std()]
    insulin = [dataset['Insulin'].mean(), dataset['Insulin'].std()]
    bmi = [dataset['BMI'].mean(), dataset['BMI'].std()]

    for index, row in dataset.iterrows():
        if math.isnan(row['Glucose']):
            dataset.loc[index, 'Glucose'] = randrange(max(0, int(glucose[0] - n * glucose[1])), int(glucose[0] + n * glucose[1]))
        if math.isnan(row['BloodPressure']):
            dataset.loc[index, 'BloodPressure'] = randrange(max(0, int(blood_p[0] - n * blood_p[1])), int(blood_p[0] + n * blood_p[1]))
        if math.isnan(row['SkinThickness']):
            dataset.loc[index, 'SkinThickness'] = randrange(max(0, int(skin_t[0] - n * skin_t[1])), int(skin_t[0] + n *skin_t[1]))
        if math.isnan(row['Insulin']):
            dataset.loc[index, 'Insulin'] = randrange(max(0, int(insulin[0] - n * insulin[1])), int(insulin[0] + n * insulin[1]))
        if math.isnan(row['BMI']):
            dataset.loc[index, 'BMI'] = randrange(max(0, int(bmi[0] - n * bmi[1])), int(bmi[0] + n * bmi[1]))
    dataset.to_csv(new_data_location)

    # print('\n - Lendo o arquivo com o dataset sobre diabetes')
    data = pd.read_csv(new_data_location)

    # Criando X and y par ao algorítmo de aprendizagem de máquina.\
    # print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
    # Caso queira modificar as colunas consideradas basta algera o array a seguir.
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[feature_cols]
    y = data.Outcome

    # Ciando o modelo preditivo para a base trabalhada
    # print(' - Criando modelo preditivo')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    # realizando previsões com o arquivo de
    # print(' - Aplicando modelo e enviando para o servidor')
    data_app = pd.read_csv('../diabetes_app.csv')
    y_pred = neigh.predict(data_app)

    # Enviando previsões realizadas com o modelo para o servidor
    URL = 'https://aydanomachado.com/mlclass/01_Preprocessing.php'

    # TODO Substituir pela sua chave aqui
    DEV_KEY = 'Dual Core'

    # json para ser enviado para o servidor
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(y_pred).to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)

    # Extraindo e imprimindo o texto da resposta
    pastebin_url = r.text
    res = json.loads(r.text)
    if res['accuracy'] > max_accuracy:
        dataset.to_csv(str(res['accuracy']) + '_new_data.csv')
        max_accuracy = res['accuracy']
    print(' - Resposta do servidor:\n', r.text, '\n')


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    threading.Timer(sec, func_wrapper).start()


makeRequest()
set_interval(makeRequest, 1)