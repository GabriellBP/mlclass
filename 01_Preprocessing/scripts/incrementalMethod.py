import json
import threading
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import math
from random import randrange
import numpy as np

global max_accuracy, missing_pairs, local_max_accuracy
max_accuracy = 0.83673469387755
local_max_accuracy = 0
n = 3
attempts = 1
best_dataset = pd.read_csv('csv/naive_diabetes_dataset.csv')
missing_dataset = pd.read_csv('../diabetes_dataset.csv')
missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def find_all_missing_pairs(dataset):
    pairs = {}

    for index, row in dataset.iterrows():
        for col in missing_cols:
            if math.isnan(row[col]):
                try:
                    pairs[index] += [col]
                except:
                    pairs[index] = [col]

    return pairs


def generate_turns(i, a, total):
    if i == 0:
        total += [a]
        return
    for j in range(2):
        generate_turns(i - 1, a + [j], total)


def make_request():
    global max_accuracy, missing_pairs, local_max_accuracy

    count = 0
    for person_id in missing_pairs:
        count += 1
        turns = []
        generate_turns(len(missing_pairs[person_id]), [], turns)
        print('#' * 15, 'NEXT PERSON [ID: {}, {}/{}]'.format(person_id, count, len(missing_pairs)), '#' * 15, missing_pairs[person_id])
        print('turns:', turns, '\n')

        best_row = best_dataset.iloc[person_id].copy()
        for i in range(attempts):
            for turn in turns:
                print('selected turn:', turn)
                for index, col in enumerate(missing_pairs[person_id]):
                    mean = best_dataset[col].mean()
                    sd = best_dataset[col].std()

                    if turn[index] == 0:
                        best_dataset.loc[person_id, col] = randrange(max(0, int(mean - n * sd)), int(mean))
                    else:
                        best_dataset.loc[person_id, col] = randrange(int(mean), int(mean + n * sd))

                print(str(best_row[missing_pairs[person_id]]) + '\n----------\n' + str(best_dataset.iloc[person_id][missing_pairs[person_id]]) + '\n')
                response = send_request(best_dataset)
                if response['accuracy'] > local_max_accuracy:
                    local_max_accuracy = response['accuracy']
                    best_row = best_dataset.iloc[person_id].copy()

                    if local_max_accuracy > max_accuracy:
                        max_accuracy = response['accuracy']
                        best_dataset.to_csv('csv/' + str(response['accuracy']) + '_new_data_brute.csv')
                print(' - Resposta do servidor:\n', response, '\n')

        best_dataset.iloc[person_id] = best_row
        print('-' * 20, 'BEST ROW', '-' * 20)
        print('- BEST LOCAL ACCURACY:', local_max_accuracy)
        print(best_row[missing_pairs[person_id]])
        print('-' * 50, '\n\n')


    make_request()


def send_request(dataset):
    global max_accuracy

    data = dataset
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

    while True:
        # Enviando requisição e salvando o objeto resposta
        r = requests.post(url=URL, data=data)

        # Extraindo e imprimindo o texto da resposta
        res = json.loads(r.text)

        try:
            if res['accuracy'] is not None:
                break
        except:
            print(' - Resposta do servidor:\n', r.text, '\n')
            continue

    return res


missing_pairs = find_all_missing_pairs(missing_dataset)
make_request()
