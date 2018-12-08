import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import math
from random import randrange
import os
import datetime

global max_accuracy, missing_pairs, local_max_accuracy, allow_invert_on_no_gaps_row
max_accuracy = 0.83673469387755
local_max_accuracy = 0.83673469387755
n = 3
attempts = 1
allow_invert_on_no_gaps_row = False
best_dataset = pd.read_csv('csv/incremental_new_data_0.83673469387755.csv')
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


def has_gaps(row, cols):
    for col in cols:
        if row[col] == 0:
            return True
    return False


def make_request():
    global max_accuracy, missing_pairs, local_max_accuracy, allow_invert_on_no_gaps_row

    count = 0
    modified_person = False
    for person_id in missing_pairs:
        count += 1
        turns = []
        generate_turns(len(missing_pairs[person_id]), [], turns)
        row_has_gaps = has_gaps(best_dataset.iloc[person_id], missing_pairs[person_id])
        original_row = best_dataset.iloc[person_id].copy()

        print('#' * 15, 'NEXT PERSON [ID: {}, {}/{}], {}GAPS FOUND'.format(person_id, count, len(missing_pairs),
                                                                           '' if row_has_gaps else 'NO '),
              '#' * 15, missing_pairs[person_id])
        print('invert values on gaps filled:', allow_invert_on_no_gaps_row)
        print('turns:', turns, '\n')

        best_row = best_dataset.iloc[person_id].copy()
        for i in range(attempts):
            for turn in turns:
                print('selected turn:', turn)
                modified = False
                for index, col in enumerate(missing_pairs[person_id]):
                    mean = missing_dataset[col].mean()
                    sd = missing_dataset[col].std()

                    if row_has_gaps:
                        if original_row[col] == 0:
                            modified = True
                            if turn[index] == 0:
                                best_dataset.loc[person_id, col] = randrange(max(0, int(mean - n * sd)), int(mean))
                            else:
                                best_dataset.loc[person_id, col] = randrange(int(mean), int(mean + n * sd))
                    elif allow_invert_on_no_gaps_row:
                        is_above = True if best_dataset.loc[person_id, col] > mean else False
                        if turn[index] == 0:
                            modified = True
                            best_dataset.loc[person_id, col] = randrange(max(0, int(mean - n * sd)), int(mean)) \
                                if is_above else randrange(int(mean), int(mean + n * sd))
                        # else:
                        #     best_dataset.loc[person_id, col] = randrange(int(mean), int(mean + n * sd)) if is_above \
                        #                                         else randrange(max(0, int(mean - n * sd)), int(mean))

                if modified:
                    modified_person = True
                    print('CURRENT BEST ROW:\n' + str(best_row[missing_pairs[person_id]]) + '\n--------\nCURRENT ROW:\n'
                          + str(best_dataset.iloc[person_id][missing_pairs[person_id]]) + '\n')
                    response = send_request(best_dataset)
                    if response['accuracy'] > local_max_accuracy:
                        if os.path.exists('csv/incremental_new_data_' + str(local_max_accuracy) + '.csv'):
                            os.remove('csv/incremental_new_data_' + str(local_max_accuracy) + '.csv')

                        local_max_accuracy = response['accuracy']
                        best_dataset.to_csv('csv/incremental_new_data_' + str(local_max_accuracy) + '.csv')
                        best_row = best_dataset.iloc[person_id].copy()

                        if local_max_accuracy > max_accuracy:
                            max_accuracy = response['accuracy']
                            best_dataset.to_csv('csv/' + str(response['accuracy']) + '_new_data_incremental.csv')
                    print(' - Resposta do servidor:\n', response, '\n')
                else:
                    print(' - Nada a enviar\n')

        best_dataset.iloc[person_id] = best_row
        print('-' * 20, 'BEST ROW', '-' * 20)
        print('- BEST LOCAL ACCURACY:', local_max_accuracy)
        print(best_row[missing_pairs[person_id]])
        print('-' * 50, '\n\n')

    allow_invert_on_no_gaps_row = not modified_person


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

    got_error = False
    while True:
        # Enviando requisição e salvando o objeto resposta
        r = requests.post(url=URL, data=data)

        # Extraindo e imprimindo o texto da resposta
        res = json.loads(r.text)

        try:
            if res['accuracy'] is not None:
                break
        except:
            if not got_error:
                got_error = True
                print(' - Resposta do servidor ({}):\n'.format(datetime.datetime.now().time()), r.text, '\n')
            continue

    return res


missing_pairs = find_all_missing_pairs(missing_dataset)
while True:
    make_request()
