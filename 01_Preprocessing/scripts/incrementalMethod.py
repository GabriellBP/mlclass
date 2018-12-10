import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import math
from random import randrange
import os
import datetime

global max_accuracy, method_max_accuracy, can_skip_person, start_at, start_at_turn, start_with_impact
max_accuracy = 0.86734693877551
method_max_accuracy = 0.86734693877551
can_skip_person = True
start_at = 347
start_at_turn = 0
start_with_impact = False

n = 3
data_app = pd.read_csv('../diabetes_app.csv')
best_dataset = pd.read_csv('csv/incremental_new_data_{}.csv'.format(method_max_accuracy))
missing_dataset = pd.read_csv('csv/incremental_missing_diabetes_dataset.csv')
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
    global max_accuracy, method_max_accuracy, can_skip_person, start_at, start_at_turn, start_with_impact

    missing_pairs = find_all_missing_pairs(missing_dataset)
    count = 0
    had_any_impact = start_with_impact
    for person_id in missing_pairs:
        count += 1
        if count < start_at:
            continue

        start_at = 0
        turns = []
        generate_turns(len(missing_pairs[person_id]), [], turns)
        row_has_gaps = has_gaps(best_dataset.iloc[person_id], missing_pairs[person_id])

        print('#' * 15, 'NEXT PERSON [ID: {}, {}/{}], {}GAPS FOUND'.format(person_id, count, len(missing_pairs),
                                                                           '' if row_has_gaps else 'NO '),
              '#' * 15, missing_pairs[person_id])
        print('can skip "meaningless" person:', can_skip_person)
        print('turns:', turns, '\n')

        best_row = best_dataset.iloc[person_id].copy()

        if row_has_gaps:
            had_impact = start_with_impact
            negative_impacts = 0
            attempted = 0

            for t, turn in enumerate(turns):
                attempted += 1

                if t < start_at_turn:
                    continue

                start_at_turn = 0
                start_with_impact = False
                print('selected turn:', turn)
                for index, col in enumerate(missing_pairs[person_id]):
                    mean = missing_dataset[col].mean()
                    sd = missing_dataset[col].std()

                    below = randrange(int(mean - n * sd), int(mean))
                    if below < 0:
                        below = 0
                    above = randrange(int(mean), int(mean + n * sd))
                    if turn[index] == 0:
                        if len(turns) == 2 and below == 0:
                            best_dataset.loc[person_id, col] = above
                        else:
                            best_dataset.loc[person_id, col] = below
                    else:
                        best_dataset.loc[person_id, col] = above

                print('CURRENT BEST ROW:\n' + str(best_row[missing_pairs[person_id]]) + '\n--------\nCURRENT ROW:\n'
                      + str(best_dataset.iloc[person_id][missing_pairs[person_id]]) + '\n')

                response = send_request(best_dataset)
                if response['accuracy'] != method_max_accuracy:
                    had_impact = True
                    had_any_impact = True

                    if response['accuracy'] < method_max_accuracy:
                        negative_impacts += 1
                    elif response['accuracy'] > method_max_accuracy:
                        if os.path.exists('csv/incremental_new_data_' + str(method_max_accuracy) + '.csv'):
                            os.remove('csv/incremental_new_data_' + str(method_max_accuracy) + '.csv')

                        method_max_accuracy = response['accuracy']
                        best_dataset.to_csv('csv/incremental_new_data_' + str(method_max_accuracy) + '.csv')
                        best_row = best_dataset.iloc[person_id].copy()

                        if method_max_accuracy > max_accuracy:
                            max_accuracy = method_max_accuracy
                            best_dataset.to_csv('csv/' + str(max_accuracy) + '_new_data_incremental.csv')
                print(' - Resposta do servidor:\n', response, '\n')

                if not had_impact and attempted == len(turns) / 2 and can_skip_person:
                    print('>' * 30, 'SKIPPING MEANINGLESS PERSON!!!\n')
                    break

            if negative_impacts == len(turns):
                print('>' * 30, 'UPDATING MISSING DATA SET!!! THIS PERSON\'S GAPS MAY BE ZEROS')
                missing_dataset.iloc[person_id] = best_row
                missing_dataset.to_csv('csv/incremental_missing_diabetes_dataset.csv')
        else:   # da forma atual, provavelmente nunca vai entrar aqui
            print(' - Nada a enviar\n')

        best_dataset.iloc[person_id] = best_row
        print('-' * 20, 'BEST ROW', '-' * 20)
        print('- BEST LOCAL ACCURACY:', method_max_accuracy)
        print(best_row[missing_pairs[person_id]])
        print('-' * 50, '\n\n')

    can_skip_person = had_any_impact


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


while True:
    make_request()
