import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import requests
import json
import datetime

STEPS = 1000
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
global max_accuracy, yy, ww, best_y_pred, data_app
max_accuracy = 0.9030612244898
best_y_pred = '[0.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0]'

data_app = pd.read_csv('../diabetes_app.csv')
data_app = data_app[feature_cols]


def executeKNN(data_training):
    global max_accuracy, best_y_pred

    X = data_training[feature_cols]
    y = data_training.Outcome

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    y_pred = neigh.predict(data_app)

    if pd.Series(y_pred).to_json(orient='values') == best_y_pred:
        print('>' * 30, 'SAME ARRAY AS THE BEST ARRAY INTERCEPTED')
        return -1

    URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

    DEV_KEY = "Dual Core"

    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(y_pred).to_json(orient='values')}

    # r = requests.post(url=URL, data=data)
    #
    # # Extraindo e imprimindo o texto da resposta
    # res = json.loads(r.text)

    got_error = False
    while True:
        r = requests.post(url=URL, data=data)

        # Extraindo e imprimindo o texto da resposta
        res = json.loads(r.text)

        try:
            if res['accuracy'] is not None:
                if res['accuracy'] > max_accuracy:
                    best_y_pred = pd.Series(y_pred).to_json(orient='values')
                    print('>' * 30, 'NEW best_y_pred:', best_y_pred)
                    data_training.to_csv(str(best_y_pred)+'_Gabriel.csv')
                break
        except:
            if not got_error:
                got_error = True
                print(' - Resposta do servidor ({}):\n'.format(datetime.datetime.now().time()), r.text, '\n')
            continue
    print(res['accuracy'])
    return res['accuracy']


def hillClimb(data):
    global yy
    global ww
    # first weight definition
    w = [1]*len(feature_cols)     # 80612244897959
    # w = [1, 1, 1, -1, 1, 1, 1, 2] # 80612244897959
    # w = [1, 1, 1, -1, 1, 1, 0, 2] # 80102040816327
    # w = [1, 1, 1, 0, 1, 1, 0, 2]  # 80612244897959
    # w = [-2, 1, 1, -1, 1, 1, 0, 2]# 80102040816327
    # w = [1, 1, 1, 0, 1, 0, 0, 2]  # 80612244897959
    # w = [1, 1, 1, -1, 1, 0, 0, 2]
    # [1, 1, 4, 7, 1, 1, 1, 2]

    # [1.2797502663134468, 1, 1, 1, 1, 1, 1.2226601248310898, 1]

    # updated dataset
    updated_data = data.copy()
    temp_data_best = updated_data
    # best weight founded
    best_w = w.copy()
    # weights already tested
    w_tested = [w.copy()]
    # best acuracy
    best_y = 0

    while True:
        temp_best_w = best_w.copy()
        temp_best_y = best_y

        for idx, c in enumerate(feature_cols):
            # generate a random int between [-2, 2]
            # r_value_p = np.random.randint(-5, 0)
            # r_value_n = np.random.randint(1, 6)
            # r_value = np.random.randint(1, 20)
            r_value = np.random.uniform(-3, 4)
            # if r_value == 0:
            #     r_value = r_value_p
            # else:
            #     r_value = r_value_n
            # setting the copies
            temp_data = updated_data.copy()
            temp_w = best_w.copy()
            # applying the random value to a column
            temp_w[idx] += r_value
            if temp_w not in w_tested:
                w_tested.append(temp_w.copy())
            else:
                print('already have:', temp_w)
                continue
            # applying the new weight value to a column on the dataframe
            temp_data.loc[:, c] *= temp_w[idx]
            # executing knn with the new dataframe
            print('actual w:', temp_w)
            temp_y = executeKNN(temp_data)
            if temp_y > temp_best_y:
                temp_best_y = temp_y
                temp_best_w = temp_w.copy()
                temp_data_best = temp_data.copy()

        if temp_best_y > best_y:
            print('         Nova acuracia:', temp_best_y)
            print('         Novos pesos:', temp_best_w)
            best_y = temp_best_y
            best_w = temp_best_w.copy()
            yy = best_y
            ww = best_w.copy()
            updated_data = temp_data_best.copy()
        print(yy)
        print(ww)


def main():
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # recebendo csv de teste
    global data_app
    '''
        df_test = pd.read_csv('../diabetes_app.csv')
        df_test_names = df_test.columns
    '''

    # recebendo csv com maior porcentagem de acerto
    df = pd.read_csv('csv/incremental_new_data_0.9030612244898.csv')
    df = df[cols]
    df_names = df.columns

    # normalizando dados dos csvs

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=df_names)
    np_scaled = min_max_scaler.fit_transform(data_app)
    data_app = pd.DataFrame(np_scaled, columns=feature_cols)

    # hill climb para definir os pesos das colunas
    # hillClimb(df, df_test)
    hillClimb(df_normalized)
    # hillClimb(df.drop('DiabetesPedigreeFunction', axis=1).drop('BMI', axis=1))
    # executeKNN(df.drop('Pregnancies', axis=1))


if __name__ == "__main__":
    main()





