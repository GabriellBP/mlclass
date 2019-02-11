import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

STEPS = 100
columns = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age', 'Glucose', 'BMI',
           'BloodPressure', 'SkinThickness', 'Insulin']
global best_w


def executeKNN(data, test):
    test_temp = test.drop('Outcome', axis=1)

    # Criando X and y para o algorítmo de aprendizagem de máquina.
    X = data[columns]
    y = data.Outcome

    # Ciando o modelo preditivo para a base trabalhada
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    # realizando previsões com o arquivo
    y_pred = neigh.predict(test_temp)
    # print(y_pred)
    # print(y)
    score = neigh.score(X, y)
    return score


def hillClimb(test, data):
    global best_w
    # definindo array de pesos inicial e matriz de pesos já testados
    w = np.random.randint(1, 10, len(columns))
    w_tested = [w]
    best_y = -1

    for i in range(0, STEPS):
        temp_data = data
        for idx, c in enumerate(columns):
            temp_data.loc[:, c] *= w[idx]
        y_pred = executeKNN(temp_data, test)
        if y_pred > best_y:
            best_y = y_pred
            print(best_y, ':', w)
            best_w = w
        # while w in w_tested:
        w = np.random.randint(1, 10, len(columns))
        w_tested.append(w)
    return best_y


def main():
    global best_w

    # recebendo csv de teste
    df_test = pd.read_csv('../diabetes_app.csv')
    df_test_names = df_test.columns

    # recebendo csv com maior porcentagem de acerto
    df = pd.read_csv('csv/0.83163265306122_new_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df_names = df.columns

    # normalizando dados dos csvs
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=df_names)
    np_scaled = min_max_scaler.fit_transform(df_test)
    df_test_normalized = pd.DataFrame(np_scaled, columns=df_test_names)

    # separando csv em csv de treino e de teste
    df_normalized['split'] = np.random.randn(df_normalized.shape[0], 1)
    msk = np.random.rand(len(df_normalized)) <= 0.7

    training = df_normalized[msk]
    training = training.drop('split', axis=1)
    test = df_normalized[~msk]
    test = test.drop('split', axis=1)

    df_normalized = df_normalized.drop('split', axis=1)
    # df_normalized.to_csv('gg.csv')
    # training.to_csv('training.csv', index=False)
    # test.to_csv('test.csv', index=False)

    # hill climb para definir os pesos das colunas
    best_y = hillClimb(test, training)
    for idx, c in enumerate(columns):
        df_normalized.loc[:, c] *= best_w[idx]
    df_normalized.to_csv('weighted_data.csv', index=False)
    df_test_normalized.to_csv('normalized_test.csv', index=False)


if __name__ == "__main__":
    main()





