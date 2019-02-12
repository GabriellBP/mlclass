from time import sleep

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from scripts.server.send_server import send_2_server


def generate_cross_val_score(clf, data, target, cv):
    """
        Can we use here, the mean of the predictions to calculate the full accuracy
    """
    return cross_val_score(clf, data, target, cv=cv)


def transform_sex_column(df):
    for idx, cell in enumerate(df['sex']):
        if cell == 'I':
            df.at[idx, 'sex'] = 0
        elif cell == 'M':
            df.at[idx, 'sex'] = 1
        else:
            df.at[idx, 'sex'] = -1


def main():
    print('\n - Lendo o arquivo com o dataset sobre abalone')
    # training
    data = pd.read_csv('abalone_dataset.csv')
    transform_sex_column(data)
    # evaluating
    data_app = pd.read_csv('abalone_app.csv')
    transform_sex_column(data_app)

    print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo abalone_dataset')
    feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
                    'viscera_weight', 'shell_weight']
    X = data[feature_cols]
    y = data.type

    # Ciando o modelo preditivo para a base trabalhada
    print(' - Criando modelo preditivo')
    classifier = RandomForestClassifier(oob_score=False,
                                        min_samples_leaf=9,
                                        n_estimators=700, random_state=101)
    classifier.fit(X, y)

    y_pred = classifier.predict(data_app)

    while True:
        try:
            print('server score: {}'.format(send_2_server(y_pred)))
            break
        except:
            sleep(600)
            continue


if __name__ == "__main__":
    main()
