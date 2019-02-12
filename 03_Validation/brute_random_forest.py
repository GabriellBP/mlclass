import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']


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
    best_score = 0
    best_params = {}

    print('\n - Lendo o arquivo com o dataset sobre abalone')
    # training
    data = pd.read_csv('abalone_dataset.csv')
    transform_sex_column(data)
    # evaluating
    data_app = pd.read_csv('abalone_app.csv')[feature_cols]
    transform_sex_column(data_app)

    print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo abalone_dataset')
    X = data[feature_cols]
    y = data.type

    # Ciando o modelo preditivo para a base trabalhada
    for oob_score in range(2):
        for min_samples_leaf in range(1, 50):
            print(' - Criando modelo preditivo')
            print({'oob_score': oob_score, 'min_samples_leaf': min_samples_leaf, 'n_estimators': 700,
                   'random_state': 101})
            classifier = RandomForestClassifier(oob_score=True if oob_score == 1 else False,
                                                min_samples_leaf=min_samples_leaf,
                                                n_estimators=700, random_state=101)
            classifier.fit(X, y)
            print('score: {}'.format(classifier.score(X, y)))

            # Cross Validation score
            score10 = np.mean(generate_cross_val_score(classifier, X, y, 10))
            score20 = np.mean(generate_cross_val_score(classifier, X, y, 20))

            score = (score10 + score20) / 2
            print('X-val score: {}'.format(score))
            print()
            if score > best_score:
                best_score = score
                best_params = {'oob_score': oob_score, 'min_samples_leaf': min_samples_leaf, 'n_estimators': 700,
                               'random_state': 101}

    print('best params:{}'.format(best_params))
    print()


if __name__ == "__main__":
    main()
