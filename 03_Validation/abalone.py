import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.server.send_server import send_2_server


def split_data(X, y, test_size, random_state):
    """
        Can split our data into training and test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def generate_accuracy_score(y_test, y_pred):
    """
        Calculate the accuracy score between the predicted y, and the test y
    """
    return accuracy_score(y_test, y_pred) * 100


def generate_cross_val_score(clf, data, target, cv):
    """
        Can we use here, the mean of the predictions to calculate the full accuracy
    """
    return cross_val_score(clf, data, target, cv=cv)


def get_classifier(option=1):
    if option == 1:  # local score: 0.6046973683089146
        # KNN classifier with n=3
        return KNeighborsClassifier(n_neighbors=3)
    elif option == 2:  # local score: 0.6497192013368424
        # KNN classifier with n=19
        return KNeighborsClassifier(n_neighbors=19)
    elif option == 3:  # local score: 0.5587182164432322
        # Tree classifier
        return DecisionTreeClassifier()
    elif option == 4:  # local score: 0.66091053909337
        # MLP classifier with adam params
        return MLPClassifier(verbose=0, random_state=0,
                             max_iter=400, solver='adam', learning_rate_init=0.01)
    elif option == 5:  # local score: 0.6251458914579832
        # MLP classifier with constant with Nesterov's momentum params
        return MLPClassifier(verbose=0, random_state=0,
                             max_iter=400, solver='sgd', learning_rate='constant', momentum=.9,
                             nesterovs_momentum=True, learning_rate_init=0.2)
    elif option == 6:  # local score: variable
        # MLP classifier with default params
        return MLPClassifier(max_iter=400)
    elif option == 7:  # local score: 0.6672972302744794 (X-validation=10); 0.6631670790979023 (X-validation=20)
        # MLP classifier with custom params
        return MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                             beta_1=0.9, beta_2=0.999, early_stopping=False,
                             epsilon=1e-08, hidden_layer_sizes=(15,),
                             learning_rate='constant', learning_rate_init=0.01,
                             max_iter=200, momentum=0.9, n_iter_no_change=10,
                             nesterovs_momentum=True, power_t=0.5, random_state=1,
                             shuffle=True, solver='lbfgs', tol=0.0001,
                             validation_fraction=0.1, verbose=False, warm_start=False)
    elif option == 8:  # local scores:
        return RandomForestClassifier(n_estimators=700, oob_score=True, random_state=73, min_samples_leaf=10, max_features=6)
    elif option == 9:  # local scores: 0.6561099300761193 (X-validation=10); 0.6564381674582185 (X-validation=20)
        return RandomForestClassifier(n_estimators=700, oob_score=True, random_state=101, min_samples_leaf=15)
    elif option == 10:  # local score: 0.6538877792887174 (X-validation=10)
        return RandomForestClassifier(n_estimators=700, oob_score=True, n_jobs=1, random_state=101, min_samples_leaf=30)
    elif option == 11:
        # 0.6555023923444976 (servidor)
        # 0.7681992337164751 (proprio)
        # 0.654523717466067 (X-validation=10)
        # 0.6580851337830741 (X-validation=15)
        # 0.6573976649369829 (X-validation=20)
        return RandomForestClassifier(n_estimators=700, oob_score=True, random_state=101, min_samples_leaf=10)


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
    data = pd.read_csv('abalone_dataset.csv')

    transform_sex_column(data)

    print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo abalone_dataset')
    feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
                    'viscera_weight', 'shell_weight']
    X = data[feature_cols]
    y = data.type

    # Ciando o modelo preditivo para a base trabalhada
    print(' - Criando modelo preditivo')
    classifier = get_classifier(11)
    classifier.fit(X, y)
    print('score: {}'.format(classifier.score(X, y)))

    # Cross Validation score
    score = np.mean(generate_cross_val_score(classifier, X, y, 15))
    print('local X-val score: {}'.format(score))

    # Realizando previsões com o arquivo abalone_app.csv
    print(' - Aplicando modelo')
    data_app = pd.read_csv('abalone_app.csv')
    transform_sex_column(data_app)
    y_pred = classifier.predict(data_app)

    # sending to the server
    send_2_server(y_pred)


if __name__ == "__main__":
    main()
