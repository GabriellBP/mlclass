# testing attributes for MLP
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]


def plot_on_dataset(X, y, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    X = MinMaxScaler().fit_transform(X)

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=400, **param)
        mlp.fit(X, y)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)


def transform_sex_column(df):
    for idx, cell in enumerate(df['sex']):
        if cell == 'I':
            df.at[idx, 'sex'] = 0
        elif cell == 'M':
            df.at[idx, 'sex'] = 1
        else:
            df.at[idx, 'sex'] = -1


# load dataset
data = pd.read_csv('abalone_dataset.csv')
feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
                'viscera_weight', 'shell_weight']

transform_sex_column(data)

X = data[feature_cols]
y = data.type

plot_on_dataset(X, y, name='abalone')
