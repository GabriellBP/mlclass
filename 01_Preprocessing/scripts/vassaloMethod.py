import sys
import pandas as pd
import json
import math

DEBUG = 0

dataset = pd.read_csv('../diabetes_dataset.csv')

# ordena o dataset de forma que as linhas com menos lacunas fiquem no topo
ordered_data = dataset.iloc[dataset.isnull().sum(axis=1).argsort()]

with open('correlations_weights.json') as f:
    weights = json.load(f)


# columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
columns = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'Glucose', 'BMI', 'BloodPressure',
           'SkinThickness', 'Insulin']
ignored_cols = []


def find_best_neighbor(rank, id, key):
    minimum = sys.maxsize
    unique = True
    for i, v in enumerate(rank):
        if v <= minimum and i != id and not math.isnan(ordered_data.loc[i, key]):
            unique = True if v < minimum else False
            minimum = ordered_data.loc[i, key]

    return minimum, unique

count = 0
for index, row in ordered_data.iterrows():
    print(count)
    count += 1
    if DEBUG:
        print('-' * 20, index)

    for c in columns:
        if math.isnan(row[c]):
            if DEBUG:
                print('---------- Column: ', c)
            other_columns = columns.copy()
            other_columns.remove(c)

            ranking = [0 for i in range(ordered_data.shape[0])]
            for k in other_columns:
                if not math.isnan(row[k]):
                    if DEBUG:
                        print('- Neighbors: ', k)

                    df = ordered_data.copy()
                    df[k] = df[k] - row[k]
                    df[k] = df[k].abs()
                    neighbors = df.sort_values(k)

                    if DEBUG:
                        print(neighbors)

                    for idx, value in neighbors.iterrows():
                        # print(idx, value[k])
                        worst_neighbor = neighbors[k].max()
                        ranking[idx] += (1.5 if math.isnan(value[k]) else (value[k] / worst_neighbor)) * weights[k]

            value, is_unique = find_best_neighbor(ranking, index, c)
            ordered_data.loc[index, c] = value

    if DEBUG:
        print('\n')

for c in ignored_cols:
    ordered_data.drop(c, 1, inplace=True)

ordered_data.to_csv('vassalo_new_data.csv')
ordered_data.to_csv('../new_data.csv')
print('- NaN count:\n', ordered_data.isnull().sum(), sep='')

