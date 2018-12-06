import sys

import pandas as pd
import numpy as np
import math

DEBUG = 0

dataset = pd.read_csv('../diabetes_dataset.csv')

# ordena o dataset de forma que as linhas com menos lacunas fiquem no topo
ordered_data = dataset.iloc[dataset.isnull().sum(axis=1).argsort()]

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def find_best_neighbor(rank, id, key):
    minimum = rank[0]
    unique = True
    for i, v in enumerate(rank):
        if v <= minimum and i != id and not math.isnan(ordered_data.loc[i, key]):
            unique = True if v < minimum else False
            minimum = v

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
                        ranking[idx] += row[k] if math.isnan(value[k]) else value[k]

            value, is_unique = find_best_neighbor(ranking, index, c)
            ordered_data.loc[index, c] = value

    if DEBUG:
        print('\n')


    # if math.isnan(row['Glucose']):
    #     dataset.loc[index, 'Glucose'] = randrange(int(dataapp['Glucose'].min()), int(dataapp['Glucose'].max()))
    # if math.isnan(row['BloodPressure']):
    #     dataset.loc[index, 'BloodPressure'] = randrange(int(dataapp['BloodPressure'].min()), int(dataapp['BloodPressure'].max()))
    # if math.isnan(row['SkinThickness']):
    #     dataset.loc[index, 'SkinThickness'] = randrange(int(dataapp['SkinThickness'].min()), int(dataapp['SkinThickness'].max()))
    # if math.isnan(row['Insulin']):
    #     dataset.loc[index, 'Insulin'] = randrange(int(dataapp['Insulin'].min()), int(dataapp['Insulin'].max()))
    # if math.isnan(row['BMI']):
    #     dataset.loc[index, 'BMI'] = randrange(int(dataapp['BMI'].min()), int(dataapp['BMI'].max()))

ordered_data.to_csv('vassalo_new_data.csv')
print("NaN count: ", ordered_data.isnull().sum())

