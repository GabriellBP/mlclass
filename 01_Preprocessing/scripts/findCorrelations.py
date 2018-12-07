import pandas as pd
import json
import operator
from collections import OrderedDict

DEBUG = 0

dataset = pd.read_csv('../diabetes_dataset.csv')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
ignored_cols = []
corr = {}
weights = {}

for a in feature_cols:
    for b in feature_cols:
        if a != b:
            corr[a + ' x ' + b] = dataset[a].corr(dataset[b])

            try:
                weights[a] += int(corr[a + ' x ' + b] * 10)
            except:
                weights[a] = int(corr[a + ' x ' + b] * 10)

with open('correlations_all.json', 'w') as fp:
    json.dump(corr, fp)

with open('correlations_weights.json', 'w') as fp:
    json.dump(weights, fp)

corr = {}
for a in feature_cols:
    for b in feature_cols:
        if a != b:
            try:
                if corr[b + ' x ' + a] is None:
                    continue
            except:
                corr[a + ' x ' + b] = dataset[a].corr(dataset[b])

corr = OrderedDict(sorted(corr.items(), key=operator.itemgetter(1), reverse=True))

with open('correlations_filtered.json', 'w') as fp:
    json.dump(corr, fp)
