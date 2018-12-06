import pandas as pd
import json
import operator
from collections import OrderedDict

DEBUG = 0

dataset = pd.read_csv('../diabetes_dataset.csv')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
ignored_cols = []
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

with open('correlations.json', 'w') as fp:
    json.dump(corr, fp)
