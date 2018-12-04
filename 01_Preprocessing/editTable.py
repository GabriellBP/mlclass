# 72%: Média da tabela completa
# 73%: Selecionando um valor qualquer da própria coluna

import pandas as pd
import numpy as np
import math
from random import randrange

# dataapp = pd.read_csv("diabetes_app.csv")
dataapp = pd.read_csv("diabetes_dataset.csv")

# ----------------- Média geral -------------------
# glucose = dataapp.loc[:,'Glucose']
# bloodP = dataapp.loc[:,'BloodPressure']
# skinT = dataapp.loc[:,'SkinThickness']
# insulin = dataapp.loc[:,'Insulin']
# bmi = dataapp.loc[:,'BMI']
#
# glucoseMean = sum(glucose)/197
# bloodPMean = sum(bloodP)/197
# skinTMean = sum(skinT)/197
# insulinMean = sum(insulin)/197
# bmiMean = sum(bmi)/197
# -------------------------------------------------
glucose = [[], []]
bloodP = [[], []]
skinT = [[], []]
insulin = [[], []]
bmi = [[], []]

for index, row in dataapp.iterrows():
    if not math.isnan(row["Glucose"]):
        glucose[int(row["Outcome"])] += [row["Glucose"]]
    if not math.isnan(row["BloodPressure"]):
        bloodP[int(row["Outcome"])] += [row["BloodPressure"]]
    if not math.isnan(row["SkinThickness"]):
        skinT[int(row["Outcome"])] += [row["SkinThickness"]]
    if not math.isnan(row["Insulin"]):
        insulin[int(row["Outcome"])] += [row["Insulin"]]
    if not math.isnan(row["BMI"]):
        bmi[int(row["Outcome"])] += [row["BMI"]]

g = {}
b = {}
s = {}
ins = {}
bm = {}
for i in range(2):
    g[i] = sum(glucose[i]) / len(glucose[i])
    b[i] = sum(bloodP[i]) / len(bloodP[i])
    s[i] = sum(skinT[i]) / len(skinT[i])
    ins[i] = sum(insulin[i]) / len(insulin[i])
    bm[i] = sum(bmi[i]) / len(bmi[i])


dataset = pd.read_csv("diabetes_dataset.csv")
for index, row in dataset.iterrows():
    if math.isnan(row["Glucose"]):
        dataset.loc[index, 'Glucose'] = g[int(row["Outcome"])]
        # element = randrange(0, 571)
        # while math.isnan(dataset.loc[element, 'Glucose']):
        #     element = randrange(0, 571)
        # dataset.loc[index, 'Glucose'] = dataset.loc[element, 'Glucose']
    if math.isnan(row["BloodPressure"]):
        dataset.loc[index, 'BloodPressure'] = b[int(row["Outcome"])]
        # element = randrange(0, 571)
        # while math.isnan(dataset.loc[element, 'BloodPressure']):
        #     element = randrange(0, 571)
        # dataset.loc[index, 'BloodPressure'] = dataset.loc[element, 'BloodPressure']
    if math.isnan(row["SkinThickness"]):
        dataset.loc[index, 'SkinThickness'] = s[int(row["Outcome"])]
        # element = randrange(0, 571)
        # while math.isnan(dataset.loc[element, 'SkinThickness']):
        #     element = randrange(0, 571)
        # dataset.loc[index, 'SkinThickness'] = dataset.loc[element, 'SkinThickness']
    if math.isnan(row["Insulin"]):
        dataset.loc[index, 'Insulin'] = ins[int(row["Outcome"])]
        # element = randrange(0, 571)
        # while math.isnan(dataset.loc[element, 'Insulin']):
        #     element = randrange(0, 571)
        # dataset.loc[index, 'Insulin'] = dataset.loc[element, 'Insulin']
    if math.isnan(row["BMI"]):
        dataset.loc[index, 'BMI'] = bm[int(row["Outcome"])]
        # element = randrange(0, 571)
        # while math.isnan(dataset.loc[element, 'BMI']):
        #     element = randrange(0, 571)
        # dataset.loc[index, 'BMI'] = dataset.loc[element, 'BMI']

# for index, row in dataset.iterrows():
#     print(row["Glucose"], row["Insulin"])

# print(dataset.loc[1,'Insulin'])
# dataset.to_csv("new_data.csv")

# preg.replace('', 20182018)
# preg.to_csv("new_data.csv")

# print(dataset.to_string())
dataset.to_csv("new_data.csv")

print(dataset.isnull().sum())
