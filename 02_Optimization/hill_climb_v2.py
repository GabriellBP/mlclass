import numpy as np
import pandas as pd
import requests
import datetime


def make_request(val):
    # generating the initial result
    try:
        req = requests.get(url.format(val['phi1'], val['theta1'], val['phi2'], val['theta2'],
                                      val['phi3'], val['theta3']))
        return float(req.text.split('\n')[0])
    except:
        make_request(val)


def print_result(val, best):
    print('-' * 30)
    print('NEW Best Result:', best)
    print("Values: ['phi1': {}, 'theta1': {}, 'phi2': {}, 'theta2': {}, 'phi3': {}, 'theta3': {}]"
          .format(val['phi1'], val['theta1'], val['phi2'], val['theta2'],
                  val['phi3'], val['theta3']))
    print('At:', datetime.datetime.now().time())
    print()


url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'
index = ['phi1', 'theta1', 'phi2', 'theta2', 'phi3', 'theta3']

# generating a initial set of values
values = pd.Series(np.random.randint(0, 360, 6), index=index)
best_values = values.copy()
best_result = make_request(values)

print_result(values, best_result)

# executing hill climb
tries = 0
while True:
    for t in index:
        memory = values.copy()
        values[t] += 1
        temp_result = make_request(values)
        while temp_result >= best_result and values[t] <= 360:
            print_result(values, best_result)
            values[t] += 1
            best_result = temp_result
            temp_result = make_request(values)
        best_values = values.copy()
        values = memory
        values[t] -= 1
        temp_result = make_request(values)
        while temp_result >= best_result and values[t] >= 0:
            print_result(values, best_result)
            values[t] -= 1
            best_result = temp_result
            temp_result = make_request(values)
        best_values = values.copy()
    print('jumping')
    # avoiding local hills
    temp_values = pd.Series(np.random.randint(0, 360, 6), index=index)
    temp_result = make_request(values)
    tries = 0
    while best_result > temp_result and tries<1000:
        tries += 1
        temp_values = pd.Series(np.random.randint(0, 360, 6), index=index)
        temp_result = make_request(values)
    if temp_result > best_result:
        values = temp_values
        best_result = temp_result