import numpy as np
import pandas as pd
import requests
import datetime

url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'
index = ['phi1', 'theta1', 'phi2', 'theta2', 'phi3', 'theta3']

# generating a initial set of values
values = pd.Series(np.random.randint(0, 360, 6), index=index)

# generating the initial result
r = requests.get(url.format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
                            values['phi3'], values['theta3']))
best_result = float(r.text.split('\n')[0])

print('-' * 30)
print('NEW Best Result:', best_result)
print("Values: ['phi1': {}, 'theta1': {}, 'phi2': {}, 'theta2': {}, 'phi3': {}, 'theta3': {}]"
      .format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
              values['phi3'], values['theta3']))
print('At:', datetime.datetime.now().time())
print()

# variation
variation = 5

# executing hill climb
tries = 0
while True:
    tries += 1
    for t in index:
        new_values = values.copy()
        interval_i = values[t] - variation if values[t] - variation >= 0 else values[t]
        interval_f = values[t] + variation if values[t] + variation <= 360 else values[t]
        new_values[t] = np.random.randint(interval_i, interval_f)
        try:
            r = requests.get(
                url.format(new_values['phi1'], new_values['theta1'], new_values['phi2'], new_values['theta2'],
                           new_values['phi3'], new_values['theta3']))
            result = float(r.text.split('\n')[0])
        except:
            continue
        if result > best_result:
            tries = 0
            best_result = result
            values = new_values.copy()
            print('-' * 30)
            print('NEW Best Result:', best_result)
            print("Values: ['phi1': {}, 'theta1': {}, 'phi2': {}, 'theta2': {}, 'phi3': {}, 'theta3': {}]"
                  .format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
                          values['phi3'], values['theta3']))
            print('At:', datetime.datetime.now().time())
            print()
    if tries == 100:
        print('jumping')
        # avoiding local hills
        tries = 0
        values = pd.Series(np.random.randint(0, 360, 6), index=index)
        r = requests.get(url.format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
                                    values['phi3'], values['theta3']))
        result = float(r.text.split('\n')[0])