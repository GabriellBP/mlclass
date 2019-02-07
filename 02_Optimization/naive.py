import datetime
import json

import requests

# url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'
url = 'https://aydanomachado.com/mlclass/02_Optimization.php?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}&dev_key=Dual Core'
angles = ['phi1', 'theta1', 'phi2', 'theta2', 'phi3', 'theta3']

values = {'phi1': 10, 'theta1': 180, 'phi2': 359, 'theta2': 60, 'phi3': 180, 'theta3': 205}
best_result = 29.3878848771

while True:
    improved = False

    for k in angles:
        print('Current Angle:', k)
        print('At:', datetime.datetime.now().time())
        print()
        best_angle = values[k]

        for i in range(360):
            values[k] = i

            while True:
                try:
                    r = requests.get(url.format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
                                                values['phi3'], values['theta3']))
                    result = float(json.loads(r.text)['gain'])
                    break
                except:
                    print('-'*15, 'DEU PAU NA REQUISIÇÃO! At {}'.format(datetime.datetime.now().time()))
                    continue
            print(result)

            if result > best_result:
                best_result = result
                best_angle = i
                improved = True

                print('-'*30)
                print('NEW Best Result:', best_result)
                print('Values:', values)
                print('At:', datetime.datetime.now().time())
                print()

        values[k] = best_angle

    if not improved:
        print('FIM!')
        break
