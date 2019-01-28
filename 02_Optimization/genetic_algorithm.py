import copy
import datetime
import random
import requests

url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'
turns = []
index = {'phi1': 0, 'theta1': 1, 'phi2': 2, 'theta2': 3, 'phi3': 4, 'theta3': 5}

intervals = {
    'phi1': (0, 360),
    'theta1': (0, 360),
    'phi2': (0, 360),
    'theta2': (0, 360),
    'phi3': (0, 360),
    'theta3': (0, 360)
}
best_result = -1000


def generate_turns(i, a, total):
    if i == 0:
        total += [a]
        return
    for j in range(2):
        generate_turns(i - 1, a + [j], total)


generate_turns(6, [], turns)
no_improvement_count = 0
while True:
    local_intervals = copy.deepcopy(intervals)
    improved = False

    for t in turns:
        values = {}
        curr_intervals = {}
        for k, pair in intervals.items():
            average = int((pair[0] + pair[1]) / 2)

            if t[index[k]] == 0:
                values[k] = random.randint(pair[0], int((average + pair[1]) / 2))
                curr_intervals[k] = (pair[0], int((average + pair[1]) / 2))
            else:
                values[k] = random.randint(int((pair[0] + average) / 2), pair[1])
                curr_intervals[k] = (int((pair[0] + average) / 2), pair[1])

        r = requests.get(url.format(values['phi1'], values['theta1'], values['phi2'], values['theta2'],
                                    values['phi3'], values['theta3']))
        result = float(r.text.split('\n')[0])

        if result > best_result:
            improved = True
            no_improvement_count = 0
            best_result = result
            local_intervals = copy.deepcopy(curr_intervals)

            print('-'*30)
            print('NEW Best Result:', best_result)
            print('NEW Intervals:', local_intervals)
            print('Values:', values)
            print('At:', datetime.datetime.now().time())
            print()

    intervals = local_intervals

    if not improved:
        no_improvement_count += 1

        if no_improvement_count == 100:
            print('\n\n\nRESETTING....\n\n\n')
            intervals = {
                'phi1': (0, 360),
                'theta1': (0, 360),
                'phi2': (0, 360),
                'theta2': (0, 360),
                'phi3': (0, 360),
                'theta3': (0, 360)
            }
