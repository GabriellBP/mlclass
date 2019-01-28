import requests

url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'
best_result = -1000
best_angles = {}

for phi1 in range(0, 360):
    for theta1 in range(0, 360):
        for phi2 in range(0, 360):
            for theta2 in range(0, 360):
                for phi3 in range(0, 360):
                    for theta3 in range(0, 360):
                        r = requests.get(url.format(phi1, theta1, phi2, theta2, phi3, theta3))
                        result = float(r.text.split('\n')[0])
                        
                        if result > best_result:
                            best_result = result
                            
                            best_angles['phi1'] = phi1
                            best_angles['theta1'] = theta1
                            best_angles['phi2'] = phi2
                            best_angles['theta2'] = theta2
                            best_angles['phi3'] = phi3
                            best_angles['theta3'] = theta3
                            
                            print('-'*30)
                            print('NEW Best Result:', best_result)
                            print('NEW Best Angles:', best_angles)
                            print()
