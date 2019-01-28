import copy
import datetime
import random
import requests

url = 'http://localhost:8080/antenna/simulate?phi1={}&theta1={}&phi2={}&theta2={}&phi3={}&theta3={}'

global best_result

initial_population_size = 1000
max_population_size = 50000
max_generations = 1000
crossing_chance = 100
mutation_chance = 100


class Chromosome:
    def __init__(self, angles, score=None):
        self.angles = angles
        self.score = score
        if score is None:
            self.score = evaluate(self)

    def __str__(self):
        return "Angles: {}; Score: {}".format(str(self.angles), self.score)


def evaluate(chromosome):
    global best_result

    while True:
        try:
            r = requests.get(url.format(chromosome.angles[0], chromosome.angles[1], chromosome.angles[2], chromosome.angles[3],
                                        chromosome.angles[4], chromosome.angles[5]))
            break
        except:
            print('DEU RUIM!')
            continue
    result = float(r.text.split('\n')[0])

    if chromosome.score is None:
        chromosome.score = result

    if result > best_result:
        best_result = result
        print('-'*30)
        print('NEW Best:', chromosome)
        print('At:', datetime.datetime.now().time())
        print()

    return result


def crossover(population):
    descendants = []
    for _ in range(100):
        top_individual = population[random.randint(0, int(len(population) * 0.20))]
        reg_individual = population[random.randint(int(len(population) * 0.20), int(len(population) * 0.6))]

        for i in range(8):
            cp = random.randint(0, 6)
            child = top_individual.angles[:cp] + reg_individual.angles[cp:]
            assert len(child) == 6
            descendants.append(Chromosome(child))
    return descendants


def mutate(population):
    mutants = copy.deepcopy(population)
    for i in range(len(mutants)):
        mp = random.randint(0, 6)
        mutants[i].angles[mp] += random.randint(0, 15) if random.randint(0, 2) == 1 else -random.randint(0, 15)
    return mutants


def main():
    global best_result
    best_result = -1000

    # generating initial population
    population = []
    for _ in range(initial_population_size):
        population.append(Chromosome([random.randint(0, 360), random.randint(0, 360), random.randint(0, 360),
                                      random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)]))

    curr_gen = 0
    while True:
        curr_gen += 1
        if curr_gen > max_generations:
            print('THE END!')
            break

        # selection
        population.sort(key=lambda c: c.score, reverse=True)

        if len(population) > max_population_size:
            population = population[:max_population_size]

        # crossing
        descendants = crossover(population[:int(len(population) * 0.6)])

        # mutation
        mutants = mutate(descendants)

        population = population + descendants + mutants


if __name__ == '__main__':
    main()
