import copy
import torch
import random

def fed_avg(local_weights):
    avg = copy.deepcopy(local_weights[0])
    for key in avg:
        for i in range(1, len(local_weights)):
            avg[key] += local_weights[i][key]
        avg[key] /= len(local_weights)
    return avg

def black_widow_optimization(pop, fitness_func, generations, mutation_rate, procreation_ratio, cannibalism_ratio, **kwargs):
    population = [copy.deepcopy(p) for p in pop]
    num_parents = len(population)
    fitness_log = []
    
    if num_parents <= 2:
        return population[0], [fitness_func(population[0])]
    else:
        for _ in range(generations):
            # Step 1: Procreation
            children = []
            for _ in range(int(num_parents * procreation_ratio)):
                p1, p2 = random.sample(population, 2)
                child = {k: (p1[k] + p2[k]) / 2 for k in p1}
                children.append(child)
            population += children

            # Step 2: Fitness & Cannibalism
            scores = [fitness_func(p) for p in population]
            survivors = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            survivors = survivors[:int(len(population) * (1 - cannibalism_ratio))]
            population = [p for _, p in survivors]

            # Step 3: Mutation
            for p in population:
                if random.random() < mutation_rate:
                    for key in p:
                        p[key] += torch.randn_like(p[key]) * 0.01

            # Step 4: Update Fitness
            scores = [fitness_func(p) for p in population]
            fitness_log.append(max(scores))

        return population[0], fitness_log
