import pandas as pd
from deap import base, creator, tools, algorithms
import random
import numpy as np


excel_file_path = 'data.xlsx'
dataset = pd.read_excel(excel_file_path)


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def create_individual():
    individual_data = dataset.sample(n=1).iloc[0].tolist()
    return creator.Individual(individual_data)


def init_population():
    population = [create_individual() for _ in range(20)]
    return population

def eval_patient(individual):
    score = 0

    fractal_dimension_worst_idx = dataset.columns.get_loc('fractal_dimension_worst')
    symmetry_worst_idx = dataset.columns.get_loc('symmetry_worst')
    concave_points_worst_idx = dataset.columns.get_loc('concave_points_worst')
    concavity_worst_idx = dataset.columns.get_loc('concavity_worst')
    compactness_worst_idx = dataset.columns.get_loc('compactness_worst')
  
    if float(individual[fractal_dimension_worst_idx]) > 0.1 :
        score += 40
    else:
        score -=5
    if float(individual[symmetry_worst_idx]) > 0.4:
        score += 20
    else:
        score -=5
    if float(individual[concave_points_worst_idx]) < 0.2:
        score += 20
    else:
        score -=5
    if float(individual[concavity_worst_idx]) <0.6 :
        score += 40
    else:
        score -=5
    if float(individual[compactness_worst_idx]) < 0.6:
        score +=40
    else:
        score -= 5
 
    return score,


def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if isinstance(individual[i], (float, np.float64)):
            if random.random() < indpb:
                individual[i] += random.uniform(-1.0, 1.0)
        elif isinstance(individual[i], int):
            if random.random() < indpb:
                individual[i] += random.randint(-1, 1)
    return (individual,)  
toolbox.register('evaluate', eval_patient)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', custom_mutate, indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)

toolbox.register('individual', tools.initIterate, creator.Individual, create_individual)
toolbox.register('population', init_population)


def main():
    pop = toolbox.population()
    hof = tools.HallOfFame(20)
    for gen in range(20):
        offspring = algorithms.varOr(pop, toolbox, lambda_=20, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, len(pop))
        hof.update(pop)

    
    discharge_threshold = 0
    for i, patient in enumerate(hof):
        fitness = patient.fitness.values[0]
        if fitness <= discharge_threshold:
            print(f"Patient {i+1} with fitness {fitness}: this patient can be discharged")
        else:
            print(f"Patient {i+1} with fitness {fitness}: this patient cannot be discharged")

    return pop, hof

if __name__ == '__main__':
    main()