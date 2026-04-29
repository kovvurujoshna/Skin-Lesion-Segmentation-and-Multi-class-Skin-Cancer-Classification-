import time

import numpy as np


def perform_selection(population, fitness_values):
    # Perform selection based on fitness values
    # Your selection logic goes here
    # Example: Select the best individuals based on their fitness values
    sorted_indices = np.argsort(fitness_values)
    selected_population = population[sorted_indices]
    return selected_population


def perform_mutation(population, mutation_rate):
    # Perform mutation on the population
    # Your mutation logic goes here
    # Example: Add random noise to each individual
    mutated_population = population + mutation_rate * np.random.randn(*population.shape)
    return mutated_population


def perform_crossover(parents, crossover_rate):
    # Perform crossover between the parents
    # Your crossover logic goes here
    # Example: Blend the parents' values to create offspring
    num_parents = len(parents)
    num_variables = parents.shape[1]
    offspring = np.zeros((num_parents, num_variables))
    for i in range(num_parents):
        for j in range(num_variables):
            if np.random.rand() < crossover_rate:
                offspring[i, j] = np.mean(parents[:, j])
            else:
                offspring[i, j] = parents[i, j]
    return offspring


def MAO(pop_size, evaluate_fitness, lb, ub, max_iterations):
    mutation_rate = 0.1
    crossover_rate = 0.8
    population, num_variables = pop_size.shape
    best_solution = None
    best_fitness = float('inf')
    ct = time.time()
    Convergence_curve = np.zeros((max_iterations, 1))

    for Iter in range(max_iterations):
        fitness_values = evaluate_fitness(population)
        best_index = np.argmin(fitness_values)
        if fitness_values[best_index] < best_fitness:
            best_solution = population[best_index]
            best_fitness = fitness_values[best_index]

        selected_population = perform_selection(population, fitness_values)
        mutated_population = perform_mutation(selected_population, mutation_rate)
        offspring = perform_crossover(mutated_population, crossover_rate)

        population = offspring
        Iter = Iter + 1
        Convergence_curve[Iter] = best_fitness
    Time = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, Time
