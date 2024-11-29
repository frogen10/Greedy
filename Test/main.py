import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np


class TSP:
    def __init__(self, cities_names, x, y, city_goods, n_population, crossover_per, mutation_per, n_generations, numbers_of_cars, car_max_capacity, minimum_cities):
        self.cities_names = cities_names
        self.x = x
        self.y = y
        self.city_coords = dict(zip(cities_names, zip(x, y)))
        self.city_goods = city_goods
        self.n_population = n_population
        self.crossover_per = crossover_per
        self.mutation_per = mutation_per
        self.n_generations = n_generations
        self.numbers_of_cars = numbers_of_cars
        self.car_max_capacity = car_max_capacity
        self.minimum_cities = minimum_cities


    def initial_population(self, cities_list, n_population = 250):
        
        """
        Generating initial population of cities randomly selected from all 
        possible permutations  of the given cities
        Input:
        1- Cities list 
        2. Number of population 
        Output:
        Generated lists of cities
        """
        
        population = []
        while (len(population)!=n_population):
            shuffle(cities_list)
            if cities_list not in population:
                population.append(cities_list[:])
        return population

    def dist_two_cities(self, city_1, city_2):
        
        """
        Calculating the distance between two cities  
        Input:
        1- City one name 
        2- City two name
        Output:
        Calculated Euclidean distance between two cities
        """
        
        city_1_coords = self.city_coords[city_1]
        city_2_coords = self.city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

    def total_dist_individual(self, individual):
        
        """
        Calculating the total distance traveled by individual, 
        one individual means one possible solution (1 permutation)
        Input:
        1- Individual list of cities 
        Output:
        Total distance traveled 
        """
        
        total_dist = 0
        for i in range(0, len(individual)):
            if(i == len(individual) - 1):
                total_dist += self.dist_two_cities(individual[i], individual[0])
            else:
                total_dist += self.dist_two_cities(individual[i], individual[i+1])
        return total_dist

    def fitness_prob(self, population):
        """
        Calculating the fitness probability 
        Input:
        1- Population  
        Output:
        Population fitness probability 
        """
        total_dist_all_individuals = []
        for i in range (0, len(population)):
            total_dist_all_individuals.append(self.total_dist_individual(population[i]))
            
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - total_dist_all_individuals
        population_fitness_sum = sum(population_fitness)
        population_fitness_probs = population_fitness / population_fitness_sum
        return population_fitness_probs

    def roulette_wheel(self, population, fitness_probs):
        """
        Implement a selection strategy based on proportionate roulette wheel
        Selection.
        Input:
        1- population
        2: fitness probabilities 
        Output:
        selected individual
        """
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0,1,1)
        selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
        return population[selected_individual_index]

    def crossover(self, parent_1, parent_2):
        """
        Implement mating strategy using simple crossover between two parents
        Input:
        1- parent 1
        2- parent 2 
        Output:
        1- offspring 1
        2- offspring 2
        """
        n_cities_cut = len(self.cities_names) - 1
        cut = round(random.uniform(1, n_cities_cut))
        offspring_1 = []
        offspring_2 = []
        
        offspring_1 = parent_1 [0:cut]
        offspring_1 += [city for city in parent_2 if city not in offspring_1]
        
        
        offspring_2 = parent_2 [0:cut]
        offspring_2 += [city for city in parent_1 if city not in offspring_2]
        
        
        return offspring_1, offspring_2

    def mutation(self, offspring):
        """
        Implement mutation strategy in a single offspring
        Input:
        1- offspring individual
        Output:
        1- mutated offspring individual
        """
        n_cities_cut = len(self.cities_names) - 1
        index_1 = round(random.uniform(0,n_cities_cut))
        index_2 = round(random.uniform(0,n_cities_cut))

        temp = offspring [index_1]
        offspring[index_1] = offspring[index_2]
        offspring[index_2] = temp
        return(offspring)

    def run_ga(self, cities_names, n_population, n_generations,
            crossover_per, mutation_per):
        
        population = self.initial_population(cities_names, n_population)
        fitness_probs = self.fitness_prob(population)
        
        parents_list = []
        for i in range(0, int(crossover_per * n_population)):
            parents_list.append(self.roulette_wheel(population,
                                            fitness_probs))

        offspring_list = []    
        for i in range(0,len(parents_list), 2):
            offspring_1, offspring_2 = self.crossover(parents_list[i], 
                                                parents_list[i+1])

        #     print(offspring_1)
        #     print(offspring_2)
        #     print()

            mutate_threashold = random.random()
            if(mutate_threashold > (1-mutation_per)):
                offspring_1 = self.mutation(offspring_1)
        #         print("Offspring 1 mutated", offspring_1)

            mutate_threashold = random.random()
            if(mutate_threashold > (1-mutation_per)):
                offspring_2 = self.mutation(offspring_2)
        #         print("Offspring 2 mutated", offspring_2)


            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list

        fitness_probs = self.fitness_prob(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:n_population]
        best_mixed_offsrping = []
        for i in best_fitness_indices:
            best_mixed_offsrping.append(mixed_offspring[i])


        for i in range(0, n_generations):
            # if (i%10 == 0):
                # print("Generation: ", i)
            
            fitness_probs = self.fitness_prob(best_mixed_offsrping)
            parents_list = []
            for i in range(0, int(crossover_per * n_population)):
                parents_list.append(self.roulette_wheel(best_mixed_offsrping, 
                                                fitness_probs))

            offspring_list = []    
            for i in range(0,len(parents_list), 2):
                offspring_1, offspring_2 = self.crossover(parents_list[i], 
                                                    parents_list[i+1])

                mutate_threashold = random.random()
                if(mutate_threashold > (1-mutation_per)):
                    offspring_1 = self.mutation(offspring_1)

                mutate_threashold = random.random()
                if(mutate_threashold > (1-mutation_per)):
                    offspring_2 = self.mutation(offspring_2)

                offspring_list.append(offspring_1)
                offspring_list.append(offspring_2)


            mixed_offspring = parents_list + offspring_list
            fitness_probs = self.fitness_prob(mixed_offspring)
            sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
            best_fitness_indices = sorted_fitness_indices[0:int(0.8*n_population)]

            best_mixed_offsrping = []
            for i in best_fitness_indices:
                best_mixed_offsrping.append(mixed_offspring[i])
                
            old_population_indices = [random.randint(0, (n_population - 1)) 
                                    for j in range(int(0.2*n_population))]
            for i in old_population_indices:
    #             print(i)
                best_mixed_offsrping.append(population[i])
                
            random.shuffle(best_mixed_offsrping)
                
        return best_mixed_offsrping

    def run(self):
        best_mixed_offsrping = self.run_ga(self.cities_names, self.n_population,
                                self.n_generations, self.crossover_per, self.mutation_per)

        total_dist_all_individuals = []
        for i in range(0, self.n_population):
            total_dist_all_individuals.append(self.total_dist_individual(best_mixed_offsrping[i]))

        index_minimum = np.argmin(total_dist_all_individuals)
        minimum_distance = min(total_dist_all_individuals)

        shortest_path = best_mixed_offsrping[index_minimum]

        print("Shortest path: ", shortest_path)
        print("Minimum distance: ", minimum_distance)
        x_shortest = []
        y_shortest = []
        for city in shortest_path:
            x_value, y_value = self.city_coords[city]
            x_shortest.append(x_value)
            y_shortest.append(y_value)
            
        x_shortest.append(x_shortest[0])
        y_shortest.append(y_shortest[0])

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()

        for i in range(len(self.x)):
            for j in range(i + 1, len(self.x)):
                ax.plot([self.x[i], self.x[j]], [self.y[i], self.y[j]], 'k-', alpha=0.09, linewidth=1)
                
        plt.title(label="TSP Best Route Using GA",
                fontsize=25,
                color="k")

        str_params = '\n'+str(self.n_generations)+' Generations\n'+str(self.n_population)+' Population Size\n'+str(self.crossover_per)+' Crossover\n'+str(self.mutation_per)+' Mutation'
        plt.suptitle("Total Distance Travelled: "+ 
                    str(round(minimum_distance, 3)) + 
                    str_params, fontsize=18, y = 1.047)

        for i, txt in enumerate(shortest_path):
            ax.annotate(str(i+1)+ "- " + txt, (x_shortest[i], y_shortest[i]), fontsize= 20)

        fig.set_size_inches(16, 12)    
        # plt.grid(color='k', linestyle='dotted')
        plt.savefig('solution.png')
        plt.show()
