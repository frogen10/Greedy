import random
import numpy as np
import pandas as pd
import unidecode
from itertools import permutations
from math import radians, sin, cos, sqrt, atan2

class TSP:
    def __init__(self, cities_names, x, y, city_goods, n_population, crossover_per, mutation_per, n_generations, numbers_of_cars, car_max_capacity, minimum_cities, main_city):
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
        self.main_city = main_city  # Main city

    def dist_two_cities(self, city_1, city_2):
        # Haversine formula to calculate the distance between two points on the Earth's surface
        R = 6371.0  # Radius of the Earth in kilometers
        lat1, lon1 = self.city_coords[city_1]
        lat2, lon2 = self.city_coords[city_2]

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    def total_dist_individual(self, individual):
        total_dist = 0
        for car_route in individual:
            if car_route:
                total_dist += self.dist_two_cities(self.main_city, car_route[0])
                for i in range(len(car_route) - 1):
                    total_dist += self.dist_two_cities(car_route[i], car_route[i+1])
                total_dist += self.dist_two_cities(car_route[-1], self.main_city)
        return total_dist

    def fitness_prob(self, population):
        total_dist_all_individuals = [self.total_dist_individual(ind) for ind in population]
        fitness = [1 / dist for dist in total_dist_all_individuals]
        total_fitness = sum(fitness)
        fitness_prob = [f / total_fitness for f in fitness]
        return fitness_prob

    def initial_population(self, city_goods, n_population=250):
        population = []
        for _ in range(n_population):
            cars = [{'capacity': self.car_max_capacity, 'cities': []} for _ in range(self.numbers_of_cars)]
            all_Cities = city_goods.copy()
            all_Cities = all_Cities[all_Cities['city'] != self.main_city]
            cars_full = set()
            while len(cars_full) < self.numbers_of_cars and len(all_Cities) != 0:
                for car in cars:
                    if(len(all_Cities) == 0):
                        break
                    row = all_Cities.sample().iloc[0]
                    city = row['city']
                    goods = row['values']
                    if car['capacity'] >= goods:
                        car['cities'].append(city)
                        all_Cities = all_Cities[all_Cities['city'] != city]
                        car['capacity'] -= goods
                    else:
                        cars_full.add(cars.index(car))
                
            for car in cars:
                car['cities'].insert(0, self.main_city)
            population.append([car['cities'] for car in cars])
        return population

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, offspring):
        for car in offspring:
            random.shuffle(car[1:])  # Shuffle the order of cities, excluding the main city
        return offspring

    def run_ga(self, city_goods, n_population, n_generations, crossover_per, mutation_per):
        population = self.initial_population(city_goods, n_population)
        for generation in range(n_generations):
            fitness = self.fitness_prob(population)
            new_population = []
            for _ in range(n_population // 2):
                parents = random.choices(population, weights=fitness, k=2)
                if random.random() < crossover_per:
                    child1, child2 = self.crossover(parents[0], parents[1])
                else:
                    child1, child2 = parents[0], parents[1]
                if random.random() < mutation_per:
                    child1 = self.mutation(child1)
                if random.random() < mutation_per:
                    child2 = self.mutation(child2)
                new_population.extend([child1, child2])
            population = new_population
        best_individual = min(population, key=self.total_dist_individual)
        return best_individual

    def run(self):
        best_individual = self.run_ga(self.city_goods, self.n_population, self.n_generations, self.crossover_per, self.mutation_per)
        print("Best individual (distribution of cities among cars):")
        total_goods_all_cars = 0
        for i, car in enumerate(best_individual):
            car_goods = sum(self.city_goods[self.city_goods['city'].isin(car)]['values'])
            car_distance = self.total_dist_individual([car])
            total_goods_all_cars += car_goods
            print(f"Car {i+1}: {car}")
            print(f"  Total goods: {car_goods}")
            print(f"  Total distance traveled: {car_distance:.2f} km")
        print("Total distance traveled by all cars:", self.total_dist_individual(best_individual))
        print("Total goods in all cars:", total_goods_all_cars)
