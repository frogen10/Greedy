import random
import numpy as np
import pandas as pd
import unidecode
from itertools import permutations
from math import radians, sin, cos, sqrt, atan2
import os
import matplotlib.pyplot as plt

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
    
    def get_best_individuals(self, city_goods):
        if os.path.exists('bestvalues.csv'):
            # Read the existing file
            best_values_df = pd.read_csv('bestvalues.csv')
            best_individuals = []
            for i in range(len(best_values_df) - 1):  # Exclude the last row which is the total
                car_cities = best_values_df.iloc[i]['Cities'].strip("[]").replace("'", "").split(", ")
                best_individuals.append(car_cities)
            return best_individuals
        raise ValueError("No best values found in the CSV file.")

    def initial_population(self, city_goods, n_population=250):
        population = []
        try:
            population.append(self.get_best_individuals(city_goods))
        except:
            pass
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
            if(len(all_Cities) == 0):
                for car in cars:
                    car['cities'].insert(0, self.main_city)
                population.append([car['cities'] for car in cars])
        return population

    def is_valid_car(self, car: list, all_cities_df):
        total_goods = sum(all_cities_df[all_cities_df['city'] == city]['values'].iloc[0] for city in car[1:])
        return total_goods <= self.car_max_capacity
    
    def check_children(self, car1, car2, all_cities_df)->bool:
        valid_child1 = self.is_valid_car(car1, all_cities_df)
        valid_child2 = self.is_valid_car(car2, all_cities_df)
        
        if valid_child1 and valid_child2:
            #print("Both children are valid after crossover and mutation.")
            return True
        return False

    def check_cities(self, parent, all_cities_df):
        cities = set(all_cities_df['city'])
        for car in parent:
            if not self.is_valid_car(car, all_cities_df):
                return False
            for city in car:
                if(city in cities):
                    cities.remove(city)
        if(len(cities) != 0):
            return False
        return True

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        car1_index = random.randint(0, len(parent1) - 1)
        car2_index = random.randint(0, len(parent2) - 1)
        
        car11 = child1[car1_index]
        car12 = child2[car2_index]
        car21 = child2[car1_index]
        car22 = child1[car2_index]
        car11_temp = car11[:crossover_point]+car12[crossover_point:]
        car12_temp = car12[:crossover_point]+car11[crossover_point:]
        car21_temp = car21[:crossover_point]+car22[crossover_point:]
        car22_temp = car22[:crossover_point]+car21[crossover_point:]
        
        child1[car1_index] = car11_temp
        child1[car2_index] = car12_temp
        child2[car1_index] = car21_temp
        child2[car2_index] = car22_temp
            
        return child1, child2

    def mutation(self, offspring):
        '''Scramble Mutation
            Randomly select two genes and swap their positions
        '''
        copy = [offspring.copy()]
        result = []
        for car in offspring:
            cities_to_shuffle = car[1:]  # Get the cities excluding the main city
            random.shuffle(cities_to_shuffle)  # Shuffle the cities
            cities_to_shuffle.insert(0, car[0])  # Insert the main city back to the car
            result.append(cities_to_shuffle)  # Assign the shuffled cities back to the car
        return result

    def run_ga(self, city_goods, n_population, n_generations, crossover_per, mutation_per):
        population = self.initial_population(city_goods, n_population)
        best_individual = population[0]
        for generation in range(n_generations):
            fitness = self.fitness_prob(population)
            new_population = []
            for i in range(n_population//2):
                parents = random.choices(population, weights=fitness, k=2)
                if random.random() < crossover_per:
                    child1, child2 = self.crossover(parents[0], parents[1])
                else:
                    child1, child2 = parents[0], parents[1]
                if random.random() < mutation_per:
                    child1 = self.mutation(child1)
                if random.random() < mutation_per:
                    child2 = self.mutation(child2)
                if(self.check_cities(child1, city_goods)):
                    new_population.append(child1)
                if(self.check_cities(child2, city_goods)):
                    new_population.append(child2)
            population = new_population
            population.append(best_individual)
            best_individual = min(population, key=self.total_dist_individual)
        return best_individual

    def run(self):
        best_individual = self.run_ga(self.city_goods, self.n_population, self.n_generations, self.crossover_per, self.mutation_per)
        print("Best individual (distribution of cities among cars):")
        total_goods_all_cars = 0
        car_values = []
        for i, car in enumerate(best_individual):
            car_goods = sum(self.city_goods[self.city_goods['city'].isin(car)]['values'])
            car_distance = self.total_dist_individual([car])
            total_goods_all_cars += car_goods
            car_values.append({
                'Car': f"Car {i+1}",
                'Cities': car,
                'Total Goods': car_goods,
                'Total Distance': car_distance
            })
            print(f"Car {i+1}: {car}")
            print(f"  Total goods: {car_goods}")
            print(f"  Total distance traveled: {car_distance:.2f} km")
        total_distance_all_cars = self.total_dist_individual(best_individual)
        print("Total distance traveled by all cars:", total_distance_all_cars)
        print("Total goods in all cars:", total_goods_all_cars)
        car_values.append({
            'Car': 'Total',
            'Cities': '',
            'Total Goods': total_goods_all_cars,
            'Total Distance': total_distance_all_cars
        })
        df = pd.DataFrame(car_values)
        if os.path.exists('bestvalues.csv'):
            # Read the existing file
            existing_df = pd.read_csv('bestvalues.csv')
            existing_total_distance = existing_df[existing_df['Car'] == 'Total']['Total Distance'].values[0]
            
            # Compare the distances
            if total_distance_all_cars < existing_total_distance:
                # Save the new solution if the new distance is less
                df.to_csv('bestvalues.csv', index=False)
                print("New solution saved to bestvalues.csv")
            else:
                print("Existing solution is better or equal. No changes made.")
        else:
            # Save the new solution if the file does not exist
            df.to_csv('bestvalues.csv', index=False)
            print("New solution saved to bestvalues.csv")
    
    def showPlot(self):

        shortest_path =self.get_best_individuals(self.city_goods)
        
        fig, ax = plt.subplots()
        # Define colors for different cars
        colors = ['r', 'g', 'b', 'c', 'm']

        for idx, car_path in enumerate(shortest_path):
            x_shortest = []
            y_shortest = []
            for city in car_path:
                if city in self.city_coords:
                    x_value, y_value = self.city_coords[city]
                    x_shortest.append(x_value)
                    y_shortest.append(y_value)

            # Close the loop by adding the starting city to the end
            if x_shortest and y_shortest:
                x_shortest.append(x_shortest[0])
                y_shortest.append(y_shortest[0])

            # Plot the shortest path for each car
            ax.plot(x_shortest, y_shortest, '--o', label=f'Car {idx+1} Route', linewidth=2.5, color=colors[idx % len(colors)])

        plt.legend()

        # Plot all possible connections (optional)
        for i in range(len(self.x)):
            for j in range(i + 1, len(self.x)):
                ax.plot([self.x[i], self.x[j]], [self.y[i], self.y[j]], 'k-', alpha=0.09, linewidth=1)

        # Add labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Shortest Paths for All Cars')
        plt.show()

