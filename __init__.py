
import pandas as pd
from main import TSP
import time

def main():
    n_population = 250
    crossover_per = 0.8
    mutation_per = 0.2
    n_generations = 200
    numbers_of_cars = 5
    car_max_capacity = 1000
    minimum_cities = 30
    dyingFactor = 0.6

    df = pd.read_csv('coords.csv')
    main_city_row = df[df['values'] == 0]
    if main_city_row.empty:
        raise ValueError("No city with values equal to 0 found in the CSV file.")
    main_city = main_city_row.iloc[0]['city']
    #df['city'] = df['city'].apply(lambda x: unidecode.unidecode(x))
    # Assign values to cities_names, x, and y
    cities_names = df['city'].tolist()
    x = df['x'].tolist()
    y = df['y'].tolist()
    city_goods = df.groupby('city')['values'].sum().reset_index()
    city_goods = city_goods.sort_values(by='values', ascending=False)
    tsp = TSP(cities_names, x,y, city_goods, n_population, crossover_per, mutation_per, n_generations, numbers_of_cars, car_max_capacity, minimum_cities, main_city, dyingFactor)
    start_time = time.time()
    tsp.run()
    tsp.showPlot()
    
    # End timer
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")




if __name__ == '__main__':
    main()