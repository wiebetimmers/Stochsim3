import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from math import radians
from itertools import product as x
from itertools import chain as c
import networkx as nx
from tqdm import tqdm

fig = plt.figure(figsize=(6,4), dpi=300)
# Sources for inspiration
# Lecture 9
# https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/anneal.html
# http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/
# https://codereview.stackexchange.com/questions/208387/2-opt-algorithm-for-the-traveling-salesman-and-or-sro

def init_cities(text_file):
    f = open(text_file, 'r')
    g = f.readlines()
    cities_raw = g[6:]
    cities = {'city': [], 'x': [], 'y': []}
    for city in cities_raw:
        city = city.strip('  ')
        city = city.strip(' ')
        city = city.strip('\n')
        city_entry = city.split(" ")
        if city_entry[0] == 'EOF':
            break
        city_entry = filter(None, city_entry)
        city_entry = list(map(int, city_entry))
        cities['city'].append(city_entry[0])
        cities['x'].append(city_entry[1])
        cities['y'].append(city_entry[2])
    return cities


def generate_rand_path(no_cities):
    max_edges = no_cities-1
    count = 0
    edges = []
    cities = list(range(1,no_cities+1, 1))
    first = np.random.choice(cities)
    cities.remove(first)
    for i in range(no_cities):
        second = np.random.choice(cities)
        coordinates = (first, second)
        cities.remove(second)
        edges.append(coordinates)
        first = second
        count +=1
        if count == max_edges:
            break
    edges.append((edges[-1][1], edges[0][0]))
    return edges

def calculate_path_distance(path, distances):
    total_distance = 0.0
    for edge in path[:-1]:
        total_distance += distances.at[edge[0], edge[1]]
    return total_distance

def swap(edge):
    edge = (edge[1], edge[0])
    return edge

def two_opt(cur_path):
    path = cur_path.copy()
    new_path = []
    for p in path:
        new_path.append(p[0])
    new_path.append(path[-1][1])
    idx1 = random.randint(1, len(new_path)-2)
    idx2 = random.randint(2, len(new_path))
    while idx2 == idx1:
        idx2 = random.randint(2, len(new_path))
    for i in range(1, len(new_path) - 2):
        for j in range(i + 1, len(new_path)):
            if j - i == 1:
                continue
            new_route = new_path[:]
            new_route[i:j] = new_path[j - 1:i - 1:-1]
            if i == idx1 and j == idx2:
                new_path = new_route
    edge_path = []
    for i in range(1, len(new_path)):
        edge_path.append((new_path[i - 1], new_path[i]))
    return edge_path


def temperature(its, total_its, temp_scheme):
    if temp_scheme == 'linear':
        return T0 - (its/total_its)
    elif temp_scheme == 'exp_multi':
        return T0 * (1+ALPHA_EXP_MULTI_COOL**its)
    elif temp_scheme == 'log_multi':
        return T0 / (1+ALPHA_LOG_MULTI_COOL*np.log(1+its))
    elif temp_scheme == 'quad_multi':
        return T0 / (1+ALPHA_QUAD_MULTI_COOL*(its**2))
    return

def simulation(path_init, dist_cities, its, temp_scheme=None):
    path = path_init
    tsp_distance = calculate_path_distance(path, dist_cities)
    distance_list = []
    path_change_count = 0
    for i in tqdm(range(its)):
        #for i in range(mutate_its): # mutate mutate_its times the 2-opt elementary edit -
        # In SA terms: we sample the next possible state here
        #print(path)
        new_path = two_opt(path)
        #print(new_path)
        new_tsp_distance = calculate_path_distance(new_path, dist_cities)
        # Sample U
        U = np.random.rand()

        # Compute next move probability
        difference = new_tsp_distance - tsp_distance
        alpha_x_prob = min(np.exp(-(difference) / temperature(i, its, temp_scheme)), 1)

        # If new distance is smaller, we make the move always!
        if U <= alpha_x_prob:
            tsp_distance = new_tsp_distance
            path = new_path
            path_change_count += 1
            distance_list.append(tsp_distance)
        else:
            distance_list.append(tsp_distance)

    print('Performed %s path changes'%(path_change_count))
    return path, distance_list

def get_city_coordinates(city, df_cities):
    x = df_cities.loc[df_cities['city'] == city, 'x'].iloc[0]
    y = df_cities.loc[df_cities['city'] == city, 'y'].iloc[0]
    return x, y

def plot_path(path, df_cities, name):
    for idx, p in enumerate(path):
        city1 = get_city_coordinates(p[0], df_cities)
        city2 = get_city_coordinates(p[1], df_cities)
        x = [city1[0], city2[0]]
        y = [city1[1], city2[1]]
        plt.plot(x, y, '-', c='r')
        plt.text(x[0], y[0], f'{p[0]}')
    plt.savefig('path_graphs/path_%s.jpg'%name)
    plt.clf()
    return


# Initialization
np.random.seed(12345)
cities = init_cities('a280.tsp.txt')
df_cities = pd.DataFrame(cities)
no_cities = len(df_cities)
distance = DistanceMetric.get_metric('euclidean')
pw_dis = distance.pairwise(df_cities[['x','y']].to_numpy())
dist_cities = pd.DataFrame(pw_dis, columns=df_cities.city.unique(), index=df_cities.city.unique())

# Number of SA iterations
ITS = 1000

# Cooling scheme parameters
T0 = 1.0
ALPHA_EXP_MULTI_COOL = 0.85
ALPHA_LOG_MULTI_COOL = 2
ALPHA_QUAD_MULTI_COOL = 3


if __name__ == '__main__':
    temp_schemes = ['linear', 'exp_multi', 'log_multi', 'quad_multi']
    distances = []
    paths = []
    path_init = generate_rand_path(no_cities)
    for ts in temp_schemes:
        print('\nPerforming SA with cooling scheme: %s'%ts)
        path, distance_list = simulation(path_init, dist_cities, ITS, temp_scheme=ts)
        plot_path(path, df_cities, ts)
        distances.append(distance_list)
        paths.append(path)
    for idx, ds in enumerate(distances):
        plt.plot(list(range(0,ITS,1)), ds, label='%s'%temp_schemes[idx])
    plt.legend()
    plt.savefig('distances.jpg')

# Run single simulation
#path, distance_list = simulation(no_cities, dist_cities, ITS, temp_scheme='log_multi')
#plot_path(path, df_cities, 'log_multi')
