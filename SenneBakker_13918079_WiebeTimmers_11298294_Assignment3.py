import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from math import radians
from itertools import product as x
from itertools import chain as c
import networkx as nx
from tqdm import tqdm


# Sources for inspiration
# Lecture 9
# https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/anneal.html
# http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/

def init_cities(text_file):
    f = open(text_file, 'r')
    g = f.readlines()
    cities_raw = g[6:]
    cities = {'city': [], 'x': [], 'y': []}

    for city in cities_raw:
        city = city.strip(' ')
        city = city.strip('\n')
        city_entry = city.split(" ")
        if city_entry[0] == 'EOF':
            break
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

def mutate_path(cur_path, path_length):
    path = cur_path.copy()
    edge1_idx = np.random.randint(1, path_length-1)
    edge2_idx = np.random.randint(1, path_length-1)
    while edge1_idx == edge2_idx:
        edge2_idx = np.random.randint(1, path_length - 1)
    if edge1_idx > edge2_idx:
        idx1 = edge2_idx
        idx2 = edge1_idx
    else:
        idx1= edge1_idx
        idx2= edge2_idx

    midsect = False
    #print('point to change:', path[idx1], path[idx2])

    count_mid = 1
    for idx, p in enumerate(path):
        if idx == idx1:
            temp_side1 = path[idx-1][1]
            temp_side2 = path[idx2+1][0]
            path[idx1-1] = (path[idx1-1][0], temp_side2)
            path[idx2+1] = (path[idx2+1][0], temp_side1)
            midsect = True
            store_id1 = path[idx1]
            path[idx1] = path[idx2]
            path[idx2] = store_id1
            path[idx1] = swap(path[idx1])
            path[idx2] = swap(path[idx2])

        if midsect == True and idx1 < idx < idx2:
            swap1 = path[idx]
            swap2 = path[idx2-count_mid]
            path[idx2-count_mid] = swap1
            path[idx] = swap2
            count_mid += 1
            path[idx] = swap(path[idx])

        if idx == idx2:
            midsect = False

    return path


def temperature(its, total_its, temp_scheme):
    if temp_scheme == 'linear':
        return T0 - (its/total_its)
    elif temp_scheme == 'exp_multi':
        return T0 * (ALPHA_EXP_MULTI_COOL**its)
    elif temp_scheme == 'log_multi':
        return T0 / (1+ALPHA_LOG_MULTI_COOL*np.log(1+its))
    elif temp_scheme == 'quad_multi':
        return T0 / (1+ALPHA_QUAD_MULTI_COOL*(its**2))
    return

def simulation(no_cities, dist_cities, its, temp_scheme=None):
    path = generate_rand_path(no_cities)
    tsp_distance = calculate_path_distance(path, dist_cities)
    distance_list = []
    path_change_count = 0
    for i in tqdm(range(its)):
        #for i in range(mutate_its): # mutate mutate_its times the 2-opt elementary edit -
        # In SA terms: we sample the next possible state here
        #print(path)
        print(path)
        new_path = mutate_path(path, no_cities-1)
        #print(new_path)
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

#showpath()
#np.random.seed(12345)
cities = init_cities('eil51.tsp.txt')
df_cities = pd.DataFrame(cities)
no_cities = len(df_cities)
distance = DistanceMetric.get_metric('euclidean')
pw_dis = distance.pairwise(df_cities[['x','y']].to_numpy())
dist_cities = pd.DataFrame(pw_dis, columns=df_cities.city.unique(), index=df_cities.city.unique())

ITS = 5

# cooling scheme parameters
T0 = 1.0
ALPHA_EXP_MULTI_COOL = 0.85
ALPHA_LOG_MULTI_COOL = 2
ALPHA_QUAD_MULTI_COOL = 3

temp_schemes = ['linear', 'exp_multi', 'log_multi', 'quad_multi']
distances = []
paths = []
'''
for ts in temp_schemes:
    print('Performing SA with cooling scheme: %s'%ts)
    path, distance_list = simulation(no_cities, dist_cities, ITS, temp_scheme=ts)
    distances.append(distance_list)
    paths.append(path)

for idx, ds in enumerate(distances):
    plt.plot(list(range(0,ITS,1)), ds, label='%s'%temp_schemes[idx])

plt.legend()
plt.savefig('distances.jpg')'''

def plot_path(path, df_cities):
    for idx, p in enumerate(path):
        city1 = get_city_coordinates(p[0], df_cities)
        city2 = get_city_coordinates(p[1], df_cities)
        x = [city1[0], city2[0]]
        y = [city1[1], city2[1]]
        plt.plot(x, y, '-', c='r')
        plt.text(x[0], y[0], f'city {idx}')
    plt.show()


path, distance_list = simulation(no_cities, dist_cities, ITS, temp_scheme='linear')
#plot_path(path, df_cities)
'''


plt.show()'''

# Plot the interconnected graph
#plt.plot(*zip(*c(*list(x(df_cities[['x','y']].to_numpy(),df_cities[['x','y']].to_numpy())))))
#plt.show()
