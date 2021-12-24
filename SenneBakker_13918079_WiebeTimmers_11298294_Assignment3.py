import random
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from math import radians
from itertools import product as x
from itertools import chain as c
import networkx as nx
from tqdm import tqdm
import statistics as st
import os
import multiprocessing   # used to improve time efficiency
from scipy import stats

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

def stats_plot(temp_dict, ts):
    mean_list = []
    stdev_list = []
    for i in range(len(temp_dict[0])):
        entry = [item[i] for item in temp_dict]
        mean_list.append(st.mean(entry))
        stdev_list.append(st.stdev(entry))

    last_means = [item[len(temp_dict[0])-1] for item in temp_dict]

    its = list(range(len(temp_dict[0])))
    plt.xlabel('iterations')
    plt.ylabel('distance')
    plt.plot(its, mean_list, 'b-', label='Distance')
    plt.fill_between(its, np.subtract(mean_list, stdev_list),
                        np.add(mean_list, stdev_list), color='b', alpha=0.2)
    plt.savefig('distances/%s_mean_%s'%(ts, SAMPLE_ITS))
    plt.clf()
    return mean_list, stdev_list, its, last_means

def stat_test(file, temp_schemes):
    methods = pd.Series(temp_schemes)
    infile = open(file, 'rb')
    data = pkl.load(infile)
    # fill in p.value for every combination, we do a Welsh t-test and test if the samples are different
    # We set our p crit value to 0.01
    # H0: method x mean = method y mean
    # H1: method x mean != method y mean
    df = pd.DataFrame(methods.apply(lambda x: methods.apply(lambda y: (stats.ttest_ind(data[x],
                                                                                       data[y],
                                                                                       equal_var=False)[1]))))
    df.index = methods
    df.columns = methods
    df.to_excel('stat_test_output.xlsx')
    for ts in temp_schemes:
        print('%s mean and stdev'%ts)
        print(st.mean(data[ts]))
        print(st.stdev(data[ts]))
        print('\n')
    return

# Initialization
np.random.seed(34537)
#cities = init_cities('eil51.tsp.txt')
cities = init_cities('a280.tsp.txt')
df_cities = pd.DataFrame(cities)
no_cities = len(df_cities)
distance = DistanceMetric.get_metric('euclidean')
pw_dis = distance.pairwise(df_cities[['x','y']].to_numpy())
dist_cities = pd.DataFrame(pw_dis, columns=df_cities.city.unique(), index=df_cities.city.unique())

# Number of SA iterations
ITS = 10000

# Cooling scheme parameters
T0 = 1.0
ALPHA_EXP_MULTI_COOL = 0.85
ALPHA_LOG_MULTI_COOL = 5
ALPHA_QUAD_MULTI_COOL = 0.95
SAMPLE_ITS = 10   # we test every method with param setting sample_Its times

# Decide what to run:
simulation_process = True
plotting = True
stat_testing = True

if __name__ == '__main__':
    temp_schemes = ['linear', 'exp_multi', 'log_multi', 'quad_multi']
    if simulation_process:
        distances = {
            'linear':[],
            'exp_multi':[],
            'log_multi':[],
            'quad_multi':[]
        }
        paths = {
            'linear':[],
            'exp_multi':[],
            'log_multi':[],
            'quad_multi':[]
        }
        path_init = generate_rand_path(no_cities)
        sample_its = list(range(1, SAMPLE_ITS + 1, 1))
        for ts in temp_schemes:
            print('\nPerforming SA with cooling scheme: %s' % ts)
            for i in range(SAMPLE_ITS):
                path, distance_list = simulation(path_init, dist_cities, ITS, temp_scheme=ts)
                plot_path(path, df_cities, ts)
                distances[ts].append(distance_list)
                paths[ts].append(path)
        file_to_write = open("results.pickle", "wb")
        pkl.dump(distances, file_to_write)
        file_to_write.close()

    if plotting:
        infile = open("results.pickle", 'rb')
        new_dict_results = pkl.load(infile)
        m_lists1 = []
        m_lists2 = []
        for ts in temp_schemes:
            mean_list, stdev_list, its, last_means = stats_plot(new_dict_results[ts], ts)
            m_lists1.append(last_means)
            m_lists2.append(mean_list)

        for idx, m in enumerate(m_lists2):
            plt.plot(its, m, label='%s'%temp_schemes[idx])
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('avg_distance')
        plt.savefig('avg_distances.jpg')
        plt.clf()
        file_to_write2 = open("stat_results.pickle", "wb")
        distances_mean = {
            'linear': m_lists1[0],
            'exp_multi': m_lists1[1],
            'log_multi': m_lists1[2],
            'quad_multi': m_lists1[3]
        }
        pkl.dump(distances_mean, file_to_write2)
        file_to_write2.close()

    if stat_testing:
        stat_test("stat_results.pickle", temp_schemes)


# Run single simulation
#path, distance_list = simulation(no_cities, dist_cities, ITS, temp_scheme='log_multi')
#plot_path(path, df_cities, 'log_multi')
