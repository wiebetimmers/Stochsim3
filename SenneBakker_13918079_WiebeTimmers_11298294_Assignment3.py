import matplotlib.pyplot as plt


def init_cities(text_file):
    f = open(text_file, 'r')
    g = f.readlines()
    cities_raw = g[6:]
    cities_clean = []

    for city in cities_raw:
        city = city.strip('\n')
        city_entry = city.split(" ")
        if city_entry[0] == 'EOF':
            break
        city_entry = list(map(int, city_entry))
        cities_clean.append(city_entry)

    return cities_clean


cities = init_cities('eil51.tsp.txt')



for city in cities:
    plt.scatter(city[1], city[2], label="%s"%city[0], color='r')
plt.show()