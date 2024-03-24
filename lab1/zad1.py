import matplotlib.pyplot as plt
import numpy as np



def read_file(filename):
    global content
    try:
        with open(filename, 'r') as file:
            content = file.read()
            print(f"Zawartość pliku {filename}:")

    except FileNotFoundError:
        print(f"Plik {filename} nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania pliku {filename}: {e}")








filename = 'ekg1.txt'

read_file(filename)


ekg = tuple(content.split("\n"))

y = [[] for i in range(12)]
for i in range(12):
    for row in ekg:
        y[i].append(int(row.split(" ")[i]))


x = tuple(range(1,len(ekg)+1))
x = [i * 0.001 for i in x]

for i in range(12):
    plt.plot(x, y[i])

plt.ylabel("Wartość")
plt.xlabel("Czas [s]")

plt.show()

