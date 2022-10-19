from os import system as lanzar
import estrategia as ES

lista_argumentos = []   #algori  k   E   g  a  b  it
# lista_argumentos.append('EGreedy -1 0.1 -1 -1 -1 -1')
# lista_argumentos.append('EGreedy -1 0.2 -1 -1 -1 -1')
# lista_argumentos.append('EGreedy -1 0.3 -1 -1 -1 -1')
# lista_argumentos.append('EGreedy -1 0.5 -1 -1 -1 -1')
# lista_argumentos.append('Random -1 -1 -1 -1 -1 -1')
# lista_argumentos.append('Greedy -1 -1 -1 -1 -1 -1')
# lista_argumentos.append('UCB -1 -1 0.1 -1 -1 -1')
# lista_argumentos.append('UCB -1 -1 0.01 -1 -1 -1')
# lista_argumentos.append('Thompson -1 -1 -1 1 100 1')
# lista_argumentos.append('Thompson -1 -1 -1 1 200 5')
# lista_argumentos.append("LinUCB -1 -1 -1 -1 -1 -1")
lista_argumentos.append("AdaptativeEGreedy -1 -1 -1 -1 -1 -1")


print("************************************************")
print("*******************EJECUTANDO*******************")
print("************************************************")

for argumentos in lista_argumentos:
    lanzar(f'python main.py {argumentos}')
