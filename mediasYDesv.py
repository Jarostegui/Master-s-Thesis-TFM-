
import numpy as np

nombreFichero = "EGreedy/0.1"


ild = []
unex = []
gini = []
nov = []
acierto = []
tiempo = []
for i in range(5):
    fichero = open("ResultadosFinales/Test"+str(i+1)+"/"+nombreFichero+"/resultados.txt")
    lineas = fichero.read().splitlines()
    ild.append(float(lineas[0].split(": ")[1]))
    unex.append(float(lineas[1].split(": ")[1]))
    nov.append(float(lineas[2].split(": ")[1]))
    gini.append(float(lineas[3].split(": ")[1]))
    acierto.append(float(lineas[4].split(": ")[1]))
    tiempo.append(float(lineas[5].split("s")[0]))

    fichero.close()

print(round(np.mean(ild), 3))
print(round(np.mean(unex), 3))
print(round(np.mean(nov), 3))
print(round(np.mean(gini), 3))
print(round(np.mean(acierto), 3))
print(round(np.mean(tiempo), 3))
print("\n")
print(round(np.std(ild), 3))
print(round(np.std(unex), 3))
print(round(np.std(nov), 3))
print(round(np.std(gini), 3))
print(round(np.std(acierto), 3))
print(round(np.std(tiempo), 3))