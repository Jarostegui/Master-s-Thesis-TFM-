import matplotlib.pyplot as plt
import numpy as np
import sys
import os

EPSILON = "Ɛ"
GAMMA = "γ"
ALPHA = "α"
BETA = "β"
SIGMA = "σ"

plt.style.use('seaborn-ticks')


aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/AdaptiveUCB/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "b", label="AdapUCB")


aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/DynamicThompson/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "g", label="DynTS")


aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/EXP3/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "r", label="EXP3")


aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/KNN/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "m", label="Knn")

aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/LinUCB/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "c", label="LinUCB")

aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/Random/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "k", label="Random")


aciertosPlot = []
fichero = open(
    "ResultadosFinales/TestGraficas/VDBE/aciertosPlot.txt", 'r')
lineas = fichero.read().splitlines()
for l in lineas:
    aciertosPlot.append(float(l))
plt.plot(aciertosPlot, "y", label="VDBE")

plt.legend(fontsize=20, ncol=2)
plt.title('Comparativa aciertos algoritmos', fontsize=26)
plt.xlabel('Iteración', fontsize=20)
plt.ylabel('Aciertos', fontsize=20)
plt.show()




# plt.rc('font', size=18)
# fig, axes = plt.subplots(1, 1)

# fig.suptitle("Dynamic Thompson Sampling para distintos valores de k")

# aciertosPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-1-100/aciertosPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "b", label="k = 1")

# aciertosPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-2-100/aciertosPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "g", label="k = 2")

# aciertosPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-5-100/aciertosPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "r", label="k = 5")

# aciertosPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-10-100/aciertosPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "c", label="k = 10")

# aciertosPlot = []
# fichero = open(
#         "ResultadosFinales/Train/DynamicThompson/1-20/1-20-20-100/aciertosPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "m", label="k = 20")

# aciertosPlot=[]
# fichero=open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-50-100/aciertosPlot.txt", 'r')
# lineas=fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "y", label="k = 50")

# aciertosPlot=[]
# fichero=open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-100-100/aciertosPlot.txt", 'r')
# lineas=fichero.read().splitlines()
# for l in lineas:
#     aciertosPlot.append(float(l))
# axes[0].plot(aciertosPlot, "k", label="k = 100")


# ##################################################################################################################################################################

# novedadPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-1-100/novedadPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "b", label="k = 1")

# novedadPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-2-100/novedadPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "g", label="k = 2")

# novedadPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-5-100/novedadPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "r", label="k = 5")

# novedadPlot = []
# fichero = open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-10-100/novedadPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "c", label="k = 10")

# novedadPlot = []
# fichero = open(
#         "ResultadosFinales/Train/DynamicThompson/1-20/1-20-20-100/novedadPlot.txt", 'r')
# lineas = fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "m", label="k = 20")

# novedadPlot=[]
# fichero=open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-50-100/novedadPlot.txt", 'r')
# lineas=fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "y", label="k = 50")

# novedadPlot=[]
# fichero=open(
#     "ResultadosFinales/Train/DynamicThompson/1-20/1-20-100-100/novedadPlot.txt", 'r')
# lineas=fichero.read().splitlines()
# for l in lineas:
#     novedadPlot.append(float(l))
# axes[1].plot(novedadPlot, "k", label="k = 100")

# axes[0].set_ylabel("Aciertos")
# axes[0].set_xlabel("Iteración")
# axes[0].set_xlim(xmin=0)
# axes[0].set_ylim(ymin=0)

# axes[1].set_ylabel("Novedad")
# axes[1].set_xlabel("Iteración")
# axes[1].set_ylim([0, 1])

# handles, labels=axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc=(0.15, 0.60))

# plt.show()


# fichero=open("ResultadosTFG/Thompson/1-100/tiempos.txt", 'r')
# lineas=fichero.read().splitlines()
# iter=[]
# tiempo=[]
# for l in lineas:
#     l=l.split(" ")
#     iter.append(float(l[0]))
#     tiempo.append(float(l[1]))

# plt.ylabel("Segundos")
# plt.xlabel("κ")
# plt.plot(iter, tiempo, marker='o')
# plt.show()


# EPSILON = "Ɛ"
# GAMMA = "γ"
# ALPHA = "α"
# BETA = "β"
#
# if sys.argv[1] == "EGreedy":
#     label = EPSILON+" = "
# if sys.argv[1] == "UCB":
#     label = GAMMA+" = "
# if sys.argv[1] == "Thomspon":
#     label = EPSILON
# if sys.argv[1] == "EGreedy":
#     label = EPSILON
# ficheros = []
# path = "ResultadosTFM/"+sys.argv[1]+"/"
# carpetas = os.listdir(path)
#
#
# for c in carpetas:
#     label = label+c
#     aciertosPlot = []
#     ficheroAciertos = open(path+c+"/aciertosPlot.txt")
#     lineas = ficheroAciertos.read().splitlines()
#     for l in lineas:
#         aciertosPlot.append(float(l))
#     axes[0].plot(aciertosPlot, "", label=)
#     ficheroNovedad = open(path+c+"/novedadPlot.txt")
