# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import entorno as EN
import time
import random
import os
import sys
from datos import datos
import estrategia as ES
random.seed()

EPSILON = "Ɛ"
GAMMA = "γ"
ALPHA = "α"
BETA = "β"
SIGMA = "σ"

# dataset = datos('Datos/ratingsML1MTrain.csv', 'Datos/tags1M.csv')
# dataName = "Train"

dataset = datos('Datos/ratingsML1MTest.csv', 'Datos/tags1M.csv')
dataName = "Test"


def ejecutar(estrategia):
    numIteraciones = 100

    dataset.reset()

    entorno = EN.entorno(numIteraciones)

    print("#################################")
    print(estrategia)
    ES.EList = []

    aciertosPlot, novedadPlot, ildList, unexpectednessList, \
        novedadActual, giniList, aciertosFinal, end, start, \
        recomendPlot = entorno.recomendar(dataset, estrategia)

    fileAciertos = estrategia.pathBase+"/aciertosPlot.txt"
    fileNovedad = estrategia.pathBase+"/novedadPlot.txt"
    fileResultados = estrategia.pathBase+"/resultados.txt"
    filePltAciertos = estrategia.pathBase+"/Aciertos.png"
    filePltNovedad = estrategia.pathBase+"/Novedad.png"
    filePltEpsilon = estrategia.pathBase+"/EvolucionEpsilon.png"

    os.makedirs(os.path.dirname(fileAciertos), exist_ok=True)
    os.makedirs(os.path.dirname(fileNovedad), exist_ok=True)
    os.makedirs(os.path.dirname(fileResultados), exist_ok=True)

    fileAciertos = open(fileAciertos, "w")
    fileNovedad = open(fileNovedad, "w")
    fileResultados = open(fileResultados, "w")

    for i in aciertosPlot:
        fileAciertos.write(str(i)+"\n")
    fileAciertos.close()

    for i in novedadPlot:
        fileNovedad.write(str(i)+"\n")
    fileNovedad.close()

    fileResultados.write("ILD@10: "+str(round(np.mean(ildList), 4))+"\n")
    fileResultados.write("Unexpectedness@10: " +
                         str(round(np.mean(unexpectednessList), 4))+"\n")
    fileResultados.write("Novedad: "+str(round(novedadActual, 4))+"\n")
    fileResultados.write("Gini: "+str(round(np.mean(giniList), 4))+"\n")
    fileResultados.write("Aciertos: "+str(round(aciertosFinal, 4))+"\n")
    fileResultados.write(str(round(end - start, 4))+"s\n")

    plt.plot(recomendPlot, aciertosPlot)
    plt.title(estrategia.title)
    plt.ylabel('Aciertos')
    plt.xlabel('Iteraciones')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.savefig(filePltAciertos)
    plt.close()

    plt.plot(recomendPlot, novedadPlot)
    plt.title(estrategia.title)
    plt.ylabel('Novedad')
    plt.xlabel('Iteraciones')
    plt.ylim(ymin=0, ymax=1)
    plt.savefig(filePltNovedad)
    plt.close()

    plt.plot(ES.EList)
    plt.title(estrategia.title)
    plt.ylabel(EPSILON)
    plt.xlabel('Iteraciones')
    plt.savefig(filePltEpsilon)
    plt.close()

    return


if __name__ == "__main__":
    print("************************************************")
    print("*******************EJECUTANDO*******************")
    print("*******************"+dataName+"*******************")



# # EGreedy
#     ejecutar(ES.EpsilonGreedyPolicy(0.1))

# # # #Random
#     ejecutar(ES.RandomPolicy())

# # # #Greedy
#     ejecutar(ES.GreedyPolicy())

# AdaptiveUCB
    ejecutar(ES.AdaptiveUCBPolicy(dataset))

# # # #UCB
    ejecutar(ES.UCBPolicy(dataset, 0.01))

# # #Thompson
#     ejecutar(ES.ThompsonSamplingPolicy(1, 100, 2))

# # #Knn
#     ejecutar(ES.knn(dataset, 10))

# # #AdaptativeEGreedy
#     ejecutar(ES.AdaptiveEpsilonGreedyPolicy(1000, 7))

# # #LinUCB
#     ejecutar(ES.LinUCBPolicy(dataset, 0.01))

# # CLUB
#     ejecutar(ES.CLUBPolicy(dataset, 0.2, 0.3))

# #VDBE
#     ejecutar(ES.VDBEEpsilonGreedyPolicy(0.04, 0.001))

# # #EpsilonFirstGreedy
    # ejecutar(ES.EpsilonFirstGreedyPolicy(100000))

# # #DynamicThomsponSampling
#     ejecutar(ES.DynamicThompsonSamplingPolicy(dataset, 1, 20, 10, 100))


# EXP3
#     ejecutar(ES.EXP3Policy(dataset, 0.6))