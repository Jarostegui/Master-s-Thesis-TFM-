# -*- coding: utf-8 -*-
import numpy as np
from datos import datos
from abc import ABCMeta, abstractmethod
import math
import random
import time
from heapq import nlargest
import networkx as nx
import matplotlib.pyplot as plt

EPSILON = "Ɛ"
GAMMA = "γ"
ALPHA = "α"
BETA = "β"
SIGMA = "σ"
DELTA = "δ"

EList = []

class estrategia(object):

    def __str__(self):
        return self.title

    def update(self, dataset, usuario, result, acierto):
        if acierto == True:
            dataset.items[result][0] += 1
        else:
            return

class EpsilonGreedyPolicy(estrategia):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.title = "EGreedy "+EPSILON+" = "+str(epsilon)
        self.pathBase = "ResultadosTFM/EGreedy/"+str(epsilon)

    def choose(self, datos, usuario, epoca):
        if np.random.random() < self.epsilon:
            return random.choice(list(datos.porRecomendar[usuario]))
        else:
            return datos.getItemPuntuacionMaxima(usuario)


class GreedyPolicy(estrategia):

    def __init__(self):
        self.title = "Greedy"
        self.pathBase = "ResultadosTFM/Greedy/"

    def choose(self, datos, usuario, epoca):
        return datos.getItemPuntuacionMaxima(usuario)


class RandomPolicy(estrategia):

    def __init__(self):
        self.title = "Random"
        self.pathBase = "ResultadosTFM/Random/"

    def choose(self, datos, usuario, epoca):
        return random.choice(list(datos.porRecomendar[usuario]))


class UCBPolicy(estrategia):

    def __init__(self, datos, gamma):
        self.gamma = gamma
        self.datos = datos
        self.epoca = 0
        self.primeraVuelta = list(datos.items.keys())
        self.title = "UCB "+GAMMA+" = "+str(gamma)
        self.pathBase = "ResultadosTFM/UCB/"+str(gamma)

    def choose(self, datos, usuario, epoca):
        self.epoca = epoca  
        if self.primeraVuelta:
            result = random.choice(self.primeraVuelta)
            self.primeraVuelta.remove(result)
            return result

        maximum = max(datos.porRecomendar[usuario], key=self.calcular)
        return maximum

    def calcular(self, k):
        if self.datos.items[k][1] == 0:
            return 0
        else:
            porcentajeAcierto = self.datos.getPorcentajeAcierto(k)
            exploration = math.sqrt(
                self.gamma * np.log(self.epoca) / self.datos.items[k][1])

            return porcentajeAcierto + exploration


class ThompsonSamplingPolicy(estrategia):

    def __init__(self, a, b, k):
        self.alfaCero = a
        self.betaCero = b
        self.k = k
        self.betaDist = {}
        self.contador = k
        self.title = "Thompson "+ALPHA+" = "+str(a)+" "+BETA+" = "+str(b)
        self.pathBase = "ResultadosTFM/Thompson/"+str(a)+"-"+str(b)

    def reCalcular(self, datos):
        for k in datos.items.keys():
            self.betaDist[k] = random.betavariate(
                datos.items[k][0] + self.alfaCero, datos.items[k][1] - datos.items[k][0] + self.betaCero)
        return

    def choose(self, datos, usuario, epoca):
        if self.contador == self.k:
            self.contador = 0
            self.reCalcular(datos)
        self.contador += 1
        maximum = max(datos.porRecomendar[usuario], key=self.betaDist.get)
        # Recalcular el item recomendado
        self.betaDist[maximum] = random.betavariate(
            datos.items[maximum][0] + self.alfaCero, datos.items[maximum][1] - datos.items[maximum][0] + self.betaCero)

        return maximum


class knn(estrategia):

    def __init__(self, dataset, k):
        #{Usuario, {Item, rating}}
        self.ratings = {}

        #{Usuario, {Usuario, interseccion}}
        self.similitudes = {}

        #{Usuario, numRatingsPositivos}
        self.numRatingsPositivos = {}

        #{Item, [Usuarios]}
        self.usuariosPuntuadoItem = {}

        self.k = k
        self.dataset = dataset
        self.usuarios = dataset.ratings
        self.items = self.dataset.items
        self.title = "KNN k = "+str(k)
        self.pathBase = "ResultadosTFM/KNN/"+str(k)

        # Interseccion |u| y |v|
        for u in self.usuarios:
            self.ratings[u] = {}
            self.similitudes[u] = {}
            self.numRatingsPositivos[u] = 0
            for v in self.usuarios:
                if u != v:
                    self.similitudes[u][v] = 0

        for i in self.items:
            self.usuariosPuntuadoItem[i] = []

    def choose(self, datos, usuario, epoca):
        self.dataset = datos
        result = {}
        usuariosSimilares = self.getTopK(usuario, self.k)

        for v in usuariosSimilares:
            interseccion = self.similitudes[usuario][v]                
            try:
                similitud = interseccion / \
                    (self.numRatingsPositivos[usuario] +
                     self.numRatingsPositivos[v] - interseccion)
            except:
                similitud = 0

            items = self.ratings[v].keys()
            keys = list(items)

            for i in keys:
                if i not in self.dataset.yaRecomendado[usuario]:
                    if i not in result:
                        result[i] = 0
                    if float(self.ratings[v][i]) > 3.0:
                        result[i] += similitud

        if not result:
            return random.choice(list(self.dataset.porRecomendar[usuario]))
        else:
            return max(result)

    def update(self, dataset, usuario, result, acierto):
        if acierto == True:
            dataset.items[result][0] += 1
            self.ratings[usuario][result] = 5.0
            self.usuariosPuntuadoItem[result].append(usuario)
            self.numRatingsPositivos[usuario] += 1
            for u in self.usuariosPuntuadoItem[result]:
                intersecc = len(
                    list(set(self.ratings[usuario]).intersection(self.ratings[u])))
                self.similitudes[usuario][u] = intersecc
                self.similitudes[u][usuario] = intersecc
        else:
            return

    def getTopK(self, usuario, k):
        keys = list(self.similitudes[usuario])
        random.shuffle(keys)
        kHighest = nlargest(k, keys, key=self.similitudes[usuario].get)
        return kHighest


class LinUCBPolicy(estrategia):

    def __init__(self, dataset, alfa):
        self.alfa = alfa
        self.title = "LinUCB "+ALPHA+" = "+" "+str(alfa)
        self.pathBase = "ResultadosTFM/LinUCB/"+str(alfa)
        #A: {item:A}
        self.A = {}
        #b: {item:b}
        self.b = {}
        #theta: {item:theta}

        self.T = {}
        #p: {item:p}
        self.p = {}
        #A_inv: {item:inversa}
        self.A_inv = {}

        vistos = set()
        for i in dataset.features.keys():
            if i not in vistos:
                # 18 = numero de posibles caracteristicas
                self.A[i] = np.identity(18)
                self.b[i] = np.zeros(18)
            self.A_inv[i] = np.linalg.inv(self.A[i])
            self.T[i] = np.matmul(self.A_inv[i], self.b[i])
            Xt = dataset.getFeaturesVector(i)
            primero = np.matmul(self.T[i].transpose(), Xt)
            segundo = self.alfa * \
                math.sqrt(
                    np.matmul(np.matmul(np.array(Xt.transpose()), self.A_inv[i]), np.array(Xt)))
            self.p[i] = primero + segundo
            vistos.add(i)

    def choose(self, datos, usuario, epoca):
        return max(datos.porRecomendar[usuario], key=self.p.get)

    def update(self, dataset, usuario, result, acierto):
        Xt = dataset.getFeaturesVector(result)
        self.A[result] = self.A[result] + np.matmul(Xt, Xt.transpose())
        self.A_inv[result] = np.linalg.inv(self.A[result])
        if acierto == True:
            self.b[result] = self.b[result] + Xt
            dataset.items[result][0] += 1
        self.T[result] = np.matmul(self.A_inv[result], self.b[result])
        primero = np.matmul(self.T[result].transpose(), Xt)
        segundo = self.alfa * \
            math.sqrt(np.matmul(np.matmul(np.array(Xt.transpose()),
                                          self.A_inv[result]), np.array(Xt)))
        self.p[result] = primero + segundo


class CLUBPolicy(estrategia):

    def __init__(self, dataset, alfa, alfa2):
        self.alfa = alfa
        self.alfa2 = alfa2
        self.title = "CLUB "+ALPHA+" = "+" " + \
            str(alfa)+" "+ALPHA+"2 = "+str(alfa2)
        self.pathBase = "ResultadosTFM/CLUB/"+str(alfa)+"-"+str(alfa2)
        #A: {user:A}
        self.A = {}
        self.Ac = {}
        #b: {user:b}
        self.b = {}
        self.bc = {}
        #theta: {user:theta}
        self.T = {}
        self.Tc = {}
        #p: {user:p}
        self.p = {}
        #A_inv: {user:inversa}
        self.A_inv = {}
        self.Ac_inv = {}
        num_users = len(dataset.ratings.keys())
        self.G = nx.erdos_renyi_graph(
            num_users, (3*np.log(num_users))/num_users)

        # Para cada usuario
        vistos = set()
        for u in dataset.porRecomendar.keys():
            if u not in vistos:
                # 18 = numero de posibles caracteristicas
                self.A[u] = np.identity(18)
                self.b[u] = np.zeros(18)
            self.A_inv[u] = np.linalg.inv(self.A[u])
            self.T[u] = np.matmul(self.A_inv[u], self.b[u])
            vistos.add(u)

        # Para cada cluser
        self.sub_graphs = list(nx.connected_component_subgraphs(self.G))
        for i in self.sub_graphs:
            self.Ac[i] = np.identity(18)
            self.bc[i] = np.zeros(18)
            self.Ac_inv[i] = np.linalg.inv(self.Ac[i])
            self.Tc[i] = np.matmul(self.Ac_inv[i], self.bc[i])

    def choose(self, datos, usuario, epoca):
        for c in self.sub_graphs:
            if usuario in c:
                for i in datos.items:
                    Xt = datos.getFeaturesVector(i)
                    primero = np.matmul(self.Tc[c].transpose(), Xt)
                    segundo = self.alfa * \
                        math.sqrt(
                            np.matmul(np.matmul(np.array(Xt.transpose()), self.Ac_inv[c]), np.array(Xt)))
                    self.p[i] = primero + segundo
        return max(datos.porRecomendar[usuario], key=self.p.get)

    def update(self, dataset, usuario, result, acierto):
        Xt = dataset.getFeaturesVector(result)
        self.A[result] = self.A[result] + np.matmul(Xt, Xt.transpose())
        self.A_inv[result] = np.linalg.inv(self.A[result])
        if acierto == True:
            self.b[result] = self.b[result] + Xt
            dataset.items[result][0] += 1
        self.T[result] = np.matmul(self.A_inv[result], self.b[result])
        primero = np.matmul(self.T[result].transpose(), Xt)
        segundo = self.alfa * \
            math.sqrt(np.matmul(np.matmul(np.array(Xt.transpose()),
                                          self.A_inv[result]), np.array(Xt)))
        self.p[result] = primero + segundo
        for i in self.sub_graphs:
            for j in i.nodes():
                self.Ac[i] += (self.A[j+1] - np.identity(18))
                self.bc[i] += self.b[j+1]
            self.Ac_inv[i] = np.linalg.inv(self.Ac[i])
            self.Tc[i] = np.matmul(self.Ac_inv[i], self.bc[i])
        vecinos = list(self.G.neighbors(usuario-1))
        for v in vecinos:
            a = np.linalg.norm(self.T[usuario] - self.T[v+1])
            b = self.alfa2 * math.sqrt((1+math.log(len(dataset.yaRecomendado[usuario])+1))/(
                1+len(dataset.yaRecomendado[usuario])))
            c = self.alfa2 * \
                math.sqrt(
                    (1+math.log(len(dataset.yaRecomendado[v+1])+1))/(1+len(dataset.yaRecomendado[v+1])))
            if a > b + c:
                self.G.remove_edge(usuario-1, v)



class AdaptiveEpsilonGreedyPolicy(estrategia):

    def __init__(self, limit, f):
        self.f = f
        self.limit = limit
        self.epsilon = 0.5
        self.aciertosAcumulados = 0
        self.maxPrev = 0
        self.maxCurr = 0
        self.k = 0
        self.cuentaAcumulada = 0
        self.title = "Adaptive EGreedy l="+str(self.limit)+" f="+str(self.f)
        self.pathBase = "ResultadosTFM/AdaptiveEGreedy/" + \
            str(self.limit)+"-"+str(self.f)

    def choose(self, datos, usuario, epoca):
        EList.append(self.epsilon)
        self.cuentaAcumulada += 1
        if np.random.random() < self.epsilon:
            self.maxCurr = self.aciertosAcumulados / self.cuentaAcumulada
            self.k += 1
            if self.k == self.limit:
                delta = (self.maxCurr - self.maxPrev) * self.f
                if delta > 0:
                    self.epsilon = (1.0 / (1.0 + math.exp(-2 * delta))) - 0.5
                elif delta < 0:
                    self.epsilon = 0.5
                self.maxPrev = self.maxCurr
                self.k = 0
                self.aciertosAcumulados = 0
                self.cuentaAcumulada = 0
            return random.choice(list(datos.porRecomendar[usuario]))
        else:
            return datos.getItemPuntuacionMaxima(usuario)

    def update(self, dataset, usuario, result, acierto):
        if acierto == True:
            self.aciertosAcumulados += 1
            dataset.items[result][0] += 1
        else:
            return


class VDBEEpsilonGreedyPolicy(estrategia):

    def __init__(self, sigma, delta):
        self.epsilon = 1
        self.sigma = sigma # > 0
        self.delta = delta #[0, 1)
        self.title = "VDBE "+EPSILON+"-Greedy "+SIGMA+" = "+str(self.sigma)+" "+DELTA+" = "+str(self.delta)
        self.pathBase = "ResultadosTFM/VDBEEGreedy/"+str(self.sigma)+"-"+str(self.delta)

    def choose(self, datos, usuario, epoca):
        EList.append(self.epsilon)
        if np.random.random() < self.epsilon:
            return random.choice(list(datos.porRecomendar[usuario]))
        else:
            return datos.getItemPuntuacionMaxima(usuario)

    def calcularF(self, meanRewardPrevia, meanRewardPost):
        return (1 - math.exp(-np.absolute(meanRewardPost - meanRewardPrevia)/self.sigma)) / (1 + math.exp(-np.absolute(meanRewardPost - meanRewardPrevia)/self.sigma))

    def update(self, dataset, usuario, result, acierto):
        meanRewardPrevia = 1 / dataset.getIntentosItem(result)
        if acierto == True:
            dataset.items[result][0] += 1
        meanRewardPost = 1 / dataset.getIntentosItem(result)

        #Actualizar Epsilon
        self.epsilon = (self.delta * self.calcularF(meanRewardPrevia, meanRewardPost)) + ((1 - self.delta) * self.epsilon)


class EpsilonFirstGreedyPolicy(estrategia):

    def __init__(self, iterations):
        self.contador = 0
        self.iterations = iterations
        self.epsilon = 0.1
        self.title = "EFirstGreedy iterations = "+str(self.iterations)
        self.pathBase = "ResultadosTFM/EFirstGreedy/"+str(self.iterations)

    def choose(self, datos, usuario, epoca):
        EList.append(self.epsilon)
        self.contador += 1
        if self.contador < self.iterations:
            return random.choice(list(datos.porRecomendar[usuario]))
        else:
            if np.random.random() < self.epsilon:
                return random.choice(list(datos.porRecomendar[usuario]))
            else:
                return datos.getItemPuntuacionMaxima(usuario)


class DynamicThompsonSamplingPolicy(estrategia):

    def __init__(self, dataset, a, b, k, C):
        self.alfaCero = a
        self.betaCero = b
        self.k = k
        self.C = C
        self.betaDist = {}
        self.alfaI = {}
        self.betaI = {}
        self.dataset = dataset
        for i in self.dataset.items.keys():
            self.alfaI[i] = 0
            self.betaI[i] = 0
        self.contador = k
        self.title = "DynamicThompson "+ALPHA+" = "+str(a)+" "+BETA+" = "+str(b)+" k = "+str(k)+" C = "+str(C)
        self.pathBase = "ResultadosTFM/DynamicThompson/"+str(a)+"-"+str(b)+"-"+str(k)+"-"+str(C)

    def reCalcular(self):
        for k in self.dataset.items.keys():
            self.betaDist[k] = random.betavariate(
                self.alfaI[k] + self.alfaCero, self.betaI[k] + self.betaCero)
        return

    def choose(self, datos, usuario, epoca):
        if self.contador == self.k:
            self.contador = 0
            self.reCalcular()
        self.contador += 1
        maximum = max(self.dataset.porRecomendar[usuario], key=self.betaDist.get)
        # Recalcular el item recomendado
        self.betaDist[maximum] = random.betavariate(
             self.alfaI[maximum] + self.alfaCero,  self.betaI[maximum] + self.betaCero)
        return maximum

    def update(self, dataset, usuario, result, acierto):
        if self.alfaI[result] + self.betaI[result] < self.C:
            if acierto == True:
                self.alfaI[result] += 1
                dataset.items[result][0] += 1
            else:
                self.betaI[result] += 1
        else:
            if acierto == True:
                dataset.items[result][0] += 1
                self.alfaI[result] = (self.alfaI[result] + 1) * (self.C/(self.C + 1))
                self.betaI[result] =(self.betaI[result] + 0) * (self.C/(self.C + 1))
            else:
                self.alfaI[result] = (self.alfaI[result] + 0) * (self.C/(self.C + 1))
                self.betaI[result] =(self.betaI[result] + 1) * (self.C/(self.C + 1))
        return


class AdaptiveUCBPolicy(estrategia):

    def __init__(self, datos):
        self.gamma = 1
        self.datos = datos
        self.epoca = 0
        self.primeraVuelta = list(datos.items.keys())
        self.title = "AdaptiveUCB "
        self.pathBase = "ResultadosTFM/AdaptiveUCB"

        self.aciertos = 1
        self.fallos = 1

    def choose(self, datos, usuario, epoca):
        EList.append(self.gamma)
        self.epoca = epoca
        if self.primeraVuelta:
            result = random.choice(self.primeraVuelta)
            self.primeraVuelta.remove(result)
            return result

        maximum = max(datos.porRecomendar[usuario], key=self.calcular)
        return maximum

    def calcular(self, k):
        if self.datos.items[k][1] == 0:
            return 0
        else:
            porcentajeAcierto = self.datos.getPorcentajeAcierto(k)
            exploration = math.sqrt(
                self.gamma * np.log(self.epoca) / self.datos.items[k][1])

            return porcentajeAcierto + exploration

    def update(self, dataset, usuario, result, acierto):
        if acierto == True:
            self.aciertos += 1
            dataset.items[result][0] += 1
        else:
            self.fallos += 1
        self.gamma = np.log(1 + self.fallos / self.aciertos)


class EXP3Policy(estrategia):

    def __init__(self, datos, gamma):
        self.gamma = gamma
        self.datos = datos
        self.pesos = {}
        for i in self.datos.items:
            self.pesos[i] = 1
        self.probas = {}
        self.title = "EXP3 "+GAMMA+" = "+str(gamma)
        self.pathBase = "ResultadosTFM/EXP3/"+str(gamma)

    def choose(self, datos, usuario, epoca):
        self.probas = {}
        itemsPorRecomendar = list({x : self.pesos[x] for x in datos.porRecomendar[usuario]}.values())
        norm = float(sum(itemsPorRecomendar))
        for i in datos.porRecomendar[usuario]:
            self.probas[i] = (1.0 - self.gamma) * (self.pesos[i] / norm) + (self.gamma / len(itemsPorRecomendar))
        
        return random.choices(list(self.probas.keys()), weights = list(self.probas.values()))[0]

    def update(self, dataset, usuario, result, acierto):
        estimatedReward = 0
        if acierto == True:
            estimatedReward = 1 / self.probas[result]
            dataset.items[result][0] += 1
        self.pesos[result] *= math.exp(estimatedReward * self.gamma / len(dataset.porRecomendar[usuario]))
        
