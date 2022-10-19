import numpy as np
import random
random.seed()


class datos:

    featuresBase = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",  "Sci-Fi", "Thriller", "War", "Western"]

    def __init__(self, nombreFichero, tagsFichero):

        #{Usuario, {Item, Rating}}
        self.ratings = {}

        #{Item, [Aciertos, Intentos]}
        self.items = {}

        #{Usuario, set(Items)}
        self.yaRecomendado = {}

        #{Usuario, [Items]}
        self.porRecomendar = {}

        #{Item, [Features]}
        self.features = {}

        # {Item, [ "features" 0,1,0,1...]}
        self.featuresVector = {}

        # Ratings
        fichero = open(nombreFichero, 'r')
        lineas = fichero.read().splitlines()
        fichero.close()

        for linea in lineas:
            linea = linea.split("::")
            usuario = int(linea[0])
            item = int(linea[1])
            rating = int(linea[2])
            if item not in self.items:
                self.items[item] = [0, 0]
            if usuario in self.ratings:
                self.ratings[usuario][item] = rating
            else:
                diccionarioAux = {item: rating}
                self.ratings[usuario] = diccionarioAux
            if usuario not in self.yaRecomendado:
                self.yaRecomendado[usuario] = []

        # porRecomendar
        k = list(self.items.keys())
        for u in self.ratings.keys():
            random.shuffle(k)
            k2 = k.copy()
            self.porRecomendar[u] = k2

        # Features
        fichero = open(tagsFichero, 'r', encoding="utf-8")
        lineas = fichero.read().splitlines()
        fichero.close()

        for linea in lineas:
            linea = linea.split("::")
            item = int(linea[0])
            tags = linea[2]
            for t in tags.split("|"):
                if item not in self.features:
                    self.features[item] = []
                    self.featuresVector[item] = np.zeros(18)
                self.features[item].append(t)
                index = self.featuresBase.index(t)
                self.featuresVector[item][index] = 1

    def getItemPuntuacionMaxima(self, usuario):
        maximum = max(self.porRecomendar[usuario],
                      key=self.getPorcentajeAcierto)
        if self.items[maximum][1] == 0:
            maximum = random.choice(list(self.porRecomendar[usuario]))
        return maximum

    def getPorcentajeAcierto(self, k):
        if self.items[k][1] == 0:
            return 0
        else:
            return self.items[k][0]/self.items[k][1]

    def getIntentosItem(self, item):
        return self.items[item][1]

    def jaccard(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def ILD(self, usuario):
        result = []
        for i in self.yaRecomendado[usuario][-10:]:
            for j in self.yaRecomendado[usuario][-10:]:
                result.append(
                    1 - self.jaccard(self.features[i], self.features[j]))
        return result

    def Unexpectedness(self, usuario):
        result = []
        for i in self.yaRecomendado[usuario][-10:]:
            for j in self.yaRecomendado[usuario][:-10]:
                result.append(
                    1 - self.jaccard(self.features[i], self.features[j]))
        return result

    def getFeaturesVector(self, i):
        return self.featuresVector[i]

    def reset(self):
        for i in self.items:
            self.items[i] = [0, 0]

        for u in self.ratings:
            self.yaRecomendado[u] = []

        k = list(self.items.keys())
        for u in self.ratings.keys():
            random.shuffle(k)
            k2 = k.copy()
            self.porRecomendar[u] = k2
