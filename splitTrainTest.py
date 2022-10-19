
import random

fichero1 = open("Datos/ratingsML1M.csv", 'r')
lineas = fichero1.read().splitlines()
fichero1.close()
fichero2 = open("Datos/ratingsML1MTrain.csv", 'w')
fichero3 = open("Datos/ratingsML1MTest.csv", 'w')


ratings = {}

for linea in lineas:
    linea = linea.split("::")
    usuario = linea[0]
    item = linea[1]
    rating = linea[2]
    if usuario in ratings:
        ratings.get(usuario)[item] = rating
    else:
        diccionarioAux = {item : rating}
        ratings[usuario] = diccionarioAux

for k in ratings:
    for j in ratings.get(k):
        r = random.uniform(0, 1)
        if r < 0.2:
            fichero2.write(k+"::"+j+"::"+ratings.get(k).get(j)+"\n")
        else:
            fichero3.write(k+"::"+j+"::"+ratings.get(k).get(j)+"\n")

fichero2.close()
fichero3.close()
