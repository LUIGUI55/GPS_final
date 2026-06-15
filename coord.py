import math
import random
import requests

# Define las funciones de distancia y evaluación de ruta
def distancia(coord1, coord2):
    lat1 = coord1[0]
    lon1 = coord1[1]
    lat2 = coord2[0]
    lon2 = coord2[1]
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def evalua_ruta(ruta, coord):
    total = 0
    for i in range(0, len(ruta) - 1):
        ciudad1 = ruta[i]
        ciudad2 = ruta[i + 1]
        total += distancia(coord[ciudad1], coord[ciudad2])
    # Cierra el ciclo
    ciudad1 = ruta[-1]
    ciudad2 = ruta[0]
    total += distancia(coord[ciudad1], coord[ciudad2])
    return total

# Algoritmo de Simulated Annealing
def simulated_annealing(ruta, coord):
    T = 20
    T_MIN = 0
    V_enfriamiento = 100

    while T > T_MIN:
        dist_actual = evalua_ruta(ruta, coord)
        for _ in range(1, V_enfriamiento):
            # Intercambio aleatorio de dos ciudades
            i = random.randint(0, len(ruta) - 1)
            j = random.randint(0, len(ruta) - 1)
            ruta_tmp = ruta[:]
            ciudad_tmp = ruta_tmp[i]
            ruta_tmp[i] = ruta_tmp[j]
            ruta_tmp[j] = ciudad_tmp
            dist = evalua_ruta(ruta_tmp, coord)
            delta = dist_actual - dist
            # Criterio de aceptación
            if dist < dist_actual:
                ruta = ruta_tmp[:]
                break
            elif random.random() < math.exp(delta / T):
                ruta = ruta_tmp[:]
                break
        T -= 0.005

    return ruta

# Función para obtener coordenadas de la API de Google Maps
def obtener_coordenadas(ciudad, clave_api):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={ciudad}&key={clave_api}'
    respuesta = requests.get(url)
    datos = respuesta.json()
    if datos['status'] == 'OK':
        ubicacion = datos['results'][0]['geometry']['location']
        return (ubicacion['lat'], ubicacion['lng'])
    else:
        print(f"Error al obtener las coordenadas de {ciudad}: {datos['status']}")
        return None

# Función para obtener ciudades intermedias usando la API de Places
def obtener_ciudades_intermedias(clave_api, ciudad_inicio, ciudad_final):
    coord_inicio = obtener_coordenadas(ciudad_inicio, clave_api)
    coord_final = obtener_coordenadas(ciudad_final, clave_api)
    if not coord_inicio or not coord_final:
        return None

    # Calculamos 3 puntos intermedios a lo largo de la línea recta
    lat_diff = coord_final[0] - coord_inicio[0]
    lon_diff = coord_final[1] - coord_inicio[1]
    
    fractions = [0.25, 0.50, 0.75]
    ciudades = [ciudad_inicio]
    
    # Buscamos ciudades reales cerca de cada punto intermedio usando Google Places API
    for f in fractions:
        lat = coord_inicio[0] + lat_diff * f
        lon = coord_inicio[1] + lon_diff * f
        
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=40000&type=locality&key={clave_api}"
        try:
            respuesta = requests.get(url)
            datos = respuesta.json()
            if datos.get('status') == 'OK' and len(datos.get('results', [])) > 0:
                for result in datos['results']:
                    name = result['name']
                    if name not in ciudades and name != ciudad_final:
                        ciudades.append(name)
                        break
        except Exception as e:
            print(f"Error al buscar ciudades intermedias: {e}")

    if ciudad_final not in ciudades:
        ciudades.append(ciudad_final)
    return ciudades

# Define tu clave API
clave_api = 'AIzaSyATseXRXJunPtS9HPA9RtoKSLbHJpRXqR8'

# Ciudades de inicio y final
ciudad_inicio = 'Toluca'
ciudad_final = 'Aguascalientes'

# Obtener ciudades intermedias
ciudades = obtener_ciudades_intermedias(clave_api, ciudad_inicio, ciudad_final)

# Obtener coordenadas de las ciudades intermedias
coord = {}
for ciudad in ciudades:
    coordenadas = obtener_coordenadas(ciudad, clave_api)
    if coordenadas:
        coord[ciudad] = coordenadas

# Genera una ruta inicial aleatoria
ruta = list(coord.keys())
random.shuffle(ruta)

# Ejecuta el algoritmo de Simulated Annealing
mejor_ruta = simulated_annealing(ruta, coord)

# Evalúa la distancia total del mejor recorrido
distancia_total = evalua_ruta(mejor_ruta, coord)

# Muestra el resultado en la terminal
print("Mejor ruta:")
for ciudad in mejor_ruta:
    print(ciudad)
print("Distancia total:", distancia_total)
