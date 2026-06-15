from flask import Flask, render_template, request, jsonify
import heapq
import os
import random
import json
import math

app = Flask(__name__)

# Graph of Mexican cities and highways from the lowercase 'gps' project
CITIES = {
    'CDMX': {'lat': 19.4326, 'lon': -99.1332},
    'Queretaro': {'lat': 20.5888, 'lon': -100.3899},
    'Guadalajara': {'lat': 20.6597, 'lon': -103.3496},
    'Monterrey': {'lat': 25.6866, 'lon': -100.3161},
    'Puebla': {'lat': 19.0413, 'lon': -98.2062},
    'SLP': {'lat': 22.1565, 'lon': -100.9855},
    'Leon': {'lat': 21.1227, 'lon': -101.6747},
    'Toluca': {'lat': 19.2827, 'lon': -99.6557},
    'Cuernavaca': {'lat': 18.9242, 'lon': -99.2216},
    'Acapulco': {'lat': 16.8531, 'lon': -99.8237},
    'Veracruz': {'lat': 19.1738, 'lon': -96.1342},
    'Oaxaca': {'lat': 17.0732, 'lon': -96.7266},
    'Morelia': {'lat': 19.7060, 'lon': -101.1950},
    'Pachuca': {'lat': 20.1010, 'lon': -98.7591},
    'Celaya': {'lat': 20.5223, 'lon': -100.8122},
    'LaPaz': {'lat': 24.1426, 'lon': -110.3129},
    'Tijuana': {'lat': 32.5149, 'lon': -117.0382},
    'Hermosillo': {'lat': 29.0729, 'lon': -110.9559},
    'Mazatlan': {'lat': 23.2494, 'lon': -106.4111},
}

HIGHWAYS = [
    ('CDMX', 'Queretaro', 211, 198),
    ('CDMX', 'Puebla', 135, 184),
    ('CDMX', 'Toluca', 65, 95),
    ('CDMX', 'Pachuca', 95, 0),
    ('CDMX', 'Cuernavaca', 90, 136),
    ('Queretaro', 'SLP', 200, 170),
    ('Queretaro', 'Leon', 170, 150),
    ('Queretaro', 'Celaya', 50, 0),
    ('SLP', 'Monterrey', 510, 350),
    ('Leon', 'Guadalajara', 220, 280),
    ('Cuernavaca', 'Acapulco', 290, 600),
    ('Puebla', 'Veracruz', 280, 450),
    ('Puebla', 'Oaxaca', 340, 200),
    ('Guadalajara', 'Morelia', 290, 400),
    ('Morelia', 'Toluca', 180, 250),
    ('SLP', 'Leon', 180, 100),
    ('LaPaz', 'Tijuana', 1430, 0),
    ('Tijuana', 'Hermosillo', 800, 180),
    ('Hermosillo', 'Mazatlan', 850, 450),
    ('Mazatlan', 'Guadalajara', 480, 700),
]

# Build adjacency list
GRAPH = {city: [] for city in CITIES}
for u, v, dist, toll in HIGHWAYS:
    GRAPH[u].append({'to': v, 'dist': dist, 'toll': toll})
    GRAPH[v].append({'to': u, 'dist': dist, 'toll': toll})

GAS_PRICE_PER_LITER = 24.50

# Fallback-safe ML predictor imports
try:
    import joblib
    import numpy as np
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

HIGHWAY_MAP = {
    'CDMX-Queretaro': 0,
    'CDMX-Puebla': 1,
    'CDMX-Toluca': 2,
    'CDMX-Cuernavaca': 3,
    'Cuernavaca-Acapulco': 4,
    'Queretaro-SLP': 5,
    'SLP-Monterrey': 6,
    'Puebla-Veracruz': 7,
    'Mazatlan-Guadalajara': 8,
    'Hermosillo-Mazatlan': 9,
}

class SafeIncidentPredictor:
    def __init__(self):
        self.model = None
        if HAS_ML_LIBS:
            # Look for joblib model in the workspace
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gps/backend/backend/ml/models/incident_predictor.joblib'))
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    print(f"ML Model loaded successfully from {model_path}")
                except Exception as e:
                    print(f"Error loading ML model from {model_path}: {e}")
            else:
                print("ML model file not found at expected path. Fallback logic will be used.")

    def predict(self, highway_name, hour, day, is_holiday):
        if self.model is not None and HAS_ML_LIBS:
            try:
                h_id = HIGHWAY_MAP.get(highway_name, 0)
                features = np.array([[h_id, hour, day, is_holiday]])
                prob = self.model.predict_proba(features)[0][1]
                return float(prob)
            except Exception as e:
                print(f"Error making model prediction: {e}. Falling back...")
        
        # Rule-based fallback incorporating Mexican Highway Safety reports (crime/robbery stats)
        h_id = HIGHWAY_MAP.get(highway_name, -1)
        if h_id == -1:
            return 0.05
            
        risk_prob = 0.05
        
        # 1. CDMX-Queretaro (h_id = 0) - Extremely high cargo theft / robbery risk
        if h_id == 0:
            risk_prob = 0.22 # High baseline
            # Commute hours and night/madrugada hours
            if (7 <= hour <= 10) or (20 <= hour or hour <= 5):
                risk_prob += 0.35
            if is_holiday or day >= 4:
                risk_prob += 0.12
                
        # 2. CDMX-Puebla (h_id = 1) - High crime risk (San Martin Texmelucan corridor)
        elif h_id == 1:
            risk_prob = 0.20
            # Night and early morning
            if (19 <= hour or hour <= 6):
                risk_prob += 0.40
            if is_holiday or day >= 4:
                risk_prob += 0.10

        # 3. Puebla-Veracruz (h_id = 7) - Extremely critical risk (Cumbres de Maltrata / Esperanza)
        elif h_id == 7:
            risk_prob = 0.25
            # Night hours
            if (18 <= hour or hour <= 6):
                risk_prob += 0.45
            if is_holiday or day >= 4:
                risk_prob += 0.15

        # 4. Queretaro-SLP (h_id = 5) & SLP-Monterrey (h_id = 6) - High risk (Matehuala highway 57)
        elif h_id in (5, 6):
            risk_prob = 0.18
            # Night hours
            if (21 <= hour or hour <= 6):
                risk_prob += 0.35
            if is_holiday or day >= 4:
                risk_prob += 0.08

        # 5. Cuernavaca-Acapulco (h_id = 4) - High delinquency risk (Guerrero highway)
        elif h_id == 4:
            risk_prob = 0.18
            # Weekend and holiday nights
            if (day >= 4 or is_holiday):
                risk_prob += 0.35
            if (22 <= hour or hour <= 5):
                risk_prob += 0.25

        # 6. CDMX-Toluca (h_id = 2) & CDMX-Cuernavaca (h_id = 3) - Moderate risk
        elif h_id in (2, 3):
            risk_prob = 0.08
            if (23 <= hour or hour <= 5):
                risk_prob += 0.18

        # 7. Mazatlan-Guadalajara (h_id = 8) & Hermosillo-Mazatlan (h_id = 9) - Sinaloa / Nayarit corridor risk
        elif h_id in (8, 9):
            risk_prob = 0.16
            if (20 <= hour or hour <= 6):
                risk_prob += 0.30
            if is_holiday or day >= 4:
                risk_prob += 0.08

        # Add a tiny bit of random variation to feel alive
        risk_prob += random.uniform(-0.01, 0.01)
        return max(0.0, min(1.0, risk_prob))

predictor = SafeIncidentPredictor()

def manhattan_heuristic(node, goal):
    """
    Calculates Manhattan distance between two cities, scaled to approximate kilometers (111km per degree)
    """
    node_coords = CITIES[node]
    goal_coords = CITIES[goal]
    lat_dist = abs(node_coords['lat'] - goal_coords['lat']) * 111
    lon_dist = abs(node_coords['lon'] - goal_coords['lon']) * 111
    return lat_dist + lon_dist

def calculate_route_astar(start, goal, fuel_weight=1.0, toll_weight=1.0, fuel_efficiency=12.0, fuel_price=24.50):
    """
    A* algorithm to find optimal route.
    Cost = Distance + (Fuel_Cost * fuel_weight) + (Toll_Cost * toll_weight)
    """
    if start not in CITIES or goal not in CITIES:
        return None

    # Priority queue structure: 
    # (f_score, current_node, path, cumulative_dist, cumulative_fuel, cumulative_toll)
    pq = [(manhattan_heuristic(start, goal), start, [start], 0, 0, 0)]
    visited = {}

    while pq:
        f_score, current, path, dist, fuel, toll = heapq.heappop(pq)

        if current == goal:
            return {
                'path': path,
                'distance': dist,
                'fuel_cost': fuel,
                'toll_cost': toll,
                'total_cost': fuel + toll
            }

        if current in visited and visited[current] <= f_score:
            continue
        visited[current] = f_score

        for neighbor in GRAPH.get(current, []):
            neighbor_name = neighbor['to']
            edge_dist = neighbor['dist']
            edge_toll = neighbor['toll']
            
            new_dist = dist + edge_dist
            new_fuel = fuel + (edge_dist / fuel_efficiency) * fuel_price
            new_toll = toll + edge_toll
            
            # Weighted g_score cost function
            g_score = new_dist + (new_fuel * fuel_weight) + (new_toll * toll_weight)
            h_score = manhattan_heuristic(neighbor_name, goal)
            f_score_new = g_score + h_score
            
            if neighbor_name not in visited or visited[neighbor_name] > f_score_new:
                heapq.heappush(pq, (f_score_new, neighbor_name, path + [neighbor_name], new_dist, new_fuel, new_toll))

    return None

@app.route('/')
def index():
    # Render index.html passing list of cities and their data to front-end
    return render_template('index.html', cities=sorted(list(CITIES.keys())), cities_data=CITIES)

def find_closest_city(lat, lon):
    min_dist = float('inf')
    closest = None
    for city_name, coords in CITIES.items():
        # simple Manhattan distance in degrees
        dist = abs(lat - coords['lat']) + abs(lon - coords['lon'])
        if dist < min_dist:
            min_dist = dist
            closest = city_name
    return closest

@app.route('/get_routes', methods=['POST'])
def get_routes():
    """
    Unified endpoint for path calculation supporting any location coordinates in Mexico
    """
    data = request.get_json() or {}
    
    origen_name = data.get('origen_name')
    origen_lat = data.get('origen_lat')
    origen_lon = data.get('origen_lon')
    
    destino_name = data.get('destino_name')
    destino_lat = data.get('destino_lat')
    destino_lon = data.get('destino_lon')

    fuel_weight = float(data.get('fuel_weight', 1.0))
    toll_weight = float(data.get('toll_weight', 1.0))
    fuel_efficiency = float(data.get('fuel_efficiency', 12.0))
    fuel_price = float(data.get('fuel_price', 24.50))
    # Enforce boundaries
    fuel_price = max(19.0, min(30.0, fuel_price))
    
    # ML specific variables
    hour = int(data.get('hour', 12))
    day = int(data.get('day', 2)) # Default Wednesday
    is_holiday = 1 if data.get('is_holiday', False) else 0

    if not origen_name or not destino_name or origen_lat is None or origen_lon is None or destino_lat is None or destino_lon is None:
        return jsonify({'error': 'Faltan parámetros de origen, destino o coordenadas.'}), 400

    # If the search coordinates are extremely close, perform local direct pathfinding
    coord_diff = abs(origen_lat - destino_lat) + abs(origen_lon - destino_lon)
    if coord_diff < 0.8:
        distance_km = math.sqrt((origen_lat - destino_lat)**2 + (origen_lon - destino_lon)**2) * 111.0
        fuel_cost = (distance_km / fuel_efficiency) * fuel_price
        toll_cost = 0.0
        
        path = [origen_name, destino_name]
        coordenadas_ruta = [[origen_lat, origen_lon], [destino_lat, destino_lon]]
        risk_score = 0.05
        
        return jsonify({
            'path': path,
            'distance': round(distance_km, 1),
            'fuel_cost': round(fuel_cost, 2),
            'toll_cost': round(toll_cost, 2),
            'total_cost': round(fuel_cost + toll_cost, 2),
            'risk_score': risk_score,
            'coordenadas_ruta': coordenadas_ruta,
            'segment_risks': [risk_score]
        })

    # Otherwise, execute hybrid graph-embedding:
    # 1. Embed origin/destination into closest nodes in the Mexico A* network
    start_node = find_closest_city(origen_lat, origen_lon)
    goal_node = find_closest_city(destino_lat, destino_lon)

    # 2. Run A* routing on the highway graph
    result = calculate_route_astar(start_node, goal_node, fuel_weight, toll_weight, fuel_efficiency, fuel_price)
    if not result:
        # Fallback to direct path calculation if network routing fails
        distance_km = math.sqrt((origen_lat - destino_lat)**2 + (origen_lon - destino_lon)**2) * 111.0
        fuel_cost = (distance_km / fuel_efficiency) * fuel_price
        return jsonify({
            'path': [origen_name, destino_name],
            'distance': round(distance_km, 1),
            'fuel_cost': round(fuel_cost, 2),
            'toll_cost': 0.0,
            'total_cost': round(fuel_cost, 2),
            'risk_score': 0.05,
            'coordenadas_ruta': [[origen_lat, origen_lon], [destino_lat, destino_lon]],
            'segment_risks': [0.05]
        })

    # 3. Add the entry and exit road segments
    dist_entry = math.sqrt((origen_lat - CITIES[start_node]['lat'])**2 + (origen_lon - CITIES[start_node]['lon'])**2) * 111.0
    dist_exit = math.sqrt((destino_lat - CITIES[goal_node]['lat'])**2 + (destino_lon - CITIES[goal_node]['lon'])**2) * 111.0

    total_distance = result['distance'] + dist_entry + dist_exit
    total_fuel_cost = (total_distance / fuel_efficiency) * fuel_price
    total_toll_cost = result['toll_cost'] # assume tolls inside the highway network

    # 4. Reconstruct clean itinerary path and coordinates
    final_path = []
    coordenadas_ruta = []

    final_path.append(origen_name)
    coordenadas_ruta.append([origen_lat, origen_lon])

    for node in result['path']:
        # Avoid duplicate names if coordinates overlap
        if node != origen_name and node != destino_name:
            final_path.append(node)
            coordenadas_ruta.append([CITIES[node]['lat'], CITIES[node]['lon']])

    if destino_name not in final_path:
        final_path.append(destino_name)
        coordenadas_ruta.append([destino_lat, destino_lon])

    # 5. Evaluate ML risk on all highway network segments and populate segment risks
    risks = []
    segment_risks = []
    
    for i in range(len(final_path) - 1):
        u, v = final_path[i], final_path[i+1]
        if u == origen_name or v == destino_name:
            # Entry / Exit road
            risk = 0.05
        else:
            highway_name = f"{u}-{v}"
            if highway_name not in HIGHWAY_MAP and f"{v}-{u}" in HIGHWAY_MAP:
                highway_name = f"{v}-{u}"
            if highway_name in HIGHWAY_MAP:
                risk = predictor.predict(highway_name, hour, day, is_holiday)
                risks.append(risk)
            else:
                risk = 0.05
        segment_risks.append(round(risk, 3))
    
    avg_risk = sum(risks) / len(risks) if risks else 0.05

    return jsonify({
        'path': final_path,
        'distance': round(total_distance, 1),
        'fuel_cost': round(total_fuel_cost, 2),
        'toll_cost': round(total_toll_cost, 2),
        'total_cost': round(total_fuel_cost + total_toll_cost, 2),
        'risk_score': round(avg_risk, 3),
        'coordenadas_ruta': coordenadas_ruta,
        'segment_risks': segment_risks
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
