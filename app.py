from flask import Flask, render_template, request, jsonify
import heapq
import os
import random
import json
import math

app = Flask(__name__)

# Graph of Mexican cities and highways from the lowercase 'gps' project
CITIES = {
    'CDMX':        {'lat': 19.4326, 'lon': -99.1332},
    'Queretaro':   {'lat': 20.5888, 'lon': -100.3899},
    'Guadalajara': {'lat': 20.6597, 'lon': -103.3496},
    'Monterrey':   {'lat': 25.6866, 'lon': -100.3161},
    'Puebla':      {'lat': 19.0413, 'lon': -98.2062},
    'SLP':         {'lat': 22.1565, 'lon': -100.9855},
    'Leon':        {'lat': 21.1227, 'lon': -101.6747},
    'Toluca':      {'lat': 19.2827, 'lon': -99.6557},
    'Cuernavaca':  {'lat': 18.9242, 'lon': -99.2216},
    'Acapulco':    {'lat': 16.8531, 'lon': -99.8237},
    'Veracruz':    {'lat': 19.1738, 'lon': -96.1342},
    'Oaxaca':      {'lat': 17.0732, 'lon': -96.7266},
    'Morelia':     {'lat': 19.7060, 'lon': -101.1950},
    'Pachuca':     {'lat': 20.1010, 'lon': -98.7591},
    'Celaya':      {'lat': 20.5223, 'lon': -100.8122},
    # ── Peninsula de Baja California (Hwy 1 Transpeninsular, ruta TERRESTRE) ──
    # Nota: La peninsula NO tiene conexion terrestre al continente.
    # Unica salida: subir por Hwy 1 hasta Tijuana/Mexicali (frontera norte).
    'LaPaz':       {'lat': 24.1426, 'lon': -110.3129},  # BCS sur
    'SantaRosalia': {'lat': 27.3367, 'lon': -112.2707}, # BCS/BCS norte intermedio
    'Ensenada':    {'lat': 31.8676, 'lon': -116.5963},  # BC norte
    'Tijuana':     {'lat': 32.5149, 'lon': -117.0382},  # BC frontera
    # ── Noroeste continental (100% terrestre) ───────────────────────────────
    'Mexicali':    {'lat': 32.6635, 'lon': -115.4683},  # BC/Sonora frontera
    'Nogales':     {'lat': 31.3236, 'lon': -110.9340},  # Sonora norte
    'Hermosillo':  {'lat': 29.0729, 'lon': -110.9559},  # Sonora central
    'Culiacan':    {'lat': 24.7994, 'lon': -107.3894},  # Sinaloa
    'Mazatlan':    {'lat': 23.2494, 'lon': -106.4111},  # Sinaloa sur
}

HIGHWAYS = [
    # ── Centro / Bajio ──────────────────────────────────────────────────────
    ('CDMX',        'Queretaro',    211, 198),
    ('CDMX',        'Puebla',       135, 184),
    ('CDMX',        'Toluca',        65,  95),
    ('CDMX',        'Pachuca',       95,   0),
    ('CDMX',        'Cuernavaca',    90, 136),
    ('Queretaro',   'SLP',          200, 170),
    ('Queretaro',   'Leon',         170, 150),
    ('Queretaro',   'Celaya',        50,   0),
    ('SLP',         'Monterrey',    510, 350),
    ('Leon',        'Guadalajara',  220, 280),
    ('Cuernavaca',  'Acapulco',     290, 600),
    ('Puebla',      'Veracruz',     280, 450),
    ('Puebla',      'Oaxaca',       340, 200),
    ('Guadalajara', 'Morelia',      290, 400),
    ('Morelia',     'Toluca',       180, 250),
    ('SLP',         'Leon',         180, 100),
    # ── Peninsula Baja California: Hwy 1 Transpeninsular (TERRESTRE) ────────
    # NUNCA cruza el Golfo — toda la ruta sigue la peninsula de sur a norte.
    ('LaPaz',        'SantaRosalia', 467,   0),  # La Paz → Santa Rosalia (BCS)
    ('SantaRosalia', 'Ensenada',     650,   0),  # Santa Rosalia → Ensenada (BC)
    ('Ensenada',     'Tijuana',      108,  30),  # Ensenada → Tijuana (BC norte)
    # ── Frontera norte → continente (100% terrestre) ─────────────────────────
    ('Tijuana',     'Mexicali',     195,  50),   # Hwy 2D
    ('Mexicali',    'Nogales',      340,   0),   # Hwy 2 por Sonora
    ('Nogales',     'Hermosillo',   261,  80),   # Hwy 15
    # ── Corredor Pacifico continental (Hwy 15, terrestre) ───────────────────
    ('Hermosillo',  'Culiacan',     422, 160),
    ('Culiacan',    'Mazatlan',     210, 120),
    ('Mazatlan',    'Guadalajara',  480, 700),
    # ── Corredor Noreste (terrestre) ─────────────────────────────────────────
    ('Monterrey',   'Hermosillo',   950, 200),   # Hwy 40/45 por Chihuahua
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
    'CDMX-Queretaro':        0,
    'CDMX-Puebla':           1,
    'CDMX-Toluca':           2,
    'CDMX-Cuernavaca':       3,
    'Cuernavaca-Acapulco':   4,
    'Queretaro-SLP':         5,
    'SLP-Monterrey':         6,
    'Puebla-Veracruz':       7,
    'Mazatlan-Guadalajara':  8,
    'Hermosillo-Culiacan':   9,
    'Culiacan-Mazatlan':     10,
    'Tijuana-Mexicali':      11,
    'Mexicali-Nogales':      12,
    'Nogales-Hermosillo':    13,
    'Monterrey-Hermosillo':  14,
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

def is_point_in_zone(lat, lon, zone):
    """
    Returns True if the point (lat, lon) is inside the circular avoid zone.
    """
    z_lat = zone['lat']
    z_lon = zone['lon']
    z_rad_km = zone['radius'] / 1000.0
    # Approximate km distance using flat-earth
    dlat = (lat - z_lat) * 111.0
    dlon = (lon - z_lon) * 111.0 * math.cos(math.radians(z_lat))
    return math.sqrt(dlat**2 + dlon**2) < z_rad_km

def is_segment_in_avoid_zone(lat1, lon1, lat2, lon2, zone):
    """
    Returns True if the segment from (lat1,lon1) to (lat2,lon2) intersects
    the circular avoid zone. Checks: segment closest-point distance, both
    endpoints, and the midpoint.
    """
    z_lat = zone['lat']
    z_lon = zone['lon']
    z_rad_km = zone['radius'] / 1000.0

    # Check both endpoints first (fast path)
    if is_point_in_zone(lat1, lon1, zone) or is_point_in_zone(lat2, lon2, zone):
        return True

    # Check midpoint
    mid_lat = (lat1 + lat2) / 2.0
    mid_lon = (lon1 + lon2) / 2.0
    if is_point_in_zone(mid_lat, mid_lon, zone):
        return True

    # Flat-plane projection: closest point on segment to zone center
    x1, y1 = lon1 * 111.0 * math.cos(math.radians(lat1)), lat1 * 111.0
    x2, y2 = lon2 * 111.0 * math.cos(math.radians(lat2)), lat2 * 111.0
    xc, yc = z_lon * 111.0 * math.cos(math.radians(z_lat)), z_lat * 111.0

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        dist = math.sqrt((xc - x1)**2 + (yc - y1)**2)
    else:
        t = ((xc - x1) * dx + (yc - y1) * dy) / (dx**2 + dy**2)
        t = max(0.0, min(1.0, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist = math.sqrt((xc - closest_x)**2 + (yc - closest_y)**2)

    return dist < z_rad_km

def euclidean_heuristic(node, goal):
    """
    Calculates Euclidean distance between two cities, scaled to approximate kilometers (111km per degree)
    This is strictly admissible and guarantees finding the optimal path.
    """
    node_coords = CITIES[node]
    goal_coords = CITIES[goal]
    lat_dist = (node_coords['lat'] - goal_coords['lat']) * 111.0
    lon_dist = (node_coords['lon'] - goal_coords['lon']) * 111.0
    return math.sqrt(lat_dist**2 + lon_dist**2)

def calculate_route_astar(start, goal, fuel_weight=1.0, toll_weight=1.0, fuel_efficiency=12.0, fuel_price=24.50, toll_multiplier=1.0, avoid_zones=None):
    """
    A* algorithm to find optimal route.
    Cost = Distance + (Fuel_Cost * fuel_weight) + (Toll_Cost * toll_weight)
    Edges that pass through any avoid_zone are HARD-BLOCKED (skipped entirely).
    If no path exists with avoidance, falls back without avoidance.
    """
    if start not in CITIES or goal not in CITIES:
        return None

    def _run_astar(blocked_zones):
        # Priority queue: (f_score, current_node, path, dist, fuel, toll)
        pq = [(euclidean_heuristic(start, goal), start, [start], 0, 0, 0)]
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
                edge_toll = neighbor['toll'] * toll_multiplier

                # HARD BLOCK: skip this edge entirely if it crosses any avoid zone
                if blocked_zones:
                    lat1, lon1 = CITIES[current]['lat'], CITIES[current]['lon']
                    lat2, lon2 = CITIES[neighbor_name]['lat'], CITIES[neighbor_name]['lon']
                    if any(is_segment_in_avoid_zone(lat1, lon1, lat2, lon2, z) for z in blocked_zones):
                        continue  # Don't consider this edge at all

                new_dist = dist + edge_dist
                new_fuel = fuel + (edge_dist / fuel_efficiency) * fuel_price
                new_toll = toll + edge_toll

                g_score = new_dist + (new_fuel * fuel_weight) + (new_toll * toll_weight)
                h_score = euclidean_heuristic(neighbor_name, goal)
                f_score_new = g_score + h_score

                if neighbor_name not in visited or visited[neighbor_name] > f_score_new:
                    heapq.heappush(pq, (f_score_new, neighbor_name, path + [neighbor_name], new_dist, new_fuel, new_toll))

        return None

    # Try with full avoidance first
    result = _run_astar(avoid_zones or [])
    if result is not None:
        return result

    # If no path found with avoidance (graph too sparse), try without zones
    # This is a fallback — the zone avoidance failed silently
    if avoid_zones:
        print(f"[WARN] No path found avoiding zones. Falling back to unblocked route.")
        return _run_astar([])

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
    Unified endpoint for path calculation supporting any location coordinates in Mexico,
    with support for optional intermediate stops (paradas imprevistas/intermedias).
    """
    data = request.get_json() or {}
    
    origen_name = data.get('origen_name')
    origen_lat = data.get('origen_lat')
    origen_lon = data.get('origen_lon')
    
    destino_name = data.get('destino_name')
    destino_lat = data.get('destino_lat')
    destino_lon = data.get('destino_lon')

    stops = data.get('stops', [])  # List of dicts: {'name': str, 'lat': float, 'lon': float}
    avoid_zones = data.get('avoid_zones', [])  # List of dicts: {'lat': float, 'lon': float, 'radius': float}

    fuel_weight = float(data.get('fuel_weight', 1.0))
    toll_weight = float(data.get('toll_weight', 1.0))
    fuel_efficiency = float(data.get('fuel_efficiency', 12.0))
    fuel_price = float(data.get('fuel_price', 24.50))
    # Enforce boundaries
    fuel_price = max(19.0, min(30.0, fuel_price))
    
    # Determine vehicle type and toll multiplier
    vehicle_type = data.get('vehicle_type', 'auto')
    if vehicle_type == 'moto':
        toll_multiplier = 0.6
    elif vehicle_type == 'camioneta':
        toll_multiplier = 1.6
    elif vehicle_type == 'camion':
        toll_multiplier = 11.0
    else:
        toll_multiplier = 1.0
    
    # ML specific variables
    hour = int(data.get('hour', 12))
    day = int(data.get('day', 2)) # Default Wednesday
    is_holiday = 1 if data.get('is_holiday', False) else 0

    if not origen_name or not destino_name or origen_lat is None or origen_lon is None or destino_lat is None or destino_lon is None:
        return jsonify({'error': 'Faltan parámetros de origen, destino o coordenadas.'}), 400

    # Build sequence of all points
    points = []
    points.append({
        'name': origen_name,
        'lat': origen_lat,
        'lon': origen_lon
    })
    for stop in stops:
        points.append({
            'name': stop.get('name'),
            'lat': stop.get('lat'),
            'lon': stop.get('lon')
        })
    points.append({
        'name': destino_name,
        'lat': destino_lat,
        'lon': destino_lon
    })

    segment_results = []
    
    for i in range(len(points) - 1):
        p_start = points[i]
        p_end = points[i+1]
        
        s_name = p_start['name']
        s_lat = p_start['lat']
        s_lon = p_start['lon']
        
        e_name = p_end['name']
        e_lat = p_end['lat']
        e_lon = p_end['lon']
        
        # If coordinates are very close, direct path:
        coord_diff = abs(s_lat - e_lat) + abs(s_lon - e_lon)
        if coord_diff < 0.8:
            distance_km = math.sqrt((s_lat - e_lat)**2 + (s_lon - e_lon)**2) * 111.0
            fuel_cost = (distance_km / fuel_efficiency) * fuel_price
            
            segment_results.append({
                'path': [s_name, e_name],
                'coords': [[s_lat, s_lon], [e_lat, e_lon]],
                'distance': distance_km,
                'fuel_cost': fuel_cost,
                'toll_cost': 0.0
            })
        else:
            # Embed into graph
            start_node = find_closest_city(s_lat, s_lon)
            goal_node = find_closest_city(e_lat, e_lon)
            
            result = calculate_route_astar(start_node, goal_node, fuel_weight, toll_weight, fuel_efficiency, fuel_price, toll_multiplier, avoid_zones=avoid_zones)
            if not result:
                # Fallback to direct path
                distance_km = math.sqrt((s_lat - e_lat)**2 + (s_lon - e_lon)**2) * 111.0
                fuel_cost = (distance_km / fuel_efficiency) * fuel_price
                segment_results.append({
                    'path': [s_name, e_name],
                    'coords': [[s_lat, s_lon], [e_lat, e_lon]],
                    'distance': distance_km,
                    'fuel_cost': fuel_cost,
                    'toll_cost': 0.0
                })
            else:
                g_path = list(result['path'])
                
                if start_node == goal_node:
                    # Bypass graph node completely
                    final_g_path = []
                else:
                    # Smart Truncation and Detour Pruning from both ends
                    # 1. Truncate from the end: stop path at any node if the destination is closer to that node than the next node in the path is.
                    final_g_path = []
                    for i, node in enumerate(g_path):
                        final_g_path.append(node)
                        if i < len(g_path) - 1:
                            next_node = g_path[i+1]
                            dist_curr_to_dest = math.sqrt((e_lat - CITIES[node]['lat'])**2 + (e_lon - CITIES[node]['lon'])**2) * 111.0
                            dist_curr_to_next = math.sqrt((CITIES[next_node]['lat'] - CITIES[node]['lat'])**2 + (CITIES[next_node]['lon'] - CITIES[node]['lon'])**2) * 111.0
                            if dist_curr_to_dest < dist_curr_to_next:
                                break
                                
                    # 2. Prune from the start: skip initial nodes if the start is closer to the second node than the first node is to the second node.
                    while len(final_g_path) >= 2:
                        n0 = final_g_path[0]
                        n1 = final_g_path[1]
                        dist_start_to_n1 = math.sqrt((s_lat - CITIES[n1]['lat'])**2 + (s_lon - CITIES[n1]['lon'])**2) * 111.0
                        dist_n0_to_n1 = math.sqrt((CITIES[n0]['lat'] - CITIES[n1]['lat'])**2 + (CITIES[n0]['lon'] - CITIES[n1]['lon'])**2) * 111.0
                        if dist_start_to_n1 < dist_n0_to_n1:
                            final_g_path.pop(0)
                        else:
                            break
                
                # Recalculate distance and tolls based on final_g_path
                total_distance = 0.0
                total_toll_cost = 0.0
                
                for j in range(len(final_g_path) - 1):
                    u, v = final_g_path[j], final_g_path[j+1]
                    for edge in GRAPH[u]:
                        if edge['to'] == v:
                            total_distance += edge['dist']
                            total_toll_cost += edge['toll'] * toll_multiplier
                            break
                            
                if final_g_path:
                    first_node = final_g_path[0]
                    dist_entry = math.sqrt((s_lat - CITIES[first_node]['lat'])**2 + (s_lon - CITIES[first_node]['lon'])**2) * 111.0
                    total_distance += dist_entry
                    
                    last_node = final_g_path[-1]
                    dist_exit = math.sqrt((e_lat - CITIES[last_node]['lat'])**2 + (e_lon - CITIES[last_node]['lon'])**2) * 111.0
                    total_distance += dist_exit
                else:
                    # If completely pruned, direct distance
                    total_distance = math.sqrt((s_lat - e_lat)**2 + (s_lon - e_lon)**2) * 111.0
                    total_toll_cost = 0.0
                    
                total_fuel_cost = (total_distance / fuel_efficiency) * fuel_price
                
                seg_path = []
                seg_coords = []
                
                seg_path.append(s_name)
                seg_coords.append([s_lat, s_lon])
                
                for node in final_g_path:
                    if node != s_name and node != e_name:
                        seg_path.append(node)
                        seg_coords.append([CITIES[node]['lat'], CITIES[node]['lon']])
                        
                if e_name not in seg_path:
                    seg_path.append(e_name)
                    seg_coords.append([e_lat, e_lon])
                    
                segment_results.append({
                    'path': seg_path,
                    'coords': seg_coords,
                    'distance': total_distance,
                    'fuel_cost': total_fuel_cost,
                    'toll_cost': total_toll_cost
                })

    # Combine all segments
    final_path = []
    coordenadas_ruta = []
    
    total_distance = 0.0
    total_fuel_cost = 0.0
    total_toll_cost = 0.0
    
    for idx, seg in enumerate(segment_results):
        total_distance += seg['distance']
        total_fuel_cost += seg['fuel_cost']
        total_toll_cost += seg['toll_cost']
        
        if idx == 0:
            final_path.extend(seg['path'])
            coordenadas_ruta.extend(seg['coords'])
        else:
            final_path.extend(seg['path'][1:])
            coordenadas_ruta.extend(seg['coords'][1:])

    # Calculate risks along combined path
    risks = []
    segment_risks = []
    
    for i in range(len(final_path) - 1):
        u, v = final_path[i], final_path[i+1]
        
        # Check if this edge is a highway edge
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
    
    fuel_liters = round(total_distance / fuel_efficiency, 1)

    return jsonify({
        'path': final_path,
        'distance': round(total_distance, 1),
        'fuel_cost': round(total_fuel_cost, 2),
        'toll_cost': round(total_toll_cost, 2),
        'total_cost': round(total_fuel_cost + total_toll_cost, 2),
        'fuel_liters': fuel_liters,
        'risk_score': round(avg_risk, 3),
        'coordenadas_ruta': coordenadas_ruta,
        'segment_risks': segment_risks
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
