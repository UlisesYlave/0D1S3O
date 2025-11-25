import math
import random
import heapq
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import requests

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# =============================================================================
# CONSTANTS - VERSIÓN CORREGIDA
# =============================================================================
MU_SUN = 1.32712440018e20
MU_EARTH = 3.986004418e14
AU = 149_597_870_700
P_WATER = 500.0                 # $500/kg - Agua de asteroides (PRECIO REALISTA)
ROI_MIN = 1                     # REDUCIDO: De 0.1 a 0.01 (1% ROI mínimo)
C_DEV = 5e8                     # Development cost ($)
C_LAUNCH = 2e7                  # Launch cost ($)
DV_BUDGET = 30000.0             # AUMENTADO: De 90000 a 120000 m/s
TIME_MAX = 3650.0               # AUMENTADO: De 1825 a 3650 días (10 años)
C_FUEL = 5000.0
M0_WET = 5000.0
ISP = 3000.0
G0 = 9.80665
MINING_TIME = 300               # AUMENTADO: De 100 a 300 días
MINING_RATE = 50000             # AUMENTADO: De 10000 a 50000 kg/día
MAX_ASTEROIDS_BEFORE_RETURN = 10

# =============================================================================
# NASA SBDB DATA LOADER
# =============================================================================
class NASADataLoader:
    SBDB_BASE = "https://ssd-api.jpl.nasa.gov/sbdb.api"

    def __init__(self, cache_dir: str = "./sbdb_cache", rate_limit_s: float = 0.12, debug: bool = False):
        self.cache_dir = cache_dir
        self.rate_limit_s = rate_limit_s
        self.debug = debug
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, obj_id: str) -> str:
        safe = obj_id.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def fetch_object(self, obj_id: str, force: bool = False) -> Dict[str, Any]:
        path = self._cache_path(obj_id)
        if not force and os.path.exists(path):
            #if self.debug:
                #print(f"[SBDB] Cache hit: {obj_id}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        if self.debug:
            print(f"[SBDB] Fetching: {obj_id}")
        
        try:
            # Request con parámetros físicos
            res = requests.get(
                self.SBDB_BASE, 
                params={
                    "des": obj_id, 
                    "phys-par": "1", 
                    "full-prec": "true"
                }, 
                timeout=30
            )
            
            if res.status_code != 200:
                if self.debug:
                    print(f"[SBDB] Failed {obj_id}: HTTP {res.status_code}")
                return None
                
            data = res.json()
            
            # Verificar si tiene datos físicos útiles
            if not self._has_useful_physical_data(data):
                if self.debug:
                    print(f"[SBDB] Skipping {obj_id}: insufficient physical data")
                return None
                
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            time.sleep(self.rate_limit_s)
            return data
            
        except Exception as e:
            if self.debug:
                print(f"[SBDB] Error fetching {obj_id}: {e}")
            return None

    def _has_useful_physical_data(self, data: Dict[str, Any]) -> bool:
        """Verificar si los datos físicos son útiles para nuestra simulación"""
        phys_par = data.get("phys_par", [])
        
        # Buscar al menos uno de estos parámetros
        useful_params = False
        for param in phys_par:
            name = param.get('name', '').lower()
            value = param.get('value')
            if value and name in ['diameter', 'extent', 'gm', 'density']:
                useful_params = True
                break
                
        return useful_params

    def fetch_many(self, ids: List[str], force: bool = False) -> List[Optional[Dict[str, Any]]]:
        results = []
        for obj_id in ids:
            try:
                result = self.fetch_object(obj_id, force)
                results.append(result)
            except Exception as e:
                if self.debug:
                    print(f"[SBDB] Failed {obj_id}: {e}")
                results.append(None)
        return results

# =============================================================================
# ASTEROID MODEL
# =============================================================================
@dataclass
class Asteroid:
    id: str
    a: float
    e: float
    i_deg: float
    om_deg: float
    w_deg: float
    ma_deg: float
    radius_m: float
    water_fraction: float
    spectral_type: str = "C"
    density_kg_m3: float = 2000.0  # NEW: Store actual density
    volume_m3: float = None  # NEW: Store actual volume if available
    is_lagrange_station: bool = False  # NEW: Flag for L2 station

    def mass_kg(self) -> float:
        """Calculate mass using actual volume if available, otherwise sphere approximation"""
        if self.is_lagrange_station:
            return 1e6  # Mass of the station (arbitrary)
        if self.volume_m3 is not None:
            return self.volume_m3 * self.density_kg_m3
        else:
            # Fallback to spherical approximation
            volume = (4/3) * math.pi * self.radius_m ** 3
            return volume * self.density_kg_m3

    def available_water_kg(self, mining_time_days: float = MINING_TIME) -> float:
        """Calculate available water based on actual asteroid mass - VERSIÓN MEJORADA"""
        if self.is_lagrange_station:
            return 0.0
        
        total_water = self.mass_kg() * self.water_fraction
        mining_rate = MINING_RATE
        mineable = mining_rate * mining_time_days
        
        # Si el asteroide tiene MUCHA agua, permitir minar más
        if total_water > 1e9:  # Si tiene más de 1 millón de toneladas
            # Minar un porcentaje significativo
            return min(total_water * 0.1, mineable * 5)  # Hasta 5 veces más
        else:
            return min(total_water, mineable)

    def orbital_position(self, date_days: float = 0.0) -> Tuple[float, float, float]:
        """Calculate orbital position CORREGIDO para animación"""
        if self.is_lagrange_station:
            # L2 point - posición más precisa
            earth_angle = (date_days / 365.25) * 2 * math.pi
            l2_distance = AU * 1.01  # 1% más allá de la Tierra (más preciso)
            x = l2_distance * math.cos(earth_angle)
            y = l2_distance * math.sin(earth_angle)
            z = 0.0
            return x, y, z
        
        # PARA ASTEROIDES REALES - usar elementos orbitales reales
        # Calcular anomalía media actualizada
        period_years = self.a ** 1.5  # Ley de Kepler
        period_days = period_years * 365.25
        mean_motion = 360.0 / period_days  # grados por día
        
        ma_current = (self.ma_deg + mean_motion * date_days) % 360.0
        
        # Convertir a anomalía excéntrica (aproximación)
        ea_current = ma_current  # Simplificación para órbitas casi circulares
        
        # Radio vector
        r = self.a * AU * (1 - self.e * math.cos(math.radians(ea_current)))
        
        # Posición en el plano orbital
        x_orb = r * math.cos(math.radians(ea_current))
        y_orb = r * math.sin(math.radians(ea_current))
        
        # Rotar a plano eclíptico usando elementos orbitales
        # (simplificado para visualización)
        cos_om = math.cos(math.radians(self.om_deg))
        sin_om = math.sin(math.radians(self.om_deg))
        cos_i = math.cos(math.radians(self.i_deg))
        sin_i = math.sin(math.radians(self.i_deg))
        
        x = x_orb * cos_om - y_orb * sin_om * cos_i
        y = x_orb * sin_om + y_orb * cos_om * cos_i
        z = y_orb * sin_i
        
        return x, y, z
    
def parse_physical_parameters(phys_par: List[Dict]) -> Dict[str, Any]:
    """
    Parse physical parameters from NASA SBDB API response.
    Returns dict with: diameter_km, extent_km, density_g_cm3, volume_km3, spectral_type
    """
    result = {
        'diameter_km': None,
        'extent_km': None,
        'density_g_cm3': None,
        'volume_km3': None,
        'spectral_type': None,
        'gm_km3_s2': None
    }
    
    for param in phys_par:
        name = param.get('name', '').lower()
        value_str = param.get('value')
        
        if value_str is None:
            continue
            
        try:
            if name == 'diameter':
                result['diameter_km'] = float(value_str)
                
            elif name == 'extent':
                # Parse extent like "34.4x11.2x11.2" km
                axes = [float(x) for x in value_str.split('x')]
                result['extent_km'] = axes
                
            elif name == 'density':
                result['density_g_cm3'] = float(value_str)
                
            elif name == 'gm':
                # GM in km^3/s^2
                result['gm_km3_s2'] = float(value_str)
                
            elif name in ['spec_t', 'spec_b']:
                # Spectral type
                if result['spectral_type'] is None:
                    result['spectral_type'] = value_str.strip().upper()
                    
        except (ValueError, AttributeError):
            continue
    
    return result

def calculate_volume_and_density(phys_params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate volume (m³) and density (kg/m³) from physical parameters.
    Priority: GM+density > GM+volume > density+volume > estimates
    Returns: (volume_m3, density_kg_m3)
    """
    volume_m3 = None
    density_kg_m3 = None
    mass_kg = None
    
    # Step 1: Calculate mass from GM if available
    if phys_params['gm_km3_s2'] is not None:
        gm = phys_params['gm_km3_s2'] * 1e9  # Convert km³/s² to m³/s²
        G = 6.67430e-11  # m³/(kg·s²)
        mass_kg = gm / G
    
    # Step 2: Get density if available
    if phys_params['density_g_cm3'] is not None:
        density_kg_m3 = phys_params['density_g_cm3'] * 1000.0  # g/cm³ to kg/m³
    
    # Step 3: Calculate volume from extent (triaxial ellipsoid) - most accurate shape
    if phys_params['extent_km'] is not None:
        axes = phys_params['extent_km']
        if len(axes) == 3:
            a, b, c = [x * 500.0 for x in axes]  # km to m (semi-axes, so divide diameter by 2)
            volume_m3 = (4/3) * math.pi * a * b * c
    
    # Step 4: Calculate volume from diameter (sphere) if no extent
    elif phys_params['diameter_km'] is not None:
        radius_m = phys_params['diameter_km'] * 500.0  # km to m (radius = diameter/2)
        volume_m3 = (4/3) * math.pi * radius_m ** 3
    
    # Step 5: Derive missing parameters from what we have
    # If we have mass and density, calculate volume
    if mass_kg is not None and density_kg_m3 is not None:
        volume_m3 = mass_kg / density_kg_m3
    
    # If we have mass and volume, calculate density
    elif mass_kg is not None and volume_m3 is not None:
        density_kg_m3 = mass_kg / volume_m3
    
    # If we have volume and density but no mass, calculate it
    elif volume_m3 is not None and density_kg_m3 is not None:
        mass_kg = volume_m3 * density_kg_m3
    
    # Step 6: Fill in missing values with estimates
    if density_kg_m3 is None:
        # Estimate from spectral type
        if phys_params['spectral_type']:
            density_kg_m3 = estimate_density_from_spectral_type(phys_params['spectral_type'])
        else:
            density_kg_m3 = 2000.0  # Default
    
    if volume_m3 is None and mass_kg is not None:
        # We have mass and density estimate, calculate volume
        volume_m3 = mass_kg / density_kg_m3
    
    return volume_m3, density_kg_m3

def estimate_density_from_spectral_type(spec_type: str) -> float:
    """Estimate density (kg/m³) based on spectral type"""
    spec_upper = spec_type.upper()[0] if spec_type else 'C'
    
    density_map = {
        'C': 1380.0,  # Carbonaceous
        'S': 2700.0,  # Silicaceous
        'M': 5320.0,  # Metallic
        'X': 2000.0,  # Unknown composition
        'V': 3500.0,  # Basaltic
        'D': 1200.0,  # Dark, possibly organic-rich
        'E': 3000.0,  # Enstatite
    }
    
    return density_map.get(spec_upper, 2000.0)

def estimate_water_fraction(spec_type: str) -> float:
    """Estimate water fraction based on spectral type"""
    spec_upper = spec_type.upper()[0] if spec_type else 'C'
    
    water_map = {
        'C': 0.10,   # Carbonaceous - high water
        'S': 0.02,   # Silicaceous - low water
        'M': 0.00,   # Metallic - no water
        'X': 0.05,   # Unknown - conservative
        'V': 0.01,   # Basaltic - very low
        'D': 0.15,   # Dark - potentially high water/organics
        'E': 0.01,   # Enstatite - low water
    }
    
    return water_map.get(spec_upper, 0.05)

# =============================================================================
# STATE REPRESENTATION
# =============================================================================
@dataclass
class State:
    seq: List[str]
    current: str
    delta_v_used: float
    m_water: float
    t_current: float
    date_abs: float
    current_position: Tuple[float, float, float] = None      # (x, y, z) en metros
    current_velocity: Tuple[float, float, float] = None      # (vx, vy, vz) en m/s  ← NUEVO
    current_epoch: float = None                             # Época orbital precisa ← NUEVO
    current_trajectory: Dict = None
    real_cost_accumulated: float = 0.0

    def __hash__(self):
        return hash((tuple(self.seq), self.current, round(self.delta_v_used, 2), 
                    round(self.m_water, 2), round(self.t_current, 2),
                    tuple(self.current_position) if self.current_position else (0,0,0),
                    tuple(self.current_velocity) if self.current_velocity else (0,0,0)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def clone_and_add(self, asteroid_id: str, dv_add: float, water_add: float, 
                     time_add: float, new_position: Tuple[float, float, float] = None,
                     new_velocity: Tuple[float, float, float] = None,
                     new_epoch: float = None,
                     cost_add: float = 0.0):
        return State(
            seq=self.seq + [asteroid_id],
            current=asteroid_id,
            delta_v_used=self.delta_v_used + dv_add,
            m_water=self.m_water + water_add,
            t_current=self.t_current + time_add,
            date_abs=self.date_abs + time_add,
            current_position=new_position,
            current_velocity=new_velocity,
            current_epoch=new_epoch,
            real_cost_accumulated=self.real_cost_accumulated + cost_add
        )

    def asteroids_visited(self) -> int:
        """Count number of asteroids visited (excluding BASE and L2_STATION)"""
        return len([x for x in self.seq if x not in ["BASE", "L2_STATION"]])

# =============================================================================
# ENVIRONMENT
# =============================================================================
class AsteroidMiningEnvironment:
    def __init__(self, asteroids: Dict[str, Asteroid], dv_budget: float = DV_BUDGET, 
                 time_max: float = TIME_MAX, roi_min: float = ROI_MIN,
                 max_asteroids_before_return: int = MAX_ASTEROIDS_BEFORE_RETURN):
        self.asteroids = asteroids
        self.dv_budget = dv_budget
        self.time_max = time_max
        self.roi_min = roi_min
        self.max_asteroids_before_return = max_asteroids_before_return

    def initial_state(self) -> State:
        return State(["BASE"], "BASE", 0.0, 0.0, 0.0, 0.0)

    def actions(self, state: State) -> List[str]:
        available = []
        current_ast = self.asteroids[state.current]
        
        # NEW: Check if we've reached maximum asteroids and must return to L2
        asteroids_visited = state.asteroids_visited()
        if asteroids_visited >= self.max_asteroids_before_return and "L2_STATION" not in state.seq:
            # Must return to L2 station
            if "L2_STATION" not in state.seq:
                dv_est = self._estimate_dv(current_ast, self.asteroids["L2_STATION"], state.date_abs)
                if state.delta_v_used + dv_est <= self.dv_budget:
                    return ["L2_STATION"]
            return []
        
        for ast_id, asteroid in self.asteroids.items():
            # Skip conditions
            if ast_id in state.seq:  # Already visited
                continue
            if ast_id == "BASE":  # Don't return to Earth
                continue
            if asteroid.water_fraction <= 0 and not asteroid.is_lagrange_station:  # Skip dry asteroids
                continue
            if asteroid.is_lagrange_station and asteroids_visited < 1:  # Can't go to station without visiting asteroids first
                continue
                
            # Check delta-v constraint
            dv_est = self._estimate_dv(current_ast, asteroid, state.date_abs)
            if state.delta_v_used + dv_est <= self.dv_budget:
                available.append(ast_id)
                
        return available

    def result(self, state: State, action: str) -> State:
        ast_from = self.asteroids[state.current]
        ast_to = self.asteroids[action]
        
        # Usar el método existente que SÍ funciona
        dv_transfer, transfer_time = self._calculate_transfer_dv_with_tof(ast_from, ast_to, state.date_abs)
        
        # Calcular nueva posición basada en tiempo de transferencia
        new_position = ast_to.orbital_position(state.date_abs + transfer_time)
        
        # Velocidad simplificada (asumir velocidad orbital circular en destino)
        r_mag = math.sqrt(sum(x**2 for x in new_position))
        orbital_speed = math.sqrt(MU_SUN / r_mag)
        tangent_dir = [-new_position[1], new_position[0], 0]  # Dirección tangencial
        tangent_mag = math.sqrt(sum(x**2 for x in tangent_dir))
        if tangent_mag > 0:
            tangent_dir = [x/tangent_mag for x in tangent_dir]
        else:
            tangent_dir = [1, 0, 0]
        new_velocity = [x * orbital_speed for x in tangent_dir]
        
        # Calcular costo REAL de este paso (sin retorno futuro)
        cost_this_step = self._calculate_real_step_cost(state, dv_transfer, transfer_time, ast_to)

        # ✅ AÑADIR COSTO DE RETORNO FINAL si vamos a L2
        if ast_to.is_lagrange_station and state.current != "BASE":
            # Calcular costo de retorno desde el asteroide anterior a L2
            cost_return_final = self._calculate_return_cost(state, ast_to)
            cost_this_step += cost_return_final

        # NEW: Only mine if it's an asteroid, not the station
        if ast_to.is_lagrange_station:
            mining_time = 0
            water_collected = 0
        else:
            mining_time = MINING_TIME
            water_collected = ast_to.available_water_kg(mining_time)
            
        total_time = transfer_time + mining_time
        return state.clone_and_add(action, dv_transfer, water_collected, total_time,
                                 new_position, new_velocity, state.date_abs + total_time,cost_this_step)

    def is_goal(self, state: State) -> bool:
        if len(state.seq) < 2 or state.delta_v_used > self.dv_budget or state.t_current > self.time_max:
            return False
        
        # NEW: Goal is now reaching L2 station with sufficient ROI
        if state.current != "L2_STATION":
            return False
            
        return self.calculate_roi(state) >= self.roi_min

    def calculate_roi(self, state: State) -> float:
        """CALCULO DE ROI CORREGIDO"""
        revenue = state.m_water * P_WATER  # $500/kg × kg de agua
        
        costs = self._total_cost(state)
        
        # ROI = (Revenue - Costs) / Costs
        if costs > 0:
            roi = (revenue - costs) / costs
        else:
            roi = 0.0
            
        return roi

    def _estimate_dv(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        """Estimación REALISTA de ΔV - VERSIÓN CORREGIDA"""
        # Para transfers entre asteroides reales, usar cálculo más pesimista
        if ast1.is_lagrange_station or ast2.is_lagrange_station:
            return self._estimate_lagrange_transfer_dv(ast1, ast2, date)
            
        # Calcular posiciones
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # ✅ AUMENTADO: Hohmann más realista
        dv_hohmann = self._hohmann_dv_positions(r1_mag, r2_mag)
        
        # ✅ AUMENTADO: Penalización por diferencia de inclinación (MUY COSTOSO)
        inc1_rad = math.radians(ast1.i_deg)
        inc2_rad = math.radians(ast2.i_deg)
        delta_inc = abs(inc2_rad - inc1_rad)
        
        v_avg = math.sqrt(MU_SUN / ((r1_mag + r2_mag)/2))
        dv_inc = 2 * v_avg * math.sin(delta_inc/2) if delta_inc > 0.01 else 0
        
        # ✅ AUMENTADO: Penalización por excentricidad
        e_penalty = (ast1.e + ast2.e) * 5000
        
        # ✅ AUMENTADO: Penalización por diferencia de argumento de perihelio
        w1_rad = math.radians(ast1.w_deg)
        w2_rad = math.radians(ast2.w_deg)
        delta_w = abs(w2_rad - w1_rad) % (2 * math.pi)
        w_penalty = min(delta_w, 2*math.pi - delta_w) * 1000
        
        # ✅ NUEVO: Penalización por distancia angular grande
        dot_product = sum(r1[i] * r2[i] for i in range(3))
        cos_theta = dot_product / (r1_mag * r2_mag)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        angle_penalty = angle_rad * 2000  # Más costoso entre ángulos grandes
        
        dv_total = dv_hohmann + dv_inc + e_penalty + w_penalty + angle_penalty
        
        # ✅ AUMENTADO: Margen de seguridad del 30%
        return dv_total * 1.3

    def _estimate_lagrange_transfer_dv(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        """Estimate delta-v for transfers involving L2 station"""
        if ast1.is_lagrange_station and ast2.is_lagrange_station:
            return 0.0
            
        if ast1.is_lagrange_station:
            # From L2 to asteroid
            r_l2 = ast1.orbital_position(date)
            r_ast = ast2.orbital_position(date)
        else:
            # From asteroid to L2
            r_ast = ast1.orbital_position(date)
            r_l2 = ast2.orbital_position(date)
            
        r_ast_mag = math.sqrt(sum(x**2 for x in r_ast))
        r_l2_mag = math.sqrt(sum(x**2 for x in r_l2))
        
        # Hohmann transfer más realista
        dv_hohmann = self._hohmann_dv_positions(r_ast_mag, r_l2_mag)
        
        return dv_hohmann * 1.1  # Pequeño margen para L2

    def _calculate_transfer_dv_with_tof(self, ast1: Asteroid, ast2: Asteroid, date: float) -> tuple:
        """CÁLCULO MEJORADO de ΔV real y tiempo de vuelo"""
        # NEW: Special handling for L2 station
        if ast1.is_lagrange_station or ast2.is_lagrange_station:
            return self._calculate_lagrange_transfer_with_tof(ast1, ast2, date)
            
        # Calcular posiciones iniciales
        r1 = ast1.orbital_position(date)
        
        # Estimación inicial de tiempo de vuelo
        time_estimate = self._estimate_transfer_time_dated(ast1, ast2, date)
        
        # Iterar para convergencia (método simple de Lambert)
        for _ in range(3):
            r2 = ast2.orbital_position(date + time_estimate)
            time_estimate = self._calculate_tof(r1, r2)
        
        # Posiciones finales
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date + time_estimate)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Velocidades orbitales circulares
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        
        # Transferencia elíptica (Hohmann)
        a_transfer = (r1_mag + r2_mag) / 2.0
        v_at_r1 = math.sqrt(MU_SUN * (2.0/r1_mag - 1.0/a_transfer))
        v_at_r2 = math.sqrt(MU_SUN * (2.0/r2_mag - 1.0/a_transfer))
        
        # ΔV para la transferencia
        dv_departure = abs(v_at_r1 - v1)
        dv_arrival = abs(v2 - v_at_r2)
        
        # Considerar cambio de inclinación si es significativo
        inc1_rad = math.radians(ast1.i_deg)
        inc2_rad = math.radians(ast2.i_deg)
        delta_inc = abs(inc2_rad - inc1_rad)
        
        if delta_inc > 0.01:  # > 0.57 grados
            # Costo adicional por cambio de inclinación
            dv_inc = 2 * v1 * math.sin(delta_inc/2)
            dv_departure += dv_inc * 0.3  # Aplicar parte al inicio
        
        dv_total = dv_departure + dv_arrival
        
        return dv_total, time_estimate

    def _calculate_precise_transfer(self, current_position, current_velocity, target_asteroid, date):
        """
        Método simplificado para calcular transferencia precisa
        """
        if current_position is None:
            # Estado inicial desde BASE
            current_position = [AU, 0, 0]  # Posición de la Tierra
            current_velocity = [0, math.sqrt(MU_SUN/AU), 0]  # Velocidad orbital terrestre
        
        # Calcular posición objetivo
        time_estimate = self._estimate_transfer_time_dated(
            self.asteroids["BASE"], target_asteroid, date  # Usar BASE como referencia
        )
        
        target_position = target_asteroid.orbital_position(date + time_estimate)
        
        # Usar Lambert para cálculo más preciso
        try:
            dv, transfer_time, new_pos, new_vel = self._calculate_lambert_transfer(
                current_position, current_velocity, target_position, date, time_estimate
            )
            return dv, transfer_time, new_pos, new_vel
        except:
            # Fallback a Hohmann
            dv, transfer_time = self._calculate_transfer_dv_with_tof(
                self.asteroids["BASE"], target_asteroid, date
            )
            return dv, transfer_time, target_position, [0, 0, 0]

    def _calculate_lagrange_transfer_with_tof(self, ast1: Asteroid, ast2: Asteroid, date: float) -> tuple:
        """Calculate transfer involving L2 station"""
        if ast1.is_lagrange_station:
            r1 = ast1.orbital_position(date)
            r2 = ast2.orbital_position(date)
        else:
            r1 = ast1.orbital_position(date)
            r2 = ast2.orbital_position(date)
            
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Transferencia de Hohmann
        a_transfer = (r1_mag + r2_mag) / 2.0
        tof_seconds = math.pi * math.sqrt(a_transfer**3 / MU_SUN)
        tof_days = tof_seconds / (24 * 3600)
        
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        
        v_at_r1 = math.sqrt(MU_SUN * (2.0 / r1_mag - 1.0 / a_transfer))
        v_at_r2 = math.sqrt(MU_SUN * (2.0 / r2_mag - 1.0 / a_transfer))
        
        dv_total = abs(v_at_r1 - v1) + abs(v2 - v_at_r2)
        
        return dv_total, tof_days

    def _hohmann_dv_positions(self, r1_mag: float, r2_mag: float) -> float:
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        at = (r1_mag + r2_mag) / 2.0
        vt1 = math.sqrt(MU_SUN * (2.0 / r1_mag - 1.0 / at))
        vt2 = math.sqrt(MU_SUN * (2.0 / r2_mag - 1.0 / at))
        return abs(vt1 - v1) + abs(v2 - vt2)

    def _estimate_transfer_time_dated(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        a_transfer = (r1_mag + r2_mag) / 2.0
        tof_seconds = math.pi * math.sqrt(a_transfer**3 / MU_SUN)
        
        tof_days = tof_seconds / (24 * 3600)
        
        return max(30.0, min(400.0, tof_days))

    def _calculate_tof(self, r1: list, r2: list) -> float:
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        dot_product = sum(r1[i] * r2[i] for i in range(3))
        cos_theta = dot_product / (r1_mag * r2_mag)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        
        c = math.sqrt(r1_mag**2 + r2_mag**2 - 2*r1_mag*r2_mag*cos_theta)
        s = (r1_mag + r2_mag + c) / 2.0
        a_min = s / 2.0
        
        tof_seconds = math.pi * math.sqrt(a_min**3 / MU_SUN)
        
        return tof_seconds / (24 * 3600)

    def _calculate_real_step_cost(self, state: State, dv_step: float, time_step: float, target_asteroid: Asteroid) -> float:
        """Calcula costo REAL de un paso específico"""
        # 1. Propelente usado en ESTE tramo
        if dv_step > 0:
            mass_ratio = math.exp(dv_step / (ISP * G0))
            m_prop = M0_WET * (1.0 - 1.0/mass_ratio)
            cost_propellant = m_prop * C_FUEL
        else:
            cost_propellant = 0.0
        
        # 2. Operaciones por tiempo de ESTE tramo
        cost_operations = time_step * 100000.0
        
        # 3. Minería si llegamos a asteroide
        cost_mining = 5000000.0 if not target_asteroid.is_lagrange_station else 0
        
        # 4. CAPEX solo en primer paso desde BASE
        cost_capex = (C_DEV + C_LAUNCH) if state.current == "BASE" and len(state.seq) == 1 else 0
        
        return cost_propellant + cost_operations + cost_mining + cost_capex

    def _calculate_return_cost(self, from_state: State, to_l2_station: Asteroid) -> float:
        """Calcula costo de retorno FINAL desde asteroide a L2"""
        if from_state.current == "BASE" or from_state.current == "L2_STATION":
            return 0.0  # No hay retorno desde BASE o L2
        
        # Calcular ΔV para retornar a L2 desde el asteroide anterior
        from_asteroid = self.asteroids[from_state.current]
        dv_return = self._estimate_dv(from_asteroid, to_l2_station, from_state.date_abs)
        
        if dv_return > 0:
            mass_ratio_return = math.exp(-dv_return / (ISP * G0))
            m_prop_return = M0_WET * (1.0 - mass_ratio_return)
            return m_prop_return * C_FUEL
        else:
            return 0.0

    def _total_cost(self, state: State) -> float:
        """CALCULO DE COSTOS CORREGIDO Y ROBUSTO"""
        try:
            # 1. Costos FIJOS (CAPEX) - se pagan una vez
            cost_development = C_DEV
            cost_launch = C_LAUNCH
            
            # 2. Costos de PROPELENTE - basado en ΔV REAL usado
            if state.delta_v_used > 0:
                # Masa inicial SIN agua (solo la nave)
                mass_ratio = math.exp(-state.delta_v_used / (ISP * G0))
                m_prop = M0_WET * (1.0 - mass_ratio)
                cost_propellant = m_prop * C_FUEL
            else:
                cost_propellant = 0.0
            
            # 3. Costos de OPERACIONES (OPEX) - tiempo real
            # $100,000 por día (más realista para misión compleja)
            cost_operations = state.t_current * 10000.0
            
            # 4. Costos de MINERÍA - $5M por asteroide (equipos especializados)
            asteroids_mined = state.asteroids_visited()
            cost_mining = asteroids_mined * 1000000.0
            
            # 5. Costo de RETORNO solo si no estamos en L2 y tenemos agua
            cost_return = 0.0
            if state.current != "L2_STATION" and state.m_water > 0:
                # Calcular ΔV aproximado para retornar a L2
                current_ast = self.asteroids[state.current]
                l2_station = self.asteroids["L2_STATION"]
                
                dv_return_est = self._estimate_dv(current_ast, l2_station, state.date_abs)
                
                print(f"dv_return_est: {dv_return_est}")

                if dv_return_est > 0:
                    mass_ratio_return = math.exp(-dv_return_est / (ISP * G0))
                    m_prop_return = M0_WET * (1.0 - mass_ratio_return)
                    cost_return = m_prop_return * C_FUEL
            
            # COSTO TOTAL SUMATORIO
            total_cost = (cost_development + cost_launch + cost_propellant + 
                         cost_operations + cost_mining + cost_return)
            
            return total_cost
            
        except Exception as e:
            print(f"  [ERROR] Cost calculation failed: {e}")
            # Fallback seguro
            return C_DEV + C_LAUNCH + (state.delta_v_used * 1000)  # Estimación conservadora

    def _calculate_lambert_transfer(self, r1, v1, r2, departure_time, flight_time, 
                                  is_from_l2=False, is_to_l2=False):
        """
        CALCULA transferencia óptima usando método de Lambert CON SOPORTE L2
        """
        try:
            # Convertir a arrays numpy
            r1 = np.array(r1)
            v1 = np.array(v1) if v1 is not None else np.array([0, 0, 0])
            r2 = np.array(r2)
            
            # DETECTAR SI ES TRANSFERENCIA L2
            l2_distance = AU + 1.5e9  # Distancia L2 desde Sol
            r2_mag = np.linalg.norm(r2)
            is_to_l2 = abs(r2_mag - l2_distance) < 0.1 * AU  # Si está cerca de L2
            is_from_l2 = np.linalg.norm(r1) < 1.1 * AU  # Si viene de cerca de la Tierra/L2
            
            # MAGNITUDES Y GEOMETRÍA
            r1_mag = np.linalg.norm(r1)
            r2_mag = np.linalg.norm(r2)
            
            # ÁNGULO ENTRE VECTORES POSICIÓN
            dot_product = np.dot(r1, r2)
            cos_theta = dot_product / (r1_mag * r2_mag)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            
            # CONSIDERACIONES ESPECIALES PARA L2
            l2_assist_factor = 1.0
            if is_to_l2:
                # Transferencias a L2 son más eficientes (asistencia terrestre)
                l2_assist_factor = 0.7
                flight_time = min(flight_time, 180.0)  # Máximo 180 días a L2
            elif is_from_l2:
                # Desde L2/Tierra podemos optimizar
                l2_assist_factor = 0.8
            
            # CÁLCULO LAMBERT SIMPLIFICADO
            c = np.linalg.norm(r2 - r1)  # Cuerda
            s = (r1_mag + r2_mag + c) / 2.0  # Semi-perímetro
            
            # Semi-eje mayor mínimo
            a_min = s / 2.0
            
            # Tiempo de vuelo mínimo
            tof_min = math.pi * math.sqrt(a_min**3 / MU_SUN)
            
            # Ajustar tiempo de vuelo
            desired_tof_seconds = flight_time * 24 * 3600
            if desired_tof_seconds >= tof_min:
                a_transfer = a_min
            else:
                a_transfer = (desired_tof_seconds / math.pi)**(2/3) * MU_SUN**(1/3)
                a_transfer = max(a_transfer, a_min)
            
            # VELOCIDADES LAMBERT (aproximación simplificada)
            v1_circ = math.sqrt(MU_SUN / r1_mag)
            v2_circ = math.sqrt(MU_SUN / r2_mag)
            v1_transfer = math.sqrt(MU_SUN * (2.0 / r1_mag - 1.0 / a_transfer))
            v2_transfer = math.sqrt(MU_SUN * (2.0 / r2_mag - 1.0 / a_transfer))
            
            # ΔV CON FACTORES L2
            if is_to_l2:
                # Menor ΔV de llegada a L2
                dv_arrival = abs(v2_circ - v2_transfer) * 0.6
                dv_departure = abs(v1_transfer - v1_circ)
            elif is_from_l2:
                # Menor ΔV de salida desde Tierra/L2
                dv_departure = abs(v1_transfer - v1_circ) * 0.7
                dv_arrival = abs(v2_circ - v2_transfer)
            else:
                dv_departure = abs(v1_transfer - v1_circ)
                dv_arrival = abs(v2_circ - v2_transfer)
            
            dv_total = (dv_departure + dv_arrival) * l2_assist_factor
            
            # Nueva posición y velocidad
            new_position = r2
            # Velocidad orbital circular en destino
            tangent_dir = np.array([-r2[1], r2[0], 0])
            tangent_mag = np.linalg.norm(tangent_dir)
            if tangent_mag > 0:
                tangent_dir = tangent_dir / tangent_mag
            else:
                tangent_dir = np.array([1, 0, 0])
            new_velocity = tangent_dir * v2_circ
            
            return dv_total, flight_time, new_position.tolist(), new_velocity.tolist()
            
        except Exception as e:
            print(f"  [Warning] Lambert failed: {e}, using fallback")
            return self._calculate_hohmann_fallback(r1, r2, is_to_l2, is_from_l2)
    
    def _calculate_hohmann_fallback(self, r1, r2):
        """Fallback a transferencia de Hohmann"""
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        
        # Transferencia de Hohmann
        a_transfer = (r1_mag + r2_mag) / 2.0
        tof_seconds = math.pi * math.sqrt(a_transfer**3 / MU_SUN)
        tof_days = tof_seconds / (24 * 3600)
        
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        
        v_at_r1 = math.sqrt(MU_SUN * (2.0 / r1_mag - 1.0 / a_transfer))
        v_at_r2 = math.sqrt(MU_SUN * (2.0 / r2_mag - 1.0 / a_transfer))
        
        dv_total = abs(v_at_r1 - v1) + abs(v2 - v_at_r2)
        
        return dv_total, tof_days, r2.tolist(), [0, 0, 0]
# =============================================================================
# R* ALGORITHM - VERSIÓN MEJORADA CON DIRECCIÓN
# =============================================================================
@dataclass
class RStarNode:
    """Nodo para R* que incluye información de dirección/orientación"""
    state: State
    g: float = field(default=0.0)
    parent: Any = field(default=None)
    direction: Tuple[float, float, float] = field(default=None)  # Vector dirección
    is_avoid: bool = field(default=False)
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        return hash(self) == hash(other)

class RStar:
    def __init__(self, env: AsteroidMiningEnvironment, w: float = 2.0, 
                 K: int = 5, max_iterations: int = 2000,
                 verbose: bool = True):
        self.env = env
        self.w = w
        self.K = K
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.gamma = {}  # Diccionario de nodos explorados
        self.iterations = 0
        self.nodes_expanded = 0
        
    def solve(self) -> Tuple[Optional[State], float]:
        initial = self.env.initial_state()
        goal_roi = -1e9
        goal_state = None
        
        # Nodo inicial con dirección inicial (desde Tierra)
        initial_direction = [1.0, 0.0, 0.0]  # Dirección inicial arbitraria
        start_node = RStarNode(state=initial, g=0.0, direction=initial_direction)
        self.gamma[hash(initial)] = start_node
        
        OPEN = []
        heapq.heappush(OPEN, (self._compute_key(start_node), start_node))
        
        CLOSED = set()
        self.iterations = 0
        
        if self.verbose:
            print(f"[R*] Starting with w={self.w}, K={self.K}")
            print(f"[R*] Initial state: {initial.current}, ΔV={initial.delta_v_used}")

        while OPEN and self.iterations < self.max_iterations:
            self.iterations += 1
            
            _, current_node = heapq.heappop(OPEN)
            current_hash = hash(current_node.state)

            if current_hash in CLOSED:
                continue
            
            # Verificar si es estado objetivo
            if self.env.is_goal(current_node.state) and not self._is_initial_state(current_node.state):
                roi = self.env.calculate_roi(current_node.state)
                if roi > goal_roi:
                    goal_roi = roi
                    goal_state = current_node.state
                    if self.verbose:
                        asteroids_visited = current_node.state.asteroids_visited()
                        print(f"  [Iter {self.iterations}] ✓ GOAL! ROI={roi:.4f}, "
                              f"Asteroids={asteroids_visited}, Water={current_node.state.m_water/1000:.1f} tons")
                continue

            CLOSED.add(current_hash)
            self.nodes_expanded += 1

            # Generar sucesores considerando dirección
            successors = self._generate_directed_successors(current_node)

            if self.verbose and self.iterations % 100 == 0:
                open_count = len(OPEN)
                closed_count = len(CLOSED)
                best_water = goal_state.m_water/1000 if goal_state else 0
                print(f"  [Iter {self.iterations}] Open={open_count}, Closed={closed_count}, "
                      f"Best ROI={goal_roi:.4f}, Water={best_water:.1f}t")
            
            for succ_node in successors:
                succ_hash = hash(succ_node.state)
                
                if succ_hash in CLOSED:
                    continue
                
                if succ_hash not in self.gamma:
                    self.gamma[succ_hash] = succ_node
                else:
                    # Actualizar si encontramos un camino mejor
                    existing_node = self.gamma[succ_hash]
                    if succ_node.g < existing_node.g:
                        existing_node.g = succ_node.g
                        existing_node.parent = current_node
                        existing_node.direction = succ_node.direction
                
                heapq.heappush(OPEN, (self._compute_key(succ_node), succ_node))

        if self.verbose:
            print(f"\n[R*] Search complete:")
            print(f"  Iterations: {self.iterations}")
            print(f"  Nodes expanded: {self.nodes_expanded}")
            print(f"  Graph size: {len(self.gamma)}")
            
            if goal_state:
                asteroids_visited = goal_state.asteroids_visited()
                print(f"  Best solution: {asteroids_visited} asteroids, "
                      f"{goal_state.m_water/1000:.1f} tons water, ROI={goal_roi:.4f}")
        
        return goal_state, goal_roi if goal_state else (None, -1e9)
    
    def _is_initial_state(self, state: State) -> bool:
        """Verificar si es el estado inicial"""
        return (state.current == "BASE" and 
                state.delta_v_used == 0.0 and 
                len(state.seq) == 1)
    
    def _compute_key(self, node: RStarNode) -> float:
        """Calcular clave de prioridad para cola"""
        h = self._heuristic(node.state)
        return node.g + self.w * h
    
    def _heuristic(self, state: State) -> float:
        """Heurística mejorada considerando dirección"""
        
        if state.current == "L2_STATION" and state.m_water > 0:
            return 0.0
        
        # Asteroides no visitados
        unvisited = [ast for aid, ast in self.env.asteroids.items()
                     if aid not in state.seq and aid not in ["BASE", "L2_STATION"] and ast.water_fraction > 0]
        
        if not unvisited:
            if state.current != "L2_STATION":
                current_ast = self.env.asteroids[state.current]
                l2_station = self.env.asteroids["L2_STATION"]
                dv_to_l2 = self.env._estimate_dv(current_ast, l2_station, state.date_abs)
                return dv_to_l2 * C_FUEL / (ISP * G0) / (C_DEV + C_LAUNCH)
            return 0.0
        
        # Potencial de agua reducido por distancia/dirección
        potential_water = sum(ast.available_water_kg() for ast in unvisited)
        potential_revenue = potential_water * P_WATER
        
        # Costo estimado
        avg_dv_per_asteroid = 5000.0
        potential_dv = len(unvisited) * avg_dv_per_asteroid
        potential_cost = potential_dv * C_FUEL / (ISP * G0)
        
        # Penalización por no estar en L2 si tenemos agua
        l2_penalty = 500000 if state.current != "L2_STATION" and state.m_water > 0 else 0
        
        heuristic_value = max(0.0, -(potential_revenue - potential_cost) / (C_DEV + C_LAUNCH) + l2_penalty)
        
        return heuristic_value
    
    def _generate_directed_successors(self, node: RStarNode) -> List[RStarNode]:
        """Generar sucesores considerando la dirección actual"""
        current_state = node.state
        current_direction = node.direction
        available_actions = self.env.actions(current_state)
        
        if not available_actions:
            return []
        
        # Calcular scores considerando dirección
        scored_actions = []
        
        for action in available_actions:
            score = self._score_action_with_direction(current_state, current_direction, action)
            scored_actions.append((score, action))
        
        # Ordenar por score (mayor primero) y tomar las mejores K
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        successors = []
        
        for score, action in scored_actions[:self.K]:
            try:
                succ_state = self.env.result(current_state, action)
                
                # Calcular nueva dirección basada en el movimiento
                new_direction = self._calculate_new_direction(node, succ_state, action)
                
                # Calcular costo acumulado
                edge_cost = self._estimate_edge_cost(current_state, succ_state)
                new_g = node.g + edge_cost
                
                succ_node = RStarNode(
                    state=succ_state,
                    g=new_g,
                    parent=node,
                    direction=new_direction
                )
                
                successors.append(succ_node)
                
            except Exception as e:
                if self.verbose:
                    print(f"  [Warning] Error generating successor for {action}: {e}")
                continue
        
        return successors

    def _score_action_with_direction(self, current_state: State, 
                                   current_direction: Tuple[float, float, float], 
                                   action: str) -> float:
        """Calcular score considerando la dirección actual"""
        
        # Acción de retorno a L2 - alta prioridad en condiciones específicas
        if action == "L2_STATION":
            asteroids_visited = current_state.asteroids_visited()
            if asteroids_visited >= self.env.max_asteroids_before_return or current_state.m_water > 50000:
                return 1000 + current_state.m_water / 1000
            else:
                return 100
        
        # Acción a asteroide
        asteroid = self.env.asteroids[action]
        current_ast = self.env.asteroids[current_state.current]
        
        # Score base por agua potencial
        water_potential = asteroid.available_water_kg()
        water_score = water_potential * P_WATER / 1000000
        
        # Penalización por delta-V estimado
        dv_est = self.env._estimate_dv(current_ast, asteroid, current_state.date_abs)
        dv_penalty = dv_est / 1000
        
        # BONUS/MALUS POR DIRECCIÓN - NUEVO
        direction_bonus = self._calculate_direction_bonus(current_state, current_direction, action)
        
        # Bonus por estar cerca del límite de asteroides
        asteroids_visited = current_state.asteroids_visited()
        asteroid_bonus = 200 if asteroids_visited < self.env.max_asteroids_before_return - 1 else 0
        
        total_score = water_score - dv_penalty + direction_bonus + asteroid_bonus
        
        return max(0, total_score)

    def _calculate_direction_bonus(self, current_state: State,
                                 current_direction: Tuple[float, float, float],
                                 action: str) -> float:
        """Calcular bonus/malus basado en la dirección del movimiento"""
        
        if action == "L2_STATION":
            return 0.0  # No aplica para L2
            
        current_ast = self.env.asteroids[current_state.current]
        target_ast = self.env.asteroids[action]
        
        # Calcular vector hacia el objetivo
        current_pos = np.array(current_ast.orbital_position(current_state.date_abs))
        target_pos = np.array(target_ast.orbital_position(current_state.date_abs))
        
        direction_to_target = target_pos - current_pos
        distance = np.linalg.norm(direction_to_target)
        
        if distance > 0:
            direction_to_target = direction_to_target / distance
            
            # Producto punto con dirección actual (coseno del ángulo)
            current_dir_np = np.array(current_direction)
            dot_product = np.dot(current_dir_np, direction_to_target)
            
            # Bonus por moverse en dirección similar
            # 1.0 = misma dirección, -1.0 = dirección opuesta
            direction_bonus = dot_product * 800  # Escalar el bonus
            
            # Bonus adicional por objetivos "cercanos" en la misma dirección
            if dot_product > 0.8:  # Ángulo < 37 grados
                direction_bonus += 500
            elif dot_product > 0.5:  # Ángulo < 60 grados  
                direction_bonus += 200
            elif dot_product < -0.5:  # Ángulo > 120 grados (cambio drástico)
                direction_bonus -= 1000
            elif dot_product < -0.2:  # Ángulo > 100 grados
                direction_bonus -= 500
            
            return direction_bonus
        
        return 0.0

    def _calculate_new_direction(self, from_node: RStarNode, to_state: State, action: str) -> Tuple[float, float, float]:
        """Calcular nueva dirección después del movimiento"""
        
        if action == "L2_STATION":
            # Para L2, mantener dirección similar
            return from_node.direction
            
        from_ast = self.env.asteroids[from_node.state.current]
        to_ast = self.env.asteroids[action]
        
        # Calcular vector de movimiento
        from_pos = np.array(from_ast.orbital_position(from_node.state.date_abs))
        to_pos = np.array(to_ast.orbital_position(to_state.date_abs))
        
        movement_vector = to_pos - from_pos
        distance = np.linalg.norm(movement_vector)
        
        if distance > 0:
            # Normalizar y suavizar con dirección anterior
            new_direction = movement_vector / distance
            old_direction = np.array(from_node.direction)
            
            # Combinar con dirección anterior (70% nueva, 30% anterior)
            blended_direction = 0.7 * new_direction + 0.3 * old_direction
            blended_direction = blended_direction / np.linalg.norm(blended_direction)
            
            return tuple(blended_direction.tolist())
        
        return from_node.direction  # Mantener dirección si no hay movimiento

    def _estimate_edge_cost(self, from_state: State, to_state: State) -> float:
        """Estimar costo de arista considerando dirección"""
        dv_diff = to_state.delta_v_used - from_state.delta_v_used
        time_diff = to_state.t_current - from_state.t_current
        
        dv_cost = dv_diff * C_FUEL / (ISP * G0)
        time_cost = time_diff * 1000
        
        return dv_cost + time_cost

    def get_search_stats(self) -> dict:
        """Obtener estadísticas de la búsqueda"""
        return {
            'iterations': self.iterations,
            'nodes_expanded': self.nodes_expanded,
            'graph_size': len(self.gamma)
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def generate_synthetic_asteroids(n: int = 15) -> Dict[str, Asteroid]:
    asteroids = {}
    for i in range(n):
        asteroids[f"A{i+1}"] = Asteroid(
            id=f"A{i+1}",
            a=random.uniform(1.05, 2.0),
            e=random.uniform(0.0, 0.15),
            i_deg=random.uniform(0.0, 10.0),
            om_deg=random.uniform(0, 360),
            w_deg=random.uniform(0, 360),
            ma_deg=random.uniform(0, 360),
            radius_m=random.uniform(200, 1200),
            water_fraction=random.uniform(0.05, 0.15),
        )
    return asteroids

def create_lagrange_station() -> Asteroid:
    """Create the L2 station asteroid object"""
    return Asteroid(
        id="L2_STATION",
        a=1.0,  # Same semi-major axis as Earth
        e=0.0,  # Circular orbit
        i_deg=0.0,  # In ecliptic plane
        om_deg=0.0,
        w_deg=0.0,
        ma_deg=0.0,
        radius_m=100.0,  # Station size
        water_fraction=0.0,  # No water to mine
        spectral_type="S",
        density_kg_m3=3000.0,
        is_lagrange_station=True  # NEW: Mark as station
    )

def print_cumulative_budget(env: AsteroidMiningEnvironment, asteroids: Dict[str, Asteroid], state: State):
    """Show cumulative ΔV, costs, and water collected at each step - VERSIÓN CORREGIDA"""
    cumulative_dv = 0.0
    cumulative_water = 0.0
    cumulative_cost = 0.0
    prev = "BASE"

    print("\n[INFO] Cumulative budget along the route:")
    print(f"{'Step':<5} {'Location':<12} {'ΔV_this_step':>12} {'ΔV_cum':>12} "
          f"{'Water_collected':>16} {'Cost_cum':>20} {'Asteroids':>10}")

    asteroids_visited = 0

    # Reconstruir la misión paso a paso
    temp_state = env.initial_state()

    for i, location_id in enumerate(state.seq[1:], 1):  # skip BASE, start from 1
        loc_from = asteroids[prev]
        loc_to = asteroids[location_id]

        location_name = "L2_STATION" if loc_to.is_lagrange_station else f"Ast-{location_id}"

        # Water collected if it's an asteroid
        if not loc_to.is_lagrange_station:
            water = loc_to.available_water_kg()
            cumulative_water += water
            asteroids_visited += 1
        else:
            water = 0

        # Calcular el estado REAL después de este paso
        next_state = env.result(temp_state, location_id)
        
        dv_this_step = next_state.delta_v_used - temp_state.delta_v_used  # ΔV REAL
        cumulative_dv = next_state.delta_v_used

        real_cost_cum = next_state.real_cost_accumulated
        
        print(f"{i:<5} {location_name:<12} {dv_this_step:>12,.0f} {cumulative_dv:>12,.0f} "
              f"{cumulative_water:>16,.0f} {real_cost_cum:>20,.0f} {asteroids_visited:>10}")

        temp_state = next_state
        prev = location_id
    
def get_asteroid_ids_rich_in_water() -> List[str]:
    """
    Retorna una lista de asteroides conocidos tipo C (carbonáceos) 
    que tienen alta probabilidad de contener agua
    """
    # Asteroides tipo C conocidos con buenos datos físicos
    c_type_asteroids = [
        # Asteroides grandes tipo C bien estudiados
        "1", "2", "4", "6", "7", "10", "11", "13", "14", "15", "16", "19", "20", 
        "24", "27", "29", "31", "39", "42", "45", "46", "49", "50", "51", "52",
        "54", "56", "59", "60", "63", "65", "66", "68", "69", "70", "74", "75",
        "76", "78", "79", "82", "84", "85", "86", "87", "88", "89", "90", "92",
        "93", "94", "95", "96", "97", "99", "100", "101", "102", "103", "104",
        "105", "106", "107", "108", "109", "110", "111", "112", "113", "114",
        "115", "116", "117", "118", "119", "120", "121", "122", "123", "124",
        "125", "126", "127", "128", "129", "130", "131", "132", "133", "134",
        "135", "136", "137", "138", "139", "140", "141", "142", "143", "144",
        "145", "146", "147", "148", "149", "150", "151", "152", "153", "154",
        "155", "156", "157", "158", "159", "160", "161", "162", "163", "164",
        "165", "166", "167", "168", "169", "170", "171", "172", "173", "174",
        "175", "176", "177", "178", "179", "180", "181", "182", "183", "184",
        "185", "186", "187", "188", "189", "190", "191", "192", "193", "194",
        "195", "196", "197", "198", "199", "200", "201", "202", "203", "204",
        "205", "206", "207", "208", "209", "210", "211", "212", "213", "214",
        "215", "216", "217", "218", "219", "220", "221", "222", "223", "224",
        "225", "226", "227", "228", "229", "230", "231", "232", "233", "234",
        "235", "236", "237", "238", "239", "240", "241", "242", "243", "244",
        "245", "246", "247", "248", "249", "250", "253", "254", "255", "256",
        "257", "258", "259", "260", "261", "262", "263", "264", "265", "266",
        "267", "268", "269", "270", "271", "272", "273", "274", "275", "276",
        "277", "278", "279", "280", "281", "282", "283", "284", "285", "286",
        "287", "288", "289", "290", "291", "292", "293", "294", "295", "296",
        "297", "298", "299", "300", "302", "303", "304", "305", "306", "307",
        "308", "309", "310", "311", "312", "313", "314", "315", "316", "317",
        "318", "319", "320", "321", "322", "323", "324", "325", "326", "327",
        "328", "329", "330", "331", "332", "333", "334", "335", "336", "337",
        "338", "339", "340", "341", "342", "343", "344", "345", "346", "347",
        "348", "349", "350", "351", "352", "353", "354", "355", "356", "357",
        "358", "359", "360", "361", "362", "363", "364", "365", "366", "367",
        "368", "369", "370", "371", "372", "373", "374", "375", "376", "377",
        "378", "379", "380", "381", "382", "383", "384", "385", "386", "387",
        "388", "389", "390", "391", "392", "393", "394", "395", "396", "397",
        "398", "399", "400", "401", "402", "403", "404", "405", "406", "407",
        "408", "409", "410", "411", "412", "413", "414", "415", "416", "417",
        "418", "419", "420", "421", "422", "423", "424", "425", "426", "427",
        "428", "429", "430", "431", "432", "433", "434", "435", "436", "437",
        "438", "439", "440", "441", "442", "443", "444", "445", "446", "447",
        "448", "449", "450", "451", "452", "453", "454", "455", "456", "457",
        "458", "459", "460", "461", "462", "463", "464", "465", "466", "467",
        "468", "469", "470", "471", "472", "473", "474", "475", "476", "477",
        "478", "479", "480", "481", "482", "483", "484", "485", "486", "487",
        "488", "489", "490", "491", "492", "493", "494", "495", "496", "497",
        "498", "499", "500"
    ]
    
    # Seleccionar un subconjunto diverso para testing
    selected_ids = [
        "1", "2", "6", "7", "10", "13", "15", "19", 
        "24", "29", "31", "39", "45", "52", "65", "85",
        "253", "324"  # Incluir algunos conocidos
    ]
    
    return selected_ids

def load_asteroids_from_nasa(ids: List[str], debug: bool = False, force_refresh: bool = False) -> Dict[str, Asteroid]:
    """Load asteroids from NASA SBDB API - VERSIÓN MEJORADA con filtrado"""
    loader = NASADataLoader(debug=debug)
    records = loader.fetch_many(ids, force=force_refresh)
    
    asteroids = {}
    successful_loads = 0
    
    for idx, rec in enumerate(records):
        if rec is None:
            continue

        obj_info = rec.get("object", {})
        asteroid_id = obj_info.get("des", f"A{idx+1}")
        
        # ----------------------------
        # Orbital elements
        # ----------------------------
        a, e, i_deg, om_deg, w_deg, ma_deg = 2.5, 0.1, 10.0, 0.0, 0.0, 0.0
        try:
            orb = rec.get("orbit", {})
            elements = orb.get("elements", [])
            for el in elements:
                name = el.get("name", "").lower()
                val = float(el.get("value", 0.0))
                if name == "a":
                    a = val
                elif name == "e":
                    e = val
                elif name == "i":
                    i_deg = val
                elif name == "om":
                    om_deg = val
                elif name == "w":
                    w_deg = val
                elif name == "ma":
                    ma_deg = val
        except Exception:
            if debug:
                print(f"  [WARNING] {asteroid_id}: Error parsing orbital elements")

        # ----------------------------
        # Physical parameters
        # ----------------------------
        phys_par = rec.get("phys_par", [])
        phys_params = parse_physical_parameters(phys_par)
        
        # Calculate volume and density
        volume_m3, density_kg_m3 = calculate_volume_and_density(phys_params)
        
        # Determine radius for position calculations
        if phys_params['diameter_km']:
            radius_m = phys_params['diameter_km'] * 500.0
        elif phys_params['extent_km']:
            radius_m = sum(phys_params['extent_km']) / (3 * 2) * 1000.0  # Average semi-axis
        else:
            if debug:
                print(f"  [SKIPPING] {asteroid_id}: No size data available")
            continue  # Saltar asteroides sin datos de tamaño
        
        # Determine spectral type and water fraction
        spectral_type = phys_params['spectral_type'] or 'C'
        water_fraction = estimate_water_fraction(spectral_type)
        
        if debug:
            mass_kg = (volume_m3 * density_kg_m3) if volume_m3 else 0
            water_kg = mass_kg * water_fraction
            
            # Format diameter safely
            if phys_params['diameter_km'] is not None:
                diam_str = f"{phys_params['diameter_km']:.1f}km"
            elif phys_params['extent_km'] is not None:
                diam_str = f"{phys_params['extent_km'][0]:.1f}x{phys_params['extent_km'][1]:.1f}x{phys_params['extent_km'][2]:.1f}km"
            else:
                diam_str = "unknown"
            
            # Show data sources used
            sources = []
            if phys_params['gm_km3_s2'] is not None:
                sources.append("GM")
            if phys_params['density_g_cm3'] is not None:
                sources.append("ρ")
            if phys_params['extent_km'] is not None:
                sources.append("extent")
            elif phys_params['diameter_km'] is not None:
                sources.append("diam")
            source_str = "+".join(sources) if sources else "estimated"
            
            """print(f"  ✓ {asteroid_id}: size={diam_str}, "
                  f"density={density_kg_m3:.0f}kg/m³, "
                  f"mass={mass_kg/1e12:.2f}×10¹²kg, "
                  f"water≈{water_kg/1e9:.2f}×10⁹kg ({spectral_type}, {source_str})")"""

        asteroids[asteroid_id] = Asteroid(
            id=asteroid_id,
            a=a, e=e, i_deg=i_deg, om_deg=om_deg,
            w_deg=w_deg, ma_deg=ma_deg,
            radius_m=radius_m,
            water_fraction=water_fraction,
            spectral_type=spectral_type,
            density_kg_m3=density_kg_m3,
            volume_m3=volume_m3
        )
        successful_loads += 1
    
    # Add BASE station (Earth)
    asteroids["BASE"] = Asteroid(
        id="BASE", a=1.0, e=0.0, i_deg=0.0, om_deg=0.0,
        w_deg=0.0, ma_deg=0.0, radius_m=100.0, water_fraction=0.0, 
        spectral_type="S", density_kg_m3=3000.0
    )
    
    # Add L2 station
    asteroids["L2_STATION"] = create_lagrange_station()
    
    if debug:
        print(f"\n[INFO] Successfully loaded {successful_loads} asteroids with physical data")
        print(f"       + BASE + L2_STATION = {len(asteroids)} total objects")
        
        # Show statistics by spectral type
        spectral_counts = {}
        for ast in asteroids.values():
            if ast.id not in ["BASE", "L2_STATION"]:
                spec_type = ast.spectral_type[0] if ast.spectral_type else 'U'
                spectral_counts[spec_type] = spectral_counts.get(spec_type, 0) + 1
        
        if spectral_counts:
            print(f"       Spectral types: {', '.join([f'{k}:{v}' for k, v in spectral_counts.items()])}")
    
    return asteroids

# =============================================================================
# CONTINUOUS TRAJECTORY CALCULATOR
# =============================================================================
class ContinuousTrajectoryCalculator:
    """Calcula trayectorias continuas que EVITAN el Sol"""
    
    def __init__(self, env, asteroids):
        self.env = env
        self.asteroids = asteroids
        
    def calculate_segment_trajectory(self, from_ast_id: str, to_ast_id: str, 
                                   start_date: float, end_date: float, 
                                   num_points: int = 100) -> List[Tuple[float, float]]:
        """Calcula trayectoria que evita el Sol entre dos asteroides"""
        from_ast = self.asteroids[from_ast_id]
        to_ast = self.asteroids[to_ast_id]
        
        # Calcular posiciones inicial y final
        start_pos = from_ast.orbital_position(start_date)
        end_pos = to_ast.orbital_position(end_date)
        
        # Generar trayectoria que EVITA el Sol
        trajectory = self._generate_solar_avoiding_trajectory(start_pos, end_pos, num_points)
        
        print(f"      Puntos generados: {len(trajectory)}, Distancia mínima al Sol: {self._min_distance_to_sun(trajectory)/AU:.2f} AU")
        return trajectory
    
    def _generate_solar_avoiding_trajectory(self, start_pos, end_pos, num_points):
        """
        Genera trayectoria que EVITA pasar cerca del Sol (0,0)
        """
        trajectory = []
        
        # Convertir a arrays numpy
        start_vec = np.array([start_pos[0], start_pos[1]])
        end_vec = np.array([end_pos[0], end_pos[1]])
        
        # Calcular si la línea recta pasa muy cerca del Sol
        min_distance = self._line_min_distance_to_sun(start_vec, end_vec)
        
        # Si pasa muy cerca del Sol, usar trayectoria elíptica
        if min_distance < AU * 0.3:  # Menos de 0.3 AU del Sol
            print("      ⚠️  Usando trayectoria elíptica para evitar el Sol")
            return self._elliptical_avoidance_trajectory(start_vec, end_vec, num_points)
        else:
            # Trayectoria normal con curvatura suave
            return self._smooth_curved_trajectory(start_vec, end_vec, num_points)
    
    def _line_min_distance_to_sun(self, start_vec, end_vec):
        """Calcula la distancia mínima al Sol en la línea recta"""
        sun_pos = np.array([0, 0])
        direction = end_vec - start_vec
        line_length = np.linalg.norm(direction)
        
        if line_length == 0:
            return np.linalg.norm(start_vec)
        
        direction_normalized = direction / line_length
        to_sun = sun_pos - start_vec
        projection = np.dot(to_sun, direction_normalized)
        
        if projection <= 0:
            closest_point = start_vec
        elif projection >= line_length:
            closest_point = end_vec
        else:
            closest_point = start_vec + direction_normalized * projection
        
        return np.linalg.norm(closest_point - sun_pos)
    
    def _elliptical_avoidance_trajectory(self, start_vec, end_vec, num_points):
        """
        Trayectoria ELÍPTICA que mantiene distancia segura del Sol
        """
        trajectory = []
        
        r1 = np.linalg.norm(start_vec)
        r2 = np.linalg.norm(end_vec)
        
        # Ángulos de los vectores posición
        theta1 = np.arctan2(start_vec[1], start_vec[0])
        theta2 = np.arctan2(end_vec[1], end_vec[0])
        
        # Asegurar dirección angular correcta
        delta_theta = theta2 - theta1
        if delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        elif delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        
        # Radio mínimo seguro (nunca más cerca de 0.4 AU del Sol)
        safe_min_radius = AU * 0.4
        
        for i in range(num_points + 1):
            t_frac = i / num_points
            theta = theta1 + delta_theta * t_frac
            
            # Interpolación base
            base_radius = r1 + (r2 - r1) * t_frac
            
            # Componente elíptica para alejarse del Sol en el punto medio
            elliptical_component = np.sin(t_frac * np.pi) * 0.4 * min(r1, r2)
            
            # Asegurar distancia mínima segura
            current_r = max(safe_min_radius, base_radius + elliptical_component)
            
            x = current_r * np.cos(theta)
            y = current_r * np.sin(theta)
            
            trajectory.append((x, y))
            
        return trajectory
    
    def _smooth_curved_trajectory(self, start_vec, end_vec, num_points):
        """
        Trayectoria suave con curvatura natural
        """
        trajectory = []
        
        direction = end_vec - start_vec
        distance = np.linalg.norm(direction)
        
        for i in range(num_points + 1):
            t_frac = i / num_points
            
            # Posición base
            base_pos = start_vec + direction * self._smooth_step(t_frac)
            
            # Añadir curvatura para evitar línea recta
            if distance > AU * 0.1:
                if np.linalg.norm(direction) > 0:
                    direction_normalized = direction / np.linalg.norm(direction)
                    perpendicular = np.array([direction_normalized[1], -direction_normalized[0]])
                    
                    curvature_magnitude = np.sin(t_frac * np.pi) * 0.15 * distance
                    curved_pos = base_pos + perpendicular * curvature_magnitude
                    
                    # Verificar que no esté muy cerca del Sol
                    if np.linalg.norm(curved_pos) > AU * 0.3:
                        trajectory.append((curved_pos[0], curved_pos[1]))
                        continue
            
            # Si está muy cerca del Sol, usar posición base
            trajectory.append((base_pos[0], base_pos[1]))
            
        return trajectory
    
    def _min_distance_to_sun(self, trajectory):
        """Calcula la distancia mínima al Sol en una trayectoria"""
        min_dist = float('inf')
        for point in trajectory:
            dist = np.linalg.norm(point)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def _smooth_step(self, t: float) -> float:
        """Función de suavizado para transiciones naturales"""
        return t * t * (3 - 2 * t)

# =============================================================================
# VISUALIZATION MODULE - VERSIÓN CORREGIDA CON TRAYECTORIAS CONTINUAS
# =============================================================================
class MissionVisualizer:
    def __init__(self, env, asteroids, solution_state):
        self.env = env
        self.asteroids = asteroids
        self.solution_state = solution_state
        self.route = solution_state.seq
        self.trajectory_calculator = ContinuousTrajectoryCalculator(env, asteroids)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Colores para la visualización
        self.colors = {
            'sun': '#FFD700',
            'earth': '#1E90FF', 
            'l2': '#00FF00',
            'asteroid': '#FF6B6B',
            'spacecraft': '#FFFFFF',
            'trajectory': '#FFA500',
            'orbit': '#444444'
        }
        
        # Almacenar trayectorias precalculadas
        self.segment_trajectories = {}
        
    def setup_plot(self):
        """Configurar el plot inicial"""
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
        # Sol en el centro
        sun = Circle((0, 0), AU * 0.1, color=self.colors['sun'], alpha=0.8)
        self.ax.add_patch(sun)
        
        # Órbita de la Tierra
        earth_orbit = Circle((0, 0), AU, fill=False, 
                           color=self.colors['orbit'], linestyle='--', alpha=0.3)
        self.ax.add_patch(earth_orbit)
        
        # Zona de peligro alrededor del Sol (0.4 AU)
        danger_zone = Circle((0, 0), AU * 0.4, fill=False, 
                           color='red', linestyle=':', alpha=0.5, linewidth=2)
        self.ax.add_patch(danger_zone)
        self.ax.text(0, AU * 0.45, 'Zona de peligro', color='red', 
                   ha='center', fontsize=8, alpha=0.7)
        
        # Configurar límites del plot
        max_distance = 3 * AU
        self.ax.set_xlim(-max_distance, max_distance)
        self.ax.set_ylim(-max_distance, max_distance)
        self.ax.set_aspect('equal')
        
        # Etiquetas y título
        self.ax.set_xlabel('Distancia X (m)', color='white')
        self.ax.set_ylabel('Distancia Y (m)', color='white')
        self.ax.set_title('Misión de Reconocimiento - Trayectorias que Evitan el Sol', 
                        color='white', fontsize=14, pad=20)
        
        # Grid
        self.ax.grid(True, alpha=0.2, color='white')
        
        # Texto de la ruta
        route_text = f'Ruta: {" → ".join(self.route)}'
        self.ax.text(0.02, 0.02, route_text, transform=self.ax.transAxes,
                   color='yellow', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    def precalculate_all_trajectories(self):
        """Precalcula TODAS las trayectorias SIN minería"""
        print("Precalculando trayectorias continuas (SIN minería)...")
        
        # Reconstruir timeline precisa SIN minería
        timeline = self._calculate_precise_timeline()
        
        for i, segment in enumerate(timeline):
            from_id = segment['from']
            to_id = segment['to']
            start_date = segment['start_date']
            end_date = segment['end_date']  # 🚫 SIN tiempo de minería
            
            print(f"  Calculando vuelo {i+1}: {from_id} → {to_id}")
            
            # Calcular trayectoria continua (solo vuelo)
            trajectory = self.trajectory_calculator.calculate_segment_trajectory(
                from_id, to_id, start_date, end_date, num_points=150
            )
            
            # 🚫 ALMACENAR SIN TIEMPOS DE MINERÍA
            self.segment_trajectories[i] = {
                'trajectory': trajectory,
                'from': from_id,
                'to': to_id,
                'start_date': start_date,
                'end_date': end_date  # 🚫 Fin directo, sin minería
            }
        
        print(f"Precálculo completado: {len(self.segment_trajectories)} vuelos continuos")
    
    def _calculate_precise_timeline(self):
        """Calcula timeline SIN minería"""
        timeline = []
        current_state = self.env.initial_state()
        
        for i in range(len(self.route) - 1):
            current_id = self.route[i]
            next_id = self.route[i + 1]
            
            try:
                # Calcular próximo estado (sin minería)
                next_state = self.env.result(current_state, next_id)
                
                # 🚫 CERO tiempo de minería
                transfer_time = next_state.t_current - current_state.t_current
                
                segment = {
                    'from': current_id,
                    'to': next_id,
                    'start_date': current_state.date_abs,
                    'transfer_time': max(10.0, transfer_time),
                    'mining_time': 0,  # 🚫 CERO minería
                    'end_date': current_state.date_abs + transfer_time  # 🚫 Fin directo
                }
                
                timeline.append(segment)
                current_state = next_state
                
            except Exception as e:
                print(f"Error calculando segmento {current_id}→{next_id}: {e}")
                # Segmento de respaldo SIN minería
                segment = {
                    'from': current_id,
                    'to': next_id,
                    'start_date': current_state.date_abs,
                    'transfer_time': 100.0,
                    'mining_time': 0,  # 🚫 CERO minería
                    'end_date': current_state.date_abs + 100.0  # 🚫 Fin directo
                }
                timeline.append(segment)
        
        return timeline
    
    def calculate_orbital_positions(self, date_days):
        """Calcular posiciones orbitales para una fecha específica"""
        positions = {}
        
        for ast_id, asteroid in self.asteroids.items():
            try:
                pos = asteroid.orbital_position(date_days)
                positions[ast_id] = (pos[0], pos[1])
            except Exception as e:
                positions[ast_id] = (AU * 1.5, 0.0)
                
        return positions
    
    def get_spacecraft_position(self, current_day):
        """Obtener posición de la nave - SOLO VUELO, SIN MINERÍA"""
        
        # Buscar en qué segmento de VUELO estamos
        for seg_index, seg_data in self.segment_trajectories.items():
            seg_start = seg_data['start_date']
            seg_end = seg_data['end_date']
            
            # 🚫 SOLO TRANSFERENCIA - NO HAY MINERÍA
            if seg_start <= current_day <= seg_end:
                trajectory = seg_data['trajectory']
                transfer_duration = seg_end - seg_start
                
                if transfer_duration > 0 and trajectory:
                    progress = (current_day - seg_start) / transfer_duration
                    point_index = int(progress * (len(trajectory) - 1))
                    point_index = max(0, min(len(trajectory) - 1, point_index))
                    return trajectory[point_index]
                else:
                    return trajectory[0] if trajectory else (0, 0)
        
        # Después de la misión, quedarse en el último destino
        if self.segment_trajectories:
            last_seg = list(self.segment_trajectories.values())[-1]
            if current_day > last_seg['end_date']:
                last_pos = self.asteroids[last_seg['to']].orbital_position(current_day)
                return (last_pos[0], last_pos[1])
        
        return self.calculate_orbital_positions(current_day)["BASE"]
    
    def animate_mission(self, save_path=None):
        """Crear animación de la misión completa - VERSIÓN CORREGIDA"""
        print("Iniciando animación con trayectorias continuas...")
        
        # Precalcular TODAS las trayectorias primero
        self.precalculate_all_trajectories()
        
        if not self.segment_trajectories:
            print("ERROR: No se pudieron calcular las trayectorias")
            return None
        
        # Configurar el plot
        self.setup_plot()
        
        # Calcular duración total
        last_segment = list(self.segment_trajectories.values())[-1]
        total_days = last_segment['end_date']
        print(f"Duración total de la misión: {total_days:.1f} días")
        
        # Elementos de la animación
        scat_asteroids = self.ax.scatter([], [], s=50, c=self.colors['asteroid'], 
                                       label='Asteroides', alpha=0.7)
        scat_earth = self.ax.scatter([], [], s=200, c=self.colors['earth'], 
                                   label='Tierra/L2', alpha=0.8)
        scat_spacecraft = self.ax.scatter([], [], s=100, c=self.colors['spacecraft'], 
                                        marker='^', label='Nave', zorder=5)
        
        # Línea de trayectoria COMPLETA
        full_trajectory_line, = self.ax.plot([], [], '-', color=self.colors['trajectory'], 
                                           linewidth=2, alpha=0.8, label='Trayectoria')
        
        # Línea de trayectoria ACTUAL (segmento en curso)
        current_trajectory_line, = self.ax.plot([], [], '-', color='cyan', 
                                              linewidth=3, alpha=0.9, label='Segmento Actual')

        # Texto informativo
        info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                verticalalignment='top', color='white', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Leyenda
        self.ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
                     labelcolor='white', fontsize=9)
        
        # Almacenar historial de posiciones para trayectoria completa
        full_trajectory_history = []
        current_segment_points = []

        def update(frame):
            nonlocal full_trajectory_history, current_segment_points
            
            total_frames = 400
            current_day = min(frame * total_days / total_frames, total_days * 1.05)
            
            # Calcular posiciones actuales de todos los cuerpos
            positions = self.calculate_orbital_positions(current_day)
            
            # Actualizar posiciones de cuerpos celestes
            asteroid_pos = []
            earth_l2_pos = []
            
            for ast_id, pos in positions.items():
                if ast_id in ["BASE", "L2_STATION"]:
                    earth_l2_pos.append([pos[0], pos[1]])
                elif ast_id in self.route:
                    asteroid_pos.append([pos[0], pos[1]])
            
            # Calcular posición ACTUAL de la nave
            sc_pos = self.get_spacecraft_position(current_day)
            spacecraft_pos = [sc_pos] if sc_pos else []
            
            # Actualizar historial de trayectoria COMPLETA
            if sc_pos:
                full_trajectory_history.append(sc_pos)
                if len(full_trajectory_history) > 1000:
                    full_trajectory_history = full_trajectory_history[-1000:]
            
            # Actualizar puntos del segmento ACTUAL
            current_segment_points = []
            current_segment_info = "En tránsito"
            sc_color = self.colors['spacecraft']
            
            for seg_index, seg_data in self.segment_trajectories.items():
                if seg_data['start_date'] <= current_day <= seg_data['end_date']:
                    # Estamos en este segmento de transferencia
                    current_segment_points = seg_data['trajectory']
                    current_segment_info = f"Tránsito: {seg_data['from']} → {seg_data['to']}"
                    break
            
            # Actualizar gráficos
            scat_asteroids.set_offsets(asteroid_pos)
            scat_earth.set_offsets(earth_l2_pos)
            scat_spacecraft.set_offsets(spacecraft_pos)
            scat_spacecraft.set_color(sc_color)
            
            # Actualizar líneas de trayectoria
            if len(full_trajectory_history) > 1:
                full_trajectory_line.set_data(*zip(*full_trajectory_history))
            
            if len(current_segment_points) > 1:
                current_trajectory_line.set_data(*zip(*current_segment_points))
            else:
                current_trajectory_line.set_data([], [])
            
            # Actualizar texto informativo
            days_from_start = current_day
            info_str = f'Día: {days_from_start:.1f}\n{current_segment_info}'
            
            # Mostrar progreso del segmento actual
            for seg_index, seg_data in self.segment_trajectories.items():
                if seg_data['start_date'] <= current_day <= seg_data['end_date']:
                    progress = (current_day - seg_data['start_date']) / (seg_data['end_date'] - seg_data['start_date'])
                    info_str += f'\nProgreso: {progress*100:.1f}%'
                    break
            
            info_text.set_text(info_str)
            
            return (scat_asteroids, scat_earth, scat_spacecraft, 
                   full_trajectory_line, current_trajectory_line, info_text)
        
        print("Generando animación...")
        anim = FuncAnimation(self.fig, update, frames=400, interval=50, blit=False, repeat=True)
        
        # Guardar o mostrar
        if save_path:
            print(f"Guardando animación en {save_path}...")
            try:
                anim.save(save_path, writer='pillow', fps=20, dpi=100)
                print("¡Animación guardada exitosamente!")
            except Exception as e:
                print(f"Error al guardar: {e}")
                plt.show()
        else:
            plt.show()
            
        return anim

# =============================================================================
# INTEGRACIÓN CON EL CÓDIGO EXISTENTE
# =============================================================================

def visualize_solution(env, asteroids, solution_state, save_path=None):
    """Función principal para visualizar la solución - VERSIÓN CORREGIDA"""
    print("=" * 70)
    print("VISUALIZACIÓN CON TRAYECTORIAS CONTINUAS Y REALISTAS")
    print("=" * 70)
    
    visualizer = MissionVisualizer(env, asteroids, solution_state)
    anim = visualizer.animate_mission(save_path)
    
    return anim

# =============================================================================
# A* ALGORITHM FOR COMPARISON - VERSIÓN CORREGIDA
# =============================================================================
@dataclass
class AStarNode:
    state: State
    g: float = field(default=0.0)  # Costo real acumulado
    h: float = field(default=0.0)  # Heurística
    parent: Any = field(default=None)
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    @property
    def f(self):
        return self.g + self.h

class AStar:
    def __init__(self, env: AsteroidMiningEnvironment, verbose: bool = True):
        self.env = env
        self.verbose = verbose
        self.nodes_expanded = 0
        self.iterations = 0
        
    def solve(self) -> Tuple[Optional[State], float]:
        initial = self.env.initial_state()
        start_node = AStarNode(state=initial, g=0.0, h=self._heuristic(initial))
        
        open_set = []
        heapq.heappush(open_set, (start_node.f, start_node))
        
        closed_set = set()
        best_roi = -1e9
        best_state = None
        
        max_iterations = 30000
        
        while open_set and self.iterations < max_iterations:
            self.iterations += 1
            _, current_node = heapq.heappop(open_set)
            
            if hash(current_node.state) in closed_set:
                continue
                
            closed_set.add(hash(current_node.state))
            self.nodes_expanded += 1
            
            # Check if goal state
            if self.env.is_goal(current_node.state):
                roi = self.env.calculate_roi(current_node.state)
                if roi > best_roi:
                    best_roi = roi
                    best_state = current_node.state
                    if self.verbose and self.iterations % 500 == 0:
                        print(f"  [A* Iter {self.iterations}] ✓ GOAL! ROI={roi:.4f}")
                continue
            
            # Generate successors
            for action in self.env.actions(current_node.state):
                try:
                    next_state = self.env.result(current_node.state, action)
                    
                    # Calculate cost and heuristic
                    edge_cost = self._calculate_edge_cost(current_node.state, next_state)
                    new_g = current_node.g + edge_cost
                    new_h = self._heuristic(next_state)
                    
                    successor_node = AStarNode(
                        state=next_state,
                        g=new_g,
                        h=new_h,
                        parent=current_node
                    )
                    
                    heapq.heappush(open_set, (successor_node.f, successor_node))
                    
                except Exception as e:
                    if self.verbose and self.iterations % 1000 == 0:
                        print(f"  [A* Warning] Error with action {action}: {e}")
                    continue
        
        if self.verbose:
            print(f"\n[A*] Search completed:")
            print(f"  Iterations: {self.iterations}")
            print(f"  Nodes expanded: {self.nodes_expanded}")
            print(f"  Open set size: {len(open_set)}")
            print(f"  Closed set size: {len(closed_set)}")
            if best_state:
                asteroids_visited = best_state.asteroids_visited()
                print(f"  Best ROI: {best_roi:.4f}, Asteroids: {asteroids_visited}")
            else:
                print(f"  No solution found")
        
        return best_state, best_roi if best_state else (None, -1e9)
    
    def _heuristic(self, state: State) -> float:
        """Misma heurística que R* para comparación justa"""
        if state.current == "L2_STATION" and state.m_water > 0:
            return 0.0
        
        # Asteroides no visitados
        unvisited = [ast for aid, ast in self.env.asteroids.items()
                    if aid not in state.seq and aid not in ["BASE", "L2_STATION"] and ast.water_fraction > 0]
        
        if not unvisited:
            if state.current != "L2_STATION":
                current_ast = self.env.asteroids[state.current]
                l2_station = self.env.asteroids["L2_STATION"]
                dv_to_l2 = self.env._estimate_dv(current_ast, l2_station, state.date_abs)
                return dv_to_l2 * 1000
            return 0.0
        
        # Potencial de agua restante
        potential_water = sum(ast.available_water_kg() for ast in unvisited[:3])
        potential_revenue = potential_water * P_WATER
        
        # Costo estimado
        avg_dv_per_asteroid = 5000.0
        potential_dv = len(unvisited) * avg_dv_per_asteroid
        potential_cost = potential_dv * C_FUEL / (ISP * G0)
        
        return max(0.0, -(potential_revenue - potential_cost) / (C_DEV + C_LAUNCH))
    
    def _calculate_edge_cost(self, from_state: State, to_state: State) -> float:
        """Costo de arista - mismo que R*"""
        dv_diff = to_state.delta_v_used - from_state.delta_v_used
        time_diff = to_state.t_current - from_state.t_current
        
        dv_cost = dv_diff * C_FUEL / (ISP * G0)
        time_cost = time_diff * 1000
        
        return dv_cost + time_cost

    def get_search_stats(self) -> dict:
        return {
            'iterations': self.iterations,
            'nodes_expanded': self.nodes_expanded
        }

# =============================================================================
# COMPARATIVE STUDY: R* vs A* - VERSIÓN COMPLETA CORREGIDA
# =============================================================================
def run_rstar_vs_astar_study():
    """
    Estudio comparativo sistemático entre R* y A* - CORREGIDO
    """
    
    print("\n" + "="*80)
    print("ESTUDIO COMPARATIVO: R* vs A*")
    print("="*80)
    
    # Configuraciones de prueba MÁS REALISTAS
    test_configs = [
        {
            'name': 'Caso Pequeño - 25k Presupuesto - Max 5 asteroides',
            'dv_budget': 25000,
            'time_max': 4000,
            'max_asteroids': 5,
            'asteroid_count': 5
        },
        {
            'name': 'Caso Medio - 30k Presupuesto - Max 8 asteroides', 
            'dv_budget': 30000,
            'time_max': 4000,
            'max_asteroids': 8,
            'asteroid_count': 8
        },
        {
            'name': 'Caso Grande - 40k Presupuesto - Max 12 asteroides',
            'dv_budget': 40000,
            'time_max': 4000,
            'max_asteroids': 12,
            'asteroid_count': 12
        },
        {
            'name': 'Caso Restrictivo - Recursos limitados',
            'dv_budget': 10000,
            'time_max': 4000, 
            'max_asteroids': 3,
            'asteroid_count': 3
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔧 CONFIGURACIÓN: {config['name']}")
        print(f"   Asteroides: {config['asteroid_count']}, ΔV: {config['dv_budget']} m/s")
        
        # Generar dataset de asteroides
        asteroid_ids = get_asteroid_ids_rich_in_water()
        asteroids = load_asteroids_from_nasa(asteroid_ids, debug=True, force_refresh=False)
        asteroids["BASE"] = Asteroid(
            id="BASE", a=1.0, e=0.0, i_deg=0.0,
            om_deg=0.0, w_deg=0.0, ma_deg=0.0,
            radius_m=100, water_fraction=0.0, spectral_type="S"
        )
        asteroids["L2_STATION"] = create_lagrange_station()
        
        env = AsteroidMiningEnvironment(
            asteroids=asteroids,
            dv_budget=config['dv_budget'],
            time_max=config['time_max'],
            max_asteroids_before_return=config['max_asteroids']
        )
        
        # EJECUCIÓN R*
        print("   🔄 Ejecutando R*...")
        start_time = time.time()
        try:
            solver_rstar = RStar(env, w=5, K=300, max_iterations=30000, verbose=False)
            state_rstar, roi_rstar = solver_rstar.solve()
            time_rstar = time.time() - start_time
            stats_rstar = solver_rstar.get_search_stats()
        except Exception as e:
            print(f"   ❌ Error en R*: {e}")
            state_rstar, roi_rstar = None, -1e9
            time_rstar = 0
            stats_rstar = {'iterations': 0, 'nodes_expanded': 0}
        
        # EJECUCIÓN A*
        print("   🔄 Ejecutando A*...")
        start_time = time.time()
        try:
            solver_astar = AStar(env, verbose=False)
            state_astar, roi_astar = solver_astar.solve()
            time_astar = time.time() - start_time
            stats_astar = solver_astar.get_search_stats()
        except Exception as e:
            print(f"   ❌ Error en A*: {e}")
            state_astar, roi_astar = None, -1e9
            time_astar = 0
            stats_astar = {'iterations': 0, 'nodes_expanded': 0}
        
        # ✅ CORRECCIÓN: Cálculo robusto de ratios de tiempo
        if time_rstar == 0 and time_astar == 0:
            time_ratio = 1.0
            speed_ratio = 1.0
        elif time_astar == 0:
            time_ratio = float('inf')
            speed_ratio = float('inf')
        elif time_rstar == 0:
            time_ratio = 0.0
            speed_ratio = 0.0
        else:
            time_ratio = time_rstar / time_astar
            speed_ratio = time_astar / time_rstar
        
        # Limitar valores extremos
        time_ratio = max(0.001, min(1000, time_ratio))
        speed_ratio = max(0.001, min(1000, speed_ratio))
        
        # Análisis comparativo
        comparison = {
            'configuration': config,
            'rstar': {
                'success': state_rstar is not None,
                'roi': roi_rstar if state_rstar else -1,
                'computation_time': time_rstar,
                'iterations': stats_rstar['iterations'],
                'nodes_expanded': stats_rstar['nodes_expanded'],
                'route_length': len(state_rstar.seq) if state_rstar else 0,
                'delta_v_used': state_rstar.delta_v_used if state_rstar else 0,
                'water_collected': state_rstar.m_water if state_rstar else 0
            },
            'astar': {
                'success': state_astar is not None,
                'roi': roi_astar if state_astar else -1,
                'computation_time': time_astar,
                'iterations': stats_astar['iterations'],
                'nodes_expanded': stats_astar['nodes_expanded'],
                'route_length': len(state_astar.seq) if state_astar else 0,
                'delta_v_used': state_astar.delta_v_used if state_astar else 0,
                'water_collected': state_astar.m_water if state_astar else 0
            },
            'time_ratio': time_ratio,
            'speed_ratio': speed_ratio
        }
        
        # Métricas de comparación CON PROTECCIÓN CONTRA DIVISIÓN POR CERO
        if state_rstar and state_astar:
            # ROI improvement (protegido)
            if roi_astar != 0:
                roi_improvement = ((roi_rstar - roi_astar) / abs(roi_astar)) * 100
            else:
                roi_improvement = 100.0 if roi_rstar > 0 else 0.0
            
            # Efficiency ratio (protegido)
            if stats_rstar['nodes_expanded'] > 0:
                efficiency_ratio = stats_astar['nodes_expanded'] / stats_rstar['nodes_expanded']
            else:
                efficiency_ratio = float('inf') if stats_astar['nodes_expanded'] > 0 else 1.0
            
            comparison['metrics'] = {
                'roi_improvement_pct': roi_improvement,
                'time_ratio': time_ratio,
                'speed_ratio': speed_ratio,
                'efficiency_ratio': efficiency_ratio,
                'rstar_better_roi': roi_improvement > 0,
                'rstar_faster': time_ratio < 1.0,
                'rstar_more_efficient': efficiency_ratio > 1.0
            }
            
            # ✅ CORRECCIÓN: Display de tiempos más claro
            time_winner = "R*" if time_ratio < 0.9 else "A*" if time_ratio > 1.1 else "EMPATE"
            if speed_ratio >= 1:
                speed_display = f"{speed_ratio:.2f}x"
            else:
                speed_display = f"{1/speed_ratio:.2f}x" if speed_ratio > 0 else "N/A"
            
            print(f"   📊 RESULTADOS:")
            print(f"      ROI: R*={roi_rstar:.4f} vs A*={roi_astar:.4f} (Δ={roi_improvement:+.1f}%)")
            print(f"      Tiempos: R*={time_rstar:.3f}s vs A*={time_astar:.3f}s")
            print(f"      Velocidad: {time_winner} más rápido ({speed_display})")
            print(f"      Nodos: R*={stats_rstar['nodes_expanded']} vs A*={stats_astar['nodes_expanded']} (ratio={efficiency_ratio:.2f}x)")
            
        elif state_rstar and not state_astar:
            comparison['metrics'] = {
                'rstar_found_solution': True, 
                'astar_found_solution': False,
                'roi_improvement_pct': 100.0,
                'time_ratio': 1.0,
                'speed_ratio': 1.0,
                'efficiency_ratio': 1.0
            }
            print(f"   ✅ R* encontró solución, A* no")
            print(f"      ROI R*: {roi_rstar:.4f}, Tiempo: {time_rstar:.3f}s")
            
        elif not state_rstar and state_astar:
            comparison['metrics'] = {
                'rstar_found_solution': False, 
                'astar_found_solution': True,
                'roi_improvement_pct': -100.0,
                'time_ratio': 1.0,
                'speed_ratio': 1.0,
                'efficiency_ratio': 1.0
            }
            print(f"   ❌ A* encontró solución, R* no")  
            print(f"      ROI A*: {roi_astar:.4f}, Tiempo: {time_astar:.3f}s")
            
        else:
            comparison['metrics'] = {
                'both_failed': True,
                'roi_improvement_pct': 0.0,
                'time_ratio': 1.0,
                'speed_ratio': 1.0,
                'efficiency_ratio': 1.0
            }
            print(f"   ❌ Ambos algoritmos fallaron")
            print(f"      Iteraciones R*: {stats_rstar['iterations']}, A*: {stats_astar['iterations']}")
        
        results.append(comparison)
    
    # ANÁLISIS AGREGADO CON PROTECCIÓN
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO FINAL")
    print("="*80)
    
    successful_comparisons = [r for r in results if r['rstar']['success'] and r['astar']['success']]
    rstar_only_success = [r for r in results if r['rstar']['success'] and not r['astar']['success']]
    astar_only_success = [r for r in results if not r['rstar']['success'] and r['astar']['success']]
    both_failed = [r for r in results if not r['rstar']['success'] and not r['astar']['success']]
    
    print(f"📈 ESTADÍSTICAS DE EJECUCIÓN:")
    print(f"   Ambos exitosos: {len(successful_comparisons)}/{len(results)}")
    print(f"   Solo R* exitoso: {len(rstar_only_success)}/{len(results)}") 
    print(f"   Solo A* exitoso: {len(astar_only_success)}/{len(results)}")
    print(f"   Ambos fallaron: {len(both_failed)}/{len(results)}")
    
    if successful_comparisons:
        # Métricas agregadas CON PROTECCIÓN
        roi_improvements = [r['metrics']['roi_improvement_pct'] for r in successful_comparisons]
        speed_ratios = [r['speed_ratio'] for r in successful_comparisons]
        efficiency_ratios = [r['metrics']['efficiency_ratio'] for r in successful_comparisons]
        
        # Filtrar valores infinitos antes de calcular promedios
        valid_roi_improvements = [x for x in roi_improvements if abs(x) < float('inf')]
        valid_speed_ratios = [x for x in speed_ratios if x < float('inf') and x > 0]
        valid_efficiency_ratios = [x for x in efficiency_ratios if x < float('inf')]
        
        avg_roi_improvement = sum(valid_roi_improvements) / len(valid_roi_improvements) if valid_roi_improvements else 0
        avg_speed_ratio = sum(valid_speed_ratios) / len(valid_speed_ratios) if valid_speed_ratios else 1.0
        avg_efficiency = sum(valid_efficiency_ratios) / len(valid_efficiency_ratios) if valid_efficiency_ratios else 1.0
        
        # ✅ CORRECCIÓN: Cálculo de victorias mejorado con umbrales
        roi_wins_rstar = sum(1 for r in successful_comparisons if r['metrics']['roi_improvement_pct'] > 0.1)
        roi_wins_astar = sum(1 for r in successful_comparisons if r['metrics']['roi_improvement_pct'] < -0.1)
        roi_ties = len(successful_comparisons) - roi_wins_rstar - roi_wins_astar
        
        speed_wins_rstar = sum(1 for r in successful_comparisons if r['speed_ratio'] > 1.1)
        speed_wins_astar = sum(1 for r in successful_comparisons if r['speed_ratio'] < 0.9)
        speed_ties = len(successful_comparisons) - speed_wins_rstar - speed_wins_astar
        
        efficiency_wins_rstar = sum(1 for r in successful_comparisons if r['metrics']['efficiency_ratio'] > 1.1)
        efficiency_wins_astar = sum(1 for r in successful_comparisons if r['metrics']['efficiency_ratio'] < 0.9)
        efficiency_ties = len(successful_comparisons) - efficiency_wins_rstar - efficiency_wins_astar
        
        print(f"\n📊 MÉTRICAS PROMEDIO ({len(successful_comparisons)} casos ambos exitosos):")
        print(f"   ROI: R* supera a A* en {avg_roi_improvement:+.3f}% en promedio")
        if avg_speed_ratio >= 1.0:
            print(f"   Velocidad: R* es {avg_speed_ratio:.2f}x más rápido que A*")
        else:
            print(f"   Velocidad: A* es {1/avg_speed_ratio:.2f}x más rápido que R*")
        print(f"   Eficiencia: R* explora {avg_efficiency:.2f}x menos nodos que A*")
        
        print(f"\n🏆 VICTORIAS POR MÉTRICA (con empates):")
        print(f"   ROI: R* {roi_wins_rstar} | A* {roi_wins_astar} | Empates {roi_ties}")
        print(f"   Velocidad: R* {speed_wins_rstar} | A* {speed_wins_astar} | Empates {speed_ties}")
        print(f"   Eficiencia: R* {efficiency_wins_rstar} | A* {efficiency_wins_astar} | Empates {efficiency_ties}")
        
        # Análisis detallado de cada caso
        print(f"\n🔍 ANÁLISIS DETALLADO POR CASO:")
        for i, result in enumerate(successful_comparisons, 1):
            roi_diff = result['metrics']['roi_improvement_pct']
            speed_ratio = result['speed_ratio']
            eff_ratio = result['metrics']['efficiency_ratio']
            
            roi_winner = "R*" if roi_diff > 0.1 else "A*" if roi_diff < -0.1 else "EMPATE"
            speed_winner = "R*" if speed_ratio > 1.1 else "A*" if speed_ratio < 0.9 else "EMPATE"
            eff_winner = "R*" if eff_ratio > 1.1 else "A*" if eff_ratio < 0.9 else "EMPATE"
            
            speed_display = f"{speed_ratio:.2f}x" if speed_ratio >= 1 else f"{1/speed_ratio:.2f}x"
            
            print(f"   Caso {i}: ROI({roi_winner}: {roi_diff:+.3f}%) | "
                  f"Velocidad({speed_winner}: {speed_display}) | "
                  f"Eficiencia({eff_winner}: {eff_ratio:.2f}x)")
        
        # Conclusiones
        print(f"\n🔍 CONCLUSIONES:")
        if avg_roi_improvement > 0.1:
            print(f"   ✅ R* ENCUENTRA MEJORES SOLUCIONES que A*")
            print(f"      (mejora promedio de {avg_roi_improvement:.3f}% en ROI)")
        elif avg_roi_improvement < -0.1:
            print(f"   ✅ A* ENCUENTRA MEJORES SOLUCIONES que R*")
            print(f"      (mejora promedio de {-avg_roi_improvement:.3f}% en ROI)")
        else:
            print(f"   ⚖️  AMBOS ALGORITMOS encuentran soluciones de CALIDAD SIMILAR")
            print(f"      (diferencia promedio de {abs(avg_roi_improvement):.3f}% en ROI)")
            
        if avg_speed_ratio > 1.1:
            print(f"   ✅ R* ES SIGNIFICATIVAMENTE MÁS RÁPIDO que A*")
            print(f"      ({avg_speed_ratio:.1f}x más rápido en promedio)")
        elif avg_speed_ratio < 0.9:
            print(f"   ✅ A* ES SIGNIFICATIVAMENTE MÁS RÁPIDO que R*")
            print(f"      ({1/avg_speed_ratio:.1f}x más rápido en promedio)")
        else:
            print(f"   ⚖️  AMBOS ALGORITMOS tienen VELOCIDAD DE CÓMPUTO SIMILAR")
            
        if avg_efficiency > 1.1:
            print(f"   ✅ R* ES SIGNIFICATIVAMENTE MÁS EFICIENTE en exploración")
            print(f"      (explora {avg_efficiency:.1f}x menos nodos)")
        elif avg_efficiency < 0.9:
            print(f"   ✅ A* ES SIGNIFICATIVAMENTE MÁS EFICIENTE en exploración")  
            print(f"      (explora {1/avg_efficiency:.1f}x menos nodos)")
        else:
            print(f"   ⚖️  AMBOS ALGORITMOS tienen EFICIENCIA DE EXPLORACIÓN SIMILAR")
    
    # Considerar también casos donde solo uno tuvo éxito
    if rstar_only_success:
        print(f"\n💡 R* demostró mayor robustez: encontró soluciones en {len(rstar_only_success)} casos donde A* falló")
    
    if astar_only_success:
        print(f"💡 A* demostró mayor robustez: encontró soluciones en {len(astar_only_success)} casos donde R* falló")
    
    # Guardar resultados detallados
    output_data = {
        'study_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_configurations': len(test_configs),
        'successful_comparisons': len(successful_comparisons),
        'rstar_only_success': len(rstar_only_success),
        'astar_only_success': len(astar_only_success),
        'both_failed': len(both_failed),
        'results': results
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/rstar_vs_astar_study.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: results/rstar_vs_astar_study.json")
    
    return results

# =============================================================================
# MAIN EXTENDIDO CON COMPARACIÓN - VERSIÓN FINAL
# =============================================================================
def main():
    print("=" * 70)
    print("ASTEROID MINING OPTIMIZER - R* vs A* COMPARISON")
    print("=" * 70)
    
    print("\nSeleccione modo de ejecución:")
    print("1. Misión estándar con R*")
    print("2. Estudio comparativo R* vs A*")
    print("3. Ambos")
    
    try:
        choice = input("\nIngrese opción (1-3, default=1): ").strip()
        if choice == "":
            choice = "1"
        choice = int(choice)
    except:
        choice = 1
    
    # Cargar asteroides
    print("\n[INFO] Loading asteroids...")
    asteroid_ids = get_asteroid_ids_rich_in_water()
    asteroids = load_asteroids_from_nasa(asteroid_ids, debug=True, force_refresh=False)
    
    if choice in [1, 3]:
        # Misión estándar R*
        print("\n" + "="*70)
        print("EJECUTANDO MISIÓN ESTÁNDAR CON R*")
        print("="*70)
        
        env = AsteroidMiningEnvironment(asteroids)
        solver = RStar(env, w=5, K=300, max_iterations=30000, verbose=True)
        best_state, best_roi = solver.solve()
        
        if best_state:
            print_cumulative_budget(env, asteroids, best_state)
            print(f"\n✓ SOLUCIÓN R*: ROI={best_roi:.4f}")
            # Visualización opcional - comentar si da problemas
            visualize_solution(env, asteroids, best_state, save_path="./mission_animation.gif")
        else:
            print("\n✗ No se encontró solución con R*")
    
    if choice in [2, 3]:
        # Estudio comparativo
        print("\n" + "="*70)
        print("INICIANDO ESTUDIO COMPARATIVO R* vs A*")
        print("="*70)
        
        results = run_rstar_vs_astar_study()
        
        # Resumen final
        successful = [r for r in results if r['rstar']['success'] and r['astar']['success']]
        if successful:
            roi_improvements = [r['metrics']['roi_improvement_pct'] for r in successful if abs(r['metrics']['roi_improvement_pct']) < float('inf')]
            if roi_improvements:
                avg_roi_imp = sum(roi_improvements) / len(roi_improvements)
                print(f"\n🎯 CONCLUSIÓN FINAL: R* supera a A* por {avg_roi_imp:+.1f}% en ROI promedio")

if __name__ == "__main__":
    main()