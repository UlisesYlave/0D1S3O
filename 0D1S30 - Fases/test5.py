import math
import random
import heapq
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import requests
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# =============================================================================
# CONSTANTS
# =============================================================================
MU_SUN = 1.32712440018e20
AU = 149_597_870_700
P_WATER = 20_000.0
ROI_MIN = 1
C_DEV = 5e8
C_LAUNCH = 2e7
DV_BUDGET = 40_000.0
TIME_MAX = 1825.0
C_FUEL = 5000.0
M0_WET = 5000.0
ISP = 3000.0
G0 = 9.80665

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
            if self.debug:
                print(f"[SBDB] Cache hit: {obj_id}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        if self.debug:
            print(f"[SBDB] Fetching: {obj_id}")
        res = requests.get(self.SBDB_BASE, params={"des": obj_id, "full-prec": "true"}, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"SBDB failed: {res.status_code}")
        data = res.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        time.sleep(self.rate_limit_s)
        return data

    def fetch_many(self, ids: List[str], force: bool = False) -> List[Optional[Dict[str, Any]]]:
        results = []
        for obj_id in ids:
            try:
                results.append(self.fetch_object(obj_id, force))
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

    def mass_kg(self) -> float:
        density = 2000.0 if self.spectral_type == "C" else 3000.0
        return (4/3) * math.pi * self.radius_m ** 3 * density

    def available_water_kg(self, mining_time_days: float = 10.0) -> float:
        total_water = self.mass_kg() * self.water_fraction
        mining_rate = 1000.0
        return min(total_water, mining_rate * mining_time_days)

    def orbital_position(self, date_days: float = 0.0) -> Tuple[float, float, float]:
        period_days = 365.25 * self.a ** 1.5
        ma_updated = self.ma_deg + (date_days / period_days) * 360.0
        r = self.a * AU
        x = r * math.cos(math.radians(ma_updated))
        y = r * math.sin(math.radians(ma_updated)) * math.cos(math.radians(self.i_deg))
        z = r * math.sin(math.radians(ma_updated)) * math.sin(math.radians(self.i_deg))
        return x, y, z

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

    def __hash__(self):
        return hash((tuple(self.seq), self.current,
                     round(self.delta_v_used, 2), round(self.m_water, 2),
                     round(self.t_current, 2)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def clone_and_add(self, asteroid_id: str, dv_add: float, water_add: float, time_add: float):
        return State(
            seq=self.seq + [asteroid_id],
            current=asteroid_id,
            delta_v_used=self.delta_v_used + dv_add,
            m_water=self.m_water + water_add,
            t_current=self.t_current + time_add,
            date_abs=self.date_abs + time_add
        )

# =============================================================================
# VISUALIZATION FUNCTIONS - CORREGIDAS
# =============================================================================
def plot_orbital_trajectory(env, asteroids: Dict[str, Asteroid], state: State, filename: str = "mission_trajectory.png"):
    """Plot XY orbital map with Sun at center and mission trajectory"""

    # OBTENER EL DIRECTORIO ACTUAL del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filename)
    
    print(f"[VIZ] Saving to: {full_path}")

    print(f"[VIZ] Creating orbital trajectory plot for route: {state.seq}")
    print(f"[VIZ] Target filename: {filename}")
    print(f"[VIZ] File exists before save: {os.path.exists(filename)}")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # CONTENIDO DEL GRÁFICO (simplificado para prueba)
        ax.scatter(0, 0, color='yellow', s=200, label='Sun')
        ax.scatter(1, 0, color='blue', s=50, label='Earth')
        
        # Plot simple trajectory
        if len(state.seq) > 1:
            x_vals = [0, 1, 2, 3]  # Valores simples para prueba
            y_vals = [0, 0.5, 0.2, 0]
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label='Trajectory')
        
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_title(f'Mission: {" → ".join(state.seq)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # INTENTAR GUARDAR con diferentes métodos
        success = False
        
        try:
            print(f"[VIZ] Attempting to save figure...")
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            print(f"[VIZ] Save completed without errors")
            success = True
        except Exception as e:
            print(f"[VIZ] Save failed: {e}")
            success = False
        
        # VERIFICAR si el archivo se creó
        if success:
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"[VIZ] SUCCESS: File created: {filename} ({file_size} bytes)")
            else:
                print(f"[VIZ] ERROR: Save reported success but file doesn't exist: {filename}")
        else:
            print(f"[VIZ] Save failed completely")
        
        plt.close(fig)
        return success
        
    except Exception as e:
        print(f"[VIZ] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_thrust_profile(env, asteroids: Dict[str, Asteroid], state: State, filename: str = "thrust_profile.png"):
    """Plot thrust magnitude vs time for the mission"""

    # OBTENER EL DIRECTORIO ACTUAL del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filename)
    
    print(f"[VIZ] Saving to: {full_path}")

    print(f"[VIZ] Creating thrust profile plot for route: {state.seq}")
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        time_points = [0.0]
        thrust_points = [0.0]
        delta_v_points = [0.0]
        water_points = [0.0]
        events = ["Start at BASE"]
        
        current_time = 0.0
        current_dv = 0.0
        current_water = 0.0
        
        # Reconstruct mission timeline
        for i in range(len(state.seq) - 1):
            ast_from_id = state.seq[i]
            ast_to_id = state.seq[i + 1]
            
            print(f"[VIZ] Processing transfer: {ast_from_id} -> {ast_to_id} at time {current_time:.1f}")
            
            if ast_from_id == "BASE":
                # First transfer from BASE - use simplified calculation
                ast_to = asteroids[ast_to_id]
                transfer_time = 60.0  # Estimated first transfer
                dv_transfer = 3000.0  # Estimated first delta-V
            else:
                ast_from = asteroids[ast_from_id]
                ast_to = asteroids[ast_to_id]
                
                try:
                    # Use environment's transfer calculation
                    dv_transfer, transfer_time = env._calculate_transfer_dv_with_tof(ast_from, ast_to, current_time)
                    print(f"[VIZ] Calculated transfer: {dv_transfer:.0f} m/s, {transfer_time:.1f} days")
                except Exception as e:
                    print(f"[VIZ] Error in transfer calculation: {e}, using defaults")
                    transfer_time = 90.0
                    dv_transfer = 2500.0
            
            # Thrust during transfer (simplified model)
            thrust_duration = transfer_time
            if thrust_duration > 0:
                thrust_magnitude = dv_transfer / thrust_duration / 86400.0  # Convert to m/s²
            else:
                thrust_magnitude = 0
            
            # Add transfer phase points
            # Start of transfer
            time_points.append(current_time)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water)
            events.append(f"Start transfer to {ast_to_id}")
            
            # Middle of transfer (peak thrust)
            time_points.append(current_time + thrust_duration/2)
            thrust_points.append(thrust_magnitude)
            delta_v_points.append(current_dv + dv_transfer/2)
            water_points.append(current_water)
            events.append(f"Mid-transfer to {ast_to_id}")
            
            # End of transfer
            current_time += thrust_duration
            current_dv += dv_transfer
            
            time_points.append(current_time)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water)
            events.append(f"Arrive at {ast_to_id}")
            
            # Mining phase
            mining_time = 10.0
            water_collected = ast_to.available_water_kg(mining_time)
            
            # Middle of mining
            time_points.append(current_time + mining_time/2)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water + water_collected/2)
            events.append(f"Mining at {ast_to_id}")
            
            # End of mining
            current_time += mining_time
            current_water += water_collected
            
            time_points.append(current_time)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water)
            events.append(f"Finish mining at {ast_to_id}")
        
        # Add return to base if needed
        if state.current != "BASE":
            print(f"[VIZ] Adding return to BASE from {state.current}")
            return_time = 120.0  # Estimated return time
            dv_return = 3000.0   # Estimated return delta-V
            
            # Start return
            time_points.append(current_time)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water)
            events.append("Start return to BASE")
            
            # Middle of return
            time_points.append(current_time + return_time/2)
            thrust_points.append(dv_return / return_time / 86400.0)
            delta_v_points.append(current_dv + dv_return/2)
            water_points.append(current_water)
            events.append("Mid-return to BASE")
            
            # End of return
            current_time += return_time
            current_dv += dv_return
            
            time_points.append(current_time)
            thrust_points.append(0)
            delta_v_points.append(current_dv)
            water_points.append(current_water)
            events.append("Arrive at BASE")
        
        # Sort all points by time
        sorted_indices = np.argsort(time_points)
        time_points = np.array(time_points)[sorted_indices]
        thrust_points = np.array(thrust_points)[sorted_indices]
        delta_v_points = np.array(delta_v_points)[sorted_indices]
        water_points = np.array(water_points)[sorted_indices]
        
        print(f"[VIZ] Final timeline: {len(time_points)} points, total time: {current_time:.1f} days")
        
        # Plot 1: Thrust profile
        ax1.plot(time_points, thrust_points, 'b-', linewidth=2, label='Thrust Magnitude')
        ax1.fill_between(time_points, thrust_points, alpha=0.3, color='blue')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Thrust (m/s²)')
        ax1.set_title('Low-Thrust Control Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Mission progress
        ax2b = ax2.twinx()
        line1 = ax2.plot(time_points, delta_v_points, 'r-', linewidth=2, label='Cumulative ΔV')[0]
        line2 = ax2b.plot(time_points, water_points, 'g-', linewidth=2, label='Water Collected')[0]
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Delta-V (m/s)', color='red')
        ax2b.set_ylabel('Water Collected (kg)', color='green')
        ax2.set_title('Mission Progress: Delta-V and Water Collection')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Thrust profile saved as {filename}")
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"[VIZ] Error creating thrust profile: {e}")
        import traceback
        traceback.print_exc()
        return None
# =============================================================================
# ENVIRONMENT
# =============================================================================
class AsteroidMiningEnvironment:
    def __init__(self, asteroids: Dict[str, Asteroid], dv_budget: float = DV_BUDGET, 
                 time_max: float = TIME_MAX, roi_min: float = ROI_MIN):
        self.asteroids = asteroids
        self.dv_budget = dv_budget
        self.time_max = time_max
        self.roi_min = roi_min

    def initial_state(self) -> State:
        return State(["BASE"], "BASE", 0.0, 0.0, 0.0, 0.0)

    def actions(self, state: State) -> List[str]:
        available = []
        current_ast = self.asteroids[state.current]
        for ast_id, asteroid in self.asteroids.items():
            if ast_id in state.seq or ast_id == "BASE" or asteroid.water_fraction <= 0:
                continue
            dv_est = self._estimate_dv(current_ast, asteroid, state.date_abs)
            if state.delta_v_used + dv_est <= self.dv_budget:
                available.append(ast_id)
        return available

    def result(self, state: State, action: str) -> State:
        ast_from = self.asteroids[state.current]
        ast_to = self.asteroids[action]
        dv_transfer, transfer_time = self._calculate_transfer_dv_with_tof(ast_from, ast_to, state.date_abs)
        mining_time = 10.0
        water_collected = ast_to.available_water_kg(mining_time)
        total_time = transfer_time + mining_time
        return state.clone_and_add(action, dv_transfer, water_collected, total_time)

    def is_goal(self, state: State) -> bool:
        if len(state.seq) < 2 or state.delta_v_used > self.dv_budget or state.t_current > self.time_max:
            return False
        return self.calculate_roi(state) >= self.roi_min

    def calculate_roi(self, state: State) -> float:
        revenue = state.m_water * P_WATER
        costs = self._total_cost(state)
        return (revenue - costs) / (C_DEV + C_LAUNCH)

    def _estimate_dv(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        dv_hohmann = self._hohmann_dv_positions(r1_mag, r2_mag)
        
        dot_product = sum(r1[i] * r2[i] for i in range(3))
        cos_theta = dot_product / (r1_mag * r2_mag)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        
        v_avg = math.sqrt(MU_SUN / ((r1_mag + r2_mag)/2))
        dv_plane = 2.0 * v_avg * math.sin(angle_rad / 2.0)
        
        return dv_hohmann + dv_plane * 0.5

    def _calculate_transfer_dv_with_tof(self, ast1: Asteroid, ast2: Asteroid, date: float) -> tuple:
        r1 = ast1.orbital_position(date)
        time_estimate = self._estimate_transfer_time_dated(ast1, ast2, date)
        
        for _ in range(2):
            r2 = ast2.orbital_position(date + time_estimate)
            time_estimate = self._calculate_tof(r1, r2)
        
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date + time_estimate)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        
        a_transfer = (r1_mag + r2_mag) / 2.0
        v_at_r1 = math.sqrt(MU_SUN * (2.0/r1_mag - 1.0/a_transfer))
        v_at_r2 = math.sqrt(MU_SUN * (2.0/r2_mag - 1.0/a_transfer))
        
        dv_total = abs(v_at_r1 - v1) + abs(v2 - v_at_r2)
        
        return dv_total, time_estimate

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

    def _total_cost(self, state: State) -> float:
        if state.delta_v_used > 0:
            mass_ratio = math.exp(state.delta_v_used / (ISP * G0))
            m_prop = M0_WET * (1.0 - 1.0/mass_ratio)
        else:
            m_prop = 0.0
        cost_prop = m_prop * C_FUEL
        
        asteroids_mined = len(state.seq) - 1
        cost_mining = asteroids_mined * 10.0 * 24 * 50.0
        
        cost_ops = state.t_current * 1000.0
        
        if state.current != "BASE":
            last_asteroid = self.asteroids[state.current]
            
            r_current = last_asteroid.orbital_position(state.date_abs)
            r_current_mag = math.sqrt(sum(x**2 for x in r_current))
            r_earth = 1.0 * AU
            
            dv_return = self._hohmann_dv_positions(r_current_mag, r_earth) + 800.0
            
            m_wet_return = M0_WET + state.m_water
            if dv_return > 0:
                mass_ratio_return = math.exp(dv_return / (ISP * G0))
                m_prop_return = m_wet_return * (1.0 - 1.0/mass_ratio_return)
            else:
                m_prop_return = 0.0
                
            cost_return = m_prop_return * C_FUEL
        else:
            cost_return = 0.0
        
        return cost_prop + cost_mining + cost_ops + cost_return

# =============================================================================
# R* ALGORITHM
# =============================================================================
@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)

class RStar:
    def __init__(self, env: AsteroidMiningEnvironment, w: float = 2.0, 
                 K: int = 5, delta_asteroids: int = 3, max_local_expansions: int = 100,
                 verbose: bool = True):
        self.env = env
        self.w = w
        self.K = K
        self.delta_asteroids = delta_asteroids
        self.max_local_expansions = max_local_expansions
        self.verbose = verbose
        
        self.gamma = {}
        self.edges = {}
        
    def solve(self) -> Tuple[Optional[State], float]:
        initial = self.env.initial_state()
        goal_roi = -1e9
        goal_state = None
        
        start_node = self._create_gamma_node(initial)
        self.gamma[hash(initial)] = start_node
        
        OPEN = []
        heapq.heappush(OPEN, PrioritizedItem(
            priority=self._compute_key(start_node),
            item=start_node
        ))
        
        CLOSED = set()
        iterations = 0
        max_iterations = 1000
        
        if self.verbose:
            print(f"[True R*] Starting with w={self.w}, K={self.K}, Δ={self.delta_asteroids}")
        
        while OPEN and iterations < max_iterations:
            iterations += 1
            
            current_node = heapq.heappop(OPEN).item
            current_hash = hash(current_node.state)
            
            if current_hash in CLOSED:
                continue
            
            if self.env.is_goal(current_node.state):
                roi = self.env.calculate_roi(current_node.state)
                if roi > goal_roi:
                    goal_roi = roi
                    goal_state = current_node.state
                    if self.verbose:
                        print(f"  [Iter {iterations}] Found goal! ROI={roi:.4f}, "
                              f"Route: {' → '.join(current_node.state.seq)}")
                continue
            
            if current_node.state != initial and not current_node.local_path_computed:
                if not self._try_compute_local_path(current_node):
                    current_node.is_avoid = True
                    current_node.local_path_computed = True
                    heapq.heappush(OPEN, PrioritizedItem(
                        priority=self._compute_key(current_node),
                        item=current_node
                    ))
                    continue
            
            CLOSED.add(current_hash)
            successors = self._generate_random_successors(current_node)
            
            if self.verbose and iterations % 50 == 0:
                print(f"  [Iter {iterations}] Expanded {len(CLOSED)} nodes, "
                      f"OPEN={len(OPEN)}, best_ROI={goal_roi:.4f}")
            
            for succ_state in successors:
                succ_hash = hash(succ_state)
                
                if succ_hash in CLOSED:
                    continue
                
                if succ_hash not in self.gamma:
                    succ_node = self._create_gamma_node(succ_state)
                    succ_node.parent = current_node
                    self.gamma[succ_hash] = succ_node
                else:
                    succ_node = self.gamma[succ_hash]
                
                edge_key = (current_hash, succ_hash)
                if edge_key not in self.edges:
                    estimated_cost = self._estimate_edge_cost(current_node.state, succ_state)
                    self.edges[edge_key] = {
                        'cost': estimated_cost,
                        'computed': False,
                        'actual_path': None
                    }
                
                tentative_g = current_node.g + self.edges[edge_key]['cost']
                if tentative_g < succ_node.g:
                    succ_node.g = tentative_g
                    succ_node.parent = current_node
                    
                    heapq.heappush(OPEN, PrioritizedItem(
                        priority=self._compute_key(succ_node),
                        item=succ_node
                    ))
        
        if self.verbose:
            print(f"[R*] Search complete. Iterations={iterations}, "
                  f"Nodes in Γ={len(self.gamma)}, Closed={len(CLOSED)}")
        
        return goal_state, goal_roi if goal_state else (None, -1e9)
    
    def _create_gamma_node(self, state: State):
        return type('GammaNode', (), {
            'state': state,
            'g': 0.0 if state.seq == ["BASE"] else float('inf'),
            'parent': None,
            'is_avoid': False,
            'local_path_computed': False
        })()
    
    def _compute_key(self, node) -> List[float]:
        h = self._heuristic(node.state)
        f = node.g + self.w * h
        
        avoid_priority = 1 if node.is_avoid else 0
        
        return [avoid_priority, f]
    
    def _heuristic(self, state: State) -> float:
        unvisited = [ast for aid, ast in self.env.asteroids.items()
                     if aid not in state.seq and aid != "BASE" and ast.water_fraction > 0]
        
        if not unvisited:
            return 0.0
        
        potential_water = sum(ast.available_water_kg() for ast in unvisited)
        potential_revenue = potential_water * P_WATER
        
        avg_dv = 5000.0
        potential_cost = len(unvisited) * avg_dv * C_FUEL / (ISP * G0)
        
        return max(0.0, -(potential_revenue - potential_cost) / (C_DEV + C_LAUNCH))
    
    def _generate_random_successors(self, node) -> List[State]:
        current_state = node.state
        available = self.env.actions(current_state)
        
        if not available:
            return []
        
        successors = []
        
        for _ in range(min(self.K, len(available))):
            if not available:
                break
            
            target = random.choice(available)
            available.remove(target)
            
            succ_state = self.env.result(current_state, target)
            successors.append(succ_state)
        
        for action in self.env.actions(current_state):
            succ = self.env.result(current_state, action)
            if self.env.is_goal(succ) and succ not in successors:
                successors.append(succ)
                break
        
        return successors
    
    def _estimate_edge_cost(self, from_state: State, to_state: State) -> float:
        dv_diff = to_state.delta_v_used - from_state.delta_v_used
        return dv_diff * C_FUEL / (ISP * G0)
    
    def _try_compute_local_path(self, node) -> bool:
        if node.parent is None:
            return True
        
        parent_state = node.parent.state
        target_state = node.state
        
        if target_state.current in self.env.actions(parent_state):
            return True
        
        dv_increase = target_state.delta_v_used - parent_state.delta_v_used
        time_increase = target_state.t_current - parent_state.t_current
        
        if dv_increase < 20000 and time_increase < 200:
            node.local_path_computed = True
            
            edge_key = (hash(parent_state), hash(target_state))
            if edge_key in self.edges:
                self.edges[edge_key]['computed'] = True
                self.edges[edge_key]['cost'] = dv_increase * C_FUEL / (ISP * G0)
            
            return True
        
        return False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_asteroids_from_nasa(ids: List[str], debug: bool = False) -> Dict[str, Asteroid]:
    loader = NASADataLoader(debug=debug)
    records = loader.fetch_many(ids)
    
    asteroids = {}
    for idx, rec in enumerate(records):
        if rec is None:
            continue

        obj_info = rec.get("object", {})
        asteroid_id = obj_info.get("des", f"A{idx+1}")
        
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
            pass

        radius_m = random.uniform(200, 1500)
        spectral_type = "C"
        water_frac = 0.05
        try:
            phys = rec.get("phys_par", {})
            diameter_km = phys.get("diameter")
            if diameter_km is not None:
                radius_m = float(diameter_km) * 500
            spectral_type = phys.get("spec_B", "C") or "C"
            water_frac = 0.05 if spectral_type.upper() == "C" else 0.02
        except Exception:
            pass

        asteroids[asteroid_id] = Asteroid(
            id=asteroid_id,
            a=a, e=e, i_deg=i_deg, om_deg=om_deg,
            w_deg=w_deg, ma_deg=ma_deg,
            radius_m=radius_m,
            water_fraction=water_frac,
            spectral_type=spectral_type
        )
    
    asteroids["BASE"] = Asteroid(
        id="BASE", a=1.0, e=0.0, i_deg=0.0, om_deg=0.0,
        w_deg=0.0, ma_deg=0.0, radius_m=100.0, water_fraction=0.0, spectral_type="S"
    )
    
    if debug:
        print(f"[INFO] Loaded {len(asteroids)-1} asteroids + BASE")
    
    return asteroids

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

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    
    print("=" * 70)
    print("ASTEROID MINING MISSION PLANNER")
    print("=" * 70)
    
    print("\n[INFO] Loading asteroids from NASA SBDB...")
    nasa_ids = ["433", "1", "2", "3", "4", "6", "7", "10", "15", "16"]
    asteroids = load_asteroids_from_nasa(nasa_ids, debug=True)
    
    print(f"[INFO] Loaded {len(asteroids)-1} asteroids")
    print(f"[INFO] Budget: ΔV={DV_BUDGET/1000:.0f} km/s, Time={TIME_MAX:.0f} days")
    
    print("\n[INFO] Sample asteroids:")
    for i, (aid, ast) in enumerate(list(asteroids.items())[:6]):
        if aid == "BASE":
            continue
        dv_est = math.sqrt(MU_SUN / (ast.a * AU)) * 0.3
        print(f"  {aid}: a={ast.a:.2f}AU, i={ast.i_deg:.1f}°, "
              f"water≈{ast.available_water_kg():.0f}kg, ΔV≈{dv_est:.0f}m/s")
    
    env = AsteroidMiningEnvironment(asteroids)
    
    print("\n" + "=" * 70)
    print("RUNNING R* ALGORITHM")
    print("=" * 70)
    
    solver = RStar(env, w=2, K=5, delta_asteroids=3, max_local_expansions=100, verbose=True)
    best_state, best_roi = solver.solve()
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if best_state:
        print(f"\n✓ SOLUTION FOUND")
        print(f"  Route: {' → '.join(best_state.seq)}")
        print(f"  Delta-V used: {best_state.delta_v_used:,.0f} m/s")
        print(f"  Water collected: {best_state.m_water:,.0f} kg") 
        print(f"  Mission time: {best_state.t_current:.1f} days")
        print(f"  ROI: {best_roi:.4f}")

        # Verificar directorio actual y permisos
        current_dir = os.getcwd()
        print(f"\n[INFO] Current directory: {current_dir}")
        print(f"[INFO] Write permission: {os.access(current_dir, os.W_OK)}")
        
        # Intentar visualizaciones
        print("\n[INFO] Generating mission visualizations...")
        
        trajectory_file = "mission_trajectory.png"
        thrust_file = "thrust_profile.png"
        
        # Verificar si los archivos existen antes
        print(f"[INFO] Files before generation:")
        print(f"  {trajectory_file}: {os.path.exists(trajectory_file)}")
        print(f"  {thrust_file}: {os.path.exists(thrust_file)}")
        
        # Generar gráficos
        success1 = plot_orbital_trajectory(env, asteroids, best_state, trajectory_file)
        success2 = plot_thrust_profile(env,asteroids,best_state, thrust_file)
        
        # Verificar resultados
        print(f"\n[INFO] Visualization results:")
        print(f"  Trajectory: {'SUCCESS' if success1 else 'FAILED'}")
        print(f"  Thrust profile: {'SUCCESS' if success2 else 'FAILED'}")
        
        print(f"\n[INFO] Files after generation:")
        print(f"  {trajectory_file}: {os.path.exists(trajectory_file)}")
        print(f"  {thrust_file}: {os.path.exists(thrust_file)}")
        
        if os.path.exists(trajectory_file):
            size = os.path.getsize(trajectory_file)
            print(f"  Trajectory file size: {size} bytes")
        
        if os.path.exists(thrust_file):
            size = os.path.getsize(thrust_file)
            print(f"  Thrust file size: {size} bytes")

        else:
            print("\n✗ No solution found")

if __name__ == "__main__":
    main()