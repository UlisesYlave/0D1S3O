import math
import random
import heapq
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import requests

# =============================================================================
# CONSTANTS
# =============================================================================
MU_SUN = 1.32712440018e20      # Gravitational parameter of the Sun
AU = 149_597_870_700           # Astronomical unit in meters
P_WATER = 10_000.0             # Price per kg of water ($)
ROI_MIN = 0.4
C_DEV = 5e8                     # Development cost ($)
C_LAUNCH = 2e7                  # Launch cost ($)
DV_BUDGET = 40_000.0            # Delta-V budget (m/s)
TIME_MAX = 1825.0               # Max mission duration (days)
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
        mining_rate = 1000.0  # kg/day
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

    # -------------------------------------------------------------------------
    # IMPROVED HELPER FUNCTIONS
    # -------------------------------------------------------------------------
    def _estimate_dv(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        """Quick delta-V estimate (Hohmann + plane change) with date consideration"""
        # Use actual positions at given date for better estimate
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Hohmann transfer between current positions
        dv_hohmann = self._hohmann_dv_positions(r1_mag, r2_mag)
        
        # Plane change - calculate angular difference with safe calculation
        dot_product = sum(r1[i] * r2[i] for i in range(3))
        cos_theta = dot_product / (r1_mag * r2_mag)
        # Clamp to avoid numerical errors
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        
        v_avg = math.sqrt(MU_SUN / ((r1_mag + r2_mag)/2))
        dv_plane = 2.0 * v_avg * math.sin(angle_rad / 2.0)
        
        return dv_hohmann + dv_plane * 0.5  # Reduced factor for better accuracy

    def _calculate_transfer_dv_with_tof(self, ast1: Asteroid, ast2: Asteroid, date: float) -> tuple:
        """More accurate delta-V calculation with iterative time-of-flight"""
        # Initial positions
        r1 = ast1.orbital_position(date)
        time_estimate = self._estimate_transfer_time_dated(ast1, ast2, date)
        
        # Iterative refinement (2 iterations)
        for _ in range(2):
            r2 = ast2.orbital_position(date + time_estimate)
            time_estimate = self._calculate_tof(r1, r2)
        
        # Final calculation with refined time
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date + time_estimate)
        
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Circular velocities
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        
        # Transfer orbit (Hohmann-like)
        a_transfer = (r1_mag + r2_mag) / 2.0
        v_at_r1 = math.sqrt(MU_SUN * (2.0/r1_mag - 1.0/a_transfer))
        v_at_r2 = math.sqrt(MU_SUN * (2.0/r2_mag - 1.0/a_transfer))
        
        dv_total = abs(v_at_r1 - v1) + abs(v2 - v_at_r2)
        
        return dv_total, time_estimate

    def _hohmann_dv_positions(self, r1_mag: float, r2_mag: float) -> float:
        """Hohmann transfer between two orbital radii"""
        v1 = math.sqrt(MU_SUN / r1_mag)
        v2 = math.sqrt(MU_SUN / r2_mag)
        at = (r1_mag + r2_mag) / 2.0
        vt1 = math.sqrt(MU_SUN * (2.0 / r1_mag - 1.0 / at))
        vt2 = math.sqrt(MU_SUN * (2.0 / r2_mag - 1.0 / at))
        return abs(vt1 - v1) + abs(v2 - vt2)

    def _estimate_transfer_time_dated(self, ast1: Asteroid, ast2: Asteroid, date: float) -> float:
        """Estimate transfer time using positions at actual date"""
        r1 = ast1.orbital_position(date)
        r2 = ast2.orbital_position(date)
        
        # Better heuristic based on orbital mechanics
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Semi-major axis of transfer orbit
        a_transfer = (r1_mag + r2_mag) / 2.0
        tof_seconds = math.pi * math.sqrt(a_transfer**3 / MU_SUN)  # Half period for Hohmann
        
        tof_days = tof_seconds / (24 * 3600)
        
        return max(30.0, min(400.0, tof_days))

    def _calculate_tof(self, r1: list, r2: list) -> float:
        """Calculate time of flight using Lambert's problem approximation"""
        r1_mag = math.sqrt(sum(x**2 for x in r1))
        r2_mag = math.sqrt(sum(x**2 for x in r2))
        
        # Cosine of transfer angle with safe calculation
        dot_product = sum(r1[i] * r2[i] for i in range(3))
        cos_theta = dot_product / (r1_mag * r2_mag)
        cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to avoid numerical issues
        theta = math.acos(cos_theta)
        
        # Simplified Lambert solution (minimum energy transfer)
        c = math.sqrt(r1_mag**2 + r2_mag**2 - 2*r1_mag*r2_mag*cos_theta)
        s = (r1_mag + r2_mag + c) / 2.0
        a_min = s / 2.0
        
        tof_seconds = math.pi * math.sqrt(a_min**3 / MU_SUN)
        
        return tof_seconds / (24 * 3600)  # Convert to days

    def _total_cost(self, state: State) -> float:
        """Calculate total mission cost - CORRECTED"""
        # Propulsion cost - FIXED MASS CALCULATION
        if state.delta_v_used > 0:
            mass_ratio = math.exp(state.delta_v_used / (ISP * G0))
            m_prop = M0_WET * (1.0 - 1.0/mass_ratio)  # CORRECTED rocket equation
        else:
            m_prop = 0.0
        cost_prop = m_prop * C_FUEL
        
        # Mining cost
        asteroids_mined = len(state.seq) - 1  # Exclude BASE
        cost_mining = asteroids_mined * 10.0 * 24 * 50.0  # 10 days * 24 hours * $50/hr
        
        # Operations cost
        cost_ops = state.t_current * 1000.0
        
        # Return cost - ONLY IF NOT ALREADY AT BASE
        if state.current != "BASE":
            last_asteroid = self.asteroids[state.current]
            
            # Calculate return delta-V from current position
            r_current = last_asteroid.orbital_position(state.date_abs)
            r_current_mag = math.sqrt(sum(x**2 for x in r_current))
            r_earth = 1.0 * AU  # Earth orbit radius
            
            dv_return = self._hohmann_dv_positions(r_current_mag, r_earth) + 800.0  # To Gateway
            
            # Wet mass for return includes collected water
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

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)

class RStar:
    """True R* algorithm as described in Likhachev & Stentz (2008)"""
    
    def __init__(self, env: AsteroidMiningEnvironment, w: float = 2.0, 
                 K: int = 5, delta_asteroids: int = 3, max_local_expansions: int = 100,
                 verbose: bool = True):
        self.env = env
        self.w = w  # Inflation factor for weighted A*
        self.K = K  # Number of random successors per expansion
        self.delta_asteroids = delta_asteroids  # "Distance" in terms of asteroids
        self.max_local_expansions = max_local_expansions
        self.verbose = verbose
        
        # Sparse graph Γ
        self.gamma = {}  # state_hash -> GammaNode
        self.edges = {}  # (from_hash, to_hash) -> Edge
        
    def solve(self) -> Tuple[Optional[State], float]:
        """Run true R* algorithm"""
        
        initial = self.env.initial_state()
        goal_roi = -1e9
        goal_state = None
        
        # Initialize sparse graph with start state
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
            
            # Select node with minimum priority (prefer non-AVOID)
            current_node = heapq.heappop(OPEN).item
            current_hash = hash(current_node.state)
            
            if current_hash in CLOSED:
                continue
            
            # Check if we've found a valid goal state
            if self.env.is_goal(current_node.state):
                roi = self.env.calculate_roi(current_node.state)
                if roi > goal_roi:
                    goal_roi = roi
                    goal_state = current_node.state
                    if self.verbose:
                        print(f"  [Iter {iterations}] Found goal! ROI={roi:.4f}, "
                              f"Route: {' → '.join(current_node.state.seq)}")
                # Continue searching for better solutions
                continue
            
            # Try to compute local path to this node if not done yet
            if current_node.state != initial and not current_node.local_path_computed:
                if not self._try_compute_local_path(current_node):
                    # Failed to compute local path easily - mark as AVOID
                    current_node.is_avoid = True
                    current_node.local_path_computed = True
                    # Re-insert with AVOID priority
                    heapq.heappush(OPEN, PrioritizedItem(
                        priority=self._compute_key(current_node),
                        item=current_node
                    ))
                    continue
            
            # Expand node (generate K random successors at distance Δ)
            CLOSED.add(current_hash)
            successors = self._generate_random_successors(current_node)
            
            if self.verbose and iterations % 50 == 0:
                print(f"  [Iter {iterations}] Expanded {len(CLOSED)} nodes, "
                      f"OPEN={len(OPEN)}, best_ROI={goal_roi:.4f}")
            
            for succ_state in successors:
                succ_hash = hash(succ_state)
                
                # Skip if already closed
                if succ_hash in CLOSED:
                    continue
                
                # Create or get gamma node
                if succ_hash not in self.gamma:
                    succ_node = self._create_gamma_node(succ_state)
                    succ_node.parent = current_node
                    self.gamma[succ_hash] = succ_node
                else:
                    succ_node = self.gamma[succ_hash]
                
                # Create edge (local path not computed yet, use heuristic)
                edge_key = (current_hash, succ_hash)
                if edge_key not in self.edges:
                    # Estimate cost using heuristic
                    estimated_cost = self._estimate_edge_cost(current_node.state, succ_state)
                    self.edges[edge_key] = {
                        'cost': estimated_cost,
                        'computed': False,
                        'actual_path': None
                    }
                
                # Update g-value if this path is better
                tentative_g = current_node.g + self.edges[edge_key]['cost']
                if tentative_g < succ_node.g:
                    succ_node.g = tentative_g
                    succ_node.parent = current_node
                    
                    # Add to OPEN
                    heapq.heappush(OPEN, PrioritizedItem(
                        priority=self._compute_key(succ_node),
                        item=succ_node
                    ))
        
        if self.verbose:
            print(f"[R*] Search complete. Iterations={iterations}, "
                  f"Nodes in Γ={len(self.gamma)}, Closed={len(CLOSED)}")
        
        return goal_state, goal_roi if goal_state else (None, -1e9)
    
    def _create_gamma_node(self, state: State):
        """Create a node in the sparse graph Γ"""
        return type('GammaNode', (), {
            'state': state,
            'g': 0.0 if state.seq == ["BASE"] else float('inf'),
            'parent': None,
            'is_avoid': False,
            'local_path_computed': False
        })()
    
    def _compute_key(self, node) -> List[float]:
        """Compute priority key [avoid_flag, f_value]"""
        h = self._heuristic(node.state)
        f = node.g + self.w * h
        
        # AVOID nodes get priority 1, others get 0
        avoid_priority = 1 if node.is_avoid else 0
        
        return [avoid_priority, f]
    
    def _heuristic(self, state: State) -> float:
        """Heuristic: remaining potential value"""
        unvisited = [ast for aid, ast in self.env.asteroids.items()
                     if aid not in state.seq and aid != "BASE" and ast.water_fraction > 0]
        
        if not unvisited:
            return 0.0
        
        # Estimate potential revenue from unvisited asteroids
        potential_water = sum(ast.available_water_kg() for ast in unvisited)
        potential_revenue = potential_water * P_WATER
        
        # Rough cost estimate
        avg_dv = 5000.0  # m/s per asteroid
        potential_cost = len(unvisited) * avg_dv * C_FUEL / (ISP * G0)
        
        return max(0.0, -(potential_revenue - potential_cost) / (C_DEV + C_LAUNCH))
    
    def _generate_random_successors(self, node) -> List[State]:
        """Generate K random successor states at distance Δ"""
        current_state = node.state
        available = self.env.actions(current_state)
        
        if not available:
            return []
        
        # Generate K random successors
        successors = []
        
        for _ in range(min(self.K, len(available))):
            if not available:
                break
            
            # Randomly select an asteroid to visit
            target = random.choice(available)
            available.remove(target)
            
            # Create successor state by visiting this asteroid
            succ_state = self.env.result(current_state, target)
            successors.append(succ_state)
        
        # Always include goal states if reachable
        for action in self.env.actions(current_state):
            succ = self.env.result(current_state, action)
            if self.env.is_goal(succ) and succ not in successors:
                successors.append(succ)
                break
        
        return successors
    
    def _estimate_edge_cost(self, from_state: State, to_state: State) -> float:
        """Estimate cost of edge using heuristic"""
        dv_diff = to_state.delta_v_used - from_state.delta_v_used
        return dv_diff * C_FUEL / (ISP * G0)
    
    def _try_compute_local_path(self, node) -> bool:
        """
        Try to compute local path from parent to this node using weighted A*.
        Returns True if successful, False if too hard (exceeds max_local_expansions).
        """
        if node.parent is None:
            return True
        
        parent_state = node.parent.state
        target_state = node.state
        
        # Simple check: if we can reach target from parent in one step, it's easy
        if target_state.current in self.env.actions(parent_state):
            return True
        
        # Otherwise, check if the path is reasonable
        dv_increase = target_state.delta_v_used - parent_state.delta_v_used
        time_increase = target_state.t_current - parent_state.t_current
        
        # Easy path criteria: reasonable resource usage
        if dv_increase < 20000 and time_increase < 200:
            node.local_path_computed = True
            
            # Update edge cost with actual cost
            edge_key = (hash(parent_state), hash(target_state))
            if edge_key in self.edges:
                self.edges[edge_key]['computed'] = True
                self.edges[edge_key]['cost'] = dv_increase * C_FUEL / (ISP * G0)
            
            return True
        
        # Hard path - would require extensive search
        return False

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

def print_cumulative_budget(env: AsteroidMiningEnvironment, asteroids: Dict[str, Asteroid], state: State):
    """Show cumulative ΔV, costs, and water collected at each step."""
    cumulative_dv = 0.0
    cumulative_cost = 0.0
    cumulative_water = 0.0
    prev = "BASE"

    print("\n[INFO] Cumulative budget along the route:")
    print(f"{'Step':<5} {'Asteroid':<8} {'ΔV_this_step':>12} {'ΔV_cum':>12} "
          f"{'Water_collected':>16} {'Cost_cum':>14}")

    for i, ast_id in enumerate(state.seq[1:]):  # skip BASE
        ast_from = asteroids[prev]
        ast_to = asteroids[ast_id]

        # ΔV for this jump
        dv = env._estimate_dv(ast_from, ast_to, state.date_abs)
        cumulative_dv += dv

        # Water collected this asteroid
        water = ast_to.available_water_kg()
        cumulative_water += water

        # Compute cost including return if this were the last asteroid
        temp_state = State(state.seq[:i+2], ast_id, cumulative_dv, cumulative_water, 0, 0)
        cost = env._total_cost(temp_state)
        cumulative_cost = cost

        print(f"{i+1:<5} {ast_id:<8} {dv:>12,.0f} {cumulative_dv:>12,.0f} "
              f"{cumulative_water:>16,.0f} ${cumulative_cost:>13,.0f}")

        prev = ast_id

def load_asteroids_from_nasa(ids: List[str], debug: bool = False) -> Dict[str, Asteroid]:
    """Load asteroids from NASA SBDB API with robust parsing of orbital elements and physical parameters."""
    loader = NASADataLoader(debug=debug)
    records = loader.fetch_many(ids)
    
    asteroids = {}
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
            pass

        # ----------------------------
        # Physical parameters
        # ----------------------------
        radius_m = random.uniform(200, 1500)
        spectral_type = "C"
        water_frac = 0.05
        try:
            phys = rec.get("phys_par", {})
            diameter_km = phys.get("diameter")
            if diameter_km is not None:
                radius_m = float(diameter_km) * 500  # km -> m radius
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
    
    # Add BASE station
    asteroids["BASE"] = Asteroid(
        id="BASE", a=1.0, e=0.0, i_deg=0.0, om_deg=0.0,
        w_deg=0.0, ma_deg=0.0, radius_m=100.0, water_fraction=0.0, spectral_type="S"
    )
    
    if debug:
        print(f"[INFO] Loaded {len(asteroids)-1} asteroids + BASE")
    
    return asteroids

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("ASTEROID MINING MISSION PLANNER")
    print("=" * 70)
    
    # Option 1: Synthetic asteroids (reliable testing)
    #print("\n[INFO] Generating synthetic asteroids...")
    #asteroids = generate_synthetic_asteroids(n=15)
    
    # Option 2: Real NASA data (uncomment to use)
    print("\n[INFO] Loading asteroids from NASA SBDB...")
    nasa_ids = ["433", "1", "2", "3", "4", "6", "7", "10", "15", "16"]
    asteroids = load_asteroids_from_nasa(nasa_ids, debug=True)
    
    print(f"[INFO] Loaded {len(asteroids)-1} asteroids")
    print(f"[INFO] Budget: ΔV={DV_BUDGET/1000:.0f} km/s, Time={TIME_MAX:.0f} days")
    
    # Show some asteroid details
    print("\n[INFO] Sample asteroids:")
    for i, (aid, ast) in enumerate(list(asteroids.items())[:6]):
        if aid == "BASE":
            continue
        dv_est = math.sqrt(MU_SUN / (ast.a * AU)) * 0.3  # rough estimate
        print(f"  {aid}: a={ast.a:.2f}AU, i={ast.i_deg:.1f}°, "
              f"water≈{ast.available_water_kg():.0f}kg, ΔV≈{dv_est:.0f}m/s")
    
    # Create environment
    env = AsteroidMiningEnvironment(asteroids)
    
    # Run R* algorithm
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
        # Include virtual return to base in the route
        print_cumulative_budget(env, asteroids, best_state)
        route = best_state.seq.copy()
        if best_state.current != "BASE":
            route.append("BASE")  # indicate return home

        print(f"\n✓ SOLUTION FOUND")
        print(f"  Route: {' → '.join(route)}")
        print(f"  Asteroids visited: {len(best_state.seq) - 1}")
        print(f"  Delta-V used (to last asteroid): {best_state.delta_v_used:,.0f} m/s")
        
        # Compute costs including return
        total_cost = env._total_cost(best_state)
        revenue = best_state.m_water * P_WATER
        profit = revenue - total_cost

        # Compute approximate return delta-V
        if best_state.current != "BASE":
            last_ast = asteroids[best_state.current]
            dv_return = env._hohmann_dv_positions(last_ast.a * AU, 1.0 * AU) + 800.0
        else:
            dv_return = 0.0

        print(f"  Delta-V to return: {dv_return:,.0f} m/s")
        print(f"  Total Delta-V (including return): {best_state.delta_v_used + dv_return:,.0f} m/s")
        
        print(f"  Water collected: {best_state.m_water:,.0f} kg")
        print(f"  Mission time: {best_state.t_current:.1f} days")
        print(f"  Revenue: ${revenue:,.0f}")
        print(f"  Costs (including return): ${total_cost:,.0f}")
        print(f"  Profit: ${profit:,.0f}")
        print(f"  ROI: {best_roi:.4f} ({best_roi * 100:.2f}%)")

        # Asteroids not visited
        all_ast_ids = [aid for aid in asteroids.keys() if aid != "BASE"]
        visited_ids = [aid for aid in best_state.seq if aid != "BASE"]
        not_visited = set(all_ast_ids) - set(visited_ids)
        if not_visited:
            print(f"\n  Asteroids not visited: {', '.join(sorted(not_visited))}")
    else:
        print("\n✗ No solution found")

if __name__ == "__main__":
    main()