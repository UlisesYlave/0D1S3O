# 0D1S3O - Asteroid Mining Route Optimizer 🚀🌌

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NASA Data](https://img.shields.io/badge/data-NASA_SBDB-green.svg)](https://ssd-api.jpl.nasa.gov/)

Un sistema avanzado de optimización de rutas para minería de asteroides que implementa el algoritmo **R*** adaptado para combinar mecánica orbital con análisis económico en la planificación de misiones comercialmente viables.

## ✨ Características Principales

- **🛰️ Optimización Orbital-Económica**: Integra mecánica orbital con criterios de ROI comercial
- **📡 Datos NASA en Tiempo Real**: Acceso a la base de datos SBDB de la NASA con parámetros físicos reales
- **🎯 Algoritmo R* Adaptado**: Búsqueda dirigida con consideración de dirección orbital
- **💰 Modelo Económico Realista**: Costos de desarrollo, lanzamiento, operaciones y minería
- **🔄 Trayectorias Continuas**: Generación de rutas que evitan el Sol
- **📊 Visualización Animada**: Animaciones 2D de misiones completas
- **⚡ Comparación R* vs A***: Estudio comparativo de algoritmos de búsqueda

## 🚀 Instalación Rápida

### Requisitos
```bash
Python 3.8+
pip install numpy matplotlib requests
```

### Ejecución
```bash
# Descargar el código y ejecutar
python 0D1S3O.py
```

## 🎯 Problemática y Marco Teórico

### Contexto de Minería Espacial
La minería de asteroides ha evolucionado desde el enfoque inicial en metales preciosos hacia la explotación de volátiles, particularmente **agua**, para su uso en el espacio. Este nuevo paradigma es significativamente más viable porque:

- **Crea un mercado in-situ** donde el agua se vende como propelente compitiendo con el alto costo de lanzamiento desde Tierra
- **Es técnicamente menos demandante** ya que la concentración de agua en asteroides tipo C es mucho mayor (5-10%)
- **Sirve como catalizador** para una economía espacial sostenible

### Fundamentos del Algoritmo R*
El algoritmo **R*** (R-star) es una técnica de búsqueda heurística randomizada que combina la completitud de A* con mecanismos para escapar de óptimos locales. Su adaptación para optimización orbital-económica incluye:

**Función de Evaluación:**
```
f(s) = g(s) + w × h(s)
```

Donde:
- **g(s) = -ROI(s)**: Costo real acumulado (negativo del ROI)
- **h(s)**: Heurística que estima el ROI potencial máximo de asteroides no visitados
- **w**: Factor de peso para balancear optimalidad vs. eficiencia

## ⚙️ Modelo Económico Integrado

### Cálculo de ROI
```
ROI(s) = (I_total(s) - C_total(s)) / (C_dev + C_launch)
```

**Ingresos:**
- `I_total(s) = s.m_water × P_water` (agua extraída × precio en espacio)

**Costos Totales:**
- `C_prop(s)`: Costo de propelente basado en delta-V consumido
- `C_min(s)`: Costos de operaciones de minería por asteroide
- `C_ops(s)`: Costos operativos diarios de la misión
- `C_return(s)`: Costo de retorno a estación L2

### Parámetros Económicos Clave
```python
P_WATER = 500.0        # $500/kg - precio realista de agua en espacio
C_DEV = 5e8           # $500M - costo de desarrollo
C_LAUNCH = 2e7        # $20M - costo de lanzamiento
ROI_MIN = 0.01        # 1% ROI mínimo aceptable
```

## 🧠 Algoritmo R* Adaptado

### Innovaciones Principales

1. **Integración Orbital-Económica**: Primera formulación que combina mecánica orbital con ROI comercial en espacio de estados unificado

2. **Dirección Orbital en Búsqueda**: Incorporación de vectores dirección para guiar la exploración hacia rutas orbitalmente eficientes

3. **Selección Adaptativa de Sucesores**: Combinación de criterios de agua, delta-V y dirección orbital para poda inteligente

### Mecanismo de Búsqueda

```python
class RStar:
    def _generate_directed_successors(self, node: RStarNode) -> List[RStarNode]:
        # Scoring de acciones considerando dirección orbital
        scored_actions = []
        for action in available_actions:
            score = self._score_action_with_direction(
                current_state, current_direction, action)
            scored_actions.append((score, action))
        
        # Selección de mejores K sucesores
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        return scored_actions[:self.K]
```

## 🚀 Uso del Sistema

### Ejecución Básica

```python
from 0D1S3O import main

# Ejecutar con configuración por defecto
main()
```

### Configuración Personalizada

```python
# Cargar asteroides específicos de la NASA
asteroid_ids = ["1", "2", "4", "6", "10"]  # Asteroides conocidos tipo-C
asteroids = load_asteroids_from_nasa(asteroid_ids, debug=True)

# Configurar ambiente de minería
env = AsteroidMiningEnvironment(
    asteroids=asteroids,
    dv_budget=20000,    # m/s
    time_max=3000,      # días
    roi_min=0.01        # 1% ROI mínimo
)

# Ejecutar R* con parámetros personalizados
solver = RStar(env, w=10, K=8, max_iterations=5000)
solution, roi = solver.solve()
```

### Modos de Ejecución

El sistema ofrece tres modos de operación:

1. **Misión Estándar con R***: Optimización de ruta única
2. **Estudio Comparativo R* vs A***: Análisis de desempeño de algoritmos  
3. **Ambos**: Ejecución completa con comparación

## 📊 Resultados y Visualización

### Salidas Generadas

- **Ruta Óptima**: Secuencia de asteroides visitados
- **Métricas Económicas**: ROI, costos detallados, ingresos proyectados
- **Métricas Técnicas**: Delta-V total, tiempo de misión, agua recolectada
- **Visualización Animada**: Trayectoria 2D de la misión completa

### Ejemplo de Salida
```
✓ SOLUCIÓN R*: ROI=0.0152
Ruta: BASE → A5 → A12 → L2_STATION
ΔV Total: 14,250 m/s, Agua: 1.2M kg, Tiempo: 890 días
```
![mission_animation](https://github.com/user-attachments/assets/c559a535-5bc3-4144-9cd2-f96ba1ff199c)

## 🔬 Estudio Comparativo

El sistema incluye un módulo de comparación sistemática entre R* y A* que evalúa:

- **Calidad de Solución**: ROI alcanzado
- **Eficiencia Computacional**: Tiempo de ejecución
- **Eficiencia de Exploración**: Nodos expandidos
- **Robustez**: Tasa de éxito en diferentes escenarios

### Ejecutar Comparación
```python
# Ejecutar estudio comparativo completo
from 0D1S3O import run_rstar_vs_astar_study
results = run_rstar_vs_astar_study()
```

## 🎨 Visualización de Trayectorias

El sistema genera animaciones que muestran:

- **Posiciones Orbitales**: Asteroides, Tierra, y estación L2
- **Trayectoria de la Nave**: Ruta continua que evita el Sol
- **Progreso de Misión**: Días transcurridos y segmentos activos
- **Zona de Peligro**: Región cercana al Sol a evitar

```python
# Generar animación de la misión
from 0D1S3O import visualize_solution
visualize_solution(env, asteroids, solution, save_path="mission.gif")
```

## 🔮 Extensiones Futuras

- Integración con bases de datos de asteroides actualizadas
- Modelos de precios dinámicos para commodities espaciales
- Consideración de incertidumbres orbitales
- Optimización multi-objetivo (ROI, riesgo, tiempo)
- Interfaz gráfica de usuario para configuración de misiones

## 📚 Referencias

[1] Hein, A. M., et al. "A techno-economic analysis of asteroid mining." Acta Astronautica (2019)

[2] Likhachev, M., & Stentz, A. "R* Search." Proceedings of the AAAI Conference on Artificial Intelligence (2008)

[3] NASA Small-Body Database - JPL Solar System Dynamics

---

**Desarrollado para la optimización de misiones de minería espacial comercialmente viables** 🌠
