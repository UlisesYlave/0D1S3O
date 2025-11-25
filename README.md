# 0D1S3O: R* Search para Optimización Económica de Minería de Asteroides

## 📖 Descripción

Este proyecto implementa el algoritmo **R*** (R-star) adaptado para la optimización económica de rutas de prospección de asteroides, con enfoque en la explotación de agua para mercados in-situ. El modelo integra criterios técnicos orbitales con evaluación económica comercial para identificar secuencias de asteroides que maximicen el retorno de inversión (ROI).

## 🚀 Instalación y Configuración

### Prerrequisitos
```bash
Python 3.8+
Git
```

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/0D1S30.git
cd 0D1S30

# Instalar dependencias
pip install -r requirements.txt
```

### Estructura de Dependencias (requirements.txt)
```txt
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pytest>=6.0.0
pytest-cov>=2.12.0
requests>=2.25.0  # Para descarga de datos NASA
astropy>=4.3.0    # Para cálculos astronómicos
```

## 🏗️ Estructura del Proyecto

```
0D1S30/
│
├── src/
│   ├── core/
│   │   ├── asteroid.py          # Modelado de asteroides y propiedades
│   │   ├── environment.py       # Ambiente de simulación orbital
│   │   ├── state.py            # Representación del estado de la misión
│   │   └── constants.py        # Parámetros físicos y económicos
│   │
│   ├── algorithms/
│   │   ├── rstar.py            # Implementación de R* adaptado
│   │   └── pso_solver.py       # Implementación PSO para comparación
│   │
│   ├── data/
│   │   └── nasa_loader.py      # Carga de datos de NEOs de NASA
│   │
│   ├── visualization/
│   │   ├── trajectory_plotter.py    # Visualización de trayectorias
│   │   └── comparison_plots.py      # Gráficos comparativos
│   │
│   └── utils/
│       └── helpers.py          # Utilidades matemáticas y orbitales
│
├── tests/                      # Suite de pruebas unitarias
├── experiments/               # Scripts de experimentación
├── results/                   # Resultados y figuras
├── config/                    # Configuraciones
└── main.py                    # Punto de entrada principal
```

## 🧪 Testing y Verificación

### Ejecutar Tests Unitarios
```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Ejecutar tests con cobertura
python -m pytest tests/ --cov=src --cov-report=html

# Ejecutar tests específicos
python -m pytest tests/test_algorithms/test_rstar.py -v
python -m pytest tests/test_core/test_environment.py -v

# Ejecutar tests de comparación con PSO
python -m pytest tests/test_paper_comparison.py -v
```

### Ejecutar Experimentos
```bash
# Experimento principal del paper
python experiments/paper_comparison_study.py

# Estudio de parámetros
python experiments/parameter_study.py

```

## 🎯 Problemática

La minería de asteroides ha evolucionado desde el enfoque inicial en metales preciosos hacia la explotación de volátiles, particularmente agua, para su uso en el espacio. Este nuevo paradigma es significativamente más viable porque:

- **Crea un mercado in-situ** donde el agua se vende como propelente compitiendo con el alto costo de lanzamiento desde Tierra (~$20,000/kg)
- **Es técnicamente menos demandante** ya que la concentración de agua en asteroides tipo C es mucho mayor (5-10%)
- **Sirve como catalizador** para una economía espacial sostenible

## 🧠 Algoritmo R* Adaptado

### Adaptación para Optimización Orbital-Económica

El algoritmo R* ha sido adaptado para explorar secuencias de asteroides considerando tanto criterios técnicos (delta-V, asistencias gravitatorias) como económicos (ROI, costos de operación).

**Función de Evaluación:**
```
f(s) = g(s) + h(s) = -ROI(s) + h(s)
```

Donde:
- **g(s) = -ROI(s)**: Costo real acumulado (negativo del ROI)
- **h(s)**: Heurística que estima el ROI máximo potencial de asteroides no visitados

### Mecanismo Principal

R* evita quedar atrapado en mínimos locales mediante:
1. **Búsquedas locales de corto alcance** hacia objetivos aleatorios
2. **Generación aleatoria de sub-objetivos** dentro de un radio Δ económico-orbital
3. **Postergación de rutas difíciles** (estados AVOID)
4. **Reconstrucción de la solución** desde caminos económicamente viables

## ⚙️ Modelo Económico Integrado

### Cálculo de ROI
```
ROI(s) = (I_total(s) - C_total(s)) / (C_dev + C_launch)
```

**Ingresos:**
- `I_total(s) = s.m_water * P_water` (agua extraída × precio en espacio)

**Costos Totales:**
- `C_prop(s) = C_fuel * m_0 * (1 - e^(-s.deltaV_used/(I_sp * g_0)))` (propulsión)
- `C_min(s) = t_mining * cost_hour * |s.seq|` (operaciones de minería)
- `C_ops(s) = s.t_current * cost_day` (operaciones de misión)
- `C_return(s) = deltaV_return * C_fuel * m_wet` (retorno a órbita comercial)

## 🚀 Uso Rápido

### Ejemplo Básico
```python
from src.core.environment import AsteroidMiningEnvironment
from src.algorithms.rstar import RStar
from src.data.nasa_loader import load_nearest_neos

# Cargar datos de asteroides
asteroids = load_nearest_neos(max_distance=0.1)  # AU

# Configurar ambiente y algoritmo
env = AsteroidMiningEnvironment(asteroids)
planner = RStar(
    env=env,
    w=2.0,
    K=5, 
    delta_threshold=5000,
    max_local_expansions=100
)

# Ejecutar optimización
solution, metrics = planner.solve()
print(f"ROI: {metrics['roi']:.2f}, Delta-V: {metrics['delta_v']:.0f} m/s")
```

### Ejemplo Avanzado con Configuración
```python
from experiments.benchmark_paper import run_comparison_experiment

# Ejecutar experimento comparativo R* vs PSO
results = run_comparison_experiment(
    scenario="commercial",
    max_asteroids=10,
    time_limit=3600  # 1 hora
)

# Generar gráficos comparativos
from src.visualization.comparison_plots import plot_roi_comparison
plot_roi_comparison(results)
```

## 📊 Metodología de Evaluación

### Enfoque de Validación

El modelo se valida mediante:

1. **Comparación con PSO de Yang et al.** en términos de eficiencia computacional y calidad de soluciones
2. **Análisis de sensibilidad** de parámetros económicos críticos
3. **Estudio de casos** con asteroides reales del catálogo NEO
4. **Métricas de desempeño**: ROI, delta-V total, tiempo de misión, tasa de éxito

### Escenarios de Prueba

**Caso 1: Prospección Focalizada**
```bash
python main.py --scenario focused --asteroids 5 --budget 1e9
```

**Caso 2: Ruta Comercial**
```bash
python main.py --scenario commercial --asteroids 12 --budget 2e9
```

**Caso 3: Expansión de Mercado**
```bash
python main.py --scenario expansion --asteroids 20 --budget 5e9
```

## 🎨 Visualización

### Generar Gráficos
```python
from src.visualization.trajectory_plotter import plot_3d_trajectory
from src.visualization.comparison_plots import plot_economic_analysis

# Visualizar trayectoria optimizada
plot_3d_trajectory(solution, asteroids)

# Análisis económico comparativo
plot_economic_analysis(rstar_results, pso_results)
```

### Comandos de Visualización
```bash
# Generar todas las figuras del paper
python -m src.visualization.trajectory_plotter --input results/data/optimized_routes.json

# Crear dashboard interactivo
python -m src.visualization.comparison_plots --interactive
```

## 🔬 Contribuciones Principales

### Avances sobre el Estado del Arte

1. **Integración Económico-Orbital**: Primer modelo que combina optimización técnica con viabilidad comercial
2. **Algoritmo R* Adaptado**: Aplicación innovadora de búsqueda heurística randomizada a dominio espacial
3. **Modelo de Mercado In-Situ**: Enfoque realista en agua como commodity espacial
4. **Framework Extensible**: Arquitectura modular para futuras extensiones

## 📈 Resultados Esperados

El modelo demuestra que:

- Las rutas multi-asteroide **superan el punto de equilibrio** en plazos compatibles con inversión privada
- La optimización económico-técnica **identifica oportunidades** no visibles para enfoques puramente técnicos
- El algoritmo R* **escala eficientemente** a problemas de planificación complejos

## 📚 Referencias

[1] Hein, A. M., et al. "A techno-economic analysis of asteroid mining." Acta Astronautica (2019)

[2] Yang, H., et al. "Low-cost transfer between asteroids with distant orbits using multiple gravity assists." Advances in Space Research (2015)

[3] Likhachev, M., & Stentz, A. "R* Search." Proceedings of the AAAI Conference on Artificial Intelligence (2008)

[4] Olympio, J.T. "Optimal control problem for low-thrust multiple asteroid tour missions." Journal of Guidance, Control, and Dynamics (2011)
