# 0D1S3O

Planificador de ruta para minería espacial usando algoritmos bioinspirados con datos reales de asteroides.

# Cinturon Asteroides

Aplicación de prueba para visualización y animación de asteroides del cinturón de asteroides usando la API JPL de la NASA

```
"https://ssd-api.jpl.nasa.gov/sbdb.api"
```

### Clases de orbitas posibles a probar
* MBA: Main-belt Asteroid
* OMB: Outer Main-belt Asteroid 
* IMB: Inner Main-belt Asteroid
* AMO: Amor
* APO: Apollo
* ATE: Aten

### Cálculo de movimiento

#### Parámetros orbitales clave
* a: Semieje mayor (tamaño órbita)
* e: Excentricidad (forma órbita)
* i: Inclinación (ángulo respecto eclíptica)
* Ω (om): Longitud nodo ascendente
* ω (w): Argumento del perihelio
* M (ma): Anomalía media (posición actual)

#### Ecuación de Kepler
```Python
# Resolver M = E - e·sin(E) para E (anomalía excéntrica)
E = M  # Aproximación inicial
for _ in range(10):
    E_new = E - (E - e * sin(E) - M) / (1 - e * cos(E))
    E = E_new

# Luego calcular anomalía verdadera (θ)
theta = 2 * atan(sqrt((1+e)/(1-e)) * tan(E/2))
```

#### Movimiento temporal
```Python
# La anomalía media cambia con el tiempo:
M(t) = M₀ + n·t
# donde n = 360°/periodo (movimiento medio)
```

