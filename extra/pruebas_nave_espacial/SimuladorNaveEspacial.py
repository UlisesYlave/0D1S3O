import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class SimuladorMisionSincronizada:
    def __init__(self):
        self.base_url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
        self.GM = 2.959122082855909e-04  # GM_sol en UA^3/día^2
        
    def _configurar_estilo_espacial(self, ax):
        """Configura el estilo espacial con fondo oscuro"""
        fig = ax.get_figure()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.line.set_color('white')
        ax.yaxis.line.set_color('white')
        ax.zaxis.line.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        ax.grid(True, color='gray', alpha=0.1)
    
    def calcular_punto_encuentro_optimo(self, params_tierra, params_ast, tiempo_max=1000):
        """Encuentra el tiempo óptimo de lanzamiento y viaje para el encuentro"""
        
        def distancia_encuentro(variables):
            tiempo_lanzamiento, tiempo_viaje = variables
            tiempo_encuentro = tiempo_lanzamiento + tiempo_viaje
            
            # Posición de la Tierra en el lanzamiento
            pos_tierra = self._calcular_posicion_exacta(params_tierra, tiempo_lanzamiento)
            
            # Posición del asteroide en el encuentro
            pos_ast = self._calcular_posicion_exacta(params_ast, tiempo_encuentro)
            
            # Distancia a minimizar
            return np.linalg.norm(pos_ast - pos_tierra)
        
        # Optimizar para encontrar el mejor punto de encuentro
        resultado = minimize(distancia_encuentro, [0, 300], 
                           bounds=[(0, tiempo_max), (50, tiempo_max)],
                           method='L-BFGS-B')
        
        tiempo_lanzamiento_optimo, tiempo_viaje_optimo = resultado.x
        distancia_minima = resultado.fun
        
        print(f"🎯 Punto de encuentro óptimo encontrado:")
        print(f"   - Tiempo de lanzamiento: {tiempo_lanzamiento_optimo:.1f} días")
        print(f"   - Tiempo de viaje: {tiempo_viaje_optimo:.1f} días") 
        print(f"   - Distancia mínima: {distancia_minima:.4f} UA")
        
        return tiempo_lanzamiento_optimo, tiempo_viaje_optimo
    
    def calcular_trayectoria_interceptacion(self, params_tierra, params_ast, 
                                          tiempo_lanzamiento, tiempo_viaje):
        """Calcula trayectoria que INTERCEPTA al asteroide"""
        
        tiempo_encuentro = tiempo_lanzamiento + tiempo_viaje
        
        # Posiciones clave
        pos_tierra_lanzamiento = self._calcular_posicion_exacta(params_tierra, tiempo_lanzamiento)
        pos_ast_encuentro = self._calcular_posicion_exacta(params_ast, tiempo_encuentro)
        
        # Velocidad orbital de la Tierra (aproximada)
        vel_orbital_tierra = self._calcular_velocidad_orbital(params_tierra, tiempo_lanzamiento)
        
        def ecuaciones_movimiento(t, estado):
            """Ecuaciones del movimiento con gravedad solar"""
            x, y, z, vx, vy, vz = estado
            r = np.sqrt(x**2 + y**2 + z**2)
            
            if r > 0.01:  # Evitar división por cero
                factor = -self.GM / (r**3)
                ax, ay, az = factor * x, factor * y, factor * z
            else:
                ax, ay, az = 0, 0, 0
            
            return [vx, vy, vz, ax, ay, az]
        
        def objetivo_impulso(impulso):
            """Optimiza el impulso inicial para llegar al asteroide"""
            # Dirección hacia el asteroide (vector unitario)
            direccion = (pos_ast_encuentro - pos_tierra_lanzamiento)
            direccion = direccion / np.linalg.norm(direccion)
            
            # Velocidad inicial = velocidad orbital + impulso
            vel_inicial = vel_orbital_tierra + direccion * impulso[0]
            estado_inicial = np.concatenate([pos_tierra_lanzamiento, vel_inicial])
            
            # Integrar trayectoria
            t_span = (0, tiempo_viaje)
            t_eval = np.linspace(0, tiempo_viaje, 100)
            
            solucion = solve_ivp(ecuaciones_movimiento, t_span, estado_inicial,
                               t_eval=t_eval, method='RK45', rtol=1e-8)
            
            # Posición final de la nave
            pos_final_nave = solucion.y[:3, -1]
            
            # Distancia al asteroide en el encuentro
            return np.linalg.norm(pos_final_nave - pos_ast_encuentro)
        
        # Optimizar el impulso para minimizar la distancia de encuentro
        resultado_impulso = minimize(objetivo_impulso, [0.02], bounds=[(0.005, 0.1)])
        impulso_optimo = resultado_impulso.x[0]
        
        print(f"🚀 Impulso óptimo calculado: {impulso_optimo:.6f} UA/día")
        print(f"   ({impulso_optimo * 1731.456:.1f} km/s)")
        
        # Calcular trayectoria final con impulso óptimo
        direccion = (pos_ast_encuentro - pos_tierra_lanzamiento)
        direccion = direccion / np.linalg.norm(direccion)
        vel_inicial = vel_orbital_tierra + direccion * impulso_optimo
        estado_inicial = np.concatenate([pos_tierra_lanzamiento, vel_inicial])
        
        t_span = (0, tiempo_viaje)
        t_eval = np.linspace(0, tiempo_viaje, 200)
        solucion_final = solve_ivp(ecuaciones_movimiento, t_span, estado_inicial,
                                 t_eval=t_eval, method='RK45', rtol=1e-8)
        
        return solucion_final.y[:3], tiempo_encuentro, impulso_optimo
    
    def _calcular_velocidad_orbital(self, params, tiempo):
        """Calcula la velocidad orbital real en un tiempo dado"""
        # Para simplificar, aproximamos la velocidad orbital circular
        a = params['a']
        velocidad = np.sqrt(self.GM / a)  # Velocidad circular
        
        # Dirección tangencial a la órbita
        pos_actual = self._calcular_posicion_exacta(params, tiempo)
        if np.linalg.norm(pos_actual) > 0:
            # Vector tangente (perpendicular al radio)
            tangente = np.cross(pos_actual, [0, 0, 1])  # Producto cruz con eje Z
            tangente = tangente / np.linalg.norm(tangente)
            return tangente * velocidad
        else:
            return np.array([0, velocidad, 0])
    
    def simular_interceptacion_precisa(self, asteroide_id):
        """Simulación PRECISA de interceptación"""
        print(f"🎯 SIMULACIÓN DE INTERCEPTACIÓN PRECISA")
        print(f"   Objetivo: Asteroide {asteroide_id}")
        
        # Obtener datos
        datos = self.obtener_datos_asteroide(asteroide_id)
        if not datos:
            print("❌ Error obteniendo datos")
            return
        
        params_ast = self.extraer_parametros_orbitales(datos)
        if not params_ast:
            print("❌ Error con parámetros orbitales")
            return
        
        nombre_ast = datos.get('object', {}).get('fullname', f'Asteroide {asteroide_id}')
        params_tierra = self._parametros_tierra()
        
        # 1. CALCULAR PUNTO DE ENCUENTRO ÓPTIMO
        print("\n1. Calculando ventana de lanzamiento óptima...")
        tiempo_lanzamiento, tiempo_viaje = self.calcular_punto_encuentro_optimo(params_tierra, params_ast)
        tiempo_encuentro = tiempo_lanzamiento + tiempo_viaje
        
        # 2. CALCULAR TRAYECTORIA DE INTERCEPTACIÓN
        print("\n2. Calculando trayectoria de interceptación...")
        trayectoria_nave, tiempo_encuentro, impulso = self.calcular_trayectoria_interceptacion(
            params_tierra, params_ast, tiempo_lanzamiento, tiempo_viaje)
        
        # 3. CALCULAR POSICIONES FUTURAS
        print("\n3. Calculando movimiento orbital...")
        # Movimiento del asteroide durante la misión
        tiempos_ast = np.linspace(0, tiempo_encuentro, 200)
        posiciones_ast = np.array([self._calcular_posicion_exacta(params_ast, t) for t in tiempos_ast]).T
        
        # Movimiento de la Tierra durante la misión
        posiciones_tierra = np.array([self._calcular_posicion_exacta(params_tierra, t) for t in tiempos_ast]).T
        
        # 4. CONFIGURAR VISUALIZACIÓN
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        self._configurar_estilo_espacial(ax)
        
        # Dibujar elementos estáticos
        self._dibujar_sistema_completo_mejorado(ax, params_tierra, params_ast, nombre_ast)
        
        # Dibujar trayectoria de la nave
        ax.plot(trayectoria_nave[0], trayectoria_nave[1], trayectoria_nave[2],
               'yellow', linewidth=3, alpha=0.6, label='Trayectoria nave')
        
        # 5. ANIMACIÓN DE INTERCEPTACIÓN
        print("\n4. Creando animación de interceptación...")
        animacion = self._crear_animacion_interceptacion(
            ax, trayectoria_nave, posiciones_ast, posiciones_tierra,
            tiempos_ast, tiempo_lanzamiento, tiempo_viaje, nombre_ast, impulso)
        
        plt.tight_layout()
        plt.show()
        return animacion
    
    def _crear_animacion_interceptacion(self, ax, trayectoria_nave, posiciones_ast, 
                                      posiciones_tierra, tiempos_ast, tiempo_lanzamiento,
                                      tiempo_viaje, nombre_ast, impulso):
        """Animación precisa de la interceptación"""
        fig = ax.get_figure()
        
        # Elementos dinámicos
        punto_nave, = ax.plot([], [], [], 'ro', markersize=12, alpha=1.0, zorder=10, label='Nave')
        punto_ast, = ax.plot([], [], [], 'co', markersize=10, alpha=1.0, zorder=9, label='Asteroide')
        punto_tierra, = ax.plot([], [], [], 'bo', markersize=8, alpha=1.0, zorder=8, label='Tierra')
        linea_trayectoria, = ax.plot([], [], [], 'yellow', linewidth=2, alpha=0.8, zorder=5)
        
        texto_info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white',
                              fontsize=10, bbox=dict(boxstyle="round", facecolor='black', alpha=0.9))
        
        def animar(frame):
            total_frames = 150
            tiempo_actual = (frame / total_frames) * (tiempo_lanzamiento + tiempo_viaje)
            
            # Índices para las posiciones
            idx_tiempo = min(len(tiempos_ast)-1, int(frame * len(tiempos_ast) / total_frames))
            
            # 1. POSICIÓN DEL ASTEROIDE (siempre correcta)
            pos_ast_ahora = posiciones_ast[:, idx_tiempo]
            punto_ast.set_data([pos_ast_ahora[0]], [pos_ast_ahora[1]])
            punto_ast.set_3d_properties([pos_ast_ahora[2]])
            
            # 2. POSICIÓN DE LA TIERRA (siempre correcta)
            pos_tierra_ahora = posiciones_tierra[:, idx_tiempo]
            punto_tierra.set_data([pos_tierra_ahora[0]], [pos_tierra_ahora[1]])
            punto_tierra.set_3d_properties([pos_tierra_ahora[2]])
            
            # 3. POSICIÓN DE LA NAVE (solo después del lanzamiento)
            tiempo_desde_lanzamiento = tiempo_actual - tiempo_lanzamiento
            
            if tiempo_desde_lanzamiento >= 0:
                # Encontrar índice en la trayectoria de la nave
                idx_nave = min(trayectoria_nave.shape[1]-1, 
                             int(tiempo_desde_lanzamiento * trayectoria_nave.shape[1] / tiempo_viaje))
                
                pos_nave = trayectoria_nave[:, idx_nave]
                punto_nave.set_data([pos_nave[0]], [pos_nave[1]])
                punto_nave.set_3d_properties([pos_nave[2]])
                
                # Trayectoria recorrida por la nave
                if idx_nave > 1:
                    linea_trayectoria.set_data(trayectoria_nave[0, :idx_nave],
                                             trayectoria_nave[1, :idx_nave])
                    linea_trayectoria.set_3d_properties(trayectoria_nave[2, :idx_nave])
                
                # Calcular distancia al asteroide
                distancia = np.linalg.norm(pos_nave - pos_ast_ahora)
                estado = "🚀 EN VUELO"
            else:
                distancia = np.linalg.norm(pos_tierra_ahora - pos_ast_ahora)
                estado = "⏰ ESPERANDO LANZAMIENTO"
                punto_nave.set_data([], [])
                punto_nave.set_3d_properties([])
                linea_trayectoria.set_data([], [])
                linea_trayectoria.set_3d_properties([])
            
            # 4. ACTUALIZAR INFORMACIÓN
            texto_info.set_text(
                f'🎯 INTERCEPTACIÓN: {nombre_ast.split()[0]}\n'
                f'📅 Día: {tiempo_actual:.0f}\n'
                f'🛰️ Estado: {estado}\n'
                f'📏 Distancia: {distancia:.4f} UA\n'
                f'🚀 Impulso: {impulso*1731.456:.1f} km/s'
            )
            
            # 5. DESTACAR INTERCEPTACIÓN
            if tiempo_desde_lanzamiento >= tiempo_viaje * 0.95 and tiempo_desde_lanzamiento <= tiempo_viaje * 1.05:
                # Casi en el punto de encuentro
                ax.plot([pos_nave[0], pos_ast_ahora[0]], 
                       [pos_nave[1], pos_ast_ahora[1]], 
                       [pos_nave[2], pos_ast_ahora[2]], 
                       'green', linewidth=2, alpha=0.8)
            
            return punto_nave, punto_ast, punto_tierra, linea_trayectoria, texto_info
        
        anim = animation.FuncAnimation(fig, animar, frames=150, interval=80, blit=True, repeat=True)
        return anim
    
    def _dibujar_sistema_completo_mejorado(self, ax, params_tierra, params_ast, nombre_ast):
        """Dibuja el sistema solar mejorado"""
        # Sol
        ax.scatter([0], [0], [0], color='yellow', s=400, alpha=0.9, label='Sol')
        
        # Órbitas
        orbita_tierra = self._calcular_orbita_completa(params_tierra)
        orbita_ast = self._calcular_orbita_completa(params_ast)
        
        ax.plot(orbita_tierra[0], orbita_tierra[1], orbita_tierra[2],
               'blue', alpha=0.3, linewidth=1.5, label='Órbita Tierra')
        ax.plot(orbita_ast[0], orbita_ast[1], orbita_ast[2],
               'cyan', alpha=0.3, linewidth=1.5, label=f'Órbita {nombre_ast.split()[0]}')
        
        ax.set_xlabel('X (UA)', color='white', fontsize=12)
        ax.set_ylabel('Y (UA)', color='white', fontsize=12)
        ax.set_zlabel('Z (UA)', color='white', fontsize=12)
        
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white',
                 loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Métodos auxiliares (mantener igual que antes)
    def _calcular_posicion_exacta(self, params, tiempo):
        a, e, i, om, w = params['a'], params['e'], np.radians(params['i']), np.radians(params['om']), np.radians(params['w'])
        n = 2 * np.pi / params['periodo']
        M = np.radians(params['ma']) + n * tiempo
        
        E = M
        for _ in range(15):
            E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            if abs(E_new - E) < 1e-10: break
            E = E_new
        
        theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        r = a * (1 - e * np.cos(E))
        x_orb, y_orb = r * np.cos(theta), r * np.sin(theta)
        
        R = self._matriz_rotacion_3d(om, i, w)
        return R @ np.array([x_orb, y_orb, 0])
    
    def _calcular_orbita_completa(self, params):
        puntos = 100
        return np.array([self._calcular_posicion_exacta(params, t) 
                        for t in np.linspace(0, params['periodo'], puntos)]).T
    
    def _parametros_tierra(self):
        return {'a': 1.000001, 'e': 0.0167086, 'i': 0.0, 'om': 0.0, 
                'w': 102.947, 'ma': 100.0, 'periodo': 365.256}
    
    def _matriz_rotacion_3d(self, om, i, w):
        Rz_om = np.array([[np.cos(om), -np.sin(om), 0], [np.sin(om), np.cos(om), 0], [0,0,1]])
        Rx_i = np.array([[1,0,0], [0,np.cos(i),-np.sin(i)], [0,np.sin(i),np.cos(i)]])
        Rz_w = np.array([[np.cos(w),-np.sin(w),0], [np.sin(w),np.cos(w),0], [0,0,1]])
        return Rz_om @ Rx_i @ Rz_w
    
    def obtener_datos_asteroide(self, asteroides_id):
        params = {'sstr': asteroides_id}
        try:
            response = requests.get(self.base_url, params=params)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def extraer_parametros_orbitales(self, datos):
        if not datos or 'orbit' not in datos: return None
        elementos = datos['orbit']['elements']
        parametros = {}
        for elemento in elementos:
            if elemento.get('name') and elemento.get('value'):
                try: parametros[elemento['name']] = float(elemento['value'])
                except: pass
        return {
            'a': parametros.get('a', 2.7), 'e': parametros.get('e', 0.1),
            'i': parametros.get('i', 5.0), 'om': parametros.get('om', 0.0),
            'w': parametros.get('w', 0.0), 'ma': parametros.get('ma', 0.0),
            'periodo': parametros.get('per', 1000)
        }

# Ejemplo de uso
if __name__ == "__main__":
    print("=== SIMULADOR DE INTERCEPTACIÓN PRECISA ===\n")
    
    simulador = SimuladorMisionSincronizada()
    
    # Probar interceptación precisa
    print("1. Interceptación a Ceres...")
    simulador.simular_interceptacion_precisa("1")
    
    print("\n2. Interceptación a Vesta...")
    simulador.simular_interceptacion_precisa("4")
    
    print("\n3. Interceptación a Eros...")
    simulador.simular_interceptacion_precisa("433")