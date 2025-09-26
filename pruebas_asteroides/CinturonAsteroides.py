import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
import json

class CinturonAsteroidesCompleto:
    def __init__(self):
        self.base_url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
        self.orbit_class_codes = {
            "MBA": "Main-belt Asteroid",
            "OMB": "Outer Main-belt Asteroid", 
            "IMB": "Inner Main-belt Asteroid",
            "AMO": "Amor",
            "APO": "Apollo",
            "ATE": "Aten"
        }
    
    def _configurar_estilo_espacial(self, ax):
        """Configura el estilo espacial con fondo oscuro"""
        # Fondo negro
        fig = ax.get_figure()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Color de ejes y labels en blanco
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        
        # Color de los ejes
        ax.xaxis.line.set_color('white')
        ax.yaxis.line.set_color('white')
        ax.zaxis.line.set_color('white')
        
        # Color de las etiquetas
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        # Color del título
        if ax.title.get_text():
            ax.title.set_color('white')
        
        # Grid con color tenue
        ax.grid(True, color='gray', alpha=0.1)
        
        # Bordes de los paneles
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    
    def buscar_asteroides_por_clase(self, clase="MBA", limite=50):
        """Busca asteroides por clase orbital usando el parámetro class"""
        params = {
            'class': clase,
            'limit': limite
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                print(f"Encontrados {len(data.get('data', []))} asteroides de clase {clase}")
                return data
            else:
                print(f"Error en búsqueda: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def obtener_lista_amplia_asteroides(self, limite=100):
        """Obtiene una lista amplia de asteroides del cinturón principal"""
        # Primero intentamos buscar por clase
        data = self.buscar_asteroides_por_clase("MBA", limite)
        
        if data and 'data' in data:
            asteroides = []
            for item in data['data']:
                if len(item) > 0:
                    # El primer elemento suele ser la designación
                    asteroides.append(item[0])
            return asteroides[:limite]
        else:
            # Fallback: lista manual de asteroides conocidos
            return self._lista_manual_asteroides(limite)
    
    def _lista_manual_asteroides(self, limite=50):
        """Lista manual de asteroides del cinturón principal"""
        # Asteroides numerados del cinturón principal
        return [str(i) for i in range(1, limite + 1)]
    
    def obtener_datos_asteroide(self, asteroides_id):
        """Obtiene datos completos de un asteroide"""
        params = {'sstr': asteroides_id}
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def extraer_parametros_orbitales(self, datos):
        """Extrae parámetros orbitales del array elements"""
        if not datos or 'orbit' not in datos or 'elements' not in datos['orbit']:
            return None
        
        elementos = datos['orbit']['elements']
        parametros = {}
        
        for elemento in elementos:
            nombre = elemento.get('name')
            valor = elemento.get('value')
            if nombre and valor:
                try:
                    parametros[nombre] = float(valor)
                except:
                    pass
        
        # Información del objeto
        objeto = datos.get('object', {})
        nombre = objeto.get('fullname', objeto.get('shortname', 'Asteroide'))
        clase = objeto.get('orbit_class', {}).get('name', 'Desconocida')
        
        return {
            'nombre': nombre,
            'clase': clase,
            'a': parametros.get('a', 2.7),  # Semieje mayor (UA)
            'e': parametros.get('e', 0.1),  # Excentricidad
            'i': parametros.get('i', 5.0),  # Inclinación (grados)
            'om': parametros.get('om', 0.0),  # Long. nodo ascendente (grados)
            'w': parametros.get('w', 0.0),   # Argumento perihelio (grados)
            'ma': parametros.get('ma', 0.0), # Anomalía media (grados)
            'periodo': parametros.get('per', 1000)  # Período orbital (días)
        }
    
    def calcular_posicion_orbital(self, params, tiempo=0):
        """Calcula la posición orbital usando ecuaciones de Kepler"""
        # Convertir a radianes
        a = params['a']
        e = params['e']
        i = np.radians(params['i'])
        om = np.radians(params['om'])
        w = np.radians(params['w'])
        
        # Anomalía media en el tiempo (simplificado)
        ma_rad = np.radians(params['ma'] + tiempo * 360 / params['periodo'])
        
        # Resolver ecuación de Kepler para anomalía excéntrica (E)
        E = ma_rad  # Aproximación inicial
        for _ in range(10):  # Iteración de Newton
            E_new = E - (E - e * np.sin(E) - ma_rad) / (1 - e * np.cos(E))
            if abs(E_new - E) < 1e-8:
                break
            E = E_new
        
        # Anomalía verdadera
        theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        
        # Coordenadas en el plano orbital
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)
        z_orb = 0
        
        # Rotación a coordenadas eclípticas
        x1 = x_orb * np.cos(om) - y_orb * np.sin(om)
        y1 = x_orb * np.sin(om) + y_orb * np.cos(om)
        z1 = z_orb
        
        x2 = x1
        y2 = y1 * np.cos(i) - z1 * np.sin(i)
        z2 = y1 * np.sin(i) + z1 * np.cos(i)
        
        x = x2 * np.cos(w) - y2 * np.sin(w)
        y = x2 * np.sin(w) + y2 * np.cos(w)
        z = z2
        
        return x, y, z, r
    
    def generar_orbita_completa(self, params, puntos=360):
        """Genera puntos completos de la órbita"""
        x_points, y_points, z_points = [], [], []
        
        for t in np.linspace(0, 2*np.pi, puntos):
            # Para órbita completa, variamos la anomalía media
            params_temp = params.copy()
            params_temp['ma'] = np.degrees(t)  # Convertir a grados
            
            x, y, z, r = self.calcular_posicion_orbital(params_temp)
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
        
        return np.array(x_points), np.array(y_points), np.array(z_points)
    
    def _dibujar_sistema_solar(self, ax):
        """Dibuja el sistema solar interno con estilo espacial"""
        # Configurar estilo primero
        self._configurar_estilo_espacial(ax)
        
        # Sol con efecto brillante
        ax.scatter([0], [0], [0], color='yellow', s=400, label='Sol', alpha=0.9)
        ax.scatter([0], [0], [0], color='orange', s=200, alpha=0.6)
        
        # Planetas internos con colores más vibrantes
        for dist, color, nombre in [(1.0, 'cyan', 'Tierra'), (1.52, 'red', 'Marte')]:
            theta = np.linspace(0, 2*np.pi, 100)
            x = dist * np.cos(theta)
            y = dist * np.sin(theta)
            ax.plot(x, y, np.zeros_like(x), color=color, linestyle='--', alpha=0.4, label=nombre)
        
        # Zona del cinturón
        theta = np.linspace(0, 2*np.pi, 100)
        for r in [2.2, 3.3]:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, np.zeros_like(x), 'g--', alpha=0.1)
    
    def visualizar_cinturon_masivo(self, limite_asteroides=20):
        """Visualización con muchos asteroides del cinturón - estilo espacial"""
        print("Obteniendo lista de asteroides...")
        asteroides_ids = self.obtener_lista_amplia_asteroides(limite_asteroides)
        
        # Crear figura con fondo oscuro
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        # Sistema solar con estilo espacial
        self._dibujar_sistema_solar(ax)
        
        asteroides_procesados = 0
        datos_asteroides = []
        
        for i, asteroide_id in enumerate(asteroides_ids):
            if asteroides_procesados >= limite_asteroides:
                break
                
            print(f"Procesando {asteroide_id} ({i+1}/{len(asteroides_ids)})...")
            datos = self.obtener_datos_asteroide(asteroide_id)
            
            if datos:
                params = self.extraer_parametros_orbitales(datos)
                if params and params['clase'] in ['Main-belt Asteroid', 'Outer Main-belt Asteroid', 'Inner Main-belt Asteroid']:
                    # Generar órbita
                    x, y, z = self.generar_orbita_completa(params)
                    
                    # Colores espaciales (azulados y verdosos)
                    distancia_media = params['a']
                    color = plt.cm.plasma(distancia_media / 4.0)
                    
                    # Dibujar órbita con efecto brillante
                    ax.plot(x, y, z, color=color, alpha=0.8, linewidth=1.2)
                    
                    # Posición actual con efecto de brillo
                    x_curr, y_curr, z_curr, r_curr = self.calcular_posicion_orbital(params)
                    ax.scatter([x_curr], [y_curr], [z_curr], color=color, s=30, alpha=0.9)
                    ax.scatter([x_curr], [y_curr], [z_curr], color='white', s=10, alpha=0.6)
                    
                    datos_asteroides.append((params, x_curr, y_curr, z_curr))
                    asteroides_procesados += 1
        
        print(f"\n✅ Asteroides procesados: {asteroides_procesados}")
        
        # Estadísticas
        if datos_asteroides:
            distancias = [params['a'] for params, _, _, _ in datos_asteroides]
            print(f"📊 Distancia media: {np.mean(distancias):.2f} UA")
            print(f"📏 Rango: {min(distancias):.2f} - {max(distancias):.2f} UA")
        
        # Configuración final del estilo
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X (UA)', color='white', fontsize=12)
        ax.set_ylabel('Y (UA)', color='white', fontsize=12)
        ax.set_zlabel('Z (UA)', color='white', fontsize=12)
        ax.set_title(f'Cinturón de Asteroides - {asteroides_procesados} objetos', 
                     color='white', fontsize=14, pad=20)
        
        # Leyenda con estilo oscuro
        legend = ax.legend(facecolor='black', edgecolor='white', 
                          labelcolor='white', loc='upper left',
                          bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.show()
    
    def crear_animacion_movimiento(self, limite_asteroides=10, frames=100):
        """Animación del movimiento orbital - estilo espacial"""
        asteroides_ids = self.obtener_lista_amplia_asteroides(limite_asteroides)
        
        # Configurar estilo oscuro
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        # Preparar datos de asteroides
        asteroides_data = []
        for asteroide_id in asteroides_ids[:limite_asteroides]:
            datos = self.obtener_datos_asteroide(asteroide_id)
            if datos:
                params = self.extraer_parametros_orbitales(datos)
                if params:
                    asteroides_data.append(params)
        
        # Almacenar trayectorias recientes
        trayectorias = {i: [] for i in range(len(asteroides_data))}
        
        def animar(frame):
            ax.clear()
            self._dibujar_sistema_solar(ax)
            
            tiempo = frame / frames * 10  # Mayor rango temporal para movimiento visible
            
            for i, params in enumerate(asteroides_data):
                # Actualizar anomalía media para simular movimiento
                params_temp = params.copy()
                params_temp['ma'] = params['ma'] + tiempo * 360 / params['periodo']
                
                x, y, z, r = self.calcular_posicion_orbital(params_temp, tiempo)
                color = plt.cm.viridis(i / len(asteroides_data))
                
                # Órbita completa (tenue)
                x_orb, y_orb, z_orb = self.generar_orbita_completa(params)
                ax.plot(x_orb, y_orb, z_orb, color=color, alpha=0.2, linewidth=0.8)
                
                # Almacenar posición actual para trayectoria
                if len(trayectorias[i]) < 20:  # Mantener últimas 20 posiciones
                    trayectorias[i].append((x, y, z))
                else:
                    trayectorias[i].pop(0)
                    trayectorias[i].append((x, y, z))
                
                # Dibujar trayectoria reciente
                if len(trayectorias[i]) > 1:
                    tray_x, tray_y, tray_z = zip(*trayectorias[i])
                    # Hacer la trayectoria más visible con gradiente de color
                    for j in range(len(tray_x) - 1):
                        alpha = (j + 1) / len(tray_x) * 0.8
                        ax.plot(tray_x[j:j+2], tray_y[j:j+2], tray_z[j:j+2], 
                               color=color, alpha=alpha, linewidth=2)
                
                # Posición actual con efecto de estrella
                ax.scatter([x], [y], [z], color=color, s=80, alpha=0.9, marker='o')
                ax.scatter([x], [y], [z], color='white', s=30, alpha=0.7, marker='*')
            
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_zlim([-2, 2])
            ax.set_xlabel('X (UA)', color='white')
            ax.set_ylabel('Y (UA)', color='white')
            ax.set_zlabel('Z (UA)', color='white')
            ax.set_title(f'Movimiento Orbital - Frame {frame}/{frames}', 
                        color='white', fontsize=12)
            return []
        
        anim = animation.FuncAnimation(fig, animar, frames=frames, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
        return anim

# Ejemplo de uso
if __name__ == "__main__":
    print("=== VISUALIZADOR COMPLETO DEL CINTURÓN - ESTILO ESPACIAL ===\n")
    
    vis = CinturonAsteroidesCompleto()
    
    # 1. Visualización estática con muchos asteroides
    print("1. Visualización del cinturón completo...")
    vis.visualizar_cinturon_masivo(limite_asteroides=30)
    
    # 2. Animación con trayectoria reciente
    print("\n2. Creando animación con trayectorias...")
    vis.crear_animacion_movimiento(limite_asteroides=5, frames=80)