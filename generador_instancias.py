import json
import random

def generar_instancia(num_empleados=20, num_escritorios=9, num_grupos=4, num_zonas=2, filename="instancia.json"):
    """
    Genera una instancia similar a tu estructura y la guarda en JSON.
    
    Args:
        num_empleados: Número de empleados (default: 20)
        num_escritorios: Número de escritorios (default: 9)
        num_grupos: Número de grupos (default: 4)
        num_zonas: Número de zonas (default: 2)
        filename: Nombre del archivo JSON (default: "instancia.json")
    """
    
    # Listas básicas
    empleados = [f"E{i}" for i in range(num_empleados)]
    escritorios = [f"D{i}" for i in range(num_escritorios)]
    dias = ["L", "Ma", "Mi", "J", "V"]
    grupos = [f"G{i}" for i in range(num_grupos)]
    zonas = [f"Z{i}" for i in range(num_zonas)]
    
    # Distribuir escritorios en zonas
    escritorios_por_zona = num_escritorios // num_zonas
    resto = num_escritorios % num_zonas
    
    escritorios_z = {}
    indice = 0
    
    for i in range(num_zonas):
        cantidad = escritorios_por_zona + (1 if i < resto else 0)
        escritorios_z[f"Z{i}"] = escritorios[indice:indice + cantidad]
        indice += cantidad
    
    # Distribuir empleados en grupos
    empleados_por_grupo = num_empleados // num_grupos
    empleados_g = {}
    for i in range(num_grupos):
        inicio = i * empleados_por_grupo
        fin = inicio + empleados_por_grupo
        if i == num_grupos - 1:  # Último grupo toma los restantes
            fin = num_empleados
        empleados_g[f"G{i}"] = empleados[inicio:fin]
    
    # Generar preferencias de escritorios para cada empleado
    escritorios_e = {}
    for empleado in empleados:
        num_prefs = random.randint(3, min(8, num_escritorios))
        escritorios_e[empleado] = random.sample(escritorios, num_prefs)
    
    # Generar días de trabajo para cada empleado
    dias_e = {}
    for empleado in empleados:
        num_dias = random.randint(1, 3)
        dias_e[empleado] = random.sample(dias, num_dias)
    
    # Crear la instancia
    instancia = {
        "Employees": empleados,
        "Desks": escritorios,
        "Days": dias,
        "Groups": grupos,
        "Zones": zonas,
        "Desks_Z": escritorios_z,
        "Desks_E": escritorios_e,
        "Employees_G": empleados_g,
        "Days_E": dias_e
    }
    
    # Guardar en JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(instancia, f, indent=4, ensure_ascii=False)
    
    print(f"Instancia guardada en: {filename}")
    return instancia

# Ejemplo de uso
if __name__ == "__main__":
    
    # Generar otra instancia con 3 zonas
    instancia2 = generar_instancia(num_empleados=100, num_escritorios=43, num_grupos=10, num_zonas=7, filename="instancia_testeo3.json")