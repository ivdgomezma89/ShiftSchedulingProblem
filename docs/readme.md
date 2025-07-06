# 🧠📅 Asignación Inteligente de Turnos con DQN + ILS

Este proyecto implementa un sistema híbrido de **optimización de asignación de turnos** basado en **Deep Reinforcement Learning (DQN)** y **búsqueda local iterada (ILS)**. Está diseñado para encontrar soluciones óptimas considerando múltiples criterios organizacionales, todo a través de una **interfaz gráfica interactiva con Streamlit**.

---

## 🚀 Descripción del proyecto

La aplicación resuelve un **problema multiobjetivo** de asignación de puestos de trabajo a empleados con múltiples restricciones y preferencias, mediante:

- 🧠 **Fase 1 - DQN**: Generación de soluciones iniciales usando aprendizaje por refuerzo profundo con enmascaramiento de acciones.
- 🔍 **Fase 2 - ILS**: Refinamiento de las soluciones mediante operadores heurísticos y estrategias de intensificación/diversificación.
- ⚙️ Priorización **lexicográfica** de objetivos definida por el usuario.
- ⚡ Soporte de procesamiento paralelo para acelerar las búsquedas.

---

## 🎯 Objetivos de optimización

El usuario puede priorizar los siguientes objetivos:

1. ✅ Maximizar empleados asignados
2. 📆 Respetar días preferidos de presencialidad
3. 🪑 Mantener consistencia en el puesto
4. 👥 Coincidencia semanal de equipos
5. 🗺️ Dispersión de los grupos en zonas

Cada objetivo se prioriza con niveles jerárquicos: `Crítica`, `Alta`, `Media`, `Baja`, `Mínima`.

---

## 🧪 Funcionalidades de la aplicación

- 🎛️ **Panel de control**: configura prioridades, iteraciones y días presenciales.
- 📊 **Resultados generales**: muestra asignaciones, métricas clave y recomendaciones de reunión.
- 🧩 **Detalles de la asignación**: visualizaciones por grupos y zonas mediante mapas de calor.
- 💾 Soporte para cargar instancias JSON personalizadas o usar ejemplos predefinidos.

---

## 💡 Requisitos mínimos

-  🐍 Python 3.11.5
-  🧠 RAM: 16 GB (recomendado: 32 GB)
-  ⚙️ Procesador con múltiples núcleos (para aprovechar la paralelización)
-  🌐 Navegador web moderno (para visualizar la interfaz en Streamlit)

## ⚙️ Instrucciones de uso

Sigue estos pasos para ejecutar la aplicación en tu equipo local:

1. **Clona este repositorio**:
   ```bash
   git clone https://github.com/ivdgomezma89/ShiftSchedulingProblem.git
   cd ShiftSchedulingProblem

2.  **Crea y activa un entorno virtual** 
  ```bash
    # Crea el entorno virtual (recomendado)
    python -m venv venv    
    # Actívalo en Windows
    venv\Scripts\activate
    
    # Actívalo en macOS/Linux
    source venv/bin/activate
```
3. **Instala las dependencias:**
```bash
pip install -r requirements.txt
```
4. **Ejecuta la aplicación:**
```bash
streamlit run main.py
```
##  🙋 Autor
-  Desarrollado por Iván Darío Gómez Marín
-   Proyecto realizado como parte de un reto académico de optimización
