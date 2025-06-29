import streamlit as st
import time
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

def run_ils_process(process_id, total_iters, progress_dict):
    """Función que ejecuta el proceso ILS con progreso compartido"""
    for i in range(total_iters):
        # Simulación de trabajo
        time.sleep(0.1)
        progress_dict[process_id] = i + 1
    return f"Process {process_id} done"

def main():
    st.title("Optimización ILS con Multiprocessing")
    
    # Configuración
    total_iters = st.slider("Total de iteraciones", 10, 200, 100)
    num_processes = st.slider("Número de procesos", 1, 8, 4)
    
    if st.button("Iniciar Optimización"):
        # Crear el manager y diccionario compartido
        manager = Manager()
        progress_dict = manager.dict()
        
        # Inicializar el diccionario de progreso
        for i in range(num_processes):
            progress_dict[f"ILS_{i}"] = 0
        
        # Crear placeholders para las barras de progreso
        progress_placeholders = {}
        for i in range(num_processes):
            pid = f"ILS_{i}"
            progress_placeholders[pid] = st.progress(0, text=f"{pid}: 0%")
        
        # Crear el contenedor para el estado
        status_container = st.empty()
        
        try:
            # Ejecutar los procesos
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for i in range(num_processes):
                    futures.append(executor.submit(
                        run_ils_process, 
                        f"ILS_{i}", 
                        total_iters, 
                        progress_dict
                    ))
                
                # Loop de actualización del progreso
                while any(not f.done() for f in futures):
                    time.sleep(0.5)
                    
                    # Actualizar barras de progreso
                    for pid in progress_placeholders:
                        iter_count = progress_dict.get(pid, 0)
                        progress = min(iter_count / total_iters, 1.0)
                        progress_placeholders[pid].progress(
                            progress, 
                            text=f"{pid}: {int(progress*100)}%"
                        )
                    
                    # Mostrar estado general
                    completed_processes = sum(1 for f in futures if f.done())
                    status_container.info(
                        f"Procesos completados: {completed_processes}/{num_processes}"
                    )
                
                # Obtener resultados
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=1)
                        results.append(result)
                    except Exception as e:
                        results.append(f"Error: {e}")
                
                # Mostrar resultados finales
                status_container.success("¡Todos los procesos han terminado!")
                
                with st.expander("Ver resultados detallados"):
                    for result in results:
                        st.write(result)
                        
        except Exception as e:
            st.error(f"Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
