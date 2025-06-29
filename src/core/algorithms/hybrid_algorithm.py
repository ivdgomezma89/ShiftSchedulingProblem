import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple,Optional

from multiprocessing import Manager
from ..enviroment.enviroment import ShiftSchedulerEnvMaskeable
from ..algorithms.local_search import DeskAssignmentILS
from ..agent.maskeabledqn import MaskeableDQN
from ..agent.dqnconfig import DQNConfig
import numpy as np

def run_single_ils_optimization(args,progress_dict) -> Tuple:
    """
    Función estática para ejecutar una optimización ILS individual
    
    Args:
        args: tupla con (solver_id, env, initial_solution, max_iterations, perturbation_strength)
        
    Returns:
        tupla con (solver_id, best_solution, best_score, best_components)
    """
    solver_id, env, initial_solution, max_iterations, perturbation_strength_interval, objective_weights= args
    
    # Crear instancia del ILS
    ils_solver = DeskAssignmentILS(entorno=env, prioridad_objetivos=objective_weights)
    
    # Ejecutar optimización
    best_solution, best_score = ils_solver.ils(
        initial_solution=initial_solution,
        max_iterations=max_iterations,
        perturbation_strength_interval=perturbation_strength_interval,
        progress_dict=progress_dict,
        solver_id=solver_id
    )
    
    return solver_id, best_solution, best_score, ils_solver
    

class ShiftSchedulerOptimizer:
    """
    Clase para optimizar horarios de turnos usando DQN + ILS
    """
    
    def __init__(self, instance: Dict,
                 dqn_timesteps: int = 50000,
                 ils_timesteps: int = 500, 
                 required_in_office_day: int = 2,
                 objective_weights: List[float] = [1, 1, 1, 1, 1]):
        """
        Inicializa el optimizador
        
        Args:
            instance: Nombre de la instancia a usar
            required_in_office_day: Días requeridos en oficina
        """
        self.instance = instance
        self.required_in_office_day = required_in_office_day
        self.dqn_timesteps= dqn_timesteps
        self.ils_timesteps= ils_timesteps
        self.objective_weights= objective_weights

        self.perturbation_strength_interval= [3,4]
        nro_emp= len(instance['Employees'])

        if nro_emp >= 20 and nro_emp <= 40:
            self.perturbation_strength_interval= np.array([2, np.ceil(0.3*nro_emp)])
        elif nro_emp <= 60:
            self.perturbation_strength_interval= np.array([2, np.ceil(0.2*nro_emp)])
        elif nro_emp > 60:
            self.perturbation_strength_interval= np.array([2, np.ceil(0.15*nro_emp)])


        self.custom_configs = [
                {
                    'solver_id': 1,
                    'solution_key': 'best_emp_no_asignados',
                    'ils_timesteps':self.ils_timesteps ,
                    'perturbation_strength_interval': self.perturbation_strength_interval,
                    'objective_weights': self.objective_weights
                },
                {
                    'solver_id': 2,
                    'solution_key': 'best_dias_pref',
                    'ils_timesteps': self.ils_timesteps,
                    'perturbation_strength_interval': self.perturbation_strength_interval,
                    'objective_weights': self.objective_weights
                },
                {
                    'solver_id': 3,
                    'solution_key': 'best_desk_consistency',
                    'ils_timesteps': self.ils_timesteps,
                    'perturbation_strength_interval': self.perturbation_strength_interval,
                    'objective_weights': self.objective_weights
                },

                {
                    'solver_id': 4,
                    'solution_key': 'best_team_proximity',
                    'ils_timesteps': self.ils_timesteps,
                    'perturbation_strength_interval': self.perturbation_strength_interval,
                    'objective_weights': self.objective_weights
                },

                {
                    'solver_id': 5,
                    'solution_key': 'best_asistencia_equipos',
                    'ils_timesteps': self.ils_timesteps,
                    'perturbation_strength_interval': self.perturbation_strength_interval,
                    'objective_weights': self.objective_weights
                },

            ]

        
        # Configuraciones predefinidas para cada instancia
        self.params =  {
            'learning_rate': 0.00073, 
            'batch_size': 1000, 
            'gamma': 0.995,
            'exploration_fraction': 0.2, 
            'exploration_final_eps': 0.012135216733922947, 
            'target_update_interval': 20000            
        }
        
        self.policy_kwargs = {'net_arch': [80] * 3}
        
        
        # Inicializar componentes
        self.env = None
        self.model = None
        self.is_trained = False
        self._setup_multiprocessing()

        self.manager = Manager()
        self.progress_dict = self.manager.dict()
        self.progress_dict_drl = self.manager.dict()
        self.is_running = False

    def _setup_multiprocessing(self):
        """
        Configura multiprocessing de forma segura
        """
        try:
            # Solo configurar si no se ha configurado antes
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn', force=True)
            print("✓ Multiprocessing configurado correctamente")
        except RuntimeError as e:
            # Ya está configurado, no es un problema
            print(f"Multiprocessing ya configurado: {e}")

    def get_progress(self):
        """Método para que la interfaz consulte el progreso del ILS"""
        return dict(self.progress_dict) 
    
    def get_progress_drl(self):
        """Método para que la interfaz consulte el progreso del DRL"""
        return dict(self.progress_dict_drl) 


    def is_optimization_running(self):
        """Verifica si la optimización está corriendo"""
        return self.is_running   
    
    def initialize_environment(self):
        """
        Inicializa el entorno de trabajo
        
        Args:
            instances_list: Diccionario con las instancias disponibles
        """
            
        self.env = ShiftSchedulerEnvMaskeable(
            instance=self.instance, 
            required_in_office_day=self.required_in_office_day
        )
        
    def train_dqn_model(self,  verbose: int = 1):
        """
        Entrena el modelo DQN
        
        Args:
            total_timesteps: Número total de pasos de entrenamiento
            seed: Semilla para reproducibilidad
            verbose: Nivel de verbosidad
        """
        if self.env is None:
            raise ValueError("Debe inicializar el entorno primero con initialize_environment()")
            
        # if self.instance not in self.params:
        #     raise ValueError(f"No hay parámetros definidos para la instancia '{self.instance}'")
            
        # Crear configuración
        config = DQNConfig(**self.params)
        
        # Crear y entrenar modelo
        self.model = MaskeableDQN(
            self.env, 
            config=config, 
            policy_kwargs=self.policy_kwargs,
            verbose=verbose
        )    

        self.progress_dict_drl['drl'] = 0  
        
        self.model.learn(total_timesteps=self.dqn_timesteps, progress_dict_drl= self.progress_dict_drl)
    
        self.is_trained = True

        
    def run_ils_optimization_parallel(self, max_workers: Optional[int] = None) -> List[Tuple]:
        """
        Ejecuta optimizaciones ILS en paralelo
        
        Args:
            optimization_configs: Lista de configuraciones para ILS. Si es None, usa configuración por defecto
            max_workers: Número máximo de workers. Si es None, usa mp.cpu_count()
            
        Returns:
            Lista de resultados (solver_id, best_solution, best_score, ils_solver)
        """
        if not self.is_trained:
            raise ValueError("Debe entrenar el modelo DQN primero con train_dqn_model()")
            
        self.is_running = False 
        # Preparar argumentos para cada proceso
        args_list = []
        for config in self.custom_configs:
            solution_key = config['solution_key']
            if solution_key not in self.model.best_solutions:
                print(f"Advertencia: Clave '{solution_key}' no encontrada en best_solutions")
                continue
                
            args_list.append((
                config['solver_id'],
                self.env,
                self.model.best_solutions[solution_key]['solution'],
                config['ils_timesteps'],
                config['perturbation_strength_interval'],
                config['objective_weights']
            ))

            self.progress_dict[config['solver_id']] = 0


            

         
        if not args_list:
            raise ValueError("No hay configuraciones válidas para ejecutar")
            
        print(f"Ejecutando {len(args_list)} optimizaciones ILS...")

        
        # Ejecutar en paralelo
        if max_workers is None:
            max_workers = mp.cpu_count()
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
      
            future_to_solver = {
                executor.submit(run_single_ils_optimization, args, self.progress_dict): args[0]
                for args in args_list
            }
            
            results = []
            for future in as_completed(future_to_solver):
                solver_id = future_to_solver[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Solver {solver_id} completado.")
                except Exception as exc:
                    print(f"Solver {solver_id} generó una excepción: {exc}")
        
        self.is_running = False
        return results

    def _run_ils_optimization_sequential(self, args_list: List[Tuple]) -> List[Tuple]:
        """
        Método de respaldo para ejecutar optimizaciones secuencialmente
        """
        print("Ejecutando optimizaciones secuencialmente...")
        results = []
        
        for i, args in enumerate(args_list):
            solver_id = args[0]
            try:
                print(f"Ejecutando solver {solver_id} ({i+1}/{len(args_list)})...")
                result = self._run_single_ils_optimization(args)
                results.append(result)
                print(f"Solver {solver_id} completado.")
            except Exception as exc:
                print(f"Solver {solver_id} generó una excepción: {exc}")
        
        return results
    

    def print_results(self, results: List[Tuple]):
        """
        Imprime los resultados de las optimizaciones
        
        Args:
            results: Lista de resultados de run_ils_optimization_parallel
        """
        for solver_id, best_solution, best_score, ils_solver in results:
            print(f"\n=== Resultados del Solver {solver_id} ===")
            print(f"Mejor score: {best_score}")
            #ils_solver.print_solution_summary(best_solution)
    
    def get_available_solutions(self) -> List[str]:
        """
        Retorna las claves de soluciones disponibles después del entrenamiento
        
        Returns:
            Lista de claves de soluciones disponibles
        """
        if not self.is_trained:
            return []
        return list(self.model.best_solutions.keys())
    
    def compare_solutions(self, results):
        " función para seleccionar la mejor solución con base en los criterios de prioridad (lexicograficamente)"

        rst = results[0]
        best_solution = rst[1]
        best_tuple = tuple(rst[2][p] for p in self.objective_weights)
        best_score = rst[2]


        for solver_id, solution, score, ils_solver in results:         
            test_tuple = tuple(score[p] for p in self.objective_weights)
            if test_tuple > best_tuple:
                best_solution = solution
                best_tuple = test_tuple
                best_score = score

        return (best_score, best_solution) 

    
    def run_complete_optimization(self,progreso_callback=None) -> List[Tuple]:
        """
        Ejecuta el proceso completo de optimización: inicialización, entrenamiento DQN e ILS
        
        Args:
            instances_list: Diccionario con las instancias disponibles
            dqn_timesteps: Pasos de entrenamiento para DQN
            optimization_configs: Configuraciones para ILS
            
        Returns:
            Tupla con (mejor score, mejor solución)
        """


        # Paso 1: Inicializar entorno
        self.initialize_environment()
        
        # Paso 2: Entrenar DQN
        self.train_dqn_model()   
        
        
        # Paso 3: Ejecutar ILS
        try:
            results = self.run_ils_optimization_parallel()
        except Exception as e:
            print(f"Error al ejecutar optimización paralela: {e}")
            results = self._run_ils_optimization_sequential(self.custom_configs)
        
        # Paso 4: Obtener la mejor solución encontrada 
        self.print_results(results)
        

        best_score, best_solution = self.compare_solutions(results)


        
        return (best_score, best_solution)
    


    
if __name__ == "__main__":
    ShiftSchedulerOptimizer().run_complete_optimization()

