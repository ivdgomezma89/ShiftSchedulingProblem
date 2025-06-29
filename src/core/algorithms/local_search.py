

import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from ..enviroment import ShiftSchedulerEnvMaskeable


class DeskAssignmentILS:
    def __init__(self, entorno: ShiftSchedulerEnvMaskeable, prioridad_objetivos: np.array):

        """
        Inicializa el objeto DeskAssignmentILS con el entorno y prioridades para resolver el problema de asignaci n de puestos.

        Args:
            entorno (ShiftSchedulerEnvMaskeable): Entorno del problema de programaci n de turnos.
            prioridad_objetivos (np.array): Vector de prioridades para los objetivos del problema (empleados no asignados, d as preferidos,
              consistencia de puestos, proximidad de equipos y asistencia de equipos).
        """

        self.entorno = entorno
        self.priorities = np.array(prioridad_objetivos)


        self.employees = entorno.employees
        self.desks = entorno.desks
        self.days = entorno.days
        self.groups = entorno.groups
        self.zones = entorno.zones
        self.desks_z = entorno.desks_z
        self.desks_e = entorno.desks_e
        self.employees_g = entorno.employees_g
        self.days_e = entorno.days_e
        self.zone_map= entorno.zone_map
        self.required_in_office_day = entorno.required_in_office_day
        self.employees_map = entorno.employees_map

        self.zone_reverse_map= entorno.zone_reverse_map
        self.desks_reverse_map = entorno.desks_reverse_map
        self.groups_reverse_map= entorno.groups_reverse_map
        self.employees_reverse_map = entorno.employees_reverse_map
        self.days_reverse_map = entorno.days_reverse_map

        self.num_presencial_dias_idx={emp: self.required_in_office_day for emp in self.employees_map.keys()}


        # Crear mapeos de índices

        # self.group_to_idx = {group: i for i, group in enumerate(groups)}

        # # Convertir diccionarios a índices
        self.desks_e_idx = {self.employees_reverse_map[emp]: [self.desks_reverse_map[desk] for desk in desks]
                           for emp, desks in self.desks_e.items()}

        self.days_e_idx = {self.employees_reverse_map[emp]: [self.days_reverse_map[day] for day in days]
                           for emp, days in self.days_e.items()}


        # Crear mapeo de zona por puesto
        self.desk_zone = {}
        for zone, desks_in_zone in self.desks_z.items():
            zone_idx = self.zone_reverse_map[zone]
            for desk in desks_in_zone:
                desk_idx = self.desks_reverse_map[desk]
                self.desk_zone[desk_idx] = zone_idx

        # Crear grupos por empleado
        self.employee_groups = {}
        for group, employees_in_group in self.employees_g.items():
            group_idx = self.groups_reverse_map[group]
            for emp in employees_in_group:
                emp_idx = self.employees_reverse_map[emp]
                if emp_idx not in self.employee_groups:
                    self.employee_groups[emp_idx] = []
                self.employee_groups[emp_idx].append(group_idx)

        # Días de reunión por grupo (variable de decisión)
        self.meeting_days = {}

    def generate_initial_solution(self) -> np.ndarray:
        """Genera una solución inicial factible"""
        solution = np.full((len(self.days), len(self.desks)), -1, dtype=int)

        # Paso 1: Asignar días de reunión para cada grupo
        self.meeting_days = {}
        for group_idx in range(len(self.groups)):
            # Elegir día aleatorio para reunión del grupo
            self.meeting_days[group_idx] = random.randint(0, len(self.days) - 1)

        # Paso 2: Crear lista de asignaciones obligatorias (días de reunión)
        mandatory_assignments = []
        for group_idx, meeting_day in self.meeting_days.items():
            group_name = self.groups[group_idx]
            if group_name in self.employees_g:
                for emp_name in self.employees_g[group_name]:
                    emp_idx = self.employees_reverse_map[emp_name]
                    mandatory_assignments.append((meeting_day, emp_idx))

        # Paso 3: Asignar puestos para días obligatorios
        for day_idx, emp_idx in mandatory_assignments:
            available_desks = [d for d in self.desks_e_idx[emp_idx]
                             if solution[day_idx, d] == -1]
            if available_desks and not emp_idx in solution[day_idx]:
                desk_idx = random.choice(available_desks)
                solution[day_idx, desk_idx] = emp_idx

        # Paso 4: Completar días faltantes para cada empleado
        for emp_idx in range(len(self.employees)):
            current_days = np.sum(solution == emp_idx)
            required_days = self.num_presencial_dias_idx[emp_idx]

            remaining_days = required_days - current_days
            if remaining_days > 0:
                # Buscar días disponibles
                preferred_days = self.days_e_idx.get(emp_idx, list(range(len(self.days))))
                available_days = []

                for day_idx in preferred_days:
                    if np.sum(solution[day_idx] == emp_idx) == 0:  # Empleado no asignado este día
                        available_desks = [d for d in self.desks_e_idx[emp_idx]
                                         if solution[day_idx, d] == -1]
                        if available_desks:
                            available_days.append(day_idx)

                # Asignar días faltantes
                days_to_assign = min(remaining_days, len(available_days))
                selected_days = random.sample(available_days, days_to_assign)

                for day_idx in selected_days:
                    available_desks = [d for d in self.desks_e_idx[emp_idx]
                                     if solution[day_idx, d] == -1]
                    if available_desks and not emp_idx in solution[day_idx]:
                        desk_idx = random.choice(available_desks)
                        solution[day_idx, desk_idx] = emp_idx
        return solution

    def perturbation(self, solution: np.ndarray, strength: int = 3) -> np.ndarray:
        """Aplica perturbación a la solución"""
        perturbed_solution = solution.copy()

        day_selected= np.random.choice(len(self.days))
        # elegir solo valores diferentes de -1
        options= perturbed_solution[day_selected]
        options= options[options != -1]
        if len(options) >= strength and np.random.rand() < 0.5:  # se realiza intercambio de puestos en el mismo dia la mitad de las veces    

            emp_selected= np.random.choice(options, size=strength, replace=False)

            # Encontrar sus asignaciones actuales
            for emp_idx in emp_selected:
                for desk_idx in range(len(self.desks)):
                    if perturbed_solution[day_selected, desk_idx] == emp_idx:
                        perturbed_solution[day_selected, desk_idx] = -1

            for emp_idx in emp_selected:
                available_desks = [d for d in self.desks_e_idx[emp_idx]
                                 if perturbed_solution[day_selected, d] == -1]
                if available_desks:
                    desk_idx = random.choice(available_desks)
                    perturbed_solution[day_selected, desk_idx] = emp_idx

        else: # se realizan strength cantidad de intercambios de dias aleatorios conservando el mismo puesto
            interchange_count= 0
            while interchange_count <= strength:
                day_selected= np.random.choice(len(self.days), size=2, replace=False)
                desk_selected= np.random.choice(len(self.desks), size=1)

                emp_selected1= perturbed_solution[day_selected[0], desk_selected[0]]
                emp_selected2= perturbed_solution[day_selected[1], desk_selected[0]]

                # validar que el intercambio sea valido
                if emp_selected1 not in perturbed_solution[day_selected[1]] and emp_selected2 not in perturbed_solution[day_selected[0]]:
                    perturbed_solution[day_selected[0], desk_selected[0]] = emp_selected2
                    perturbed_solution[day_selected[1], desk_selected[0]] = emp_selected1
                    interchange_count += 1
      
        

        return perturbed_solution

    def ils(self, initial_solution: Optional[np.array]= None,
            max_iterations: int = 100, 
            perturbation_strength_interval: np.ndarray = np.array([1, 5]), 
            progress_dict: Optional[Dict[str, int]] = None,
            solver_id: Optional[int] = None,
            verbose=True,) -> Tuple[np.ndarray, float]:
        """Algoritmo ILS principal"""
        print("Realizando busqueda local...")

        strength = int(perturbation_strength_interval[0])
        contador_no_mejora = 0

        current_solution = self.generate_initial_solution() if initial_solution is None else initial_solution.copy()
        #print(f"Solución inicial: {current_solution}")
        current_solution_fitness= [value for key, value in self.entorno.evaluate_solution(current_solution)['score_discr'].items()]

        self.best_solution = current_solution.copy()
        best_score = current_solution_fitness

        local_optimum= self.local_search_smart(current_solution)
        local_score= [value for key, value in self.entorno.evaluate_solution(local_optimum)['score_discr'].items()]

        if self.evaluate_acceptance_criterio(best_score, local_score): # evalua la solución optimizando lexicograficamente
            self.best_solution = local_optimum.copy()
            best_score = local_score


        for iteration in range(max_iterations):
            
            # Perturbación
            perturbed_solution = self.perturbation(self.best_solution, strength)

            # Búsqueda local
            #local_optimum = self.local_search(perturbed_solution)
            local_optimum= self.local_search_smart(perturbed_solution)
            local_score = [value for key, value in self.entorno.evaluate_solution(local_optimum)['score_discr'].items()]

            # Actualizar mejor solución
            if self.evaluate_acceptance_criterio(best_score, local_score): # evalua la solución optimizando lexicograficamente
                self.best_solution = local_optimum.copy()
                best_score = local_score
                #print(f"Iteración {iteration + 1}: Mejor solución encontrada")
                strength = int(perturbation_strength_interval[0])
                contador_no_mejora = 0
                #print(f"Iteración {iteration + 1}: Nueva mejor solución - Score: {best_score:.2f}")
            else:
                contador_no_mejora += 1
                if contador_no_mejora % 10 == 0 and strength < int(perturbation_strength_interval[1]):
                    strength += 1
                #print(f"Iteración {iteration + 1}: Se aunto la perturbación a {strength}")
           
            progress_dict[solver_id] = iteration+1        
                    


            # Cada 50 iteraciones, mostrar progreso
            # if (iteration + 1) % 20 == 0 and verbose:
            #     print(f"Iteración {iteration + 1}: Mejor solución encontrada")
            #   pbar.set_postfix({'best_score': best_score})

        #print(f"\nOptimización completada. Mejor score: {best_score:.2f}")
        return self.best_solution, best_score

    def print_solution_summary(self, solution: np.ndarray):
        """Imprime un resumen de la solución"""
        print("\n=== RESUMEN DE LA SOLUCIÓN ===")

        # print("\nAsignaciones por día:")
        # for day_idx, day_name in enumerate(self.days):
        #     print(f"\n{day_name}:")
        #     for desk_idx, desk_name in enumerate(self.desks):
        #         emp_idx = solution[day_idx, desk_idx]
        #         if emp_idx != -1:
        #             emp_name = self.employees[emp_idx]
        #             print(f"  {desk_name}: {emp_name}")

        print("\nDías de presencialidad por empleado:")
        for emp_idx, emp_name in enumerate(self.employees):
            assigned_days = []
            for day_idx, day_name in enumerate(self.days):
                if np.sum(solution[day_idx] == emp_idx) > 0:
                    assigned_days.append(day_name)
            required = self.num_presencial_dias_idx[emp_idx]
            print(f"{emp_name}: {len(assigned_days)}/{required} días - {assigned_days}")

        print("\nScore discriminado:")
        print(self.entorno.evaluate_solution(solution)['score_discr'])

        # print(f"\nDías de reunión por grupo:")
        # for group_idx, meeting_day_idx in self.meeting_days.items():
        #     group_name = self.groups[group_idx]
        #     day_name = self.days[meeting_day_idx]
        #     print(f"{group_name}: {day_name}")

    def _find_unscheduled_employees(self, solution):

        employees_programados, nro_dias= np.unique(solution[solution!=-1], return_counts=True)

        # Crear una máscara para filtrar los empleados con nro_dias < required_in_office_day
        mask = nro_dias < self.entorno.required_in_office_day
        empleados_filtrados = employees_programados[mask]
        dias_filtrados = nro_dias[mask]

        # empleados programados_incompletos
        employees_programados_incompletos = dict(zip(empleados_filtrados, dias_filtrados))


        # Obtener todos los empleados (convertir a numpy array para compatibilidad)
        empleados = np.array(list(self.entorno.employees_map.keys()))

        # Calcular la diferencia
        empleados_no_programados = np.setdiff1d(empleados, employees_programados)

        # empleados a quienes se les hará el local search
        for i in empleados_no_programados:
          employees_programados_incompletos[i]=self.entorno.required_in_office_day # {idx_emple: nro_dias}
        return employees_programados_incompletos

    def _assign_available_desk(self, under_scheduled_employees, solution):

        current_solution= solution.copy()
        for emp_idx, days in under_scheduled_employees.items():
            for i in range(days):
                assigned=False
                preferred_days_emp = self.days_e_idx.get(emp_idx, [])
                preferred_desks_emp= self.desks_e_idx.get(emp_idx, [])
                # intentar asignar inicialmente a un dia preferido si está disponible
                for day in preferred_days_emp:
                    for desk_idx in preferred_desks_emp:
                        if (current_solution[day, desk_idx]==-1) and (emp_idx not in current_solution[day]):
                            current_solution[day, desk_idx]= emp_idx
                            assigned=True
                            #print(f"Empleado {empleado_idx} asignado al puesto {puesto_idx} en el día {dia}")

                # Sino asignar a otro dia no preferido
                if not assigned:
                    for dia in range(len(self.days)):
                        for desk_idx in preferred_desks_emp:
                            if (current_solution[day, desk_idx]==-1) and (emp_idx not in current_solution[day]):
                                current_solution[day, desk_idx]= emp_idx
                            #print(f"Empleado {empleado_idx} asignado al puesto {puesto_idx} en el día {dia} (no deseado)")
        return current_solution

    def evaluate_acceptance_criterio(self, best_score, test_score):
        " función para evaluar si se acepta una solución con base en los criterios de prioridad (lexicograficamente)"
        test_tuple = tuple(test_score[p] for p in self.priorities)
        best_tuple = tuple(best_score[p] for p in self.priorities)
        return test_tuple > best_tuple


    def local_search_smart(self, solution: np.ndarray) -> np.ndarray:
        """Búsqueda local inteligente que identifica intercambios prometedores sin enumerar todos los pares"""

        current_solution = solution.copy()
        current_solution_fitness= [value for key, value in self.entorno.evaluate_solution(current_solution)['score_discr'].items()]
        improved = True

        while improved:
            improved = False
            best_solution = current_solution.copy()
            best_score = current_solution_fitness

            # Primero se intentan asignar los empleados con asignaciones pendientes:
            under_scheduled_employees= self._find_unscheduled_employees(current_solution)
            current_solution= self._assign_available_desk(under_scheduled_employees, current_solution)

            #if np.random.random() <0.5:
            # Movimiento 1: Intercambios dirigidos por problemas identificados (Empleados en días no preferidos,Equipos fragmentados, Empleados solos en zona)
            swaps_in_days,swaps_between_days = self._identify_problem_based_swaps(current_solution)

            for day_idx1, desk1, emp1, day_idx2, desk2, emp2 in swaps_between_days:
                # Verificar si que el intercambio se valido (que ambos empleados ya tengan asignaciones en sus nuevas posiciones)
                if (emp2 not in current_solution[day_idx1] and emp1 not in current_solution[day_idx2]):

                    # Realizar intercambio
                    test_solution = current_solution.copy()
                    test_solution[day_idx1, desk1] = emp2
                    test_solution[day_idx2, desk2] = emp1

                    test_score = [value for key, value in self.entorno.evaluate_solution(test_solution)['score_discr'].items()]
                    if self.evaluate_acceptance_criterio(best_score, test_score): # evalua la solución optimizando lexicograficamente
                        best_solution = test_solution
                        best_score = test_score
                        #improved = True
                        break  # Tomar el primer intercambio que mejore

            for day_idx, desk1, emp1, desk2, emp2 in swaps_in_days:
                # Verificar si el intercambio es válido
                if (desk1 in self.desks_e_idx[emp2] and desk2 in self.desks_e_idx[emp1] and emp1 != emp2):
                    # Realizar intercambio
                    test_solution = current_solution.copy()
                    test_solution[day_idx, desk1] = emp2
                    test_solution[day_idx, desk2] = emp1

                    test_score = [value for key, value in self.entorno.evaluate_solution(test_solution)['score_discr'].items()]
                    if self.evaluate_acceptance_criterio(best_score, test_score): # evalua la solución optimizando lexicograficamente
                        best_solution = test_solution
                        best_score = test_score
                        #improved = True
                        break  # Tomar el primer intercambio que mejore
        #else:
            promising_relocations = self._identify_opportunity_based_relocations(current_solution)

            for day_idx, old_desk, emp_idx, new_desk in promising_relocations:
                test_solution = current_solution.copy()
                test_solution[day_idx, old_desk] = -1
                test_solution[day_idx, new_desk] = emp_idx

                test_score = [value for key, value in self.entorno.evaluate_solution(test_solution)['score_discr'].items()]
                if self.evaluate_acceptance_criterio(best_score, test_score): # evalua la solución optimizando lexicograficamente
                    best_solution = test_solution
                    best_score = test_score
                    #improved = True
                    break

            current_solution = best_solution
            current_score = best_score

        return current_solution

    def _identify_problem_based_swaps(self, solution: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """Identifica intercambios prometedores basados en problemas específicos detectados"""
        promising_swaps = []

        for day_idx in range(len(self.days)):
            # Problema 1: Empleados en días no preferidos
            misplaced_employees= self._find_misplaced_employees(day_idx, solution)

            # Problema 2: Equipos fragmentados
            #print("localizando equipos fragmentados")
            fragmented_teams = self._find_fragmented_teams(day_idx, solution)

            # Problema 3: Empleados solos en zonas donde tienen compañeros en otras zonas
            #print("localizando empleados aislados")
            isolated_employees = self._find_isolated_employees(day_idx, solution)

            # Generar intercambios dirigidos para resolver estos problemas
            swaps, swaps_between_days= self._generate_swaps_for_misplaced(day_idx, misplaced_employees, solution)
            promising_swaps.extend(swaps)
            promising_swaps.extend(self._generate_swaps_for_fragmented_teams(day_idx, fragmented_teams, solution))
            promising_swaps.extend(self._generate_swaps_for_isolated(day_idx, isolated_employees, solution))

        random.shuffle(promising_swaps)
        promising_swaps = promising_swaps[:15]


        return promising_swaps, swaps_between_days

    def _find_misplaced_employees(self, day_idx: int, solution: np.ndarray) -> List[Tuple[int, int]]:
        """Encuentra empleados asignados en días que no prefieren"""
        misplaced = []

        for desk_idx in range(len(self.desks)):
            emp_idx = solution[day_idx, desk_idx]
            if emp_idx != -1:
                preferred_days = set(self.days_e_idx.get(emp_idx, list(range(len(self.days)))))
                if day_idx not in preferred_days:
                    misplaced.append((emp_idx, desk_idx))

        return misplaced

    def _find_fragmented_teams(self, day_idx: int, solution: np.ndarray) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Encuentra equipos que están fragmentados en múltiples zonas"""
        fragmented_teams = []

        for group_name, employees_in_group in self.employees_g.items():
            group_idx = self.groups_reverse_map[group_name]
            present_employees = []
            zones_used = set()

            # Encontrar empleados del grupo presentes este día
            for emp_name in employees_in_group:
                emp_idx = self.employees_reverse_map[emp_name]
                for desk_idx in range(len(self.desks)):
                    if solution[day_idx, desk_idx] == emp_idx:
                        zone = self.desk_zone.get(desk_idx, -1)
                        present_employees.append((emp_idx, desk_idx))
                        if zone != -1:
                            zones_used.add(zone)
                        break

            # Si el equipo está en más de 2 zonas, es candidato para consolidación
            if len(zones_used) > 2 and len(present_employees) > 1:
                fragmented_teams.append((group_idx, present_employees))

        return fragmented_teams

    def _find_isolated_employees(self, day_idx: int, solution: np.ndarray) -> List[Tuple[int, int]]:
        """Encuentra empleados que están solos en una zona mientras sus compañeros están en otra"""
        isolated = []

        for desk_idx in range(len(self.desks)):
            emp_idx = solution[day_idx, desk_idx]
            if emp_idx != -1:
                current_zone = self.desk_zone.get(desk_idx, -1)
                if current_zone != -1:
                    teammates_in_current_zone = self._count_teammates_in_zone(emp_idx, day_idx, current_zone, solution)

                    # Si está solo, verificar si tiene compañeros en otras zonas
                    if teammates_in_current_zone == 0:
                        total_teammates_present = self._count_total_teammates_present(emp_idx, day_idx, solution)
                        if total_teammates_present > 0:
                            isolated.append((emp_idx, desk_idx))

        return isolated

    def _generate_swaps_for_misplaced(self, day_idx: int, misplaced: List[Tuple[int, int]],
                                    solution: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """Genera intercambios para resolver empleados mal ubicados"""
        swaps = []
        swaps_between_days=[]

        for emp1, desk1 in misplaced:
            preferred_days_emp1 = set(self.days_e_idx.get(emp1, []))

            # Buscar empleados que SÍ prefieren este día pero están en días que no prefieren
            for other_day in preferred_days_emp1:
                if other_day != day_idx:
                    for desk_idx in range(len(self.desks)):
                        emp2 = solution[other_day, desk_idx]
                        if emp2 != -1:
                            preferred_days_emp2 = set(self.days_e_idx.get(emp2, []))
                            # Si emp2 no prefiere other_day
                            if other_day not in preferred_days_emp2 :
                                # Verificar compatibilidad de puestos
                                if (desk1 in self.desks_e_idx[emp2] and desk_idx in self.desks_e_idx[emp1]):
                                    swaps_between_days.append((day_idx, desk1, emp1, other_day, desk_idx, emp2))

            # Buscar intercambios dentro del mismo día con empleados que prefieren otros días
            for desk2 in range(len(self.desks)):
                emp2 = solution[day_idx, desk2]
                if emp2 != -1 and emp2 != emp1:
                    preferred_days_emp2 = set(self.days_e_idx.get(emp2, []))
                    # Si emp2 tampoco prefiere este día, podrían beneficiarse mutuamente
                    if day_idx not in preferred_days_emp2:
                        swaps.append((day_idx, desk1, emp1, desk2, emp2))

        return swaps, swaps_between_days

    def _generate_swaps_for_fragmented_teams(self, day_idx: int, fragmented_teams: List[Tuple[int, List[Tuple[int, int]]]],
                                        solution: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """Genera intercambios para consolidar equipos fragmentados"""
        swaps = []

        for group_idx, team_members in fragmented_teams:
            # Identificar la zona con más miembros del equipo
            zone_counts = {}
            for emp_idx, desk_idx in team_members:
                zone = self.desk_zone.get(desk_idx, -1)
                if zone != -1:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1

            if not zone_counts:
                continue

            target_zone = max(zone_counts, key=zone_counts.get)

            # Buscar miembros que están fuera de la zona target
            for emp_idx, desk_idx in team_members:
                current_zone = self.desk_zone.get(desk_idx, -1)
                if current_zone != target_zone:
                    # Buscar empleados de otros equipos en la zona target que podrían intercambiar
                    for target_desk in range(len(self.desks)):
                        if (solution[day_idx, target_desk] != -1 and
                            self.desk_zone.get(target_desk, -1) == target_zone):
                            other_emp = solution[day_idx, target_desk]
                            # Verificar que no sea del mismo equipo
                            if not self._employees_in_same_group(emp_idx, other_emp):
                                swaps.append((day_idx, desk_idx, emp_idx, target_desk, other_emp))

        return swaps

    def _generate_swaps_for_isolated(self, day_idx: int, isolated: List[Tuple[int, int]],
                                solution: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """Genera intercambios para acercar empleados aislados a sus compañeros"""
        swaps = []

        for emp_idx, desk_idx in isolated:
            # Encontrar la zona donde están sus compañeros
            best_zone = self._find_best_zone_for_employee(emp_idx, day_idx, solution)

            if best_zone != -1:
                # Buscar empleados en esa zona que podrían intercambiar
                for target_desk in range(len(self.desks)):
                    if (solution[day_idx, target_desk] != -1 and
                        self.desk_zone.get(target_desk, -1) == best_zone):
                        other_emp = solution[day_idx, target_desk]
                        # Preferir intercambiar con empleados de otros equipos
                        if not self._employees_in_same_group(emp_idx, other_emp):
                            swaps.append((day_idx, desk_idx, emp_idx, target_desk, other_emp))

        return swaps

    def _identify_opportunity_based_relocations(self, solution: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Identifica reubicaciones basadas en oportunidades específicas detectadas"""
        relocations = []

        for day_idx in range(len(self.days)):
            # Oportunidad 1: Espacios vacíos en zonas con compañeros de equipo
            team_consolidation_moves = self._find_team_consolidation_opportunities(day_idx, solution)
            relocations.extend(team_consolidation_moves)

            # Oportunidad 2: Mejora de consistencia de puesto
            consistency_moves = self._find_consistency_opportunities(day_idx, solution)
            relocations.extend(consistency_moves)

        # seleccionar 10 elementos aleatoriamente
        random.shuffle(relocations)
        relocations = relocations[:20]

        return relocations

    def _find_team_consolidation_opportunities(self, day_idx: int, solution: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Encuentra oportunidades de consolidar equipos en espacios vacíos"""
        opportunities = []

        # Encontrar espacios vacíos
        empty_desks = [desk_idx for desk_idx in range(len(self.desks))
                    if solution[day_idx, desk_idx] == -1]

        for empty_desk in empty_desks:
            empty_zone = self.desk_zone.get(empty_desk, -1)
            if empty_zone == -1:
                continue

            # Buscar empleados que podrían beneficiarse de moverse a esta zona
            for current_desk in range(len(self.desks)):
                emp_idx = solution[day_idx, current_desk]
                if emp_idx != -1 and empty_desk in self.desks_e_idx[emp_idx]:
                    # Contar compañeros en la zona vacía vs zona actual
                    teammates_in_empty_zone = self._count_teammates_in_zone(emp_idx, day_idx, empty_zone, solution)
                    current_zone = self.desk_zone.get(current_desk, -1)
                    teammates_in_current_zone = self._count_teammates_in_zone(emp_idx, day_idx, current_zone, solution)

                    if teammates_in_empty_zone > teammates_in_current_zone:
                        opportunities.append((day_idx, current_desk, emp_idx, empty_desk))

        return opportunities

    def _find_consistency_opportunities(self, day_idx: int, solution: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Encuentra oportunidades de mejorar consistencia de puesto"""
        opportunities = []

        for desk_idx in range(len(self.desks)):
            emp_idx = solution[day_idx, desk_idx]
            if emp_idx != -1:
                # Encontrar el puesto más usado por este empleado
                most_used_desk = self._find_most_used_desk(emp_idx, solution)
                if (most_used_desk != desk_idx and
                    most_used_desk in self.desks_e_idx[emp_idx] and
                    solution[day_idx, most_used_desk] == -1):
                    opportunities.append((day_idx, desk_idx, emp_idx, most_used_desk))

        return opportunities

    def _find_best_zone_for_employee(self, emp_idx: int, day_idx: int, solution: np.ndarray) -> int:
        """Encuentra la mejor zona para un empleado basada en la ubicación de sus compañeros"""
        zone_teammates = {}

        # Contar compañeros en cada zona
        for zone_idx in range(len(self.zones)):
            count = self._count_teammates_in_zone(emp_idx, day_idx, zone_idx, solution)
            if count > 0:
                zone_teammates[zone_idx] = count

        if zone_teammates:
            return max(zone_teammates, key=zone_teammates.get)
        return -1

    def _find_most_used_desk(self, emp_idx: int, solution: np.ndarray) -> int:
        """Encuentra el puesto más usado por un empleado"""
        desk_usage = {}

        for day_idx in range(len(self.days)):
            for desk_idx in range(len(self.desks)):
                if solution[day_idx, desk_idx] == emp_idx:
                    desk_usage[desk_idx] = desk_usage.get(desk_idx, 0) + 1

        if desk_usage:
            return max(desk_usage, key=desk_usage.get)
        return -1

    def _count_total_teammates_present(self, emp_idx: int, day_idx: int, solution: np.ndarray) -> int:
        """Cuenta el total de compañeros de equipo presentes en un día"""
        count = 0
        emp_groups = self.employee_groups.get(emp_idx, [])

        for group_idx in emp_groups:
            group_name = self.groups[group_idx]
            if group_name in self.employees_g:
                for teammate_name in self.employees_g[group_name]:
                    teammate_idx = self.employees_reverse_map[teammate_name]
                    if teammate_idx != emp_idx:
                        # Verificar si el compañero está presente este día
                        for desk_idx in range(len(self.desks)):
                            if solution[day_idx, desk_idx] == teammate_idx:
                                count += 1
                                break

        return count

    def _count_teammates_in_zone(self, emp_idx: int, day_idx: int, zone: int, solution: np.ndarray) -> int:
        """Cuenta compañeros de equipo en una zona específica en un día"""
        if zone == -1:
            return 0

        count = 0
        emp_groups = self.employee_groups.get(emp_idx, [])

        for group_idx in emp_groups:
            group_name = self.groups[group_idx]
            if group_name in self.employees_g:
                for teammate_name in self.employees_g[group_name]:
                    teammate_idx = self.employees_reverse_map[teammate_name]
                    if teammate_idx != emp_idx:
                        for desk_idx in range(len(self.desks)):
                            if (solution[day_idx, desk_idx] == teammate_idx and
                                self.desk_zone.get(desk_idx, -1) == zone):
                                count += 1
                                break

        return count

    def _employees_in_same_group(self, emp1: int, emp2: int) -> bool:
        """Verifica si dos empleados están en el mismo grupo"""
        groups1 = set(self.employee_groups.get(emp1, []))
        groups2 = set(self.employee_groups.get(emp2, []))
        return len(groups1.intersection(groups2)) > 0