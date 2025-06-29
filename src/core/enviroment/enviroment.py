import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
from gymnasium import spaces

class ShiftSchedulerEnvMaskeable(gym.Env):
    """
    A Gymnasium environment for a shift scheduling problem formulated for reinforcement learning.

    The environment's goal is to assign employees to specific desks on given days.
    The assignments are evaluated based on several criteria:
    - Employee preferences for both days and desks.
    - Diversity of employee groups within designated office zones.
    - Proximity of team members to encourage collaboration.
    - Ensuring each employee fulfills their required number of office days.

    The environment uses action masking to ensure that the agent only selects valid
    actions (i.e., unoccupied desks that an employee is permitted to use).

    Attributes:
        instance (Dict[str, Any]): The raw data defining the scheduling problem.
        required_in_office_day (int): The number of days each employee must be in the office.
        best_solution (Dict): Stores the best reward and solution found during an episode.
    """

    def __init__(self, instance: Dict[str, Any], required_in_office_day: int):
        """
        Initializes the shift scheduling environment.

        Args:
            instance: A dictionary containing all the necessary data for the environment,
                      such as lists of employees, desks, days, and their relationships.
            required_in_office_day: The number of days each employee is required to work.
        """
        super().__init__()

        # --- Instance Data and Mappings ---
        self.instance = instance
        self.required_in_office_day = required_in_office_day

        self._create_mappings()

        # --- Environment Spaces ---
        self._setup_spaces()

        # --- State and Episode Tracking ---
        self._initialize_state()
        self._create_assignment_schedule()

        # --- Solution Tracking ---
        self.best_solution_final_reward = {'best_reward': -np.inf}
        self.best_solution_reward= {'best_reward': -np.inf}
        self.best_solutions={'best_emp_no_asignados':{'solution':None, 'reward':-np.inf},
                             'best_dias_pref':{'solution':None, 'reward':-np.inf},
                             'best_desk_consistency':{'solution':None, 'reward':-np.inf},
                             'best_team_proximity':{'solution':None, 'reward':-np.inf},
                             'best_asistencia_equipos':{'solution':None, 'reward':-np.inf}
                             }

        self.episode_info = []
        self.suma_reward=0
        self.assignment_counter=0



    # --------------------------------------------------------------------------
    # Initialization and Setup Methods
    # --------------------------------------------------------------------------



    def _create_mappings(self):
        """Creates dictionaries for efficient lookup of instance data."""


        self.employees = self.instance['Employees']
        self.desks = self.instance['Desks']
        self.days = self.instance['Days']
        self.groups = self.instance['Groups']
        self.zones = self.instance['Zones']
        self.desks_z = self.instance['Desks_Z']
        self.desks_e = self.instance['Desks_E']
        self.employees_g = self.instance['Employees_G']
        self.days_e = self.instance['Days_E']

        self.days_map = {idx: value for idx, value in enumerate(self.days)}
        self.desks_map = {idx: value for idx, value in enumerate(self.desks)}
        self.employees_map = {idx: value for idx, value in enumerate(self.employees)}
        self.zone_map= {idx: zone for idx, zone in enumerate(self.zones)}
        self.groups_map= {idx: group for idx, group in enumerate(self.groups)}

        # Reverse mappings for lookup
        self.days_reverse_map = {v: k for k, v in self.days_map.items()}
        self.desks_reverse_map = {v: k for k, v in self.desks_map.items()}
        self.employees_reverse_map = {v: k for k, v in self.employees_map.items()}
        self.zone_reverse_map= {v: k for k, v in self.zone_map.items()}
        self.groups_reverse_map= {v: k for k, v in self.groups_map.items()}

        # Employee group mapping
        self.employee_group_map = self._create_reverse_lookup(self.employees_g)

        self.num_presencial_dias_idx={emp: self.required_in_office_day for emp in self.employees_map.keys()}

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

    def _setup_spaces(self):
        """Defines the observation and action spaces for the environment."""
        num_days = len(self.days)
        num_desks = len(self.desks)
        num_employees = len(self.employees)
        total_positions = num_days * num_desks

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-1,
                high=num_employees - 1,
                shape=(total_positions,),
                dtype=np.int32
            ),
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(total_positions,),
                dtype=np.int8
            )
        })

        self.action_space = spaces.Discrete(total_positions)

    def _initialize_state(self):
        """Initializes the state matrix and other tracking variables."""
        num_days = len(self.days)
        num_desks = len(self.desks)
        self._state_matrix = np.full((num_days, num_desks), -1, dtype=np.int32)
        self.assignment_counter = 0

    def _create_assignment_schedule(self):
        """Creates a sequential list of employees to be assigned."""
        self.assignment_schedule = []
        for employee_name in self.employees:
            self.assignment_schedule.extend([employee_name] * self.required_in_office_day)

        if not self.assignment_schedule:
            raise ValueError("Assignment schedule is empty. Check employee data.")

        self.current_employee = self.assignment_schedule[0]
        self.employee_idx = self.employees_reverse_map[self.current_employee]

    # --------------------------------------------------------------------------
    # Core Gym Methods (step, reset, action_masks)
    # --------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resets the environment to its initial state for a new episode."""
        super().reset(seed=seed)

        self._initialize_state()
        self._create_assignment_schedule()


        self.current_employee = self.assignment_schedule[0]
        self.employee_idx = self.employees_reverse_map[self.current_employee]

        self.episode_info = []
        self.suma_reward= 0
        self.best_solution_final_reward = {'best_reward': -np.inf}
        self.best_solution_reward= {'best_reward': -np.inf}

        self.best_solutions={'best_emp_no_asignados':{'solution':None, 'reward':-np.inf},
                             'best_dias_pref':{'solution':None, 'reward':-np.inf},
                             'best_desk_consistency':{'solution':None, 'reward':-np.inf},
                             'best_team_proximity':{'solution':None, 'reward':-np.inf},
                             'best_asistencia_equipos':{'solution':None, 'reward':-np.inf}
                             }


        info = {
            'current_employee': self.current_employee,
            'total_assignments': len(self.assignment_schedule)
        }
        self.assignment_counter = 0

        return self.state, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Executes a single step in the environment by assigning an employee to a desk.

        Args:
            action: The action chosen by the agent, corresponding to a desk-day slot.

        Returns:
            A tuple containing the new state, reward, terminated flag, truncated flag, and info dict.
        """

        day, desk = self._action_to_coordinates(action)

        # --- Validate Action ---
        is_occupied = self._get_position(day, desk) != -1
        is_already_assigned_today = self.employee_idx in self._get_day_assignments(day)

        if is_occupied or is_already_assigned_today:
            reward = -30
            was_assigned = False
        else:
            # --- Calculate Reward and Update State ---
            reward = self._calculate_reward(day, desk)
            was_assigned = True

        # --- Logging and Info ---
        info = {
            'current_employee': self.assignment_schedule[self.assignment_counter],
            'status': "Assigned" if was_assigned else "Not Assigned",
            'chosen_slot': (self.days_map[day], self.desks_map[desk]),
            'reward': reward
        }

        self.assignment_counter += 1

        # --- Check for Termination ---
        terminated = self.assignment_counter >= len(self.assignment_schedule)
        if terminated:
          results= self.evaluate_solution(self._state_matrix)
          self.final_reward = results['score']
          self.emp_no_asignados= results['score_discr']['empleados_no_asignados']
          self.dias_pref= results['score_discr']['dias_preferidos_asignados']
          self.desk_consistency= results['score_discr']['consistencia_puestos']
          self.team_proximity= results['score_discr']['proximidad_equipos']
          self.asistencia_equipos= results['score_discr']['asistencia_equipos']

          reward += self.final_reward

        else:
            self.current_employee = self.assignment_schedule[self.assignment_counter]
            self.employee_idx = self.employees_reverse_map[self.current_employee]


        self.suma_reward=+reward

        #self.episode_info.append(info)
        #if terminated:
          # if self.final_reward > self.best_solution_final_reward.get('best_reward', -np.inf):
          #   self.best_solution_final_reward = {
          #       'best_reward': self.final_reward,
          #       'best_solution': pd.DataFrame(self.episode_info),
          #       'final_state': self._state_matrix.copy(),
          #       }
          # if self.suma_reward > self.best_solution_reward.get('best_reward', -np.inf):
          #     self.best_solution_reward = {
          #       'best_reward': reward,
          #       'best_solution': pd.DataFrame(self.episode_info),
          #       'final_state': self._state_matrix.copy(),

          #   }
          # if self.emp_no_asignados > self.best_solutions['best_emp_no_asignados']['reward']:
          #   self.best_solutions['best_emp_no_asignados']['reward']= emp_no_asignados
          #   self.best_solutions['best_emp_no_asignados']['solution']= self._state_matrix.copy()
          # if dias_pref > self.best_solutions['best_dias_pref']['reward']:
          #   self.best_solutions['best_dias_pref']['reward']= dias_pref
          #   self.best_solutions['best_dias_pref']['solution']= self._state_matrix.copy()
          # if desk_consistency > self.best_solutions['best_desk_consistency']['reward']:
          #   self.best_solutions['best_desk_consistency']['reward']= desk_consistency
          #   self.best_solutions['best_desk_consistency']['solution']= self._state_matrix.copy()
          # if team_proximity > self.best_solutions['best_team_proximity']['reward']:
          #   self.best_solutions['best_team_proximity']['reward']= team_proximity
          #   self.best_solutions['best_team_proximity']['solution']= self._state_matrix
          # if asistencia_equipos > self.best_solutions['best_asistencia_equipos']['reward']:
          #   self.best_solutions['best_asistencia_equipos']['reward']= asistencia_equipos
          #   self.best_solutions['best_asistencia_equipos']['solution']= self._state_matrix.copy()



        return self.state, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """
        Generates a binary mask indicating valid actions for the current employee.
        A valid action is an unassigned desk that the employee is permitted to use.
        """
        num_actions = self.action_space.n
        mask = np.zeros(num_actions, dtype=np.int8)

        # Get desks the current employee is allowed to use
        employee_preferred_desk_ids = [
            self.desks_reverse_map[d]
            for d in self.desks_e.get(self.current_employee, [])
        ]

        if not employee_preferred_desk_ids:
            # Fallback: if no preferred desks, allow any available desk
            available_slots = self._get_all_available_slots()
            valid_actions = [self._coordinates_to_action(d, p) for d, p in available_slots]
        else:
            # Find available slots among the preferred desks
            valid_actions = []
            for day_idx in range(len(self.days)):
                for desk_idx in employee_preferred_desk_ids:
                    if self._state_matrix[day_idx, desk_idx] == -1 and self.employee_idx not in self._state_matrix[day_idx]:
                        valid_actions.append(self._coordinates_to_action(day_idx, desk_idx))

        if valid_actions:
            mask[valid_actions] = 1
        else:
            # Fallback if no actions are available (e.g., all preferred desks are taken)
            # Assign the first available desk as a last resort.
            all_available = np.argwhere(self._state_matrix == -1)
            if all_available.size > 0:
                fallback_action = self._coordinates_to_action(all_available[0, 0], all_available[0, 1])
                mask[fallback_action] = 1
            else:
                mask[0] = 1 # Should not happen if episode terminates correctly

        return mask

    # --------------------------------------------------------------------------
    # Reward Calculation Methods
    # --------------------------------------------------------------------------

    def _calculate_reward(self, day: int, desk: int) -> float:
        """
        Calculates the immediate reward for assigning the current employee to a given slot.
        """
        day_name = self.days_map[day]
        desk_name = self.desks_map[desk]
        reward = 0.0

        self._set_position(day, desk, self.employee_idx)

        # satisfactions
        reward_satisfactions= self.evaluate_day_preferences(self._state_matrix, day_idx=day, emp_idx=self.employee_idx)

        #desk_consistency
        reward_desk_consistency= self.evaluate_desk_consistency(self._state_matrix, emp_idx=self.employee_idx)

        #team_proximity
        reward_team_proximity= self.evaluate_team_proximity(self._state_matrix, day_idx= day)

        # team attendance
        reward_team_attendance= self.evaluate_attendance(self._state_matrix, day_index=day)

        reward += reward_satisfactions+reward_desk_consistency+reward_team_proximity+reward_team_attendance


        return reward

    def _calculate_zone_diversity_reward(self, day: int, desk_name: str) -> float:
        """Calculates a reward based on the diversity of groups within a zone."""
        target_zone = next((z for z, d in self.desks_z.items() if desk_name in d), None)
        if not target_zone:
            return 0.0

        zone_desks = self.desks_z[target_zone]
        zone_desk_indices = [self.desks_reverse_map[d] for d in zone_desks if d in self.desks_reverse_map]

        assigned_employees = self._state_matrix[day, zone_desk_indices]
        valid_employees = assigned_employees[assigned_employees != -1]

        if len(valid_employees) <= 1:
            return 0.0  # Diversity is not applicable with 0 or 1 person

        zone_groups = [self.employee_group_map[self.employees_map[emp]] for emp in valid_employees]
        impurity = self._normalized_gini(zone_groups)

        # Reward diversity (low impurity) and penalize homogeneity (high impurity)
        return 3.0 if impurity == 0 else -2.0 * impurity

    def _calculate_team_proximity_reward(self) -> float:
        """Calculates a reward based on the proximity of team members."""
        distances = 0
        num_groups_in_day = 0

        for day_idx in range(self._state_matrix.shape[0]):
            day_assignments = self._state_matrix[day_idx, :]
            assigned_indices = np.where(day_assignments != -1)[0]

            if len(assigned_indices) < 2:
                continue

            employee_ids = day_assignments[assigned_indices]
            groups = np.array([self.employee_group_map[self.employees_map[eid]] for eid in employee_ids])

            unique_groups = np.unique(groups)
            num_groups_in_day += len(unique_groups)

            for group in unique_groups:
                member_indices = assigned_indices[groups == group]
                if len(member_indices) > 1:
                    distances += np.sum(np.diff(np.sort(member_indices)))

        if num_groups_in_day == 0:
            return 0.0

        # Penalize large distances; closer teams are better.
        # Normalization helps keep the reward value stable.
        penalty = distances / (self._state_matrix.shape[1] * num_groups_in_day)
        return -penalty

    def evaluate_solution(self,solution) -> float:
        """Evalúa la calidad de una solución"""
        score = 0

        # Verificar restricciones hard
        assigned_days= self.check_hard_constraints(solution)
        score += assigned_days

        # Evaluar criterios soft
        preferred_days = self.evaluate_day_preferences(solution)
        score += preferred_days
        desk_consistency = self.evaluate_desk_consistency(solution)
        score += desk_consistency
        team_proximity= self.evaluate_team_proximity(solution)
        score += team_proximity
        team_attendance= self.evaluate_attendance(solution)
        score += team_attendance

        solution_quality= {'empleados_no_asignados': assigned_days,
                           'dias_preferidos_asignados': preferred_days,
                           'consistencia_puestos':desk_consistency,
                           'proximidad_equipos':team_proximity,
                           'asistencia_equipos': team_attendance
                           }

        return {'score': score,
                'score_discr': solution_quality
                }

    def check_hard_constraints(self, solution) -> int:
        """Verifica restricciones obligatorias y devuelve penalización"""
        penalty = 0
        # 1. Verificar que cada empleado tenga el número correcto de días
        for emp_idx in range(len(self.employees)):
            assigned_days = int(np.sum(solution == emp_idx))
            required_days = self.num_presencial_dias_idx[emp_idx]
            penalty += abs(assigned_days - required_days)


          #print(f"Empleado: {self.employees_map[emp_idx]}, Asignados: {assigned_days}, Necesarios: {required_days}")

        # 2. Verificar días de reunión de grupo
        # for group_idx, meeting_day in self.meeting_days.items():
        #     group_name = self.groups[group_idx]
        #     if group_name in self.employees_g:
        #         for emp_name in self.employees_g[group_name]:
        #             emp_idx = self.employees_reverse_map[emp_name]
        #             if np.sum(self._state_matrix[meeting_day] == emp_idx) == 0:
        #                 penalty += 10  # Penalización alta por no asistir a reunión

        # 3. Verificar compatibilidad de puestos
        # for day_idx in range(len(self.days)):
        #     for desk_idx in range(len(self.desks)):
        #         emp_idx = self._state_matrix[day_idx, desk_idx]
        #         if emp_idx != -1:
        #             if desk_idx not in self.desks_e_idx[emp_idx]:
        #                 penalty += 25  # Puesto no compatible

        return -penalty

    def evaluate_day_preferences(self, solution, day_idx= None, emp_idx=None) -> float:
        """Evalúa satisfacción de preferencias de días"""
        score = 0
        if emp_idx is not None:
            preferred_days = self.days_e_idx.get(emp_idx, [])
            if day_idx in preferred_days:
                score = 2
            else:
                score =1
        else:
            for emp_idx in range(len(self.employees)):
                preferred_days = set(self.days_e_idx.get(emp_idx, []))
                for day_idx in range(len(self.days)):
                    if np.sum(solution[day_idx] == emp_idx) > 0:
                        if day_idx in preferred_days:
                            score += 1
                        else:
                            score -= 1
        return score

    def evaluate_desk_consistency(self,solution, emp_idx=None ) -> float:
        """Evalúa consistencia del mismo puesto para el mismo empleado"""
        score = 0
        if emp_idx is not None:
            used_desks = set()
            prefered_desks= self.desks_e_idx.get(emp_idx, [])
            for day_idx in range(len(self.days)):
                for desk_idx in prefered_desks:
                    if solution[day_idx, desk_idx] == emp_idx:
                        used_desks.add(desk_idx)

            if len(used_desks) == 1:
                score = 2
            elif len(used_desks) > 1:
                score = -(len(used_desks)) # Penalización por cambiar de puesto
        else:
            for emp_idx in range(len(self.employees)):
                used_desks = set()
                for day_idx in range(len(self.days)):
                    for desk_idx in range(len(self.desks)):
                        if solution[day_idx, desk_idx] == emp_idx:
                            used_desks.add(desk_idx)

                if len(used_desks) == 1:
                    score += 2
                elif len(used_desks) > 1:
                    score -= (len(used_desks)) # Penalización por cambiar de puesto

        return score

    def evaluate_team_proximity(self,solution, day_idx=None ) -> float:
        """Evalúa proximidad de miembros del mismo equipo"""
        score = 0
        if day_idx is not None:
            for group_name, employees_in_group in self.employees_g.items():
                emp_indices = [self.employees_reverse_map[emp] for emp in employees_in_group]
                zones_used = set()

                for emp_idx in emp_indices:
                    for desk_idx in range(len(self.desks)):
                        if solution[day_idx, desk_idx] == emp_idx:
                            if desk_idx in self.desk_zone:
                                zones_used.add(self.desk_zone[desk_idx])

                if len(zones_used) == 1:
                    score += 1 # Bonus por estar en la misma zona
                elif len(zones_used)>1:
                    score -= len(zones_used)/len(self.zones)    # Bonus menor por estar en 2 zonas o mas
        else:
            for day_idx in range(len(self.days)):
                for group_name, employees_in_group in self.employees_g.items():
                    emp_indices = [self.employees_reverse_map[emp] for emp in employees_in_group]
                    zones_used = set()

                    for emp_idx in emp_indices:
                        for desk_idx in range(len(self.desks)):
                            if solution[day_idx, desk_idx] == emp_idx:
                                if desk_idx in self.desk_zone:
                                    zones_used.add(self.desk_zone[desk_idx])

                    if len(zones_used) == 1:
                        score += 1 # Bonus por estar en la misma zona
                    elif len(zones_used) >1:
                        score -= len(zones_used)/ len(self.zones)   # Bonus menor por estar en máximo 2 zonas

        return score

    def evaluate_attendance(self,solution, day_index=None) -> float:
        """Evalúa la cantidad de veces que TODOS los integrantes del equipo asisten el mismo dia"""
        score = 0
        scores=[]
        if day_index is None:
            for group_name, employees_in_group in self.employees_g.items():
                att_max=0
                for day_idx in range(len(self.days)):
                    conteo=0
                    emp_indices = [self.employees_reverse_map[emp] for emp in employees_in_group]
                    for emp_idx in emp_indices:
                        if emp_idx in solution[day_idx]:
                            conteo+=1
                    attendace_g= conteo/len(emp_indices)
                    if attendace_g> att_max:
                        att_max= attendace_g

                scores.append(att_max)
            score=np.mean(scores)
        else:
            conteo=0
            emp_group= self.employee_group_map.get(self.current_employee, [])
            employees_in_group= self.employees_g.get(emp_group, [])
            emp_indices = [self.employees_reverse_map[emp] for emp in employees_in_group]
            for emp_idx in emp_indices:
                if emp_idx in solution[day_index]:
                    conteo+=1
            score= conteo/len(emp_indices)
        return score

    # --------------------------------------------------------------------------
    # State and Property Accessor Methods
    # --------------------------------------------------------------------------

    @property
    def state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment, including the action mask."""
        return {
            'observation': self._state_matrix.flatten(),
            'action_mask': self.action_masks()
        }

    def _get_position(self, day: int, desk: int) -> int:
        """Gets the employee ID at a specific (day, desk) position."""
        return self._state_matrix[day, desk]

    def _set_position(self, day: int, desk: int, employee_id: int):
        """Sets an employee ID at a specific (day, desk) position."""
        self._state_matrix[day, desk] = employee_id

    def _get_day_assignments(self, day: int) -> np.ndarray:
        """Gets all assignments for a specific day."""
        return self._state_matrix[day, :]

    def _get_desk_status_all_days(self, desk_idx: int) -> np.ndarray:
        """Gets the status of a specific desk across all days."""
        return self._state_matrix[:, desk_idx]

    def _get_all_available_slots(self) -> List[Tuple[int, int]]:
        """Returns all available (day, desk) slots as a list of coordinates."""
        rows, cols = np.where(self._state_matrix == -1)
        return list(zip(rows, cols))

    # --------------------------------------------------------------------------
    # Helper and Utility Methods
    # --------------------------------------------------------------------------

    def _action_to_coordinates(self, action: int) -> Tuple[int, int]:
        """Converts a flat action index into (day, desk) coordinates."""
        num_desks = len(self.instance['Desks'])
        day = action // num_desks
        desk = action % num_desks
        return day, desk

    def _coordinates_to_action(self, day: int, desk: int) -> int:
        """Converts (day, desk) coordinates into a flat action index."""
        num_desks = len(self.desks)
        return day * num_desks + desk

    def _create_reverse_lookup(self, data: Dict[Any, List]) -> Dict:
        """Creates a reverse mapping from a value to its key."""
        reverse_dict = {}
        for key, values in data.items():
            for value in values:
                reverse_dict[value] = key
        return reverse_dict

    def _gini_impurity(self, labels: List) -> float:
        """Calculates the Gini impurity for a list of labels."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        n = len(labels)
        return 1.0 - sum((count / n) ** 2 for count in counts.values())

    def _normalized_gini(self, labels: List) -> float:
        """Calculates the normalized Gini impurity."""
        if not labels:
            return 0.0
        unique_classes = len(set(labels))
        if unique_classes <= 1:
            return 0.0

        gini = self._gini_impurity(labels)
        max_gini = (unique_classes - 1) / unique_classes
        return gini / max_gini

    def _get_group_attendance_ratio(self) -> Dict[str, float]:
        """
        Calculates the ratio of attending members for each group on their best day.
        """
        attendance_ratios = {}
        assigned_employee_names = np.vectorize(self.employees_map.get)(self._state_matrix)

        for group, members in self.employees_g.items():
            max_attendance = 0
            for day_idx in range(len(self.days)):
                present_members = set(members) & set(assigned_employee_names[day_idx])
                max_attendance = max(max_attendance, len(present_members))

            if not members:
                attendance_ratios[group] = 0.0
            else:
                attendance_ratios[group] = max_attendance / len(members)
        return attendance_ratios

    def _get_preference_satisfaction_count(self) -> int:
        """Counts how many assignments match an employee's preferred day."""
        satisfaction_count = 0
        for day_idx, row in enumerate(self._state_matrix):
            for desk_idx, employee_id in enumerate(row):
                if employee_id != -1:
                    employee_name = self.employees_map[employee_id]
                    day_name = self.days_map[day_idx]
                    if day_name in self.days_e.get(employee_name, []):
                        satisfaction_count += 1
        return satisfaction_count