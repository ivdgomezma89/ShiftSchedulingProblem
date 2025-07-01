import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import json
from instances import InstanceReader
import seaborn as sns

from src.core.algorithms.hybrid_algorithm import ShiftSchedulerOptimizer
import time
import threading
import pickle


class DeskAssignmentVisualization:

    def __init__(self):
        # Configuración de la página
        st.set_page_config(
            page_title="🏢 Sistema de Asignación de Puestos",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # CSS personalizado para mejorar la apariencia
        st.markdown(
            """
        <style>
            .main-header {
               background: linear-gradient(90deg, #a8edea 0%, #5ee7df 50%, #2ecc71 100%);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                color: white;
                text-align: center;
            }
                    
            html, body, [class*="css"]  {
                font-family: 'Segoe UI', sans-serif;
            }
            
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            
            .success-metric {
                border-left-color: #28a745;
            }
            
            .warning-metric {
                border-left-color: #ffc107;
            }
            
            .danger-metric {
                border-left-color: #dc3545;
            }
            
            .info-box {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #17a2b8;
                margin: 1rem 0;
            }
            
            .team-header {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 0.5rem;
                border-radius: 5px;
                margin: 0.5rem 0;
                text-align: center;
                font-weight: bold;
            }
            
            .sidebar-title {
                background: linear-gradient(90deg, #a8edea 0%, #5ee7df 50%, #2ecc71 100%);
                color: white;
                padding: 0.8rem;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: bold;
            }
                    
            .progress-container {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                border: 2px solid #20b2aa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .progress-title {
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 15px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            
            .ils-progress-label {
                color: #20b2aa;
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 5px;
            }
            
            /* Estilo personalizado para las barras de progreso */
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #20b2aa 0%, #48cae4 50%, #90e0ef 100%) !important;
                height: 12px !important;
                border-radius: 6px !important;
            }
            
            .stProgress > div > div > div {
                background-color: #e8f4f8 !important;
                border-radius: 6px !important;
                height: 12px !important;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # CSS personalizado
        st.markdown(
            """
            <style>
                /* Cambiar el fondo del sidebar */
                section[data-testid="stSidebar"] {
                    background-color: #f0f2f6;  /* Cambia este color a tu gusto */
                    padding: 20px;
                }

                /* También puedes cambiar el color del texto si quieres */
                section[data-testid="stSidebar"] * {
                    color: #333;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        if "nro_puestos" not in st.session_state:
            st.session_state.nro_puestos = None

        if "nro_empleados" not in st.session_state:
            st.session_state.nro_empleados = None

        if "nro_dias" not in st.session_state:
            st.session_state.nro_dias = None

        if "nro_zonas" not in st.session_state:
            st.session_state.nro_zonas = None

        if "instancia_programada" not in st.session_state:
            st.session_state.instancia_programada = None

        if "instancia_cargada" not in st.session_state:
            st.session_state.instancia_cargada = None

        if "prioridad_objetivos" not in st.session_state:
            st.session_state.prioridad_objetivos = None

        if "best_schedule" not in st.session_state:
            st.session_state.best_schedule = None

        if "best_score" not in st.session_state:
            st.session_state.best_score = None

        if "total_asignaciones" not in st.session_state:
            st.session_state.total_asignaciones = 0

        if "progress_bar" not in st.session_state:
            st.session_state.progress_bar = None

        if "resultados_corridas" not in st.session_state:
            st.session_state.resultados_corridas = {}

        if "start_time" not in st.session_state:
            st.session_state.start_time = None

        if "end_time" not in st.session_state:
            st.session_state.end_time = None

        if "locked" not in st.session_state:
            st.session_state.locked = False

        self.dias_presencialidad = 2

        self.instance_reader = InstanceReader()

    def save_results(self, results_dict):

        try:
            # Cargar el diccionario
            with open("results/resultados.pkl", "rb") as f:
                loaded_data = pickle.load(f)
            # guardar los resultados en el diccionario
            loaded_data.update(results_dict)
            # Guardar el diccionario actualizado
            with open("results/resultados.pkl", "wb") as f:
                pickle.dump(loaded_data, f)
        except FileNotFoundError:
            # Guardar el diccionario
            with open("results/resultados.pkl", "wb") as f:
                pickle.dump(results_dict, f)

    # Función que se ejecuta al hacer clic
    def bloquear_inputs(self):
        st.session_state.locked = True

    def load_results(self, instance):
        try:
            # Cargar el diccionario
            with open("results/resultados.pkl", "rb") as f:
                loaded_data = pickle.load(f)
            return loaded_data.get(instance, None)

        except FileNotFoundError:
            return None

    def _validar_prioridades(
        self,
    ):

        prioridades_dict_map = {
            "Crítico": 5,
            "Alta": 4,
            "Media": 3,
            "Baja": 2,
            "Mínima": 1,
        }

        conjunto_prioridades = set(
            [
                self.prioridad_asignacion_empleados,
                self.prioridad_asistencia_grupos,
                self.prioridad_satisfaccion_empleados,
                self.prioridad_dispersion_equipos,
                self.prioridad_consistencia_puestos,
            ]
        )

        if len(conjunto_prioridades) != 5:
            st.error(
                "Error: Todos los objetivos deben tener prioridades diferentes.",
                icon="🚨",
            )
            return False

        else:
            prioridades = [
                prioridades_dict_map[self.prioridad_asignacion_empleados],
                prioridades_dict_map[self.prioridad_satisfaccion_empleados],
                prioridades_dict_map[self.prioridad_consistencia_puestos],
                prioridades_dict_map[self.prioridad_dispersion_equipos],
                prioridades_dict_map[self.prioridad_asistencia_grupos],
            ]

            prioridades = np.argsort(prioridades)
            st.session_state.prioridad_objetivos = prioridades[::-1]

            return True

    def _create_reverse_lookup(self, data):
        """Creates a reverse mapping from a value to its key."""
        reverse_dict = {}
        for key, values in data.items():
            for value in values:
                reverse_dict[value] = key
        return reverse_dict

    def _create_mappings(self, instance):
        """Creates dictionaries for efficient lookup of instance data."""

        self.employees = instance["Employees"]
        self.desks = instance["Desks"]
        self.days = instance["Days"]
        self.groups = instance["Groups"]
        self.zones = instance["Zones"]
        self.desks_z = instance["Desks_Z"]
        self.desks_e = instance["Desks_E"]
        self.employees_g = instance["Employees_G"]
        self.days_e = instance["Days_E"]

        self.days_map = {idx: value for idx, value in enumerate(self.days)}
        self.desks_map = {idx: value for idx, value in enumerate(self.desks)}
        self.employees_map = {idx: value for idx, value in enumerate(self.employees)}
        self.zone_map = {idx: zone for idx, zone in enumerate(self.zones)}
        self.groups_map = {idx: group for idx, group in enumerate(self.groups)}

        # Reverse mappings for lookup
        self.days_reverse_map = {v: k for k, v in self.days_map.items()}
        self.desks_reverse_map = {v: k for k, v in self.desks_map.items()}
        self.employees_reverse_map = {v: k for k, v in self.employees_map.items()}
        self.zone_reverse_map = {v: k for k, v in self.zone_map.items()}
        self.groups_reverse_map = {v: k for k, v in self.groups_map.items()}

        # Employee group mapping
        self.employee_group_map = self._create_reverse_lookup(self.employees_g)

        self.num_presencial_dias_idx = {
            emp: self.dias_presencialidad for emp in self.employees_map.keys()
        }

        self.desks_e_idx = {
            self.employees_reverse_map[emp]: [
                self.desks_reverse_map[desk] for desk in desks
            ]
            for emp, desks in self.desks_e.items()
        }

        self.days_e_idx = {
            self.employees_reverse_map[emp]: [
                self.days_reverse_map[day] for day in days
            ]
            for emp, days in self.days_e.items()
        }

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

    def obtener_porcentaje_dias_preferidos(
        self,
    ):
        instance = st.session_state.instancia_programada
        if instance is not None and st.session_state.best_schedule is not None:
            self._create_mappings(instance)
            satisfaccion = 0
            max_satisfaccion = 0
            for day in range(len(self.days)):
                for desk in range(len(self.desks)):
                    empployee_idx = st.session_state.best_schedule[day][desk]
                    if empployee_idx == -1:
                        continue
                    else:
                        if day in self.days_e_idx[empployee_idx]:
                            satisfaccion += 1
            # conteo de satisfacciones maximas posibles
            for empl_name, days in self.days_e.items():
                if len(days) > self.dias_presencialidad:
                    max_satisfaccion += self.dias_presencialidad
                else:
                    max_satisfaccion += len(days)
            return (satisfaccion, max_satisfaccion)
        else:
            return (None, None)

    def consistencia_puestos(
        self,
    ):
        instance = st.session_state.instancia_programada
        if instance is not None and st.session_state.best_schedule is not None:
            self._create_mappings(instance)
            consistencia = 0
            days, desks = st.session_state.best_schedule.shape

            for desk in range(desks):
                values, counts = np.unique(
                    st.session_state.best_schedule[:, desk], return_counts=True
                )
                for val, count in zip(values, counts):
                    if count > 1 and val != -1:
                        consistencia += 1
            return consistencia
        else:
            return None

    def obtener_dia_reunion_equipo(
        self,
    ):
        instance = st.session_state.instancia_programada
        if instance is not None and st.session_state.best_schedule is not None:
            self._create_mappings(instance)
            dia_reuniones = []
            for group_name, employees_in_group in self.employees_g.items():
                att_max = 0
                day_max = 0
                max_conteo = 0
                for day_idx in range(len(self.days)):
                    conteo = 0
                    emp_indices = [
                        self.employees_reverse_map[emp] for emp in employees_in_group
                    ]
                    for emp_idx in emp_indices:
                        if emp_idx in st.session_state.best_schedule[day_idx]:
                            conteo += 1
                    attendace_g = conteo / len(emp_indices)
                    if attendace_g > att_max:
                        att_max = attendace_g
                        day_max = day_idx
                        max_conteo = conteo
                dia_reuniones.append(
                    {
                        "Grupo": group_name,
                        "Dia": self.days_map[day_max],
                        "% asistencia": f"{max_conteo}/{len(emp_indices)} ({int(att_max*100)}%)",
                    }
                )
            return dia_reuniones

        else:
            return None

    def preparar_datos_instancia(self, best_solution):
        """
        Carga los datos de la instancia proporcionada
        """

        instance = st.session_state.instancia_programada
        if instance is not None and best_solution is not None:
            self._create_mappings(instance)

            self.group_color_map = {}
            category_colors = sns.color_palette("husl", n_colors=len(self.groups))
            i = 0
            for emp in self.groups:
                self.group_color_map[emp] = (
                    f"background-color: rgb({int(category_colors[i][0]*255)}, {int(category_colors[i][1]*255)}, {int(category_colors[i][2]*255)})"
                )
                i += 1

            # Crear DataFrame de Empleados (ID y Grupo)
            employee_data = []
            for group_name, employee_list in instance["Employees_G"].items():
                for emp_id in employee_list:
                    employee_data.append(
                        {
                            "empleado_id": emp_id,
                            "grupo": group_name,
                            "color": self.group_color_map[group_name],
                        }
                    )
            df_empleados = pd.DataFrame(employee_data)

            # Crear DataFrame de Puestos (ID y Zona)

            self.zones_color_map = {}
            category_colors = sns.color_palette("husl", n_colors=len(self.zones))
            i = 0
            for zn in self.zones:
                self.zones_color_map[zn] = (
                    f"background-color: rgb({int(category_colors[i][0]*255)}, {int(category_colors[i][1]*255)}, {int(category_colors[i][2]*255)})"
                )
                i += 1

            desk_data = []
            for zone_name, desk_list in instance["Desks_Z"].items():
                for desk_id in desk_list:
                    desk_data.append(
                        {
                            "puesto_id": desk_id,
                            "zona": zone_name,
                            "color": self.zones_color_map[zone_name],
                        }
                    )
            df_puestos = pd.DataFrame(desk_data)

            assignment_matrix = np.full(
                (len(self.days), len(self.desks)), None, dtype=object
            )
            # convertir los id de empleados a los nombres originales
            for i in range(len(self.days)):
                for j in range(len(self.desks)):
                    employee_id = best_solution[i][j]
                    if employee_id != -1:
                        assignment_matrix[i][j] = self.employees_map[employee_id]

            df_assignments = pd.DataFrame(
                assignment_matrix, index=self.days, columns=self.desks
            )

            return df_empleados, df_puestos, df_assignments

    def seleccionar_colores(m: int):
        """
        Genera una paleta de 'm' colores distintivos usando la paleta 'husl' de Seaborn.
        Ideal para un gran número de categorías.

        Args:
            m (int): El número deseado de colores diferentes.

        Returns:
            list: Una lista de tuplas RGB (Rojo, Verde, Azul) de los colores.
        """
        if m <= 0:
            raise ValueError("La cantidad de colores (m) debe ser un entero positivo.")

        # Genera 'm' colores de la paleta 'husl'
        colores = sns.color_palette("husl", n_colors=m)
        return colores

    def run_assignment_algorithm(
        self,
    ):
        progreso = st.progress(0)
        mensaje = st.empty()
        algoritmo_rl_ils = ShiftSchedulerOptimizer(
            instance=st.session_state.instancia_programada,
            dqn_timesteps=self.num_iterations_rl,
            ils_timesteps=self.num_iterations_ils,
            required_in_office_day=self.dias_presencialidad,
            objective_weights=st.session_state.prioridad_objetivos,
        )

        def callback(p):
            progreso.progress(p)
            mensaje.text(f"Progreso: {int(p*100)}%")

        print("Running assignment algorithm...")
        # algoritmo_rl_ils.run_complete_optimization(callback=callback)

    def main(
        self,
    ):
        # Título principal
        st.markdown(
            """
        <div class="main-header">
            <h1>🧩 Sistema Inteligente de Asignación de Puestos (SIAP) </h1>
            <p>Optimización de espacios de trabajo híbridos mediante reinforcement learning y heurística ILS</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # st.logo("assets/logo2.png", size="large", icon_image="assets/logo2.png")

        if "prioridades_list" not in st.session_state:
            st.session_state.prioridades_list = [
                "Crítico",
                "Alta",
                "Media",
                "Baja",
                "Mínima",
            ]

        # Sidebar con información general
        with st.sidebar:

            st.image("assets/logo2.png", use_container_width=True, width=20)
            st.markdown(
                '<div class="sidebar-title">📊 Panel de Control</div>',
                unsafe_allow_html=True,
            )

            with st.expander("### ⚙️ Ajuste de parámetros del algoritmo"):
                # Seleccionar cantidad de iteraciones
                self.num_iterations_rl = st.number_input(
                    "Cantidad de Iteraciones agente RL",
                    min_value=10000,
                    max_value=300000,
                    value=60000,
                    step=1000,
                    key="rl_iterations",
                    disabled=st.session_state.locked,
                )
                self.num_iterations_ils = st.number_input(
                    "Cantidad de Iteraciones ILS",
                    min_value=500,
                    max_value=50000,
                    value=4000,
                    step=100,
                    key="ils_iterations",
                    disabled=st.session_state.locked,
                )

            with st.expander("### 📅 Número de dias presencialidad"):
                self.dias_presencialidad = st.slider(
                    "Días presencialidad",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    key="days",
                    disabled=st.session_state.locked,
                )

            with st.expander("### 🎯 Prioridades en la optimización"):
                with st.container():
                    # Seleccionar prioridades

                    self.prioridad_asignacion_empleados = st.pills(
                        "Asignación de empleados",
                        options=st.session_state.prioridades_list,
                        key="empleados",
                        default=st.session_state.prioridades_list[0],
                        disabled=st.session_state.locked,
                    )

                    self.prioridad_asistencia_grupos = st.pills(
                        "Reunión de seguimiento",
                        options=st.session_state.prioridades_list,
                        key="seguimiento",
                        default=st.session_state.prioridades_list[1],
                        disabled=st.session_state.locked,
                    )

                    self.prioridad_satisfaccion_empleados = st.pills(
                        "Satisfacción de requerimientos de días",
                        options=st.session_state.prioridades_list,
                        default=st.session_state.prioridades_list[2],
                        key="satisfaccion",
                        disabled=st.session_state.locked,
                    )

                    self.prioridad_dispersion_equipos = st.pills(
                        "Dispersión de equipos",
                        options=st.session_state.prioridades_list,
                        default=st.session_state.prioridades_list[3],
                        key="dispersion",
                        disabled=st.session_state.locked,
                    )

                    self.prioridad_consistencia_puestos = st.pills(
                        "Consistencia de puestos",
                        options=st.session_state.prioridades_list,
                        key="consistencia",
                        default=st.session_state.prioridades_list[4],
                        disabled=st.session_state.locked,
                    )

        # Contenido principal
        tab1, tab2, tab3 = st.tabs(
            [
                "🧠✨ Realizar programación",
                "📅 Resultados Generales",
                "👥 Detalles de la asignación",
            ]
        )

        with tab1:

            # Ejecutar algoritmo de asignación de puestos
            col1, col2 = st.columns(2)

            with col1:
                # Seleccionar instancia
                with st.container():
                    st.markdown("### 📤 Seleccionar Instancia")
                    instancia_seleccion = st.radio(
                        "Instancia",
                        [
                            "Instancias predefinidas",
                            "Cargar instancia nueva",
                            "Ver resultados",
                        ],
                        horizontal=True,
                        disabled=st.session_state.locked,
                    )
                    instancias_predefinidas = self.instance_reader.list_instances()
                    if instancia_seleccion == "Instancias predefinidas":

                        # instancias_predefinidas= []
                        # st.markdown('<span style="color:red; font-style:italic;">Instancia seleccionada por defecto</span>', unsafe_allow_html=True)
                        selected_instance = st.selectbox(
                            "Instancia",
                            instancias_predefinidas,
                            key="instance_selection",
                            label_visibility="hidden",
                        )
                        instance = self.instance_reader.read_instance(selected_instance)
                        if isinstance(instance, str):
                            st.warning(instance)
                            st.session_state.nro_puestos = None
                            st.session_state.nro_empleados = None
                            st.session_state.nro_dias = None
                            st.session_state.nro_zonas = None
                            return
                        else:
                            st.session_state.instancia_cargada = instance
                            st.session_state.nro_puestos = len(instance["Desks"])
                            st.session_state.nro_empleados = len(instance["Employees"])
                            st.session_state.nro_dias = len(instance["Days"])
                            st.session_state.nro_zonas = len(instance["Zones"])
                            ejecutar_btn = st.button(
                                " ▶️ Ejecutar", 
                                on_click=self.bloquear_inputs,
                                disabled=st.session_state.locked
                            )
                    elif instancia_seleccion == "Cargar instancia nueva":
                        uploaded_file = st.file_uploader(
                            "Cargar archivo JSON", type="json"
                        )
                        if uploaded_file is not None:
                            selected_instance = uploaded_file.name
                            # Leer una instancia
                            instance = self.instance_reader.read_instance(uploaded_file)
                            if isinstance(instance, str):
                                st.error(instance)
                                st.session_state.nro_puestos = None
                                st.session_state.nro_empleados = None
                                st.session_state.nro_dias = None
                                st.session_state.nro_zonas = None
                            else:
                                st.success("Instancia cargada correctamente")
                                st.session_state.instancia_cargada = instance
                                st.session_state.nro_puestos = len(instance["Desks"])
                                st.session_state.nro_empleados = len(
                                    instance["Employees"]
                                )
                                st.session_state.nro_dias = len(instance["Days"])
                                st.session_state.nro_zonas = len(instance["Groups"])
                                ejecutar_btn = st.button(
                                    " ▶️ Ejecutar", 
                                    on_click=self.bloquear_inputs,
                                    disabled=st.session_state.locked
                                )
                        else:
                            st.warning(
                                "Por favor, cargue una instancia antes de ejecutar el algoritmo."
                            )
                            st.session_state.nro_puestos = None
                            st.session_state.nro_empleados = None
                            st.session_state.nro_dias = None
                            st.session_state.nro_zonas = None
                            return
                    else:
                        selected_instance = st.selectbox(
                            "Instancia",
                            instancias_predefinidas,
                            key="instance_selection",
                            label_visibility="hidden",
                        )
                        instance = self.instance_reader.read_instance(selected_instance)
                        st.session_state.instancia_cargada = instance
                        st.session_state.nro_puestos = len(instance["Desks"])
                        st.session_state.nro_empleados = len(instance["Employees"])
                        st.session_state.nro_dias = len(instance["Days"])
                        st.session_state.nro_zonas = len(instance["Groups"])
                        ver_resultados = st.button("🔍 Ver resultados")

                if instancia_seleccion != "Ver resultados":
                    # Botón para ejecutar el algoritmo
                    if ejecutar_btn and st.session_state.instancia_cargada is not None:

                        # st.session_state.locked = True

                        if self._validar_prioridades():
                            st.session_state.instancia_programada = (
                                st.session_state.instancia_cargada
                            )

                            # Crear el contenedor principal
                            with st.container():
                                progress_placeholders = {}
                                st.markdown(
                                    '<div class="progress-title">⏳ Progreso de Optimización DRL + ILS </div>',
                                    unsafe_allow_html=True,
                                )

                                st.markdown(
                                    f'<div class="ils-progress-label">DRL Agent</div>',
                                    unsafe_allow_html=True,
                                )
                                progress_placeholders["drl"] = st.progress(
                                    0, text=f"DRL: Preparando..."
                                )

                                for i in range(1, 6):
                                    pid = i
                                    st.markdown(
                                        f'<div class="ils-progress-label">Solver ILS #{pid}</div>',
                                        unsafe_allow_html=True,
                                    )
                                    progress_placeholders[pid] = st.progress(
                                        0, text=f"S{pid}: Preparando..."
                                    )

                                st.markdown("</div>", unsafe_allow_html=True)

                            # Crear placeholders para las barras de progreso
                            # progress_placeholders = {}
                            # for i in range(1,6):
                            #     pid = i
                            #     progress_placeholders[pid] = st.progress(0, text=f"{pid}: 0%")

                            algoritmo_rl_ils = ShiftSchedulerOptimizer(
                                instance=st.session_state.instancia_programada,
                                dqn_timesteps=self.num_iterations_rl,
                                ils_timesteps=self.num_iterations_ils,
                                required_in_office_day=self.dias_presencialidad,
                                objective_weights=st.session_state.prioridad_objetivos,
                            )

                            with st.spinner(
                                "Ejecutando algoritmo DRL + ILS...", show_time=True
                            ):

                                # Diccionario para comunicar resultados entre hilos
                                results = {
                                    "best_score": None,
                                    "best_schedule": None,
                                    "finished": False,
                                    "start_time": None,
                                    "end_time": None,
                                }

                                def run_optimization():
                                    start_time = time.time()
                                    results["start_time"] = start_time
                                    best_score, best_schedule = (
                                        algoritmo_rl_ils.run_complete_optimization()
                                    )
                                    results["end_time"] = time.time()
                                    # Guardar en el diccionario compartido
                                    results["best_score"] = best_score
                                    results["best_schedule"] = best_schedule
                                    results["finished"] = True

                                # Iniciar optimización en hilo separado
                                thread = threading.Thread(target=run_optimization)
                                thread.start()

                                # Actualizar progress bars mientras corre
                                status_container = st.empty()

                                while (
                                    algoritmo_rl_ils.is_optimization_running()
                                    or thread.is_alive()
                                ):
                                    time.sleep(0.5)
                                    # Obtener progreso actual

                                    current_progress = algoritmo_rl_ils.get_progress()
                                    current_progress_drl = (
                                        algoritmo_rl_ils.get_progress_drl()
                                    )

                                    # Actualizar cada barra
                                    for (
                                        solver_id,
                                        progress_bar,
                                    ) in progress_placeholders.items():

                                        if solver_id == "drl":
                                            current_iter = current_progress_drl.get(
                                                "drl", 0
                                            )
                                            total_iterations = self.num_iterations_rl
                                            progress = min(
                                                current_iter / total_iterations, 1.0
                                            )
                                            progress_bar.progress(
                                                progress,
                                                text=f"{solver_id}: {current_iter}/{total_iterations}",
                                            )
                                        else:
                                            current_iter = current_progress.get(
                                                solver_id, 0
                                            )

                                            total_iterations = self.num_iterations_ils
                                            progress = min(
                                                current_iter / total_iterations, 1.0
                                            )
                                            progress_bar.progress(
                                                progress,
                                                text=f"{solver_id}: {current_iter}/{total_iterations}",
                                            )

                                        if progress == 1.0:
                                            text = f"✅ {solver_id}: ¡Completado! ({current_iter}/{total_iterations})"
                                        elif progress > 0:
                                            text = f"⚡ {solver_id}: {current_iter}/{total_iterations} ({int(progress*100)}%)"
                                        else:
                                            text = f"🔄 {solver_id}: En proceso DRL..."
                                        progress_bar.progress(progress, text=text)

                                thread.join()

                                if results["finished"]:
                                    st.session_state.best_score = results["best_score"]
                                    st.session_state.best_schedule = results[
                                        "best_schedule"
                                    ]

                                    start_time = results["start_time"]
                                    end_time = results["end_time"]
                                    minutes, seconds = divmod(end_time - start_time, 60)
                                    st.session_state.start_time = start_time
                                    st.session_state.end_time = end_time
                                    status_container.success(
                                        f"El algoritmo se ha ejecutado correctamente en {minutes:.0f} minutos y {seconds:.0f} segundos.",
                                        icon="✅",
                                    )
                                    time.sleep(5)
                                    results_dict = {}
                                    results_dict[selected_instance] = {
                                        "best_score": st.session_state.best_score,
                                        "best_schedule": st.session_state.best_schedule,
                                        "start_time": start_time,
                                        "end_time": end_time,
                                    }

                                    # self.save_results(results_dict)
                                else:
                                    status_container.error(
                                        "Optimización no completada. Por favor, vuelva a intentarlo.",
                                        icon="⚠️",
                                    )
                                    st.session_state.best_score = None
                                    st.session_state.best_schedule = None
                                st.session_state.locked = False
                                st.rerun()
                elif ver_resultados:

                    load_results = self.load_results(selected_instance)
                    if load_results is None:
                        st.error(
                            "No se encontraron resultados para la instancia seleccionada."
                        )
                        return
                    else:
                        st.session_state.instancia_programada = (
                            st.session_state.instancia_cargada
                        )
                        st.session_state.best_score = load_results["best_score"]
                        st.session_state.best_schedule = load_results["best_schedule"]
                        st.session_state.start_time = load_results["start_time"]
                        st.session_state.end_time = load_results["end_time"]
                        st.success(
                            "Resultados cargados correctamente. \n \n Puede visualizarlos en las pestañas Resultados Generales y Detalles de la asignación.",
                            icon="✅",
                        )

            with col2:
                st.markdown("### 📋 Información de la instancia")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("👥 Número empleados", st.session_state.nro_empleados)
                    st.metric("💺 Cantidad Puestos", st.session_state.nro_puestos)
                with col2:
                    st.metric("📅 Días", st.session_state.nro_dias)
                    st.metric("🏢 Número Zonas", st.session_state.nro_zonas)

                st.markdown("#### 📦 Instancia cargada")
                st.json(st.session_state.instancia_cargada)

        with tab2:
            st.session_state.total_asignaciones = 0
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("## 👥 Asignaciones")
                if st.session_state.best_schedule is not None:
                    self._create_mappings(st.session_state.instancia_programada)
                    asignaciones = {}
                    try:
                        for day in range(st.session_state.best_schedule.shape[0]):
                            for desk in range(st.session_state.best_schedule.shape[1]):
                                employee_idx = st.session_state.best_schedule[day][desk]
                                if employee_idx != -1:
                                    st.session_state.total_asignaciones += 1
                                    try:
                                        asignaciones[employee_idx].append(
                                            [self.days[day], self.desks[desk]]
                                        )
                                    except:
                                        asignaciones[employee_idx] = [
                                            [self.days[day], self.desks[desk]]
                                        ]

                        # mostrar como un dataframe
                        df = pd.DataFrame(
                            list(asignaciones.items()),
                            columns=["Empleado_idx", "Asignación"],
                        )
                        df.sort_values(by="Empleado_idx", inplace=True, ascending=True)
                        df["Empleado"] = df["Empleado_idx"].map(self.employees_map)
                        df.set_index("Empleado", inplace=True)
                        del df["Empleado_idx"]
                        st.dataframe(
                            df.style.set_properties(
                                **{
                                    "background-color": "#eff8ea",
                                    "color": "#021A02",
                                    "font-weight": "bold",
                                    "text-align": "center",
                                }
                            ),
                            use_container_width=True,
                            # hide_index=True
                        )

                    except Exception as e:
                        st.error(f"Error al mostrar la matriz de asignación: {e}")
                else:
                    st.warning("No se ha ejecutado el algoritmo.")

            with col2:

                st.markdown("## 📊 Métricas de la solución")
                col1, col2 = st.columns(2)
                if st.session_state.best_schedule is not None:

                    satisfaccion_dias, max_satisfaccion_dias = (
                        self.obtener_porcentaje_dias_preferidos()
                    )
                    consistencia_puestos = self.consistencia_puestos()

                    with col1:

                        st.metric(
                            "✅ Total Asignaciones",
                            f"{st.session_state.total_asignaciones}/{(len(self.employees)*self.dias_presencialidad)}",
                            f"{(st.session_state.total_asignaciones/(len(self.employees)*self.dias_presencialidad))*100:.0f}%",
                            help="El porcentaje de asignaciones realizadas del total  -> numero empleados * dias presencialidad",
                            border=True,
                        )

                        st.metric(
                            "⭐ Satisfacción dias preferidos",
                            f"{satisfaccion_dias}/{max_satisfaccion_dias}",
                            f"{(satisfaccion_dias/max_satisfaccion_dias*100):.1f}%",
                            help="El porcentaje de dias preferidos por los empleados que se asignaron",
                            border=True,
                        )
                        minutes, seconds = divmod(
                            st.session_state.end_time - st.session_state.start_time, 60
                        )
                        st.metric(
                            "⏳ Tiempo de ejecución",
                            f"{{minutes:.0f}} min y {{seconds:.0f}} seg".format(
                                minutes=minutes, seconds=seconds
                            ),
                            help="Tiempo de ejecución del algoritmo",
                            border=True,
                        )

                    with col2:
                        st.metric(
                            "🎯 Consistencia Puestos",
                            f"{consistencia_puestos}/{len(self.employees)}",
                            f"{consistencia_puestos / len(self.employees)*100:.1f}%",
                            help="Número de empleados que tienen asignado el mismo puesto toda la semana",
                            border=True,
                        )

                        data = self.obtener_dia_reunion_equipo()  # lista de dicts
                        df = pd.DataFrame(data)
                        st.dataframe(
                            df.style.set_properties(
                                **{
                                    "background-color": "#f0faf5",
                                    "color": "#021A02",
                                    "font-weight": "bold",
                                    "text-align": "center",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                else:
                    st.warning("No se ha ejecutado el algoritmo.")

        with tab3:
            st.markdown("## 👥 Distribución por grupos")
            st.markdown(
                "En este diagrama se esquematiza la distribución de empleados para cada puesto.\n\n"
                "El color de la celda representa el grupo al que pertenece el empleado asignado."
            )
            if st.session_state.best_schedule is not None:
                self._create_mappings(st.session_state.instancia_programada)
                groups = ["Seleccione un grupo"] + [g for g in self.groups]
                col1, col2 = st.columns(2)
                with col1:
                    selected_group = st.selectbox(
                        "Seleccione un grupo", groups, key="group_selected"
                    )
            if st.session_state.best_schedule is not None:
                try:
                    if (
                        selected_group == "Seleccione un grupo"
                    ):  # muestra toda la matriz de asugnaciones
                        df_empleados, df_puestos, df_assignments = (
                            self.preparar_datos_instancia(
                                st.session_state.best_schedule
                            )
                        )
                        df_display = df_assignments.fillna("Libre")

                        def resaltar_negativos(val):
                            """
                            Resalta el texto en rojo si el valor es -1, de lo contrario, no aplica estilo.
                            """
                            if val == "Libre":
                                return "color:red"
                            else:
                                return f'{df_empleados[df_empleados["empleado_id"] == val]["color"].values[0]}'

                        df_display = df_display.style.applymap(resaltar_negativos)
                        st.dataframe(df_display, use_container_width=False)
                    else:
                        df_empleados, df_puestos, df_assignments = (
                            self.preparar_datos_instancia(
                                st.session_state.best_schedule
                            )
                        )
                        df_display = df_assignments.fillna("Libre")
                        # iterar por todos los elementos del dataframe
                        for index, row in df_display.iterrows():
                            for column in df_display.columns:
                                value = row[column]
                                if (
                                    value == "Libre"
                                    or self.employee_group_map[value] != selected_group
                                ):
                                    df_display.at[index, column] = "-"

                        def resaltar_otro_grupo(val):
                            if val == "-":
                                return "background-color: #444444; color: white;color: darkgray;"
                            else:
                                return "background-color: #a8d5ba; color: black;"

                        styled_df = df_display.style.applymap(resaltar_otro_grupo)
                        st.dataframe(styled_df, use_container_width=False)

                except Exception as e:
                    st.error(f"Error al mostrar la matriz de asignación: {e}")

                st.markdown("## 🏢 Distribución por Zonas")
                st.markdown(
                    "En este diagrama se esquematiza la distribución de los **grupos** por zonas.\n\n "
                    "Cada color representa a una zona diferente."
                )
                try:
                    if selected_group == "Seleccione un grupo":
                        df_empleados, df_puestos, df_assignments = (
                            self.preparar_datos_instancia(
                                st.session_state.best_schedule
                            )
                        )
                        df_display = df_assignments.fillna("Libre")
                        df_display = df_display.map(
                            lambda x: (
                                "Libre" if x == "Libre" else self.employee_group_map[x]
                            )
                        )

                        # Función para aplicar colores basados en el contenido
                        def aplicar_colores(val, row_idx, col_idx):
                            desk = f"D{col_idx}"
                            if val == "Libre":
                                return "color: red; font-weight: bold"
                            else:
                                return f'{df_puestos[df_puestos["puesto_id"] == desk]["color"].values[0]}'

                        # Aplicar estilos usando una función que recibe el DataFrame completo
                        def aplicar_estilos(df):
                            styles = pd.DataFrame(
                                "", index=df.index, columns=df.columns
                            )
                            for row in range(df.shape[0]):
                                for col in range(df.shape[1]):
                                    styles.iloc[row, col] = aplicar_colores(
                                        df.iloc[row, col], row, col
                                    )
                            return styles

                        # Aplicar los estilos al DataFrame
                        styled_df = df_display.style.apply(aplicar_estilos, axis=None)

                        # Mostrar el DataFrame estilizado
                        st.dataframe(styled_df, use_container_width=False)
                    else:
                        df_empleados, df_puestos, df_assignments = (
                            self.preparar_datos_instancia(
                                st.session_state.best_schedule
                            )
                        )
                        df_display = df_assignments.fillna("Libre")

                        def aplicar_colores(val, row_idx, col_idx):
                            desk = f"D{col_idx}"
                            if val == "Libre":
                                return "background-color: #444444; color: white; color: balck;"
                            else:
                                return f'{df_puestos[df_puestos["puesto_id"] == desk]["color"].values[0]}'

                        # Aplicar estilos usando una función que recibe el DataFrame completo
                        def aplicar_estilos(df):
                            styles = pd.DataFrame(
                                "", index=df.index, columns=df.columns
                            )
                            for row in range(df.shape[0]):
                                for col in range(df.shape[1]):
                                    styles.iloc[row, col] = aplicar_colores(
                                        df.iloc[row, col], row, col
                                    )
                            return styles

                        df_display = df_display.map(
                            lambda x: (
                                "Libre" if x == "Libre" else self.employee_group_map[x]
                            )
                        )
                        df_display = df_display.map(
                            lambda x: "Libre" if x != selected_group else x
                        )
                        df_display = df_display.style.apply(aplicar_estilos, axis=None)
                        st.dataframe(df_display, use_container_width=False)

                except Exception as e:
                    st.error(f"Error al procesar los datos: {e}")
            else:
                st.warning("No se ha ejecutado el algoritmo de programación.")


if __name__ == "__main__":
    DeskAssignmentVisualization().main()
