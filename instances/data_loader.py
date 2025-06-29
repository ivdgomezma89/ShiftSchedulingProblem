import os
import json
from pathlib import Path


class InstanceReader:
    def __init__(self, instances_folder="data"):
        """
        Inicializa el lector con la carpeta de instancias.
        
        Args:
            instances_folder: nombre de la carpeta donde están las instancias
        """
        # La carpeta de instancias está al mismo nivel que data_loader.py
        current_file = Path(__file__)
        self.instances_path = current_file.parent / instances_folder
        #self.instances_path.mkdir(exist_ok=False)
    
    def read_instance(self, source):
        """
        Lee una instancia desde archivo local o archivo subido.
        
        Args:
            source: nombre del archivo (str) o archivo subido
            instance_name: nombre específico si se busca por nombre
        
        Returns:
            dict: datos de la instancia
        """
        if isinstance(source, str):
            # Leer desde archivo local
            file_path = self.instances_path / f"{source}.json"
            if not file_path.exists():
                file_path = self.instances_path / source
            
            with open(file_path, 'r') as f:
                instance = json.load(f)
        else:
            # Leer desde archivo subido
            instance = json.load(source)
        
        valid = self._validate_instance(instance)
        if not valid:
            return ("La instancia no es valida porque no tiene las claves requeridas.")
        return instance
    
    def _validate_instance(self, instance):
        """Valida que la instancia tenga las claves requeridas."""
        required_keys = ['Days', 'Desks', 'Employees', 'Days_E', 'Desks_E', 'Employees_G', 'Desks_Z']
        
        for key in required_keys:
            if key not in instance:
                return False
        
        return True
    
    def list_instances(self):
        """Lista las instancias disponibles."""
        if not self.instances_path.exists():
            return []
        
        return [f.stem for f in self.instances_path.glob("*.json")]