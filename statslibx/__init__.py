"""
StatsLibx - Librería de Estadística para Python
Autor: Emmanuel Ascendra
Versión: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "Emmanuel Ascendra"

# Importar las clases principales
from .descriptive import DescriptiveStats, DescriptiveSummary
from .inferential import InferentialStats, TestResult
from .computacional import ComputationalStats
from .utils import UtilsStats
from .preprocessing import Preprocessing
from .datasets import load_dataset, generate_dataset

# Definir qué se expone cuando se hace: from statslib import *
__all__ = [
    # Clases principales
    'DescriptiveStats',
    'InferentialStats',
    'ProbabilityStats',
    'ComputationalStats',
    'UtilsStats',
    'Preprocessing',
    'load_dataset',
    'generate_dataset'
]

# Mensaje de bienvenida (opcional)
def welcome():
    """Muestra información sobre la librería"""
    print(f"StatsLibx v{__version__}")
    print(f"Librería de estadística descriptiva e inferencial")
    print(f"Autor: {__author__}")
    print(f"\nClases disponibles:")
    print(f"  - DescriptiveStats: Estadística descriptiva")
    print(f"  - InferentialStats: Estadística inferencial")
    print(f"  - ComputacionalStats: En desarrollo")
    print(f"  - UtilsStats: Utilidades Extras")
    print(f"\nMódulos disponibles:")
    print(f"  - Datasets: Carga de Datasets")
    print(f"  - Preprocessing: Preprocesamiento de datos")
    print(f"\nPara más información: help(statslibx)")