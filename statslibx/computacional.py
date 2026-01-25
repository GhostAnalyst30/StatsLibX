from typing import Union, Optional, Literal
import numpy as np
import pandas as pd
import polars as pl
import os

class ComputationalStats:
    """
    Class for computational statistics
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray],
                sep: str = None,
                decimal: str = None,
                thousand: str = None,
                backend: Literal['pandas', 'polars'] = 'pandas'):
        """
        # Initialize DataFrame
        
        ## **Parameters:**

        - **data** : Data to analyze
        - **sep** : Column separator
        - **decimal** : Decimal separator
        - **thousand** : Thousand separator
        - **backend** : 'pandas' or 'polars' for processing
        (Proximamente estara habilitado polars para big data)

        **Examples:**

        ``Example 1:
        stats = DescriptiveStats(data)
        ``
        """

        if isinstance(data, str) and os.path.exists(data):
                data = ComputationalStats.from_file(data).data

        if isinstance(data, pl.DataFrame):
            raise TypeError(
                "Polars aún no soportado. Use pandas.DataFrame."
            )


        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])]) \
                    if isinstance(data, pd.DataFrame) else pl.DataFrame(data, )
        
        self.data = data
        self.backend = backend
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.sep = sep
        self.decimal = decimal
        self.thousand = thousand
    
    @classmethod
    def from_file(self, path: str):
        """
        Carga automática de archivos y devuelve instancia de Intelligence.
        Soporta CSV, Excel, TXT, JSON, Parquet, Feather, TSV.
        Automatic file upload and returns Intelligence instance. 
        Supports CSV, Excel, TXT, JSON, Parquet, Feather, TSV.

        Parametros / Parameters:
        ------------------------
        path : str
            Ruta del archivo
            File path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado / File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path, sep=self.sep, decimal=self.decimal, thousand=self.thousand)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, decimal=self.decimal, thousand=self.thousand)

        elif ext in [".txt", ".tsv"]:
            df = pd.read_table(path, sep=self.sep, decimal=self.decimal, thousand=self.thousand)

        elif ext == ".json":
            df = pd.read_json(path)

        elif ext == ".parquet":
            df = pd.read_parquet(path)

        elif ext == ".feather":
            df = pd.read_feather(path)

        else:
            raise ValueError(f"Formato no soportado / Unsupported format: {ext}")

        return ComputationalStats(df)
    
    def monte_carlo(self, function, n: int = 100, return_simulations: bool = False, **kwargs) -> pd.DataFrame:
        """
        Realiza simulaciones de Monte Carlo para una función y devuelve un DataFrame con las simulaciones y sus resultados.
        """
        samples = []

        for _ in range(n):
            sample = function(**kwargs)
            samples.append(float(sample))

        mean = sum(samples) / n
        variance = sum((x - mean)**2 for x in samples) / n
        std = variance**0.5

        if return_simulations:
            return {
                "mean": float(mean),
                "std": float(std),
                "samples": samples
            }

        else:
            return {
                "mean": float(mean),
                "std": float(std)
            }
