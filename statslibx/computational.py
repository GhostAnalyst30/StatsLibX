from typing import Union, Optional, Literal
import pandas as pd

class ComputationalStats:
    """
    Class for computational statistics
    """
    
    def __init__(self, seed: Optional[int] = None):
        pass
    
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
