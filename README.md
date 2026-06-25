<p align="center">
  <img src="https://raw.githubusercontent.com/GhostAnalyst30/StatsLibX/main/StatsLibX.png" alt="StatsLibX" width="420"/>
</p>

<h1 align="center">StatsLibX</h1>

<p align="center">
  <strong>Estadística descriptiva, inferencial y computacional para Python — con pandas, polars y ViewX.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/statslibx/"><img src="https://img.shields.io/pypi/v/statslibx?label=PyPI&color=7c6af7" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/statslibx/"><img src="https://img.shields.io/pypi/pyversions/statslibx?label=Python&color=4fd1c5" alt="Python versions"/></a>
  <a href="https://github.com/GhostAnalyst30/StatsLibX/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License MIT"/></a>
  <a href="https://github.com/GhostAnalyst30/StatsLibX"><img src="https://img.shields.io/github/stars/GhostAnalyst30/StatsLibX?style=social" alt="GitHub stars"/></a>
</p>

<p align="center">
  <a href="https://ghostanalyst30.github.io/StatsLibX/">Documentación</a> ·
  <a href="https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb">Notebook API</a> ·
  <a href="https://github.com/GhostAnalyst30/StatsLibX/issues">Issues</a> ·
  <a href="https://ghostanalyst30.github.io/ViewX/">ViewX</a>
</p>

---

**StatsLibX** es una librería de Python moderna para análisis estadístico y ciencia de datos. Ofrece una API clara basada en clases, soporte dual **pandas / polars**, datasets embebidos, preprocesamiento, estadística computacional y un puente de reportes con **ViewX**.

> **Versión actual:** `0.2.9` · **Autor:** Emmanuel Ascendra

---

## Novedades en v0.2.9

| Área | Cambio |
|------|--------|
| **Arquitectura** | Capa `Backend` unificada en todos los módulos de dominio |
| **Polars** | `load_dataset(backend="polars")` y constructores compatibles con `pl.DataFrame` |
| **API** | `DescriptiveStats.from_file()`, `InferentialStats.from_file()`, `ComputationalStats.help()` |
| **Preprocessing** | `clean_data()` ampliado (escalado, outliers, transforms) y `change_dtypes()` con polars |
| **ViewX** | `to_report_data()` — serializa resultados statslibx para `Report` / `HTML` |
| **Packaging** | `pyproject.toml`, extras opcionales, CLI `statslibx`, marcador `py.typed` |
| **Docs web** | Sitio Next.js v0.2.9, playground Pyodide alineado con la API real |

---

## Instalación

```bash
pip install statslibx
```

### Extras opcionales

```bash
# ViewX (reportes HTML, slides, matrices)
pip install statslibx[viewx]

# Regresión avanzada (statsmodels / sklearn)
pip install statslibx[statsmodels,sklearn]

# Excel + todo incluido
pip install statslibx[excel]
pip install statslibx[all]
```

| Extra | Paquetes |
|-------|----------|
| `viewx` | viewx ≥ 0.2.3 |
| `statsmodels` | statsmodels ≥ 0.13 |
| `sklearn` | scikit-learn ≥ 1.0 |
| `excel` | openpyxl ≥ 3.0 |
| `all` | Todos los anteriores |

**Requisitos:** Python ≥ 3.9 · numpy · pandas · scipy · matplotlib · seaborn · plotly · sympy

---

## Inicio rápido

```python
import statslibx as slx
from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, Preprocessing
from statslibx.datasets import load_iris, generate_dataset

print(f"StatsLibX v{slx.__version__}")

# Cargar dataset embebido (iris, penguins, titanic)
iris = load_iris()
print(iris.head())

# Estadística descriptiva
ds = DescriptiveStats(iris)
print(ds.mean("sepal_length"))
print(ds.summary())

# Prueba inferencial
inf = InferentialStats(iris)
print(inf.t_test_1sample("sepal_length", popmean=5.8))

# Desde archivo
stats = DescriptiveStats.from_file("mi_datos.csv")

# Datos sintéticos
schema = {
    "age": {"dist": "normal", "mean": 35, "std": 10, "type": "int"},
    "group": {"dist": "categorical", "choices": ["A", "B", "C"]},
}
df = generate_dataset(n_rows=500, schema=schema, seed=42)
```

### Motores de datos (pandas / polars)

Todas las clases que reciben DataFrames soportan el parámetro `backend`:

```python
from statslibx import DescriptiveStats, InferentialStats, ComputationalStats, Preprocessing

df = load_iris()

# Auto-detecta: pandas DataFrame → pandas, polars DataFrame → polars
DescriptiveStats(df)
InferentialStats(df)
ComputationalStats(df)
Preprocessing(df)

# Forzar motor polars (convierte pandas → polars internamente)
DescriptiveStats(df, backend="polars")
InferentialStats(df, backend="polars")
ComputationalStats(df, backend="polars")
Preprocessing(df, backend="polars")

# Forzar motor pandas (convierte polars → pandas)
# InferentialStats(pl_df, backend="pandas")

# Desde archivo
DescriptiveStats.from_file("datos.csv", backend="polars")
InferentialStats.from_file("datos.csv", backend="polars")
ComputationalStats.from_file("datos.csv", backend="polars")
Preprocessing.from_file("datos.csv", backend="polars")

# Inspeccionar motor activo
stats = DescriptiveStats(df, backend="polars")
print(stats.backend)  # "polars"
```

Carga directa con polars:

```python
from statslibx.datasets import load_dataset

df = load_dataset("iris.csv", backend="polars")  # requiere pip install polars
stats = DescriptiveStats(df)
print(stats.backend)  # "polars" (auto-detectado)
```

---

## Módulos

| Clase / Módulo | Descripción |
|----------------|-------------|
| **`DescriptiveStats`** | Media, mediana, varianza, correlación, regresión lineal, outliers, resúmenes |
| **`InferentialStats`** | t-tests, ANOVA, chi-cuadrado, intervalos de confianza, normalidad |
| **`ComputationalStats`** | Regresión polinomial, bootstrap, k-means, interpolación, correlación |
| **`Preprocessing`** | Limpieza, nulos, escalado, outliers, calidad de datos, dtypes |
| **`UtilsStats`** | Carga de archivos, visualización (matplotlib / seaborn / plotly), effect size |
| **`datasets`** | `load_dataset`, `load_iris`, `load_penguins`, `generate_dataset` |
| **`Backend`** | Abstracción pandas / polars (`statslibx.backend`) |
| **`viewx`** | `HTML`, `Slides`, `Report`, `DataMatrix`, `to_report_data` |

---

## Integración ViewX

StatsLibX se conecta con [ViewX](https://ghostanalyst30.github.io/ViewX/) para generar reportes y visualizaciones a partir de resultados estadísticos.

```python
from statslibx import DescriptiveStats, Report, to_report_data

df = load_iris()
summary = DescriptiveStats(df).summary()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/GhostAnalyst30/ViewX/main/images_for_git/DashBoard_Example.png" alt="ViewX example" width="600"/>
</p>

```bash
pip install statslibx[viewx]
```

---

## CLI — Terminal

StatsLibX incluye una interfaz de línea de comandos para explorar CSV sin escribir código.

```bash
statslibx data iris.csv --summary --types --missing
statslibx describe iris.csv --numeric
statslibx describe iris.csv --categorical
statslibx quality iris.csv --verbose
statslibx preview iris.csv -n 10
statslibx info iris.csv --detailed
statslibx --help
```

---

## Estadística computacional

```python
from statslibx import ComputationalStats

cs = ComputationalStats(df, seed=42)

# Regresión con términos de interacción
model = cs.regression(X=["age", "score"], y="income", interaction_terms=True)
print(model.get_formula())
print(model.summary())

# Bootstrap
boot = cs.bootstrapping("income", n_samples=1000, statistic="mean")
print(boot.summary())

# Clustering
kmeans = cs.k_means(k=3)
elbow = cs.elbow_method(max_k=10)
```

---

## Preprocesamiento

```python
pp = Preprocessing(df)

pp.data_quality()
pp.clean_data(
    drop_duplicates=True,
    handle_missing=True,
    missing_strategy="median",
    scale=True,
    scaling_method="standard",
    remove_outliers=True,
)
pp.preview_data(n=5)
```

---

## Documentación

| Recurso | Enlace |
|---------|--------|
| Documentación estática | [GitHub Pages](https://statslibx.vercel.app/) |
| Notebook completo (181 celdas) | [how_use_statslibx.ipynb](https://github.com/GhostAnalyst30/StatsLibX/blob/main/how_use_statslibx.ipynb) |
| Repositorio | [github.com/GhostAnalyst30/StatsLibX](https://github.com/GhostAnalyst30/StatsLibX) |
| ViewX | [ViewX Page](https://viewx.vercel.app/) |

---

## Estructura del paquete

```
statslibx/
├── descriptive.py      # DescriptiveStats, DescriptiveSummary, LinearRegressionResult
├── inferential.py      # InferentialStats, TestResult
├── computational.py    # ComputationalStats, RegressionResult, BootstrappingResult
├── preprocessing/      # Preprocessing
├── datasets/           # iris, penguins, titanic + generate_dataset
├── utils.py            # UtilsStats (I/O, plots, outliers)
├── backend.py          # Backend pandas / polars
├── viewx/              # Puente ViewX + to_report_data
├── cli.py              # statslibx CLI
└── py.typed            # PEP 561 typed package
```
---

## Contribuciones

¡Todas las mejoras e ideas son bienvenidas!

Abre un [issue](https://github.com/GhostAnalyst30/StatsLibX/issues) o un pull request en GitHub.

**Contacto:** [ascendraemmanuel@gmail.com](mailto:ascendraemmanuel@gmail.com)

---

<p align="center">
  Desarrollado por <strong>Emmanuel Ascendra</strong> · StatsLibX v0.2.9 · MIT License
</p>
