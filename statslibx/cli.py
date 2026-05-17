import argparse
import statslibx as slx
from statslibx.datasets import load_dataset
from statslibx.preprocessing import Preprocessing
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        prog="statslibx",
        description="Statslibx - Data analysis from terminal"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # describe - Estadísticas descriptivas completas
    describe = subparsers.add_parser("describe", help="Estadísticas descriptivas")
    describe.add_argument("file", help="Ruta del archivo o nombre del dataset (ej: iris, titanic)")
    describe.add_argument("-n", "--numeric", action="store_true", help="Solo columnas numéricas")
    describe.add_argument("-c", "--categorical", action="store_true", help="Solo columnas categóricas")

    # quality - Calidad de datos
    quality = subparsers.add_parser("quality", help="Reporte de calidad de datos")
    quality.add_argument("file", help="Ruta del archivo o nombre del dataset")
    quality.add_argument("-v", "--verbose", action="store_true", help="Mostrar detalles")

    # preview - Vista previa
    preview = subparsers.add_parser("preview", help="Vista previa de los datos")
    preview.add_argument("file", help="Ruta del archivo o nombre del dataset")
    preview.add_argument("-n", "--rows", type=int, default=5, help="Número de filas (default: 5)")
    preview.add_argument("-s", "--sample", action="store_true", help="Muestra aleatoria")

    # info - Información completa del dataset (NUEVO)
    info = subparsers.add_parser("info", help="Información completa del dataset")
    info.add_argument("file", help="Ruta del archivo o nombre del dataset")
    info.add_argument("-d", "--detailed", action="store_true", help="Información detallada (tipos, nulos, memoria)")

    # data - Comando específico que pediste
    data = subparsers.add_parser("data", help="Resumen del dataset (filas, columnas, tipos)")
    data.add_argument("file", help="Ruta del archivo o nombre del dataset (ej: iris, titanic, o archivo.csv)")
    data.add_argument("-s", "--summary", action="store_true", help="Mostrar resumen estadístico básico")
    data.add_argument("-t", "--types", action="store_true", help="Mostrar tipos de datos")
    data.add_argument("-m", "--missing", action="store_true", help="Mostrar valores faltantes")

    args = parser.parse_args()

    if not args.command:
        print(slx.welcome())
        return

    # Cargar datos (soporta datasets internos y archivos externos)
    df = load_dataset(args.file)
    
    # Verificar si los datos se cargaron correctamente
    if df is None or df.empty:
        print(f"❌ Error: No se pudieron cargar los datos desde '{args.file}'")
        return

    pp = Preprocessing(df)

    # Comando: describe
    if args.command == "describe":
        print("\n" + "="*80)
        print(f"📊 ESTADÍSTICAS DESCRIPTIVAS - {args.file.upper()}")
        print("="*80)
        
        if args.numeric:
            print(pp.describe_numeric())
        elif args.categorical:
            print(pp.describe_categorical() if hasattr(pp, 'describe_categorical') else df.describe(include=['object', 'category']))
        else:
            # Mostrar ambas
            print("\n📈 Variables Numéricas:")
            print(pp.describe_numeric())
            print("\n📝 Variables Categóricas:")
            if hasattr(pp, 'describe_categorical'):
                print(pp.describe_categorical())
            else:
                print(df.describe(include=['object', 'category']))

    # Comando: quality
    elif args.command == "quality":
        print("\n" + "="*80)
        print(f"🔍 CALIDAD DE DATOS - {args.file.upper()}")
        print("="*80)
        quality_report = pp.data_quality()
        print(quality_report)
        
        if args.verbose and hasattr(pp, 'missing_details'):
            print("\n📋 Detalle de valores faltantes:")
            print(pp.missing_details())

    # Comando: preview
    elif args.command == "preview":
        print("\n" + "="*80)
        print(f"👁️ VISTA PREVIA - {args.file.upper()}")
        print("="*80)
        
        if args.sample:
            print(f"\n🎲 Muestra aleatoria de {args.rows} filas:")
            print(df.sample(min(args.rows, len(df))))
        else:
            print(f"\n📄 Primeras {args.rows} filas:")
            print(pp.preview_data(args.rows))

    # Comando: info (nuevo - información completa)
    elif args.command == "info":
        print("\n" + "="*80)
        print(f"ℹ️ INFORMACIÓN DEL DATASET - {args.file.upper()}")
        print("="*80)
        
        # Información básica
        print(f"\n📏 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"💾 Memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        print(f"\n📋 Columnas ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:3d}. {col}")
        
        if args.detailed:
            print("\n🔧 Tipos de datos:")
            print(df.dtypes)
            
            print("\n⚠️ Valores nulos:")
            nulls = df.isnull().sum()
            null_pct = (nulls / len(df)) * 100
            null_df = pd.DataFrame({
                'Nulos': nulls,
                'Porcentaje': null_pct
            })
            print(null_df[null_df['Nulos'] > 0] if (nulls > 0).any() else "✅ No hay valores nulos")
            
            print("\n🔄 Valores únicos por columna:")
            for col in df.columns:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count:,} únicos")

    # Comando: data (el que pediste específicamente)
    elif args.command == "data":
        print("\n" + "="*80)
        print(f"📊 RESUMEN DEL DATASET - {args.file.upper()}")
        print("="*80)
        
        # Información básica siempre visible
        print(f"\n📏 Dimensiones:")
        print(f"   • Filas:    {df.shape[0]:,}")
        print(f"   • Columnas: {df.shape[1]:,}")
        
        print(f"\n📋 Columnas:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()
            print(f"   {i:2d}. {col:20s} | Tipo: {str(dtype):12s} | Nulos: {nulls:4d} | Únicos: {unique:,}")
        
        if args.types:
            print(f"\n🔧 Tipos de datos detallados:")
            print(df.dtypes.to_string())
        
        if args.missing:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_data = missing[missing > 0]
            if len(missing_data) > 0:
                print(f"\n⚠️ Valores faltantes:")
                for col, nulos in missing_data.items():
                    print(f"   • {col}: {nulos:,} ({missing_pct[col]:.1f}%)")
            else:
                print(f"\n✅ No hay valores faltantes")
        
        if args.summary:
            print(f"\n📈 Resumen estadístico rápido:")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"\n   Variables numéricas ({len(numeric_cols)}):")
                for col in numeric_cols[:5]:  # Mostrar primeras 5 numéricas
                    print(f"   • {col}:")
                    print(f"       Mín: {df[col].min():.2f} | Máx: {df[col].max():.2f}")
                    print(f"       Media: {df[col].mean():.2f} | Mediana: {df[col].median():.2f}")
                if len(numeric_cols) > 5:
                    print(f"   ... y {len(numeric_cols)-5} columnas numéricas más")
            else:
                print("   No hay variables numéricas")
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                print(f"\n   Variables categóricas ({len(categorical_cols)}):")
                for col in categorical_cols[:3]:
                    top_value = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                    print(f"   • {col}: {df[col].nunique():,} categorías (Moda: {top_value})")
                if len(categorical_cols) > 3:
                    print(f"   ... y {len(categorical_cols)-3} columnas categóricas más")

    else:
        print(f"❌ Comando desconocido: {args.command}")

"""
# Básico - muestra dimensiones y columnas
statslibx data iris.csv

# Con resumen estadístico
statslibx data iris.csv --summary

# Con tipos de datos
statslibx data iris.csv --types

# Con valores faltantes
statslibx data iris.csv --missing

statslibx data mi_archivo.csv --summary --types --missing

# Información completa
statslibx info iris.csv

# Con detalles avanzados
statslibx info iris.csv --detailed

# Solo numéricas
statslibx describe iris.csv --numeric

# Solo categóricas
statslibx describe iris.csv --categorical

statslibx describe iris.csv

# Reporte básico
statslibx quality iris.csv

# Con verbose
statslibx quality iris.csv --verbose

# Primeras filas
statslibx preview iris.csv -n 10

# Muestra aleatoria
statslibx preview iris.csv -n 5 --sample
"""


if __name__ == "__main__":
    main()