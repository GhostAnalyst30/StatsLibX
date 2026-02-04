# Paquetes
install.packages("reticulate")

# Inicializacion
library(reticulate)
py_install("pandas")
py_install("statslibx")
py_install("flet")
py_install("matplotlib")
py_install("seaborn")
py_install("plotly")


slx <- import("statslibx")

slx$welcome()

df <- slx$load_dataset("tests/bank (1).csv", sep=";")

stats <- slx$DescriptiveStats(df)

print(stats$mean())
