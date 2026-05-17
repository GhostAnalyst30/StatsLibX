from statslibx.viewx import HTML
from statslibx.datasets import load_iris
import plotly.express as px

data = load_iris()

page = HTML(
    data=data,
    title="Iris DashBoard",
    num_cols=2,
    num_rows=2,
    theme="dark"
)

page.add_text(
    content="""Este dashboard fue hecho con el fin
    de poner a prueba la herramienta de Viewx\n
    \t\t\t- Emmanuel Ascendra""",
    slot_grid=(1, 1, 1, 1)
)

page.add_table(
    df=data.head(10),
    title="Dataset de Iris",
    slot_grid=(1, 2, 1, 1)
)

scatter_plot = px.scatter(
    data_frame=data,
    x="sepal_length",
    y="petal_length"
)

page.add_plot(
    fig=scatter_plot,
    slot_grid=(2, 1, 1, 4)
)

page.generate(filename="Dashboard_Iris.html")