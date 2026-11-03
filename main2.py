import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv")
df["sana"] = pd.to_datetime(df["sana"])
df["oy"] = df["sana"].dt.month

natija = df.groupby(["oy", "tolov"])["summa"].sum().unstack(fill_value=0)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=natija.index,
    y=natija["naqd"],
    name="Naqd"
))

fig.add_trace(go.Bar(
    x=natija.index,
    y=natija["karta"],
    name="Karta"
))

fig.update_layout(
    title="Oylar bo'yicha naqd va karta to'lovlari",
    xaxis_title="Oy",
    yaxis_title="Jami summa",
    barmode="group"
)

fig.show()

print(natija[natija["naqd"] > natija["karta"]])