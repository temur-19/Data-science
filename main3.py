import plotly.express as px
import pandas as pd

df = pd.read_csv("data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv")
shim = df[df["toifa"] == "Shim"].copy()
shim["sana"] = pd.to_datetime(shim["sana"])
shim["oy"] = shim["sana"].dt.month
natija = shim.groupby("oy", as_index=False)["summa"].sum()
fig = px.bar(
    natija,
    x="oy",
    y="summa",
    title="Shim savdosining oyma-oy o'zgarishi",
    labels={
        "oy": "Oy",
        "summa": "Jami savdo"
    },
    text_auto=True
)
fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=[
            "Yanvar", "Fevral", "Mart", "Aprel",
            "May", "Iyun", "Iyul", "Avgust",
            "Sentabr", "Oktabr", "Noyabr", "Dekabr"
        ]
    )
)

fig.show()