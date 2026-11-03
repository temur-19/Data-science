import plotly.express as px
import pandas as pd


df = pd.read_csv("data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv")
kurtka = df[df["toifa"] == "Kurtka"]
oylik = kurtka.groupby("sana")["summa"].sum().reset_index()
fig = px.line(
    oylik,
    x="sana",
    y="summa",
    title="Kurtka savdosining oylik dinamikasi",
    markers=True
)
fig.show()
eng_kop = oylik.loc[oylik["summa"].idxmax()]
print("Eng ko'p sotilgan oy:", eng_kop["sana"])
print("Savdo summasi:", eng_kop["summa"])