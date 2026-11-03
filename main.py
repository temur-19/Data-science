import plotly.express as px
import pandas as pd

data = pd.read_csv('data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv')
poyabzal = data[data['toifa'] == 'Poyabzal']
eng_katta = poyabzal.loc[poyabzal["summa"].idxmax()]
grafik = px.bar(
    poyabzal,
    x="sana",
    y="summa",
    color="tolov",
    title="Poyabzal buyurtmalari"
)
grafik.show()
print(eng_katta[["sana", "tolov", "summa"]])
print(poyabzal.head())
