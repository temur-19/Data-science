import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataframe = 'data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv'


df = pd.read_csv(dataframe)

df["sana"] = pd.to_datetime(df["sana"])

df["oy"] = df["sana"].dt.to_period("M")

oylik_savdo = df.groupby("oy")["summa"].sum()

plt.plot(oylik_savdo.index.astype(str), oylik_savdo.values, marker="o")
plt.title("Oylik savdo o'zgarishi")
plt.xlabel("Oy")
plt.ylabel("Jami savdo")
plt.savefig("shop/grafik.png")
print("Grafik saqlandi!")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="toifa", y="summa")
plt.title("Toifalar bo'yicha summa taqsimoti")
plt.xlabel("Toifa")
plt.ylabel("Summa")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('shop/result.png')
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="toifa", y="summa", hue="tolov")
plt.title("Toifa va to'lov turi bo'yicha summa taqsimoti")
plt.xlabel("Toifa")
plt.ylabel("Summa")
plt.savefig('shop/result2.png')
plt.show()





