import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataframe = 'data/Oq3OsQ0HJQPqkO75oqFvgyqLAZadw5ymjsAmYagZ.csv'


df = pd.read_csv(dataframe)


print(pd.DataFrame(df).head())

import pandas as pd
import matplotlib.pyplot as plt

df["sana"] = pd.to_datetime(df["sana"])
df["oy"] = df["sana"].dt.to_period("M")

oylik_savdo = df.groupby(["oy", "toifa"])["summa"].sum().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 7))

for toifa in oylik_savdo.columns:
    y = oylik_savdo[toifa].values
    x = range(len(y))

    if toifa == "Kurtka":
        ax.plot(x, y, color="royalblue", marker="o", linewidth=3.5)
    else:
        ax.plot(x, y, color="gray", marker="o", linewidth=1.5, alpha=0.6)

    ax.annotate(
        toifa,
        (len(y) - 1, y[-1]),
        xytext=(10, 0),
        textcoords="offset points",
        color="royalblue" if toifa == "Kurtka" else "gray",
        fontsize=11,
        fontweight="bold" if toifa == "Kurtka" else "normal"
    )

if "Kurtka" in oylik_savdo.columns:
    y = oylik_savdo["Kurtka"].values
    x = len(y) - 1

ax.set_xticks(range(len(oylik_savdo.index)))
ax.set_xticklabels(oylik_savdo.index.astype(str), rotation=45)

fig.suptitle("Toifalar bo‘yicha oylik savdo", fontsize=17, fontweight="bold")
fig.text(
    0.5, 0.91,
    "Xulosa: Kurtka savdosi o‘sib borib, dekabr oyida eng yuqori natijaga erishdi.",
    ha="center",
    fontsize=10
)

ax.set_xlabel("Oy")
ax.set_ylabel("Jami savdo (summa)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig("shop/grafik.png", dpi=300, bbox_inches="tight")
plt.show()




