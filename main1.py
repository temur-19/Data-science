import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


kun = np.array([
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12
])

Ishlagan_soati = np.array([
    4, 5, 6, 7, 8, 9,
    5, 10, 7, 11, 8, 6
])

xomashyo_sarfi = np.array([
    55, 60, 68, 75, 80, 88,
    65, 92, 78, 96, 85, 70
])

mahsulot_soni = np.array([
    42, 50, 58, 67, 74, 82,
    55, 90, 70, 95, 80, 62
])

df = pd.DataFrame({
    "Kun": kun,
    "Ishlagan_soati": Ishlagan_soati,
    "Xomashyo_sarfi": xomashyo_sarfi,
    "Mahsulot_soni": mahsulot_soni
})


print(df)

plt.figure(figsize=(10, 5))
plt.bar(kun, mahsulot_soni)
plt.title("Mahsulot soni test natijalari")
plt.xlabel("Kun")
plt.ylabel("Mahsulot soni")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.savefig("grafik/bar_chart.png")

plt.figure(figsize=(8, 5))
plt.hist(
mahsulot_soni,
bins=5,
edgecolor="black"
)
plt.title("Mahsulot soni taqsimoti")
plt.xlabel("Mahsulot soni")
plt.ylabel("Takrorlanish soni")
plt.grid(axis="y", alpha=0.3)
plt.savefig("grafik/histogram.png")



plt.figure(figsize=(8, 5))
plt.scatter(
Ishlagan_soati,
mahsulot_soni,
s=100,
label="Talabalar"
)
m, b = np.polyfit(Ishlagan_soati,
mahsulot_soni,
1
)
plt.plot(
Ishlagan_soati,
m * Ishlagan_soati + b,
label="Trend"
)
plt.title("Ishlagan soati va mahsulot soni: trend")
plt.xlabel("Ishlagan soati")
plt.ylabel("Mahsulot soni")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("grafik/scatter.png")




plt.figure(figsize=(6, 5))
plt.boxplot(mahsulot_soni)
plt.title("Mahsulot soni Boxplot ko'rinishi")
plt.ylabel("Mahsulot soni")
plt.grid(axis="y", alpha=0.3)
plt.savefig("grafik/boxplot.png")




plt.figure(figsize=(9, 5))
plt.boxplot(
[
Ishlagan_soati,
xomashyo_sarfi,
mahsulot_soni
],
tick_labels=[
"Ishlagan soati",
"Xomashyo sarfi",
"Mahsulot soni"
]
)
plt.title("Xususiyatlarning Boxplot orqali taqqoslanishi")
plt.grid(axis="y", alpha=0.3)
plt.savefig("grafik/boxplot_comparison.png")





raqamli_data = df[
[
"Kun",
"Ishlagan_soati",
"Mahsulot_soni"
]
]
corr = raqamli_data.corr()
print(corr)




plt.figure(figsize=(7, 6))
plt.imshow(
corr,
cmap="coolwarm",
vmin=-1,
vmax=1
)
plt.colorbar()
plt.xticks(
range(len(corr.columns)),
corr.columns,
rotation=45
)
plt.yticks(
range(len(corr.columns)),
corr.columns
)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
        j,
        i,
        f"{corr.iloc[i, j]:.2f}",
        ha="center",
        va="center"
        )
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("grafik/correlation_matrix.png")


natija = np.where(
mahsulot_soni >= 60,
"O'tdi",
"O'tmadi"
)
print(natija)


otdi = np.sum(natija == "O'tdi")
otmadi = np.sum(natija == "O'tmadi")
print("O'tdi:", otdi)
print("O'tmadi:", otmadi)


labels = [
"O'tdi",
"O'tmadi"
]
sizes = [otdi,
otmadi
]
plt.figure(figsize=(6, 6))
plt.pie(
sizes,
labels=labels,
autopct="%1.1f%%",
startangle=90
)
plt.title("Mahsulot soni natijalarining ulushi")
plt.axis("equal")
plt.savefig("grafik/pie_chart.png")



epoch = np.array([
1, 2, 3, 4, 5,
6, 7, 8, 9, 10
])
accuracy = np.array([
0.55, 0.63, 0.69, 0.74, 0.78,
0.81, 0.84, 0.86, 0.88, 0.90
])
plt.figure(figsize=(8, 5))
plt.plot(
    epoch,
    accuracy,
    marker="o"
)
plt.title("Model accuracy qiymatining epochlar bo'yicha o'zgarishi")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.savefig("grafik/accuracy_plot.png")






