# ==============================================================
# KNN (K TA ENG YAQIN QO'SHNI) TOZA PYTHONDA
# Pifagor (Evklid) masofasi va ovoz berish usuli
# ==============================================================
import math
from collections import Counter

# Mavjud mevalar bazasi
# Har bir element: [Vazni(gramm), Qizilligi(1-10), Meva_Nomi]
fruits_dataset = [
    [150, 8, "Olma"],
    [170, 9, "Olma"],
    [160, 7, "Olma"],
    [130, 2, "Apelsin"],
    [140, 3, "Apelsin"],
    [125, 1, "Apelsin"]
]


# 1. Pifagor masofasini hisoblaydigan funksiya: d = sqrt((x2-x1)^2 + (y2-y1)^2)
def calculate_distance(point1, point2):
    squared_sum = 0
    for i in range(len(point1)):
        squared_sum += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_sum)


# 2. KNN asosiy algoritmi
def knn_classify(dataset, new_point, k=3):
    distances = []

    # Yangi nuqtadan barcha eski nuqtalargacha masofalarni o'lchaymiz
    for item in dataset:
        features = item[:-1]
        label = item[-1]
        dist = calculate_distance(features, new_point)
        distances.append((dist, label))

    # Masofa bo'yicha kichigidan kattasiga saralaymiz
    distances.sort(key=lambda x: x[0])

    # Eng yaqin K ta qo'shnini olamiz
    nearest_k = distances[:k]
    print(f"\nEng yaqin {k} ta qo'shni:")
    for dist, label in nearest_k:
        print(f"-> Meva: {label:7} | Masofa: {dist:.2f}")

    # Ovoz berish: eng ko'p uchragan mevani topamiz
    labels = [label for _, label in nearest_k]
    vote_result = Counter(labels).most_common(1)[0][0]
    return vote_result


# Sinov: Yangi noma'lum meva keldi (Vazni: 145 gr, Qizilligi: 7 ball)
unknown_fruit = [145, 7]
k_neighbors = 3

predicted_fruit = knn_classify(fruits_dataset, unknown_fruit, k=k_neighbors)
print(f"\nYANGI MEVA TASHXISI: {predicted_fruit} 🍎🍊")