a = input("Kodni kiriting: ")
sum1 = 0
sum2 = 0
for i in range(len(a)):
    if i % 2 == 1:
        sum1 += int(a[i])
    else:
        sum2 += int(a[i])
sum1 *= 3
total_sum = sum1 + sum2-int(a[-1])
print(f"Summa: {total_sum}")
if 10-total_sum % 10 == int(a[-1]):
    print("Kiritilgan kod to'g'ri.")
else:
    print("Kiritilgan kod noto'g'ri.")