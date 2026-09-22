"""E-commerce ma'lumotlari uchun umumiy vizual tahlil.

Ishga tushirish:
	python main.py

Natijalar ``outputs/`` papkasiga PNG va CSV ko'rinishida yoziladi.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ecommerce_sales_customer_analytics_150k.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_data() -> pd.DataFrame:
	"""CSV faylni o'qiydi va tahlil uchun ustunlarni tayyorlaydi."""
	if not DATA_FILE.exists():
		raise FileNotFoundError(f"Dataset topilmadi: {DATA_FILE}")

	data = pd.read_csv(DATA_FILE, low_memory=False)
	data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")

	numeric_columns = [
		"net_sales",
		"profit",
		"gross_sales",
		"discount_amount",
		"quantity",
		"customer_rating",
	]
	for column in numeric_columns:
		data[column] = pd.to_numeric(data[column], errors="coerce")

	data = data.dropna(subset=["order_date", "net_sales", "profit"])
	data["month"] = data["order_date"].dt.to_period("M").dt.to_timestamp()
	return data


def save_chart(figure: plt.Figure, filename: str) -> None:
	"""Grafikni bir xil o'lcham va sifatda saqlaydi."""
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
	plt.close(figure)


def create_charts(data: pd.DataFrame) -> None:
	"""Asosiy savdo ko'rsatkichlari uchun alohida PNG grafiklar yaratadi."""
	sns.set_theme(style="whitegrid", palette="deep")

	monthly = data.groupby("month", as_index=False).agg(
		revenue=("net_sales", "sum"), profit=("profit", "sum")
	)
	figure, axis = plt.subplots(figsize=(12, 5))
	axis.plot(monthly["month"], monthly["revenue"], label="Sof savdo", linewidth=2)
	axis.plot(monthly["month"], monthly["profit"], label="Foyda", linewidth=2)
	axis.set_title("Oylik sof savdo va foyda")
	axis.set_xlabel("Oy")
	axis.set_ylabel("Miqdor")
	axis.legend()
	save_chart(figure, "01_monthly_sales_profit.png")

	channel = data.groupby("sales_channel", as_index=False).agg(
		revenue=("net_sales", "sum"), profit=("profit", "sum")
	).sort_values("revenue", ascending=False)
	figure, axes = plt.subplots(1, 2, figsize=(13, 5))
	sns.barplot(data=channel, x="revenue", y="sales_channel", ax=axes[0], color="#2878b5")
	axes[0].set_title("Savdo kanali bo'yicha tushum")
	axes[0].set_xlabel("Sof savdo")
	axes[0].set_ylabel("")
	sns.barplot(data=channel, x="profit", y="sales_channel", ax=axes[1], color="#49a078")
	axes[1].set_title("Savdo kanali bo'yicha foyda")
	axes[1].set_xlabel("Foyda")
	axes[1].set_ylabel("")
	save_chart(figure, "02_sales_channels.png")

	region = data.groupby("region", as_index=False).agg(
		revenue=("net_sales", "sum"), profit=("profit", "sum")
	).sort_values("revenue", ascending=False)
	figure, axis = plt.subplots(figsize=(10, 5))
	sns.barplot(data=region, x="revenue", y="region", hue="region", legend=False, ax=axis)
	axis.set_title("Regionlar bo'yicha tushum")
	axis.set_xlabel("Sof savdo")
	axis.set_ylabel("")
	save_chart(figure, "03_revenue_by_region.png")

	status = data["order_status"].value_counts().rename_axis("status").reset_index(name="orders")
	figure, axis = plt.subplots(figsize=(8, 6))
	axis.pie(status["orders"], labels=status["status"], autopct="%1.1f%%", startangle=90)
	axis.set_title("Buyurtmalar holati")
	save_chart(figure, "04_order_status.png")

	figure, axis = plt.subplots(figsize=(10, 6))
	sns.scatterplot(
		data=data.sample(min(len(data), 12000), random_state=42),
		x="discount_amount",
		y="profit",
		hue="customer_segment",
		alpha=0.45,
		ax=axis,
	)
	axis.set_title("Chegirma va foyda o'rtasidagi bog'liqlik")
	axis.set_xlabel("Chegirma miqdori")
	axis.set_ylabel("Foyda")
	save_chart(figure, "05_discount_vs_profit.png")

	rating = data.groupby("customer_segment", as_index=False).agg(
		average_rating=("customer_rating", "mean"),
		average_order_value=("net_sales", "mean"),
	)
	figure, axes = plt.subplots(1, 2, figsize=(12, 5))
	sns.barplot(data=rating, x="customer_segment", y="average_rating", hue="customer_segment", legend=False, ax=axes[0])
	axes[0].set_title("Segment bo'yicha o'rtacha reyting")
	axes[0].set_xlabel("")
	axes[0].set_ylabel("Reyting")
	sns.barplot(data=rating, x="customer_segment", y="average_order_value", hue="customer_segment", legend=False, ax=axes[1])
	axes[1].set_title("Segment bo'yicha o'rtacha buyurtma")
	axes[1].set_xlabel("")
	axes[1].set_ylabel("O'rtacha buyurtma qiymati")
	save_chart(figure, "06_customer_segments.png")


def create_dashboard(data: pd.DataFrame) -> None:
	"""Barcha asosiy grafiklarni bitta dashboardga jamlaydi."""
	monthly = data.groupby("month", as_index=False).agg(
		revenue=("net_sales", "sum"), profit=("profit", "sum")
	)
	channel = data.groupby("sales_channel", as_index=False)["net_sales"].sum().sort_values("net_sales")
	region = data.groupby("region", as_index=False)["net_sales"].sum().sort_values("net_sales")
	status = data["order_status"].value_counts()

	figure, axes = plt.subplots(2, 2, figsize=(16, 10))
	figure.suptitle("E-commerce savdo dashboardi", fontsize=18, fontweight="bold")
	axes[0, 0].plot(monthly["month"], monthly["revenue"], label="Sof savdo")
	axes[0, 0].plot(monthly["month"], monthly["profit"], label="Foyda")
	axes[0, 0].set_title("Oylik dinamika")
	axes[0, 0].legend()
	axes[0, 1].barh(channel["sales_channel"], channel["net_sales"], color="#2878b5")
	axes[0, 1].set_title("Kanallar bo'yicha tushum")
	axes[1, 0].barh(region["region"], region["net_sales"], color="#49a078")
	axes[1, 0].set_title("Regionlar bo'yicha tushum")
	axes[1, 1].pie(status.values, labels=status.index, autopct="%1.1f%%", startangle=90)
	axes[1, 1].set_title("Buyurtmalar holati")
	save_chart(figure, "dashboard.png")


def print_summary(data: pd.DataFrame) -> None:
	"""Konsolga tezkor KPI xulosasini chiqaradi."""
	total_revenue = data["net_sales"].sum()
	total_profit = data["profit"].sum()
	print(f"Buyurtmalar soni: {len(data):,}")
	print(f"Sof savdo: ${total_revenue:,.2f}")
	print(f"Foyda: ${total_profit:,.2f}")
	print(f"O'rtacha buyurtma: ${data['net_sales'].mean():,.2f}")
	print(f"Grafiklar saqlandi: {OUTPUT_DIR}")


def main() -> None:
	OUTPUT_DIR.mkdir(exist_ok=True)
	data = load_data()
	create_charts(data)
	create_dashboard(data)
	data.groupby("sales_channel", as_index=False).agg(
		orders=("order_id", "nunique"),
		revenue=("net_sales", "sum"),
		profit=("profit", "sum"),
	).sort_values("revenue", ascending=False).to_csv(
		OUTPUT_DIR / "sales_channel_summary.csv", index=False
	)
	print_summary(data)


if __name__ == "__main__":
	main()
