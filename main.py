import numpy as np
import pandas as pd
import kagglehub


data = kagglehub.dataset_download("abcsds/pokemon")
file_path = data + '/Pokemon.csv'
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


data = load_data(file_path)


def count_columns(data):
    return len(data.columns)

def count_rows(data):
    return len(data)

def legendary_pokemon(data):
	legendary = data[data['Legendary'] == True]
	return legendary

def eng_kop_turlar(data):
	return data['Type 1'].value_counts().idxmax()

def fast_pokemon(data):
	fastest = data[data['Speed'] == data['Speed'].max()]
	return fastest

def generation1(data):
	gen1 = data[data['Generation'] == 1]
	return gen1

def average_attack_by_type(data):
	attack_types = data.groupby('Type 1')['Attack'].mean().sort_values(ascending=False)
	return attack_types

print(data.head(10))
print(f"Ustunlar soni: {count_columns(data)}")
print(f"Qatorlar soni: {count_rows(data)}")
print(f"Legendary Pokemonlar soni: {len(legendary_pokemon(data))}")
print(f"Pokemon turlari eng ko'p: {eng_kop_turlar(data)}") 
print(f"Eng tez Pokemon: {fast_pokemon(data)['Name'].values[0]}")
print(f"1-chi avlod Pokemonlar soni: {len(generation1(data))}")
print("Har bir turdagi Pokemonlarning o'rtacha hujum kuchi:",max(average_attack_by_type(data)))

