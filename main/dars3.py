import pandas as pd
import numpy as np
import kagglehub

data = 'data/6uFf5n4dAoQCJyDPfNwguUxofJRfupUGMKtiA2m0.csv'

def get_data(file_path):
    data = pd.read_csv(file_path)
    return data


def mijozlar(data):
    yangi = data[data['mijoz_turi'] == 'yangi']
    doimiy = data[data['mijoz_turi'] == 'doimiy']
    return {
        'yangi': len(yangi),
        'doimiy': len(doimiy)
    }

def analyze_data(data):
    if mijozlar(data)['doimiy'] > mijozlar(data)['yangi']:
        return "Doimiy mijozlar ko'p"
    elif mijozlar(data)['doimiy'] < mijozlar(data)['yangi']:
        return "Yangi mijozlar ko'p"
    else:
        return "Mijozlar soni teng"

def get_payments(data):
    card_payments = data[data['tolov'] == 'karta'].shape[0]
    non_card_payments = data[data['tolov'] == 'naqd'].shape[0]
    return{
        'kartadan tolovlar': card_payments,
        'naqd tolovlar': non_card_payments
    }
def card_payments(data):

    if get_payments(data)['kartadan tolovlar'] > get_payments(data)['naqd tolovlar']:
        return "Kartadan tolovlar ko'p"
    elif get_payments(data)['kartadan tolovlar'] < get_payments(data)['naqd tolovlar']:
        return "Naqd tolovlar ko'p"
    else:   
        return "Teng tolovlar" 


def baholash(data):
    result = data.groupby('yetkazish_kun')['baho'].mean()
    yaxshi_kunlar = result[result > 4.5]
    print(yaxshi_kunlar)
    return result


def new_gipoteza(data):
    result = data.groupby('mijoz_turi')['yetkazish_kun'].mean()
    if result['yangi'] < result['doimiy']:
        print("Yangi mijozlar tezroq yetkazib beriladi")
    elif result['yangi'] > result['doimiy']:
        print("Doimiy mijozlar tezroq yetkazib beriladi")
    else:
        print("Yangi va doimiy mijozlar uchun yetkazib berish vaqti teng")
    return result



data = get_data(data)


print(f"Ustunlar soni: {len(data.columns)}")
print(f"Yangi mijozlar soni: {mijozlar(data)['yangi']}")
print(f"Doimiy mijozlar soni: {mijozlar(data)['doimiy']}")
print(analyze_data(data))
print(f"Kartadan tolovlar soni: {get_payments(data)['kartadan tolovlar']}")
print(f"Naqd tolovlar soni: {get_payments(data)['naqd tolovlar']}")
print(card_payments(data))
print(baholash(data))
print(new_gipoteza(data))