import pandas as pd
import numpy as np
import geopandas as gpd
import folium


data = pd.read_csv('data/kuSazUvM7khSVz7bsLxLUmeNdUf6zWYm4NCxYECn.csv')

def geodata(df):
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return gdf


def make_map(df):
    m = folium.Map(
        location=[41.0, 66.0],
        zoom_start=6
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs="EPSG:4326"
    )

    for _, row in gdf.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['name'],
            tooltip=row['name'],
            icon=folium.Icon(color='red')
        ).add_to(m)

    os.makedirs('data', exist_ok=True)

    m.save('data/map.html')
    print('Map saved')
print(data)
print(data.shape)
print(geodata(data))



