"""Main module."""

# For ipyleaflet
import ipyleaflet
from ipyleaflet import basemaps

# For spatial_tsp
import numpy as np
import random
from scipy.spatial import distance, Voronoi
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
from shapely.geometry import Point

class Map(ipyleaflet.Map):
    
    def __init__(self, center=[20, 0], zoom=2, **kwargs):
        super().__init__(center=center, zoom=zoom, **kwargs)

    def add_tile_layer(self, url, name, **kwargs):
        layer = ipyleaflet.TileLayer(url=url, name=name, **kwargs)
        self.add(layer)

    def add_basemap(self, name):
        if isinstance(name, str):
            url = eval(f"basemaps.{name}").build_url()
            self.add_tile_layer(url, name)
        else:
            self.add(name)

    def add_layers_control(self, position = 'topright'):
        self.add_control(ipyleaflet.LayersControl(position=position))


class Spatial_TSP():
    def is_far_enough(new_point, existing_points, min_distance=3):
        for point in existing_points:
            if np.sqrt((new_point[0] - point[0])**2 + (new_point[1] - point[1])**2) < min_distance:
                return False
        return True
        
    def generate_clustered_points(num_points, std_dev=5, cluster_centers=[(13, 13), (37, 37)], x_max=50, y_max=50, min_distance=3, seed=None):
        np.random.seed(seed)
        all_points = set()

        points_per_cluster = num_points // len(cluster_centers)
        extra_points = num_points % len(cluster_centers)     
        
        for index, center in enumerate(cluster_centers):
            points = set()
            extra = 1 if index < extra_points else 0
            while len(points) < points_per_cluster + extra:
                x = np.random.normal(center[0], std_dev)
                y = np.random.normal(center[1], std_dev)
                x, y = int(round(x)), int(round(y))
                if x > 1 and y > 1 and x <= x_max and y <= y_max and is_far_enough((x, y), all_points, min_distance):
                    points.add((x, y))
            all_points.update(points)
        
        points_list = list(all_points)
        gdf_points = gpd.GeoDataFrame({'geometry': [Point(p) for p in points_list]}, crs="EPSG:4326")
        
        return gdf_points


    def calculate_distance_matrix(gdf_points):
        points = np.array([[point.x, point.y] for point in gdf_points.geometry])
        num_points = len(points)
        distance_matrix = np.zeros((num_points, num_points), dtype=int)
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)*100
                    distance_matrix[i, j] = round(distance)
                else:
                    distance_matrix[i, j] = 0
        return distance_matrix