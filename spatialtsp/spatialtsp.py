"""Main module."""

# For ipyleaflet
import ipyleaflet
from ipyleaflet import basemaps

# For spatialtsp
import numpy as np
import random
from scipy.spatial import distance, Voronoi
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
from shapely.geometry import Point

class Map(ipyleaflet.Map):
    """This is the map class that inherits from ipyleaflet.Map.

    Args:
        ipyleaflet (Map): The ipyleaflet.Map class.
    """
    def __init__(self, center=[20, 0], zoom=2, **kwargs):
        """Initialize the map.

        Args:
            center (list, optional): Set the center of the map. Defaults to [20, 0].
            zoom (int, optional): Set the zoom level of the map. Defaults to 2.
        """
        super().__init__(center=center, zoom=zoom, **kwargs)

    def add_tile_layer(self, url, name, **kwargs):
        """Add tile layer

        Args:
            url (str): The address of url.
            name (str): The name of the layer. 
        """
        layer = ipyleaflet.TileLayer(url=url, name=name, **kwargs)
        self.add(layer)

    def add_basemap(self, name):
        """Adds a basemap to the current map.

        Args:
            name (str or object): The name of the base map as a string or object
        """
        if isinstance(name, str):
            url = eval(f"basemaps.{name}").build_url()
            self.add_tile_layer(url, name)
        else:
            self.add(name)

    def add_layers_control(self, position = "topright"):
        """Adds a layers control to the map.

        Args:
            position (str, optional): The position of the layers control. Defaults to "topright".
        """
        self.add_control(ipyleaflet.LayersControl(position=position))

    def add_geojson(self, data, name="geojson", **kwargs):
        """Adds a GeoJSON layer to the map.

        Args:
            data (str | dict): GeoJSON data as a string or a dictionary.
            name (str, optional): The name of the layer. Defaults to "geojson"
        """

        import json

        if isinstance(data, str):
            with open(data) as f:
                data = json.load(f)

        if "style" not in kwargs:
            kwargs["style"]={"color": "blue", "weight":1, "fillOpacity":0}

        if"hover_style" not in kwargs:
            kwargs["hover_style"]={"fillcolor": "blue", "fillOpacity":0.8}

        layer = ipyleaflet.GeoJSON(data=data, name=name, **kwargs)
        self.add(layer)

    def add_shp(self, data, name='shp', **kwargs):
        """Adds a shapefile to the map 

        Args:
            data (str or dict): The path to the map
            name (str, optional): The name of the shapefile. Defaults to 'shp'.
        """
        import shapefile
        import json

        if isinstance(data, str):
            with shapefile.Reader(data) as shp:
                data = shp.__geo_interface__
        
        self.add_geojson(data, name, **kwargs)

class tsp(spatialtsp.tsp):
    def is_far_enough(new_point, existing_points, min_distance=3):
        """Check if a new point is far enough from existing points.

        Args:
            new_point (any type of number data): Make a new point
            existing_points (any type of number data): For get a enough distance from existing points
            min_distance (int, optional): The minimum distance between points. Defaults to 3.

        Returns:
            _type_: _description_
        """
        for point in existing_points:
            if np.sqrt((new_point[0] - point[0])**2 + (new_point[1] - point[1])**2) < min_distance:
                return False
        return True

    def generate_clustered_points(num_points, std_dev=5, cluster_centers=[(13, 13), (37, 37)], x_max=50, y_max=50, min_distance=3, seed=None):
        """Generate clustered points.

        Args:
            num_points (int): The number of points to generate.
            std_dev (int, optional): make std dev. Defaults to 5.
            cluster_centers (list, optional): The centers of the clusters. Defaults to [(13, 13), (37, 37)].
            x_max (int, optional): Limitation of x value. Defaults to 50.
            y_max (int, optional): Limitation of y value Defaults to 50.
            min_distance (int, optional): The minimum distance between points. Defaults to 3.
            seed (_type_, optional): Defaults to None.

        Returns:
            _type_: _description_
        """
        np.random.seed(seed)
        all_points = set()

        points_per_cluster = num_points // len(cluster_centers)
        extra_points = num_points % len(cluster_centers)  #    
        
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