"""Main module."""

# For ipyleaflet
import ipyleaflet
from ipyleaflet import basemaps

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