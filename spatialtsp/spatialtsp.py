"""Main module."""

# For ipyleaflet
import requests
import ipyleaflet
from ipyleaflet import basemaps, GeoJSON, ImageOverlay, WidgetControl
from ipywidgets import widgets

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

    def add_basemap_gui(self, basemaps=None, position="topright"):
        """
        Adds a basemap GUI to the map. The GUI includes a dropdown list for selecting the basemap and a toggle button for showing and hiding the dropdown.

        The dropdown list includes options for different basemaps, such as "OpenStreetMap", "OpenTopoMap", "Esri.WorldImagery", and "Esri.NatGeoWorldMap". When a different option is selected in the dropdown, the basemap of the map is updated accordingly.

        The toggle button, represented by a 'times' icon when the dropdown is visible and a 'plus' icon when the dropdown is hidden, allows the user to show and hide the dropdown list. When the button is clicked, the visibility of the dropdown list is toggled.

        Args:
            basemaps (list, optional): A list of basemaps to include in the dropdown. If not provided, a default list of basemaps is used.
            position (str, optional): The position of the basemap GUI on the map. Defaults to "topright".
        """
        if self.basemap_gui_control is not None:  # Check if the basemap GUI is already displayed
            return  # If it is, do nothing and return
        basemap_selector = widgets.Dropdown(
            options=[
                "OpenStreetMap",
                "OpenTopoMap",
                "Esri.WorldImagery",
                "Esri.NatGeoWorldMap",
                "USGS Hydrography",
            ],
            description="Basemap",
        )

        toggle_button = widgets.Button(
            description="",
            button_style="primary",
            tooltip="Toggle dropdown",
            icon="times",
        )
        toggle_button.layout.width = "35px"

        def toggle_dropdown(b):
            if basemap_selector.layout.display == "none":
                basemap_selector.layout.display = ""
                toggle_button.icon = "times"
            else:
                basemap_selector.layout.display = "none"
                toggle_button.icon = "plus"
        toggle_button.on_click(toggle_dropdown)

        def update_basemap(change):
            self.add_basemap(change["new"])
        basemap_selector.observe(update_basemap, "value")


        # Create a box to hold the dropdown and the button
        box = widgets.HBox([basemap_selector, toggle_button])

        self.basemap_gui_control = WidgetControl(widget=box, position=position)
        self.add_control(self.basemap_gui_control)
     

    def add_layers_control(self, position = "topright"):
        """Adds a layers control to the map.

        Args:
            position (str, optional): The position of the layers control. Defaults to "topright".
        """
        self.add_control(ipyleaflet.LayersControl(position=position))

    def add_geojson(self, data, name="geojson", **kwargs):
        """Adds a GeoJSON layer to the map.

        Args:
            data (str | dict): GeoJSON data as a string, a dictionary, or a URL.
            name (str, optional): The name of the layer. Defaults to "geojson".
        """
        import requests
        import json

        if isinstance(data, str):
            if data.startswith('http://') or data.startswith('https://'):
                # data is a URL
                response = requests.get(data)
                data = response.json()
            else:
                # data is a file path
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
    
    def add_vector(self, data, name="VectorLayer", **kwargs):
        """Adds a vector layer to the map from any GeoPandas-supported vector data format.

        Args:
            data (str, dict, or geopandas.GeoDataFrame): The vector data. It can be a path to a file (GeoJSON, shapefile), a GeoJSON dict, or a GeoDataFrame.
            name (str, optional): The name of the layer. Defaults to "VectorLayer".
        """
        import geopandas as gpd
        import json

        # Check the data type
        if isinstance(data, gpd.GeoDataFrame):
            geojson_data = json.loads(data.to_json())
        # if data is a string or a dictionary
        elif isinstance(data, (str, dict)):
            # if data is a string
            if isinstance(data, str):
                data = gpd.read_file(data)
                geojson_data = json.loads(data.to_json())
            else:  # if data is a dictionary
                geojson_data = data
        else:
            raise ValueError("Unsupported data format")

        # Add the GeoJSON data to the map
        self.add_geojson(geojson_data, name, **kwargs)

    def add_basemap_gui(self, basemaps=None, position="topright"):
        """Adds a basemap GUI to the map.

        Args:
            position (str, optional): The position of the basemap GUI. Defaults to "topright".
        """

        basemap_selector = widgets.Dropdown(
            options=[
                "OpenStreetMap",
                "OpenTopoMap",
                "Esri.WorldImagery",
                "Esri.NatGeoWorldMap",
            ],
            description="Basemap",
        )

        def update_basemap(change):
            self.add_basemap(change["new"])

        basemap_selector.observe(update_basemap, "value")

        control = ipyleaflet.WidgetControl(widget=basemap_selector, position=position)
        self.add(control)


    def add_image(self, url, bounds, name="image", **kwargs):
        """Adds an image overlay to the map.

        Args:
            url (str): The URL of the image.
            bounds (list): The bounds of the image.
            name (str, optional): The name of the layer. Defaults to "image".
        """
        layer = ipyleaflet.ImageOverlay(url=url, bounds=bounds, name=name, **kwargs)
        self.add(layer)

    def add_opacity_slider(
        self, layer_index=-1, description="Opacity", position="topright"
    ):
        """Adds an opacity slider to the map.

        Args:
            layer (object): The layer to which the opacity slider is added.
            description (str, optional): The description of the opacity slider. Defaults to "Opacity".
            position (str, optional): The position of the opacity slider. Defaults to "topright".
        """
        layer = self.layers[layer_index]
        opacity_slider = widgets.FloatSlider(
            description=description,
            min=0,
            max=1,
            value=layer.opacity,
            style={"description_width": "initial"},
        )

        def update_opacity(change):
            layer.opacity = change["new"]

        opacity_slider.observe(update_opacity, "value")

        control = ipyleaflet.WidgetControl(widget=opacity_slider, position=position)
        self.add(control)


    def add_raster(self, data, name="raster", zoom_to_layer=True, **kwargs):
        """Adds a raster layer to the map.

        Args:
            data (str): The path to the raster file.
            name (str, optional): The name of the layer. Defaults to "raster".
        """

        try:
            from localtileserver import TileClient, get_leaflet_tile_layer
        except ImportError:
            raise ImportError("Please install the localtileserver package.")

        client = TileClient(data)
        layer = get_leaflet_tile_layer(client, name=name, **kwargs)
        self.add(layer)

        if zoom_to_layer:
            self.center = client.center()
            self.zoom = client.default_zoom

    def add_widget(self, widget, position="topright"):
        """Adds a widget to the map.

        Args:
            widget (object): The widget to be added.
            position (str, optional): The position of the widget. Defaults to "topright".
        """
        control = ipyleaflet.WidgetControl(widget=widget, position=position)
        self.add(control)

    def add_zoom_slider(
        self, description="Zoom level", min=0, max=24, value=10, position="topright"
    ):
        """Adds a zoom slider to the map.

        Args:
            position (str, optional): The position of the zoom slider. Defaults to "topright".
        """
        zoom_slider = widgets.IntSlider(
            description=description, min=min, max=max, value=value
        )

        control = ipyleaflet.WidgetControl(widget=zoom_slider, position=position)
        self.add(control)
        widgets.jslink((zoom_slider, "value"), (self, "zoom"))

    def add_toolbar(self, position="topright"):
        """Adds a toolbar to the map.

        Args:
            position (str, optional): The position of the toolbar. Defaults to "topright".
        """

        padding = "0px 0px 0px 5px"  # upper, right, bottom, left

        toolbar_button = widgets.ToggleButton(
            value=False,
            tooltip="Toolbar",
            icon="wrench",
            layout=widgets.Layout(width="28px", height="28px", padding=padding),
        )

        close_button = widgets.ToggleButton(
            value=False,
            tooltip="Close the tool",
            icon="times",
            button_style="primary",
            layout=widgets.Layout(height="28px", width="28px", padding=padding),
        )

        toolbar = widgets.VBox([toolbar_button])

        def close_click(change):
            if change["new"]:
                toolbar_button.close()
                close_button.close()
                toolbar.close()

        close_button.observe(close_click, "value")

        rows = 2
        cols = 2
        grid = widgets.GridspecLayout(
            rows, cols, grid_gap="0px", layout=widgets.Layout(width="65px")
        )

        icons = ["folder-open", "map", "info", "question"]

        for i in range(rows):
            for j in range(cols):
                grid[i, j] = widgets.Button(
                    description="",
                    button_style="primary",
                    icon=icons[i * rows + j],
                    layout=widgets.Layout(width="28px", padding="0px"),
                )

        def toolbar_click(change):
            if change["new"]:
                toolbar.children = [widgets.HBox([close_button, toolbar_button]), grid]
            else:
                toolbar.children = [toolbar_button]

        # Add a new button to the toolbar for the basemap GUI

        basemap_gui_button = widgets.Button(
            description="",
            button_style="primary",
            tooltip='Toggle',  # Set tooltip to a shorter string
            icon="globe",  # Use a different icon for the basemap GUI button
            layout=widgets.Layout(width="28px", padding="0px"),
        )

        basemap_gui_button.description = "off"
        grid[0, 0] = basemap_gui_button  # Replace this with the desired position

        toolbar_button.observe(toolbar_click, "value")
        toolbar_ctrl = WidgetControl(widget=toolbar, position="topright")
        self.add(toolbar_ctrl)        


        output = widgets.Output()
        output_control = WidgetControl(widget=output, position="bottomright")
        self.add(output_control)

        def toolbar_callback(change):
            with output:
                output.clear_output()
                if change.icon == "folder-open":
                    print(f"You can open a file")
                elif change.icon == "map":
                    print(f"You can add a layer")
                elif change.icon == "globe":
                    if basemap_gui_button.description == "off" and self.basemap_gui_control is None:  # Check if the basemap GUI is not displayed and not already added
                        self.add_basemap_gui()  # Call the add_basemap_gui function
                        basemap_gui_button.description = "on"  # Update the state of the button
                        print(f"Basemap GUI added")
                    else:  # If the basemap GUI is displayed
                        self.remove(self.basemap_gui_control)  # Remove the basemap GUI
                        self.basemap_gui_control = None  # Reset the basemap GUI control
                        basemap_gui_button.description = "off"  # Update the state of the button
                        print(f"Basemap GUI removed")

                else:
                    with output:
                        output.clear_output()
                    print(f"Icon: {change.icon}")

        for tool in grid.children:
            tool.on_click(toolbar_callback)



    def add_wms_layer(
        self,
        url,
        layers,
        name=None,
        attribution="",
        format="image/png",
        transparent=True,
        opacity=1.0,
        shown=True,
        **kwargs,
    ):
        """Add a WMS layer to the map.

        Args:
            url (str): The URL of the WMS web service.
            layers (str): Comma-separated list of WMS layers to show.
            name (str, optional): The layer name to use on the layer control. Defaults to None.
            attribution (str, optional): The attribution of the data layer. Defaults to ''.
            format (str, optional): WMS image format (use ‘image/png’ for layers with transparency). Defaults to 'image/png'.
            transparent (bool, optional): If True, the WMS service will return images with transparency. Defaults to True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            shown (bool, optional): A flag indicating whether the layer should be on by default. Defaults to True.
        """

        if name is None:
            name = str(layers)

        try:
            wms_layer = ipyleaflet.WMSLayer(
                url=url,
                layers=layers,
                name=name,
                attribution=attribution,
                format=format,
                transparent=transparent,
                opacity=opacity,
                visible=shown,
                **kwargs,
            )
            self.add(wms_layer)

        except Exception as e:
            print("Failed to add the specified WMS TileLayer.")
            raise Exception(e)

## TSP Functions

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


## Calculate Standard distance
def calculate_distance_matrix(gdf_points):
    """Calculate the distance matrix between points.

    Args:
        gdf_points (GeoDataFrame): A GeoDataFrame containing points with x and y coordinates.

    Returns:
        np.array: A 2D numpy array representing the distance matrix. The distance is calculated as the Euclidean distance between points, multiplied by 100 and rounded to analyze with integer.
    """
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