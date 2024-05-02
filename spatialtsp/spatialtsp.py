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
from shapely.geometry import Point, box, Polygon
import re
import os


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
        """Adds GUI to map. Includes basemap options under 'basemap selector'

        Args:
            position (str, optional): Position of GUI. Defaults to "topright".
        """

        basemap_selector = widgets.Dropdown(
            options=[
                "OpenStreetMap",
                "OpenTopoMap",
                "Esri.WorldImagery",
                "Esri.NatGeoWorldMap",
                "CartoDB.DarkMatter"
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

    def add_toolbar(self, position="topright"): #add toolbar functionality, basemap gui button, how keep toolbar from disappearing, remove basemap widget
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


        #click signal to backend/frontend
        def on_click(change):
            if change["new"]:
                toolbar.children = [widgets.HBox([close_button, toolbar_button]), grid]
            else:
                toolbar.children = [toolbar_button]

        toolbar_button.observe(on_click, "value")
        toolbar_ctrl = WidgetControl(widget=toolbar, position="topright")
        self.add(toolbar_ctrl)

        #output widget confirming button click
        output = widgets.Output()
        output_control = WidgetControl(widget=output, position="bottomright")
        self.add(output_control)





        def toolbar_callback(change): #links to actions to buttons,
            if change.icon == "folder-open":
                with output:
                    output.clear_output()
                    print(f"You can open a file")
            elif change.icon == "map":
                self.add_basemap_gui() #call basemap selector
                with output:           #how to clear?
                    # close_button.on_click(close_click)
                    output.clear_output()
                    print("change the basemap")
            elif change.icon == "info":
                with output:
                    output.clear_output()
                    print("There is no info here.")
            elif change.icon == "question":
                with output:
                    output.clear_output()
                    print("There is no help here.")
            else:
                with output:
                    output.clear_output()
                    print(f"Icon: {change.icon}")

        for tool in grid.children:
            tool.on_click(toolbar_callback)


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

def generate_random_points(num_points, num_points_per_cell=1, x_max=50, y_max=50, min_distance=1, seed=0):
    np.random.seed() 
    n=num_points
    points_list = []

    # Calculate the total number of cells (n by n grid)
    num_cells = n ** 2
    cell_width = x_max / n
    cell_height = y_max / n
    points_list = []

    for i in range(n):
        for j in range(n):
            cell_points = set()
            while len(cell_points) < num_points_per_cell:
                x_min, x_max = i * cell_width, (i + 1) * cell_width
                y_min, y_max = j * cell_height, (j + 1) * cell_height
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                new_point = (x, y)

                if is_far_enough(new_point, cell_points, min_distance):
                    cell_points.add(new_point)

            points_list.extend(cell_points)
    
    gdf_points = gpd.GeoDataFrame({'geometry': [Point(p) for p in points_list]}, crs="EPSG:4326")
    return gdf_points


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

## Voronoi Adjacency Distance
def voronoi_adjacency_distance(gdf_points, clip_box=box(0, 0, 50, 50)):
    points = np.array([[point.x, point.y] for point in gdf_points.geometry])
    points2=points
    points2 = np.append(points2, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
    vor = Voronoi(points2)
    point_to_region = vor.point_region[:len(points)]
    polygons = []
    ids = []
    for point_idx, region_idx in enumerate(point_to_region):
        region = vor.regions[region_idx]
        if -1 in region:
            continue
        polygon = Polygon([vor.vertices[i] for i in region])
        clipped_polygon = polygon.intersection(clip_box)
        if not clipped_polygon.is_empty:
            polygons.append(clipped_polygon)
            ids.append(point_idx)  # point index
    
    # Generate a GeoDataFrame from the Voronoi polygons
    voronoi_gdf = gpd.GeoDataFrame({'id': ids, 'geometry': polygons}, crs="EPSG:4326")

    # Investigate the adjacency of the Voronoi polygons
    num_points = len(points)
    distances = np.full((num_points, num_points), 99999)
    for i in range(num_points):
        distances[i, i] = 0
        for j in range(num_points):
            if i != j:
                # find the Voronoi polygon of the two points
                if voronoi_gdf.geometry[i].touches(voronoi_gdf.geometry[j]):
                    # if the two polygons are adjacent, calculate the distance between the two points
                    distance = gdf_points.geometry[i].distance(gdf_points.geometry[j])*100
                    distances[i, j] = round(distance)    
    return distances


## KNN Adjacency Distance
def knn_adjacency_distance(gdf_points, k):
    # Extract point coordinates from the GeoDataFrame
    points = np.array([[point.x, point.y] for point in gdf_points.geometry])

    # Create and fit the k-nearest neighbors model
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)

    # Find the k-nearest neighbors for each point
    distances, indices = neigh.kneighbors(points)

    # initialize the distance matrix(99999; if not adjacent)
    distance_matrix = np.full((len(points), len(points)), 99999)
        # Fill in the actual distances for k-nearest neighbors
    for i in range(len(points)):
        for j in indices[i]:
            if i != j:  # Exclude self
                actual_distance = np.linalg.norm(points[i] - points[j])*100
                distance_matrix[i][j] = round(actual_distance)  # Round to 0 decimal places

    # 0 for the cost of moving from a city to itself
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

## Combine Distance Matrices
def combine_distance_matrices(knn_distances, voronoi_distances):
    # Generate a matrix to store the final distances
    final_distances = np.full(knn_distances.shape, 99999) # initialize the distance matrix(99999; if not adjacent)

    # Combine the two distance matrices
    for i in range(final_distances.shape[0]):
        for j in range(final_distances.shape[1]):
            # If both distances are not 99999, take the minimum
            if knn_distances[i][j] != 99999 and voronoi_distances[i][j] != 99999.00:
                final_distances[i][j] = min(knn_distances[i][j], voronoi_distances[i][j])
            elif knn_distances[i][j] != 99999:
                final_distances[i][j] = knn_distances[i][j]
            elif voronoi_distances[i][j] != 99999.00:
                final_distances[i][j] = voronoi_distances[i][j]
    return final_distances

## Generate LP Model
def generate_lp_model(distance_matrix):
    n = len(distance_matrix)  # The number of points
   
    # 1. Generate the objective function
    objective_function = "Minimize\nobj: "
    variables_str = " + ".join(f"{distance_matrix[i][j]} X_{i+1}_{j+1}"
                               for i in range(n) for j in range(n) if i != j)
    objective_function += variables_str
   
    # 2. Generate the subject to constraints
    subject_to = "\n\nSubject To\n"
    for i in range(1, n + 1):
        subject_to += f"Con_{i}: " + " + ".join(f"X_{i}_{j}" for j in range(1, n + 1) if j != i) + " = 1\n"
   
    for j in range(1, n + 1):
        subject_to += f"Con_{n + j}: " + " + ".join(f"X_{i}_{j}" for i in range(1, n + 1) if i != j) + " = 1\n"
   
    # 3. MTZ constraints(prevent subtours)
    mtz_constraints = "\n"
    for i in range(2, n + 1):
        for j in range(2, n + 1):
            if i != j:
                mtz_constraints += f"MTZ_{i}_{j}: U_{i} - U_{j} + {n} X_{i}_{j} <= {n - 1}\n"
   
    # 4. Bounds
    bounds = "\nBounds\n"
    for i in range(2, n + 1):
        bounds += f"1 <= U_{i} <= {n-1}\n"
   
    # 5. Binaries
    binaries = "\nBinaries\n"
    for i in range(1, n + 1):
        binaries += " ".join(f"X_{i}_{j}" for j in range(1, n + 1) if i != j) + "\n"
   
    generals = "\nGenerals\n" + " ".join(f"U_{i}" for i in range(2, n + 1))
   
    # 6. Combine all the parts
    lp_model = objective_function + subject_to + mtz_constraints + bounds + binaries + generals + "\n\nEnd"
    print(f"\* Generated LP model for {n} points.*/")
    return lp_model

## Write LP File
def writeLpFile_func(k, distance_matrix, i, path, num_points):
    # 'k' = number of nearest neighbors in KNN
    # 'distance_matrix' = distance matrix
    # 'i'= iteration number
    # 'workdir' = directory to save the LP files
    
    # Generate the LP model
    lp_model = generate_lp_model(distance_matrix)
    
    # Save the LP model to a file
    file_path = f"{path}/final_work/03_LPFiles/TSP_num{num_points}_k{k}.lp"

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the LP model to a file
    with open(file_path, 'w') as file:
        file.write(lp_model)
    print(f"LP file saved to {file_path}")

## 6. Get attribute from cplex optimization results
def get_attributes_cplex(result):
    """
    Extracts various attributes from the result of a CPLEX optimization.

    Args:
        result (str): The result string from a CPLEX optimization.

    Returns:
        timenb (str or None): The solution time in seconds, if found in the result string.
        iternb (str or None): The number of iterations, if found in the result string.
        nodenb (str or None): The number of nodes, if found in the result string.
        objval (float): The objective value, if found in the result string. Defaults to 0.0 if not found.
        dettime (str or None): The deterministic time in ticks, if found in the result string.

    Example:
        result = "Solution time = 10 sec. Iterations = 5 Nodes = 3 Objective = 100.0 Deterministic time = 20 ticks"
        timenb, iternb, nodenb, objval, dettime = get_attributes_cplex(result)
    """
    # Initialize default values
    timenb = iternb = nodenb = objval = dettime = None

    # Use regular expressions to find matches
    time_match = re.search(r'Solution time =\s+([\d\.]+) sec.', result)
    iter_match = re.search(r'Iterations = (\d+)', result)
    node_match = re.search(r'Nodes = (\d+)', result)
    objval_match = re.search(r'Objective =\s+([\d\.e\+\-]+)', result)
    dettime_match = re.search(r'Deterministic time =\s+([\d\.]+) ticks', result)

    # Extract values if matches are found
    if time_match:
        timenb = time_match.group(1)
    if iter_match:
        iternb = iter_match.group(1)
    if node_match:
        nodenb = node_match.group(1)
    if objval_match:
        objval = objval_match.group(1)
    if dettime_match:
        dettime = dettime_match.group(1)
    objval = objval if objval is not None else 0.0 
    return timenb, iternb, nodenb, float(objval), dettime