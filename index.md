# Welcome to spatialtsp


[![image](https://img.shields.io/pypi/v/spatialtsp.svg)](https://pypi.python.org/pypi/spatialtsp)


**A python package demo for spatially informed TSP**


-   Free software: MIT License
-   Documentation: <https://whkim15.github.io/spatialtsp>


## Background
### Traveling Salesman Problem(TSP)
-   The Traveling Salesman Problem(TSP) is the problem of finding a minimum cost complete tour of a set of cities without sub-tour. 

![TSP: Newsweek, July 26, 1954](https://www.math.uwaterloo.ca/tsp/usa50/img/newsweek_medium.jpg)


### Challenge in the Traveling Salesman Problem (TSP)
-   TSP is a cornerstone challenge in Location Science, known as NP-Hard. 
-   It means that 'Finding an optimized route is theoretically possible but computationally intensive and impractical for large datasets.' 
-   Traditional approaches largely rely on heuristic methods to provide feasible solutions within a reasonable time frame. 
-   However, the ‘heuristic approach’ prioritizes computational efficiency over finding the exact solution (Genetic Algorithms, Simulated Annealing).

## Aim of this package
-   Aims to increase computing performance while ensuring optimal solutions
-   Application of the spatial partitioning informed approach. Applies Spatial Adjacency (Voronoi polygons) & Proximity Searching Methods (K-NN) to define connections among nodes


## Features
-   Generate distance matrix based on the spatial information
-   Analyze Traveling Salesman Problem

## Demos
-   Successfully worked in simulating toy data consisting of from 10 to 50 points(100 times of each toy data)
-   Successfully worked in the real data(48 Capitals of USA)