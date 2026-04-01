# mobility-tools

This library bundles code from the [hiWalk](https://gitlab.heigit.org/climate-action/plugins/walkability) and [hiBike](https://gitlab.heigit.org/climate-action/plugins/bikeability) plugins.
First and foremost as a trial it includes the detour factors.


## Description
Currently this library provides two main features:
1. A facility to compute detour factors based on an h3 hexgrid based on ors routing.
2. A settings class to store ORS settings like API-Keys


## Installation
Add this library to your poetry project by using
```shell
poetry add git+ssh://git@gitlab.heigit.org:2022/climate-action/utilities/mobility-tools.git#1.0.3
```

## Usage
### Detour Factors
To get detour factors in your project you can use the of the following function:
```python
detour_factors.get_detour_factors(aoi, paths, ors_settings, 'foot-walking')
```
This will return a geodataframe with hexcells and a detour factor ranging from 0 to `numpy.inf`
that tells you the ratio between the path length to neighbouring cells in your selected profile and the distance as the crow flies.
`numpy.inf` indicates a missing connection.
`numpy.nan` implies that there were no routable edges within the cell in the openrouteservice graph.
### Slope
To get slope for a path network you can use the flollowing function:
```python
slope.get_paths_slopes(paths, s3settings)
```
This will use elevation pmtiles in s3storage to calculate slope for the paths in `paths` as a percentage.
Note that due to the discontinous nature of raster elevation data, there might be some small error depending on the exact positon of the path in reference to the tiles and pixels of the tile.

## License
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3

