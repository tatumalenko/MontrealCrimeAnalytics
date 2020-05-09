from time import time
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gp
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from shapely.geometry import Polygon, Point

plt.rcParams.update({'figure.dpi': 350})


class GridVertex:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class GridCell:
    def __init__(self, polygon: Polygon, is_block: bool):
        self.polygon = polygon
        self.is_block = is_block
        bounds = polygon.bounds
        x_min = bounds[0]
        y_min = bounds[1]
        x_max = bounds[2]
        y_max = bounds[3]
        self.vertices = dict(ul=(x_min, y_max), ur=(x_max, y_max), ll=(x_min, y_min), lr=(x_max, y_min))

        self.up = None


class Grid:
    def __init__(self, xs: List[Tuple[float, float]], ys: List[Tuple[float, float]]):
        self._xs = xs
        self._ys = ys
        self._n_dims: Tuple[int, int] = (len(xs), len(ys))
        self._grid: np.ndarray = np.array([(x, y) for y in ys for x in xs])

    # def closest_position(self, position: Tuple[float, float]) -> Tuple[float, float]:


class CoordIntensityPair:
    def __init__(self, coord: Tuple[float, float]):
        self._coord: Tuple[float, float] = coord
        self._intensity: int = 0
        self._is_block = False

    @property
    def coord(self):
        return self._coord

    @property
    def intensity(self) -> int:
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value


class PolygonIntensityPair:
    def __init__(self, polygon: Polygon, order: int):
        self._polygon: Polygon = polygon
        self._intensity: int = 0
        self._order: int = order
        self._is_block: bool = False

    @staticmethod
    def make(polygon: Polygon, intensity: int, order: int):
        this = PolygonIntensityPair(polygon, order)
        this.intensity = intensity
        return this

    @property
    def polygon(self):
        return self._polygon

    @property
    def intensity(self) -> int:
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value

    @property
    def order(self):
        return self._order

    @property
    def is_block(self):
        return self._is_block

    @is_block.setter
    def is_block(self, value):
        self._is_block = value


class CrimeMap:
    def __init__(self, shape_file: str, grid_delta: float, threshold_percent: int):
        self._shape_file: str = shape_file
        self._delta: float = grid_delta
        self._threshold_percent: int = threshold_percent
        self._geo_data_frame: GeoDataFrame = gp.read_file(shape_file)
        bounds = self._geo_data_frame.total_bounds
        self._x_min: int = bounds[0]
        self._y_min: int = bounds[1]
        self._x_max: int = bounds[2]
        self._y_max: int = bounds[3]
        self._x_range: np.ndarray = np.arange(self._x_min, self._x_max, self._delta)
        self._y_range: np.ndarray = np.arange(self._y_min, self._y_max, self._delta)
        self._coord_size: int = len(self._y_range)

        self._points: List[Point] = self._make_points()

        polygons, centroids = self._make_polygons2()
        self._polygons = polygons
        self._centroids = centroids

        computed_statistics = self.compute_statistics()
        self._vertices: List[Tuple[float, float]] = computed_statistics['vertices']
        self._coord_intensities: List[CoordIntensityPair] = computed_statistics['coord_intensities']
        self._polygon_intensities: List[PolygonIntensityPair] = computed_statistics['polygon_intensities']

        self._values: List[int] = computed_statistics['values']

        self._sorted_polygon_intensities: List[PolygonIntensityPair] = computed_statistics['sorted_polygon_intensities']
        self._sorted_polygons: List[Polygon] = computed_statistics['sorted_polygons']
        self._sorted_centroids: List[Polygon] = computed_statistics['sorted_centroids']
        self._sorted_values: List[int] = computed_statistics['sorted_values']

        self._inflection_index: int = computed_statistics['inflection_index']
        self._colors: List[str] = computed_statistics['colors']
        self._sorted_norm_values: List[int] = computed_statistics['sorted_norm_values']
        self._v_min: int = computed_statistics['v_min']
        self._v_max: int = computed_statistics['v_max']
        self._v_median: float = computed_statistics['median']
        self._v_sum: int = computed_statistics['sum_values']
        self._v_std_dev: float = computed_statistics['std_dev']
        self._v_avg: float = computed_statistics['avg']

        self._xs: List[float] = list(self._x_range)
        self._xs.append(self._x_max)
        self._ys: List[float] = list(self._y_range)
        self._ys.append(self._y_max)

        self._grid: List[Tuple[float, float]] = self._make_grid()


    def plot_crime_points(self):
        self._geo_data_frame.plot()
        plt.show()

    def _make_points(self) -> List[Point]:
        coord_pairs: List[Point] = [v[3] for v in self._geo_data_frame.values]
        return coord_pairs

    def _make_polygons2(self):
        delta = self._delta
        x_left_origin = self._x_min
        x_right_origin = self._x_min + delta
        y_bottom_origin = self._y_min
        y_top_origin = self._y_min + delta
        polygons = []
        centroids = []

        for _ in self._y_range:
            x_left = x_left_origin
            x_right = x_right_origin

            for _ in self._x_range:
                polygons.append(Polygon(
                    [(x_left, y_top_origin),
                     (x_right, y_top_origin),
                     (x_right, y_bottom_origin),
                     (x_left, y_bottom_origin)]))
                centroids.append(((x_left + x_right) / 2, (y_bottom_origin + y_top_origin) / 2))
                x_left += delta
                x_right += delta

            y_bottom_origin += delta
            y_top_origin += delta

        return polygons, centroids

    def _make_grid(self) -> List[Tuple[float, float]]:
        xs: List[float] = list(self._x_range.copy())
        xs.append(self._x_max)
        ys: List[float] = list(self._y_range.copy())
        ys.append(self._y_max)

        grid = []
        for y in ys:
            for x in xs:
                grid.append((x, y))

        return grid

    def _closest_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        (x, y) = point
        x_index = int(np.floor(max(x - self._x_min, 0) / self._delta))
        y_index = int(np.floor(max(y - self._y_min, 0) / self._delta))
        return self._xs[x_index], self._ys[y_index]

    def travel(self, initial: Tuple[float, float], goal: Tuple[float, float]):
        pi = self._closest_point(initial)
        pf = self._closest_point(goal)
        ndim: int = len(self._grid)
        grid: np.ndarray = np.reshape(self._grid, (ndim, ndim))
        pass

    def compute_statistics(self):
        start: float = time()
        vertices: List[Tuple[float, float]] = []
        coord_intensities: List[CoordIntensityPair] = []

        points = self._points
        delta = self._delta
        x_min = self._x_min
        y_min = self._y_min
        x_range = self._x_range.copy()
        y_range = self._y_range.copy()
        threshold = self._threshold_percent

        for y in y_range:
            for x in x_range:
                vertices.append((x, y))

        for vertex in vertices:
            coord_intensities.append(CoordIntensityPair(vertex))

        for point in points:
            x_index = int(np.floor((point.x - x_min) / delta))
            y_index = int(np.floor((point.y - y_min) / delta))
            coord_intensities[y_index * len(self._y_range) + x_index].intensity += 1

        values = [ci.intensity for ci in coord_intensities]
        polygons = self._polygons

        pips: List[PolygonIntensityPair] = []

        for i in np.arange(0, len(polygons), 1):
            pips.append(PolygonIntensityPair.make(polygons[i], values[i], i))

        sorted_pips = sorted(pips, key=lambda pip: pip.intensity, reverse=True)
        sorted_values = [pip.intensity for pip in sorted_pips]
        sorted_polygons = [pip.polygon for pip in sorted_pips]

        inflection_index = int(len(sorted_polygons) * threshold / 100)
        yellow_colors = ['yellow'] * (len(sorted_polygons) - inflection_index)
        purple_colors = ['purple'] * (len(sorted_polygons) - len(yellow_colors))
        colors = yellow_colors + purple_colors
        norm_values = [1 if v == 'yellow' else 0 for v in colors]
        print(sorted_values)
        print(norm_values)
        print('inflection_index: ' + str(inflection_index))

        for i in np.arange(0, len(colors), 1):
            if colors[i] == 'yellow':
                sorted_pips[i].is_block = True

        v_min = min(values)
        v_max = max(values)
        median = threshold / 100 * (v_max - v_min) + v_min
        sum_values = sum(values)
        std_dev = np.std(values)
        avg = np.average(values)
        print('vmin: ' + str(v_min))
        print('vmax: ' + str(v_max))
        print('median: ' + str(median))
        print('sum_values: ' + str(sum_values))
        print('std_dev: ' + str(std_dev))
        print('avg: ' + str(avg))

        end: float = time()
        plot_data_time = end - start

        print('plot_data_time: ' + str(plot_data_time))

        return dict(
            vertices=vertices,
            coord_intensities=coord_intensities,
            polygon_intensities=sorted(sorted_pips, key=lambda pip: pip.order),

            values=values,

            sorted_polygon_intensities=sorted_pips,
            sorted_polygons=sorted_polygons,
            sorted_centroids=[polygon.centroid for polygon in sorted_polygons],
            sorted_values=sorted_values,

            inflection_index=inflection_index,
            colors=colors,
            sorted_norm_values=norm_values,

            v_min=v_min,
            v_max=v_max,
            median=median,
            sum_values=sum_values,
            std_dev=std_dev,
            avg=avg,
            plot_data_time=plot_data_time
        )

    def plot(self):
        gdf = gp.GeoDataFrame({'values': self._sorted_norm_values, 'colors': self._colors},
                              geometry=self._sorted_polygons)
        ax: Axes = gdf.plot(column='values', cmap='viridis')

        # fig: Figure = ax.get_figure()

        for i in np.arange(0, len(self._sorted_centroids), 1):
            centroid = self._sorted_centroids[i].xy
            x = centroid[0][0]
            y = centroid[1][0]
            val = self._sorted_values[i]
            color = 'white' if self._colors[i] == 'purple' else 'black'
            plt.text(x, y, str(val), fontdict=dict(color=color, fontsize=5, ha='center', va='center'))

        xs = list(self._x_range)
        xs.append(self._x_max)
        ys = list(self._y_range)
        ys.append(self._y_max)

        x_ticks = [x for (x, i) in zip(xs, np.arange(0, len(xs), 1)) if i % 2 == 0]
        y_ticks = [y for (y, i) in zip(ys, np.arange(0, len(ys), 1)) if i % 2 == 0]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['{:.3f}'.format(x_tick) for x_tick in x_ticks], fontdict=dict(fontsize=6))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.3f}'.format(y_tick) for y_tick in y_ticks], fontdict=dict(fontsize=6))
        ax.set_title(
            'std_dev = {std_dev}, avg = {avg}'.format(std_dev='{:.2f}'.format(self._v_std_dev),
                                                      avg='{:.2f}'.format(self._v_avg)),
            pad=10,
            fontdict=dict(fontsize=8))
        plt.show()
        # self._geo_data_frame.plot(ax=ax)


def main():
    start_time = time()
    crime_map = CrimeMap('./Shape/crime_dt.shp', 0.002, 50)
    crime_map.plot()
    # crime_map.travel(start=(1,1), goal=(2,2))
    end_time = time()
    exec_time = end_time - start_time
    print('exec_time: ' + str(exec_time))


if __name__ == '__main__':
    main()
