from queue import PriorityQueue
from time import time
from typing import Tuple, List

import tkinter
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import geopandas as gp
from geopandas import GeoDataFrame
from matplotlib import animation
from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from shapely.geometry import Polygon, Point

plt.rcParams.update({'figure.dpi': 350})
matplotlib.use("TkAgg")


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
    def __init__(self, shape_file: str, grid_delta: float, threshold_percent: float):
        self._shape_file: str = shape_file
        self._delta: float = grid_delta
        self._threshold_percent: float = threshold_percent
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

        self._n_high: int = computed_statistics['n_high']
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
        self._grid_x_dim: int = len(self._xs)

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

        grid = [(x, y) for y in ys for x in xs]

        return grid

    def _closest_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        xi, yi = self._closest_indices(point)
        return self._xs[xi], self._ys[yi]

    def _closest_indices(self, point: Tuple[float, float]):
        (x, y) = point
        x_index = min(int(np.round(max(x - self._x_min, 0) / self._delta)), len(self._x_range))
        y_index = min(int(np.round(max(y - self._y_min, 0) / self._delta)), len(self._x_range))
        return x_index, y_index

    def travel(self, ax: Axes, initial: Tuple[float, float], final: Tuple[float, float]):
        delta = self._delta
        grid = self._grid
        goal = self._closest_point(final)
        xf, yf = self._closest_indices(goal)
        xi, yi = self._closest_indices(initial)
        n = self._grid_x_dim
        pq = PriorityQueue()
        is_goal_reached: bool = False
        came_from_set = []
        came_from_cost_point_set = []
        cameFrom = {}

        def i1d2(p_index):
            (x_index, y_index) = p_index
            return y_index * int(np.sqrt(len(self._polygon_intensities))) + x_index

        def i1d(x_index, y_index):
            return y_index * n + x_index

        def g(p_index: Tuple[int, int], neighbour_index: Tuple[int, int]) -> float:
            (x_index, y_index) = p_index
            (x_index_neighbour, y_index_neighbour) = neighbour_index
            return 0.0

        def h(x_index, y_index) -> float:
            (px, py) = grid[i1d(x_index, y_index)]
            (gx, gy) = goal
            return (((gx - px) ** 2) + ((gy - py) ** 2)) ** 0.5

        def get_cost(p_index: Tuple[int, int], neighbour_index: Tuple[int, int]) -> float:
            (x_index, y_index) = p_index
            return g(p_index, neighbour_index) + h(x_index, y_index)

        def is_in_range2(p_index) -> bool:
            (x_index, y_index) = p_index
            return is_in_range(x_index, y_index)

        def is_in_range(x_index, y_index) -> bool:
            return 0 <= x_index < n and 0 <= y_index < n

        def get_neighbours(x_index, y_index):
            indices = [
                (x_index, y_index + 1),      # up
                (x_index + 1, y_index + 1),  # up-right
                (x_index + 1, y_index),      # right
                (x_index + 1, y_index - 1),  # down-right
                (x_index, y_index - 1),      # down
                (x_index - 1, y_index - 1),  # down-left
                (x_index - 1, y_index),      # left
                (x_index - 1, y_index + 1)   # up-left
            ]

            polygon_indices = [
                (x_index, y_index),          # up-right
                (x_index, y_index - 1),      # down-right
                (x_index - 1, y_index - 1),  # down-left
                (x_index - 1, y_index)       # up-left
            ]

            def get_cost_non_diagonal(is_block_1, is_block_2):
                if is_block_1 and is_block_2:
                    return None
                if is_block_1:
                    return 1.3
                return 1.0

            cost_index_pairs = []

            def is_polygon_in_range(polygon_index):
                x_polygon_index, y_polygon_index = polygon_index
                n_polygon = int(np.sqrt(len(self._polygon_intensities)))
                return 0 <= x_polygon_index < n_polygon and 0 <= y_polygon_index < n_polygon

            if is_in_range2(indices[0]) and is_polygon_in_range(polygon_indices[0]) and is_polygon_in_range(polygon_indices[3]):
                is_block_right = self._polygon_intensities[i1d2(polygon_indices[0])].is_block
                is_block_left = self._polygon_intensities[i1d2(polygon_indices[3])].is_block
                cost_non_diagonal = get_cost_non_diagonal(is_block_right, is_block_left)
                if cost_non_diagonal is not None:
                    cost_index_pairs.append((cost_non_diagonal, indices[0]))

            if is_in_range2(indices[1]) and is_polygon_in_range(polygon_indices[0]):
                is_block = self._polygon_intensities[i1d2(polygon_indices[0])].is_block
                if not is_block:
                    cost_index_pairs.append((1.5, indices[1]))

            if is_in_range2(indices[2]) and is_polygon_in_range(polygon_indices[0]) and is_polygon_in_range(polygon_indices[1]):
                is_block_up = self._polygon_intensities[i1d2(polygon_indices[0])].is_block
                is_block_down = self._polygon_intensities[i1d2(polygon_indices[1])].is_block
                cost_non_diagonal = get_cost_non_diagonal(is_block_up, is_block_down)
                if cost_non_diagonal is not None:
                    cost_index_pairs.append((cost_non_diagonal, indices[2]))

            if is_in_range2(indices[3]) and is_polygon_in_range(polygon_indices[1]):
                is_block = self._polygon_intensities[i1d2(polygon_indices[1])].is_block
                if not is_block:
                    cost_index_pairs.append((1.5, indices[3]))

            if is_in_range2(indices[4]) and is_polygon_in_range(polygon_indices[1]) and is_polygon_in_range(polygon_indices[2]):
                is_block_right = self._polygon_intensities[i1d2(polygon_indices[1])].is_block
                is_block_left = self._polygon_intensities[i1d2(polygon_indices[2])].is_block
                cost_non_diagonal = get_cost_non_diagonal(is_block_right, is_block_left)
                if cost_non_diagonal is not None:
                    cost_index_pairs.append((cost_non_diagonal, indices[4]))

            if is_in_range2(indices[5]) and is_polygon_in_range(polygon_indices[2]):
                is_block = self._polygon_intensities[i1d2(polygon_indices[2])].is_block
                if not is_block:
                    cost_index_pairs.append((1.5, indices[5]))

            if is_in_range2(indices[6]) and is_polygon_in_range(polygon_indices[3]) and is_polygon_in_range(polygon_indices[2]):
                is_block_up = self._polygon_intensities[i1d2(polygon_indices[3])].is_block
                is_block_down = self._polygon_intensities[i1d2(polygon_indices[2])].is_block
                cost_non_diagonal = get_cost_non_diagonal(is_block_up, is_block_down)
                if cost_non_diagonal is not None:
                    cost_index_pairs.append((cost_non_diagonal, indices[6]))

            if is_in_range2(indices[7]) and is_polygon_in_range(polygon_indices[3]):
                is_block = self._polygon_intensities[i1d2(polygon_indices[3])].is_block
                if not is_block:
                    cost_index_pairs.append((1.5, indices[7]))

            return cost_index_pairs

        def draw_markers():
            ax.plot([xi], [yi], color='black')
            plt.draw()  # re-draw the figure
            plt.pause(0.000000000001)
            ax.plot([xf], [yf], color='black')
            plt.draw()  # re-draw the figure
            plt.pause(0.000000000001)

        def draw_lines(point: Tuple[int, int], points: List[Tuple[int, int]], color=None):
            xs = self._xs
            ys = self._ys
            (xi1, yi1) = point

            lines = [([xs[xi1], xs[xi2]], [ys[yi1], ys[yi2]]) for xi2, yi2 in points]

            for line in lines:
                x_values, y_values = line
                ax.plot(x_values, y_values, color=color if color is not None else 'white')
                # fig: Figure = ax.get_figure()
                # plt.draw()  # re-draw the figure
                # plt.pause(0.000000000001)

        # draw_markers()
        pq.put((100000000.0, (xi, yi)))

        closed_set = []

        path = []

        while not pq.empty():
            cost, pi = pq.get()
            (xi, yi) = pi

            if pi == (xf, yf):
                is_goal_reached = True
                path = [pi]
                while pi in cameFrom:
                    pi = cameFrom[pi]
                    path.append(pi)
                path.reverse()
                break

            closed_set.append(pi)

            cips = get_neighbours(xi, yi)
            pis = [pi for (c, pi) in cips]
            draw_lines((xi, yi), pis)

            for (g_cost_neighbour, neighbour) in cips:
                (x_neighbour, y_neighbour) = neighbour
                cost_neighbour = g_cost_neighbour + 0* h(x_neighbour, y_neighbour)

                if cost_neighbour < cost and pi not in came_from_set:
                    came_from_set.append(pi)
                    came_from_cost_point_set.append((cost, pi))

                if cost_neighbour < cost:
                    cameFrom[neighbour] = pi

                # if neighbour not in closed_set \
                #         and neighbour not in [p for (c, p) in pq.queue] \
                #         and cost_neighbour < cost:
                #     cameFrom[neighbour] = pi

                if neighbour not in closed_set and neighbour not in [p for (c, p) in pq.queue]:
                    pq.put((cost_neighbour, neighbour))
                    # pq.put((cost + cost_neighbour, neighbour))
                    # if cost_neighbour < cost and pi not in came_from_set:
                    #     came_from_set.append(pi)
                    #     came_from_cost_point_set.append((cost, pi))
                elif neighbour in [p for (c, p) in pq.queue]:
                    queue = [p for (c, p) in pq.queue]
                    same_neighbour_index = queue.index(neighbour)
                    (same_neighbour_cost, same_neighbour) = pq.queue[same_neighbour_index]
                    # if cost_neighbour < cost and pi not in came_from_set:
                    #     came_from_set.append(pi)
                    #     came_from_cost_point_set.append((cost, pi))

                    if same_neighbour_cost > cost_neighbour:
                    # if same_neighbour_cost > cost + cost_neighbour:
                        if same_neighbour in came_from_set:
                            print('here')

                        new_pq = PriorityQueue()
                        for (c, p) in pq.queue:
                            if not p == neighbour:
                                new_pq.put((c, p))
                        new_pq.put((cost_neighbour, neighbour))
                        # new_pq.put((cost + cost_neighbour, neighbour))
                        pq = new_pq

        print('is_goal_reached: ' + str(is_goal_reached))

        if is_goal_reached:
            for i in np.arange(1, len(path), 1):
                draw_lines(path[i - 1], [path[i]], color='black')

        if is_goal_reached:
            for i in np.arange(1, len(came_from_set), 1):
                draw_lines(came_from_set[i - 1], [came_from_set[i]], color='black')

        # if is_goal_reached:
        #     path = []
        #     came_from_cost_point_set_reversed = came_from_cost_point_set[::-1]
        #     p_current_cost, (p_current_x, p_current_y) = came_from_cost_point_set_reversed[0]
        #     for i in np.arange(1, len(came_from_cost_point_set_reversed), 1):
        #         p_next_cost, (p_next_x, p_next_y) = came_from_cost_point_set_reversed[i]
        #         if p_next_cost > p_current_cost:
        #             path.append((p_current_x, p_current_y))
        #             p_current_x = p_next_x
        #             p_current_y = p_next_y
        #     path_reversed = path[::-1]
        #     for i in np.arange(1, len(path_reversed), 1):
        #         draw_lines(path_reversed[i - 1], [path_reversed[i]], color='black')

        # if is_goal_reached:
        #     path = []
        #     p_current_cost, (p_current_x, p_current_y) = came_from_cost_point_set[0]
        #     for i in np.arange(1, len(came_from_cost_point_set), 1):
        #         p_next_cost, (p_next_x, p_next_y) = came_from_cost_point_set[i]
        #         if p_next_cost < p_current_cost and (p_next_x, p_next_y) not in path:
        #             path.append((p_current_x, p_current_y))
        #             p_current_x = p_next_x
        #             p_current_y = p_next_y
        #     for i in np.arange(1, len(path), 1):
        #         draw_lines(path[i - 1], [path[i]], color='black')

        # if is_goal_reached:
        #     path = []
        #     came_from_set_reversed = list(reversed(came_from_set))
        #     (p_current_x, p_current_y) = came_from_set_reversed[0]
        #
        #     for i in np.arange(1, len(came_from_set), 1):
        #         (p_next_x, p_next_y) = came_from_set_reversed[i]
        #         dx = abs(p_current_x - p_next_x)
        #         dy = abs(p_current_y - p_next_y)
        #         if (dx > 1 or dy > 1) or (dx == 0 and dy == 0):
        #             continue
        #         else:
        #             path.append((p_current_x, p_current_y))
        #             p_current_x = p_next_x
        #             p_current_y = p_next_y
        #
        #     path_reversed = path[::-1]
        #     for i in np.arange(1, len(path_reversed), 1):
        #         draw_lines(path_reversed[i - 1], [path_reversed[i]], color='black')

        # if is_goal_reached:
        #     path = []
        #     path_costs = []
        #     came_from_cost_point_set_reversed = came_from_cost_point_set[::-1]
        #     p_current_cost, (p_current_x, p_current_y) = came_from_cost_point_set_reversed[0]
        #
        #     for i in np.arange(1, len(came_from_cost_point_set_reversed), 1):
        #         p_next_cost, (p_next_x, p_next_y) = came_from_cost_point_set_reversed[i]
        #         dx = abs(p_current_x - p_next_x)
        #         dy = abs(p_current_y - p_next_y)
        #         # if (dx > 1 or dy > 1) or (dx == 0 and dy == 0):
        #         #     #continue
        #         #     pass
        #         if p_next_cost > p_current_cost:
        #             path.append((p_current_x, p_current_y))
        #             path_costs.append((p_current_cost, (p_current_x, p_current_y)))
        #             p_current_x = p_next_x
        #             p_current_y = p_next_y
        #             p_current_cost = p_next_cost
        #
        #     path_reversed = path[::-1]
        #     path_costs_reversed = path_costs[::-1]
        #     for i in np.arange(1, len(path_reversed), 1):
        #         draw_lines(path_reversed[i - 1], [path_reversed[i]], color='black')

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
            x_index = min(int(np.floor((point.x - x_min) / delta)), len(self._x_range) - 1)
            y_index = min(int(np.floor((point.y - y_min) / delta)), len(self._y_range) - 1)
            coord_intensities[min(y_index * len(self._y_range) + x_index, len(vertices) - 1)].intensity += 1

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

            n_high=len(yellow_colors),
            v_min=v_min,
            v_max=v_max,
            median=median,
            sum_values=sum_values,
            std_dev=std_dev,
            avg=avg,
            plot_data_time=plot_data_time
        )

    def plot(self) -> Axes:
        gdf = gp.GeoDataFrame({'values': self._sorted_norm_values, 'colors': self._colors},
                              geometry=self._sorted_polygons)

        ax: Axes = gdf.plot(column='values', cmap='viridis') if self._threshold_percent != 0 else gdf.plot(
            color='yellow')

        for i in np.arange(0, len(self._sorted_centroids), 1):
            centroid = self._sorted_centroids[i].xy
            x = centroid[0][0]
            y = centroid[1][0]
            val = self._sorted_values[i]
            color = 'white' if self._colors[i] == 'purple' else 'black'
            plt.text(x, y, str(val), fontdict=dict(color=color, fontsize=3, ha='center', va='center'))

        xs = list(self._x_range)
        xs.append(self._x_max)
        ys = list(self._y_range)
        ys.append(self._y_max)

        x_ticks = [x for (x, i) in zip(xs, np.arange(0, len(xs), 1)) if i % 2 == 0]
        y_ticks = [y for (y, i) in zip(ys, np.arange(0, len(ys), 1)) if i % 2 == 0]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['{:.3f}'.format(x_tick) for x_tick in x_ticks], fontdict=dict(fontsize=4))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.3f}'.format(y_tick) for y_tick in y_ticks], fontdict=dict(fontsize=4))
        ax.set_title(
            'n_high={n_high}, std_dev={std_dev}, avg={avg}, delta={delta}, threshold={threshold}%'.format(
                std_dev='{:.2f}'.format(self._v_std_dev),
                avg='{:.2f}'.format(self._v_avg),
                delta='{:.3f}'.format(self._delta),
                threshold=str(self._threshold_percent),
                n_high=self._n_high
            ),
            pad=10,
            fontdict=dict(fontsize=8))

        plt.draw()
        # self._geo_data_frame.plot(ax=ax)

        return ax

        # root = tkinter.Tk()
        # fig = plt.Figure()
        # canvas = FigureCanvasTkAgg(fig, root)
        # canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        # ax_gui: Axes = fig.add_subplot(111)
        #
        # fig.subplots_adjust(bottom=0.25)
        # y_values = [random.randrange(20, 40, 1) for _ in range(40)]
        # x_values = [i for i in range(40)]
        #
        # # ax.axis([0, 9, 20, 40])
        # # ax.plot(x_values, y_values)
        # gdf.plot(column='values', cmap='viridis', ax=ax_gui) if self._threshold_percent != 0 else gdf.plot(
        #     color='yellow', ax=ax_gui)
        #
        # for i in np.arange(0, len(self._sorted_centroids), 1):
        #     centroid = self._sorted_centroids[i].xy
        #     x = centroid[0][0]
        #     y = centroid[1][0]
        #     val = self._sorted_values[i]
        #     color = 'white' if self._colors[i] == 'purple' else 'black'
        #
        #     ax_gui.text(x, y, str(val), fontdict=dict(color=color, fontsize=3, ha='center', va='center'))
        #
        # ax_threshold = fig.add_axes([(0.25+0.25/2), 0.1, 0.25, 0.03])
        # s_threshold = Slider(ax_threshold, 'Threshold (%)', 0, 100, valinit=50, valstep=1)
        #
        # def update(threshold):
        #     ax_gui.set_title('std_dev = {std_dev}, avg = {avg} ({aperture}, {threshold}%)'.format(
        #         std_dev='{:.2f}'.format(self._v_std_dev),
        #         avg='{:.2f}'.format(self._v_avg), aperture='{:.3f}'.format(self._delta),
        #         threshold='{:.2f}'.format(threshold)))
        #     fig.canvas.draw_idle()
        #
        # s_threshold.on_changed(update)
        #
        # tkinter.mainloop()


def main():
    plt.ion()
    start_time = time()
    crime_map = CrimeMap('./Shape/crime_dt.shp', 0.002, 65)
    ax = crime_map.plot()
    # crime_map.travel(ax=ax, initial=(-73.586, 45.510), final=(-73.552, 45.494))
    # crime_map.travel(ax=ax, initial=(-73.566, 45.528), final=(-73.552, 45.494))
    # crime_map.travel(ax=ax, initial=(-73.588, 45.494), final=(-73.554, 45.526))
    crime_map.travel(ax=ax, initial=(-73.588, 45.494), final=(-73.555, 45.514))
    end_time = time()
    exec_time = end_time - start_time
    print('exec_time: ' + str(exec_time))

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
