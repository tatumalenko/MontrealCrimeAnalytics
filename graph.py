import tkinter
from queue import PriorityQueue
from time import time
from typing import Tuple, Optional, List, Set, Any, Union

from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import TextBox
from shapely.geometry import Polygon, Point
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({'figure.dpi': 350})
# matplotlib.use("TkAgg")


class HashVertex:
    _x: float
    _y: float
    _point: Tuple[float, float]
    _previous: Optional['HashVertex']
    _next: Optional['HashVertex']
    _cells: Optional[List['HashCell']]
    _hash_value: int
    _bottom_left_cell: Optional['HashCell']
    _bottom_right_cell: Optional['HashCell']
    _top_left_cell: Optional['HashCell']
    _top_right_cell: Optional['HashCell']
    _cost_g: float
    _cost_h: float
    _cost_f: float

    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y
        self._point = (x, y)
        self._previous = None
        self._next = None
        self._bottom_left_cell = None
        self._bottom_right_cell = None
        self._top_left_cell = None
        self._top_right_cell = None
        self._cells = []
        self._hash_value = hash(self._point)
        self.top: Optional['HashVertex'] = None
        self.top_right: Optional['HashVertex'] = None
        self.right: Optional['HashVertex'] = None
        self.bottom_right: Optional['HashVertex'] = None
        self.bottom: Optional['HashVertex'] = None
        self.bottom_left: Optional['HashVertex'] = None
        self.left: Optional['HashVertex'] = None
        self.top_left: Optional['HashVertex'] = None
        self._cost_g: float = 0.0
        self._cost_h: float = 0.0
        self._cost_f: float = 0.0

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def point(self) -> Tuple[float, float]:
        return self._point

    @property
    def previous(self) -> Optional['HashVertex']:
        return self._previous

    @previous.setter
    def previous(self, vertex: 'HashVertex'):
        self._previous = vertex

    @property
    def next(self) -> Optional['HashVertex']:
        return self._next

    @next.setter
    def next(self, vertex: 'HashVertex'):
        self._next = vertex

    @property
    def bottom_left_cell(self):
        return self._bottom_left_cell

    @bottom_left_cell.setter
    def bottom_left_cell(self, cell: 'HashCell'):
        self._bottom_left_cell = cell

    @property
    def bottom_right_cell(self):
        return self._bottom_right_cell

    @bottom_right_cell.setter
    def bottom_right_cell(self, cell: 'HashCell'):
        self._bottom_right_cell = cell

    @property
    def top_left_cell(self):
        return self._top_left_cell

    @top_left_cell.setter
    def top_left_cell(self, cell: 'HashCell'):
        self._top_left_cell = cell

    @property
    def top_right_cell(self):
        return self._top_right_cell

    @top_right_cell.setter
    def top_right_cell(self, cell: 'HashCell'):
        self._top_right_cell = cell

    @property
    def cells(self) -> List['HashCell']:
        return self._cells

    @property
    def cost_g(self) -> float:
        return self._cost_g

    @cost_g.setter
    def cost_g(self, cost_g: float):
        self._cost_g = cost_g

    @property
    def cost_h(self) -> float:
        return self._cost_h

    @cost_h.setter
    def cost_h(self, cost_h: float):
        self._cost_h = cost_h

    @property
    def cost_f(self) -> float:
        return self.cost_g + self.cost_h

    def distance(self, vertex: 'HashVertex'):
        return (((self.x - vertex.x) ** 2) + ((self.y - vertex.y) ** 2)) ** 0.5

    def __eq__(self, other: 'HashVertex') -> bool:
        return self.point == other.point

    def __hash__(self) -> int:
        return self._hash_value

    def __lt__(self, other: 'HashVertex') -> bool:
        return self._hash_value < other._hash_value


class HashCell:
    _polygon: Polygon
    _value: int
    _norm_value: int
    _is_block: bool
    _x_min: float
    _y_min: float
    _x_max: float
    _y_max: float
    _centroid: Point
    _bottom_right_vertex: HashVertex
    _top_left_vertex: HashVertex
    _bottom_left_vertex: HashVertex
    _top_right_vertex: HashVertex
    _vertices: Set[HashVertex]
    _points: Tuple[HashVertex, HashVertex, HashVertex, HashVertex]
    _hash_value: int
    left: Optional['HashCell']
    right: Optional['HashCell']
    top: Optional['HashCell']
    bottom: Optional['HashCell']

    def __init__(self, polygon: Polygon):
        self._polygon: Polygon = polygon
        (x_min, y_min, x_max, y_max) = polygon.bounds
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max
        self._centroid = Point((self.x_max - self.x_min) / 2 + self.x_min, (self.y_max - self.y_min) / 2 + self.y_min)
        self._value = 0
        self._norm_value = 0
        self._is_block = False
        self._bottom_left_vertex = HashVertex(self._x_min, self._y_min)
        self._top_left_vertex = HashVertex(self._x_min, self._y_max)
        self._bottom_right_vertex = HashVertex(self._x_max, self._y_min)
        self._top_right_vertex = HashVertex(self._x_max, self._y_max)
        self._vertices: Set[HashVertex] = {
            self._bottom_left_vertex,
            self._top_left_vertex,
            self._bottom_right_vertex,
            self._top_right_vertex
        }
        self._points = (
            self._bottom_left_vertex,
            self._top_left_vertex,
            self._bottom_right_vertex,
            self._top_right_vertex
        )
        self._hash_value = hash(self._points)
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def y_max(self) -> float:
        return self._y_max

    @property
    def centroid(self) -> Point:
        return self._centroid

    @property
    def bottom_left_vertex(self) -> 'HashVertex':
        return self._bottom_left_vertex

    @property
    def bottom_right_vertex(self) -> 'HashVertex':
        return self._bottom_right_vertex

    @property
    def top_right_vertex(self) -> 'HashVertex':
        return self._top_right_vertex

    @property
    def top_left_vertex(self) -> 'HashVertex':
        return self._top_left_vertex

    @property
    def vertices(self) -> Set[HashVertex]:
        return self._vertices

    @property
    def points(self) -> Tuple[HashVertex, HashVertex, HashVertex, HashVertex]:
        return self._points

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, val: int):
        self._value = val

    @property
    def norm_value(self) -> int:
        return self._norm_value

    @norm_value.setter
    def norm_value(self, val: int):
        self._norm_value = val

    @property
    def is_block(self) -> bool:
        return self._is_block

    @is_block.setter
    def is_block(self, val: bool):
        self._is_block = val

    def __eq__(self, other: 'HashCell') -> bool:
        return self.points == other.points

    def __hash__(self) -> int:
        return self._hash_value


class VertexSearchQueue(PriorityQueue):
    def vertices(self) -> List[HashVertex]:
        return [v for (_, v) in self.queue]

    def find(self, vertex: HashVertex) -> Tuple[Any, HashVertex]:
        vs: List[HashVertex] = self.vertices()
        idx: int = vs.index(vertex)
        return self.queue[idx]

    def remove(self, vertex: HashVertex) -> 'VertexSearchQueue[Tuple[Any, HashVertex]]':
        pq: VertexSearchQueue[Tuple[Any, HashVertex]] = VertexSearchQueue()
        for (priority, v) in self.queue:
            if not v == vertex:
                pq.put((priority, v))
        return pq

    def replace(self,
                to_remove: HashVertex,
                to_put: Tuple[Any, HashVertex]) -> 'VertexSearchQueue[Tuple[Any, HashVertex]]':
        pq: VertexSearchQueue[Tuple[Any, HashVertex]] = self.remove(to_remove)
        pq.put(to_put)
        return pq


class Graph:
    _bounds: Tuple[float, float, float, float]
    _x_min: float
    _y_min: float
    _x_max: float
    _y_max: float
    _delta: float
    _xs: List[float]
    _ys: List[float]
    _nx: int
    _ny: int
    _cells: List[HashCell]
    _vertices: List[HashVertex]

    def __init__(self, bounds: Tuple[float, float, float, float], delta: float):
        self._bounds = bounds
        (x_min, y_min, x_max, y_max) = bounds
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max
        self._delta = delta
        self._xs = list(np.arange(x_min, x_max, delta))
        self._ys = list(np.arange(y_min, y_max, delta))
        self._nx = len(self._xs)
        self._ny = len(self._ys)
        self._cells, self._vertices = self._make_grid()

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def y_max(self) -> float:
        return self._y_max

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def xs(self) -> List[float]:
        return self._xs

    @property
    def ys(self) -> List[float]:
        return self._ys

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def cells(self) -> List[HashCell]:
        return self._cells

    @property
    def vertices(self) -> List[HashVertex]:
        return self._vertices

    def polygons(self) -> List[Polygon]:
        return [cell.polygon for cell in self.cells]

    def ij2k(self, i: int, j: int) -> int:
        return j * self.ny + i

    def _make_grid(self) -> Tuple[List[HashCell], List[HashVertex]]:
        cells: List[HashCell] = []
        vertices: List[HashVertex] = []

        x_left_origin = self.x_min
        x_right_origin = self.x_min + self.delta
        y_bottom_origin = self.y_min
        y_top_origin = self.y_min + self.delta

        for _ in self.ys:
            x_left = x_left_origin
            x_right = x_right_origin
            for _ in self.xs:
                cell = HashCell(Polygon(
                    [(x_left, y_top_origin),
                     (x_right, y_top_origin),
                     (x_right, y_bottom_origin),
                     (x_left, y_bottom_origin)]))
                cells.append(cell)
                x_left += self.delta
                x_right += self.delta
            y_bottom_origin += self.delta
            y_top_origin += self.delta

        for j in range(0, self.ny):
            for i in range(0, self.nx):
                k = self.ij2k(i, j)
                cells[k].left = cells[self.ij2k(i - 1, j)] if 0 <= i - 1 < self.nx else None
                cells[k].right = cells[self.ij2k(i + 1, j)] if 0 <= i + 1 < self.nx else None
                cells[k].bottom = cells[self.ij2k(i, j - 1)] if 0 <= j - 1 < self.ny else None
                cells[k].top = cells[self.ij2k(i, j + 1)] if 0 <= j + 1 < self.ny else None

                vertices.append(cells[k].bottom_left_vertex)

                if i == self.nx - 1:
                    vertices.append(cells[k].bottom_right_vertex)

                if j == self.ny - 1 and not i == self.nx - 1:
                    vertices.append(cells[k].top_left_vertex)

                if j == self.ny - 1 and i == self.nx - 1:
                    vertices.append(cells[k].top_right_vertex)

        for j in range(0, self.ny):
            for i in range(0, self.nx):
                k = self.ij2k(i, j)

                cells[k].bottom_right_vertex.top_left_cell = cells[k]
                cells[k].bottom_right_vertex.top_right_cell = cells[k].right
                cells[k].bottom_right_vertex.bottom_left_cell = cells[k].bottom
                cells[k].bottom_right_vertex.bottom_right_cell = cells[k].right.bottom if cells[
                                                                                              k].right is not None else None

                cells[k].top_right_vertex.bottom_left_cell = cells[k]
                cells[k].top_right_vertex.bottom_right_cell = cells[k].right
                cells[k].top_right_vertex.top_left_cell = cells[k].top
                cells[k].top_right_vertex.top_right_cell = cells[k].right.top if cells[k].right is not None else None

                cells[k].bottom_left_vertex.top_right_cell = cells[k]
                cells[k].bottom_left_vertex.top_left_cell = cells[k].left
                cells[k].bottom_left_vertex.bottom_right_cell = cells[k].bottom
                cells[k].bottom_left_vertex.bottom_left_cell = cells[k].left.bottom if cells[
                                                                                           k].left is not None else None

                cells[k].top_left_vertex.bottom_right_cell = cells[k]
                cells[k].top_left_vertex.bottom_left_cell = cells[k].left
                cells[k].top_left_vertex.top_right_cell = cells[k].top
                cells[k].top_left_vertex.top_left_cell = cells[k].left.top if cells[k].left is not None else None

        return cells, vertices

    def closest_indices(self, point: Union[Point, HashVertex]) -> Tuple[int, int]:
        xi = min(int(np.round(max(point.x - self.x_min, 0) / self.delta)), self.nx)
        yi = min(int(np.round(max(point.y - self.y_min, 0) / self.delta)), self.ny)
        return xi, yi

    def closest_vertex(self, point: Point) -> HashVertex:
        closest_distance = float('inf')
        closest_vertex: Optional[HashVertex] = None
        for vertex in self.vertices:
            distance = vertex.distance(HashVertex(point.x, point.y))
            if distance < closest_distance:
                closest_distance = distance
                closest_vertex = vertex
        return closest_vertex

    def closest_point(self, point: Point) -> Point:
        vertex = self.closest_vertex(point)
        return Point(vertex.x, vertex.y)

    @staticmethod
    def cost_orthogonal(ca: HashCell, cb: HashCell):
        if ca.is_block and cb.is_block:
            return None
        elif ca.is_block or cb.is_block:
            return 1.3
        return 1.0

    @staticmethod
    def cost_diagonal(cell: HashCell):
        if cell.is_block:
            return None
        return 1.5

    def get_cost_neighbour_pairs(self, vertex: HashVertex) -> Set[Tuple[float, HashVertex]]:
        cost_neighbour_pairs: List[Tuple[float, HashVertex]] = []

        # top
        if vertex.top_left_cell is not None and vertex.top_right_cell is not None:
            cost = self.cost_orthogonal(vertex.top_left_cell, vertex.top_right_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.top_right_cell.top_left_vertex))

        # right
        if vertex.top_right_cell is not None and vertex.bottom_right_cell is not None:
            cost = self.cost_orthogonal(vertex.top_right_cell, vertex.bottom_right_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.top_right_cell.bottom_right_vertex))

        # bottom
        if vertex.bottom_left_cell is not None and vertex.bottom_right_cell is not None:
            cost = self.cost_orthogonal(vertex.bottom_left_cell, vertex.bottom_right_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.bottom_right_cell.bottom_left_vertex))

        # left
        if vertex.top_left_cell is not None and vertex.bottom_left_cell is not None:
            cost = self.cost_orthogonal(vertex.top_left_cell, vertex.bottom_left_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.top_left_cell.bottom_left_vertex))

        # top-right
        if vertex.top_right_cell is not None and vertex.top_right_cell.top_right_vertex is not None:
            cost = self.cost_diagonal(vertex.top_right_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.top_right_cell.top_right_vertex))

        # bottom-right
        if vertex.bottom_right_cell is not None and vertex.bottom_right_cell.bottom_right_vertex is not None:
            cost = self.cost_diagonal(vertex.bottom_right_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.bottom_right_cell.bottom_right_vertex))

        # bottom-left
        if vertex.bottom_left_cell is not None and vertex.bottom_left_cell.bottom_left_vertex is not None:
            cost = self.cost_diagonal(vertex.bottom_left_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.bottom_left_cell.bottom_left_vertex))

        # top-left
        if vertex.top_left_cell is not None and vertex.top_left_cell.top_left_vertex is not None:
            cost = self.cost_diagonal(vertex.top_left_cell)
            if cost is not None:
                cost_neighbour_pairs.append((cost, vertex.top_left_cell.top_left_vertex))

        return set(cost_neighbour_pairs)

    def h(self, vn: HashVertex, vf: HashVertex) -> float:
        xn, yn = self.closest_indices(vn)
        xf, yf = self.closest_indices(vf)
        delta_x = abs(xf - xn)
        delta_y = abs(yf - yn)
        delta_min = min(delta_x, delta_y)
        delta_max = max(delta_x, delta_y)
        delta_diagonal = delta_min
        delta_orthogonal = delta_max - delta_diagonal
        return delta_orthogonal * 1.0 + delta_diagonal * 1.5

    def search(self, ax: Axes, start: Point, end: Point) -> float:
        pq: VertexSearchQueue[Tuple[float, HashVertex]] = VertexSearchQueue()
        is_goal_reached: bool = False
        vi = self.closest_vertex(start)
        vf = self.closest_vertex(end)
        visited: Set[HashVertex] = set()
        path: List[HashVertex] = []
        cost_path: float = float('Inf')
        cost_table = []

        pq.put((0, vi))

        while not pq.empty():
            v: HashVertex
            cost_f, v = pq.get()

            if v == vf:
                cost_path = v.cost_g
                is_goal_reached = True
                path.append(v)

                while v.previous is not None:
                    step_cost = v.cost_g - v.previous.cost_g
                    h_prime = v.cost_h
                    h = v.previous.cost_h
                    step_cost_plus_h_prime = step_cost + h_prime
                    h_star = cost_path - v.previous.cost_g
                    h_str = '{:.2f}'.format(h)
                    h_star_str = '{:.2f}'.format(h_star)
                    step_cost_plus_h_prime_str = '{:.2f}'.format(step_cost_plus_h_prime)
                    cost_table.append(f'h*(n)={h_star_str}, h(n)={h_str}, h(n)<=h*(n): {h <= h_star}, c(n,n\')+h(n\')={step_cost_plus_h_prime_str}, c(n,n\')+h(n\')>=h(n): {step_cost_plus_h_prime >= h}')
                    path.append(v.previous)
                    v = v.previous

                path.reverse()
                break

            visited.add(v)

            cost_neighbour_pairs = self.get_cost_neighbour_pairs(v)
            self.draw_path_attempt(ax=ax, vertex=v)

            for cost_neighbour_pair in cost_neighbour_pairs:
                (g, vn) = cost_neighbour_pair
                cost_h = self.h(vn, vf)
                cost_g = v.cost_g + g

                print('cost_gn: ' + str(cost_g) + '; cost_hn: ' + str(cost_h) + '; cost_fn: ' + str(cost_g + cost_h))
                queue_vertices = pq.vertices()

                if vn not in visited and vn not in queue_vertices:
                    vn.previous = v
                    vn.cost_g = cost_g
                    vn.cost_h = cost_h

                    pq.put((vn.cost_f, vn))
                elif vn in queue_vertices:
                    (cost_fn_same, vn_same) = pq.find(vn)

                    # replace lower cost vertex
                    if vn_same.cost_g > cost_g:
                        vn.previous = v

                        vn.cost_g = cost_g
                        vn.cost_h = cost_h

                        pq = pq.replace(to_remove=vn, to_put=(vn.cost_f, vn))

        print('is_goal_reached: ' + str(is_goal_reached))

        if is_goal_reached:
            for i in range(1, len(path)):
                self.draw_lines(ax=ax, from_vertex=path[i - 1], to_vertices=[path[i]], color='red', linewidth=2)

        cost_table.reverse()
        for cost_element in cost_table:
            print(cost_element)

        return cost_path

    @staticmethod
    def draw_path_attempt(ax: Axes, vertex: HashVertex, color='white', linewidth=0.5):
        if vertex.previous is not None:
            ax.plot([vertex.previous.x, vertex.x], [vertex.previous.y, vertex.y], color=color, linewidth=linewidth)

    @staticmethod
    def draw_lines(ax: Axes, from_vertex: HashVertex, to_vertices: List[HashVertex], color=None, linewidth=0.5):
        lines = [([to_vertex.x, from_vertex.x], [to_vertex.y, from_vertex.y]) for to_vertex in to_vertices]

        for line in lines:
            xs, ys = line
            ax.plot(xs, ys, color=color if color is not None else 'white', linewidth=linewidth)


class CrimeMap:
    _shape_file: str
    _delta: float
    _threshold_percent: float
    _threshold_index: int
    _threshold_value: int
    _geo_data_frame: GeoDataFrame
    _points: List[Point]
    _graph: Graph

    def __init__(self, shape_file: str, delta: float, threshold_percent: float):
        self._shape_file = shape_file
        self._delta = delta
        self._threshold_percent = threshold_percent
        self._geo_data_frame = gp.read_file(shape_file)
        self._points = [v[3] for v in self._geo_data_frame.values]
        bounds = self._geo_data_frame.total_bounds
        self._graph = Graph(bounds=(bounds[0], bounds[1], bounds[2], bounds[3]), delta=delta)
        self._set_values_to_cells()
        self._threshold_index, self._threshold_value = self._compute_thresholds_and_set_is_block_to_cells()

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def threshold(self) -> float:
        return self._threshold_percent

    @property
    def points(self) -> List[Point]:
        return self._points

    @property
    def graph(self):
        return self._graph

    @property
    def threshold_index(self):
        return self._threshold_index

    @property
    def threshold_value(self):
        return self._threshold_value

    def _set_values_to_cells(self):
        for point in self.points:
            xi = min(int(np.floor((point.x - self.graph.x_min) / self.delta)), self.graph.nx - 1)
            yi = min(int(np.floor((point.y - self.graph.y_min) / self.delta)), self.graph.ny - 1)
            self.graph.cells[self.graph.ij2k(xi, yi)].value += 1

    def _compute_thresholds_and_set_is_block_to_cells(self):
        cells_reverse_sorted: List[HashCell] = sorted(self.graph.cells, key=lambda c: c.value, reverse=True)
        threshold_index: int = int(len(cells_reverse_sorted) * (100 - self.threshold) / 100)
        threshold_value: int = 0

        for cell in self.graph.cells:
            if self.threshold == 100 or threshold_index == 0:
                threshold_value = max([cell.value for cell in self.graph.cells]) + 1
                cell.is_block = False
            elif self.threshold == 0 or threshold_index == len(cells_reverse_sorted) - 1:
                threshold_value = min([cell.value for cell in self.graph.cells])
                cell.is_block = True
            else:
                threshold_value = cells_reverse_sorted[0:threshold_index][-1].value
                cell.is_block = cell.value >= threshold_value

            cell.norm_value = 1 if cell.is_block else 0

        return threshold_index, threshold_value

    def plot(self, ax: Axes) -> Axes:
        cmap = plt.cm.jet
        cmaplist = ['grey', 'black']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        gdf = gp.GeoDataFrame({'values': [cell.norm_value for cell in self.graph.cells],
                               'colors': ['black' if cell.is_block else 'white' for cell in self.graph.cells]},
                              geometry=[cell.polygon for cell in self.graph.cells])

        ax: Axes = gdf.plot(color='black') if all([cell.is_block for cell in self.graph.cells]) else gdf.plot(
            column='values', cmap=cmap, ax=ax)

        for cell in self.graph.cells:
            ax.text(cell.centroid.x, cell.centroid.y, str(cell.value),
                    fontdict=dict(color='white' if cell.is_block else 'black', fontsize=3, ha='center', va='center'))

        x_ticks = [x for (x, i) in zip(self.graph.xs, np.arange(0, self.graph.nx, 1)) if i % 2 == 0]
        y_ticks = [y for (y, i) in zip(self.graph.ys, np.arange(0, self.graph.ny, 1)) if i % 2 == 0]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['{:.3f}'.format(x_tick) for x_tick in x_ticks], fontdict=dict(fontsize=4))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.3f}'.format(y_tick) for y_tick in y_ticks], fontdict=dict(fontsize=4))
        ax.set_title(
            'τ={threshold_value} ({threshold}%), σ={std_dev}, μ={avg}, '
            'δ={delta}'.format(
                std_dev='{:.2f}'.format(np.std([cell.value for cell in self.graph.cells])),
                avg='{:.2f}'.format(np.average([cell.value for cell in self.graph.cells])),
                delta='{:.3f}'.format(self.delta),
                threshold=str(self.threshold),
                threshold_value=self.threshold_value
            ),
            pad=10,
            fontdict=dict(fontsize=8))

        return ax


class Main:
    fig: Figure
    fig_controls: Figure
    ax: Axes
    ax_controls: Axes
    crime_map: CrimeMap
    start: Optional[Point]
    goal: Optional[Point]
    _path_handles: List[Line2D]
    _should_set_start_next: bool
    _delta: float
    _threshold_percent: float

    def __init__(self):
        self.root = tkinter.Tk()
        self.start = None
        self.goal = None
        self._should_set_start_next = True
        self._delta = 0.002
        self._threshold_percent = 50

    def run(self):
        # plt.ion()
        # self.fig, (self.ax, self.ax_controls) = plt.subplots(1, 2)
        self.fig = plt.figure(figsize=(5, 3))
        # self.fig_controls = plt.figure(figsize=(2, 2))
        self.ax = plt.subplot2grid((1, 4), (0, 0), colspan=3, fig=self.fig)
        # self.ax_controls = self.fig_controls.gca()  # plt.subplot2grid((1, 3), (0, 2), fig=self.fig)

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self._draw_map()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self._onclick)

        self.sv1 = tkinter.StringVar(value='0.002')
        self.sv2 = tkinter.StringVar(value='50')

        l1 = tkinter.Label(self.root, text='Width:', width=8)
        l1.pack(side=tkinter.LEFT)
        e1 = tkinter.Entry(self.root, textvariable=self.sv1, width=10)
        e1.pack(side=tkinter.LEFT)
        self.e1 = e1

        l2 = tkinter.Label(self.root, text='Threshold%:', width=12)
        l2.pack(side=tkinter.LEFT, padx=10)
        e2 = tkinter.Entry(self.root, textvariable=self.sv2, width=5)
        e2.pack(side=tkinter.LEFT)
        self.e2 = e2

        e1.bind('<Return>', self._onsubmit_delta)
        e2.bind('<Return>', self._onsubmit_threshold_percent)

        self.root.mainloop()

    def _draw_map(self):
        self.crime_map = CrimeMap(shape_file="./Shape/crime_dt.shp",
                                  delta=self._delta,
                                  threshold_percent=self._threshold_percent)
        self.crime_map.plot(self.ax)
        # self.canvas.draw_idle()
        self.fig.canvas.draw_idle()

    def _draw_marker(self, point: Point, is_start: bool):
        self.ax.plot([point.x], [point.y],
                     marker='o' if is_start else '*',
                     color='green' if is_start else 'yellow', zorder=10)
        self.fig.canvas.draw_idle()

    def _onclick(self, event):
        x, y = event.xdata, event.ydata

        if x is not None and y is not None and event.inaxes in [self.ax]:
            if self._should_set_start_next:
                self.ax.lines.clear()
                self.start = self.crime_map.graph.closest_point(Point(x, y))
                self._should_set_start_next = False
                self._draw_marker(self.start, True)
            else:
                self.goal = self.crime_map.graph.closest_point(Point(x, y))
                self._draw_marker(self.goal, False)

                self._should_set_start_next = True
                cost_path = self.crime_map.graph.search(ax=self.ax, start=self.start, end=self.goal)
                self.ax.set_title(
                    self.ax.get_title() + ', f=' + ('{:.2f}'.format(cost_path) if not cost_path == float('Inf') else '∞'),
                    fontdict=dict(fontsize=self.ax.title.get_fontsize()))
                self.fig.canvas.draw_idle()

    def _onsubmit_delta(self, event):
        print('delta')
        self._delta = float(self.sv1.get())
        self.ax.clear()
        self._draw_map()

    def _onsubmit_threshold_percent(self, event):
        print('threshold')
        self._threshold_percent = float(self.sv2.get())
        self.ax.clear()
        self._draw_map()

    def _ontextchange(self, event):
        self.tb_delta.text_disp.set_fontsize(6)
        self.tb_threshold.text_disp.set_fontsize(6)

    def _init_text_boxes(self):
        matplotlib.rcParams.update({'font.size': 6})

        ax_delta = self.fig.add_axes([0.8, 0.5, 0.1, 0.05])
        self.tb_delta = TextBox(ax_delta, 'Width', initial=str(self._delta))
        self.tb_delta.on_submit(self._onsubmit_delta)

        ax_threshold = self.fig.add_axes([0.8, 0.4, 0.1, 0.05])
        self.tb_threshold = TextBox(ax_threshold, 'Threshold%', initial=str(self._threshold_percent))
        self.tb_threshold.on_submit(self._onsubmit_threshold_percent)

        # self.tb_delta.label.set_fontsize(6)
        # self.tb_delta.text_disp.set_fontsize(6)
        # # self.tb_delta.connect_event('key_press_event', self._ontextchange)
        # self.tb_threshold.label.set_fontsize(6)
        # self.tb_threshold.text_disp.set_fontsize(6)
        # self.tb_threshold.connect_event('key_press_event', self._ontextchange)


if __name__ == '__main__':
    Main().run()
