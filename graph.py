from queue import PriorityQueue
from time import time
from typing import Tuple, Optional, List, Set, Any

from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from shapely.geometry import Polygon, Point
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({'figure.dpi': 350})
matplotlib.use("TkAgg")


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
        self.cost_ancestor_pairs: List[Tuple[float, 'HashVertex']] = []

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

                # # top_right_vertex
                # cells[k].top_right_vertex.top = cells[k].top.top_right_vertex
                # cells[k].top_right_vertex.right = cells[k].right.top_right_vertex
                # cells[k].top_right_vertex.bottom = cells[k].bottom_right_vertex
                # cells[k].top_right_vertex.left = cells[k].top_left_vertex
                # cells[k].top_right_vertex.top_right = cells[k].top.right.top_right_vertex
                # cells[k].top_right_vertex.bottom_right = cells[k].bottom.right.bottom_right_vertex
                # cells[k].top_right_vertex.bottom_left = cells[k].bottom_left_vertex
                # cells[k].top_right_vertex.top_left = cells[k].top.top_left_vertex
                #
                # # bottom_right_vertex
                # cells[k].bottom_right_vertex.top = cells[k].top_right_vertex
                # cells[k].bottom_right_vertex.right = cells[k].right.bottom_right_vertex
                # cells[k].bottom_right_vertex.bottom = cells[k].bottom.bottom_right_vertex
                # cells[k].bottom_right_vertex.left = cells[k].bottom_left_vertex
                # cells[k].bottom_right_vertex.top_right = cells[k].right.top_right_vertex
                # cells[k].bottom_right_vertex.bottom_right = cells[k].bottom.right.bottom_right_vertex
                # cells[k].bottom_right_vertex.bottom_left = cells[k].bottom.bottom_left_vertex
                # cells[k].bottom_right_vertex.top_left = cells[k].top_left_vertex
                #
                # # bottom_left_vertex
                # cells[k].bottom_left_vertex.top = cells[k].top_left_vertex
                # cells[k].bottom_left_vertex.right = cells[k].bottom_right_vertex
                # cells[k].bottom_left_vertex.bottom = cells[k].bottom.bottom_left_vertex
                # cells[k].bottom_left_vertex.left = cells[k].left.bottom_left_vertex
                # cells[k].bottom_left_vertex.top_right = cells[k].top_right_vertex
                # cells[k].bottom_left_vertex.bottom_right = cells[k].bottom.bottom_right_vertex
                # cells[k].bottom_left_vertex.bottom_left = cells[k].bottom.left.bottom_left_vertex
                # cells[k].bottom_left_vertex.top_left = cells[k].left.top_left_vertex
                #
                # # top_left_vertex
                # cells[k].top_left_vertex.top = cells[k].top.top_left_vertex
                # cells[k].top_left_vertex.right = cells[k].top_right_vertex
                # cells[k].top_left_vertex.bottom = cells[k].bottom_left_vertex
                # cells[k].top_left_vertex.left = cells[k].left.top_left_vertex
                # cells[k].top_left_vertex.top_right = cells[k].top.top_right_vertex
                # cells[k].top_left_vertex.bottom_right = cells[k].bottom_right_vertex
                # cells[k].top_left_vertex.bottom_left = cells[k].left.bottom_left_vertex
                # cells[k].top_left_vertex.top_left = cells[k].top.left.top_left_vertex

                # ---------------------

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

                # print('hey')

        return cells, vertices

    def closest_indices(self, point: Point) -> Tuple[int, int]:
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

    def h(self, a: HashVertex, b: HashVertex):
        return a.distance(b)

    @staticmethod
    def scale() -> float:
        return 1

    def search(self, ax: Axes, start: Point, end: Point):
        pq: VertexSearchQueue[Tuple[float, HashVertex]] = VertexSearchQueue()
        is_goal_reached: bool = False
        vi = self.closest_vertex(start)
        vf = self.closest_vertex(end)
        visited: Set[HashVertex] = set()
        path: List[HashVertex] = []

        self.draw_markers(ax=ax, start=vi, end=vf)

        pq.put((0, vi))

        while not pq.empty():
            cost_v: float
            v: HashVertex
            cost_v, v = pq.get()

            if v == vf:
                is_goal_reached = True
                path.append(v)
                # while not len(v.cost_ancestor_pairs) == 0:
                #     vas = [vertex for (_, vertex) in v.cost_ancestor_pairs]
                #     cas = [cost for (cost, _) in v.cost_ancestor_pairs]
                #     v = vas[cas.index(min(cas))]
                #     path.append(v)

                while v.previous is not None:
                    path.append(v.previous)
                    v = v.previous

                path.reverse()
                break

            visited.add(v)

            cost_neighbour_pairs = self.get_cost_neighbour_pairs(v)
            self.draw_lines(ax=ax, from_vertex=v, to_vertices=[neighbour for (_, neighbour) in cost_neighbour_pairs])

            for cost_neighbour_pair in cost_neighbour_pairs:
                (g, vn) = cost_neighbour_pair
                cost_vn = cost_v + g + self.scale() * self.h(v, vf)

                queue_vertices = pq.vertices()
                print([_c for (_c, _v) in pq.queue])

                if vn not in visited and vn not in queue_vertices:
                    vn.previous = v
                    vn.cost_ancestor_pairs.append((cost_v, v))
                    pq.put((cost_vn, vn))
                elif vn in queue_vertices:
                    (cost_vn_same, vn_same) = pq.find(vn)

                    # replace lower cost vertex
                    if cost_vn_same > cost_vn:
                        vn.previous = v
                        vn.cost_ancestor_pairs.append((cost_v, v))
                        pq = pq.replace(to_remove=vn, to_put=(cost_vn, vn))

        print('is_goal_reached: ' + str(is_goal_reached))

        if is_goal_reached:
            for i in range(1, len(path)):
                self.draw_lines(ax=ax, from_vertex=path[i - 1], to_vertices=[path[i]], color='black')

    @staticmethod
    def draw_markers(ax: Axes, start: HashVertex, end: HashVertex):
        ci = plt.Circle(xy=(start.x, start.y), radius=0.0002, color='green')
        cf = plt.Circle(xy=(end.x, end.y), radius=0.0002, color='green')
        ax.add_artist(ci)
        ax.add_artist(cf)

    @staticmethod
    def draw_lines(ax: Axes, from_vertex: HashVertex, to_vertices: List[HashVertex], color=None):
        lines = [([to_vertex.x, from_vertex.x], [to_vertex.y, from_vertex.y]) for to_vertex in to_vertices]

        for line in lines:
            xs, ys = line
            ax.plot(xs, ys, color=color if color is not None else 'white')
            # fig: Figure = ax.get_figure()
            # plt.draw()  # re-draw the figure
            # plt.pause(0.000000000001)


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
        self._graph = Graph(bounds=(bounds[0], bounds[1], bounds[2], bounds[3]), delta=0.002)
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

    def plot(self) -> Axes:
        gdf = gp.GeoDataFrame({'values': [cell.norm_value for cell in self.graph.cells],
                               'colors': ['yellow' if cell.is_block else 'purple' for cell in self.graph.cells]},
                              geometry=[cell.polygon for cell in self.graph.cells])

        ax: Axes = gdf.plot(color='yellow') if all([cell.is_block for cell in self.graph.cells]) else gdf.plot(
            column='values', cmap='viridis')

        for cell in self.graph.cells:
            plt.text(cell.centroid.x, cell.centroid.y, str(cell.value),
                     fontdict=dict(color='black' if cell.is_block else 'white', fontsize=3, ha='center', va='center'))

        x_ticks = [x for (x, i) in zip(self.graph.xs, np.arange(0, self.graph.nx, 1)) if i % 2 == 0]
        y_ticks = [y for (y, i) in zip(self.graph.ys, np.arange(0, self.graph.ny, 1)) if i % 2 == 0]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(['{:.3f}'.format(x_tick) for x_tick in x_ticks], fontdict=dict(fontsize=4))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.3f}'.format(y_tick) for y_tick in y_ticks], fontdict=dict(fontsize=4))
        ax.set_title(
            'threshold_value={threshold_value}, std_dev={std_dev}, avg={avg}, '
            'delta={delta}, threshold={threshold}%'.format(
                std_dev='{:.2f}'.format(np.std([cell.value for cell in self.graph.cells])),
                avg='{:.2f}'.format(np.average([cell.value for cell in self.graph.cells])),
                delta='{:.3f}'.format(self.delta),
                threshold=str(self.threshold),
                threshold_value=self.threshold_value
            ),
            pad=10,
            fontdict=dict(fontsize=8))

        # def onclick(event):
        #     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #           ('double' if event.dblclick else 'single', event.button,
        #            event.x, event.y, event.xdata, event.ydata))
        #
        # cid = ax.get_figure().canvas.mpl_connect('button_press_event', onclick)

        plt.draw()

        return ax


def main():
    plt.ion()
    start_time = time()
    crime_map: CrimeMap = CrimeMap(shape_file="./Shape/crime_dt.shp", delta=0.002, threshold_percent=65)
    ax: Axes = crime_map.plot()
    # crime_map.graph.search(ax=ax, start=Point(-73.588, 45.494), end=Point(-73.555, 45.514))
    crime_map.graph.search(ax=ax, start=Point(-73.588, 45.494), end=Point(-73.562, 45.506))
    end_time = time()
    exec_time = end_time - start_time
    print('exec_time: ' + str(exec_time))
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
