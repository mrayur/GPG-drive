import casadi as cs
from . import feature as feature
from . import lane as lane
from . import constraints as constraints
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
from pyglet import shapes, sprite
from .helpers.tree_generator import generate_trees, generate_trees_Gabor
from .helpers.visualize_helpers import centered_image


class Scene:
    #通常，构造函数不需要显式地返回值，因为它的主要目的是初始化对象的属性。这里的 return 语句实际上是多余的，它会隐式地返回 None。
    def __init__(self):
        return
    #目前这个 Scene 类的定义只是一个基本的框架，构造函数和 draw 方法都没有实际的功能。在后续开发中，你可以在这些方法中添加具体的代码来实现相应的功能
    def draw(self, magnify):
        return

class Highway(Scene):
    """
    A class used to represent a highway with multiple lanes

    Attributes
    ----------
    lanes : list
        the list of Lane objects of the highway
    n : list
        the normal vector on the highway direction
    """
    def __init__(self, p, q, w, nb_lanes, length_list=None):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the 'first' lane
        q : list
            the second point on the center line of the 'first' lane
        w : float
            the width of each lane
        nb_lanes : int
            the number of lanes of the highway
        length_list : list
            the lengths of the different lanes
        """
        center_lane = lane.StraightLane(p, q, w)
        self.lanes = [center_lane]
        for n in range(1, nb_lanes):
            self.lanes += [center_lane.shifted(n)]
        self.n = self.lanes[0].n
        if length_list is not None:
            assert(len(length_list) == nb_lanes)
            for i, L in enumerate(length_list):
                self.lanes[i].length = L
        self.setup_trees()

        
    def setup_trees(self): #设置高速公路上的树木。通过定义森林区域的判断函数，生成树木的位置，创建树木的精灵对象，并设置它们的位置、大小等属性。
        def is_forest_zone(position):
            distances = np.vstack(self.boundary_distances()(position))
            return np.min(distances) < -2.5
        self.tree_locations = generate_trees_Gabor(100, 200, 10, radius=4.5, valid_check=is_forest_zone)
        self.tree_batch = graphics.Batch()
        self.tree_sprites = []
        for tree_location in self.tree_locations:
            index = np.random.randint(1,5)
            self.tree_sprites.append(sprite.Sprite(centered_image('GPGdrive/images/Trees/Treedark{}.png'.format(index)), subpixel=True, batch=self.tree_batch))
            tree_size = np.random.uniform(3,6)
            self.tree_sprites[-1].scale = tree_size/256
            self.tree_sprites[-1].x, self.tree_sprites[-1].y = tree_location
            # sprite.rotation = -x[2]*180./math.pi
            # sprite.opacity = opacity

    def get_lanes(self):
        """ Return the number of lanes of the highway """
        return self.lanes

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane

        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].gaussian(width)(x, u, x_other)
        return f

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].quadratic()(x, u, x_other, k)
        return f

    def linear(self):
        """ Returns a linear cost feature penalizing driving along the normal vector """
        @feature.feature
        def f(x, u, x_other, k):
            return self.lanes[0].linear()(x, u, x_other)
        return f

    def boundary_distances(self):
        """ Returns the distance to the edges of the road

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def h(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            h = []
            h.append((x[0] - edge1[0]) * self.lanes[0].n[0] + (x[1] - edge1[1]) * self.lanes[0].n[1])
            h.append(- (x[0] - edge2[0]) * self.lanes[-1].n[0] - (x[1] - edge2[1]) * self.lanes[-1].n[1])
            return h
        h.length = 2
        h.type = "inequality"
        return h


    def boundary_constraint(self, car, *args):
        """ Returns the 8 inequality boundary constraints ensuring the car remains withing the boundaries of the highway

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def h(x, u=None):
            edge1 = self.lanes[0].get_edges()[0]
            edge2 = self.lanes[-1].get_edges()[1]
            vehicle_corners = car.corners_x(x[car.id])
            h = []
            for corner in vehicle_corners:
                h.append((corner[0] - edge1[0]) * self.lanes[0].n[0] + (corner[1] - edge1[1]) * self.lanes[0].n[1])
                h.append(- (corner[0] - edge2[0]) * self.lanes[-1].n[0] - (corner[1] - edge2[1]) * self.lanes[-1].n[1])
            return h
        h.length = 8
        h.type = "inequality"
        h.id_list = (car.id,)
        return h

    def right_lane_constraint(self, car):
        """ Returns the equality constraint ensuring the car remains withing the boundaries of lane zero

        Parameters
        ----------
        car : Car object
            the ego vehicle
        """
        @constraints.stageconstraints
        def g(x, u=None):
            upper_edge = self.lanes[0].get_edges()[1]
            vehicle_corners = car.corners_x(x[car.id])
            n = self.lanes[0].n
            m = self.lanes[0].m
            g = []
            for corner in vehicle_corners:
                h1 = - (corner[0] - upper_edge[0]) * n[0] - (corner[1] - upper_edge[1]) * n[1]
                h2 = - (corner[0] - self.lanes[1].length) * m[0] - (corner[1] - upper_edge[1]) * m[1]
                g.append(cs.fmin(h1, 0) * cs.fmin(h2, 0))
                # g.append(cs.fmax(cs.fmin(h1, 0), cs.fmin(h2, 0)))
            return g
        g.length = 4
        g.type = "equality"
        g.id_list = (car.id,)
        return g

    def aligned(self, factor=1.):
        """ Returns a quadratic cost feature penalizing deviations from driving along the direction of the highway

        Parameters
        ----------
        factor : float
            the cost feature importance
        """
        @feature.feature
        def f(x, u, x_other, k):
            return - factor * (x[2] - cs.arctan2(-self.n[0], self.n[1]))**2
        return f

    def draw(self, magnify):
        """ Draws the road

        Parameters
        ----------
        magnify : float
            the manification of the visualizer
        """
        for lane in self.lanes:
            lane.draw_lane_surface()
        gl.glLineWidth(1/magnify)

        def left_line(lane):
            return np.hstack([lane.p - lane.m * lane.length + 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        def right_line(lane):
            return np.hstack([lane.p - lane.m * lane.length - 0.5 * lane.w * lane.n,
                        lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n])

        def end_line(lane):
            return np.hstack([lane.p + lane.m * lane.length - 0.5 * lane.w * lane.n,
                               lane.p + lane.m * lane.length + 0.5 * lane.w * lane.n])

        def draw_dashed_line(line, number_of_sections=1000):
            x0, y0, x1, y1 = line
            xs=np.linspace(x0,x1,number_of_sections+1)
            ys=np.linspace(y0,y1,number_of_sections+1)
            # batch = graphics.Batch()

            for i in range(number_of_sections//4):
                # shapes.Line(xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1], width = 1, batch=batch)
                graphics.draw(2, gl.GL_LINES, ('v2f', [xs[4*i], ys[4*i], xs[4*i+1], ys[4*i+1]]))

            # batch.draw()


        if len(self.lanes) == 1:
            self.lanes[0].draw_simple_lane_lines(1/magnify)
        else:
            for k, lane in enumerate(self.lanes):
                if k == 0:
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', right_line(lane)))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                elif k == len(self.lanes)-1:
                    gl.glColor3f(1., 1., 1.)
                    draw_dashed_line(right_line(lane))

                    #Hard coded parking slots for merging experiment
                    W = 43
                    for i in range(1):
                        graphics.draw(4, gl.GL_LINE_LOOP, ('v2f',
                           np.hstack(
                               [lane.p + lane.m * (
                                           W + 6 * i) - 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                            W + 6 * i) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) + 0.5 * lane.w * lane.n,
                                lane.p + lane.m * (
                                        W + 6 * (i + 1)) - 0.5 * lane.w * lane.n])
                           ))

                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', left_line(lane)))
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
                else:
                    gl.glColor3f(1., 1., 1.)
                    draw_dashed_line(left_line(lane))
                    draw_dashed_line(right_line(lane))
                    gl.glColor3f(1., 1., 0.)
                    graphics.draw(2, gl.GL_LINES, ('v2f', end_line(lane)))
        gl.glColor3f(1., 1., 1.)

        self.tree_batch.draw()

