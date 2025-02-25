import numpy as np
import casadi as cs
from . import feature as feature
import pyglet.gl as gl
import pyglet.graphics as graphics

class Lane(object):
    pass
# 虽然当前Lane类为空，但它预留了扩展的空间。如果后续需要为所有车道类型添加通用的属性或方法，
# 可以直接在Lane类中进行添加，那么所有继承自Lane的子类（如StraightLane）都会自动继承这些新的属性和方法。

class StraightLane(Lane):
    """
    A class used to represent a straight lane

    Attributes
    ----------
    p : list
        the first point on the center line of the lane
    q : list
        the second point on the center line of the lane
    w : float
        the width of the lane
    m : list
        the vector along the straight lane，沿直车道的向量
    n : list
        the normal vector on the straight lane，直车道的法向量
    """
    def __init__(self, p, q, w):
        """
        Parameters
        ----------
        p : list
            the first point on the center line of the lane
        q : list
            the second point on the center line of the lane
        w : float
            the width of the lane
        """
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
        self.length = 1000

    def shifted(self, m):
        """ Returns a new lane with position shifted by m times the width of the lane relatively

        Parameters
        ----------
        m : int
            the amount of times to shift the lane
        """
        return StraightLane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)

    def dist2(self, x):
        """ Returns the squared distance of a point p from the center line of the lane

        Parameters
        ----------
        x : list
            the position of the point p
        """
        r = (x[0] - self.p[0]) * self.n[0] + (x[1] - self.p[1]) * self.n[1]
        return r * r

    def quadratic(self):
        """ Returns a quadratic cost feature penalizing deviations from the center line of the lane """
        @feature.feature
        def f(x, u, x_other, k):
            return self.dist2(x)
        return f

    def linear(self, id_other_vehicle=None):
        """ Returns a linear cost feature penalizing driving along the normal vector """
        if id_other_vehicle is None:
            @feature.feature
            def f(x, u, x_other, k):
                return (x[0] - (self.p - self.n * self.w / 2)[0]) * self.n[0] + \
                       (x[1] - (self.p - self.n * self.w / 2)[1]) * self.n[1]
        else:
            @feature.feature
            def f(x, u, x_other, k):
                return (x_other[id_other_vehicle][0]-(self.p - self.n * self.w / 2)[0])*self.n[0] + \
                       (x_other[id_other_vehicle][1]-(self.p - self.n * self.w / 2)[1])*self.n[1]
        return f

    def get_edges(self):
        """ Returns a point on an edge of the lane and the corresponding nearest point on the other edge of the lane """
        return self.p - self.n * self.w / 2, self.p + self.n * self.w / 2

    def gaussian(self, width=0.5):
        """ Returns a gaussian cost feature penalizing deviations from the center line of the lane
        返回一个高斯代价特征函数，用于惩罚车辆偏离车道中心线的行为，width 参数控制高斯函数的宽度。
        Parameters
        ----------
        width : float
            the width of the gaussian
        """
        @feature.feature
        def f(x, u, x_other, k):
            return cs.exp(-0.5*self.dist2(x)/(width**2*self.w*self.w/4.))
        return f

    def draw_lane_surface(self):
        """ Draws the surface of this lane 绘制车道的表面，使用 OpenGL 相关函数设置颜色并绘制四边形条带。"""
        gl.glColor3f(0.3, 0.3, 0.3)
        graphics.draw(4, gl.GL_QUAD_STRIP,
                      ('v2f', np.hstack([self.p-self.m*self.length-0.55*self.w*self.n,
                                         self.p-self.m*self.length+0.55*self.w*self.n,
                                         self.p+self.m*self.length-0.55*self.w*self.n,
                                         self.p+self.m*self.length+0.55*self.w*self.n])))

    def draw_simple_lane_lines(self, line_width):
        """ Draws simple white lines for this lane 绘制简单的白色车道线，使用 OpenGL 相关函数设置颜色和线宽，并绘制直线。"""
        gl.glColor3f(1., 1., 1.)
        gl.glLineWidth(1 * line_width)
        graphics.draw(2, gl.GL_LINES,
                      ('v2f', np.hstack([self.p - self.m*self.length-0.5*self.w*self.n,
                                         self.p + self.m*self.length-0.5*self.w*self.n])))
        graphics.draw(2, gl.GL_LINES,
                      ('v2f', np.hstack([self.p + self.m*self.length+0.5*self.w*self.n,
                                         self.p - self.m*self.length+0.5*self.w*self.n])))
