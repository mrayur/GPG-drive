import casadi as cs

class Dynamics(object):
    #在 Python 中，class 继承自 object 是一种显式声明，表示该类是一个新式类（new-style class）。
    # 在 Python 2 中，类可以选择继承 object 或不继承（这会创建老式类）。
    # 而从 Python 3 开始，所有类默认都是新式类，即所有类都隐式地继承自 object，即使没有显式写出来。
    """
    A class used to represent the general dynamics of a vehicle
    通过一个通用的框架来表示车辆的状态演变，并根据输入和状态计算车辆下一时刻的状态。
    Attributes
    ----------
    nx : int
        the number of states
    nu : int
        the number of inputs
    dt : float
        the sampling time
    bounds : list
        the control bounds of the vehicle
    lr : float
        the length between the mass center and the rear end
    lf : float
        the length between the mass center and the front end
    width : float
        the width of the vehicle
    f : function
        the dynamics of the vehicle
    """
    def __init__(self, nx, nu, f, dt=0.25, bounds=None, lr=0, lf=0, width=0, use_rk4=True):
        #8个属性定义
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.bounds = bounds
        self.lr = lr
        self.lf = lf
        self.width = width
        self.f = f
        if use_rk4:
            def f_discrete(x,u):
                k1 = f(x, u)
                k2 = f(x + k1 * (self.dt / 2), u)
                k3 = f(x + k2 * (self.dt / 2), u)
                k4 = f(x + k3 * self.dt, u)
                return x + (1/6) * self.dt * (k1 + 2*k2 + 2*k3 + k4)
        else:
            def f_discrete(x,u):
                k1 = f(x, u)
                k2 = f(x + k1 * (self.dt / 2), u)
                return x + (1/2) * self.dt * (k1 + k2)
            
            # def f_discrete(x,u):
            #     return x + self.dt * f(x,u)
        self.f_discrete = f_discrete

    def __call__(self, x, u):
        return self.f_discrete(x, u)
    #__call__ 方法是一个特殊方法，允许你让一个类的实例对象像函数一样使用，自动执行并返回类中指定函数运算结果
    # 创建 Dynamics 类实例      dynamics = Dynamics(nx=4, nu=2, f=f)
    # 通过()直接调用实例来获得结果，result = dynamics(x, u)  # 相当于调用 dynamics.__call__(x, u)
    #                                vs
    # 创建 Dynamics 类实例          dynamics = Dynamics(nx=4, nu=2, f=f)
    # 必须通过 `calculate` 方法来调用 result = dynamics.calculate(x, u)
    # 总结：省了函数"calculate"的书写

class CarDynamics(Dynamics): #明确表示 CarDynamics 继承自 Dynamics，因此 CarDynamics 类可以访问 Dynamics 类中的所有属性和方法。
    """
    A class used to represent the dynamics of a vehicle using the kinematic bicycle model
    """
    def __init__(self, dt=0.25, bounds=None, friction=0., lr=4, lf=0, width=2, use_rk4=True):
        """
        Parameters
        ----------
        dt : float, optional
            the sampling time
        bounds : list, optional
            the control bounds of the vehicle
        friction : float, optional
            the friction of the kinematic vehicle model
        lr : float, optional
            the length between the mass center and the rear end
        lf : float, optional
            the length between the mass center and the front end
        width : float, optional
            the width of the vehicle
        """
        self.friction = friction
        if bounds is None:
            bounds = [[-2., -0.5], [2., 0.5]]

        def f(x, u):
            if lf == 0:
                beta = u[1]
            else:
                beta = cs.arctan(lr / (lf + lr) * cs.tan(u[1]))
            if isinstance(x, cs.SX) or isinstance(x, cs.MX):
                # 这行检查状态向量 x 是否是 CasADi 中的符号变量（cs.SX 或 cs.MX）。
                # 这些符号变量通常用于优化问题或符号计算。如果是符号变量，就用 CasADi 的方式来处理计算。
                x_next = cs.SX(4, 1)
                x_next[0] = x[3] * cs.cos(x[2] + beta)
                x_next[1] = x[3] * cs.sin(x[2] + beta)
                x_next[2] = (x[3] / lr) * cs.sin(beta)
                x_next[3] = u[0] - x[3] * friction
                return x_next
            else:
                return cs.vertcat(x[3] * cs.cos(x[2] + beta), x[3] * cs.sin(x[2] + beta), (x[3] / lr) * cs.sin(beta),
                                u[0] - x[3] * friction)
        Dynamics.__init__(self, 4, 2, f, dt, bounds, lr, lf, width, use_rk4)

    # def __getstate__(self):
    #     return self.dt, self.bounds, self.friction, self.lr, self.lf, self.width

    # def __setstate__(self, dt, bounds, friction, lr, lf, width):
    #     def f(x, u):
    #         beta = cs.arctan(lr / (lf + lr) * cs.tan(u[1]))
    #         if isinstance(x, cs.SX) or isinstance(x, cs.MX):
    #             x_next = cs.SX(4, 1)
    #             x_next[0] = x[3] * cs.cos(x[2] + beta)
    #             x_next[1] = x[3] * cs.sin(x[2] + beta)
    #             x_next[2] = (x[3] / lr) * cs.sin(beta)
    #             x_next[3] = u[0] - x[3] * friction
    #             return x_next
    #         else:
    #             return np.array([x[3] * cs.cos(x[2] + beta), x[3] * cs.sin(x[2] + beta), (x[3] / lr) * cs.sin(beta),
    #                             u[0] - x[3] * friction])
    #     Dynamics.__init__(self, 4, 2, f, dt, bounds, lr, lf, width)


class CarDynamicsLongitudinal(Dynamics):
    """
    A class used to represent the dynamics of a vehicle using the one-dimensional, longitudinal model
    """
    def __init__(self, dt=0.25, bounds=None, friction=0., lr=4, lf=0, width=2, use_rk4=True):
        """
        Parameters
        ----------
        dt : float, optional
            the sampling time
        bounds : list, optional
            the control bounds of the vehicle
        friction : float, optional
            the friction of the kinematic vehicle model
        lr : float, optional
            the length between the mass center and the rear end
        lf : float, optional
            the length between the mass center and the front end
        width : float, optional
            the width of the vehicle
        """
        self.friction = friction
        if bounds is None:
            bounds = [[-2.], [2.]]

        # Hacky way to get parametric bounds in OpEn by substituting u = U_{min}*v + (1+v)/2*U_{max} where v \in [-1,1]
        # Currently OpEn only supports u \in U and not u \in U(p)
        if any([isinstance(bound[0], cs.SX) for bound in bounds]) or any([isinstance(bound[0], cs.MX) for bound in bounds]):
            self.parametric_bounds = bounds
            def f(x, v):
                u0 = self.parametric_bounds[0][0] + (1+v[0])/2*(self.parametric_bounds[1][0] - self.parametric_bounds[0][0])
                x_next = cs.SX(4, 1)
                x_next[0] = x[3] * cs.cos(x[2])
                x_next[1] = x[3] * cs.sin(x[2])
                x_next[2] = 0
                x_next[3] = u0 - x[3] * friction
                return x_next
            bounds = [[-1.], [1.]]
        else:
            def f(x, u):
                if isinstance(x, cs.SX) or isinstance(x, cs.MX):
                    x_next = cs.SX(4, 1)
                    x_next[0] = x[3] #x[3] * cs.cos(x[2])
                    x_next[1] = 0 #x[3] * cs.sin(x[2])
                    x_next[2] = 0
                    x_next[3] = u[0] - x[3] * friction
                    return x_next
                else:
                    return cs.vertcat(x[3]*cs.cos(x[2]), x[3]*cs.sin(x[2]), 0, u[0]-x[3]*friction)
        Dynamics.__init__(self, 4, 1, f, dt, bounds, lr, lf, width, use_rk4)

    # def __getstate__(self):
    #     return self.dt, self.bounds, self.friction, self.lr, self.lf, self.width

    # def __setstate__(self, dt, bounds, friction, lr, lf, width):
    #     def f(x, u):
    #         if isinstance(x, cs.SX) or isinstance(x, cs.MX):
    #             x_next = cs.SX(4, 1)
    #             x_next[0] = x[3] * cs.cos(x[2])
    #             x_next[1] = x[3] * cs.sin(x[2])
    #             x_next[2] = 0
    #             x_next[3] = u[0] - x[3] * friction
    #             return x_next
    #         else:
    #             return np.array([x[3]*cs.cos(x[2]), x[3]*cs.sin(x[2]), 0, u[0]-x[3]*friction])
    #     Dynamics.__init__(self, 4, 1, f, dt, bounds, lr, lf, width)

