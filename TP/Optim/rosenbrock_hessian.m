function [a] = rosenbrock_hessian(x, y)
a = [1200*x.^2-400*y+2, -400*x; -400*x, 200];