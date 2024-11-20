function [a] = rosenbrock_diff(x, y)
a = [-400*x*(y-x.^2)-2*(1-x); 200*(y-x.^2)];