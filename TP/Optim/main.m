% main.m
% Sample MATLAB code

% Clear workspace and command window
clear;
clc;

% Define a sample function
f = @(x) x.^2 - 4*x + 4;

% Define the range for x
x = linspace(-10, 10, 100);

% Calculate the function values
y = f(x);

% Plot the function
figure;
plot(x, y, 'LineWidth', 2);
title('Sample Function Plot');
xlabel('x');
ylabel('f(x)');
grid on;