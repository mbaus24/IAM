close all
clear all
MS=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialisation manuelle avec la souris


[x, y] = meshgrid(-2.2:0.1:2.2, -2.5:0.1:3);


hold on

mesh(x,y,rosenbrock(x,y));
title('choix de l estime initial');
[xk,yk]=ginput(1)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Representation courbes de niveau
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xk  = [xk; yk];


[xi, yi] = meshgrid(-2.2:0.1:2.2, -2.5:0.1:3);
contour(xi, yi, rosenbrock(xi, yi), 100);
hold on;
plot3(xk(1),xk(2),rosenbrock(xk(1),xk(2)),'+k','markersize', MS); %estimé initial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialisations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k  = 0;
mu = 0.000001;
pas=0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Introduction d'un xk+1 pour pouvoir rentrer dans la boucle while qui suit
%calcule par une methode de gradient à pas constant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xkk      = xk -pas* rosenbrock_diff(xk(1),xk(2));

tic;
while k <= 1000 && norm(xkk-xk) > mu 
    xk=xkk;
    h = rosenbrock_hessian(xk(1),xk(2)) \ rosenbrock_diff(xk(1),xk(2));
    xkk = xk - h;
    plot3(xkk(1),xkk(2),rosenbrock(xkk(1),xkk(2)),'b.','markersize', MS);
    
    k = k + 1;
end
toc;
plot3(xkk(1),xkk(2),rosenbrock(xkk(1),xkk(2)),'or','markersize', MS);

%affichage des résulats
iterations = k
extrema = xkk