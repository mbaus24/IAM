close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% representation de la fonction a optimiser
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hold on;
plot(-20:0.01:20,forme_quad(-20:0.01:20));
title('fonction a optimiser');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialisations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xk = 10;
k       = 1;
pas = 0.5;
pas_opt=(xk)^2/(xk*2*xk);

MS=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%estime initial croix noire
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(xk,forme_quad(xk),'k+','markersize', MS);
   legend('fonction a optimiser','initialisation') 

erreur      = 0.001;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%iteration 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xkk      = xk -pas *2*xk; 

while k <= 1000 && abs(xkk-xk) > erreur
    xk = xkk;
    xkk = xk - pas *2*xk;%pas constant
    
    %xkk = xk - pas_opt *2*xk;%pas optimal
    
  
    %xkk = xk - 2*xk/2; %newton 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %representation des iterations successives points bleus
    plot(xkk,forme_quad(xkk),'b.','markersize', MS);
    
   
    
    k = k + 1;
end
plot(xkk,forme_quad(xkk),'ro','markersize', MS);
legend('fonction a optimiser','initialisation','iterations')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% affichage des resultats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iterations = k-2
extrema = xkk