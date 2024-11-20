close all
clear all
MS=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Programme qui optimise la fonction de Rosenblock 
%% par des methodes de gradient 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Discretisation de l intervalle d etude
figure(1);
[x, y] = meshgrid(-2.2:0.1:2.2, -1.5:0.1:5);


hold on

% Initialisation manuelle avec la souris
mesh(x,y,rosenbrock(x,y));
title('choix de l estime initial');
[xk,yk]=ginput(1)
hold off


%%Representation
figure(2);contour(x,y,rosenbrock(x,y),30);
title('fonction de Rosenbrock a optimiser');

hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%pas des algorithmes de gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pas constant
pas = 0.001;

%%%
%initialisation du pas optimal
pas_opt=0.01;

% iteration initiale
k = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Affichage de la fonction a optimiser et son iteration initiale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2);plot3(xk,yk,rosenbrock(xk,yk),'+k','markersize', MS);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dxk = -2+2*xk-400*(xk*yk-(xk).^3);	% calcul de la derivee par rapport ? x
dyk =     200*( yk-  (xk).^2);          		% calcul de la derivee par rapport ? y

%%%%%%%premiere iteration

xkk = xk - pas*dxk;
ykk = yk - pas*dyk;

erreur      = 0.001;
mu      = 0.000001;
solx=xk;
soly=yk;
x=[xk,yk];
xx=[xkk,ykk];
tic;

%%%%%%%%Critere d'arret des iterations
   while k <= 40000 && norm(x - xx) > mu
    % calcul de l itere suivant
xk = xkk;
yk = ykk;
%
dxk = -2+2*xk-400*(xk*yk-(xk).^3);	% calcul de la d�riv�e par rapport ? x
dyk =     200*( yk-  (xk).^2);

% %%%%%%%%%%%%%%%%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %xkk = xk - pas*dxk;
   %ykk = yk - pas*dyk; 


 
% %%%%%%%%%%%%%%%%%%%PAS OPTIMAL
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%


% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Recherche par golden section Matlab 
%recherche_pas=@(pas_opt)(rosenbrock(xk-pas_opt*dxk,yk-pas_opt*dyk));
%[pas_opt]=fminbnd(recherche_pas,0,2);


% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%Recherche � l'aide de la fonction fminsearch (utilisant une m�thode de
% %simplexe).
  recherche_pas=@(pas_opt)(rosenbrock(xk-pas_opt*dxk,yk-pas_opt*dyk)); 
  options = optimset('MaxFunEvals',10000,'MaxIter',10000);
  [pas_opt]=fminsearch(recherche_pas,0,options);
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      
%      
   xkk = xk - pas_opt*dxk;
   ykk = yk - pas_opt*dyk;
% 
  x=[xk,yk];
xx=[xkk,ykk];
    
    k = k + 1;
    
solx=[solx,xkk];
soly=[soly,ykk];

plot3(xkk,ykk,rosenbrock(xkk,ykk),'.-b','markersize', MS);   			% On affiche les iterations successives



 
   end
toc;
%Affichage de la solution
plot3(1,1,0,'or','markersize', MS);  
legend('fonction a optimiser','initialisation','iterations') 

%plot(solx,soly)
% affichage des resultats
iterations = k

xk
yk