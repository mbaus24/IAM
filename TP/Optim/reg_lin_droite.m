clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Synthese donnees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a=[0:0.01:10];
donnees=25*a+50
donnees_b=donnees+10*randn(1,length(a));
figure(2);plot(a,donnees_b,'*')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Representation de la fontion objectif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:100
    for j=1:100
  fonc_a_opt(i,j)=sum(donnees_b-(i.*a+j)).^2;
    end
end
figure(1);mesh(fonc_a_opt);
title('Fonctionnelle qui est optimisee')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xk=0;
yk=0;
%pas constant
pas = 0.000001;

% iteration initiale
k = 1;

%fonc_a_optimimiser= [0]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient de la fonction objectif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fonc_x=-2*a.*(donnees_b - xk.* a- yk);
fonc_y=-2*(donnees_b - xk.* a- yk);

dxk = sum(fonc_x);	% somme des gradients
dyk =     sum(fonc_y) ;        		% somme des gradients

xkk = xk - pas*dxk;
ykk = yk - pas*dyk;

mu      = .0001;
x=[xk,yk];
xx=[xkk,ykk];


while k <= 20000 && norm(x - xx) > mu
    % calcul de l itere suivant
    xk = xkk;
    yk = ykk;
    fonc_x=-2*a.*(donnees_b - xk.* a- yk);
    fonc_y=-2*(donnees_b - xk.* a- yk);

    dxk = sum(fonc_x);	% somme des gradients
    dyk =     sum(fonc_y) ;        		% somme des gradients
   
% %%%%%%%%%%%%%%%%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   xkk = xk - pas*dxk;
   ykk = yk - pas*dyk;
   x=[xk,yk];
   %JX= sum(donnees_b-(xk.*a+yk)).^2;
   %fonc_a_optimimiser= cat(2,fonc_a_optimimiser,JX);
   xx=[xkk,ykk]
   k = k + 1
    
end
    
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Representation de la solution avec les parametres obtenus par gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
hold on 
plot(a,xx(1).*a+xx(2),'r')
%plot(a,25.*a+50,'r')
title('Donnees fittees')
hold off