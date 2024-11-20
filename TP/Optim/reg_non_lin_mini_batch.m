clear all;
close all;
a=[0:0.1:100];
donnees=exp(-((a-25).^2)/8);%+exp(-((x-10).^2)/5)+exp(-((x-54).^2)/8)+exp(-((x-85).^2)/12);
donnees_b=donnees+0.01*randn(1,1001);
figure(2);plot(a,donnees_b,'*')
title('Fonction qui est optimisee')
for i=1:100
    for j=1:100
  fonc_a_opt(i,j)=sum((donnees_b-exp(-((a-i).^2)/j)).^2) ;
    end
end
figure(1);mesh(fonc_a_opt);

%initialisation
xk=20;
yk=8;

%pas constant
pas = 0.000001;

% iteration initiale
k = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient de la fonction objectif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iirage au sort d'un lot de donnees qui seront utilisees pour le calcul du gradient
N=200000; %portion d'echantillons consideres
taille_max_lot=round(length(a)/N);
liste_indice_echant=unique(randi(length(a),1,taille_max_lot));

%gradient
fonc_x=-(4* exp(-((-xk + a(liste_indice_echant)).^2/yk)) .*(-exp(-((-xk + a(liste_indice_echant)).^2/yk)) + donnees_b(liste_indice_echant)).* (-xk + a(liste_indice_echant)))/yk;
fonc_y=-((2* exp(-((-xk + a(liste_indice_echant)).^2/yk)).* (-exp(-((-xk + a(liste_indice_echant)).^2/yk)) + donnees_b(liste_indice_echant)).* (-xk + a(liste_indice_echant)).^2))/yk^2;

dxk = sum(fonc_x);	% somme des gradients
dyk =     sum(fonc_y) ;        		% somme des gradients

xkk = xk - pas*dxk;
ykk = yk - pas*dyk;


mu      = 0.00000001;
x=[xk,yk];
xx=[xkk,ykk];

k;
norm(x - xx);
%while sqrt(dxk^2 + dyk^2) > erreur 		% Critere d'arret des iterations
while k <= 40000 %&& norm(x - xx) > mu
    %disp(k);
    norm(x - xx);
    % calcul de l itere suivant
    xk = xkk;
    yk = ykk;
    
    N=20; %portion d'echantillons consideres
    taille_max_lot=round(length(a)/N);
    liste_indice_echant=unique(randi(length(a),1,taille_max_lot));
    
   fonc_x=-(4* exp(-((-xk + a(liste_indice_echant)).^2/yk)) .*(-exp(-((-xk + a(liste_indice_echant)).^2/yk)) + donnees_b(liste_indice_echant)).* (-xk + a(liste_indice_echant)))/yk;
   fonc_y=-((2* exp(-((-xk + a(liste_indice_echant)).^2/yk)).* (-exp(-((-xk + a(liste_indice_echant)).^2/yk)) + donnees_b(liste_indice_echant)).* (-xk + a(liste_indice_echant)).^2))/yk^2;

   
   dxk = sum(fonc_x);	% somme des gradients
   dyk =     sum(fonc_y) ;        		% somme des gradients
    
   
   
%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   xkk = xk - pas*dxk;
    ykk = yk - pas*dyk;
    x=[xk,yk];
    xx=[xkk,ykk];
    k = k + 1;
    
    
   end
    
disp(k)
disp(xx)
%
figure(2);
hold on 
plot(a,exp(-((a-xk).^2)/yk),'r')
title('Donnees fittees')
hold off