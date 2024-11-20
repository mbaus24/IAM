clear all;
close all;
a=[0:0.1:100];
donnees=exp(-((a-25).^2)/8);
donnees_b=donnees+0.01*randn(1,1001);
figure(2);plot(a,donnees_b,'*')



for i=1:100
    for j=1:100
  fonc_a_opt(i,j)=sum((donnees_b-exp(-((a-i).^2)/j)).^2) ;
    end
end
figure(1);mesh(fonc_a_opt);
title('Fonction qui est optimisee')


%initialisation
xk=25;
yk=3;

%[xk,yk]=[0,0];
%pas constant
pas = 0.0001;

% iteration initiale
k = 1;


%gradient
fonc_x=-(4* exp(-((-xk + a).^2/yk)) .*(-exp(-((-xk + a).^2/yk)) + donnees_b).* (-xk + a))/yk;
fonc_y=-((2* exp(-((-xk + a).^2/yk)).* (-exp(-((-xk + a).^2/yk)) + donnees_b).* (-xk + a).^2)/yk^2);
dxk = sum(fonc_x);	% calcul de la derivee par rapport ? x
dyk =     sum(fonc_y) ;       		% calcul de la derivee par rapport ? y


xkk = xk - pas*dxk;
ykk = yk - pas*dyk;



mu      = .0000001;
 x=[xk,yk];
 xx=[xkk,ykk];

tic;
%while sqrt(dxk^2 + dyk^2) > erreur 		% Critere d'arret des iterations
   while k <= 40000 && norm(x - xx) > mu
    % calcul de l itere suivant
xk = xkk;
yk = ykk;

fonc_x=-(4* exp(-((-xk + a).^2/yk)) .*(-exp(-((-xk + a).^2/yk)) + donnees_b).* (-xk + a))/yk;
fonc_y=-((2* exp(-((-xk + a).^2/yk)).* (-exp(-((-xk + a).^2/yk)) + donnees_b).* (-xk + a).^2))/yk^2;
dxk = sum(fonc_x);	% calcul de la derivee par rapport ? x
dyk =     sum(fonc_y);          		% calcul de la derivee par rapport ? y;

% %%%%%%%%%%%%%%%%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   xkk = xk - pas*dxk;
   ykk = yk - pas*dyk; 

x=[xk,yk];
xx=[xkk,ykk];
    
    k = k + 1
    
   end
    
   
%donnees=exp(-((a-25).^2)/2);%+exp(-((x-10).^2)/5)+exp(-((x-54).^2)/8)+exp(-((x-85).^2)/12);
%
xx
figure(2);
hold on 
plot(a,exp(-((a-xk).^2)/yk),'r')
title('Donnees fittees')
hold off