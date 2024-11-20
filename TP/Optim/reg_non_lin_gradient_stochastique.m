clear all;
close all;
a=[0:0.1:100];
donnees=exp(-((a-25).^2)/8);%+exp(-((x-10).^2)/5)+exp(-((x-54).^2)/8)+exp(-((x-85).^2)/12);
donnees_b=donnees+0.01*randn(1,1001);
figure(2);plot(a,donnees_b,'*')

for i=1:100
    for j=1:100
  fonc_a_opt(i,j)=sum((donnees_b-exp(-((a-i).^2)/j)).^2) ;
    end
end
figure(1);mesh(fonc_a_opt);
title('Fonction qui est optimisee');
%initialisation
xk=20;
yk=3;

%pas constant
pas = 10;

% iteration initiale
k = 1;

%tirage aleatoire d'un coupe de donnees
ind= randi(length(a));
echan_abs=a(ind);
echan_ord=donnees_b(ind);

%gradient
fonc_x=-(4* exp(-((-xk + echan_abs).^2/yk)) .*(-exp(-((-xk + echan_abs).^2/yk)) + echan_ord).* (-xk + echan_abs))/yk
fonc_y=-((2* exp(-((-xk + echan_abs).^2/yk)).* (-exp(-((-xk + echan_abs).^2/yk)) + echan_ord).* (-xk + echan_abs).^2))/yk^2
%dxk = sum(fonc_x);	% calcul de la derivee par rapport ? x
%dyk =     sum(fonc_y) ;        		% calcul de la derivee par rapport ? y


xkk = xk - pas*fonc_x
ykk = yk - pas*fonc_y



mu      = .00001;
x=[xk,yk];
xx=[xkk,ykk];

k
norm(x - xx)
%while sqrt(dxk^2 + dyk^2) > erreur 		% Critere d'arret des iterations
while k <= 40000 %&& norm(x - xx) > mu
    disp(k) 
    norm(x - xx)
    % calcul de l itere suivant
    xk = xkk
    yk = ykk
    ind= randi(length(a));
    echan_abs=a(ind);
    echan_ord=donnees_b(ind);
    fonc_x=-(4* exp(-((-xk + echan_abs).^2/yk)) .*(-exp(-((-xk + echan_abs).^2/yk)) + echan_ord).* (-xk + echan_abs))/yk
    fonc_y=-((2* exp(-((-xk + echan_abs).^2/yk)).* (-exp(-((-xk + echan_abs).^2/yk)) + echan_ord).* (-xk + echan_abs).^2)/yk^2)

% %%%%%%%%%%%%%%%%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   xkk = xk - pas*fonc_x;
   ykk = yk - pas*fonc_y; 
   x=[xk,yk];
   xx=[xkk,ykk];
   k = k + 1
    
   end
    
   
%donnees=exp(-((a-25).^2)/2);%+exp(-((x-10).^2)/5)+exp(-((x-54).^2)/8)+exp(-((x-85).^2)/12);
%
figure(2);
hold on 
plot(a,exp(-((a-xk).^2)/yk),'r')
title('Donnees fittees')
hold off