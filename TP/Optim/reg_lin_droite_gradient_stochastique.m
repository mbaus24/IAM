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
title('Fonction qui est optimisee')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xk=0;
yk=0;

%pas constant
pas = 0.00001;

% iteration initiale
k = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient de la fonction objectif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind= randi(length(a));
echan_abs=a(ind);
echan_ord=donnees_b(ind);

fonc_x=-2*echan_abs.*(echan_ord + xk.* echan_abs- yk);
fonc_y=-2*(echan_ord + xk.* echan_abs- yk);

%G= sqrt(fonc_x^2+fonc_y^2+eps);
%pas_adapt=pas/G; 
pas_adapt = pas;
xkk = xk - pas_adapt*fonc_x;
ykk = yk - pas_adapt*fonc_y;

mu      = .000000001;
x=[xk,yk];
xx=[xkk,ykk];


while k <= 200000 && norm(x - xx) > mu
    % calcul de l itere suivant
    xk = xkk;
    yk = ykk;
    
    ind= randi(length(a));
    echan_abs=a(ind);
    echan_ord=donnees_b(ind);

    fonc_x=-2*echan_abs.*(echan_ord - xk.* echan_abs- yk);
    fonc_y=-2*(echan_ord - xk.* echan_abs- yk);
    
    %G= sqrt(fonc_x^2+fonc_y^2+eps);



% %%%%%%%%%%%%%%%%%%%%GRADIENT PAS CONSTANT
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %pas_adapt=pas/G; 
    pas_adapt = pas;
    xkk = xk - pas_adapt*fonc_x;
    ykk = yk - pas_adapt*fonc_y;
    x=[xk,yk];
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