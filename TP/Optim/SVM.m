%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Algorithme SVM pour separation lineaire de deux classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generation des donnees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 35;
X1 = rand(N,2);
X2 = rand(N,2) + 0.5*ones(N,1)*[1 2];
y = [ones(N,1) ; -ones(N,1)];

figure(1)
plot(X1(:,1),X1(:,2),'+r','LineWidth',2); hold on
plot(X2(:,1),X2(:,2),'ob','LineWidth',2);
legend('classe 1','classe 2', 'Location','NorthWest')
hold off

% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Implementation de la minimisation sous contrainte
% %% Solution avec quadprog
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% X = [X1;X2];
% [nbre_ech,p] = size(X);
% 
% H = [1,0,0;0,1,0;0,0,0];
% f = [0; 0; 0];
% A = -[diag(y)*X y];
% b = -ones(nbre_ech,1);
% 
% 
% [v] = quadprog(H,f,A,b);
% %[v] = fmincon(f,A,b);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% visualisation des résultats
% %% Equation du plan  a partir des parametres estimes
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% x1 = 0;
% y1 = (-v(3)-v(1)*x1)/v(2);
% D = [x1  y1];
% x2 = 1.5;
% y2 = (-v(3)-v(1)*x2)/v(2);
% D = [D ; [x2 y2]];
% 
% figure(2)
% plot(X1(:,1),X1(:,2),'+r','LineWidth',2); hold on
% plot(X2(:,1),X2(:,2),'ob','LineWidth',2);
% h = plot(D(:,1),D(:,2),'k','LineWidth',2);
% nw = sqrt(v(1)^2+v(2)^2);%marge : norme de w
% plot(D(:,1),D(:,2)+1/nw,'--k') 
% plot(D(:,1),D(:,2)-1/nw,'--k') 
% legend('classe 1','classe 2','frontière ','+ marge ','- marge','Location','NorthWest')
% axis square
% hold off
% 
