
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Chargement des images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load homer.mat
load matrice_flou.mat %degradation: flou de bouge
[n,m]=size(homer);
figure;subplot(3,3,1);imagesc(homer);colormap(gray);title('image originale')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Degradation de l image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
homer_flou=matrice_flou*homer;
subplot(3,3,2);imagesc(homer_flou);colormap(gray);title('image floue')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Restauration de l image par m�thode directe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recons=inv(matrice_flou)*homer_flou;
subplot(3,3,3);imagesc(recons);colormap(gray);title('image restauree par meth.directe')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Introduction d un bruit electronique
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bruit=10*randn(256,256);
homer_flou_bruit=matrice_flou*homer+bruit;
subplot(3,3,4);imagesc(homer_flou_bruit);colormap(gray);title('image floue et avec bruit')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Restauration de l imagepar m�thode directe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recons_avec_bruit=inv(matrice_flou'*matrice_flou)*(matrice_flou'*homer_flou_bruit);
subplot(3,3,5);imagesc(recons_avec_bruit);colormap(gray);title('image restauree par meth.directe');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reconstruction par algorithme de gradient
%pour la minimisation de la solution au sens des moindres carres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kmax=500;
recons=1/(m*n)*ones(n,m,kmax);
erreur=zeros(1,kmax-1);
for k=1:kmax
    Hx=matrice_flou*recons(:,:,k);
    grad=matrice_flou'*Hx-matrice_flou'*homer_flou_bruit;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Algorithme de gradient a pas constant
%%pour minimiser J(x)=||y-Hx||^2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % 
 %alpha=1;
 % recons(:,:,k+1)=recons(:,:,k)-alpha*grad ;
 % erreur(1,k)=sum(sum(((homer-recons(:,:,k+1)).^2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Algorithme de gradient 
%%pour minimiser J(x)=||y-Hx||^2 
%%avec recherche unidimensionnelle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % 
  %alpha_opt=fminbnd(@(alpha) sum(sum( (homer_flou_bruit- matrice_flou*(recons(:,:,k)-alpha*grad )).^2)),0,100);
  %recons(:,:,k+1)=recons(:,:,k)-alpha_opt*grad ;
  %erreur(1,k)=sum(sum(((homer-recons(:,:,k+1)).^2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Algorithme du gradient regularise pour optimiser J(x)=||y-Hx||^2 +gamma *||x-p||^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%gamma=0.02;
%alpha=0.05; 
 %p=1/(n*m)*ones(n,m);
 %recons(:,:,k+1)=recons(:,:,k)-alpha*grad-alpha*gamma*(2*recons(:,:,k)-2*p) ;     
 %erreur(1,k)=sum(sum(((homer-recons(:,:,k+1)).^2)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Algorithme du gradient regularise pour optimiser J(x)=||y-Hx||^2 +gamma *||x-Ax||^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=[0,1/4,0;1/4,0,1/4;0,1/4,0];
 gamma=0.02;
 alpha=0.5; 
 recons(:,:,k+1)=recons(:,:,k)-alpha*grad-alpha*gamma*(-2*conv2(recons(:,:,k),A,'same')+2*recons(:,:,k)) ;
 erreur(1,k)=sum(sum(((homer-recons(:,:,k+1)).^2)));
 


end
% % % 
  subplot(3,3,7);plot(erreur);title('Erreur de reconstruction')
  [mini,pos]=min(erreur(2:kmax));
  subplot(3,3,8);imagesc(recons(:,:,pos));colormap(gray);title('Iteration qui donne le minimum de l erreur ')
  ite=pos
% % 
  figure;imagesc(recons(:,:,100));colormap(gray);title('Iteration 100 ')
    figure;imagesc(recons(:,:,101));colormap(gray);title('Iteration 101 ')