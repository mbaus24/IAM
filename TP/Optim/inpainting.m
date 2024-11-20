clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Chargement des images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tortue=double(imread('turtle.png'))/255;
tortue=tortue(:,:,1);
tortue_crop= tortue(30:285,100:355);


[n,m]=size(tortue_crop);

figure(1);subplot(2,3,1);imagesc(tortue_crop);colormap(gray);title('image originale: X')
masque=eye(256,256);
masque(228,228)= 0;
masque(229,229)= 0;
masque(100,100)= 0;
masque(101,101)= 0;
masque(102,102)= 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Degradation de l image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tortue_degradee=masque*tortue_crop+0.05*randn(256,256);
figure(1);subplot(2,3,2);imagesc(masque);colormap(gray);title('masque de l image : H')
figure(1);subplot(2,3,3);imagesc(tortue_degradee);colormap(gray);title('image degradee: Y= HX + b')

tortue_filtree=medfilt2(tortue_degradee,[10,3]);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Reconstruction 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Reconstruction par algorithme de gradient
% %pour la minimisation de la solution au sens des moindres carres
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 kmax=100;
 recons=1/(m*n)*ones(n,m,kmax);
recons_reg=1/(m*n)*ones(n,m,kmax);
 erreur=zeros(1,kmax-1);
 erreur_reg=zeros(1,kmax-1);
 lambda=1;
% 
% 
 for k=1:kmax
     Hx=masque*recons(:,:,k);
     grad=masque'*Hx-masque'*tortue_degradee;
     Hx_reg=masque*recons_reg(:,:,k);
     grad_reg=masque'*Hx_reg-masque'*tortue_degradee;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%Algorithme de gradient a pas constant
% %%pour minimiser J(x)=||y-Hx||^2 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
  alpha=0.1;
% 
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%Reconstruction avec gradient � pas constant
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  recons(:,:,k+1)=recons(:,:,k)-alpha*grad ;
  erreur(1,k)=sum(sum(((tortue_crop-recons(:,:,k+1)).^2)));
  end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%Reconstruction avec gradient � pas constant r�gularis�
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  A=[0,1/4,0;1/4,0,0;0,1/4,0];
  gamma=0.8;
   
  recons_reg(:,:,k+1)=recons_reg(:,:,k)-alpha*grad_reg-alpha*gamma*(conv2(recons_reg(:,:,k),A,'same')) ;
  erreur_reg(1,k)=sum(sum(((tortue_crop-recons_reg(:,:,k+1)).^2)));
% 
 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualisation reconstruction avec gradient � pas constant r�gularis�
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 figure(1);subplot(2,3,4);plot(erreur);colormap(gray);title('Erreur de reconstruction')
 [mini,pos]=min(erreur(2:kmax));
 ite=pos
 subplot(2,3,5);imagesc(recons(:,:,pos));colormap(gray);title('Iteration qui donne le minimum de l erreur ')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Visualisation reconstruction avec gradient � pas constant r�gularis�
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
subplot(2,3,4);plot(erreur);colormap(gray);
 hold on 
 subplot(2,3,4);plot(erreur_reg,'r');colormap(gray);legend('erreur sans reg','erreur avec reg');title('Erreur de reconstruction')
 hold off
 [mini,pos_reg]=min(erreur_reg(2:kmax));
  ite_reg=pos_reg
 subplot(2,3,6);imagesc(recons_reg(:,:,pos_reg));colormap(gray);title('Iteration qui donne le minimum de l erreur avec regularisation')
 
