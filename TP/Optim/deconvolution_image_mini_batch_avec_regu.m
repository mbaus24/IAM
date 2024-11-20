
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Chargement des images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%Restauration de l image par methode directe
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
%Restauration de l imagepar methode directe
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
kmax=1000;
recons=1/(m*n)*ones(n,m,kmax);
erreur=zeros(1,kmax-1);
recons_reg=1/(m*n)*ones(n,m,kmax);
erreur_reg=zeros(1,kmax-1);
%Tirage au sort d'un echantillon aleatoire(un pixel)


for k=1:kmax
    
    %ind_i= randi(n);
    %ind_j= randi(n);
    
    N=0.3; %portion d'echantillons consideres
    taille_max_lot=round(n/N);
    liste_indice_echant_i=(randi(n,1,taille_max_lot));
    liste_indice_echant_j=(randi(n,1,taille_max_lot));
    liste_indice_echant_i=[liste_indice_echant_i,circshift(liste_indice_echant_i,14),circshift(liste_indice_echant_i,256)];
    liste_indice_echant_j=[liste_indice_echant_j,circshift(liste_indice_echant_j,145),circshift(liste_indice_echant_j,18)];
    selectionechantillons=zeros(n,m);
    for w= 1:length(liste_indice_echant_i)
        for v=1:length(liste_indice_echant_j)
            selectionechantillons(liste_indice_echant_i(w),liste_indice_echant_j(v))=1;
        end
    end
    
    ech_sel=homer_flou_bruit.*selectionechantillons;
    Hx=matrice_flou*recons(:,:,k);
    grad=matrice_flou'*Hx-matrice_flou'*ech_sel;
    Hx_reg=matrice_flou*recons_reg(:,:,k);
    grad_reg=matrice_flou'*Hx_reg-matrice_flou'*ech_sel;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Algorithme de gradient a pas constant
%%pour minimiser J(x)=||y-Hx||^2 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
 alpha=1;
 recons(:,:,k+1)=recons(:,:,k)-alpha*grad ;



  A=[0,-1,0;-1,4,-1;0,-1,0];
 gamma=0.01;
%  
 recons_reg(:,:,k+1)=recons_reg(:,:,k)-alpha*grad_reg-alpha*gamma*(conv2(homer_flou_bruit,A,'same')) ;
 erreur_reg(1,k)=sum(sum(((homer-recons_reg(:,:,k+1)).^2)));

 erreur(1,k)=sum(sum(((homer-recons(:,:,k+1)).^2)));
% 



end
% % % 
 subplot(3,3,7);plot(erreur);title('Erreur de reconstruction')
 [mini,pos]=min(erreur(2:kmax));
 subplot(3,3,8);imagesc(recons(:,:,pos));colormap(gray);title('Iteration qui donne le minimum de l erreur ')
 ite=pos
[minireg,posreg]=min(erreur_reg(2:kmax));
 %subplot(3,3,9);plot(erreur_reg);title('Erreur de reconstruction')

 subplot(3,3,9);imagesc(recons_reg(:,:,pos));colormap(gray);title('Iteration qui donne le minimum de l erreur ')
 ite=posreg