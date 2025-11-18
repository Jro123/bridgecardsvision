// Copyright Jacques ROSSILLOL 2024
//
//
#define _USE_MATH_DEFINES

#ifdef _WIN32
#include <Windows.h>
#include <tchar.h>
#else
#include <algorithm>
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp> // Inclure le module ximgproc pour LSD
#include <opencv2/freetype.hpp>
#include "config.h"

// constantes 
extern const char* NESO[4];
//const char* couleurs[] = {"P", "C", "K", "T"}; // ♠, ♥, ♦, ♣
extern const char* nomval[14];
extern const char* valeurcarte[14];
extern const char* nomcouleur[4]; 
extern const char* couleurcarte[4]; 

extern int distribution[4][13][2]; // les 4 mains en cours de décodage
extern unecarte carteMort[13]; // la main du mort

void traiterLaCarte(cv::Mat& image,  unecarte& carte, std::vector<unecartePrec> cartesPrec, 
  std::vector<uncoin> Coins, unpli& monpli, bool estvideo, config& maconf);



// pour chaque carte de la frame analysée:
// déterminer si la carte à déjà été trouvée dans une frame précédente
// déterminer si elle est déjà dans le pli en cours
// déterminer si la carte est complète (au moins deux coins)
// vérifier qu'elle ne contient pas des coins d'autres cartes
// vérifier que ses coins ne sont pas sur des bords d'autres cartes
// vérifier qu'il y a une seule nouvelle carte complète
// ordonner les coins de la carte complète
// compléter en ajoutant les sommets non trouvés
// décoder
// vérifier que la carte n'a pas déjà été jouée dans un pli précédent
// vérifier par rapport au cartes du mort
// ajouter au pli en cours



void traiterCartes(cv::Mat image, config& maconf,std::vector<unecarte>& cartes, 
  std::vector<uncoin>& Coins, std::vector<unecartePrec>& cartesPrec,
   const std::vector<ligne>&  lignes, unpli& monpli) {

  // calculer les bords des cartes
  std::vector<unbord> bords = calculerBords(cartes, maconf);
  std::vector<unbord> bordsPrec = calculerBords(cartesPrec, maconf);


  int printoption = maconf.printoption;
  if (printoption > 1) std::cout<< std::endl<<"================== recherche des nouvelles cartes ======"<<std::endl;
  bool estvideo = true;
  int epsilon = std::max(maconf.hauteurcarte / 10, 2*maconf.deltacadre);
  int nca = 0; // numéro de carte complète à analyser
  int nc = 0; // numéro de carte cherchée, cartes numérotées à partir de 1
  int numcol, valcarte; // couleur et valeur de carte
  int h0 = -1; // coin de la frame précédente
  int h1 = -1;
  int dc = maconf.deltacadre;
  
  int attendre = 0; // pour pouvoir actualiser l'affichage en debug 
  if(attendre) cv::waitKey(0);

  nc = 0;
  for (nc = 0; nc < cartes.size(); nc++){
    auto& carte = cartes[nc];
    auto coins = carte.coins;
    bool existeHauteur(false), existeLargeur(false);
    int nbHauteur(0), nbLargeur(0);

    // déterminer si les coins de la carte fournissent une hauteur ou largeur de carte
    // ignorer les coins qui sont sur un bord de carte ou de carte précédente
    for (int n = 0; n < coins.size(); n++){
      auto cn = coins[n];
      if (cn->elimine) continue;
      if (cn->couleur >= 0 && cn->valeur > 0) continue; // coin déjà analysé
      // nouvelle carte

      // ignorer s'il est sur un bord de carte
      
      bool coinSurBord = cn->estSurBord(bordsPrec, maconf);
      if(! coinSurBord)  coinSurBord = cn->estSurBord(bords, maconf);
      if(coinSurBord){
        cn->elimine = true;
        continue; // ignorer le coin qui est sur un bord ou bord précédent
      } 

      // noter que par construction des numéros de cartes, 
      //  les numéros de cartes sont en ordre croissant des numéros de coins
      // considérer tous les coins de cette carte nc
      // vérifier qu'on a au moins 2 sommets, limitant une hauteur ou une largeur de carte
      cv::Point2i P1(cn->sommet), P2;
      int nbc = 0;
      int nbCoinsTrouves = 0;  // comptage des coins trouvés dans la frame précédente

      for (int m = n+1; m < coins.size(); m++) {
        uncoin* coin = coins[m];
        cv::Point2i P = coin->sommet;
        // ignorer un coin qui est sur un bord d'une carte de cette frame
        coinSurBord = coin->estSurBord(bordsPrec, maconf);
        if(! coinSurBord)  coinSurBord = coin->estSurBord(bords, maconf);
        if(coinSurBord){
          coin->elimine = true;
          continue; // ignorer le coin qui est sur un bord ou bord précédent
        } 

        nbc++;
        /*if (nbc == 1) P1 = P;
        else */{
          float lg = std::sqrt((P1-P).x * (P1-P).x + (P1-P).y * (P1-P).y) +1;
          if (std::abs(lg - maconf.hauteurcarte) < epsilon)
            {existeHauteur = true; nbHauteur++;}
          if (std::abs(lg - maconf.largeurcarte) < epsilon)
            {existeLargeur = true; nbLargeur++;}
        }
      } // for m
    } // for n

    // déterminer si la carte était déjà dans la frame précédente
    // dans la frame précédente si au moins deux coins y étaient, sur la même carte précédente
    // la distance entre un sommet de la carte et un coin de la frame précédente
    //         doit être commune à tous les sommets de la carte
    bool estcarte(true);
    bool carteTrouvee(false);
    if (existeHauteur || existeLargeur) {
      for (auto& ucp : cartesPrec) {
        if (ucp.contient (carte, epsilon) ) {
          carteTrouvee = true;
          carte.couleur = ucp.couleur;
          carte.valeur = ucp.valeur;
          for (auto cn : carte.coins) {
            cn->couleur = ucp.couleur;
            cn->valeur = ucp.valeur;
          }
          std::string nomcarte = "";
          if (ucp.couleur >= 0 && ucp.couleur <= 3) nomcarte = nomcouleur[ucp.couleur];
          if (ucp.valeur > 0 && ucp.valeur < 14){
            nomcarte += " ";
            nomcarte += nomval[ucp.valeur];
          }
          if(printoption > 1)  std::cout<< " carte "<< nc << " ("<< nomcarte 
            << ") déjà dans la frame précédente." << std::endl;
          break;
        }
      } // cartes précédentes
    }

    if (!carteTrouvee && (existeHauteur || existeLargeur)
     && nbHauteur <= 2 && nbLargeur <= 2) {
      for (int n=0; n < carte.coins.size(); n++) {
        auto cn = carte.coins[n];
        cv::Point2i P1 = cn->sommet;
        if (n == 0 && printoption > 0) std::cout<< "nouvelle carte "<< nc<<" nouveau sommet "<< n <<P1<<std::endl;
        estcarte = true;
        if (n> 0 && printoption > 0) std::cout<< "         carte "<< nc<<" autre sommet "<< n <<cn->sommet<<std::endl;
      }
      if (printoption > 0) std::cout<<" carte complète "<< nc<<std::endl;
      // vérifier si la carte est dans le pli en cours
      bool carteDansPli = carte.estDansPli (monpli, epsilon);

      if (carteDansPli) {
        estcarte = false;
        std::string nomcarte= "??";
        if(carte.couleur >= 0 && carte.couleur <= 3) nomcarte = nomcouleur[carte.couleur];
        if (carte.valeur >0 && carte.valeur <= 13) nomcarte +=  std::string(" ") + std::string(nomval[carte.valeur]);
        if (printoption > 0) std::cout<<" carte "<<nc <<" ("<<nomcarte<<") dans le pli en cours "<< std::endl;
        // valoriser tous les coins de cette carte
        for (auto& coin : coins){
          coin->couleur = carte.couleur;
          coin->valeur = carte.valeur;
        }
      }

        
      
      
      if (estcarte) { // carte avec au moins deux coins et absente de la frame précédente et du pli en cours
        // analyser la nouvelle carte
        if (nca != 0){
          if (printoption> 0) std::cout<< " plusieurs cartes complètes "<< nca << ","<<nc<<std::endl;
          // ignorer la carte dont un sommet est sur un bord de carte de la frame précédente
          // parcourir les coins de la carte nc
          // comparer chaque coin aux bords des cartes du pli en cours
          // vérifier les deux cartes NC puis NCA
          // vérifier les sommets de la carte nc
          bool ignorerNC = false;
          for (int i = 0; i < coins.size(); i++){
            //if (coins[i].numCarte != nc) continue;
            cv::Point2i P(coins[i]->sommet);
            cv::Point2i U, V;
            // comparer aux bords de cartes du pli en cours
            for (int j = 0; j < monpli.nbcartes; j++ ) { // chaque carte du pli
              numcol = monpli.cartes[j].couleur;
              valcarte = monpli.cartes[j].valeur;
              for (int k = 0; k < 4; k++){ // chaque sommet
                U = monpli.cartes[j].sommet[k];
                for (int l=k+1; l < 4; l++) { // chaque autre sommet de la même carte du pli
                  V = monpli.cartes[j].sommet[l];
                  // on a un bord UV
                  double dist = calculerDistance(P, U, V);
                  if (std::abs(dist) > maconf.deltacadre) continue;
                  // sur le segment ?
                  int ps = ((P-U).x * (P-V).x + (P-U).y * (P-V).y);
                  if (ps >= 0) continue;
                  ignorerNC = true;
                  break;
                } // for l
                if (ignorerNC) break;
              } // for k
              if (ignorerNC) break;
            } // for j
            if (ignorerNC) break;
          } // for i
          if (ignorerNC) {
            if (printoption > 1) std::cout<<" carte "<<nc<< " ignorée car dans la frame précédente"<<std::endl;
            //nc = 0;
          }
          else {
            // vérifier la carte nca
            for (int i = 0; i < coins.size(); i++){
              //if (coins[i].numCarte != nca) continue;
              cv::Point2i P(coins[i]->sommet);
              // comparer aux bords de cartes du pli en cours
              for (int j = 0; j < monpli.nbcartes; j++ ) { // chaque carte du pli
                numcol = monpli.cartes[j].couleur;
                numcol = monpli.cartes[j].valeur;
                for (int k = 0; k < 4; k++){ // chaque sommet
                  cv::Point2i U(monpli.cartes[j].sommet[k]);
                  for (int l=k+1; l < 4; l++) { // chaque autre sommet de la même carte du pli
                    cv::Point2i V(monpli.cartes[j].sommet[l]);
                    // on a un bord UV
                    double dist = calculerDistance(P, U, V);
                    if (std::abs(dist) > maconf.deltacadre) continue;
                    ignorerNC = true;
                    break;
                  } // for l
                  if (ignorerNC) break;
                } // for k
                if (ignorerNC) break;
              } // for j
              if (ignorerNC) break;
            } // for i
            if (ignorerNC) {
              if (printoption > 1) std::cout<<" carte "<<nca<< " ignorée car dans la frame précédente"<<std::endl;
              nca = nc + 1;
              //nc = 0;
            }
          } // else
        } else { // première nouvelle carte
          nca = nc + 1;
          break;
        }
      } else if(!carteDansPli) { // coin (n) absent de la frame précédente et non opposé à un autre coin
        if(printoption > 1) std::cout<< " carte "<<nc<< " a un seul coin: "<<carte.coins[0]->sommet<<std::endl;
      }
    } // carte trouvée avec au moins un bord hauteur ou largeur de carte

  } // cartes
  
  if (nca){
    if (printoption > 0) std::cout<<"==> analyse de la carte "<<nca<<std::endl;
    auto& carte = cartes[nca -1];
    traiterLaCarte(image, carte, cartesPrec, Coins, monpli, estvideo, maconf);
  }
  return;
}


// analyser une carte 
void traiterLaCarte(cv::Mat& image,  unecarte& carte, std::vector<unecartePrec> cartesPrec,
  std::vector<uncoin> Coins,  unpli& monpli, bool estvideo, config& maconf){

  int printoption = maconf.printoption;
    // obtenir les 4 sommets du rectangle de la carte
    // trouver l'encombrement des coins de la carte
    // partir d'un coin le plus  à gauche, le plus haut s'il y en a plusieurs, (P)
    // rechercher un coin opposé (Q) à distance proche de la longueur ou largeur de carte
    // Q doit être proche d'une des lignes du coin P
    // et la distance PQ doit être la plus proche possible de la longueur ou largeur de carte
    // si la distance n'est pas exactement la longueur ou largeur
    // repartir du coin Q (--> P) et chercher un coin Q à distance convenable.
    // on a alors deux coins P et Q opposés PQ étant un coté de la carte
    // chercher ou calculer les deux autres sommets

    auto coins = carte.coins;
    cv::Point2i P, Q, U, V;
    int n1(-1), n2(-1), n3(-1), n4(-1); // les indices des coins de la carte analysée
    int nbpts;
    int dc = std::max(maconf.hauteurcarte/15 , maconf.deltacadre);
    // rechercher le coin le plus à gauche (et le plus haut s'il y en a plusieurs)
    int xmin(12345); // une grande valeur
    for (auto cn : coins) {
      if (cn->elimine) continue; // coin éliminé
      if (cn->sommet.x <= xmin ) {
        xmin = cn->sommet.x;  //limite gauche de la carte
      }
    }
    int ymin(12345); // une grande valeur
    for (int n = 0; n < coins.size(); n++) {
      const auto cn = coins[n];
      if (cn->elimine) continue; // coin éliminé
      if (cn->sommet.x <= xmin ) {    // un des coins les plus à gauche
        if (cn->sommet.y < ymin ) {
          n1 = n;
          ymin = cn->sommet.y;
        }
      }
    }
    P = coins[n1]->sommet;
    if(printoption > 1) std::cout<<" coin "<< n1 <<" "<<P<<std::endl;
    // rechercher un coin opposé sur le premier coté du coin P puis sur le deuxième
    n2 = -1;
    int lgref, lgref2;
    float a,b,c, aa,bb,cc;
    cv::Vec4i ln = coins[n1]->l1->ln;
    cv::Vec4i ln2 = coins[n1]->l2->ln;
    a = coins[n1]->l1->a; b = coins[n1]->l1->b; c = coins[n1]->l1->c;
    aa = coins[n1]->l2->a; bb = coins[n1]->l2->b; cc = coins[n1]->l2->c;
    
    for (int i=0; n2< 0 && i < 2; i++) {
      n2 = -1;
      float ecart, ecartmin(dc);

      for (int n = 0; n < coins.size(); n++) { // les coins de la carte
        const auto& cn = coins[n];
        if (cn->elimine) continue;
        if (n == n1) continue;
        Q = cn->sommet;
        float dist = a*Q.x + b*Q.y + c;
        float lg = std::abs(dist);
        if (lg > dc) continue; // Q n'est pas proche d'une arête du coin n (P)
        lg = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
        if (lg > maconf.hauteurcarte + dc) continue; // PQ trop grand
        if (lg < maconf.largeurcarte - dc) continue; //PQ trop court
        if (lg > maconf.largeurcarte + dc && lg < maconf.hauteurcarte - dc) continue;
        // PQ est hauteur ou largeur de carte ?
        if (lg > (maconf.largeurcarte + maconf.hauteurcarte)/2) {
          lgref = maconf.hauteurcarte;
          lgref2 = maconf.largeurcarte;
        }
        else {
          lgref = maconf.largeurcarte;
          lgref2 = maconf.hauteurcarte;
        }
        ecart = lgref - lg;
        if (ecart <= ecartmin){
          ecartmin = ecart;
          n2 = n;
        }
      }
      if (n2 < 0) {
        ln2 = ln;
        ln = coins[n1]->l2->ln;
        aa = a; bb=b; cc=c;
        a = coins[n1]->l2->a; b = coins[n1]->l2->b; c = coins[n1]->l2->c;
      }
    }  // les deux arêtes du coin n


    if (n2 < 0) {
      // aucun coin opposé au coin P
      nbpts = 1;
    } else {
      Q = coins[n2]->sommet;
      if(printoption > 1) std::cout<<" coin opposé "<< n2 <<" "<<Q<<std::endl;
      // on a deux sommets de la carte P (n1) et Q (n2)
      // rechercher ou calculer les deux autres sommets U et V
      // la ligne commune est : coin n1 , ligne ln, coefficient a b c
      // rechercher le coin U sur l'autre ligne du coin P
      cv::Vec4i ln = ln2;
      float ecmin = 3;
      n3 = -1;
      for (int n = 0; n < coins.size(); n++) {
        const auto cn = coins[n];
        if (cn->elimine) continue;
        if (n == n1 || n == n2) continue;
        U = cn->sommet;
        float dist = aa*U.x + bb*U.y + cc;
        if (std::abs(dist) > 1) continue; // U n'est pas sur l'autre ligne (que PQ)  du coin P
        float lg = std::sqrt((U.x - P.x)*(U.x - P.x) + (U.y - P.y)*(U.y - P.y));
        if (std::abs(lg - lgref2) > ecmin) continue; // U n'est pas à distance convenable de P (hauteur ou largeur de carte)
        if(std::abs(lg - lgref2) < ecmin) {
          ecmin = std::abs(lg - lgref2);
          n3 = n;
        }
      }
      if(n3 >= 0) {
        U = cv::Point2i(coins[n3]->sommet);
        if(printoption > 1) std::cout<<" coin trois "<< n3 <<" "<<U<<std::endl;
      }
      ligne *l1 = coins[n2]->l1;
      ligne *l2 = coins[n2]->l2;
      // rechercher le sommet V sur le coté orthogonal à PQ du coin Q (n2)
      float ps = a*coins[n2]->l1->a + b*coins[n2]->l1->b;
      if(std::abs(ps) < 0.5) {
        l2 = l1;
        l1 =coins[n2]->l2;
      }
      // ii est la ligne // PQ  jj est la ligne ortogonale à PQ
      // rechercher un coin V de cette carte tel que QV = lgref2 (à deltacadre près)
      // distance de V à la droite PQ = lgref2   et QV orthogonal à PQ
      n4 = -1;
      ecmin = 3;
      for (int n = 0; n < coins.size(); n++) {
        const auto cn = coins[n];
        if (cn->elimine) continue;
        if (n == n1 || n == n2 || n == n3) continue;
        V = cn->sommet;
        // distance de V à cette ligne jj orthogonale à PQ:
        float d = V.x * l2->a + V.y * l2->b + l2->c;
        if (std::abs(d) > dc) continue;

        float dist = std::abs(a*V.x + b*V.y + c);
        if (std::abs(dist - lgref2) > ecmin) continue;
        if (std::abs(dist - lgref2) < ecmin) {
          ecmin = std::abs(dist - lgref2);
          n4 = n;
        }
      }
      if(n4 >= 0) {
        V = cv::Point2i(coins[n4]->sommet);
        if(printoption > 1) std::cout<<" coin quatre "<< n4 <<" "<<V<<std::endl;
      }


      if (n3 < 0) {
        // pas de coin sur l'autre ligne du coin P
        U = V;
        n3 = n4;
        n4 = -1;
      }
    }
    nbpts = 1;
    if (n4 >= 0) nbpts = 4;
    else if (n3 >= 0) nbpts = 3;
    else if (n2 >= 0) nbpts = 2;
    int pts[4][2]; // les 4 points de la carte
    pts[0][0] = P.x; pts[0][1] = P.y;
    if (n2 >= 0) pts[1][0] = Q.x; pts[1][1] = Q.y;
    if (n3 >= 0) pts[2][0] = U.x; pts[2][1] = U.y;
    if (n4 >= 0) pts[3][0] = V.x; pts[3][1] = V.y;

    // TODO  si on n'a pas les 4 sommets, compléter à partir des deux premiers;
    //
    //  U_________V
    //  |         |
    //  |         |
    //
    double lgW; // longueur de l'autre coté
    double lg, lg2;
    int nbptsW = nbpts;
    if ((nbpts == 2 || nbpts == 3)){
      // les deux premiers points sont la longueur ou la largeur ou la diagonale

      // on dispose du vecteur normal des lignes des coins
      cv::Point2i P(coins[n1]->sommet);
      cv::Point2i Q(coins[n2]->sommet);
      // vérifier que PQ est la hauteur ou la largeur de carte
      cv::Point2i C; // autre sommet opposé au point P
      cv::Point2i D; // autre sommet opposé au point Q
      // PQ hauteur ou largeur de carte ?
      lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
      lg = std::sqrt(lg2);
      if (lg > 5*maconf.hauteurcarte/4 && nbpts == 3){
        // c'est la diagonale.
        // s'il y a un 3ème sommet, remplacer le 2ème sommet
        Q = coins[n3]->sommet;
        pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1]; 
        lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
        lg = std::sqrt(lg2);
        int w = n2; n2 = n3; n3 = w;
      }
      if (std::abs (lg - maconf.hauteurcarte) <= dc ) lgW = maconf.largeurcarte;
      else if (std::abs (lg - maconf.largeurcarte) <= dc ) lgW = maconf.hauteurcarte;
      else lgW = 0;
      if (lgW > dc) {

        double a = coins[n1]->l1->a;
        double b = coins[n1]->l1->b;
        double c = coins[n1]->l1->c;
        double dist = a*Q.x + b*Q.y +c;
        cv::Point2i R(coins[n1]->K);
        if (abs(dist) > dc) { // PQ // autre ligne du coin P
          a = coins[n1]->l2->a;
          b = coins[n1]->l2->b;
          c = coins[n1]->l2->c;
          R = coins[n1]->H;
        }
        // distance de R à la droite (i)
        double dR = a*R.x + b*R.y + c;
        // dR > 0 : vecteur normal dirigé vers l'intérieur 
        if (dR < 0) lgW = -lgW;
        C.x = Q.x + a*lgW;
        C.y = Q.y + b*lgW;
        D.x = P.x + a*lgW;
        D.y = P.y + b*lgW;
        if (nbpts == 3) {
          // remplacer C ou D par le point n3
          cv::Point2i K(coins[n3]->sommet);
          if (std::abs(C.x - K.x) < dc && std::abs(C.y - K.y) < dc ) {
            C = K;
          } else if (std::abs(D.x - K.x) < dc && std::abs(D.y - K.y) < dc ) {
            D = K;
          }
        }
        if (C.x >= 0 && C.x < image.cols && C.y >= 0 && C.y < image.rows 
        && D.x >= 0 && D.x < image.cols && D.y >= 0 && D.y < image.rows ) {
          pts[2][0] = C.x;
          pts[2][1] = C.y;
          pts[3][0] = D.x;
          pts[3][1] = D.y;
          nbpts = 4;   // on vient de compléter
        } else {
          if(printoption > 1) std::cout<<" hors de l'ecran"<<std::endl;
        }
      } //if lgW > dc
    }

    if (nbpts == 4){
      // vérifier qu'aucun autre coin d'une autre carte n'est dans celle-ci
      // réordonner les sommets ABCD : partir du premier point A= pts[0]
      // si le segment 0-1 n'est pas un coté de carte (longueur > hauteurcarte), inverser 1 et 2
      // si 2-3 est dans le même sens que 0-1, inverser 2 et 3 
      int lg = (pts[1][0] - pts[0][0])*(pts[1][0] - pts[0][0]) + (pts[1][1] - pts[0][1])*(pts[1][1] - pts[0][1]);
      lg = std::sqrt(lg);
      if (lg > 5*maconf.hauteurcarte/4) { // inverser les points 1 et 2
        int x= pts[1][0]; int y = pts[1][1];
        pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1];
        pts[2][0] = x; pts[2][1] = y;
      }
      // CD en sens inverse de AB ?
      // calcul du produit scalaire AB*CD
      int ps = (pts[1][0] - pts[0][0])*(pts[3][0]-pts[2][0]) 
          + (pts[1][1] - pts[0][1])*(pts[3][1]-pts[2][1]);
      if (ps > 0) { // inverser 2 et 3
        int x= pts[3][0]; int y = pts[3][1];
        pts[3][0] = pts[2][0]; pts[3][1] = pts[2][1];
        pts[2][0] = x; pts[2][1] = y;
      }
      for (int n = 0; n < Coins.size(); n++){
        const auto& cn = Coins[n];
        if (coins.size() > 0 && &cn == (coins[0])) continue; // un coin de la carte analysée
        if (coins.size() > 1 && &cn == (coins[1])) continue;
        if (coins.size() > 2 && &cn == (coins[2])) continue;
        if (coins.size() > 3 && &cn == (coins[3])) continue;
        cv::Point2i P(cn.sommet);
        // ignorer ce coin s'il était déjà dans la frame précédente
        if (estvideo) {
          bool trouve = false;
          for (auto ucp : cartesPrec) {
            if (! ucp.contient(P, maconf.deltacoin)) continue; // ignorer ce coin à l'intérieur de la nouvelle carte
          }
          // P à l'intérieur de la carte (rectangle) ?
          // la projection de P sur chacun des 4 coté doit être à l'intérieur du segment
          // (c'est un rectangle, il suffirait de considérer les deux premiers cotés)
          // coté UV : UP.UV doit être positif et inférieur à UV.UV
          bool estdehors = false;
          int x = pts[3][0];
          int y = pts[3][1];
          cv::Point2i U(x,y);
          cv::Point2i V;
          int ps;
          for (int i= 0; i < 4; i++){
            V = cv::Point2i(pts[i][0], pts[i][1]);
            //ps = (P.x - U.x)*(V.x - U.x) + (P.y - U.y)*(V.y - U.y);
            ps = unvecteur(U,P)*unvecteur(U,V);
            if (ps < 3*maconf.hauteurcarte) {
              estdehors = true; break;}
            //if (ps > (V.x - U.x)*(V.x - U.x) + (V.y - U.y)*(V.y - U.y) - 2*maconf.hauteurcarte) {
            if (ps > unvecteur(U,V)*unvecteur(U,V) - 2*maconf.hauteurcarte) {
              estdehors = true; break;
            }
            U = V;
          }
          if (!estdehors ){
            nbpts = 0;
            if(printoption > 1) std::cout<< " coin "<< n << P <<" dans la carte "<<std::endl; 
          }
        }
      }
      if (nbpts == 4) {
        //        comparer aux cartes du pli en cours
        //        comparer à une carte (C)du pli en cours
        //        comparer chaque sommet aux 4 sommets de la carte (C)
        //        obtenir l'écart minimum
        //        obtenir pour les 4 sommets de (C) : ecart minimum et maximum des écarts minimaux
        //        obtenir l'écart global mini et maxi des minima 
        //        si l'écart maxi est faible, ignorer cette nouvelle carte 
        //        sinon : décoder et ajouter la carte au pli en cours
        float dmin2 = image.cols + image.rows;
        int iproche;
        for (int i = 0; i < monpli.nbcartes; i++){ // chaque carte du pli en cours
          float dmax1 = 0; // distance de la carte analysée à la carte du pli
          for (int k = 0; k < nbpts; k++){ // chaque sommet de la carte analysée
            unpoint M(pts[k][0], pts[k][1]); // un point de la carte en cours
            float dmin1 = image.cols + image.rows;
            for (int j = 0; j< 4; j++){ // chaque sommet de la carte du pli en cours
              unpoint S(monpli.cartes[i].sommet[j]) ;
              float d = unvecteur(M,S).lg();
              if (d < dmin1) dmin1 = d;
            }
            if (dmax1 < dmin1) dmax1 = dmin1;
          }
          // dmax1 = distance de la carte analysée à la carte i du pli
          if (dmin2 > dmax1) {dmin2 = dmax1; iproche = i;}
        }
        int numcol= -1;
        int valcarte = 0;
        bool ajouteraupli = false;
        // carte analysée proche d'une carte du pli ?
        if (dmin2 < maconf.hauteurcarte/4) {
          numcol = monpli.cartes[iproche].couleur;
          valcarte = monpli.cartes[iproche].valeur;
          carte.couleur = numcol;
          carte.valeur = valcarte;
          if (printoption > 1){
            std::cout<<"carte proche d'une carte du pli en cours "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
          }
        }
        else {  // Décoder cette nouvelle carte
          valcarte = decoderCarte(image, pts, maconf, numcol);
          // vérifier qu'elle n'est pas déjà dans la distribution
          // puis l'ajouter dans la distribution

          carte.couleur = numcol;
          carte.valeur = valcarte;
          if (numcol >= 0 && valcarte > 0 && valcarte <= 13) { // on a trouvé
            bool duplique = false;
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 13; j++) {
                if ( numcol == distribution[i][j][0] && valcarte == distribution[i][j][1]) {
                  std::cout<<" cette carte ("<<couleurcarte[numcol]<<" "<<valcarte<<")"<<" est deja dans le pli "
                    <<j+1<<" du joueur "<<i<<std::endl;
                  duplique = true; break;
                }
              }
              if (duplique) break;
            }
          }
          // mémoriser la valeur obtenue sur tous les coins de la carte
          // uniquement si on a trouvé couleur et valeur
          if (numcol >= 0 && valcarte > 0 && valcarte <= 13) {
            for (auto coin : carte.coins){
              //if (coin.numCarte != nca) continue; // pas un coin de cette carte
              coin->couleur = numcol;
              coin->valeur = valcarte;
              if (valcarte > 10) coin->estunRDV = true;
            }
          }
          // vérifier la présence des coins de cette carte dans la frame pécédente ou le pli en cours
          // avec couleur et valeur différentes
          // ignorer les coins correspondants
          bool estDansPrec(false);
          for (auto cn : coins){
            cv::Point2i P(cn->sommet);
            for (auto ucp : cartesPrec) {
              if (ucp.couleur != carte.couleur || ucp.valeur != carte.valeur) continue;
              for (int i= 0; i<4; i++) {
                if (std::abs(ucp.sommet[i].x - P.x) > dc ) continue;
                if (std::abs(ucp.sommet[i].y - P.y) > dc ) continue;
                estDansPrec = true;
                break;
              }
              for (auto& up : ucp.coinsPrec){
                if (std::abs(up.x - P.x) > dc || std::abs(up.y - P.y) > dc ) continue;
                estDansPrec = true;
                break;
              }
              if (estDansPrec) break;
            }
            // vérifier l'absence de ce coin de cette carte dans le pli en cours
            if (!estDansPrec){
              for (int n = 0; n < monpli.nbcartes; n++){
                for (int i = 0; i<4; i++){
                  if (std::abs(monpli.cartes[n].sommet[i].x - P.x ) > dc ) continue;
                  if (std::abs(monpli.cartes[n].sommet[i].y - P.y ) > dc ) continue;
                  if (monpli.cartes[n].couleur != carte.couleur || monpli.cartes[n].valeur != carte.valeur) continue;
                  estDansPrec = true;
                  break;
                }
                if(estDansPrec) break;
              }
              if(estDansPrec) break;
            }
          }
          if (estDansPrec) {
            // désactiver tous les coins de cette carte
            if (printoption > 0) 
              std::cout<<"!!! carte invalide : "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
            for (auto& cn : coins) {
              cn->elimine = true;
              cn->couleur = -1; cn->valeur = 0;
            }
          }

          if (!estDansPrec) {
            ajouteraupli = true;
            if (printoption > 0) {
              if (numcol >= 0 && valcarte > 0 && valcarte <= 13)  // on a trouvé
                std::cout<<"==> carte : "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
              else
                std::cout<<"==> carte valeur "<<valcarte<<" couleur "<<numcol<<std::endl;
            }
          }
          
        }
        // si l'option d'analyse des coins isolés est active
        //        noter : ne pas analyser les coins pas encore analysés et sur un bord de cette carte
        if (maconf.coinsoption){
          cv::Point2i P(pts[3][0], pts[3][1]);
          for (int i = 0; i < 4; i++){
            cv::Point2i Q(pts[i][0], pts[i][1]);
            float a, b, c; // normale de la droite PQ et equation de PQ
            float lg = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
            a = (Q.y - P.y) /lg;
            b = -(Q.x - P.x) / lg;
            c = -a*P.x - b*P.y; 
            for (auto& coin:coins){
              if (coin->elimine) continue;
              if (coin->couleur >= 0 && coin->valeur > 0) continue; // déjà décodé
              cv::Point2i M(coin->sommet);
              // sommet (M) du coin n sur le segment PQ ?
              if (std::abs(a*M.x + b*M.y + c) < maconf.deltacadre ) {
                // MP.MQ < 0 --> M entre P et Q ?
                if ((P.x - M.x)*(Q.x - M.x) + (P.y - M.y)*(Q.y - M.y) < 0) {
                  coin->elimine = true;
                  break;
                }
              }
              P = Q;
            }
          }
        }

        //        
        if (ajouteraupli){
          // ajouter la carte au pli en cours
          // sauf si elle y est déjà
          for (int i = 0; i < monpli.nbcartes; i++){
            if (monpli.cartes[i].couleur == numcol && monpli.cartes[i].valeur == valcarte) {
              ajouteraupli = false;
              break;
            }
          }
        }
        if (ajouteraupli){
          
          if (monpli.nbcartes < 4) {
            int nc = monpli.nbcartes;
            monpli.cartes[nc].couleur = numcol;
            monpli.cartes[nc].valeur = valcarte;
            for (int i=0; i < 4;i++) {
              int x = pts[i][0]; int y = pts[i][1];
              monpli.cartes[nc].sommet[i].x = x;
              monpli.cartes[nc].sommet[i].y = y;
            }
            monpli.nbcartes++;
            if (printoption > 0) 
              std::cout<<" carte ajoutée au pli en cours : "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
          }
        }
      }
    }
    return;
}


