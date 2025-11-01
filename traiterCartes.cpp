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



void traiterCartes(cv::Mat image, config& maconf, std::vector<uncoin>& Coins, std::vector<uncoinPrec>& coinsPrec,
   const std::vector<ligne>&  lignes, unpli& monpli) {
  int printoption = maconf.printoption;
  if (printoption > 1) std::cout<< std::endl<<"================== recherche des nouvelles cartes ======"<<std::endl;
  bool estvideo = true;
  int epsilon = std::max(4,2*maconf.deltacadre);
  int nca = 0; // numéro de carte complète à analyser
  int nc = 0; // numéro de carte cherchée, cartes numérotées à partir de 1
  int numcol, valcarte; // couleur et valeur de carte
  int h0 = -1; // coin de la frame précédente
  int h1 = -1;
  int dc = maconf.deltacadre;
  
  int attendre = 0; // pour pouvoir actualiser l'affichage en debug 
  if(attendre) cv::waitKey(0);

  // calculer les bords des cartes
  std::vector<unbord> bords;
  nc = 1;
  while(nc >=0) {
    bool existecarte(false);
    for (int i = 0; i < Coins.size(); i++){
      uncoin& c1 = Coins[i]; 
      if (c1.numCarte < nc) continue;
      existecarte = true;
      cv::Point2i P=c1.sommet;
      for (int j = i+1; j < Coins.size(); j++) {
        uncoin& c2 = Coins[j];
        if (c2.numCarte != nc) continue;
        cv::Point2i Q=c2.sommet;
        if (std::abs(std::sqrt((Q-P).x * (Q-P).x + (Q-P).y * (Q-P).y) - maconf.hauteurcarte) > epsilon
        && std::abs(std::sqrt((Q-P).x * (Q-P).x + (Q-P).y * (Q-P).y) - maconf.largeurcarte) > epsilon ) continue;
        unbord ub(nc, c1, c2);
        bords.push_back(ub);
      }
    }
    nc++;
    if (!existecarte) nc = -1;
  } 

  // calculer les bords de cartes de la frame précédente
  std::vector<unbord> bordsPrec;
  for (int h=0; h<coinsPrec.size(); h++){
    uncoinPrec& up = coinsPrec[h];
    cv::Point2i P; P.x = up.x; P.y = up.y;
    int nc = up.numcarte;
    for (int hh = h+1; hh<coinsPrec.size(); hh++){
      uncoinPrec& up2 = coinsPrec[hh];
      if (up2.numcarte != nc) continue;
      cv::Point2i Q; Q.x = up2.x; Q.y = up2.y;
      int d = std::sqrt( (Q-P).x * (Q-P).x + (Q-P).y * (Q-P).y);
      if (std::abs(d - maconf.hauteurcarte) > epsilon 
      && std::abs(d - maconf.largeurcarte) > epsilon ) continue;
      unbord ub(0);
      ub.numcarte = nc;
      ub.A = P; ub.B = Q;
      bordsPrec.push_back(ub);
    }
  }

  nc = 0;
  // déterminer si la carte était déjà dans la frame précédente

  // dans la frame précédente si au moins deux coins y étaient
  // la distance entre un sommet de la carte et un coin de la frame précédente
  //         doit être commune à tous les sommets de la carte
  for (int n = 0; n < Coins.size(); n++){
    const auto& cn = Coins[n];
    if (cn.elimine) continue;
    if (cn.numCarte <= nc) continue; // coin d'une carte déjà recherchée
    if (cn.couleur >= 0 && cn.valeur > 0) continue; // coin déjà analysé
    // nouvelle carte
    nc = cn.numCarte; // numéro de carte suivante de cette frame

#ifdef SURLIGNE    
    // ignorer s'il est au milieu d'une ligne
    bool coinSurLigne(false);
    for (const auto& ln:lignes){
      if (ln.contient(cn.sommet)) {coinSurLigne = true; break;}
    }
    if (coinSurLigne) continue;
#endif

    // ignorer s'il est sur un bord de carte
    bool coinSurBord(false);
    for (auto& ub:bords){
      //if (ub.numcarte == nc) continue;
      if (ub.contient(cn.sommet, epsilon)) {
          coinSurBord = true;   break;}
    }
    if(coinSurBord) continue; // ignorer le coin qui est sur un bord d'une carte

    // ignorer le coin s'il est sur un bord de carte de la frame précédente
    for (auto& ub:bordsPrec){
      if (ub.contient(cn.sommet, epsilon)) {
          coinSurBord = true;   break;}
    }
    if(coinSurBord) continue; // ignorer le coin qui est sur un bord d'une carte précédente

    // noter que par construction des numéros de cartes, 
    //  les numéros de cartes sont en ordre croissant des numéros de coins
    // considérer tous les coins de cette carte nc
    // vérifier qu'on a au moins 2 sommets, limitant une hauteur ou une largeur de carte
    bool existeHauteur(false), existeLargeur(false);
    cv::Point2i P1(cn.sommet), P2;
    int nbc = 0;
    int nbCoinsTrouves = 0;  // comptage des coins trouvés dans la frame précédente
    for (int m = 0; m < Coins.size(); m++) {
      uncoin& coin = Coins[m];
      if (coin.numCarte != nc) continue;
      cv::Point2i P = coin.sommet;
#ifdef SURLIGNE      
      // ignorer s'il est au milieu d'une ligne
      coinSurLigne =false;
      for (const auto& ligne:lignes){
        if (ligne.contient(cn.sommet)) {coinSurLigne = true; break;}
      }
      if (coinSurLigne) continue;
#endif      
      // ignorer un coin qui est sur un bord d'une carte de cette frame
      coinSurBord =false;
      for (auto& ub:bords){
        //if (ub.numcarte == nc) continue;
        if (ub.contient(P, dc)) {
           coinSurBord = true;   break;}
      }
      if(coinSurBord) continue; // ignorer le coin qui est sur un bord d'une carte

      nbc++;
      if (nbc == 1) P1 = P;
      else {
        float lg = std::sqrt((P1-P).x * (P1-P).x + (P1-P).y * (P1-P).y) +1;
        if (std::abs(lg - maconf.hauteurcarte) < epsilon)
          existeHauteur = true;
        if (std::abs(lg - maconf.largeurcarte) < epsilon)
          existeLargeur = true;
      }
      if (nbc == 2) P2 = P;
      if(nbc > 2){
        float lg = std::sqrt((P2-P).x * (P2-P).x + (P2-P).y * (P2-P).y) +1;
        if (std::abs(lg - maconf.hauteurcarte) < epsilon)
          existeHauteur = true;
        if (std::abs(lg - maconf.largeurcarte) < epsilon)
          existeLargeur = true;
      }
    }

    if (nbc >= 2 && (existeHauteur || existeLargeur)) {
      h0 = -1;  // a priori pas présente dans la frame précédente
      h1 = -1;
      bool estNouveau = false; // nouveau sommet dans cette frame ?
      cv::Point2i  P1(cn.sommet);
      int dmin(100), dmax(0);
      std::string nomcarte="";
      int numcol(-1), valcarte(0); // couleur et valeur d'une carte antérieure (= d'une frameprécédente)
      for(int  m= n; m< Coins.size(); m++ ){ // chercher tous les coins de cette carte 
        const auto& cm = Coins[m];
        if (cm.numCarte != nc) continue; // coin d'une autre carte (analysée déjà ou à venir) )
        cv::Point2i PT(cm.sommet);
        // rechercher dans le vecteur coinsPrec des coins  mémorisés des frames précédentes
        bool trouve(false);
        h0 = -1;
        dmin = 100; // distance minimale entre le coin PT de la carte (nc) et la frame précédente
        
        for (int h = 0; h < coinsPrec.size(); h++){
          uncoinPrec up = coinsPrec[h];
          cv::Point2i QT(up.x, up.y);
          if (std::abs(PT.x - QT.x) > epsilon ) continue;
          if (std::abs(PT.y - QT.y) > epsilon ) continue;
          if (h1 >= 0) { // on a trouvé un coin précédent proche de cette carte
            if (numcol >= 0 && valcarte > 0) {
              if (numcol != up.couleur) continue; // couleur différente
              if (valcarte != up.valeur) continue; // valeur différente :pas un coin de la même carte antérieure
            }
          }
          int d = std::max(std::abs(PT.x - QT.x), std::abs(PT.y - QT.y));
          if (dmin > d ) { 
            dmin = d;
            h0 = h;
          }
        }
        if (h0 >= 0){ // on a trouvé un coin précédent proche du coin PT de cette carte
          if (h1 < 0 || numcol < 0 || valcarte <= 0){ // pas encore trouvé un coin proche dans la frame précédente
            // mémoriser couleur et valeur du coin précédent
            numcol = coinsPrec[h0].couleur;   // couleur et valeur de la carte antérieure proche
            valcarte = coinsPrec[h0].valeur;
            h1 = h0;
          }
        }
        if (h0 < 0){ // le coin PT n'est pas trouvé dans la frame précédente
          // mémoriser qu'il y a un nouveau sommet de la carte
          // et attendre la recherche de tous les sommets de la carte
          estNouveau = true;
        }
        else {
          if (dmin < epsilon) nbCoinsTrouves++;
         if (dmax < dmin) dmax = dmin; // maximum entre les coins de la carte et de la carte de la frame précédente
        }
        // if (h1 >= 0 && dmax > maconf.hauteurcarte / 10) break; // un coin de la carte est trop loin de la carte antérieure
      } // for m

      // si un des sommets de la carte est nouveau dans cette frame
      // et si on a trouvé une carte correspondant à celle-ci dans la frame précédente
      //  qui n'a pas de couleur et valeur
      //  ==> ne pas reconnaitre que c'est la même carte
      if (nbCoinsTrouves < 2 && estNouveau && h1 >= 0) {
        if (numcol < 0 || valcarte <= 0) dmax = 100; // carte antérieure pas identifiée
      }
      if(nbCoinsTrouves < 2 && estNouveau) dmax = 100; // nouveau coin donc nouvelle carte
      if (nbCoinsTrouves >= 2 || ( h1 >= 0 && dmax < maconf.hauteurcarte / 10)) { // carte proche d'une carte identifiée de la frame précédente
        nomcarte = "";
        if (numcol >= 0 && numcol <= 3) nomcarte = nomcouleur[numcol];
        if (coinsPrec[h1].valeur > 0 && coinsPrec[h1].valeur < 14){
          nomcarte += " ";
          nomcarte += nomval[coinsPrec[h1].valeur];
        }
        if(printoption > 1)  std::cout<< " carte "<< nc << " ("<< nomcarte 
          << ") déjà dans la frame précédente." << std::endl;
        // noter qu'il est inutile d'analyser ce coin
        //cn.couleur = numcol;
        //cn.valeur = valcarte;
        // ainsi que tous les coins de cette carte
        for (auto &coin:Coins){
          if (coin.numCarte == nc) { coin.couleur = numcol; coin.valeur = valcarte;}
        } 
        continue; // for n
      }
    } else {
      // moins de 2 coins ou absence de hauteur et largeur de carte
      // on considère que c'est une nouvelle carte
    }

    // nouveau coin, nouvelle carte dans cette frame
    // vérifier que c'est la seule nouvelle carte
    // analyser les coins à partir du coin n+1 nouveaux 
    //   de la même carte --> les angles de la carte
    //   d'une autre carte --> erreur : plusieurs nouvelles cartes
    if (printoption > 0) std::cout<< "nouvelle carte "<< nc<<" nouveau sommet "<< n <<P1<<std::endl;
    bool estcarte(false);
    for (int m =n+1; m< Coins.size(); m++){
      const auto& cm = Coins[m];
      if (cm.numCarte != nc) continue;
      if (cm.elimine) continue;
      cv::Point2i P2(cm.sommet);
      // coin de la même carte ?
      if (nc == cm.numCarte) { // même carte
        // mémoriser
        if (printoption > 0) std::cout << " autre sommet "<< m<< " "<< P2<<std::endl;
        // distance entre les deux points proche de longueur ou largeur de carte
        if (!estcarte) {
          double dist = std::sqrt((P1.x - P2.x)*(P1.x - P2.x) + (P1.y - P2.y)*(P1.y - P2.y));
          if (std::abs(dist - maconf.hauteurcarte) < maconf.hauteurcarte / 10) estcarte =true;
          if (std::abs(dist - maconf.largeurcarte) < maconf.hauteurcarte / 10) estcarte =true;
        }
      }               
    } // for m

    bool carteDansPli(false);
    if (estcarte) { // carte avec au moins deux coins et un bord et absente de la frame précédente
      if (printoption > 0) std::cout<<" carte complète "<< nc<<std::endl;
      // TODO : vérifier si la carte est dans le pli en cours
      // chaque coin de cette carte doit être proche d'une carte du pli en cours

      uncoin cn; 
      for (auto& cx:Coins) {
        if (cx.numCarte != nc) continue;
        cn = cx;
        break;
      }
      cv::Point2i P(cn.sommet);
      unecarte carte;
      int isom1; // indice du sommet de la carte du pli qui est proche du premier coin
      bool trouveCarteDuPli(false);
      for (auto cartex:monpli.cartes) {
        for ( isom1= 0; isom1<4; isom1++){
          cv::Point2i Q(cartex.sommet[isom1]);
          if (std::abs((Q-P).x ) <= epsilon && (std::abs((Q-P).y ) <= epsilon)) {
            // on a trouvé un sommet d'une carte du pli qui est proche du coin
            trouveCarteDuPli = true;
            carte = cartex;
            break;
          }
        }
        if (trouveCarteDuPli) break;
      }
      if (trouveCarteDuPli) {
        // vérifier que les autres coins sont proches d'un (autre) sommet de la même carte du pli
        carteDansPli=true; // au moins un coin dans le pli 
        for (auto& cm: Coins){
          if (cm.numCarte != nc) continue;
          bool coinDansPli(true);
          for (int j = 0; j< 4; j++){
            if (j == isom1) continue; // ce sommet est déjà proche de la carte du pli
            cv::Point2i R(cm.sommet);
            // proche d'un sommet de la carte du pli ?
            bool estproche(false);
            for (int k=0; k < 4; k++){
              if (std::abs((R - carte.sommet[k]).x) < epsilon &&  std::abs((R - carte.sommet[k]).y) < epsilon  ){
                estproche=true;
                break;
              }
            }
            if (!estproche) {
              carteDansPli = false;
              break;
            }
          } // sommets de la carte
          if (!coinDansPli){ carteDansPli = false; break;}
        } // for cm::Coins
      } 
      if (carteDansPli) {
        estcarte = false;
        std::string nomcarte = nomcouleur[carte.couleur];
        nomcarte +=  std::string(" ") + std::string(nomval[carte.valeur]);
        if (printoption > 0) std::cout<<" carte "<<nc <<" ("<<nomcarte<<") dans le pli en cours "<< std::endl;
        // valoriser tous les coins de cette carte
        for (auto& coin : Coins){
          if (coin.numCarte != nc) continue;
          //if (coin.elimine) continue;
          coin.couleur = carte.couleur;
          coin.valeur = carte.valeur;
        }
      }
    } // if(estcarte)

      
    
    
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
        for (int i = 0; i < Coins.size(); i++){
          if (Coins[i].numCarte != nc) continue;
          cv::Point2i P(Coins[i].sommet);
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
          nc = 0;
        }
        else {
          // vérifier la carte nca
          for (int i = 0; i < Coins.size(); i++){
            if (Coins[i].numCarte != nca) continue;
            cv::Point2i P(Coins[i].sommet);
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
            nca = nc;
            //nc = 0;
          }
        } // else
      } else { // première nouvelle carte
        nca = nc;
      }
    } else if(!carteDansPli) { // coin (n) absent de la frame précédente et non opposé à un autre coin
      if(printoption > 1) std::cout<< " carte "<<nc<< " a un seul coin: "<< n<<cn.sommet<<std::endl;
    }
  } // for n

  
  if (nca){
    if (printoption > 1) std::cout<<"==> analyse de la carte "<<nca<<std::endl;
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
    cv::Point2i P, Q, U, V;
    int n1(-1), n2(-1), n3(-1), n4(-1);
    int nbpts;
    //int i, j;
    int dc = std::max(4, maconf.deltacadre);
    // rechercher le coin le plus à gauche (et le plus haut s'il y en a plusieurs)
    int xmin(12345); // une grande valeur
    for (auto& cn:Coins) {
      if (cn.numCarte != nca) continue; // autre carte
      if (cn.elimine) continue; // coin éliminé
      if (cn.sommet.x <= xmin ) {
        xmin = cn.sommet.x;  //limite gauche de la carte
      }
    }
    int ymin(12345); // une grande valeur
    for (int n = 0; n < Coins.size(); n++) {
      const auto& cn = Coins[n];
      if (cn.numCarte != nca) continue; // autre carte
      if (cn.elimine) continue; // coin éliminé
      if (cn.sommet.x <= xmin ) {    // un des coins les plus à gauche
        if (cn.sommet.y < ymin ) {
          n1 = n;
          ymin = cn.sommet.y;
        }
      }
    }
    P = Coins[n1].sommet;
    if(printoption > 1) std::cout<<" coin "<< n1 <<" "<<P<<std::endl;
    // rechercher un coin opposé sur le premier coté du coin P puis sur le deuxième
    n2 = -1;
    int lgref, lgref2;
    float a,b,c, aa,bb,cc;
    cv::Vec4i ln = Coins[n1].l1->ln;
    cv::Vec4i ln2 = Coins[n1].l2->ln;
    a = Coins[n1].l1->a; b = Coins[n1].l1->b; c = Coins[n1].l1->c;
    aa = Coins[n1].l2->a; bb = Coins[n1].l2->b; cc = Coins[n1].l2->c;
    for (int i=0; n2< 0 && i < 2; i++) {
      n2 = -1;
      float ecart, ecartmin(dc);
      for (int n = 0; n < Coins.size(); n++) {
        const auto& cn = Coins[n];
        if (cn.numCarte != nca || cn.elimine) continue;
        if (n == n1) continue;
        Q = cn.sommet;
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
        ln = Coins[n1].l2->ln;
        aa = a; bb=b; cc=c;
        a = Coins[n1].l2->a; b = Coins[n1].l2->b; c = Coins[n1].l2->c;
      }
    } // while n2


    if (n2 < 0) {
      // aucun coin opposé au coin P
      nbpts = 1;
    } else {
      Q = Coins[n2].sommet;
      if(printoption > 1) std::cout<<" coin opposé "<< n2 <<" "<<Q<<std::endl;
      // on a deux sommets de la carte P (n1) et Q (n2)
      // rechercher ou calculer les deux autres sommets U et V
      // la ligne commune est : coin n1 indice i, ligne ln, coefficient a b c
      // rechercher le coin U sur l'autre ligne (j) du coin P
      cv::Vec4i ln = ln2;
      float ecmin = 3;
      n3 = -1;
      for (int n = 0; n < Coins.size(); n++) {
        const auto& cn = Coins[n];
        if (cn.numCarte != nca || cn.elimine) continue;
        if (n == n1 || n == n2) continue;
        U = cn.sommet;
        float dist = aa*U.x + bb*U.y + cc;
        if (std::abs(dist) > 1) continue; // U n'est pas sur l'autre ligne (que PQ) j du coin P
        float lg = std::sqrt((U.x - P.x)*(U.x - P.x) + (U.y - P.y)*(U.y - P.y));
        if (std::abs(lg - lgref2) > ecmin) continue; // U n'est pas à distance convenable de P (hauteur ou largeur de carte)
        if(std::abs(lg - lgref2) < ecmin) {
          ecmin = std::abs(lg - lgref2);
          n3 = n;
        }
      }
      if(n3 >= 0) {
        U = cv::Point2i(Coins[n3].sommet);
        if(printoption > 1) std::cout<<" coin trois "<< n3 <<" "<<U<<std::endl;
      }
      ligne *l1 = Coins[n2].l1;
      ligne *l2 = Coins[n2].l2;
      // rechercher le sommet V sur le coté orthogonal à PQ du coin Q (n2)
      float ps = a*Coins[n2].l1->a + b*Coins[n2].l1->b;
      if(std::abs(ps) < 0.5) {
        l2 = l1;
        l1 =Coins[n2].l2;
      }
      // ii est la ligne // PQ  jj est la ligne ortogonale à PQ
      // rechercher un coin V de cette carte tel que QV = lgref2 (à deltacadre près)
      // distance de V à la droite PQ = lgref2   et QV orthogonal à PQ
      n4 = -1;
      ecmin = 3;
      for (int n = 0; n < Coins.size(); n++) {
        const auto& cn = Coins[n];
        if (cn.numCarte != nca || cn.elimine) continue;
        if (n == n1 || n == n2 || n == n3) continue;
        V = cn.sommet;
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
        V = cv::Point2i(Coins[n4].sommet);
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
      cv::Point2i P(Coins[n1].sommet);
      cv::Point2i Q(Coins[n2].sommet);
      // vérifier que PQ est la hauteur ou la largeur de carte
      cv::Point2i C; // autre sommet opposé au point P
      cv::Point2i D; // autre sommet opposé au point Q
      // PQ hauteur ou largeur de carte ?
      lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
      lg = std::sqrt(lg2);
      if (lg > 5*maconf.hauteurcarte/4 && nbpts == 3){
        // c'est la diagonale.
        // s'il y a un 3ème sommet, remplacer le 2ème sommet
        Q = Coins[n3].sommet;
        pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1]; 
        lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
        lg = std::sqrt(lg2);
        int w = n2; n2 = n3; n3 = w;
      }
      if (std::abs (lg - maconf.hauteurcarte) <= dc ) lgW = maconf.largeurcarte;
      else if (std::abs (lg - maconf.largeurcarte) <= dc ) lgW = maconf.hauteurcarte;
      else lgW = 0;
      if (lgW > dc) {

        double a = Coins[n1].l1->a;
        double b = Coins[n1].l1->b;
        double c = Coins[n1].l1->c;
        double dist = a*Q.x + b*Q.y +c;
        cv::Point2i R(Coins[n1].K);
        if (abs(dist) > dc) { // PQ // autre ligne du coin P
          a = Coins[n1].l2->a;
          b = Coins[n1].l2->b;
          c = Coins[n1].l2->c;
          R = Coins[n1].H;
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
          cv::Point2i K(Coins[n3].sommet);
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
        if (cn.numCarte == nca) continue; // un coin de cette carte
        cv::Point2i P(cn.sommet);
        // ignorer ce coin s'il était déjà dans la frame précédente
        if (estvideo) {
          bool trouve = false;
          for (int h = 0; h < coinsPrec.size(); h++){
            uncoinPrec up = coinsPrec[h];
            if (std::abs(P.x - coinsPrec[h].x) <= 1 
            && std::abs(P.y - coinsPrec[h].y) <= 1 ) {
              trouve = true;
              break;
            }
          }
          if (!trouve) continue; // ignorer ce coin à l'intérieur de la nouvelle carte
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
          if (printoption > 1){
            std::cout<<"carte proche d'une carte du pli en cours "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
          }
        }
        else {  // Décoder cette nouvelle carte
          valcarte = decoderCarte(image, pts, maconf, numcol);
          // vérifier qu'elle n'est pas déjà dans la distribution
          // puis l'ajouter dans la distribution

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
          ajouteraupli = true;
          if (printoption > 0) {
            if (numcol >= 0 && valcarte > 0 && valcarte <= 13)  // on a trouvé
              std::cout<<"==> carte : "<<couleurcarte[numcol]<<" "<<valcarte<<std::endl;
            else
              std::cout<<"==> carte valeur "<<valcarte<<" couleur "<<numcol<<std::endl;
          }
        }
        // mémoriser la valeur obtenue sur tous les coins de la carte
        // uniquement si on a trouvé couleur et valeur
        if (numcol >= 0 && valcarte > 0 && valcarte <= 13) {
          for (auto &coin:Coins){
            if (coin.numCarte != nca) continue; // pas un coin de cette carte
            coin.couleur = numcol;
            coin.valeur = valcarte;
            if (valcarte > 10) coin.estunRDV = true;
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
            for (auto& coin:Coins){
              if (coin.elimine) continue;
              if (coin.couleur >= 0 && coin.valeur > 0) continue; // déjà décodé
              cv::Point2i M(coin.sommet);
              // sommet (M) du coin n sur le segment PQ ?
              if (std::abs(a*M.x + b*M.y + c) < maconf.deltacadre ) {
                // MP.MQ < 0 --> M entre P et Q ?
                if ((P.x - M.x)*(Q.x - M.x) + (P.y - M.y)*(Q.y - M.y) < 0) {
                  coin.elimine = true;
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
          }
        }
      }
    }
  } // traitement carte nca
return;
}
