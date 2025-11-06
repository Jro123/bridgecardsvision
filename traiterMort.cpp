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

// décoder les cartes du mort et enregistrer sa main
void traiterMort(config& maconf, cv::Mat imaMort, unecarte *carteMort) {
  int printoption = maconf.printoption;
  int waitoption = maconf.waitoption;
  std::chrono::duration<double> duree;
  auto t1 = std::chrono::high_resolution_clock::now();

// procéder de gauche à droite
// extraire la couleur d'une colonne de 1 pixel à gauche
  int icarteMort(1); // indice des 13 cartes du mort. première carte déjà jouée
  cv::Scalar couleurFond; // couleur du fond
  cv::Rect r;
  cv::Mat lig;
  cv::Scalar m0, m1, m2;
  int numcol(-1), valcarte;   // couleur (0 à 3) et valeur de carte (1 à 13)
  int xcol(0), ycol(0); // position de la colonne de cartes
  int xbas; // position du coin le plus bas de la carte

  r.x = 0;
  r.y = 0;
  r.width = 1;
  r.height = imaMort.rows;
  lig = imaMort(r); couleurFond = cv::mean(lig); // couleur du fond

  // !!!!! spécifique pour mise au point avec une vidéo de FUNBRIDGE
  // vérifier que c'est sans effet sur une vidéo réelle
  // remplacer le morceau de plaquette contenant S et nom d'utilisateur funbridge
  // à gauche d'une éventuelle carte
  //   par la couleur du fond
  // rechercher la présence d'une carte en bas de l'image (au dessus de la plaquette)
  cv::Vec3b coul(couleurFond[0], couleurFond[1], couleurFond[2]);
  int x= 0;
  for (x = 0; x < imaMort.cols; x++){
    m2 = imaMort.at<cv::Vec3b>(imaMort.rows - 12,x);
    if (m2[0] > 60 + couleurFond[0]) {  // il y a une carte en bas
      // nettoyer à gauche de la carte
      int xlim = x;
      for (int y=imaMort.rows - 10; y < imaMort.rows; y++){
        for (int x=0; x < xlim; x++){
          imaMort.at<cv::Vec3b>(y,x) = coul;
        }
      }
      break;
    }
  }
  // remplacer les deux dernières lignes par la couleur du fond
  for (int y=imaMort.rows - 2; y < imaMort.rows; y++){
    for (int x=0; x < imaMort.cols; x++){
      imaMort.at<cv::Vec3b>(y,x) = coul;
    }
  }
  // trouver la ligne noire en haut (10 lignes) et à gauche (10 pixels) de imaMort
  // puis remplacer les pixels noirs par la couleur du fond
  r.x = 0; r.width = 10;
  r.height = 1;
  for (r.y = 0; r.y < 10; r.y++){
    lig = imaMort(r); m1 = cv::mean(lig);
    if (m1[0] < 20 && m1[1] < 20 && m1[2] < 20) { // ligne noire
      for (int x = 0; x < imaMort.cols; x ++) {
        cv::Scalar pix = imaMort.at<cv::Vec3b>(r.y, x);
        if (pix[0] < 20 && pix[1] < 20 && pix[2] < 20)
            imaMort.at<cv::Vec3b>(r.y,x) = coul;
      }
      break;
    }
  }
  // ajouter une bordure supérieure et inférieure de la couleur du fond
  cv::Mat imaW;
  cv::copyMakeBorder(imaMort, imaW, 5,5,0,0, cv::BORDER_CONSTANT, couleurFond);
  imaMort = imaW.clone();

  // find de la partie spécifique à une vidéo d'un replay de FUNBRIDGE

  cv::Mat mortCopie = imaMort.clone(); // pour affichages de mise au point 
  int ybas = imaMort.rows - 1; // position du bas de la colonne de cartes

  bool estPremier = true; // indique qu'on analyse la carte la plus basse de la colonne
  int pts[4][2];  // les 4 sommets d'une carte, entière ou limitée à la partie supérieure
  xcol = 1; // position gauche de la colonne de cartes
  //
  // analyser les cartes, en repérant le changement de colonne de cartes et la fin des colonnes
  // nettoyer à gauche de chaque nouvelle colonne
  //
while(true) {
  r.y = 6; // à cause du trait supérieur de la vidéo FUNBRIDGE et de l'ajout de la bordure de 5 lignes
  if (estPremier) { // on traite la carte du bas, qui est complètement visible
    // nettoyer à gauche de la colonne de cartes
    cv::Rect rr;
    rr.x = 0; rr.width = xcol; rr.y = 0; rr.height = imaMort.rows;
    cv::rectangle(imaMort, rr, couleurFond, cv::FILLED);
    r.height = imaMort.rows - r.y - 1; // toute la hauteur de la colonne de cartes
  } else r.height = std::max(pts[2][1], pts[3][1]); // partie au dessus de la carte qu'on vient d'analyser
  // rechercher une colonne plus claire (au moins en vert) -> première colonne de cartes (atout)
  // puis colonnes suivantes
  r.width = 1;
  r.height = std::min(r.height, imaMort.rows - r.y - 1);
  for (r.x = xcol; r.x < imaMort.cols - maconf.largeurcarte; r.x++){
    lig = imaMort(r); m1 = cv::mean(lig); // couleur de cette colonne de 1 pixel
    if (m1[1] - couleurFond[1] > 40) {xcol = r.x; break;}
  }
  // extraire un rectangle de largeur de moitié de largeur de carte
  r.width = maconf.largeurcarte / 2;
  lig = imaMort(r);
  // trouver le bas de la dernière carte de la colonne. coin bas gauche (ou au dessous)
  r.y = imaMort.rows - 1;
  if (!estPremier) r.y = std::max(pts[2][1], pts[3][1]);
  r.height = 1;
  while (r.y > 0) {
    lig = imaMort(r); m2 = cv::mean(lig);
    if (m2[1] - couleurFond[1] > 10) {ybas = r.y; break;}
    r.y--;
  }
  if (r.y <= 6 + maconf.taillegros / 2) {  // 0 --> 6 à cause de la vidéo FUNBRIDGE
    // on a trouvé toutes les cartes de cette colonne
    //xcol += 5 + maconf.largeurcarte;
    estPremier = true; // pour passage à la colonne suivante (xcol)
    if (xcol > imaMort.cols - maconf.largeurcarte) break; // on est arrivé au bout des colonnes
    // nettoyer ce qui est à gauche de la nouvelle colonne
    cv::Rect rr;
    rr.x = 0; rr.width = xcol; rr.y = 0; rr.height = imaMort.rows;
    cv::rectangle(imaMort, rr, couleurFond, cv::FILLED);
    continue;
  }

  // limiter au bas de la carte. au moins 1/5 de la carte doit être visible pour déterminer sa valeur
  r.height = std::min(ybas ,maconf.hauteurcarte /5);
  r.y = std::max(6,ybas - r.height); // ignorer le trait blanc en haut de funbridge et la bordure de 5 lignes
  // rechercher la position à gauche de la carte 
  for (r.x = xcol; r.x < maconf.largeurcarte; r.x++){
    lig = imaMort(r); m1 = cv::mean(lig); // couleur de cette colonne de 1 pixel
    if (std::abs(m1[1] - couleurFond[1]) > 20) {xcol = r.x; break;}
  }

  // on a : xcol=gauche du bas de la carte   ybas= bas de la carte
  // extraire le bas de la carte un peu élargi à gauche et dessous
  r.x = std::max(0,xcol -8); r.width = 14 + maconf.largeurcarte;
  r.height = std::min(ybas+8, maconf.hauteurcarte /2);
  r.y = std::min(imaMort.rows - 1, ybas + 8) - r.height;
  r.y = std::max(1, r.y);
  if (printoption > 0) tracerRectangle(r, mortCopie, "Mort", cv::Scalar(0,0,0));

  cv::Mat imaCol = imaMort(r); // bas de la carte à analyser
  xcol = r.x; ycol = r.y; // position de imaCol dans imaMort

  std::vector<ligne> lignes;
  // rechercher les lignes  dans cette image (imaCol)
  int save = maconf.nbpoints;
  maconf.nbpoints = 5; // on recherche même des lignes très courtes
  cv::Mat gray;
  cv::cvtColor(imaCol, gray, cv::COLOR_BGR2GRAY);
  trouverLignes(maconf, gray, lignes, true);
  maconf.nbpoints = save;

  // afficher les lignes;
  if (printoption > 0) {
    for (auto ligne : lignes){
      if (ligne.ln[0] < 0) continue; // ligne éliminée
      cv::Point2i A(ligne.ln[0], ligne.ln[1]);
      cv::Point2i B(ligne.ln[2], ligne.ln[3]);
      A.x += xcol; A.y += ycol;
      B.x += xcol; B.y += ycol;
      cv::line(mortCopie,A,B,cv::Scalar(255,0,0),1);
      std::cout<< " ligne "<<A<<"->"<<B<<std::endl;
    }
  }

  // trouver les coins
  std::vector<uncoin> Coins;
  trouverCoins(maconf, lignes, Coins);
  // afficher les coins
  if (printoption > 0) {
    for(auto moncoin : Coins ){
      cv::Point2i P(moncoin.sommet);
      P.x += xcol; P.y += ycol;
      cv::circle(mortCopie, P, 2, cv::Scalar(0,0,255),-1);
    }
    if (printoption > 0) afficherImage("Mort", mortCopie);
    if (waitoption > 0) cv::waitKey(0); else cv::waitKey(1);
  }
  // rechercher la ligne longue plutot horizontale la plus basse
  // puis les lignes plutot verticales à gauche et à droite 
  // puis calculer les coins bas gauche et droite
  // si on ne trouve pas, rechercher lescoins gauche et droit sur (proche de)  cette ligne
  ligne ligneBas, ligneGauche, ligneDroite;
  int yLigneBas(0);
  int xLigneGauche(12345); // position ligne #verticale gauche
  int xLigneDroite (0); // position ligne #verticale à droite
  for (auto ligne : lignes){
    if (ligne.ln[0] < 0) continue; // ligne éliminée
    if (ligne.lg < maconf.largeurcarte /2) continue; // trop courte
    if (std::abs(ligne.a) > 0.5) continue; // pas assez horizontale
    if (ligne.ln[1] > yLigneBas) {yLigneBas = ligne.ln[1]; ligneBas = ligne;}
    if (ligne.ln[3] > yLigneBas) {yLigneBas = ligne.ln[3]; ligneBas = ligne;}
  }

  cv::Point2i P(0,0), PG(0,0), PD(0,0); // coins bas gauche et droite dans imaCol
  int xg(12345), yg(0);
  uncoin coinGauche, coinDroit;
  coinGauche.elimine = true; // a priori non trouvé
  coinDroit.elimine = true;

  // si on a trouvé la ligne du bas, rechercher une ligne plutot verticale
  // la plus à gauche (son point le plus bas le plus à gauche)
  // calculer le point d'intersection
  if (yLigneBas > 0){
    for (auto ligne: lignes){ // ligne AB
      if (ligne.ln[0] < 0) continue; // ligne éliminée
      //float ps = ligne.a * ligneBas.a + ligne.b * ligneBas.b;
      //if (std::abs(ps) > 0.5 ) continue; // pas assez verticale
      if (std::abs(ligne.b) > 0.5 ) continue;  // pas assez verticale
      if (ligne.ln[1] < ligne.ln[3]){ // B le plus bas
        if (ligne.ln[2] < xLigneGauche) {ligneGauche=ligne; xLigneGauche = ligne.ln[2];}
      } else if (ligne.ln[0] < xLigneGauche) {ligneGauche=ligne; xLigneGauche = ligne.ln[0];}
    } 
    if (xLigneGauche < imaCol.cols) { // on a trouvé une ligne verticale
      PG.x = xLigneGauche;
      PG.y = yLigneBas; // TODO : il faudrait calculer l'intersection
    }

    xLigneDroite = xLigneGauche + maconf.largeurcarte - maconf.deltacadre; 
    // rechercher le coin de carte bas droit par une ligne verticale à droite
    for (auto ligne: lignes){ // ligne AB
      if (ligne.ln[0] < 0) continue; // ligne éliminée
      if (std::abs(ligne.b) > 0.5) continue; // pas assez verticale
      if (ligne.ln[1] < ligne.ln[3]){ // B le plus bas
        if (ligne.ln[2] > xLigneDroite) {ligneDroite=ligne; xLigneDroite = ligne.ln[2];}
      } else if (ligne.ln[0] > xLigneDroite) {ligneDroite=ligne; xLigneDroite = ligne.ln[0];}
    } 
    if (xLigneDroite < imaCol.cols) {
      PD.x = xLigneDroite;
      PD.y = yLigneBas; // il faudrait calculer l'intersection des lignes 
    }
  }

  // si on n'a pas trouvé le coin gauche (PG) rechercher le coin le plus bas à gauche,
  if (PG.x == 0){ // pas encore trouvé le coin bas gauche
    for(auto moncoin : Coins ){
      P = moncoin.sommet;
      if (P.x > maconf.largeurcarte / 2) continue; // coin pas assez à gauche
      if (yLigneBas > 0) {
        if (P.x > ligneBas.ln[0]  ) continue; // coin à droite de A  de la ligne du bas AB
        if (P.x > ligneBas.ln[2]  ) continue; // coin à droite de B  de la ligne du bas AB
        float dist = ligneBas.dist(P);
        if (std::abs(dist) > maconf.deltacadre) continue; // pas sur la ligne du bas
      }
      if (P.y > yg) {yg = P.y; coinGauche = moncoin; coinGauche.elimine = false;}
      if (P.y == yg && P.x < xg) {xg = P.x; coinGauche = moncoin; coinGauche.elimine=false; PG = P;}
    }
  }
  
  cv::Point2i PGG = PG;  // coordonnées dans imaCol
  PG.x += xcol; PG.y += ycol; // coordonnées de PG dans l'image imaMort

    int xd(0), yd(0), ecart(12345);
  // si on n'a pas trouvé le coin bas droit, rechercher le coin le plus bas à droite
  if (PD.x == 0){
    for(auto moncoin : Coins ){
      P = moncoin.sommet;
      if (P.x < maconf.largeurcarte / 2) continue; //coin trop à gauche
      if (yLigneBas > 0) {
        if (P.x < ligneBas.ln[0]  ) continue; // coin à gauche de A  de la ligne du bas AB
        if (P.x < ligneBas.ln[2]  ) continue; // coin à gauche de B  de la ligne du bas AB
        float dist = ligneBas.dist(P);
        if (std::abs(dist) > maconf.deltacadre) continue; // pas sur la ligne du bas
      }
      if (P.y > yd) {yd = P.y; coinDroit = moncoin; }
      if (P.y == yd) {
        int lg = std::sqrt((P.x - PGG.x)*(P.x - PGG.x) + (P.y - PGG.y)*(P.y - PGG.y));
        if (ecart > std::abs(lg - maconf.largeurcarte)) {
          ecart = std::abs(lg - maconf.largeurcarte);
          if (ecart < maconf.deltacoin) {
            PD = P;
            coinDroit = moncoin; coinDroit.elimine = false;
          }
        }
      }
    }
  }
  cv::Point2i PDD = PD; // coordonnées dans imaCol
  PD.x += xcol; PD.y += ycol; // coordonnées dans imaMort
  // si on a un seul coin (PGG ou PDD nul), on peut calculer l'autre 
  if (PDD.x == 0 && PDD.y == 0) {  // coin bas droit non trouvé
    float a, b;
    if (PGG.x > 0 || PGG.y > 0 ){
      if (yLigneBas > 0 ){ // calculer à partir de la ligne du bas
        a = ligneBas.b; b = ligneBas.a; if (a < 0) {a = -a; b = -b;}
        PDD.x = PGG.x + a*maconf.largeurcarte;
        PDD.y = PGG.y + b*maconf.largeurcarte;
      } else if(!coinGauche.elimine) { // calculer à partir du coin bas gauche
        int lg  = coinGauche.l1->lg;
        if (lg < coinGauche.l2->lg) {
          a = std::abs(coinGauche.l2->a);
          b = std::abs(coinGauche.l2->b);
        } else {
          a = std::abs(coinGauche.l1->a);
          b = std::abs(coinGauche.l1->b);
        }
        if (b > a) {int w = a; a = b; b = w;}
        PDD.x = PGG.x + a*maconf.largeurcarte;
        PDD.y = PGG.y + b*maconf.largeurcarte;
      }
      PD.x = PDD.x + xcol; PD.y = PDD.y + ycol;
    }
  } else if (PGG.x == 0 && PGG.y == 0 ) { // calculer à partir du coin droit
    if (yLigneBas > 0 ){ // calculer à partir de la ligne du bas
      float a,b;
      a = ligneBas.b; b = ligneBas.a; if (a < 0) {a = -a; b = -b;}
      PGG.x = PDD.x - a*maconf.largeurcarte;
      PGG.y = PDD.y - b*maconf.largeurcarte;
    } else { // calculer à partir du coin droit
      float a, b;
      int lg  = coinDroit.l1->lg;
      if (lg < coinDroit.l2->lg) {
        a = std::abs(coinDroit.l2->a);
        b = std::abs(coinDroit.l2->b);
      } else {
        a = std::abs(coinDroit.l1->a);
        b = std::abs(coinDroit.l1->b);
      }
      if (b > a) {int w = a; a = b; b = w;}
      PGG.x = PDD.x - a*maconf.largeurcarte;
      PGG.y = PDD.y - b*maconf.largeurcarte;
    }
    PG.x = PGG.x + xcol; PG.y = PGG.y + ycol;
  }

  // ajuster la position du coin bas droit : à droite du coin bas gauche à distance largeurcarte
  {
    float lg = std::sqrt((PD.x - PG.x)*(PD.x - PG.x) + (PD.y - PG.y)*(PD.y - PG.y));
    if (std::abs(lg - maconf.largeurcarte) > 2*maconf.deltacadre ) {
      PD.x = PG.x + maconf.largeurcarte * float(PD.x - PG.x) / lg;
      PD.y = PG.y + maconf.largeurcarte * float(PD.y - PG.y) / lg;
      PDD.x = PGG.x + maconf.largeurcarte * float(PDD.x - PGG.x) / lg;
      PDD.y = PGG.y + maconf.largeurcarte * float(PDD.y - PGG.y) / lg;
    }
  }
  if (printoption > 0){
    cv::circle(mortCopie, PG, 4, cv::Scalar(255,255,0),1);
    cv::circle(mortCopie, PD, 4, cv::Scalar(255,255,0),1);
    afficherImage("Mort", mortCopie);
  }

  // on a les deux coins inférieurs coinGauche et CoinDroit   PG et PD
  // on peut reconstituer les 2 autres coins de la carte
  //   (uniquement pour la carte du bas de la colonne )

  if (!estPremier) {  // ce n'est pas la première carte de la colonne
    //  si ce n'est pas la première carte en bas  de la colonne rechercher les deux coins supérieurs
    //    situés sur la ligne longue (au moins la moitié de largeur de carte) la plus basse
    //     mais au dessus du haut de la carte précédente (min de pts[*][1])
    // rechercher la ligne longue du haut de carte #Horizontale la plus basse 
    //      au dessus des coins bas gauche et droit  qu'on vient de déterminer
    // coin haut gauche (droit): intersection entre l'arête #verticale du coin bas gauche (droit)
    //      et la ligne longue
    //
    // le bord supérieur de carte peut être morcelé en plusieurs lignes courtes
    //      à cause de la carte juste au dessus, qu'elle recouvre
    //      ==> accepter les lignes courtes alignées (fusionables)
    cv::Point2i HG(0,0), HD(0,0); // coin haut gauche et droit
    int ymax(0);
    int ylim = imaCol.rows - maconf.taillechiffre; // pas trop bas
    if (PGG.x > 0) ylim = std::max(0,PGG.y - maconf.taillechiffre);
    ligne ligneHaut; 
    cv::Point2i A(imaCol.cols,0), B(0, 0); // ligne AB reconstituée
    for (auto ligne1 : lignes) {
      if (ligne1.ln[0] < 0) ligne1.ln[0] = - ligne1.ln[0];  // utiliser les petites lignes (invalidées)
      if (std::abs(ligne1.a) > 0.5) continue; // ligne pas assez horizontale (30 degrés)
      if (ligne1.ln[1] > ylim) continue; // ligne trop basse 
      if (ligne1.ln[3] > ylim) continue; // ligne trop basse
      if (ligne1.lg < maconf.largeurcarte / 12) continue; // ligne vraiement trop courte
      if (ligne1.lg < maconf.largeurcarte / 3) { // ligne trop courte
        // vérifier s'il y a d'autres lines alignées
        //  ayant meme normale (ou opposée) dont une extrémité est sur cette ligne
        // calculer les points extrèmes (en x) des lignes cumulées
        // créer une nouvelle ligne supérieure
        int lgtot = 0; // cumuler les petites lignes alignées AB
        A.x = ligne1.ln[0]; A.y = ligne1.ln[1];
        B.x = ligne1.ln[2]; B.y = ligne1.ln[3];
        if (B.x < A.x) {
          A.x = ligne1.ln[2]; A.y = ligne1.ln[3];
          B.x = ligne1.ln[0]; B.y = ligne1.ln[1];
        }
        for (auto ligne2 : lignes) {
          if (ligne2.ln[0] < 0) ligne2.ln[0] = -ligne2.ln[0];
          if (ligne2.lg < maconf.largeurcarte / 12) continue; // ligne trop courte
          if (ligne2.ln[1] > ylim + 2) continue; // ligne trop basse 
          if (ligne2.ln[3] > ylim + 2) continue; // ligne trop basse
          float dist  = ligne1.dist(cv::Point2i(ligne2.ln[0], ligne2.ln[1]));
          if (std::abs(dist) > maconf.deltacadre) continue;
          dist  = ligne1.dist(cv::Point2i(ligne2.ln[2], ligne2.ln[3]));
          if (std::abs(dist) > maconf.deltacadre) continue;
          if (A.x > ligne2.ln[0]) {A.x = ligne2.ln[0]; A.y = ligne2.ln[1];}
          if (A.x > ligne2.ln[2]) {A.x = ligne2.ln[2]; A.y = ligne2.ln[3];}
          if (B.x < ligne2.ln[0]) {B.x = ligne2.ln[0]; B.y = ligne2.ln[1];}
          if (B.x < ligne2.ln[2]) {B.x = ligne2.ln[2]; B.y = ligne2.ln[3];}
          lgtot += ligne2.lg;
        }
        if (lgtot < maconf.largeurcarte / 4 || (B.x - A.x) < maconf.largeurcarte / 3)  continue;
        if (A.y >= ymax) {
          ligne lW;
          lW.ln[0] = A.x; lW.ln[1] = A.y; lW.ln[2] = B.x; lW.ln[3] = B.y;
          float lg = std::sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
          lW.lg = lg;
          lW.a = float(B.y - A.y )/lg;
          lW.b = float(A.x - B.x) / lg;
          lW.c = -A.x * lW.a - A.y*lW.b;
          ymax = A.y;
          ligneHaut = lW;
        }
      } // if ligne courte
      else if (ligne1.ln[1] > ymax) {ymax = ligne1.ln[1]; ligneHaut = ligne1;}
    }

    if (ymax > 0 ){ // on a trouvé la ligne horizontale bord supérieur de la carte
      // calculer l'intersection avec l'arête verticale du coin bas gauche puis droit
      // en fait on calcule la projection du coin bas gauche sur le bord supérieur de la carte
      if (PGG.x > 0 ) { // coin bas gauche trouvé ou calculé
        float dist = ligneHaut.dist(PGG);
        HG.x = PGG.x - dist*ligneHaut.a;
        HG.y = PGG.y - dist*ligneHaut.b;
      }
      if (PDD.x > 0) { // coin bas droit trouvé ou calculé
        float dist = ligneHaut.dist(PDD);
        HD.x = PDD.x - dist*ligneHaut.a;
        HD.y = PDD.y - dist*ligneHaut.b;
      }
      float a, b;
      a = ligneHaut.b; b = -ligneHaut.a; // vecteur directeur de la ligne du bord supérieur de la carte
      if (a < 0) {a = -a; b = -b;} 
      if (HD.x == 0 && HG.x > 0) { // calculer le coin haut droit à partir du gauche
        HD.x = HG.x + a * maconf.largeurcarte;
        HD.y = HG.y + b * maconf.largeurcarte;
      }
      else if (HG.x == 0 && HD.x > 0) { // calculer le coin haut gauche à partir du droit
        HG.x = HD.x - a * maconf.largeurcarte;
        HG.y = HD.y - b * maconf.largeurcarte;
      }
    }

    // le coin haut droit (HD) doit etre à distance largeurcarte du coin haut gauche (HG)
    // au besoin, le recalculer et déplacer le coin bas droit de la même translation
    if (HG.x > 0 && HD.x > 0) {
      float lgHaut = std::sqrt((HD.x - HG.x)*(HD.x - HG.x) + (HD.y - HG.y)*(HD.y - HG.y));
      if (std::abs(lgHaut - maconf.largeurcarte) > maconf.deltacadre) {
        float dx = HD.x; float dy = HD.y;
        HD.x = HG.x + maconf.largeurcarte * (HD.x - HG.x) / lgHaut;
        HD.y = HG.y + maconf.largeurcarte * (HD.y - HG.y) / lgHaut;
        dx = HD.x - dx; dy = HD.y - dy;
        // déplacer le coin bas droit de la même valeur
        PD.x += dx;
        PD.y += dy;
      }
    }
    // rechercher le coin supérieur bien orienté le plus bas, plutot à gauche
    // dont l'arête plutot horizontale est longue (au moins 3/4 de largeurcarte)
    // c'est alors le coin supérieur gauche de la carte
    // sinon, rechercher le coin droit ...
    //                                             _____________
    //                                      _________
    //      _____                     ______
    //     /      ______              \
    //    /             _____          \
    //   /                              \
    //   coins bien orientés
    float a, b, aa, bb; // vecteurs directeurs de l'arête horizontale et verticale
    int ymaxi = 0;
    for(auto moncoin : Coins ){
      P = moncoin.sommet;
      if (P.x > maconf.largeurcarte / 2) continue; // trop à droite
      // vérifier que P est proche de l'arête verticale du coin bas gauche (s'il existe)
      if (PGG.x > 0 && !coinGauche.elimine) {
        float dist;
        if (std::abs(coinGauche.l1->a) > 0.5) dist = coinGauche.l1->dist(P);
        else  dist = coinGauche.l2->dist(P);
        if (std::abs(dist) > maconf.deltacadre) continue; // coin trop loin de l'arête verticale 
      }
      // utiliser le plus haut de H et K  (H.y ou K.y minimum)
      cv::Point2i M, N;
      if (moncoin.H.y < ycol + moncoin.K.y) {M = moncoin.H; N=moncoin.K;}
      else {M = moncoin.K; N=moncoin.H;}

      if (M.x < P.x) continue; // coin droit
      if (N.y < P.y) continue; // coin bas
      // donc coin haut gauche
      int dx = M.x - P.x;
      if (dx < maconf.largeurcarte / 4) continue; // pas sur le bord supérieur (largeur) de carte
      // bon candidat, choisir le plus bas
      if (P.y > ymaxi){ ymaxi = P.y; coinGauche  = moncoin; coinGauche.elimine = false;}
    }
    if (HG.x > 0 ) {
      coinGauche.elimine = true; // choisir la position HG, déjà calculée
      ymaxi = 0;
    }

    if (ymaxi == 0){ // on n'a pas trouvé (ou choisi) le coin gauche sur une arête longue
      // chercher le coin haut droit de la carte
      for(auto moncoin : Coins ){
        P = moncoin.sommet;
        if (P.x < maconf.largeurcarte / 2) continue; // trop à gauche
        // vérifier que P est proche de l'arête verticale du coin bas droit (s'il existe)
        if (PDD.x > 0 && yd > 0) {
          float dist;
          if (std::abs(coinDroit.l1->a) > 0.5) dist = coinDroit.l1->dist(P);
          else dist = coinDroit.l2->dist(P);
          if (std::abs(dist) > maconf.deltacadre) continue; // loin de l'arête verticale
        }
        // utiliser le plus haut de H et K  (H.y ou K.y minimum)
        cv::Point2i M, N; // M le plus haut, N le plus bas
        if (moncoin.H.y < moncoin.K.y) {M = moncoin.H; N=moncoin.K;}
        else {M = moncoin.K; N=moncoin.H;}

        if (M.y < P.y - maconf.deltacadre) continue; // coin bas
        if (M.x > P.x + maconf.deltacadre) continue; // coin (haut) gauche
        // donc coin haut droit
        int dx = P.x - M.x;
        if (dx < maconf.largeurcarte / 4) continue; // pas sur le bord supérieur (largeur) de carte
        // bon candidat, choisir le plus bas
        if (P.y > ymaxi){ ymaxi = P.y; coinDroit  = moncoin; coinDroit.elimine = false;}
      }
      if (HD.x > 0) {
        coinDroit.elimine = true; // choisir HD, calculé à partir des lignes 
        ymaxi = 0;
      }

      if (ymaxi == 0) { // pas trouvé (ou choisi) le coin haut gauche ni droit
        // utiliser les coins calculés à partir du bord supérieur et des coins bas

        if (HG.x > 0 && HD.x > 0) {
          if (printoption > 0){
            cv::circle(mortCopie, cv::Point2i(xcol+HG.x, ycol+HG.y), 4, cv::Scalar(0,255,255),1);
            cv::circle(mortCopie, cv::Point2i(xcol+HD.x, ycol+HD.y), 4, cv::Scalar(0,255,255),1);
            afficherImage("Mort", mortCopie);
          }
          pts[0][0] = xcol + HG.x; // haut gauche
          pts[0][1] = ycol + HG.y;
          pts[1][0] = xcol + HD.x; // haut droit
          pts[1][1] = ycol + HD.y;
          pts[2][0] = PD.x; // bas droit
          pts[2][1] = PD.y;
          pts[3][0] = PG.x; // bas gauche
          pts[3][1] = PG.y;
        } else {
          // impossible de trouver les deux angles supérieurs de la carte
          std::cout<< "!!!! impossible de trouver le bord supérieur de la carte"<<std::endl;
          afficherImage("imacol", imaCol);
          cv::waitKey(0);
        }
        //cv::waitKey(0);
      }
      else { // coin haut droit trouvé. pas le gauche
        // calculer le coin gauche le long de l'arête plutot horizontale
        cv::Point2i Phd(coinDroit.sommet + cv::Point2i(xcol, ycol));
        cv::circle(mortCopie, Phd, 4, cv::Scalar(255,255,0),1);
        // on a le coin haut droit, donc ses deux arêtes
        // calculer les 3 autres sommets à partir des normales (et donc vecteurs directeurs) des arêtes
        pts[1][0] = Phd.x; pts[1][1] = Phd.y;
        // arête vers la gauche, celle dont la normale est plutot verticale
        if (std::abs(coinDroit.l1->b) > std::abs(coinDroit.l1->a) ) { // arête horizontale ?
          a = coinDroit.l1->b; b = -coinDroit.l1->a;
          aa = coinDroit.l2->b; bb = -coinDroit.l2->a;
        } else {
          a = coinDroit.l2->b; b = -coinDroit.l2->a;
          aa = coinDroit.l1->b; bb = -coinDroit.l1->a;
        }
        if (a < 0) { a = -a; b = - b;}
        if (bb < 0 ) {aa = -aa; bb = -bb;}
        // utiliser la normale de l'arête horizontale plutot que le vecteur directeur de la petite arête vericale
        aa = -b; bb = a; 
        // premier point en bas à gauche suivant en bas à droite
        //pts[0] est en haut à gauche!
        // créer les autres points en laissant pts[0] en haut à gauche
        pts[0][0] = pts[1][0] - a*maconf.largeurcarte; // coin haut gauche
        pts[0][1] = pts[1][1] - b*maconf.largeurcarte;

        pts[2][0] = pts[1][0] + aa*maconf.hauteurcarte;
        pts[2][1] = pts[1][1] + bb*maconf.hauteurcarte;
        pts[3][0] = pts[0][0] + aa*maconf.hauteurcarte;
        pts[3][1] = pts[0][1] + bb*maconf.hauteurcarte;
        // les coins bas ont été calculés : PG et PD
        if (PDD.x > 0) pts[2][0] = PD.x; pts[2][1] = PD.y;
        if (PGG.x > 0) pts[3][0] = PG.x; pts[3][1] = PG.y;
      }
    }
    else { // on a trouvé le coin supérieur gauche
      cv::Point2i Phg(coinGauche.sommet + cv::Point2i(xcol, ycol));
      cv::circle(mortCopie, Phg, 4, cv::Scalar(255,255,0),1);
    
      // on a le coin haut gauche, donc ses deux arêtes
      // calculer les 3 autres sommets à partir des normales (et donc vecteurs directeurs) des arêtes
      pts[0][0] = xcol + coinGauche.sommet.x; pts[0][1] = ycol + coinGauche.sommet.y;
      pts[0][0] = Phg.x; pts[0][1] = Phg.y;
      // arête vers la droite, celle dont la normale est plutot verticale
      if (std::abs(coinGauche.l1->b) > std::abs(coinGauche.l1->a) ) { // arête horizontale ?
        a = coinGauche.l1->b; b = -coinGauche.l1->a;
        aa = coinGauche.l2->b; bb = -coinGauche.l2->a;
      } else {
        a = coinGauche.l2->b; b = -coinGauche.l2->a;
        aa = coinGauche.l1->b; bb = -coinGauche.l1->a;
      }
      if (a < 0) { a = -a; b = - b;}
      if (bb < 0 ) {aa = -aa; bb = -bb;}
      // utiliser la normale de l'arête horizontale plutot que le vecteur directeur de la petite arête vericale
      aa = -b; bb = a; 
      // premier point en bas à gauche suivant en bas à droite
      //pts[0] est en haut à gauche!
      // créer les autres points en laissant pts[0] en haut à gauche
      pts[1][0] = pts[0][0] + a*maconf.largeurcarte;
      pts[1][1] = pts[0][1] + b*maconf.largeurcarte;

      pts[2][0] = pts[1][0] + aa*maconf.hauteurcarte;
      pts[2][1] = pts[1][1] + bb*maconf.hauteurcarte;
      pts[3][0] = pts[0][0] + aa*maconf.hauteurcarte;
      pts[3][1] = pts[0][1] + bb*maconf.hauteurcarte;
      // les coins bas ont été calculés : PG et PD
      if (PDD.x > 0) pts[2][0] = PD.x; pts[2][1] = PD.y;
      if (PGG.x > 0) pts[3][0] = PG.x; pts[3][1] = PG.y;
  }
    if (printoption > 0) afficherImage("Mort", mortCopie);

    // on a les 4 sommets de la carte, dans l'image imaMort
    // seul le haut de la carte est présent
    // décoder la carte :
    int numcolW(numcol); // la couleur a été déterminée par la carte du bas de la colonne
    cv::Mat imacarte = extraireCarteIncomplete(imaMort, pts, maconf);

    valcarte = decoderLaCarte(imacarte, maconf, numcolW);
    {
    std::string s = carteToString(numcol, valcarte);
    //if (printoption > 0) 
      std::cout<<"==> carte du mort :"<< s<<std::endl;
    }
    // s'assurer que pts[2] et pts[3] sont les coins hauts
    int xx = pts[3][0]; int yy = pts[3][1];
    pts[3][0] = pts[0][0]; pts[3][1] = pts[0][1]; // coin haut gauche de la carte
    pts[0][0] = xx; pts[0][1] = yy;
    xx = pts[1][0]; yy = pts[1][1];
    pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1];
    pts[2][0] = xx; pts[2][1] = yy;
  }
  else { // on traite la première carte en bas, donc complète
    // chercher la ligne la plus longue = la ligne du bas = bord inférieur de carte
    // ligne avec une des extrémités la plus basse et plutot horizontale
    int lgmax = 0;
    int ymax = 0;
    for(auto ligne:lignes){
      cv::Point2i A(ligne.ln[0], ligne.ln[1]);
      cv::Point2i B(ligne.ln[2], ligne.ln[3]);
      int lg = ligne.lg;
      if (std::max(A.y, B.y) > ymax) {
        if (std::abs(ligne.a) < 0.5) { // ligne plutot horizontale (30 degrés)
          if (lg > maconf.largeurcarte / 2) {ymax = std::max(A.y, B.y);ligneBas = ligne;}
        }
      }
    }
    cv::Point2i A(ligneBas.ln[0], ligneBas.ln[1]);
    cv::Point2i B(ligneBas.ln[2], ligneBas.ln[3]);
    cv::Point2i AA = A + cv::Point2i(xcol, ycol);
    cv::Point2i BB = B + cv::Point2i(xcol, ycol);
    if (printoption > 1){
      cv::line(mortCopie,AA,BB,cv::Scalar(255,128,0),2);
      afficherImage("Mort", mortCopie); //cv::waitKey(0);
    }

    pts[0][0] = PG.x; pts[0][1] = PG.y;
    pts[1][0] = PD.x; pts[1][1] = PD.y;
    float aa = ligneBas.a; float bb = ligneBas.b; 
    if (bb > 0) {aa = -aa; bb = -bb;}
    pts[2][0] = pts[1][0] + aa*maconf.hauteurcarte;
    pts[2][1] = pts[1][1] + bb*maconf.hauteurcarte;
    pts[3][0] = pts[0][0] + aa*maconf.hauteurcarte;
    pts[3][1] = pts[0][1] + bb*maconf.hauteurcarte;

    // décoder la carte :
    numcol = -1; valcarte = 0;
    valcarte = decoderCarte(imaMort, pts, maconf, numcol);
    {
      std::string s = carteToString(numcol, valcarte);
      //if (printoption > 0)
       std::cout<<std::endl<<"==> carte du mort:"<< s<<std::endl;
    //cv::waitKey(0);
    }
  }
  carteMort[icarteMort].couleur = numcol;
  carteMort[icarteMort].valeur = valcarte;
  icarteMort++;
  //
  {
    //
    // remplir la zone de la carte (élargie) avec la couleur du fond
    // Définir les 4 points du rectangle incliné
    // 0: bas gauche   1: bas droit  2: haut droit    3: haut gauche
    // si la carte est en haut de imaMort, nettoyer à partir du haut de imaMort
    if (pts[3][1] < maconf.taillegros) pts[3][1] = pts[2][1] = 2;
    // elargir suffisamment à droite, sans déborder sur la prochaine colonne de cartes
    // pour cela, décaler le coin haut droit à droite jusqu'à la couleur de fond 
    // sur quelques pixels (10% de la largeur de carte)
    int dx(0); //décalage à droite
    int x = pts[2][0]; if(estPremier) x-= 2;
    cv::Scalar px;
    while(x < pts[2][0] + maconf.largeurcarte/8) {
      px = imaMort.at<cv::Vec3b>(pts[2][1], x);
      if (std::abs(px[0] - couleurFond[0]) + std::abs(px[0] - couleurFond[0])
       + std::abs(px[0] - couleurFond[0]) < 60) break;
       x++;
    }
    /************************************ 
    while(x < pts[2][0] + maconf.largeurcarte/8) {
      px = imaMort.at<cv::Vec3b>(pts[2][1], x);
      if (std::abs(px[0] - couleurFond[0]) + std::abs(px[0] - couleurFond[0])
       + std::abs(px[0] - couleurFond[0]) > 30) {
        break;
       }
       x++;
    }
    *************************************/
    dx = x - pts[2][0];
    
    cv::Point points[1][4];
    cv::Point2i ptsW[4];
    ptsW[0].x = std::max(0, pts[0][0] - 2);
    ptsW[0].y = std::min(imaMort.rows -1, pts[0][1] +1);
    ptsW[1].x = std::min(imaMort.cols -1 , pts[1][0] + dx);
    ptsW[1].y = std::min(imaMort.rows -1, pts[1][1] +1);
    ptsW[2].x = std::min(imaMort.cols -1 , pts[2][0] + dx);
    ptsW[2].y = std::max(0, pts[2][1] -2);
    ptsW[3].x = std::max(0, pts[3][0] -2);
    ptsW[3].y = std::max(0, pts[3][1] -2);


    points[0][0] = ptsW[0];
    points[0][1] = ptsW[1];
    points[0][2] = ptsW[2];
    points[0][3] = ptsW[3];

    // Convertir en structure compatible
    const cv::Point* ppt[1] = { points[0] };
    int npt[] = { 4 };
    // Remplir le polygone
    cv::fillPoly(imaMort, ppt, npt, 1, couleurFond);
    // ajouter une ligne noire sur le bord supérieur de la carte qu'on vient de décoder
    cv::line(imaMort, cv::Point2i(pts[3][0], std::max(0,pts[3][1]-1)),
      cv::Point2i(pts[2][0], std::max(0,pts[2][1] - 1)), cv::Scalar(0,0,0), 1);
  }
  // si la carte est en haut de la colonne, nettoyer l'image en incluant cette colonne
  if (pts[3][1] < 6 + maconf.taillechiffre) {
    cv::Rect rr;
    rr.x = 0; rr.width = pts[2][0] + 2; // imprécision sur le bord droit : +2
     rr.y = 0; rr.height = imaMort.rows;
    cv::rectangle(imaMort, rr, couleurFond, cv::FILLED);
    if (pts[2][0] > imaMort.cols - maconf.largeurcarte) break; // dernière colonne
    if (maconf.waitoption) {afficherImage("mort", imaMort); cv::waitKey(0);}
  }

  mortCopie = imaMort.clone();
  if  (printoption > 0 ) afficherImage("Mort", mortCopie); //cv::waitKey(0);

  // traiter les autres cartes, limitées à la partie supérieure, de la colonne
  // rappel : haut gauche de la colonne : xcol, ycol
  // le bas de la colonne est le plus bas de pts[2][1]  et pts[3][1]

  estPremier = false;
} // while(true)
auto t2 = std::chrono::high_resolution_clock::now();
duree = t2 - t1;
std::cout << "Duree de décodage du Mort : " << duree.count() << " secondes" << std::endl;

}
