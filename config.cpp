#include "config.h"
/////////////////////////////////////////////////
// définitions des méthodes 
///////////////////////////////////////////////


std::vector<unbord> calculerBords(std::vector<unecarte>& cartes, const config& maconf) {
  int epsilon = std::max(4,2*maconf.deltacadre);
  std::vector<unbord> bords;
  for (auto& ucp : cartes){
    auto coins= ucp.coins;
    for (int h=0; h<coins.size(); h++){
      auto uc = coins[h];
      if (uc->elimine) continue;
      cv::Point2i P; P.x = uc->sommet.x; P.y = uc->sommet.y;
      //int nc = up.numcarte;
      for (int hh = h+1; hh<coins.size(); hh++){
        auto up2 = coins[hh];
        if (up2->elimine) continue;
        //if (up2.numcarte != nc) continue;
        cv::Point2i Q; Q.x = up2->sommet.x; Q.y = up2->sommet.y;
        int d = std::sqrt( (Q-P).x * (Q-P).x + (Q-P).y * (Q-P).y);
        if (std::abs(d - maconf.hauteurcarte) > epsilon 
        && std::abs(d - maconf.largeurcarte) > epsilon ) continue;
        unbord ub(0);
        //ub.numcarte = nc;
        ub.A = P; ub.B = Q;
        bords.push_back(ub);
      }
    }
  }
  return bords;
}

std::vector<unbord> calculerBords(std::vector<unecartePrec>& cartesPrec, const config& maconf) {
  int epsilon = std::max(4,2*maconf.deltacadre);
  std::vector<unbord> bords;
  for (auto& ucp : cartesPrec){
    auto coinsPrec = ucp.coinsPrec;
    for (int h=0; h<coinsPrec.size(); h++){
      auto uc = coinsPrec[h];
      cv::Point2i P; P.x = uc.x; P.y = uc.y;
      //int nc = up.numcarte;
      for (int hh = h+1; hh<coinsPrec.size(); hh++){
        auto up2 = coinsPrec[hh];
        //if (up2.numcarte != nc) continue;
        cv::Point2i Q; Q.x = up2.x; Q.y = up2.y;
        int d = std::sqrt( (Q-P).x * (Q-P).x + (Q-P).y * (Q-P).y);
        if (std::abs(d - maconf.hauteurcarte) > epsilon 
        && std::abs(d - maconf.largeurcarte) > epsilon ) continue;
        unbord ub(0);
        //ub.numcarte = nc;
        ub.A = P; ub.B = Q;
        bords.push_back(ub);
      }
    }
  }
  return bords;
}

std::vector<unbord> calculerBords(std::vector<uncoinPrec>& coinsPrec, const config& maconf) {
  int epsilon = std::max(4,2*maconf.deltacadre);
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
  return bordsPrec;
}

std::vector<unbord> calculerBords(std::vector<uncoin>& coins, const config& maconf){

  int epsilon = std::max(4,2*maconf.deltacadre);
  std::vector<unbord> bords;
  int nc = 1;
  while(nc >=0) {
    bool existecarte(false);
    for (int i = 0; i < coins.size(); i++){
      uncoin& c1 = coins[i]; 
      if (c1.elimine) continue;
      if (c1.numCarte < nc) continue;
      existecarte = true;
      cv::Point2i P=c1.sommet;
      for (int j = i+1; j < coins.size(); j++) {
        uncoin& c2 = coins[j];
        if (c2.elimine) continue;
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
  return bords;
}



bool uncoin::estSurBord(std::vector<unbord> bords, const config& maconf){
  int epsilon = std::max(4,2*maconf.deltacadre);
  bool coinSurBord(false);
  for (unbord& ub: bords){
    if (ub.contient(this->sommet, epsilon)) {
        coinSurBord = true;
        break;
      }
  }
  return coinSurBord;
}

bool uncoin::estproche(const uncoin& coin, config& maconf, int ecart) const{
  // les arêtes doivent être parallèles
  // les arètes parallèles doivent bavoir le même sens
  // les sommets doivent être proches
  if (ecart <= 0) ecart = maconf.deltacadre;
  ligne l1a = *this->l1;
  ligne l1b = *coin.l1;
  ligne l2a = *this->l2;
  ligne l2b = *coin.l2;
  if (std::abs( (this->sommet - coin.sommet).x) > maconf.deltacadre) return false;
  if (std::abs( (this->sommet - coin.sommet).y) > maconf.deltacadre) return false;

  if (l1a.a * l2b.a + l1a.b * l2b.b < maconf.cosOrtho) {
    // l1a // l1b vérifier l'orientation
    if (( this->sommet - this->H) * (coin.sommet - coin.H) < 0 ) return false; // sens opposé
    if (( this->sommet - this->K) * (coin.sommet - coin.K) < 0 ) return false; // sens opposé
  } else if (l1a.a * l1b.a + l1a.b * l1b.b < maconf.cosOrtho) {
    // l1a // l2b vérifier l'orientation
    if (( this->sommet - this->H) * (coin.sommet - coin.K) < 0 ) return false; // sens opposé
    if (( this->sommet - this->K) * (coin.sommet - coin.H) < 0 ) return false; // sens opposé
  }
  return true;
}

bool uncoin::estoppose(const uncoin& coin, config& maconf, int ecart) const{
  // les arêtes doivent être parallèles et proches
  // les arètes parallèles doivent avoir le même sens
  // la distance entre les sommets doit être proche de la hauteur ou largeur de carte

  if (ecart <= 0) ecart = std::max(4, maconf.deltacadre * 2);
  ligne l1a = *this->l1;
  ligne l1b = *coin.l1;
  ligne l2a = *this->l2;
  ligne l2b = *coin.l2;

  float d1= l1a.dist(coin.sommet);
  float d2= l2a.dist(coin.sommet);
  if (std::min(std::abs(d1), std::abs(d2)) > maconf.deltacadre) return false;
  float d11= l1b.dist(this->sommet);
  float d22= l2b.dist(this->sommet);
  if (std::min(std::abs(d11), std::abs(d22)) > maconf.deltacadre) return false;

  // vérifier le sens des arêtes communes
  if (std::abs(d1) < std::abs(d2)) {
    if (std::abs(d11) < std::abs(d22)) {
      // l1a et l1b parallèles proches
      if (( this->sommet - this->H) * (coin.sommet - coin.H) > 0 ) return false; // même sens
      if (( this->sommet - this->K) * (coin.sommet - coin.K) < 0 ) return false; // sens opposé
    } else {
      // l1a // l2b et proches
      if (( this->sommet - this->H) * (coin.sommet - coin.K) > 0 ) return false; // même sens
      if (( this->sommet - this->K) * (coin.sommet - coin.H) < 0 ) return false; // sens opposé
    }
  } else { // ligne commune l2a
    if (std::abs(d11) < std::abs(d22)) {
      // l2a et l1b parallèles proches
      if (( this->sommet - this->K) * (coin.sommet - coin.H) > 0 ) return false; // même sens
      if (( this->sommet - this->H) * (coin.sommet - coin.K) < 0 ) return false; // sens opposé
    } else {
      // l2a // l2b et proches
      if (( this->sommet - this->K) * (coin.sommet - coin.K) > 0 ) return false; // même sens
      if (( this->sommet - this->H) * (coin.sommet - coin.H) < 0 ) return false; // sens opposé
    }
  }
  cv::Point2i pq = (this->sommet - coin.sommet);
  int dist = std::sqrt(pq.x * pq.x + pq.y * pq.y);
  if ( std::abs(dist - maconf.hauteurcarte) > ecart &&
       std::abs(dist - maconf.largeurcarte) > ecart) 
        return false;
  return true;
}

bool uncoin::estDans(const uncoin& coin, config& maconf, int ecart) const{
  if (ecart <= 0) ecart = maconf.deltacadre;
  if (! this->estproche(coin, maconf, ecart)) return false;
  float d1= coin.l1->dist(this->sommet);
  float d2= coin.l2->dist(this->sommet);
  if (d1*coin.l1->dist(coin.K) <= 0) return false; // dehors
  if (d1*coin.l2->dist(coin.H) <= 0) return false; // dehors
  return true;
}

bool unbord::contient(cv::Point2i P, int ecart){
  // P proche des extrémités du segment ?
  if (std::abs((A-P).x) < ecart && std::abs((A-P).y) <= ecart) return false; 
  if (std::abs((B-P).x) < ecart && std::abs((B-P).y) <= ecart) return false; 
  ligne ln (A,B);
  float dist = ln.dist(P);
  if (std::abs(dist) >= ecart) return false; // loin de la droite
  // vérifier que P est entre A et B
  float ps = (A-P).x * (B-P).x + (A-P).y * (B-P).y;
  if (ps >= 0) return false;
  return true;
}

bool unecartePrec::contient(uncoin cn, int ecart) {
  // la carte précédente contient un coin si le sommet du coin est proche d'un sommet de la carte
  if (cn.elimine) return false;
  cv::Point2i P = cn.sommet;
  return this->contient(P, ecart);
}

bool unecartePrec::contient(cv::Point2i P, int ecart) {
  // la carte précédente contient le point P s'il est proche d'un sommet de la carte
  for (auto up : this->coinsPrec)
  if (std::abs(P.x - up.x) < ecart && std::abs(P.y - up.y) < ecart) return true;
  for ( int i=0; i<4; i++){
    cv::Point2i Q = this->sommet[i];
    if (std::abs(P.x - Q.x) < ecart && std::abs(P.y - Q.y) < ecart) return true;
  }
  return false;
}

bool unecartePrec::contient(unecarte carte, int ecart) {
  // la carte précédente contient la carte si elle a au moins deux sommets proches
  if (carte.coins.size() < 2 ) return false;
  int nb = 0;
  if (carte.sommet[0].x != 0 && carte.sommet[0].y != 0){
    for (int i=0; i<4; i++){
      cv::Point2i Q = carte.sommet[i];
      bool trouve(false);
      if (this->contient(Q, ecart)) {
        if (nb >= 1) return true;
        nb++;
      }
    }
  }
  else {
    for (auto coin : carte.coins){
      cv::Point2i Q = coin->sommet;
      if (this->contient(Q, ecart)) {
        if (nb >= 1) return true;
        nb++;
      }
    }
  }
  return false;
}

bool unecarte::estDansPli(unpli monpli, int epsilon){
  if (this->coins.size() <= 0) return false;
  bool carteDansPli(false);
  unecarte& carte = monpli.cartes[0];
  auto cn = coins[0]; // premier coin de la carte 
  cv::Point2i P(cn->sommet);
  int isom1; // indice du sommet de la carte du pli qui est proche du premier coin
  bool trouveCarteDuPli(false);
  for (int ic = 0; ic < monpli.nbcartes; ic++) {
    auto& cartex = monpli.cartes[ic];
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
  if (!trouveCarteDuPli) return false;

  // vérifier que les autres coins sont proches d'un (autre) sommet de la même carte du pli
  carteDansPli=true; // au moins un coin dans le pli
  for (int icn = 1; icn < coins.size(); icn++) {
    auto& cm = coins[icn];
    cv::Point2i R(cm->sommet);
    bool coinDansPli(true);
    bool estproche(false);
    for (int k = 0; k< 4; k++){
      if (k == isom1) continue; // ce sommet est déjà proche de la carte du pli
      // proche d'un sommet de la carte du pli 
      int dx = std::abs(R.x - carte.sommet[k].x);
      int dy = std::abs(R.y - carte.sommet[k].y);
      if (dx < epsilon &&  dy < epsilon  ){
        estproche=true;
        break;
      }
    } // sommets de la carte
    if (!estproche) {
      carteDansPli = false;
      break;
    }
    if (!coinDansPli){ carteDansPli = false; break;}
  } // for cm::Coins
  if(carteDansPli){
    this->couleur = carte.couleur;
    this->valeur = carte.valeur;
    // actualiser l'encombrement de la carte dans le pli
    for (auto coin: this->coins) {
      if (coin->sommet.x < carte.ymin) carte.ymin = coin->sommet.x;
      if (coin->sommet.x > carte.xmax) carte.xmax = coin->sommet.x;
      if (coin->sommet.y < carte.ymin) carte.ymin = coin->sommet.y;
      if (coin->sommet.y > carte.ymax) carte.ymax = coin->sommet.y;
    }
    // récupérer les sommets de la carte du pli
    for (int i=0; i<4; i++){
      this->sommet[i] = carte.sommet[i];
    }
  }
  return carteDansPli;
}

void unecarte::calculSommets(config& maconf){
  // calculer les sommets de la carte à partir des coins
  if (this->coins.size() < 2) return; // pas assez de coins pour calculer les sommets
  if (this->coins.size() == 4){
    for (int i=0; i<4; i++){
      this->sommet[i] = this->coins[i]->sommet;
    }
    return;
  }
  // au moins 2 coins
  // trouver les coins opposés
  int ecart = std::max(maconf.hauteurcarte / 12, maconf.deltacadre * 2);
  uncoin* coin1 = this->coins[0];
  uncoin* coin2 = nullptr;
  for (int i=1; i<this->coins.size(); i++){
    if (coin1->estoppose(*this->coins[i], maconf, ecart)){
      coin2 = this->coins[i];
      break;
    }
  }
  if (coin2 == nullptr) return; // pas de coins opposés
  // coins opposés trouvés : coin1 et coin2
  // calculer les autres sommets à distance hauteur ou largeur de carte
  cv::Point2i P1 = coin1->sommet;
  cv::Point2i P2 = coin2->sommet;
  cv::Point2i dir = P2 - P1;
  float longueur = std::sqrt(dir.x * dir.x + dir.y * dir.y);
  int lg = maconf.largeurcarte;
  if (longueur < (maconf.hauteurcarte + maconf.largeurcarte) / 2)
    lg = maconf.hauteurcarte;
  // on a le vecteur normal à chaque arête de chaque coin
  ligne l1a = *coin1->l1;
  ligne l2a = *coin1->l2;
  // choisir la ligne la plus proche de P2
  float d1 = l1a.dist(P2);
  float d2 = l2a.dist(P2);
  cv::Point2i normale;
  if (std::abs(d1) < std::abs(d2)) {
    normale.x = l1a.a;
    normale.y = l1a.b;
  } else {
    normale.x = l2a.a;
    normale.y = l2a.b;
  }
  // déjà normé

  cv::Point2i offset;
  offset.x = int(normale.x * lg);
  offset.y = int(normale.y * lg);
  cv::Point2i P3 = P1 - offset;
  cv::Point2i P4 = P2 - offset; 
  // ordre des sommets : P1, P2, P4, P3
  this->sommet[0] = P1;
  this->sommet[1] = P2;
  this->sommet[2] = P4;
  this->sommet[3] = P3;
}
  