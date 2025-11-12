#include "config.h"
/////////////////////////////////////////////////
// définitions des méthodes 
///////////////////////////////////////////////

bool uncoin::estSurBord(std::vector<unbord> bords, const config& maconf){
  int epsilon = std::max(4,2*maconf.deltacadre);
  bool coinSurBord(false);
  for (unbord& ub: bords){
    if (ub.contient(sommet, epsilon)) {
        coinSurBord = true;
        elimine = true;
        break;
      }
  }
  return coinSurBord;
}

bool unbord::contient(cv::Point2i P, int ecart){
  ligne ln (A,B);
  float dist = ln.dist(P);
  if (std::abs(dist) >= ecart) return false;
  float ps = (A-P).x * (B-P).x + (A-P).y * (B-P).y;
  if (ps >= 0) return false;
  if (std::abs((A-P).x) < ecart && std::abs((A-P).y) <= ecart) return false; 
  if (std::abs((B-P).x) < ecart && std::abs((B-P).y) <= ecart) return false; 
  return true;
}

