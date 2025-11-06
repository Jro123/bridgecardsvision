#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <stack>
#include <cmath>
#include "config.h"

extern int threadoption;
extern int printoption;

const char* NESO[4]=  {"Nord", "Est", "Sud", "Ouest"};
const char* couleurcarteA[4]  = {"P", "C", "K", "T"}; 

// Convertit numéro joueur en texte
std::string joueurToString(int j) {
  //const char* noms[] = {"Nord", "Est", "Sud", "Ouest"};
  return (j >= 0 && j < 4) ? NESO[j] : "Inconnu";
}


// Convertit couleur/valeur en chaîne lisible
std::string carteToString(int couleur, int valeur) {
  if (couleur < 0 || couleur > 3 || valeur < 1 || valeur > 13) return "??";
  std::string val;
  if (valeur == 1)  val = "A";
  else if (valeur <= 10) val = std::to_string(valeur);
  else if (valeur == 11) val = "V";
  else if (valeur == 12) val = "D";
  else if (valeur == 13) val = "R";
  
  return std::string(couleurcarteA[couleur]) + val;
}


bool calculerRouge(cv::Mat GS, bool estunRDV, const config& maconf, cv::Scalar mbl);

// tracer le rectangle r sur une copie de l'image et afficher la fenêtre dont le nom est s 
void tracerRectangle(cv::Rect r, cv::Mat copie, std::string s, cv::Scalar couleur) {
    if (printoption < 0) return;
    if (r.width >= 1) cv::line(copie, cv::Point2i(r.x, r.y), cv::Point2i(r.x + r.width-1, r.y), couleur);
    if (r.height > 1){
        if (r.width > 1) cv::line(copie, cv::Point2i(r.x + r.width - 1, r.y + r.height-1), cv::Point2i(r.x, r.y + r.height-1), couleur);
        cv::line(copie, cv::Point2i(r.x, r.y), cv::Point2i(r.x, r.y + r.height-1), couleur);
        if (r.width> 1) cv::line(copie, cv::Point2i(r.x + r.width - 1, r.y + r.height-1), cv::Point2i(r.x + r.width - 1, r.y), couleur);
    }
    afficherImage(s, copie); //cv::waitKey(1);
}

void afficherImage(std::string nom, cv::Mat image) {
    if (threadoption) return;
    if (printoption < 0) return;
#ifndef _WIN32
    //cv::namedWindow(nom, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(nom, cv::WINDOW_NORMAL);

    //cv::resizeWindow(nom, image.cols, image.rows);
    cv::imshow(nom, image);
    //cv::waitKey(1);
    //std::this_thread::sleep_for(std::chrono::seconds(5)); // ou moins
#else
    cv::imshow(nom, image);
#endif
}


double calculerDistance(cv::Vec4i& l1, cv::Vec4i& l2) {
    // les droites AB et CD sont suppos�es paralleles
    // n �tant la normale � AB
    // calculer AC.n
    // plus directement : AC ^ AB / longueur AB
    double d = -(l2[0] - l1[0]) * (l1[3] - l1[1]) + (l2[1] - l1[1]) * (l1[2] - l1[0]);
    double lgab = (l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]);
    d /= sqrt(lgab);
    return d;
}
double calculerDistance(cv::Point2i& Q, cv::Point2i P, cv::Point2i R) {
    // calcul de la distance alg�brique du point Q � la droite PR
    // n �tant la normale � PR
    // calculer PQ.n
    // plus directement avec le produit vectoriel  : PQ ^ PR / longueur PR
    double d = (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x);
    double lgpr = (R.x - P.x) * (R.x - P.x) + (R.y - P.y) * (R.y - P.y);
    d /= sqrt(lgpr);
    return d;
}

double calculerSinus(cv::Vec4i& l1, cv::Vec4i& l2) {
    // calculer le produit vectoriel AB ^ CD
    double pv;
    pv = -(l1[2] - l1[0]) * (l2[3] - l2[1]) + (l1[3] - l1[1]) * (l2[2] - l2[0]);
    double lgab = (l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]);
    double lgcd = (l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]);
    pv /= sqrt(lgab * lgcd);
    return pv;
}

void calculerBlanc(uncoin& moncoin, const config& maconf) {
    cv::Mat lig;
    cv::Rect r;
    int db = 1; // d�calage de la zone de test du blanc par rapport au bord
    if (maconf.deltacadre > 5) db = 2;
    r.width = 2 * maconf.taillechiffre;
    r.height = std::max(1, maconf.deltacadre - 2 * db);

    if (moncoin.UU.x > moncoin.PP.x) { // � droite
        r.x = moncoin.PP.x + maconf.taillechiffre;
        r.x = std::min(moncoin.ima_coin.cols - r.width, r.x);
    }
    else {
        r.x = std::max(0, moncoin.PP.x - 3 * maconf.taillechiffre);
    }
    if (moncoin.U.y > moncoin.PP.y) { // au dessous
        r.y = moncoin.PP.y + db;
        r.y = std::min(moncoin.ima_coin.rows - r.height, r.y);
    }
    else {
        r.y = std::max(0,moncoin.PP.y - db - r.height);
    }
    cv::Mat zone = moncoin.ima_coin(r);
    cv::Scalar moyblanc = cv::mean(zone);
    moncoin.moyblanc = moyblanc;
}

// calculer l'intensité moyenne des pixels (foncés) du symbole
void calculerMoyenneSymbole(const cv::Mat& image, const cv::Scalar ref, cv::Scalar& moy) {
    double sommeRouge = 0, sommeVert = 0, sommeBleu = 0;
   int compteurPixels = 0;
   // déterminer l'intensité maximale -->référence du blanc
   // puis ne considérer que les pixels plus foncés
   int maxbgr = 0;
   int maxb = 0;
   int maxg = 0;
   int maxr = 0;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            int bleu = pixel[0];
            int vert = pixel[1];
            int rouge = pixel[2];
            if (bleu+ vert + rouge > maxbgr) {
                maxbgr = bleu + vert + rouge;
                maxb = bleu;
                maxg = vert;
                maxr = rouge;
            }
        }
    }

   // Parcourir tous les pixels
   int ecart = 90;
    while(compteurPixels < image.rows*image.cols / 4 && ecart > 1) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                int bleu = pixel[0];
                int vert = pixel[1];
                int rouge = pixel[2];
                    // Vérifier la condition
                if ( maxbgr - (bleu + vert + rouge )> ecart )
                {
                    sommeRouge += rouge;
                    sommeVert += vert;
                    sommeBleu += bleu;
                    compteurPixels++;
                }
            }
        }
        ecart /=2;
    }

   if (compteurPixels > 0) {
       moy[0] = sommeBleu / compteurPixels;
       moy[1] = sommeVert / compteurPixels;
       moy[2] = sommeRouge / compteurPixels;
       std::cout << "Moyennes BGR : " << moy << std::endl;
   } else {
       std::cout << "!!! max b g r "<< maxb<<" "<<maxg<<" "<<maxr<<std::endl;
       moy = ref;
   }
}



void calculerMinimum(const cv::Mat& image, cv::Scalar& minimum) {
    int minb = 255;
    int ming = 255;
    int minr = 255;
    int minbgr = 255;

    // Parcourir tous les pixels
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            int bleu = pixel[0];
            int vert = pixel[1];
            int rouge = pixel[2];
            minb = std::min(minb, bleu);
            ming = std::min(ming, vert);
            minr = std::min(minr, rouge);
            minbgr = std::min (minbgr, (bleu + vert + rouge) / 3);
        }
    }
    minimum[0] = minb;
    minimum[1] = ming;
    minimum[2] = minr;
    minimum[3] = minbgr;
}


// calculerl'encombrement d'un symbole de hauteur ts et de largeur ls 
// et fournir la moyenne des intensités du symbole et du fonds
// on ignore le rouge, à cause des symboles rouges
void calculerBox(const cv::Mat& imageW, const int ts, const int ls, cv::Scalar& moy,
    int *pBox, cv::Scalar& moyext, const config& maconf){
    int printoption = maconf.printoption;
    cv::Rect r;
    cv::Mat image = imageW.clone();
    r.width = std::min(ls, image.cols);
    r.height = std::min(ts, image.rows);
    cv::Scalar m;
    cv::Scalar mopt; 
    double intmin = 255*2;
    int xopt=0, yopt=0;
    for ( r.x = 0; r.x <= image.cols - r.width; r.x++){
        for (r.y=0; r.y <= image.rows - r.height; r.y++){
            cv::Mat box = image(r);
            m = cv::mean(box);
            double mbg = m[0] + m[1];
            if (intmin >= mbg ) {
                intmin = mbg;
                xopt = r.x;
                yopt = r.y;
                mopt = m;
            }
        }
    }
    moy = mopt;
    cv::Scalar mtot = mean(image);
    cv::Scalar mext = mtot;
    if(ls < image.cols || ts < image.rows)
        mext = (mtot*image.rows*image.cols - moy*ts*ls) / (image.rows*image.cols - ts*ls);
    else mext = mtot;
    moyext = mext;
    if(pBox){
        pBox[0] = xopt;
        pBox[1] = xopt + r.width - 1;
        pBox[2] = yopt;
        pBox[3] = yopt + r.height - 1;
        if (printoption > 2) std::cout<<"BOX "<<pBox[0]<<","<<pBox[1]<<","<<pBox[2]<<","<<pBox[3]<<
        " / "<< image.cols<<"x"<<image.rows<<" Moyennes " << moy << moyext <<  std::endl;
    }
}




// déterminer l'encombrement d'un caractère ou symbole dans une image
// on ne considère que les pixels les plus foncés, au moins un quart
// on synthétise en calculant les positions xmin, xmax, ymin, ymax
// ensuite on élargit par les bordures claires et on fournit l'image recalculée
// ref : pas utilisé, moy:moyennes du symbole, moyext : moyennes hors du symbole
void calculerEncombrement(const cv::Mat& image, const cv::Scalar ref, cv::Scalar& moy,
    int *pBox, cv::Mat& imaR, cv::Scalar& moyext) {
    double sommeRouge = 0, sommeVert = 0, sommeBleu = 0;
   int compteurPixels = 0;
   double sRx=0, sVx=0, sBx=0; int ctrx = 0;
   // mise au point : déterminer l'écart maximum avec la référence
   int maxb = 0;
   int maxg = 0;
   int maxr = 0;

   // déterminer l'intensité maximale -->référence du blanc
   // puis ne considérer que les pixels plus foncés
   int maxbgr = 0;
   int maxbg = 0; 
   int maxext = 0;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            int bleu = pixel[0];
            int vert = pixel[1];
            int rouge = pixel[2];
            if (bleu+ vert + rouge > maxbgr) { maxbgr = bleu + vert + rouge;}
            if (bleu+ vert > maxbg) {
                maxbg = bleu+ vert;
                maxb = bleu;
                maxg = vert;
                maxr = rouge;
            }
        }
    }

    // Parcourir tous les pixels, en ne considérant que les plus foncés
    // au plus une moitié de l'image
    int ecart = 10;  // 10 : valeur expérimentale à confirmer
    int xmin(image.cols - 1), xmax(0), ymin(image.rows - 1), ymax(0);
    compteurPixels = (image.rows)*(image.cols);
    while(compteurPixels > (image.rows)*(image.cols) / 2 && ecart < 200){
        xmin = image.cols - 1;
        xmax = 0; 
        ymin = image.rows - 1;
        ymax = 0;
        sommeRouge = 0; sommeVert = 0; sommeBleu = 0; compteurPixels = 0;
        sRx = 0; sVx = 0; sBx = 0; ctrx = 0;
        maxext = 0; // intensité maximale hors du symbole
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                int bleu = pixel[0];
                int vert = pixel[1];
                int rouge = pixel[2];
                // Vérifier la condition
                int bv = bleu + vert;
                if ( maxbg - bv > ecart)  {
                    sommeRouge += rouge;
                    sommeVert += vert;
                    sommeBleu += bleu;
                    compteurPixels++;
                    xmin = std::min(xmin, x);
                    xmax = std::max(xmax, x);
                    ymin = std::min(ymin, y);
                    ymax = std::max(ymax, y);
                } else {
                    sRx += rouge;
                    sVx += vert;
                    sBx += bleu;
                    //if (maxext < (rouge + vert + bleu)) maxext = rouge + vert + bleu;
                    ctrx++;
                }
            }
        }
        if (compteurPixels > 0) {
            moy[0] = sommeBleu / compteurPixels;
            moy[1] = sommeVert / compteurPixels;
            moy[2] = sommeRouge / compteurPixels;
        } else {
            moy = ref;
        }
        if (ctrx) {
                moyext[0] = sBx / ctrx;
                moyext[1] = sVx / ctrx;
                moyext[2] = sRx / ctrx;
        } //else moyext = moy;
        ecart = 3*ecart/2;
    }

    std::cout << "!!! max b g r "<< maxb<<" "<<maxg<<" "<<maxr<<std::endl;
       std::cout << "Moyennes BGR : " << moy << moyext << std::endl;
    if (pBox){
        pBox[0] = xmin;
        pBox[1] = xmax;
        pBox[2] = ymin;
        pBox[3] = ymax;
    }
    std::cout<<"encombrement "<<xmin<<","<<xmax<<","<<ymin<<","<<ymax<<
        " / "<< image.cols<<"x"<<image.rows<<std::endl;
    if (xmin > xmax) {  // protection contre artefact
        imaR = image.clone();
        if (pBox){
            pBox[0] = 0;
            pBox[1] = image.cols - 1;
            pBox[2] = 0;
            pBox[3] = image.rows - 1;
        }
    
        return;
    }
    // pour obtenir l'image ,
    // ajouter une bordure blanche maximale, limitée à 3 pixels
    xmin = std::max(0, xmin - 3);
    xmax = std::min(image.cols - 1, xmax + 3);
    ymin = std::max(0, ymin -3);
    ymax = std::min(image.rows - 1, ymax + 3);
    cv::Rect r;
    r.x = xmin;
    r.y = ymin;
    r.width = std::max(1, xmax - xmin + 1);
    r.height = std::max(1, ymax - ymin + 1);
    imaR = image(r).clone();
}

// validation du caractère obtenu par OCR
std::string ValiderOCR(std::string output, bool estserveur, bool inverse, uncoin& moncoin, const config& maconf) {
    // valider le caractère obtenu.
    // chiffre 1 : absence de gros symbole à coté du chiffre ni au centre
    // chiffre 2 ou 3 : absence de gros symbole à coté du chiffre, présence au centre
    // chiffre 4 à 9 ou 10 : présence de gros symbole
    // VDR : dessus à gauche ou dessous à droite : présence de gros symbole
    // VDR : (sinon) pas de gros symbole
    int printoption = maconf.printoption;
    int threadoption = maconf.threadoption;
    bool estGS = false; // gros symbole présent dans le coin ?
    bool estGScent = false; // GS central haut présent ? 
    bool estGScentral = false; // GS central présent ? 
    cv::Rect rr;
    cv::Mat GS;
    cv::Mat lig;
    cv::Mat extrait;
    cv::Scalar m, ect;
    bool estunRDV = false;
    extrait = moncoin.ima_coin.clone();

    // réactiver après ajout d'une option dans la configuration
    //if (moncoin.estunRDV && output == "J") output = "V";  // Valet anglais (J) 
    //if (moncoin.estunRDV && output == "Q") output = "D";  // Dame anglaise (Q)
    //if (moncoin.estunRDV && output == "K") output = "R";  // Roi anglais (R)
    //if (moncoin.estunRDV && output == "0") output = "D";  // Dame anglaise (Q) confondue avec 0
    //if (moncoin.estunRDV && output == "1") output = "V";  // Valet anglais (J) confondue avec 1

    if (output == "V" || output == "D" || output == "R")
    {
            //TODO : vérifier la présence de la tête ou du corps d'un personnage au mileu des cotés du coin
        // déterminer si c'est un personnage (Roi Dame ou Valet)
        // obtenir la valeur du blanc dans une petite colonne à droite de AB
        // au milieu de la bordure entre le bord de carte (AB) et le cadre éventuel
        // en évitant le caractère et le petit symbole en A et B
        // 
        // considérer les 4 zones entre le bord (exclus) et la position d'un éventuel gros symbole
        // si elles ne sont pas toutes blanches, c'est un personnage, on retourne 0
        //
        int valcarte = -1;
        cv::Mat GS; // gros symbole pour déterminer la couleur 
        int exclure = maconf. deltachiffre + maconf.taillechiffre + maconf.taillesymbole + maconf.deltachsymb + 2;
        cv::Mat lig;
        cv::Rect r;
        cv::Scalar m, mbl, mbl2;
        // déterminer la valeur du blanc
        // on trouve une partie blanche sous le caractère et le symbole entre le cadre et le gros symbole
        // tester les deux orientations possibles
        // orientation verticale:
        if (moncoin.estunRDV){
          r.width = maconf.deltagrosRDV - maconf.deltacadre - 2;
          if (moncoin.UU.x > moncoin.PP.x)  r.x = moncoin.QQ.x + 1;
          else r.x = moncoin.QQ.x - 1 - r.width;
          r.height = maconf.tailleVDR / 2;
          if(moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltaVDR + maconf.taillesymbole + 2;  
          else r.y = moncoin.QQ.y - maconf.deltaVDR - maconf.taillesymbole - 2;
        } else {
          r.width = maconf.deltagrosRDV - maconf.deltacadre - 2;
          if (moncoin.UU.x > moncoin.PP.x)  r.x = moncoin.PP.x + maconf.deltacadre+ 1;
          else r.x = moncoin.PP.x - maconf.deltacadre - 1 - r.width;
          r.height = maconf.tailleVDR / 2;
          if(moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltaVDR + maconf.taillesymbole + 2;  
          else r.y = moncoin.QQ.y - maconf.deltaVDR - maconf.taillesymbole - 2;
        }
        lig = moncoin.ima_coin(r);
        mbl = cv::mean(lig);
        if (mbl[0] < moncoin.moyblanc[0]) mbl = moncoin.moyblanc;
        // orientation horizontale
        if (moncoin.estunRDV){
          r.height = maconf.deltagrosRDV - maconf.deltacadre - 2;
          if (moncoin.U.y > moncoin.PP.y)  r.y = moncoin.QQ.y + 1;
          else r.y = moncoin.QQ.y - 1 - r.height;
          r.width = maconf.tailleVDR / 2;
          if(moncoin.UU.x > moncoin.QQ.x) r.x = moncoin.QQ.x + maconf.deltaVDR + maconf.taillesymbole + 2;  
          else r.x = moncoin.QQ.x - maconf.deltaVDR - maconf.taillesymbole - 2;
        } else {
          r.height = maconf.deltagrosRDV - maconf.deltacadre - 2;
          if (moncoin.U.y > moncoin.PP.y)  r.y = moncoin.PP.y + maconf.deltacadre + 1;
          else r.y = moncoin.PP.y - maconf.deltacadre - 1 - r.height;
          r.width = maconf.tailleVDR / 2;
          if(moncoin.UU.x > moncoin.PP.x) r.x = moncoin.PP.x + maconf.deltacadre + maconf.deltaVDR + maconf.taillesymbole + 2;  
          else r.x = moncoin.PP.x - maconf.deltacadre - maconf.deltaVDR - maconf.taillesymbole - 2;
        }
        lig = moncoin.ima_coin(r);
        mbl2 = cv::mean(lig);
        if (mbl2[0] > mbl[0]) mbl = mbl2;

        moncoin.moyblanc = mbl;
        if (!inverse) {
          // analyse de la tête éventuelle à gauche ou à droite (bord de carte)

          r.width = maconf.largeurgrosRDV;
          if (moncoin.UU.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.largeurcarte/2 - r.width / 2;
          else r.x = moncoin.PP.x - maconf.largeurcarte/2 - r.width / 2;
          r.height = maconf.taillegrosRDV;
          if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltacadre +1;
          else r.y = moncoin.PP.y - maconf.deltacadre - 1 - r.height;
          if (r.x >= 0 && r.y >= 0 && r.x < moncoin.ima_coin.cols - r.width && r.y < moncoin.ima_coin.rows - r.height ) {
              lig = moncoin.ima_coin(r);
              m = cv::mean(lig);
              if (mbl[0] > m[0] + 10) estunRDV = true;
          }
        } else { // inverse
          // analyse de la tête éventuelle dessus ou dessous (bord de carte)
          r.height = maconf.deltagros - 2;
          if (moncoin.U.y > moncoin.PP.y ) r.y = moncoin.PP.y + std::max(2,maconf.deltacadre -1);
          else r.y = moncoin.PP.y - std::max(2,maconf.deltacadre -1) - r.height;
          r.width = maconf.taillegrosRDV; // environ la taille de la tête
          if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.PP.x + maconf.deltacadre + 1;
          else r.x = moncoin.PP.x - maconf.deltacadre - 1 - r.width;
          if (r.x >= 0 && r.y >= 0 && r.x < moncoin.ima_coin.cols - r.width && r.y < moncoin.ima_coin.rows - r.height ) {
            lig = moncoin.ima_coin(r);
            m = cv::mean(lig);
            if (mbl[0] > m[0] + 10) {
              estunRDV = true;
            }
          }
        }
        if (!moncoin.estunRDV && estunRDV) {
          if (moncoin.UU.x > moncoin.PP.x) moncoin.QQ.x = moncoin.PP.x + maconf.deltacadre;
          else moncoin.QQ.x = moncoin.PP.x - maconf.deltacadre;
          if (moncoin.U.y > moncoin.PP.y) moncoin.QQ.y = moncoin.PP.y + maconf.deltacadre;
          else moncoin.QQ.y = moncoin.PP.y - maconf.deltacadre;
          moncoin.estunRDV = true;
        }
        if (!estunRDV){
            if (printoption > 1) std::cout<<output<<" !! incohérence RDV"<<std::endl;
            if (maconf.waitoption > 1) cv::waitKey(0);
            if (!moncoin.estunRDV) return ""; // ce n'est pas un personnage            
        }
            
        if (inverse){
            rr.width = maconf.taillegrosRDV;
            rr.height = 3 * maconf.largeurgrosRDV / 4;
        }else {
            rr.width = 3 * maconf.largeurgrosRDV / 4;
            rr.height = maconf.taillegrosRDV;
        }
        if (output == "D") { // couper la fleur tenue par la reine de carreau
            if (inverse) rr.width /= 2;
            else rr.height /= 2; 
        }
        if (!moncoin.estunRDV) {  // chiffre attendu ou indéterminé (et on a trouvé un R D V) 
            rr.width = maconf.largeurgros;
            rr.height = 3*maconf.taillegros/4;
            if (!inverse) { // OCR vertical ou OCR par le serveur (= orientation indéterminée)
                // Gros symbole à coté du caractère
                if (moncoin.UU.x > moncoin.PP.x)
                    rr.x = moncoin.PP.x  + maconf.deltagros + maconf.deltacadre;
                else
                    rr.x = moncoin.PP.x - maconf.deltagros - maconf.deltacadre - rr.width;
                if (moncoin.U.y > moncoin.PP.y)
                    rr.y = moncoin.PP.y + maconf.deltacadre + maconf.deltagroshaut;
                else
                    rr.y = moncoin.PP.y - maconf.deltacadre - maconf.deltagroshaut - rr.height;
            } else { // OCR inverse
                // gros symbole au dessus ou au dessous du caractère
                if (moncoin.UU.x > moncoin.PP.x)
                    rr.x = moncoin.PP.x + maconf.deltacadre + maconf.deltagroshaut;
                else
                    rr.x = moncoin.PP.x - maconf.deltacadre - maconf.deltagroshaut - rr.width;
                if (moncoin.U.y > moncoin.PP.y)
                    rr.y = moncoin.PP.y + maconf.deltacadre + maconf.deltagros;
                else
                    rr.y = moncoin.PP.y - maconf.deltacadre - maconf.deltagroshaut - rr.height;
            }
        } else // le caractère attendu est un Roi Dame ou Valet ( et trouvé)
        {
          rr.width = maconf.largeurgrosRDV;
          rr.height = 3*maconf.taillegrosRDV/4; // éviter l'épaule du roi ou la fleur de la reine
            if (!inverse) {
                if (moncoin.UU.x > moncoin.QQ.x)
                    rr.x = moncoin.QQ.x + 1 + maconf.deltagrosRDV;
                else
                    rr.x = moncoin.QQ.x - 1 - maconf.deltagrosRDV - rr.width;
                if (moncoin.U.y > moncoin.QQ.y)
                    rr.y = moncoin.QQ.y + maconf.deltagroshautRDV + 1;
                else
                    rr.y = moncoin.QQ.y - maconf.deltagroshautRDV - rr.height - 1;
            } else {
                if (moncoin.UU.x > moncoin.QQ.x)
                    rr.x = moncoin.QQ.x + maconf.deltagroshautRDV;
                else
                    rr.x = moncoin.QQ.x - maconf.deltagroshautRDV - rr.width;
                if (moncoin.U.y > moncoin.QQ.y)
                    rr.y = moncoin.QQ.y + 1 + maconf.deltagrosRDV + 1;
                else
                    rr.y = moncoin.QQ.y - 1 - maconf.deltagrosRDV - rr.height - 1;
            }
        }
        if ( rr.x < 0 || rr.y < 0 || rr.width > moncoin.ima_coin.cols - rr.x || rr.height > moncoin.ima_coin.rows - rr.y){
            if(printoption > 1) std::cout << output << " !! gros symbole hors de l'image " << std::endl;
            return "";
        }
        // extraire la zone du gros symbole attendu selon le caractère trouvé par OCR
        if (printoption > 1 && !threadoption){
            tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
        }
        GS = moncoin.ima_coin(rr).clone();
        // amplifyContrast(GS);
        cv::meanStdDev(GS, m, ect);
        if (m[0] - mbl[0] < -mbl[0]/5) estGS = true;
        //if (ect[0] > 40 + (255 - m[0]) / 5) estGS = true;
        bool invalide = false;
        if (inverse) {
          if (estGS) {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y)
            || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y)) 
              invalide = true;
          } else {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y)
            || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y))
              invalide = true;
          }
        } else {
          if (estGS) {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y)
            || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y)) 
              invalide = true;
          } else {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y)
            || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y))
              invalide = true;
          }
        }
        if (invalide) {
            if (!estserveur)
            {
                if (maconf.printoption > 1) {
                    std::cout << output << " !! caractere incompatible avec gros symbole ";
                    if (estGS) std::cout<<" present ";
                    else std::cout<< " absent ";
                    std::cout << std::endl;
                    if (maconf.waitoption) cv::waitKey(0);
                }
                return "";
            }
            else
            { // détecté même si caractère tourné de 90 degrés (ex: par trocr)
                return output;
            }
        }
        return output;


    //////////////////////////////////////////////
    } else  { // le caractère trouvé par OCR est un chiffre ( ou autre sauf V D R)
        if (output.size() == 2) {
            if (output[0] == '1'){  // 1x --> x 
                if (output[1] >= '1' && output[1] <= '9') {
                    std::cout<<output<<" --> "<<output[1]<<std::endl; 
                    output = output[1];
                }
                else if (output[1] != '0') output = "";
            } else output = "";
        }

        if (output != "10" && !(output >= "1" && output <= "9")) { return "";
        }
        if (output == "") return "";
        //
        // recalculer la valeur du blanc pour une carte entre 1 et 10
        // à l'intérieur du bord horizontal à coté de la zone chiffre + symbole
        // sur la largeur d'un chiffre, sur une hauteur jusqu'à un éventuel GS
        rr.width = maconf.largeurchiffre;
        rr.height = std::min(maconf.deltagros, maconf.deltagroshaut) - 2;
            if (moncoin.UU.x > moncoin.PP.x)
             rr.x = moncoin.PP.x + maconf.deltahaut + maconf.taillechiffre + maconf.taillesymbole + 2;
            else  rr.x = moncoin.PP.x - maconf.deltahaut - maconf.taillechiffre - maconf.taillesymbole - 2 - rr.width;
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.A.y +1;
            else rr.y = moncoin.PP.y - 1 - rr.height;
            lig = moncoin.ima_coin(rr);
            moncoin.moyblanc = cv::mean(lig);
        // TODO : à reprogrammer !
        // GS du coin absent: valide 1 2 ou 3
        //     4 : --> 1
        //     8 : --> 3
        //     7 : --> 1 2 ou 3
        //     >4 : --> invalide
        //     GS central à coté :
        //          2 ou 3 --> OK, 1 --> invalide
        // GS du coin présent: valide 4 à 10
        //         --> 4 (0 GS sous le GS du coin, 0 GS dans la colonne centrale)
        //         --> 5 (0 GS sous le GS du coin, 1 GS central )
        //         --> 6 (1 GS sous le GS coin, 0 GS cans la colonne centrale)
        //         --> 7 (1 GS au centre sous le coin, 1 GS dans la colonne centrale)
        //         --> 8 (1 GS central sous le coin, 2 GS dans la colonne centrale)
        //               ou (2 GS sous le GS du coin, 0 GS dans la colonne centrale) 
        //         --> 9 (2 GS sous le GS du coin, 1 GS dans la colonne centrale)
        //         --> 10 (2 GS sous le GS du coin, (2 GS dans la colonne centrale)
        //   
        //      
        rr.width = maconf.largeurgros;
        rr.height = maconf.taillegros;
        if (inverse){
            rr.width = maconf.taillegros;
            rr.height = maconf.largeurgros;
        }
        // zone du GS dans le coin
        if (inverse){
            if (moncoin.UU.x > moncoin.PP.x ) rr.x = moncoin.PP.x + maconf.deltagroshaut;
            else rr.x = moncoin.PP.x - maconf.deltagroshaut - rr.width;
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagros;
            else rr.y = moncoin.PP.y - maconf.deltagros - rr.height;

        } else {
            if (moncoin.UU.x > moncoin.PP.x ) rr.x = moncoin.PP.x + maconf.deltagros;
            else rr.x = moncoin.PP.x - maconf.deltagros - rr.width;
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagroshaut;
            else rr.y = moncoin.PP.y - maconf.deltagroshaut - rr.height;
        }
        if ( rr.x < 0 || rr.y < 0 || rr.width > moncoin.ima_coin.cols - rr.x || rr.height > moncoin.ima_coin.rows - rr.y){
            if(printoption > 1) std::cout << output << " !! GS du coin hors de l'image " << std::endl;
            return "";
        }
        lig = moncoin.ima_coin(rr);
        m = cv::mean(lig);
        if (moncoin.moyblanc[0] - m[0] > 20 ) estGS = true;

        if (printoption > 1 && !threadoption){
            tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
        }

        std::string retour = "";

        if (!estGS){   // pas de GS dans le coin : valide 1 2 ou 3
            //  GS central haut ? absent --> 1, présent --> 2 ou 3
            estGScent = false;
            if (inverse){
                rr.width = maconf.taillegros;
                rr.height = maconf.largeurgros;
                if(moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagroshaut;
                else rr.x = moncoin.PP.x - maconf.deltagroshaut - rr.width;
                if (moncoin.U.y  > moncoin.PP.y) rr.y = moncoin.PP.y + (maconf.largeurcarte - rr.height) / 2;
                else rr.y = moncoin.PP.y -(maconf.largeurcarte + rr.height) / 2;
            } else {
                rr.height = maconf.taillegros;
                rr.width = 2*maconf.largeurgros;
                if(moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagroshaut;
                else rr.y = moncoin.PP.y - maconf.deltagroshaut - rr.height;
                if (moncoin.UU.x  > moncoin.PP.x) rr.x = moncoin.PP.x + (maconf.largeurcarte - rr.width) / 2;
                else rr.x = moncoin.PP.x -(maconf.largeurcarte + rr.width) / 2;
            }
            if (rr.x < 0 || rr.y < 0 ||   rr.width > moncoin.ima_coin.cols - rr.x
                || rr.height > moncoin.ima_coin.rows - rr.y) {
                if (maconf.printoption > 1) std::cout<<output<<" GS centre haut  inaccessible"<<std::endl;
                return output; // controle impossible
                }
            lig = moncoin.ima_coin(rr);
            m = cv::mean(lig);
            if (moncoin.moyblanc[0] - m[0] > 20 ) estGScent = true;
            if (printoption > 1 && !threadoption){
                tracerRectangle(rr, extrait, "valider", cv::Scalar(0, 255, 0));
            }
            if (!estGScent) {
                retour = "1";
            } else { // GS centre haut présent : --> 2 ou 3
                // TODO : vérifier avec GS central
                if (inverse){
                    rr.width = maconf.taillegros;
                    rr.height = maconf.largeurgros;
                } else {
                    rr.width = maconf.largeurgros;
                    rr.height = maconf.taillegros;
                    if (moncoin.UU.x > moncoin.PP.x )
                         rr.x = moncoin.PP.x + (maconf.largeurcarte - rr.width) / 2;
                    else rr.x = moncoin.PP.x - (maconf.largeurcarte + rr.width) / 2;
                    if (moncoin.U.y > moncoin.PP.y) 
                         rr.y = moncoin.PP.y + (maconf.hauteurcarte - rr.height) / 2;
                    else rr.y = moncoin.PP.y - (maconf.hauteurcarte + rr.height) / 2;
                }
                if (rr.x < 0 || rr.y < 0 ||   rr.width > moncoin.ima_coin.cols - rr.x
                    || rr.height > moncoin.ima_coin.rows - rr.y) {
                    if (maconf.printoption > 1) std::cout<<output<<" GS central  inaccessible"<<std::endl;
                    return output; // controle impossible
                    }
                lig = moncoin.ima_coin(rr);
                m = cv::mean(lig);
                if (moncoin.moyblanc[0] - m[0] > 20 ) estGScentral = true;
                if (printoption > 1 && !threadoption){
                    tracerRectangle(rr, extrait, "valider", cv::Scalar(0, 255, 0));
                }
                if (estGScentral) retour = "3";
                else retour = "2";
                // manque de fiabilitité sur le GS central, peut-être caché par une autre carte :
                if (output == "2" || output == "3") retour = output;
            }
            if (printoption > 1) {
                if (retour == "") std::cout<<output<<" invalide"<<std::endl;
                else if (retour != output) std::cout<< output <<" --> "<< retour<<std::endl;
            }
            return retour;
        }
        // il y a un GS dans le coin --> 4 à 10
        // TODO 
        // compter le nombre de GS dans la colonne sous le GS du coin
        //      et le nombre de GS dans la colonne centrale
        // 
        int nbGScoin = 0;
        int nbGScent = 0;
        //
        if (false) {  // TODO : à réactiver après comptages
        if (nbGScoin == 0){ // 4 ou 5
            if (nbGScent > 0) retour = "4";
            else retour = "5";
        } else if (nbGScoin == 1){ // 6 7 ou 8
            if (nbGScent == 0) retour = "6";
            else if (nbGScent == 1) retour = "7";
            else retour = "8"; // 3 + 2 + 3 GS
        } else { // 2 GS sous le coin :  8 9 ou 10
            if (nbGScent == 0) retour = "8"; // 4 + 4 GS   pour certains modèles de cartes
            else if (nbGScent == 1) retour = "9"; // 4 + 1 + 4
            else retour = "10"; // 4 + 2 + 4 GS
        }
        }
        // selon le caractère:
        // 1 : --> 7 ou 4 ou 10   selon GS centre gauche ( TODO ) ou 5 ou 6 ou 8 ou 9
        // 2 : --> invalide
        // 3 : --> 8 ou 5    selon GS au centre (présent --> 5)
        // TODO :
        // 4 : vérifier abs GS 2∕4 et GS central gauche
        // 5 : ""
        // 6 : vérifier GS colonne centrale milieu haut et bas (--> 8    ou 7 si un seul)
        // 7 : GS central gauche absent --> 4 ou 5 (si GS au centre)
        // 8 : vérifier GS colonne centrale milieu haut et bas ( absents --> 6)
        if (output == "2") return ""; 
        // présence d'un GS  au dessous (position 2 sur 4) ou au milieu
        if (inverse) {
            rr.width = 2*maconf.taillegros ; // y compris GS au centre (ex : 7 ou 8)
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagros;
            else rr.y = moncoin.PP.y - maconf.deltagros - rr.height;
            if (moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagroshaut + 4*maconf.taillegros/3;
            else rr.x = moncoin.PP.x - maconf.deltagroshaut - 4*maconf.taillegros/3 - rr.width;

        } else {
            rr.height = 2*maconf.taillegros;
            if (moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagros;
            else rr.x = moncoin.PP.x - maconf.deltagros - rr.width;
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagroshaut + 4*maconf.taillegros/3;
            else rr.y = moncoin.PP.y - maconf.deltagroshaut - 4*maconf.taillegros/3 - rr.height;
        }
        if (rr.x < 0 || rr.y < 0 ||   rr.width > moncoin.ima_coin.cols - rr.x
            || rr.height > moncoin.ima_coin.rows - rr.y) {
            if (maconf.printoption > 1) std::cout<<output<<" GS gauche 2/4 inaccessible"<<std::endl;
            // return output; // controle impossible
            if (estGS && output == "1") output = "";  // ce pourrait être un 4 ou un 7
            if (estGS && output == "3") output = "";  // ce pourrait être un 8 ou un 5
        } else {
            std::string retour = "";
            // TODO : ne faire ce test que si c'est une nouvelle carte avec 4 cotés (donc complète)
            lig = moncoin.ima_coin(rr);
            m = cv::mean(lig);
            if (printoption > 1 && !threadoption){
                tracerRectangle(rr, extrait, "valider", cv::Scalar(0, 128, 128));
            }
            if (moncoin.moyblanc[0] - m[0] > 20 ) { // GS présent sous le coin 2/4 ou 2/3 ou 2&3/4
                // 1 --> 6 à 10    probable 10   possible 8
                // TODO :    tester GS 2/4  présent --> 10   absent --> 8
                // confirmer avec la colonne centrale : 0GS-->6 1GS-->7 2GS-->8 ou 10


                if (output == "1") {
                    retour =  "10";
                    std::cout<<output<<"-->"<<retour<<std::endl;
                } 
                else if (output == "3") {
                    retour =  "8";
                    std::cout<<output<<"-->"<<retour<<std::endl;
                }
                else if (output != "2" && output != "4" && output != "5") return  output;
                if (printoption > 1) {
                    if (retour == "") std::cout<<output<<" invalide"<<std::endl;
                    else std::cout << output <<" -->"<<retour <<std::endl;
                } 
                return retour;
            } else { // pas de GS à gauche sous le coin ( 4 ou 5)
                // TODO : discriminer avec le GS au centre (présent 7 --> 5   sinon 4)
                if (output == "7" ){
                    output = "5"; // TODO : pourrait être 4
                    std::cout<<output<<"-->"<<retour<<std::endl;
                } 
                if (output == "4" || output == "5" ) return output;
                if (output == "1") {
                    retour = "4";
                    std::cout<<output<<"-->"<<retour<<std::endl;
                } 
                if (output == "3") {
                    retour = "5";
                    std::cout<<output<<"-->"<<retour<<std::endl;
                } 
                if (printoption > 1) {
                    if (retour == "") std::cout<<output<<" invalide"<<std::endl;
                    else std::cout << output <<" -->"<<retour<<std::endl;
                } 
                return retour;
            }
            return output;
        }
    }
    return output;
}



// orientation et couleur
void calculerOrientation(uncoin& moncoin, const config& maconf) {
    // rechercher la pr�sence d'un gros symbole dans le coin
    // on sait si la carte est un Roi une Dame ou un Valet, ou un chiffre
    // si on trouve un gros symbole, on en d�duit l'orientation
    // sinon, on sait qu'il n'y a pas de gros symbole, donc on recherche la position du petit symbole
    // on en d�duit l'orientation.
    // on analyse le symbole, gros ou petit pour d�terminer la couleur (rouge ou noir)
    // on analyse la forme du symbole pour d�terminer la couleur Pique ou Trefle ou Coeur ou carreau
    // 
    // le gros symbole est dans l'angle du coin.
    // on consid�re un carr� � distance des bords au dela du caract�re et du petit symbole horizontal ou vertical
    // on se limite au quart du gros symbole proche du coin
    // il est pr�sent si la couleur moyenne de cette zone n'est pas blanche
    // 
    // d�terminer la couleur du "blanc"
    // on consid�re une ligne entre le bord horizontal et la position th�orique du cadre
    //   un peu au del� du coin pour �viter le bord arrondi
    int printoption = maconf.printoption;
    int threadoption = maconf.threadoption;
    cv::Mat lig;
    cv::Rect r;
    cv::Scalar m, moyblanc = moncoin.moyblanc;
    int bleulim = 20; // �cart de bleu entre le blanc et la zone test�e pour le petit symbole (*3 pour GS)
    bool estgrossymb = false;
    bool inverse(false);
    bool estDroit(false);
    cv::Mat GS;
    int numcol;
    cv::Mat z1, z2, z3;  // PS vertical, horizontal, sous PS vertical
    cv::Scalar m1, m2, m3;
    cv::Rect r2;
    cv::Mat imacop = moncoin.ima_coin.clone();
    int ts,ls;
    if(printoption > 0) std::cout<<"calcul orientation"<<std::endl;
    bool oriente = false;
    // si c'est un personnage, extraire la zone du gros symbole
    // extraire la partie commune aux deux orientations 
    // déterminer si c'est un GS
    // si c'est le cas, extraire le GS et obtenir la couleur
    //  _____________
    // |   ######
    // |   ######
    // |***++###
    // |***++###
    // |******
    if (moncoin.estunRDV){
      bool estGS = false;
      r.width = r.height = maconf.largeurgrosRDV - maconf.deltagrosRDV;
      if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagrosRDV;
      else                             r.x = moncoin.QQ.x - maconf.deltagrosRDV - r.width;
      if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagrosRDV;
      else                            r.y = moncoin.QQ.y - maconf.deltagrosRDV - r.height;
      lig = moncoin.ima_coin(r); m = cv::mean(lig);
      if (moyblanc[0] - m[0] > 20) estGS = true;
      if (moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y && !estGS) inverse = true;
      if (moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y && !estGS) inverse = true;
      if (moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y && estGS) inverse = true;
      if (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y && estGS) inverse = true;

      if (inverse) {
        r.width = maconf.taillegrosRDV; r.height = maconf.largeurgrosRDV;
        if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagroshautRDV;
        else                             r.x = moncoin.QQ.x - maconf.deltagroshautRDV - r.width;
        if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagrosRDV;
        else                            r.y = moncoin.QQ.y - maconf.deltagrosRDV - r.height;
      } else {
        r.height = maconf.taillegrosRDV; r.width = maconf.largeurgrosRDV;
        if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagrosRDV;
        else                             r.x = moncoin.QQ.x - maconf.deltagrosRDV - r.width;
        if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagroshautRDV;
        else                            r.y = moncoin.QQ.y - maconf.deltagroshautRDV - r.height;
      }
      moncoin.inverse = inverse;
      oriente = true;
      if (estGS) {
        GS = moncoin.ima_coin(r).clone();
        // redresser
        if (inverse){
          if (moncoin.U.y < moncoin.PP.y) { // dessus à droite, tourner de 90 degrés
            cv::rotate(GS, GS, cv::ROTATE_90_CLOCKWISE);
          } else {
            cv::rotate(GS, GS, cv::ROTATE_90_COUNTERCLOCKWISE);
          }
        } else {
          if (moncoin.U.y < moncoin.PP.y) cv::rotate(GS, GS, cv::ROTATE_180);
        }
        if (calculerRouge(GS,true,maconf,moyblanc)) moncoin.estRouge = true;
        else moncoin.estRouge = false;
        bool testdebug(false);
        if (testdebug) cv::imshow("GS", GS);
        return;
      }
    }

    //  |                    |
    //  | z1              z1 |
    //  |                    |  
    //  | W                W |
    //  | W  z2        z2  W |
    //  |_________  _________| 
    //  |                    |
    //  | W  z2        z2  W | 
    //  | W                W |
    //  |                    |
    //  | z1              z1 |

  if (!oriente) {
    // analyser la zone du petit symbole vertical (z1  UV)
    r.width = maconf.largeursymbole; r.height = 3*maconf.taillesymbole/2;
    // grossir sous la zone du symbole vertical (c'est blanc)
    r.x = moncoin.U.x;
    /**************************
    if (!moncoin.estunRDV){ // ???
        r.width -= (maconf.largeursymbole / 3); // donc 2/3
        if (moncoin.UU.x < moncoin.PP.x) r.x = moncoin.PP.x - maconf.deltasymbole - 1 - r.width;
        else r.x = moncoin.PP.x + maconf.deltasymbole +1;
    }
    *********************************/
    if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.U.y + 1;
    else r.y = std::max(0, moncoin.V.y - 1 - r.height);
    if(r.height > imacop.rows - r.y) r.height = imacop.rows - r.y;
    if (printoption > 1 && !threadoption) {
        cv::circle(imacop, moncoin.PP, 1, cv::Scalar(0,255, 0));
        if (moncoin.estunRDV) cv::circle(imacop, moncoin.QQ, 1, cv::Scalar(0,0, 255));
        tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,255)); // petit symbole vertical
    }
    if (r.width > moncoin.ima_coin.cols - r.x) r.width = moncoin.ima_coin.cols - r.x;
    if (r.height > moncoin.ima_coin.rows - r.y) r.height = moncoin.ima_coin.rows - r.y;

    z1 = moncoin.ima_coin(r).clone();
    cv::Rect r1 = r;
    m1 = cv::mean(z1);  // zone du petit symbole vertical

    // si on n'a pas encore déterminé l'orientation, analyser les deux zones du petit symbole
    if (!moncoin.inverse && !moncoin.estDroit){
        //TODO : petit symbole vertical? suivi d'une zone blanche de même taille
        //  sinon (zone du PS blanche ou zone suivante non blanche) PS horizontal
        // Gros symbole dans l'autre coté
        // ___________________
        //|            | 
        //| 9    GS    | 9  z2
        //| z1   GS    | GS
        //|      GS    | GS

        // U V UU VV mal recalculés si l'OCR n'a pas permis de déterminer l'orientation
        // dans ce cas , l'orientation a été fixée à ***inverse***

        // analyser la zone du petit symbole horizontal (z2)
        r.width = 3*maconf.taillesymbole / 2; r.height = maconf.largeursymbole;
        r.x = moncoin.UU.x; r.y = moncoin.UU.y;
        if (moncoin.UU.x < moncoin.PP.x) r.x = moncoin.VV.x - r.width;
        /*****************************
        if (!moncoin.estunRDV) {
            r.height -= (maconf.largeursymbole / 3); // gros symbole!
            r.x = moncoin.UU.x + 1;
            if (moncoin.U.y < moncoin.PP.y) r.y = moncoin.UU.y;
            else r.y = moncoin.UU.y + 1;
        }
        *****************************/
        if (printoption > 1 && threadoption == 0)
         tracerRectangle(r, imacop, "orient", cv::Scalar(255,255,0)); // PS horizontal
        z2 = moncoin.ima_coin(r).clone();
        m2 = cv::mean(z2);
        r2 = r;

        // amplifyContrast(z1);  // attention moyblanc pas amplifié
        if (moyblanc[0] + moyblanc[1] - m1[1] - m1[0 ] > 2*bleulim )
        {   // zone z1 (verticale) non blanche
            // c'est un petit symbole 
            // sauf si c'est un personnage (à droite dessus) ou (à gauche dessous)
            //     et c'est alors un gros symbole
            // vérifier maintenant la zone du symbole horizontal
            //         si elle est blanche, on valide la zone z1
            if (moyblanc[0] + moyblanc[1] - m2[1] - m2[0 ] < 2*bleulim ){ 
                // z2  blanche : z1 = petit symbole vertical
                // z2 blanche : valider z1
            } else {
                // z1 non blanche, z2 non blanche. z1 ou z2 morceau de gros symbole
                // ne se produit que pour R D ou V
                // cas possibles : 
                // non inversé 
                // dessus à gauche : seule possibilité petit symbole vertical (z1) et GS à sa gauche
                // dessous à droite : seule possibilité petit symbole vertical (z1) et GS à droite
                //
                // inversé
                // dessus à droite : ps horizontal (z2) et gS au dessus
                // dessous à gauche : ps horizontal (z2) et GS dessous
                inverse = false;
                if (moncoin.U.y < moncoin.PP.y && moncoin.UU.x > moncoin.PP.x) inverse = true;
                if (moncoin.U.y > moncoin.PP.y && moncoin.UU.x < moncoin.PP.x) inverse = true;
            }
        } // z1 non blanche
        else { // zone z1 blanche, le petit symbole est sur le coté horizontal
            // sauf si la zone 2 est encore plus blanche
            if (m2[0] + m2[1] + m2[2] - m1[0] - m1[1] - m1[2] < 0) inverse = true;
        }
        if(inverse) { // extraire le petit symbole et afficher
            r.x = moncoin.UU.x; r.y = moncoin.UU.y;
            r.width = moncoin.VV.x - moncoin.UU.x + 1;
            r.height = moncoin.VV.y -moncoin.UU.y + 1;
            if (printoption > 1 && !threadoption) tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,0));
            z1 = moncoin.ima_coin(r).clone();
            r1 = r;
        }
        moncoin.inverse = inverse;
    } else { // orientation déjà obtenue
        inverse = moncoin.inverse;
        estDroit = moncoin.estDroit;
        if (estDroit) inverse = false;
    }
  }
    //

    // TODO : utiliser la zone du caractère ou du gros symbole
    // petit symbole parfois en surcharge du sceptre du roi
    estgrossymb = false;
    if (moncoin.estunRDV) {
      if (inverse){
        if(( moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y )
        ||(moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y) ) {
            estgrossymb = true;
            r.width = ls = maconf.taillegrosRDV;
            r.height = ts = maconf.largeurgrosRDV;
            if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagroshautRDV;
            else r.x = moncoin.QQ.x - maconf.deltagroshautRDV - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagrosRDV;
            else r.y = moncoin.QQ.y - maconf.deltagrosRDV - r.height;
        }
      } else {
        if(( moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y )
        ||(moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y) ) {
            estgrossymb = true;
            r.width = ls = maconf.largeurgrosRDV;
            r.height = ts = maconf.taillegrosRDV;
            if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagrosRDV;
            else r.x = moncoin.QQ.x - maconf.deltagrosRDV - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagroshautRDV;
            else r.y = moncoin.QQ.y - maconf.deltagroshautRDV - r.height;
        }
      }
    } else if (moncoin.caractere == 'X' || moncoin.caractere >= '3' && moncoin.caractere <= '9') {
    // considérer aussi le gros symbole centrale pour carte 2 ou 3
        estgrossymb = true;
        int dG = (maconf.largeurcarte/2) - maconf.largeurgros / 2; // position gros symbole du 2 ou 3
        if (inverse) {
            r.width = ls = maconf.taillegros;
            r.height = ts = maconf.largeurgros;
            if (moncoin.caractere == '2' || moncoin.caractere == '3' ){
                if (moncoin.U.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.deltagroshaut;
                else r.x = moncoin.PP.x - maconf.deltagroshaut - r.width;
                if (moncoin.UU.y > moncoin.PP.y) r.y = moncoin.PP.y + dG;
                else r.y = moncoin.PP.y - dG - r.height;

            } else {
                if (moncoin.U.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.deltagroshaut;
                else r.x = moncoin.PP.x - maconf.deltagroshaut - r.width;
                if (moncoin.UU.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltagros;
                else r.y = moncoin.PP.y - maconf.deltagros - r.height;
            }
        }else {
            r.width = ls = maconf.largeurgros;
            r.height = ts = maconf.taillegros;
            if (moncoin.caractere == '2' || moncoin.caractere == '3' ){
                if (moncoin.U.x > moncoin.PP.x ) r.x = moncoin.PP.x + dG;
                else r.x = moncoin.PP.x - dG - r.width;
                if (moncoin.UU.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltagroshaut;
                else r.y = moncoin.PP.y - maconf.deltagroshaut - r.height;
            } else {
                if (moncoin.U.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.deltagros;
                else r.x = moncoin.PP.x - maconf.deltagros - r.width;
                if (moncoin.UU.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltagroshaut;
                else r.y = moncoin.PP.y - maconf.deltagroshaut - r.height;
            }
        }
    
    }
    if (!estgrossymb){
    // utiliser la zone du chiffre
      if (inverse){
        r.x = moncoin.UU.x;
        r.y = moncoin.UU.y;
        r.width = moncoin.VV.x + 1 - moncoin.UU.x;
        r.height = moncoin.VV.y + 1 - moncoin.UU.y;
      } else {
        r.x = moncoin.U.x;
        r.y = moncoin.U.y;
        r.width = moncoin.V.x + 1 - moncoin.U.x;
        r.height = moncoin.V.y + 1 - moncoin.U.y;
      }
    }

    // petit symbole s'il n'y a pas de gros symbole
    if (!estgrossymb){
    if (inverse) {
        ls = maconf.taillesymbole;
        ts = maconf.largeursymbole;
        r.x = moncoin.UU.x; r.y = moncoin.UU.y;
      } else {
        ts = maconf.taillesymbole;
        ls = maconf.largeursymbole;
        r.x = moncoin.U.x; r.y = moncoin.U.y;
      }
      r.width = ls; r.height = ts;
    }

    // z1 = gros symbole ou caractère ou petitsymbole à utiliser pour obtenir la couleur
    {
    int Box[4];
    cv::Scalar moy, moyext;
    cv::Mat imaR;
     // le symbole est soit rouge, soit noir
    // dans les deux cas, l'intensité bleue est plus faible que l'extérieur
    // déterminer la moyenne de bleu
    // ne considérer que les pixels moins bleus que cette moyenne
    // cumuler les intensités bleues et rouges de ces pixels
    // s'il y a plus de rouge que de bleu, c'est rouge

    // pour déterminer la couleur, agrandir la zone !!DESACTIVE
    // r = r1;
    // r.x--; r.y--; r.width += 2, r.height += 2;  // risque d'inclure un trait de bord
    if (r.x < 0) r.x = 0; if (r.y < 0) r.y = 0;
    if (r.width > moncoin.ima_coin.cols - r.x) r.width = moncoin.ima_coin.cols - r.x;
    if (r.height > moncoin.ima_coin.rows - r.y ) r.height = moncoin.ima_coin.rows - r.y;
    if (printoption > 1 && !threadoption) tracerRectangle(r, imacop, "orient", cv::Scalar(0,0,255));
    z1 = moncoin.ima_coin(r).clone();
    amplifyContrast(z1);
    calculerBox(z1, ts, ls, moy, Box, moyext, maconf);
    if (ts >= z1.rows && ls >= z1.cols)
     moyext = moncoin.moyblanc;
    // considérer les pixels moins bleus que la moyenne
    double cumb(0), cumr(0); 
    for(int y = Box[2]; y <= Box[3]; y++){
        for (int x = Box[0]; x <= Box[1]; x++){
            cv::Scalar pix = z1.at<cv::Vec3b>(y,x);
            if (pix[0] <= (moy[0] + moyext[0])/2) {
                cumb += pix[0];
                cumr += pix[2];
            }
        }
    }
    double ecartW = (cumr / cumb) / (moyext[2] / moyext[0]);
    if ( ecartW > 2.0) {   // préciser la limite après les tests (souvent >3 pour rouge)
        moncoin.estRouge = true; if (printoption > 1) std::cout << " Rouge! ecart = "<< ecartW<<std::endl;
    } else {
        moncoin.estRouge = false; if (printoption > 1) std::cout << " Noir ecart = "<< ecartW<<std::endl;
    }
#ifdef ACTIVER
    // c'est rouge si au moins un pixel est significativement rouge
    int rmb = -255; // rouge - bleu
    for (int x = 0; x < z1.cols; x++){
       for (int y=0; y< z1.rows; y++){
           cv::Vec3b pixel = z1.at<cv::Vec3b>(y, x);
           rmb = std::max(rmb,(pixel[2] - pixel[0]) );
       }
    }
    rmb -= (moyext[2] - moyext[0]);
    if (rmb  > 100 ){
        moncoin.estRouge = true; if (printoption) std::cout << " Rouge! rmb= "<< rmb;
    } else if (rmb - (moyext[2] - moyext[0]) < 50 ){
        moncoin.estRouge = false; if (printoption) std::cout << " Noir rmb= "<< rmb;
    }
    else {
        int ecartrelatif = 100.0*((double)moyext[2]/moyext[0]) / ((double)moy[2] / moy[0]);
        if ( ecartrelatif > 105 ){ // expérimental: rouge < 77   noir > 105
            moncoin.estRouge = false; if (printoption) std::cout << " Noir rmb "<< rmb<< ", "<< ecartrelatif;
        } else if (ecartrelatif < 77) {
            moncoin.estRouge = true; if (printoption) std::cout << " Rouge "<< rmb<< ", "<<ecartrelatif;
        } else {
            // c'est rouge si l'écart rouge-bleu est significatif
            double ecart = (moy[2] - moy[0]) / (256-moy[0]);
            if (ecart > 0.1){
                moncoin.estRouge = true; if (printoption) std::cout 
                << " Rouge! rmb "<< rmb<< ", "<<ecartrelatif<<", "<<ecart;
            } else {
                moncoin.estRouge = false; if (printoption) std::cout 
                << " Noir! rmb "<< rmb<< ", "<<ecartrelatif<<", "<< ecart;
            }
        }
    }
#endif    
}
moncoin.inverse = inverse;
return;

}



// Fonction de tri pour ordonner les points
std::vector<cv::Point2f> orderPoints(int pts[4][2]) {
    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; ++i)
        points.emplace_back(cv::Point2f(pts[i][0], pts[i][1]));

    // Calcul centre
    cv::Point2f center(0, 0);
    for (const auto& p : points)
        center += p;
    center *= (1.0f / points.size());

    // Séparer selon leur position par rapport au centre
    std::vector<cv::Point2f> ordered(4);
    for (const auto& p : points) {
        if (p.x < center.x && p.y < center.y) ordered[0] = p; // A (haut gauche)
        else if (p.x < center.x && p.y > center.y) ordered[1] = p; // B (bas gauche)
        else if (p.x > center.x && p.y > center.y) ordered[2] = p; // C (bas droit)
        else ordered[3] = p; // D (haut droit)
    }
    return ordered;
}



// détermine si l'image GS est un symbole rouge (true) ou noir (false)
bool calculerRouge(cv::Mat GS, bool estunRDV, const config& maconf, cv::Scalar mbl){
    // mbl : valeur du blanc
    int printoption = maconf.printoption;
    bool threadoption = maconf.threadoption;
    int waitoption = maconf.waitoption;
    cv::Mat symbgros = GS.clone();
    cv::Scalar moy, moyext, ect, m1;
    cv::Rect r;
    int ts, ls; // taille et largeur du symbole
    bool estRouge(false);
    if (estunRDV) {
        // c'est un personnage
        ts = maconf.taillegrosRDV;
        ls = maconf.largeurgrosRDV;
    } else {
        ts = maconf.taillegros;
        ls = maconf.largeurgros;
    }
    ts = std::min(ts, GS.rows);
    ls = std::min(ls, GS.cols);
    amplifyContrast(GS); // parfois contre productif
    if (printoption > 1 && !threadoption) {
        afficherImage("symbole", symbgros);
        afficherImage("gros", symbgros);
        //cv::waitKey(1);
    }
    // déterminer la couleur : Rouge ou noir ?
    int Box[4];
    cv::Mat z1, imaR;
     // le symbole est soit rouge, soit noir
    // dans les deux cas, l'intensité bleue est plus faible que l'extérieur
    // déterminer la moyenne de bleu
    // ne considérer que les pixels moins bleus que cette moyenne
    // en se limitant à la moitié gauche si c'est un personnage
    // cumuler les intensités bleues et rouges de ces pixels
    // s'il y a plus de rouge que de bleu, c'est rouge
    if (estunRDV) { // personnage
      r.x = 0; r.y = 0; r.width = ls / 2; r.height = ts;
      z1 = GS(r);
      calculerBox(z1, ts, ls/2, moy, Box, moyext, maconf);
      if (ts >= z1.rows && ls/2 >= z1.cols)  moyext = mbl; //cv::Scalar(255,255,255);
    } else { 
      z1 = GS.clone();
      calculerBox(z1, ts, ls, moy, Box, moyext, maconf);
      if (ts >= z1.rows && ls >= z1.cols)  moyext = mbl; //cv::Scalar(255,255,255);
    }
    // considérer les pixels moins bleus que la moyenne 
    float cumb(0), cumr(0);
    float rapportext = float(moyext[0]) / moyext[2];
    int nbpix(0);
    for(int y = Box[2]; y <= Box[3]; y++){
        for (int x = Box[0]; x <= Box[1]; x++){
            cv::Scalar pix = z1.at<cv::Vec3b>(y,x);
            //if (pix[0] <= (9*moy[0] + moyext[0]) / 10) { // un peu plus que la moyenne
            if (pix[0] <= moy[0]) { // un peu plus foncé que la moyenne
                cumb += pix[0];
                cumr += pix[2];
                nbpix++;
            }
        }
    }
    float ecartW = (cumr / cumb) * rapportext;
    if ( ecartW > 1.5 && cumr / nbpix > 50) {   // préciser la limite après les tests
        estRouge = true; if (printoption > 1) std::cout << " Rouge! ecart = "<< ecartW<<std::endl;
        return true;
    } else {
        estRouge = false; if (printoption > 1) std::cout << " Noir ecart = "<< ecartW<<std::endl;
        return false;
    }
#ifdef ACTIVER
    // c'est rouge si au moins un pixel est significativement rouge
    int rmb = -255; // rouge - bleu
    for (int x = 0; x < z1.cols; x++){
       for (int y=0; y< z1.rows; y++){
           cv::Vec3b pixel = z1.at<cv::Vec3b>(y, x);
           rmb = std::max(rmb,(pixel[2] - pixel[0]) );
       }
    }
    rmb -= (moyext[2] - moyext[0]);
    if (rmb  > 100 ){
        moncoin.estRouge = true; if (printoption) std::cout << " Rouge! rmb= "<< rmb;
    } else if (rmb - (moyext[2] - moyext[0]) < 50 ){
        moncoin.estRouge = false; if (printoption) std::cout << " Noir rmb= "<< rmb;
    }
    else {
        int ecartrelatif = 100.0*((double)moyext[2]/moyext[0]) / ((double)moy[2] / moy[0]);
        if ( ecartrelatif > 105 ){ // expérimental: rouge < 77   noir > 105
            moncoin.estRouge = false; if (printoption) std::cout << " Noir rmb "<< rmb<< ", "<< ecartrelatif;
        } else if (ecartrelatif < 77) {
            moncoin.estRouge = true; if (printoption) std::cout << " Rouge "<< rmb<< ", "<<ecartrelatif;
        } else {
            // c'est rouge si l'écart rouge-bleu est significatif
            double ecart = (moy[2] - moy[0]) / (256-moy[0]);
            if (ecart > 0.1){
                moncoin.estRouge = true; if (printoption) std::cout 
                << " Rouge! rmb "<< rmb<< ", "<<ecartrelatif<<", "<<ecart;
            } else {
                moncoin.estRouge = false; if (printoption) std::cout 
                << " Noir! rmb "<< rmb<< ", "<<ecartrelatif<<", "<< ecart;
            }
        }
    }
#endif    

}


// calcul de la couleur de bridge à partir du gros symbole
int calculerCouleur(cv::Mat GS, bool estunRDV, const config& maconf, cv::Scalar mbl){
    // mbl : valeur du blanc
    int printoption = maconf.printoption;
    bool threadoption = maconf.threadoption;
    int waitoption = maconf.waitoption;
    cv::Mat symbgros = GS.clone();
    cv::Scalar moy, moyext, ect, m1;
    int moyb; // intensité moyenne bleue
    cv::Rect r, rr;
    int yBH, hBH; // position et taille bande horizontale la plus foncée
    cv::Mat bande, centre, lig;
    int ts, ls; // taille et largeur du symbole
    bool estRouge(false);
    int numcol(-1); // 0=Pique, 2=Coeur, 3=carreau, 4=Trefle, -1=indéterminé
    if (estunRDV) {
        // c'est un personnage
        ts = maconf.taillegrosRDV;
        ls = maconf.largeurgrosRDV;
    } else {
        ts = maconf.taillegros;
        ls = maconf.largeurgros;
    }
    ts = std::min(ts, GS.rows);
    ls = std::min(ls, GS.cols);
    amplifyContrast(GS); // parfois contre productif
    if (printoption > 1 && !threadoption) {
        afficherImage("symbole", symbgros);
        afficherImage("gros", symbgros);
        //cv::waitKey(1);
    }
    // déterminer la couleur : Rouge ou noir ?
    estRouge = calculerRouge(GS, estunRDV, maconf, mbl);
if (false) {
    int Box[4];
    cv::Mat z1, imaR;
     // le symbole est soit rouge, soit noir
    // dans les deux cas, l'intensité bleue est plus faible que l'extérieur
    // déterminer la moyenne de bleu
    // ne considérer que les pixels moins bleus que cette moyenne
    // en se limitant au quart supérieure gauche si c'est un personnage
    // cumuler les intensités bleues et rouges de ces pixels
    // s'il y a plus de rouge que de bleu, c'est rouge
    if (estunRDV) { // personnage
      r.x = 0; r.y = 0; r.width = ls / 2; r.height = ts / 2;
      z1 = GS(r);
      calculerBox(z1, ts/2, ls/2, moy, Box, moyext, maconf);
    }
    else { z1 = GS.clone();
      calculerBox(z1, ts, ls, moy, Box, moyext, maconf);
    }
    if (ts >= z1.rows && ls >= z1.cols) {
     moyext = mbl; //cv::Scalar(255,255,255);
    }
    // considérer les pixels moins bleus que la moyenne pondérée entre la zone et l'extérieur (clair)
    double cumb(0), cumr(0); 
    for(int y = Box[2]; y <= Box[3]; y++){
        for (int x = Box[0]; x <= Box[1]; x++){
            cv::Scalar pix = z1.at<cv::Vec3b>(y,x);
            if (pix[0] <= (9*moy[0] + moyext[0]) / 10) { // un peu plus que la moyenne
                cumb += pix[0];
                cumr += pix[2];
            }
        }
    }
    double ecartW = (cumr / cumb) / (moyext[2] / moyext[0]);
    //double ecartW = (moy[2] / moy[0]) / (mbl[2] / mbl[0]);
    if ( ecartW > 1.8) {   // préciser la limite après les tests
        estRouge = true; if (printoption > 1) std::cout << " Rouge! ecart = "<< ecartW<<std::endl;
    } else {
        estRouge = false; if (printoption > 1) std::cout << " Noir ecart = "<< ecartW<<std::endl;
    }
}

    if (estRouge )
        eclaircirfond(GS);
    if (printoption > 1 && !threadoption) {
        afficherImage("symbole", symbgros);
        afficherImage("gros", symbgros);
        // cv::waitKey(1);
    }
    // calculer l'encombrement du symbole et les moyennes d'intensite du symbole complet
    int Box[4]; // xmin xmax ymin ymax
    calculerBox(GS, ts, ls, moy, Box, moyext, maconf);

    // si le symbole est rouge et si sa largeur est suffisante, 
    // on compare la partie supérieure à la partie inférieure
    // en se limitant à la partie gauche car la partie droite contient l'épaule du personnage
    // on compare le quart supérieur gauche au quart inférieur gauche
    // si c'est du coeur , il y a beaucoip moins de bleu en haut
    // sinon on utilise un autre moyen.
    if (estRouge && ls > 8) {

      if (estunRDV && ((ts - ls) >= ls/4 )) { // c'est un symbole RDV tronqué
        // considérer la ligne au quart supérieur
        // comparer le pixel en Box[0] + 1
        // au pixel en Box[1] +1 - ls/4
        int y = Box[2] - 1 + ts/4;
        cv::Scalar pixg, pixd;
        pixg = GS.at<cv::Vec3b>(y, Box[0] + 1);
        pixd = GS.at<cv::Vec3b>(y, Box[1] + 1 - ls/4);
        int bg = pixg[0]; 
        int bd = pixd[0];
        if (bd - bg > 100) numcol = 2; // clair à droite : carreau
        else if (bd - bg < 20) numcol = 1; // comparables : coeur
      }
      if (numcol >= 0) return numcol;


      // TODO : convertir en image monochrome
      // rechercher le haut du symbole : au moins un pixel rouge sur la ligne
      // puis rechercher une ligne .....*---*....    - > *+20
      // sur cette ligne rechercher un pixel clair bordé à gauche et à droite par des pixels rouges
      // si on trouve : compter le nombre de pixels rouges : au moins la moitié --> coeur
      //                sinon : carreau
      // sinon (pas rouge blanc rouge) : carreau
      // chercher le pixel rouge en haut de l'axe vertical
      cv::Scalar m;
      int x0, x1, x2; // positions des pixels foncés qui encadrent une zone claire
      x0 = 0;
      if (estunRDV) x0 = 1;
      r.y = 0;
      cv::Scalar pix, pix2;
      int pixrgb, pixrgb2;
      int maxrouge(0); // écart maximal rouge - bleu
      int minrgb(1000);
      int mblrgb = mbl[0] + mbl[1] + mbl[2];
      // rechercher une ligne ayant au moins un pixel foncé sur 3/4 gauche
      int ylim = std::min(GS.rows, Box[2] + 4);
      while(r.y < ylim){ // rechercher un pixel foncé, le plus foncé
        for (r.x = x0; r.x <= 3*GS.cols / 4 ; r.x++){
          pix = GS.at<cv::Vec3b>(r.y,r.x);
          pixrgb = pix[0] + pix[1] + pix[2];
          minrgb = std::min(minrgb, pixrgb);
        }
        if (mblrgb - minrgb < 150) r.y++;
        if (mblrgb - minrgb > 100) break;
      }
      if (r.y >= ylim) { // aucune ligne contrastée. 
        // on choisit arbitrairement coeur
        std::cout<< " symbole trop uniforme --> coeur par défaut"<<std::endl;
        return 1;
      }
      // on a une ligne contrastée
      // chercher un écart foncé puis clair significatif
      r.x = Box[0]; if (estunRDV) r.x = std::max(1, r.x);
      int minval(1000), valrgb;
      while(r.x < Box[1]){
        pix = GS.at<cv::Vec3b>(r.y,r.x);
        pixrgb = pix[0] + pix[1] + pix[2];
        minval = std::min(minval,pixrgb); // valeur du pixel le plus foncé
        pix2 = GS.at<cv::Vec3b>(r.y,r.x + 1);
        pixrgb2 = pix2[0] + pix2[1] + pix2[2];
        valrgb = pixrgb2;
        r.x++;
        if (valrgb - minval > 100) break; // un pixel plus clair 
      }
      if (r.x >= Box[1]){ // on n'a pas trouvé de transition foncé vers clair
        // probablement la partie supérieure du symbole coeur
        return 1;
      }
      x1 = r.x - 1;
      // ... foncé clair  : rechercher un pixel plus foncé à droite
      x2 = 0;
      while(r.x <= Box[1]){
        pix = GS.at<cv::Vec3b>(r.y,r.x);
        pixrgb = pix[0] + pix[1] + pix[2];
        if (valrgb - pixrgb > 100) {
          x2 = r.x;
          break;
        }
        valrgb = std::max(valrgb, pixrgb); // le plus clair dans le creux du coeur
        r.x++;
      }
      // compter le nombre de pixels foncés sur cette ligne. au moins un tiers --> coeur
      int nbfonce = 0;
      for (int x = x0; x <= Box[1]; x++){
        pix = GS.at<cv::Vec3b>(r.y,x);
        pixrgb = pix[0] + pix[1] + pix[2];
        if (pixrgb < valrgb) nbfonce++;
      }

      r.x = 0; r.width = GS.cols; r.height = 1;
      if (x2 > 0) {r.x = x1; r.width = x2 - x1 + 1;}
      if (nbfonce >= (Box[1] - Box[0])/ 2) {
        if(printoption > 0) tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0));
        return 1; // ... foncé clair ... foncé    et  très foncé: coeur
      } else if (nbfonce < (Box[1] - Box[0])/ 3){
        if(printoption > 0) tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
        return 2; 
      }

      // situation indécise : le nombre de pixels foncés est intermédiaire
      // effectuer une analyse plus précise

      int mh, mb, dm;
      if (!estunRDV || ((ts - ls) < ls/4 )) { // c'est un chiffre ou un symbole non tronqué
        r.x = Box[0]; r.width = (Box[1] - Box[0] + 1);
        //r.y = Box[2]; 
        r.height = (Box[3] - Box[2] + 1) / 2; // moitié supérieure
        lig = GS(r); mh = cv::mean(lig)[0];
        //r.y = Box[3] - r.height + 1; // moitié inférieure
        r.y += r.height;
        //if(r.y > GS.rows - r.height) r.y = GS.rows - r.height;
        if (r.height > Box[3] + 1 - r.y) r.height = Box[3] + 1 - r.y;
        lig = GS(r); mb = cv::mean(lig)[0];
        dm = mh - mb; //  bleu en haut  - en bas 
        if (dm < -50) numcol = 1; // coeur
        else if (dm > -10) numcol = 2; // carreau
      }
      else 
      if (false) { // c'est un personnage avec un symbole tronqué
        // TODO : ne fonctionne pas !!!!!!!!!!!! désactivation !!!!!!!!!!!!!!!!!!!!!
        // sur cerains jeux de cartes, le gros symbole est tronqué près du bord de carte
        // TODO : examiner la ligne supérieure Box[2]
        //  partir de la gauche, chercher un pixel foncé en position xcarreau (rouge - bleu >  ??)
        //  puis chercher un pixel clair en position xcoeur (rouge - bleu < ???)
        //  puis chercher un pixel foncé.
        //      si on trouve, xcoeur est le creux du coeur
        //      sinon xcarreau est le sommet du symbole carreau
        int bleuref;
        int xcoeur(0), xcarreau(0);
        int y = Box[2];
        int x = Box[0];
        while (xcarreau == 0 && y < Box[2] + 3){
          x = Box[0];
          while ( x <= Box[1]){ // chercher un pixel rouge
            cv:: Scalar pix = GS.at<cv::Vec3b>(y, x);
            if (pix[2] - pix[0] > 50) {
              xcarreau = x;
              bleuref = pix[0];
              x++;
              break;
            }
            x++;
          }
          if (xcarreau > 0) break;
          y++;
        }
        r.y = y;
        while ( x <= Box[1]){ // chercher un pixel blanc (plus de bleu que sur le premier pixel rouge)
          cv:: Scalar pix = GS.at<cv::Vec3b>(Box[2], x);
          if (pix[0] - bleuref > 40) {
            xcoeur = x; // pixel clair. creux du coeur ou à droite de la pointe du carreau 
            x++;
            break;
          }
          x++;
        }
        while ( x <= Box[1]){ // chercher un pixel rouge
          cv:: Scalar pix = GS.at<cv::Vec3b>(Box[2], x);
          if (pix[2] - pix[0] > 50) { // partie droite du coeur
            numcol = 1;
            xcoeur = x - 1;
            r.x = xcoeur;
            break;
          }
          x++;
        }
        if (numcol != 1) {numcol = 2; r.x = xcarreau;} 
        //return numcol;

        //r.x = (Box[0] + Box[1]) / 2; r.width = (Box[1] - Box[0] + 1) /2; // moitié droite
        // r.x = Box[0]; 
        r.width = (Box[1] - Box[0] + 1) /2; // moitié droite
        if (r.width > GS.cols - r.x) r.width = GS.cols - r.x;
        r.height = (Box[3] - Box[2] + 1) / 2; // quart supérieur
        r.y = Box[2];
        if (printoption > 1 && !threadoption)
          tracerRectangle (r, symbgros, "gros", cv::Scalar(255,0,0));
        lig = GS(r); mh = cv::mean(lig)[0];
        //r.x = Box[0];
        //r.y = Box[3] - r.height + 1; // quart inférieur gauche
        r.y += r.height;  if(r.y > GS.rows - r.height) r.y = GS.rows - r.height;
        if (printoption > 1 && !threadoption)
          tracerRectangle (r, symbgros, "gros", cv::Scalar(255,0,0));
        if (waitoption > 1) cv::waitKey(0);
        lig = GS(r); mb = cv::mean(lig)[0];
        dm = mh - mb; //  bleu en haut  - en bas : > 0 --> carreau
        if (dm > -10) numcol = 2; // c'est du carreau
        if (dm < -30) numcol = 1; // c'est du coeur
      }
    }

    // chercher la bande horizontale la plus sombre
    // hauteur paire si la taille du symbole est paire
    // en ignorant la partie centrale si la couleur est noire
    r.x = Box[0];
    r.width = Box[1] + 1 - Box[0];
    hBH = ts / 3;
    if ((ts & 1) != (hBH & 1)) { hBH--; if (hBH <= 0) hBH += 2; } // même parité
    // if (ts < 8) hBH = 1;
    r.height = hBH;
    int minb = 255;
    yBH = Box[2] + (ts - hBH) / 2;
    r.y = Box[2] + 1;
    rr = r;
    rr.x = r.x + ls / 3;
    rr.width = ls / 3;
    while (r.y <= Box[3] - r.height) {
        bande = GS(r);
        moy = cv::mean(bande);
        if (!estRouge) {
            centre = GS(rr);
            m1 = cv::mean(centre);
            moyb = (r.width * moy[0] - rr.width * m1[0]) / (r.width - rr.width);
            rr.y++;
        } else  moyb = moy[0];
        if (minb > moyb) {
            minb = moy[0];
            yBH = r.y;
        }
        r.y++;
    }
    r.y = yBH;
    cv::Scalar coulBH = cv::Scalar(0, 0, 255);
    if (estRouge) coulBH = cv::Scalar(0, 0, 0);
    cv::Mat bandeH = symbgros(r).clone();
    if (printoption > 1 && !threadoption)
        tracerRectangle(r, symbgros, "gros", coulBH);
    int xgBH = Box[0];

    int xmin, xmax;
    xmin = Box[0];
    xmax = Box[1];
    int xopt = -1; // position de l'axe verical du symbole
    // on a xmin et xmax du symbole
    r.x = xgBH; // gauche de la bande horizontale
    r.y = yBH;  // haut de cette bande
    r.width = std::min(Box[1] - r.x + 1, xmax - xmin + 1);
    // rechercher la position de la bande verticale optimale la plus sombre (= le moins de bleu)
    //  uniquement dans le tiers inférieur du symbole (en forme de V pour coeur et carreau)
    //      ou dans la partie supérieure pour symbole noir (pointe pique ou feuille supérieure du trefle)
    // essayer avec deux largeurs (2 et 1 ou 3) et choisir le résultat le moins clair
    //
    // TODO : rechercher la pointe inférieure du symbole coeur ou carreau
    // partir d'une ligne horizontale au milieu du symbole et descendre
    // jusqu'à trouver un pixel clair parmi les 3 premiers pixels
    // puis trouver le premier pixel rouge puis le dernier suivi d'un pixel clair
    // l'axe vertical est au milieu de ce segment rouge
    //
    //   |*****
    //   |*****
    //   | ***    <--- segment rouge recherché
    //   |  *
    if (estRouge) {
      cv::Scalar pix;
      int x= -1;
      int xr1; // premier pixel rouge, après un pixel blanc 
      int y = (Box[0] + Box[1] + 1)/ 2; // position au milieu 
      if (y >= Box[3])  y = Box[3];
      while (y <= Box[3]){
        while (y <= Box[3]){
          pix = GS.at<cv::Vec3b>(y,Box[0]);
          if (pix[0] - pix[2] > -20 ){ // pixel clair (pas trop rouge)
            x = Box[0];
            break;
          } else {
            pix = GS.at<cv::Vec3b>(y,Box[0] + 1);
            if (pix[0] - pix[2] > -20 ){ // pixel clair (pas trop rouge)
              x = Box[0] + 1;
              break;
            }
            y++;
          }
        }

        if (x >= Box[0]){ // on a trouvé une ligne avec un pixel clair en 0 ou 1
          // rechercher le prochain pixel rouge
          x++;
          while(x <= Box[1]){
            pix = GS.at<cv::Vec3b>(y,x);
            if (pix[2] - pix[0] > 20) break;
            x++;
          }
          if (x <Box[1]){
            // rechercher le prochain pixel clair
            xr1=x; x++;
            while(x <= Box[1]){
              pix = GS.at<cv::Vec3b>(y,x);
              if (pix[2] - pix[0] < 20) break;
              x++;
            }
          }
          if (x <= Box[1]) {
            // le segment xr1--> x est rouge
            // l'axe vertical est au milieu de ce segment
            r.width = 1;
            xopt = (xr1 + x) /2; break;
          }
        } else {
          // impossible de trouver le bas du symbole
        }
        y++;
      }



      r.y = yBH + (hBH + 1) / 2; // position au milieu la bande horizontale et au dessous
      if (r.y >= Box[3])  r.y = Box[3];
      r.height = Box[3] + 1 - r.y;
      if (r.height < 1) r.height = 1;
    }
    else
    { // symbole noir
      r.y = Box[2];
      r.height = std::max(1,ts / 3);
    }
    int largeurcol = 2;
    if (xopt < 0){ // on n'a pas encore trouvé l'axe vertical
      r.x = Box[0] + ls / 4;
      xmax = Box[1] + 1 - ls / 4;
      
      r.width = 2; // calcul avec largeur 2
      xopt = r.x;
      double minb3 = 255 * 2;
      double minb4 = minb3;
      while (r.x <= xmax - r.width)
      {
          bande = GS(r);
          moy = cv::mean(bande);
          double mbg = moy[0] + moy[1];
          if (mbg <= minb3) { minb3 = mbg; xopt = r.x; }
          r.x++;
      }
      int xopt3 = xopt;
      r.width = 1;
      if (ls > 8)  r.width = 3; // calcul avec largeur 1 ou 3
      r.x = xgBH;      // gauche de la bande horizontale
      if (ls >= 18) {
          r.x += ls / 6;
          xmax = Box[1] - ls / 4;
      }

      xopt = r.x;
      while (r.x <= xmax - r.width)
      {
          bande = GS(r);
          moy = cv::mean(bande);
          double mbg = moy[0] + moy[1];
          if (mbg <= minb4)
          {
              minb4 = mbg;
              xopt = r.x;
              largeurcol = r.width;
          }
          r.x++;
      }
      if (minb4 > minb3)
      {
          xopt = xopt3;
          r.width = 2;
      }

    }

//
/////////////////////// traitement symbole rouge ////////////////////////
//
    if (estRouge && numcol < 0)
    {
        // repositionner en hauteur en utilisant cette colonne centrale
        r.x = xopt;
        r.y = 0;
        r.height = GS.rows;
        bande = GS(r);
        int Box2[4];
        calculerBox(bande, ts, r.width, moy, Box2, moyext, maconf);
        Box[2] = Box2[2];
        Box[3] = Box2[3];

        int ybas = Box[3];
        // rechercher le haut du symbole, dans cette bande verticale, minimiser la moyenne  bleue
        r.x = xopt;
        int xaxe = xopt; // position de la gauche de  l'axe vertical central du symbole
        // ce devrait être au milieu de l'encombrement du symbole
        int xm = (Box[0] + Box[1] + 1 - r.width) / 2;
        if (xaxe != xm)
            if (printoption > 1)
                std::cout << " ecart calcul axe symbole rouge " << xm - xaxe << std::endl;
        // xaxe = xm; xopt = xm;
        r.y = Box[2]; // haut du symbole
        // on a le haut de cette colonne centrale étroite.
        // peut-être trop haut au milieu du coeur?
        //
        //
        //  coeur  carreau  carreau     tester les pixels x :
        // RR?RR    ?      R?R             x plus blanc que ? pour carreau sinon coeur
        // xRWRx  xRWRx   xRWRx
        // RRRRR  RRRRR
        //
        cv::Rect rr;
        rr.x = xaxe;
        rr.width = largeurcol;
        rr.y = r.y;
        rr.height = 1;
        bande = GS(rr);
        moy = cv::mean(bande); // 1 ou 3 pixels
        rr.y++;                // ligne en dessous
        if (rr.x > 0)
            rr.x--;
        rr.width += 2; // zone centrale élargie ( Zone RWR)
        if (rr.width > GS.cols - rr.x)
            rr.width = GS.cols - rr.x;
        bande = GS(rr);
        cv::Scalar moy1 = cv::mean(bande); // milieu (zone RWR)

        rr.x = xmin + 1; 
        xmax = Box[1];
        rr.width = xmax - xmin - 1; // ligne en dessous moins pixels du bord
        bande = GS(rr);
        cv::Scalar moy2 = cv::mean(bande);
        double mb = (moy2[0] * rr.width - moy1[0] * (largeurcol + 2)) / (rr.width - largeurcol - 2); // bords
        r.height = Box[3] + 1 - r.y;
        r.x = xaxe;
        r.width = largeurcol;
        if (printoption > 1 && !threadoption)
            tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0)); // bande vericale centrée
        if (moy[0] - mb > 20)
        { // centre moins rouge que les bords : coeur
            r.y++;
            numcol = 1;
        }
        else
            numcol = 2;

        if (numcol < 0)
        {
            // on a le haut du symbole dans la colonne centrale (r.y ) :
            // la pointe du carreau ou sous le creux du coeur
            r.x = xopt; // largeur  1 2 ou 3
            r.height = Box[3] + 1 - r.y;
            int wv = r.width; // largeur de la bande verticale
            if (printoption > 1 && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0)); // bande vericale centrée

            /*if (largeurcol == 1)*/ { // seul cas où il y a un doute TEST TOUJOURS --> OK experimental
                cv::Scalar m1(0), m2(0);
                int l1, l2, ytop;
                ytop = std::max(Box[2], r.y);
                cv::Rect rr;
                rr.x = xmin;
                rr.y = ytop;
                rr.height = 1;
                l1 = xaxe - xmin;
                if (l1 > 0)
                {
                    rr.x = xmin;
                    rr.width = xaxe - xmin;
                    rr.y = ytop + 1;
                    rr.height = 1;
                    bande = GS(rr);
                    m1 = mean(bande);
                }
                l2 = xmax - xaxe - largeurcol;
                if (l2 > 0)
                {
                    rr.x = xaxe + largeurcol;
                    rr.width = l2;
                    bande = GS(rr);
                    m2 = mean(bande);
                }
                else
                    l2 = 0;
                m1 = (m1 * l1 + m2 * l2) / (l1 + l2);
                double ecartbleu = m1[0] - moy[0];
                if (printoption > 1)
                    std::cout << "Ecart dessous - sommet " << ecartbleu << std::endl;
                if (ecartbleu > -30)
                {               // sommet rouge --> carreau
                    numcol = 2; // carreau
                }
                else if (ecartbleu < -100)
                    numcol = 1; // coeur
                // sinon indéterminé
            }
        }
        if (numcol < 0)
        {
            // !!!!!!! ceci ne devrait jamais arriver !!!!!!
            // peut-être trop haut au milieu du coeur?
            // c'est le cas si la ligne en dessous est nettement plus rouge (moins bleue)
            // creux du coeur si une petite ligne  à droite ou à gauche est plus rouge (= moins bleu)
            // analyser une petite ligne à gauche et à droite
            // cv::Rect rr; rr.x = Box[0]; rr.width = Box[1] - Box[0] + 1;
            // c'est du coeur si la ligne complète (moins la partie centrale) est plus rouge
            rr.x = xmin + 1;
            rr.width = xmax - xmin - 1;
            // rr.y = Box[2]; rr.height = 1;  // inchangé
            bande = GS(rr);
            cv::Scalar moy2 = cv::mean(bande);
            double mb = (moy2[0] * rr.width - moy[0] * largeurcol) / (rr.width - largeurcol);
            // rr.y++; bande = roi_image(rr); cv::Scalar moy2 = cv::mean(bande);
            if (moy[0] - mb > 30) // centre moins rouge que les bords : coeur probablement
                r.y++;
            // on a le haut du symbole dans la colonne centrale (r.y ) :
            // la pointe du carreau ou le creux du coeur
            r.x = xopt; // largeur  1 2 ou 3
            r.height = Box[3] + 1 - r.y;
            int wv = r.width; // largeur de la bande verticale
            if (printoption > 1 && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0)); // bande vericale centrée
            // on a la position haute du symbole : le haut du carreau, le creux du coeur
            // analyser un segment à gauche ou à droite à cette hauteur : très bleu (blanc)  pour carreau, moins pour coeur (un peu rouge)
            // si c'est un gros symbole, le bas du symbole peut être proche d'un autre gros symbole
            // il faut donc considérer un petit segment à droite ou à gauche, dirigé vers le bord de carte
            //   que la carte soit un chiffre ou un RDV (à gauche en cas de RDV)
            // si c'est un petit symbole, on peut choisir à droite ou à gauche
            // comparer à une ligne de même taille en bas du symbole
            // à droite si c'est un gros symbole
            //       droit dessous à gauche ou droit dessus à droite
            //   ou inverse dessus à gauche ou inverse dessous à droite
            // sinon : à gauche
            // bas du symbole : ybas
            bool adroite = true;

            int limblanc = 20; // valeur expérimentale
            r.height = 2;      // ligne haute de 2 pixels
            if (ts < 10)
                r.height = 1;
            r.width = ls / 3;
            if (adroite)
            {
                // entre xaxe + largeur axe   et Box[1] inclus
                r.x = xaxe + wv;
                r.width = Box[1] + 1 - r.x;
                r.width = 2 * r.width / 3;
                if (r.width > GS.cols - r.x)
                    r.width = GS.cols - r.x;
            }
            else
            {
                // entre Box[0] et xaxe-1
                r.width = std::max(1, xaxe - Box[0]);
                r.width = 2 * r.width / 3;
                r.x = std::max(0, xaxe - r.width);
            }
            if (r.height > Box[3] + 1 - r.y)
                r.height = Box[3] + 1 - r.y;
            if (printoption > 1 && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne en haut à gauche ou droite
            lig = GS(r);
            moy = cv::mean(lig);
            cv::Scalar moyHaut = moy;

            r.y = std::max(0, ybas + 1 - r.height);
            // r.x et r.width inchangés
            if (printoption > 1 && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne de test en haut à gauche
            lig = GS(r);
            moy = cv::mean(lig);
            // comparer l'intensité bleue entre le segment en haut et le segment en bas
            // coeur s'il y a significativement plus de bleu en bas, sinon carreau
            int ecartbleu = moy[0] - moyHaut[0];
            if (ecartbleu > 100) numcol = 1; // 100 : experimental
            else if (ecartbleu < 50) numcol = 2;
            // sinon : indéterminé
            if (maconf.printoption > 1)
            {
                if (numcol == 1)  std::cout << " coeur  ";
                if (numcol == 2)  std::cout << " carreau ";
                std::cout << " intensite bleu bas - haut " << ecartbleu << std::endl;
            }
            if (maconf.printoption > 1 && !maconf.threadoption)
                afficherImage("gros", symbgros);
        }
    }
    if (estRouge && numcol > 0)
    {
        if (printoption > 1 && !threadoption)
        {
            if (numcol == 1)  std::cout << " coeur" << std::endl;
            else              std::cout << " carreau " << std::endl;
            if (waitoption > 2) cv::waitKey(0); //else cv::waitKey(1);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// traitement du symbole noir ///////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    bool estNoir = !estRouge;
    if (estNoir)  {
        int BoxW[4];
        calculerBox(GS, ts, ls, moy, BoxW, moyext, maconf);
        numcol = -1;
            // considérer la partie  supérieure centrale : pointe du pique ou feuille supérieure du trefle
            //       *          ***** 
            //      ***        *******
            //     *****       ******* 
            // A  *******      *******
            // L  ********      *****   **
            // R **********      ***   *****
            //     pique        trefle

            // considérer la ligne en haut de la bande horizontale foncée : R
            // limité à la pointe du pique ou feuille du trefle
            // pique : très foncé
            // trefle : probablement assez foncé
            // considérer la ligne juste au dessus : L
            // pique : plus claire, Trefle : plus foncée
            // considérer la ligne précédente : A
            // pique : plus claire que L et R
            // Trefle : plus foncée que L et R
        // méthode du bas de la feuille supérieure du trefle
        // considérer le tiers central horizontal du symbole
        // partir du haut de la feuille (ignorer la première ligne)
        // descendre sur 40% du symbole
        // si on trouve une ligne significativement plus claire (que celle du milieu) : c'est du trefle 
        if (ts >= 10){

            cv::Rect r;
            r.height = ts/2;   // 50% semble correct après tests
            if (r.height < 5) r.height = 5; // au moins 5 pixels
            // r.width = (ls +2) /3;
            // r.x = BoxW[0] + ls/3;  // tiers central 
            r.width = largeurcol;
            r.x = xopt;  // l'axe vertical (sa gauche) a été calculé pour rouge et noir
            r.y = BoxW[2]; // haut du symbole
            if (printoption > 1 && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
            r.height = 1;
            cv::Mat lig = GS(r).clone();
            int mref = cv::mean(lig)[0]; // ref pour Pique limité à la composante bleue
            int mrefT = mref; // référence pour trefle : minimum (plus foncé) des 3 premières lignes
            int m;
            r.y++; // ligne suivante
            int ymax = BoxW[2] + std::max(5, ts/2);
            while(r.y < ymax) {
                lig = GS(r).clone();
                m = cv::mean(lig)[0];
                if (m > mrefT + 10) {numcol = 3; break;} // tige de la feuille de trefle
                else if (r.y > ymax - 2  && m < mref - 10) {numcol = 0; break;} // pique
                if (r.y <= BoxW[2] + 1) mref = std::min(m,mref);
                if (r.y <= BoxW[2] + 2) mrefT = std::min(m, mrefT);
                if (r.y == BoxW[2] + 2) {if (r.x) r.x--; r.width += 2;}
                r.y++;
            }
            if (printoption > 1 && !maconf.threadoption) {
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne discriminante
            }
        }

        // méthode du tiers central
        // si on n'a pas encore trouvé 
        // si le symbole est petit, calculer l'écart type du carré central (1/3)
        // réduite d'1 pixel de chaque coté
        // si c'est un Roi et s'il n'y a pas de gros symbole, on peut distinguer
        // le roi de pique du roi de trefle : le roi de pique porte une lyre sur son épaule gauche
        // TODO : si c'est une reine, en l'absence de gros symbole, la dame de pique a une couronne inclinée
        //  proche du caractère D dans la zone sous le symbole absent
        cv::Mat haut, centre;
        if (numcol < 0 ) {
            // écart type du tiers central:
            r.x = BoxW[0] + ls / 3;
            r.width = BoxW[1] + 1 - (ls) / 3 - r.x;
            // r.y = BoxW[2] + (ts)/3 ; r.height = (ts) / 3;
            r.y = yBH;
            r.height = hBH;
            if (maconf.printoption > 1 && !maconf.threadoption)
             tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 255, 0));
            centre = GS(r);
            // amplifyContrast(centre);
            cv::meanStdDev(centre, moy, ect);
            if (maconf.printoption > 1 )
                std::cout << " P/T ? écart type " << ect << std::endl;
            if (ect[0] < 10) numcol = 0;
            if (ect[0] > 60) numcol = 3;
            // Pique : écart type < 10
            // Trefle : > 60
            // indéterminé entre 10 et 60
            //
        }
        if (numcol < 0) // pas encore trouvé
        {
            // chercher au dessus de la bande sombre horizontale la colonne centrale la plus sombre
            // rectangle supérieur au dessus
            // largeur paire ls/3 pixels par excès ou ls/3 impair par défaut
            cv::Rect rc;
            int intmin = 765;
            cv::Scalar moyh, moyc;
            int wopt(2); // largeur optimale
            // avec largeur paire ?;
            r.width = ts / 3;
            if (r.width & 1)
                r.width++; // valeur paire par excès
            rc.width = r.width;
            r.y = Box[2];
            rc.y = yBH; // c'est la meilleure position
            rc.height = hBH;
            r.height = std::max(1, yBH - r.y);
            r.x = Box[0];
            int xopt = Box[0];
            while (r.x < Box[1] + 1 - r.width)
            {
                haut = GS(r);
                moyh = cv::mean(haut);
                if (moyh[0] + moyh[1] + moyh[2] < intmin)
                {
                    intmin = moyh[0] + moyh[1] + moyh[2];
                    xopt = r.x;
                }
                r.x++;
            }
            r.x = xopt;
            rc.x = r.x; // optimum pour largeur 2
            // pour largeur impaire:
            r.width--;
            rc.width--;
            r.y = Box[2];
            rc.y = yBH; // c'est la meilleure position
            rc.height = hBH;
            r.height = std::max(1, yBH - r.y);
            r.x = Box[0];
            while (r.x < Box[1] + 1 - r.width)
            {
                haut = GS(r);
                moyh = cv::mean(haut);
                if (moyh[0] + moyh[1] + moyh[2] < intmin)
                {
                    intmin = moyh[0] + moyh[1] + moyh[2];
                    xopt = r.x;
                    wopt = r.width;
                }
                r.x++;
            }
            r.x = xopt;
            rc.x = xopt;
            r.width = wopt;
            rc.width = wopt;

            if (maconf.printoption > 1 && !maconf.threadoption)
            {
                tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
                tracerRectangle(rc, symbgros, "gros", cv::Scalar(0, 255, 0));
            }

            haut = GS(r);
            centre = GS(rc);
            moyh = cv::mean(haut);
            moyc = cv::mean(centre);
            int moycent = moyc[0] + moyc[1] + moyc[2];
            int moyhaut = moyh[0] + moyh[1] + moyh[2];
            if (moyhaut == 0) moyhaut = 1; // ne pas diviser par 0
            int ecr = 100 * (moyhaut - moycent) / moyhaut;
            if (maconf.printoption > 1)
                std::cout << " intensite centre " << moycent << ", haut  " << moyhaut << ", ecr " << ecr << std::endl;
                if (ecr > 0) {
                    numcol = 0;
                    if (maconf.printoption > 1)  std::cout << "Pique";
                } else {
                    numcol = 3;
                    if (maconf.printoption > 1)  std::cout << "Trefle";
                }
        }
    }

    return numcol;
}

// extraire la carte dont on connait les 4 coins coins haut gauche et droit et bas droit et gauche
cv::Mat extraireCarteIncomplete(cv::Mat& image, int pts[4][2], config& maconf) {
  int printoption = maconf.printoption;
  int waitoption = maconf.waitoption;
  int threadoption = maconf.threadoption;
  bool estunRDV = false;
  //
  //  A_________________B     A est à gauche  AB est presque horizontal
  //  |                 |
  //  |                 |
  //  |_________________C     C est en bas à droite
  //
  cv::Mat imacarte;
  cv::Point2i AA(pts[0][0], pts[0][1]);
  cv::Point2i BB(pts[1][0], pts[1][1]);
  cv::Point2i CC(pts[2][0], pts[2][1]);
  cv::Point2i AB(BB-AA); //vecteur
  cv::Point2i BC(CC-BB);
  cv::Point2i WW;
  cv::Point2f W;
  cv::Point2f OO(0,0);
  int pv = (AA.x - BB.x)*(CC.y - BB.y) -  (AA.y - BB.y)*(CC.x - BB.x);
  float lgAB = cv::norm(AB); // largeur de la carte
  float lgBC = cv::norm(BC); // hauteur de la carte incomplète
  float angle, angle2;
    cv::Mat rot, A_mat;
    // TODO : tourner autour de A
  {
    angle2 = std::atan2(AB.y, AB.x);
      // tourner autour de A selon AB
      angle = angle2;
      WW = AA;

    // amener AA en (0,0)
  }
  rot = cv::getRotationMatrix2D(OO, angle * 180.0 / CV_PI, 1.0);
  std::vector<cv::Point2f> inputPoints = {WW};
  std::vector<cv::Point2f> rotatedPoints;
  cv::transform(inputPoints, rotatedPoints, rot);

  // rotation autour de l'origine
  cv::warpAffine(image, imacarte, rot, image.size());
  // affichage après rotation
  //cv::imshow("imrot0", imacarte); if (attendre) cv::waitKey(0);

  // déplacer

  W = rotatedPoints[0];
  double dx = -W.x;
  double dy = -W.y;
  rot.at<double>(0, 2) += dx;
  rot.at<double>(1, 2) += dy;
  
  // image déplacée
  cv::warpAffine(image, imacarte, rot, image.size());
  // affichage après rotation
  //cv::imshow("imrot", imacarte); if (attendre) cv::waitKey(0);
  // image redimensionnée
  int la = lgAB;
  int lg = lgBC;
  cv::Rect rW(0, 0, std::min(la, imacarte.cols), std::min(lg, imacarte.rows) );
  imacarte = imacarte(rW);
  if (maconf.waitoption >0){ cv::imshow("imrot2", imacarte); cv::waitKey(0);}
  return imacarte;
}


cv::Mat extraireCarte(cv::Mat& image, int pts[4][2], config& maconf);
cv::Mat extraireCarte(cv::Mat& image, int pts[4][2], config& maconf) {
  int printoption = maconf.printoption;
  int waitoption = maconf.waitoption;
  int threadoption = maconf.threadoption;
  bool estunRDV = false;
  cv::Mat imacarte;

  // considérer les deux premiers points A et B de longueur lgAB
  //   on sait que c'est un bord de carte soit la hauteur soit la largeur
  // si c'est la hauteur
  //  si les autres points sont à droite de AB (BA^BC > 0 )
  //      tourner autour de B, selon BA
  //  sinon tourner autour de A selon AB
  // si c'est une largeur
  //  si les autres points sont à droite de AB
  //     tourner autour de A selon AB
  //  sinon tourner autour de B selon BA
  bool attendre = false;
  // calcul de la matrice de déplacement
  cv::Point2i AA(pts[0][0], pts[0][1]);
  cv::Point2i BB(pts[1][0], pts[1][1]);
  cv::Point2i CC(pts[2][0], pts[2][1]);
  cv::Point2i AB(BB-AA); //vecteur
  cv::Point2i BC(CC-BB);
  cv::Point2i WW;
  cv::Point2f W;
  cv::Point2f OO(0,0);
  int pv = (AA.x - BB.x)*(CC.y - BB.y) -  (AA.y - BB.y)*(CC.x - BB.x);
  float lgAB = cv::norm(AB); // coté long ou court 
  float lgBC = cv::norm(BC);
  float angle, angle2;
    cv::Mat rot, A_mat;
    // tourner autour du coté le plus long et ajuster l'angle de 90 degrés
  if (lgAB > lgBC) {
    angle2 = std::atan2(AB.y, AB.x);
    if (pv > 0) {
      // tourner autour de A selon BC
      angle = angle2 - CV_PI / 2;
      WW = AA;
    }
    else {
      // tourner autour de B selon BC
      angle = angle2 + CV_PI / 2;
      WW = BB;
    }
    // amener BB en (0,0)
  } else { // AB largeur BC longueur
    angle2 = std::atan2(BC.y, BC.x);
    if (pv > 0 ) {
      // tourner autour de B selon BA
      angle = angle2 - CV_PI / 2;
      WW = BB;
    } else {
      // tourner autour de A selon AB
      angle = angle2 - CV_PI / 2; 
      WW = AA;
    }
  }
  rot = cv::getRotationMatrix2D(OO, angle * 180.0 / CV_PI, 1.0);
  std::vector<cv::Point2f> inputPoints = {WW};
  std::vector<cv::Point2f> rotatedPoints;
  cv::transform(inputPoints, rotatedPoints, rot);

  // rotation autour de l'origine
  cv::warpAffine(image, imacarte, rot, image.size());
  // affichage après rotation
  //cv::imshow("imrot0", imacarte); if (attendre) cv::waitKey(0);

  // déplacer

  W = rotatedPoints[0];
  double dx = -W.x;
  double dy = -W.y;
  rot.at<double>(0, 2) += dx;
  rot.at<double>(1, 2) += dy;
  
  // image déplacée
  cv::warpAffine(image, imacarte, rot, image.size());
  // affichage après rotation
  //cv::imshow("imrot", imacarte); if (attendre) cv::waitKey(0);
  // image redimensionnée
  int la = std::min(lgAB, lgBC);
  int lg = std::max(lgAB, lgBC);
  cv::Rect rW(0, 0, std::min(la, imacarte.cols), std::min(lg, imacarte.rows) );
  imacarte = imacarte(rW);
  //cv::imshow("imrot2", imacarte); if (attendre) cv::waitKey(0);
  return imacarte;
}


// obtenir la valeur et la couleur d'une carte

int decoderCarte1a10 (cv::Mat imacarte, config& maconf, int& numcol);

int decoderCarte(cv::Mat& image, int pts[4][2], config& maconf, int& numcol) {
    int printoption = maconf.printoption;
    int waitoption = maconf.waitoption;
    int threadoption = maconf.threadoption;
    bool estunRDV = false;
    cv::Mat imacarte = extraireCarte(image, pts,maconf);
    return decoderLaCarte(imacarte, maconf, numcol);
}

int decoderLaCarte(cv::Mat& imacarte, config& maconf, int& numcol) {
    int printoption = maconf.printoption;
    int waitoption = maconf.waitoption;
    bool estunRDV = false;

    cv::Mat carte = imacarte.clone();
    // afficher l'image extraite :
    if (printoption > 1) afficherImage("imacarte", imacarte);

    // 
    // l'image obtenue correspond aux 4 points:
    // A (0,0)
    // B (0, imacarte.rows - 1)
    // C (imacarte.cols - 1, imacarte.rows-1)
    // D (imacarte.cols - 1, 0)
    // analyser l'image redressée
    //
    // déterminer si c'est un personnage (Roi Dame ou Valet)
    // obtenir la valeur du blanc dans une petite colonne à droite de AB
    // au milieu de la bordure entre le bord de carte (AB) et le cadre éventuel
    // en évitant le caractère et le petit symbole en A et B
    // 
    // considérer les 4 zones entre le bord (exclus) et la position d'un éventuel gros symbole
    // si elles ne sont pas toutes blanches, c'est un personnage, on retourne 0
    //

    int valcarte = -1;
    cv::Mat GS; // gros symbole pour déterminer la couleur 
    int exclure = maconf. deltachiffre + maconf.taillechiffre + maconf.taillesymbole + maconf.deltachsymb + 2;
    cv::Mat lig;
    cv::Rect r, rr;
    cv::Scalar m, mbl, mbl2;
    // déterminer la valeur du blanc
    // on trouve une partie blanche dans le coin haut droit entre la tête du personnage et le caractère
    // sous le cadre et de hauteur la moitié de la hauteur du gros symbole
    // ou sous le chiffre et le symbole du bord gauche de la carte
    // choisir le plus clair (en bleu)
    // pour un personnage: à gauche du coin haut droit
    r.width = maconf.largeurVDR;
    r.x = imacarte.cols - maconf.deltacadre - maconf.largeurVDR - r.width - 2; // à gauche du caractère de droite
    r.y = maconf.deltacadre + maconf.tailleVDR; // nettement sous le cadre
    r.height = maconf.tailleVDR / 2; // pas trop haut
    r.height = std::min(r.height, imacarte.rows - r.y);
    if (r.y >= 0 && r.height > 0 && r.y + r.height <= imacarte.rows) {
      lig = imacarte(r);
      mbl = cv::mean(lig); // valeur du blanc
      if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,255,0));
    }
    // pour un personnage ou une autre carte (1 à 10): sous caractère et symbole du coin haut gauche
    if (maconf.deltagros > 5) {
        r.x = 3;
        r.width = maconf.deltagros - 4;
    } else {
        r.x = 2;
        r.width = maconf.deltagros - 2;
    }
    r.y = exclure; // sous le caractère et le symbole
    r.height = maconf.tailleVDR;
    if (r.y < imacarte.rows - r.height) {
      lig = imacarte(r);
      mbl2 = cv::mean(lig); // valeur du blanc
      if (mbl2[0] > mbl[0]) mbl = mbl2;
      if (printoption > 1 && !threadoption){tracerRectangle(r,carte, "carte",cv::Scalar(0,255,0));}
    }
    // déterminer si c'est un personnage ou un chiffre
    // la zone centrale verticale est alors colorée (corps du R D V)
    // ________________________________cadre haut
    // |    **         ** : zone du haut de la tête
    // |    **         ++ : zone du GS central haut
    // |    ++         == : zone du GS central milieu haut
    // |    ++         ## : zone du GS central
    // |    ==               pour un RDV les 4 zones sont foncées
    // |    ==
    // |    ## 
    // |    ##
    // |_______________________
    // GS central blanc (2 4 6 7 10) ou GS haut gauche blanc (1 2 3)
    //  GS haut centre blanc (1 4 5 6 7 8 9 10) 
    // on teste GS haut centre : blanc --> forcément un chiffre
    //    sinon on teste GS haut gauche : blanc --> 1 2 ou 3 --> 
    //
    estunRDV = true; // a priori c'est un personnage
    // position et largeur communes pour les 4 zones
    r.width = maconf.largeurgros;
    r.x = (maconf.largeurcarte - maconf.largeurgros)/2;
    if (r.x < maconf.deltagros + maconf.largeurgros){
        r.x = maconf.deltagros + maconf.taillegros +1;
        r.width = imacarte.cols - 2*r.x;
    }
    if (r.width <= 0) return 0; // carte (!) trop étroite

    // zone de la tête (**):
    r.y = maconf.deltacadre;
    r.height = std::min(imacarte.rows - r.y, 2*maconf.taillegros/3);
    r.height = std::min(r.height, maconf.deltagroshaut - r.y); // éviter le GS de la carte 2 ou 3
    rr = r;
    lig = imacarte(r);
    m = cv::mean(lig);
    int dm0 = m[0] - mbl[0];
    if (dm0 >=  - 40) { // zone claire : chiffre
        estunRDV = false; 
    } 
    if (estunRDV) { // pas encore identifié NON RDV
        // zone du GS Haut central (++)
        r.y = maconf.deltagroshaut; // zone du GS central haut (chiffre 2 ou 3  ou personnage)
        r.height = std::min(imacarte.rows - r.y, 2*maconf.taillegros/3);
        lig = imacarte(r);
        m = cv::mean(lig);
        int dm0 = m[0] - mbl[0]; // maximum observé pour un personnage: -33
        if (dm0 >=  - 20) {  // clair, donc pas un personnage (ni 2 ni 3) 
                estunRDV = false; 
        } 
    }
    if (estunRDV) { // pas encore identifié NON RDV
        // zone du GS milieu  central haut (==) exemple 8 9 ou 10
        r.y += maconf.taillegros;
        if (r.y + r.height < imacarte.rows) {
          lig = imacarte(r);
          m = cv::mean(lig);
          int dm0 = m[0] - mbl[0]; // observé uniquement sur des personnages < - 100
          if (dm0 >=  - 50) {  // clair,donc un chiffre (ni 1 ni 2 ni 3 ni 4 ni 5 ni 6 )
                  estunRDV = false; 
          } 
        }
    }
    if (estunRDV) { // pas encore identifié NON RDV
        // zone du GS milieu (##) exemple 3 ou 5
        r.y = (imacarte.rows - r.height)/2;
        if (r.y + r.height < imacarte.rows) {
          lig = imacarte(r);
          m = cv::mean(lig);
          int dm0 = m[0] - mbl[0]; // observé uniquement sur des personnages. < -77
          if (dm0 >=  - 50) {  // clair, donc un chiffre (ni 1 ni 3 ni 5 ni 7 )
                  estunRDV = false; 
          } 
        }
    }
    if (printoption > 1 && !threadoption){
        if (estunRDV)  tracerRectangle(rr,carte, "carte",cv::Scalar(0,0,255));
        else  tracerRectangle(rr,carte, "carte",cv::Scalar(255,0,0));
    }

    // décodage d'une petite carte :
    if (!estunRDV) return decoderCarte1a10 (imacarte, maconf, numcol);

    ////////////////// traitement d'un personnage ////////////////////////
      std::string output;
      cv::Mat ima_CARG; // image pour OCR caractère gauche
      // positionner précisément en recherchant les cadres haut et gauche

      //TODO : tester la situation "normale" : noir (noir) blanc (blanc) noir (noir) blanc
      //    avec la dernière position au maximum à deltacadre + 1
      //     sinon : faire le test complet 
      // distinguer les cas en considérant la partie droite 
      //
      //  ____________     ________________     <--- position sur la ligne foncée
      //    **       |                 =??|
      //   ****      |     ________________
      //             |      **          DD| <-- (=) espace entre la tête et le caractère
      //                   ***        = DD| <--     étroit pour R Pique et coeur et d'autres
      //                   ***        = DD|      (=) : zone à analyser
      //                   ***          DD|  
      //

      //        partir de la droite de la tête du personnage, sous la position maximale possible du cadre
      //        hauteur taille du caractère / 2, largeur 1 pixel
      //        aller à droite tant qu'il y a un pixel foncé
      //        --> position gauche
      r.x = imacarte.cols / 2 + maconf.largeurgros/2; // éviter la figure claire de la tête
      r.y = 2*maconf.deltacadre; // devrait être sous le cadre réel
      r.width = 1;
      r.height = maconf.taillechiffre;
      for (; r.x < imacarte.cols - maconf.deltacadre - maconf.largeurVDR; r.x++ ){
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] > mbl[0] - 30) break; // on a trouvé la zone blanche à droite de la tête
      } 

      //        largeur : largeur caractère /2    maximum 3 pixels
      //        hauteur 1 pixel
      //        remonter jusqu'au cadre (ligne non blanche) 
      //          
      // partir sous la position attendue du cadre, (position sous le caractère)
      // chercher en remontantant une ligne foncée (mbl - 30): cadre 
      int cadresup = maconf.deltacadre + maconf.tailleVDR; // position nettement blanche
      r.width = maconf.largeurVDR / 2;
      r.height = 1;
      for (r.y=cadresup; r.y > 0; r.y--){
          lig = imacarte(r); m = cv::mean(lig);
          if (m[0] < mbl[0] - 50) break; // ligne foncée = cadre
      }
      cadresup = r.y;
      //TODO : vérifier que c'est bien un trait foncé à gauche (au dessus du caractère)
      //      sinon remonter d'un pixel

      // rechercher le cadre bas, pour une carte complètement visible
      // partir nettement au dessus de la position attendue du cadre, (position sur le caractère)
      // chercher en descendant une ligne foncée (mbl - 30): cadre
      int cadreinf = 0;
      if (imacarte.rows >= maconf.hauteurcarte) { // carte complète
        cadreinf = imacarte.rows - maconf.deltacadre - maconf.tailleVDR; // position nettement blanche
        r.width = maconf.largeurVDR / 2;
        r.x = maconf.deltacadre + maconf.deltaVDR + maconf.largeurVDR + 2;
        r.height = 1;
        for (r.y=cadreinf; r.y < imacarte.rows; r.y++){
            lig = imacarte(r); m = cv::mean(lig);
            if (m[0] < mbl[0] - 30) break; // ligne foncée = cadre
        }
        cadreinf = r.y;
      }
      // on a trouvé la position du cadre en haut et en bas
      // considérer le caractère en haut à droite (pas géné par un gros symbole)
      //        partir d'une position bien à gauche du caractère
      //        chercher une petite colonne non blanche
      //
      // ____________________   <-- position du cadre supérieur
      //       ****    R |  |
      //       ****    R |  |  
      //
      r.y = cadresup + 1; r.height = maconf.tailleVDR + 1;
      r.x = imacarte.cols - maconf.deltacadre - maconf.deltaVDR - maconf.largeurVDR - 4;
      r.width = 1;
      cv::Scalar m1;
      lig = imacarte(r); m1 = cv::mean(lig); // colonne claire, utilisée comme référence
      for (;r.x < imacarte.cols + 2 - maconf.largeurVDR; r.x++ ){
          lig = imacarte(r); m = cv::mean(lig);
          if (m[0] < m1[0] - 20) { // colonne un peu foncée
              // c'est le bord gauche du caractère
              break;
          }
      }
      int xcaractere = r.x;  // début du caractère en haut à droite

      // chercher la position du caractère bas gauche
      int xcaracterebas;
      int cadregauche = maconf.deltacadre;
      if (cadreinf > 0 ){ // carte complètement visible
        r.height = maconf.tailleVDR;
        r.y = cadreinf - 1 - r.height;
        r.x = maconf.deltacadre + maconf.deltaVDR + maconf.largeurVDR + 4;
        r.width = 1;
        lig = imacarte(r); m1 = cv::mean(lig);
        for (;r.x >= maconf.largeurVDR; r.x-- ){
            lig = imacarte(r); m = cv::mean(lig);
            if (m[0] < m1[0] - 20) { // colonne un peu foncée
                // c'est le bord droit du caractère
                break;
            }
        }
        xcaracterebas = std::max(0, r.x - maconf.largeurVDR + 1);
        cadregauche = std::max(0,xcaracterebas - maconf.deltaVDR);
      }

      // on a la position du cadre à gauche et en haut

      // extraire le gros symbole en haut à gauche (toujours présent pour un personnnage)
      //        et calculer la couleur (GS de taille différente pour RDV)
      
      // ne pas élargir à gauche et à droite (présence du caractère et de la tête)
      // considérer la moitié gauche, élargie en bas 
      // puis chercher le rectangle intérieur le plus foncé
      // on obtient le haut du symbole
      int box[4];
      // rechercher une petite colonne blanche à gauche du symbole en partant du milieu
      r.x = cadregauche + maconf.deltagrosRDV + (maconf.largeurgrosRDV + 1) / 2;
      r.width = 1;
      r.y = cadresup + maconf.deltagroshautRDV; // + maconf.taillegrosRDV /4;
      r.height = std::min(imacarte.rows - r.y, maconf.taillegrosRDV / 2);
      lig = imacarte(r); m1 = cv::mean(lig); // colonne au centre du symbole
      r.x--;
      while (r.x >= cadregauche + maconf.deltagrosRDV) {
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] - m1[0] > 50) break;
        if (m[0] < m1[0]) m1 = m; // colonne de référence la plus sombre
        r.x --;
      }
      // on ne fait pas confiance au calcul de cadregauche
      //r.x = cadregauche + maconf.deltagrosRDV;
      r.y = cadresup + maconf.deltagroshautRDV;
      r.width = maconf.largeurgrosRDV / 2; // moitié gauche
      r.height = std::min(imacarte.rows - r.y, maconf.taillegrosRDV + 2);
      GS = imacarte(r).clone();
      calculerBox(GS,maconf.taillegrosRDV, r.width, m, box, m1, maconf );
      r.y += box[2]; r.height = std::min(imacarte.rows - r.y, maconf.taillegrosRDV);
      r.width = maconf.largeurgrosRDV; // totalité du gros symbole
      GS = imacarte(r).clone();
      if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,255,0));
      if (numcol < 0) numcol = calculerCouleur(GS, true,  maconf, mbl);
      // fausse détection possible entre coeur et carreau
      // la couleur (rouge ou noir) est fiable
      // si c'est rouge, partir de la ligne au centre du symbole
      // et remonter tant qu'il y a au moins un pixel rouge
      if (numcol == 1 || numcol == 2) {
        int y = cadresup + maconf.taillegrosRDV / 2;
        cv::Scalar pix;
        bool estpixelrouge(false);
        while (y >= cadresup + maconf.deltagroshautRDV) {
          estpixelrouge = false;
          for (int x = r.x + 1 ; x < r.x + maconf.largeurgrosRDV - 1; x++) {
            pix = imacarte.at<cv::Vec3b>(y,x);
            if (pix[2] - pix[0] > 50) {
              estpixelrouge = true;
              break;
            }
          }
          if (!estpixelrouge) { // on a trouvé une ligne (y)  sans pixel rouge
            y++; // haut du symbole
            r.height = maconf.taillegrosRDV;
            r.y = y; if (r.height > imacarte.rows - r.y) r.height = imacarte.rows - r.y;
            GS = imacarte(r).clone();
            bool testdebug(false);
            if (testdebug) afficherImage("GS", GS);
            numcol = calculerCouleur(GS, true,  maconf, mbl);
            break;
          }
          y--;
        }
      }
      int xGS = r.x;
      int yGS = r.y + box[3]; // bas du Gros Symbole

      // ne considérer que le haut de la carte (à cause des cartes recouvertes du mort)
      //
      // déterminer la valeur R D V sans décoder le caractère R D V
      //
      if (numcol == 3) { // Trefle
      // couleur T Trefle :
      //   V si le chapeau est bleu ou rouge (plus que la moyenne des 3 couleurs)
      //   D si la droite de la couronnne est claire
      //   R sinon
      //   onpeut vérifier : 
      //   R si on trouve le haut du sceptre sous le petit symbole haut gauche
        // chapeau du Valet. bleu ou rouge selon le jeu de cartes
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR / 2;
        r.x = maconf.largeurcarte/3; // gauche du chapeau
        r.width = maconf.largeurcarte/3; // largeur du chapeau
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,255,0));
        lig = imacarte(r); m = cv::mean(lig);
        // tester chapeau rouge 
        if (m[1] < mbl[1] - 80){
          if (m[2] - m[1] > 30) {
            valcarte = 11; // on a trouvé le chapeau du Valet (rouge)
            return 11;
          }
        }
        // tester chapeau bleu
        if (m[0] + m[1] - 2*m[2] > 30) {
            valcarte = 11; // on a trouvé le chapeau du Valet (bleu ou turquoise) 
            return 11;
        }
        
        // Roi ou Dame ? tester la  droite de la couronne
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR / 2;
        r.x = maconf.largeurcarte*7/12; // partie droite de la couronne
        r.width = maconf.largeurcarte/12; // largeur de ...
        lig = imacarte(r); m = cv::mean(lig);
        if (maconf.waitoption) {
          tracerRectangle(r, imacarte, "carte", cv::Scalar(255,0,0));
          if (maconf.waitoption) cv::waitKey(0);
        }
        if (m[0] > mbl[0] - 20){ // zone claire, pas de couronne
          valcarte = 12; // 
          return 12; // c'est donc la Dame
        }
        return 13; // ni V ni D donc Roi
      }
      if (numcol == 0) { // Pique
      // couleur P  Pique :
      //   V si le chapeau est rouge
      //   D si la gauche du chapeau est claire
      //   R  sinon 
      // ...VV....
      // .....RR...
      //........DD...
      //
r.x = maconf.largeurcarte* 5/16; // position du béret du valet
        r.width = maconf.largeurcarte * 3/16;
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR / 2;
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,255,0));
        lig = imacarte(r); m = cv::mean(lig);
        //if (m[0] < mbl[0] - 50){
        //  valcarte = 11; // on a trouvé le chapeau du Valet (rouge)
        //  return 11;
        //}
        // compter les pixels rouges
        int nbpixtot(0), nbpixrouge(0);
        for (int y = cadresup+1; y < cadresup + maconf.tailleVDR; y++)
          for (int x = r.x; x < maconf.largeurcarte / 2; x++) {
            nbpixtot++;
            cv::Scalar pix = imacarte.at<cv::Vec3b>(y,x);
            if (pix[2] >pix[0] + 20 && pix[2] > pix[1] + 20) nbpixrouge++;
          }
        if (100*nbpixrouge / nbpixtot > 30 ) 
          return 11;

        // Dame ou Roi ? tester la gauche de la couronne
        r.x = maconf.largeurcarte *5/12;
        r.width = maconf.largeurcarte / 12;
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR / 2;
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] > mbl[0] - 30){
          valcarte = 12; // zone claire : pas de couronne du Roi
          return 12;
        } else 
          return 13; // ni V ni D donc Roi
      }
      if (numcol == 0 || numcol == 3) { // Pique ou Trefle (jamais atteint)
        r.y = cadresup + maconf.tailleVDR + maconf.deltachsymb + maconf.taillesymbole + 2;
        r.height = maconf.tailleVDR;
        r.x = std::max(0,xGS - maconf.largeursymbole);
        r.width = maconf.largeursymbole;
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(255,0,0));
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] < mbl[0] - 40){
            valcarte = 13; // on a trouvé le sceptre du roi
          return 13;
        }
      }

      if (numcol == 1) { // Coeur
      // couleur Coeur :
      //   R si la partie gauche de la couronne est jaune (plus de Rouge et vert que bleu )
      //   D si c'est clair à gauche de l'épaule
      //   V sinon (le chapeu est bleu, mais comme pour la Dame)
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR;
        r.x = maconf.largeurcarte * 14/40;
        r.width = maconf.largeurcarte / 8;
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(255,0,0));
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] < (m[1] + m[2])/ 2  - 15){ // jaune ?
            valcarte = 13; // on a trouvé la couronne du roi
          return 13;
        }
        r.x = xGS + maconf.largeurgrosRDV / 2;
        r.width = maconf.largeurgrosRDV / 3;
        r.y = yGS + 1;
        r.height = maconf.taillegrosRDV / 2;
        if (r.y + r.height < imacarte.rows) {
          if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(255,0,0));
          lig = imacarte(r); m = cv::mean(lig);
          if (m[0] <  mbl[0] - 100){ // foncé à gauche de l'épaule : Valet
            return 11;
          }
          return 12; // ni R ni V  donc D
        } else { // probablement une carte du mort, écrètée par la carte qui la recouvre
          // tester la gauche du chapeau : bleu pour un valet
          // à droite il y a le reste du chapeau du valet ou la coiffe bleue de la Dame
          // position : de 25/80 à 28 ∕ 80 
          r.x = maconf.largeurcarte * 25 / 80;
          r.width = maconf.largeurcarte * 3 / 80;
          r.y = cadresup + 1;
          lig = imacarte(r); m = cv::mean(lig);
          if (m[0] - m[2] > 30)
             return 11; // béret bleu : coeur
          else return 12; // ni Roi ni Valet
        }
      }
      if (numcol == 2){ // carreau
      // couleur K carreau :
      //   V si le chapeau est rouge
      //   R si la couronne est plus à droite (que la Dame)
      //   D sinon (ou si elle tient une fleur rouge en haut à droite)
      //      ou si la zone centrale à coté du cadre gauche est claire (sinon R ou V)
        r.x = maconf.largeurcarte * 14/40; // bord gauche du chapeau du Valet
        r.width = maconf.largeurcarte * 8/40; // largeur du chapeau
        r.y = cadresup + 1;
        r.height = maconf.tailleVDR;
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(255,0,0));
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] < m[2] - 40){ // chapeau rouge : Valet
          return 11;
        }
        //r.x = cadregauche; r.width = maconf.largeurcarte /20;
        // à droite de la couronne de la reine 
        r.height = maconf.tailleVDR / 2;
        r.x = maconf.largeurcarte* 34/64; r.width = maconf.largeurcarte * 6/64;
        if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,0,0));
        lig = imacarte(r); m = cv::mean(lig);
        if (m[0] >  mbl[0] - 60   &&  m[1] >  mbl[1] - 60  ){ // clair : pas de couronne du Roi
          valcarte = 12;
          return 12;
        } else
          return 13;
      }

      // tout ce qui suit a été testé mais n'est plus utilisé 

      // appel à l' OCR
      // tester le caractère en haut à gauche
      // puis, si on ne trouve pas, le caractère en haut à droite
      double confiance(0), angle(0);
      cv::Mat ima_car;
      cv::Mat ima_carW;

      cv::Mat ima_CARV;
      int largeurcar; // largeur du caractère gauche. à utiliser pour le caractère à droite
      std::string outOCR1 = "";  // résultat OCR du caractère à droite
      for (int i = 0; i < 2; i++){
          if (i == 0){ // analyse du caractère à gauche
              r.y = cadresup + 1;
              r.x = xcaracterebas; // cadregauche + 1;
              r.width = maconf.largeurVDR + maconf.deltacadre +1; // sur le GS ou  juste avant le GS à droite
              r.height = maconf.tailleVDR + maconf.deltacadrehaut + 1; // le petit symbole est juste au dessous
              // si la petite colonne en r.x + r.width - 1 est foncée (gros symbole) r.width--;
              rr = r;
              rr.x += r.width - 1; rr.width = 1;
              lig = imacarte(rr); m = cv::mean(lig);
              if (m[0] < mbl[0] - 30){
                  r.width--;
                  rr.x--; lig = imacarte(rr); m = cv::mean(lig);
                  if (m[0] < mbl[0] - 30) r.width--;
              }
              largeurcar = r.width;                
              r.height = maconf.tailleVDR + 1; 
              /************************
              // vérifier que la petite ligne juste au dessus du caractère est blanche
              // sinon (line foncée) considérer la ligne au dessus. cadre ?
              //
              //   1       2         3         4
              // ______  _______             _______
              //  *                ______    _______
              //  ***    ***       *         ***          <--- position r.y
              //  * **   * **      ***       * **
              //  * **   * **      * **      * **
              //  ***    ***       * **      ***
              //                   ***                    <--- ??
              //
              // tester le cas 3
              int rsavey = r.y;
              r.height = 1;
              r.y += maconf.tailleVDR; // sous le caractère
              lig = imacarte(r); m = cv::mean(lig);
              if (m[0] < mbl[0] - 30){ // ligne foncée (cas 3)
                  r.y = rsavey + 1;
              } else { // cas 1 ou 2
                  r.y = rsavey - 1;
                  lig = imacarte(r); m = cv::mean(lig);
                  if (m[0] > mbl[0] - 30){ // ligne blanche (cas 2)
                      r.y++;
                  } else { // cas 1 : ne rien changer si la ligne au dessus est le cadre
                      r.y--; lig = imacarte(r); m = cv::mean(lig);
                      if (m[0]  < mbl[0] - 30 ) { // ligne foncée : c'est le cadre (cas 1)
                          r.y +=2; // rien ne change
                      } else { // ligne blanche : juste au dessus du cadre
                          r.y += 3; // une ligne plus bas 
                      }
                  }
              }
              **************************/
          } else if(i == 1) {  // en prévision du test des caractères en bas
              // analyse du caractère à droite
              r.x = xcaractere - 1; r.width = maconf.largeurVDR + 1;
              r.y = cadresup + 1; r.height = maconf.tailleVDR + 1;
              if (r.width > imacarte.cols - r.x) r.width = imacarte.cols - r.x;

              /***********************
              // ceci n'est plus utile, la détection de la gauche du caractère est fiable
              // le bord droit est éventuellement absent, le cadre également
              // si la colonne de droite est foncée, réduire la largeur
              if (r.width > largeurcar + 1) r.width = largeurcar +1; // +1 à cause du blanc à gauche
              rr = r;
              rr.x += r.width - 1; rr.width = 1;  lig = imacarte(rr); m = cv::mean(lig);
              if (m[0] < mbl[0] - 50) r.width--;

              // si la dernière colonne de droite noire est précédée d'une colonne blanche,
              // c'est le trait de cadre. la supprimer
              rr = r;
              // ignorer les colonnes blanches
              rr.x += r.width - 1; rr.width = 1;  lig = imacarte(rr); m = cv::mean(lig);
              while (m[0] > mbl[0] - 30) {
                  rr.x--;
                  lig = imacarte(rr); m = cv::mean(lig);
              }
              if (m[0] < 2*mbl[0]/3) { // colonne bien foncée
                  rr.x--; lig = imacarte(rr); m = cv::mean(lig);
                  if (m[0] > mbl[0] - 30) { // colonnes blanche et noires à droite
                      r.width = rr.x - r.x; // éliminer la ligne noire (trait de cadre)
                  }
              }
              ********************/
              if (r.width > imacarte.cols - r.x) r.width = imacarte.cols - r.x;
          }
          if (printoption > 1 && !threadoption) tracerRectangle(r,carte, "carte",cv::Scalar(0,128, 0));
          ima_car = imacarte(r);
          cv::cvtColor(ima_car, ima_carW, cv::COLOR_BGR2GRAY);
          cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
          // ajouter une bordure blanche
          {
              cv::Mat image_bordee;
              int tb = 2;
              
              cv::copyMakeBorder(ima_carW, image_bordee, tb, tb, tb, tb, cv::BORDER_CONSTANT, cv::Scalar(255));
              ima_carW = image_bordee;
          }
          ima_CARV = ima_carW.clone();
          if (printoption > 1 && !threadoption){
              afficherImage("V1c", ima_car);
              afficherImage("V1", ima_CARV);
          }
          if (printoption){
              if (waitoption > 2)
                  cv::waitKey(0);
              // else cv::waitKey(1);
          }
          if (i == 1) { // caractère haut gauche. on a déjà localisé le caractère de droite
              //////////////////// analyse géométrique ////////////////
              // on vient de tester le caractère à droite. image seuillée : "V1" ima_CARV
              // 
              //
              //       Valet     Valet     Valet        Roi        Dame      Dame
              //  ____## ##    ######     #######     #######     #######   #######
              //  ____## ##    ##  ##     #######     #######     #######   #######
              //  ....         ##  ##     #######     ##   ##     ##   ##   ##   ##
              //  ....            #          #        ##   ##     ##   ##   ##
              //                                      #######     [#####]   ##       <=== xg3 xd3
              //                                      ##   ##     _______   ##   ##
              int xd(-1), xg, xd1(-1), xg1;
              int xg3(0), xd3(0); // pixels noirs en début et fin de la troisième ligne
              int tl[4] = {0,0,0,0}, nbl = 0; // types de lignes : 0=noire, 1= noir blanc noir
              if (output == "" && !threadoption){
                if (printoption > 1) {
                  afficherImage("V1c", ima_car);
                  afficherImage("V1", ima_CARV);
                  //cv::waitKey(0);
                }

                  //        (W) : 1 ou plusieurs caractères W
                  //        [X] : 0 ou plusieurs caractères X
                  //         W  : un caractère W 
                  bool nontrouve = true;
                  bool ligne1noire = false;
                  bool ligneBNB = false;
                  int y1 = -1;
                  int x, y;
                  int ynoirfin = -1;
                  int pix;
                  // rechercher la première ligne non blanche
                  for (y = 0; nontrouve && y < ima_CARV.rows; y++){
                      for (x = 0; x < ima_CARV.cols; x++){
                          int pix = ima_CARV.at<uchar>(y,x);
                          if (pix < 128) { // premier pixel noir de la ligne : (B)N
                              y1 = y;
                              break;
                          }
                      }
                      if(y1 >= 0) break;
                  }
                  if (y1 < 0) return 0; // image vide !


                  for (y = y1; y < ima_CARV.cols; y++) {
                      // déterminer le format de la ligne :
                      // 0 : ligne noire : [B] (N) [B]
                      // 1 : ligne [B] (N) (B) (N)
                      for (x = 0; x < ima_CARV.cols; x++){
                          pix = ima_CARV.at<uchar>(y,x);
                          if (pix < 128) break; // premier pixel noir de la ligne : [B]N
                      }
                      if (pix >= 128) continue; // ligne blanche ignorée
                      
                    // déterminer le dernier pixel noir de la ligne (il existe)
                      xd1 = xd; xg1 = xg;  // mémoriser 1er et dernier pixel de la ligne précédente
                      for (int xx=ima_CARV.cols - 1; xx > 0; xx--) {
                        pix = ima_CARV.at<uchar>(y,xx);
                        if (pix < 128){ // dernier pixel noir de la ligne
                          xd = xx;
                          break;
                        }
                      }

                      xg = x; // position du premier pixel noir de la ligne
                      x++; // rechercher un pixel blanc
                      for (; x < ima_CARV.cols; x++){
                          pix = ima_CARV.at<uchar>(y,x);
                          if (pix > 128) break;
                      }
                      if (pix > 128) { // [B](N)B
                          x++; // rechercher un pixel noir
                          for (; x < ima_CARV.cols; x++){
                              pix = ima_CARV.at<uchar>(y,x);
                              if (pix < 128) break; // pixel noir
                          }
                          if (pix < 128) { // [B](N)(B)N
                              if (nbl == 0 || tl[nbl - 1] != 1) { // format différent de celui de la ligne précédente
                                  tl[nbl] = 1;
                                  nbl++;
                              }
                              continue; // ligne suivante
                          }
                          else { // ligne [B](N)(B)
                              if (nbl == 0 || tl[nbl - 1] != 0) {
                                  tl[nbl] = 0;
                                  nbl++;
                                  if (nbl == 3) {xd3 = xd; xg3 = xg;} // pour distinguer entre R et D
                              }
                          }
                      } else { // ligne [B](N)
                          xd = x; // en fait = ima_CARV.cols - 1
                          if (nbl == 0 || tl[nbl - 1] != 0) {
                              tl[nbl] = 0;
                              nbl++;
                              if (nbl == 3) {xd3 = xd; xg3 = xg;}
                          }
                          continue; // ligne suivante
                      }
                  }
                  if (printoption > 1)
                    std::cout<<nbl<<" " <<tl[0]<<"," <<tl[1]<<"," <<tl[2]<<"," <<tl[3]<<std::endl;
                  // ATTENTION : éviter l'indétermination
                  // Roi si 0 1 0 1   ou 0 1
                  // Dame   0 1 0   et bas large  ou 1 0 1 0
                  // Dame   0 1
                  // Dame si 0 1 0 1   et 3ème trait large
                  // Valet  1 0
                  // Valet  0 1 0   et bas pointu (exceptionnel avec un morceau du cadre)
                  // Valet  0       et bas pointu
                  if ( nbl >= 4 && tl[0] == 0) {
                    output = "R";
                    if (xd3 - xg3 +1 < maconf.largeurVDR / 2) output = "D";
                  }
                  else if (nbl >= 4 && tl[0] == 1 && tl[1] == 0 && tl[2] == 1 && tl[3] == 0) output = "D";
                  else if (nbl == 2 && tl[0] == 1 && tl[1] == 0) output = "V";
                  else if (nbl == 2 && tl[0] == 0 && tl[1] == 1) output = "D";
                  else if (nbl == 2 && tl[0] == 0 && tl[1] == 1) output = "R";
                  else if (nbl == 3 && tl[0] == 1 && tl[1] == 0 && tl[2] == 1){ // R
                    output = "R"; 
                  }
                  else if (nbl == 3 && tl[0] == 0 && tl[1] == 1 && tl[2] == 0){ // D ou V
                    output = "D";

                    if (xd - xg + 1 <= maconf.largeurVDR / 2) {
                      output = "V"; // 
                    
                      // TODO : tester aussi la ligne précédente
                      //    Dame          Valet
                      //  ********       ********
                      //  **    **        **  **
                      //  **    **         *  *
                      //  ........       ........
                      //  ********          ***     <--- tester le pixel à gauche 
                      //      *              *      <--- pixel noir parasite pour une Dame
                      if (xd1 >= 0)
                        if ( xd1 - xg1 + 1 <= maconf.largeurVDR / 2 && xd1 - xg1 >= xd - xg) output = "V";
                        else output = "D";
                    }
                  }
                  else if (nbl == 1 && tl[0] == 0){
                      if (xd - xg + 1 <= maconf.largeurVDR / 3) output = "V";
                  }
                  if (output == "") {
                    output = "R";  // jamais indéterminé !!!!
                  }
              }
              
          }
          if (i == 0) { // analyse du caractère gauche
              // différer la recherche OCR
              ima_CARG = ima_carW.clone();
              continue;
          }
          //  compléter l'analyse géométrique du caractère à droite
          // par la recherche par OCR :
          if (output == "") {
            std::string outOCR;
            if (maconf.tesOCR >= 1) { 
              outOCR = tesOCR(ima_carW, true, &confiance, &angle);
              if (outOCR == "" && maconf.tesOCR == 1) {
                outOCR = execOCR("SERVEUR", ima_carW, &confiance, &angle);
                if (outOCR == "0") outOCR = "D";
                if (outOCR == "1") outOCR = "V"; // parfois c'est D
              }
            }
            else  outOCR = execOCR("SERVEUR", ima_carW, &confiance, &angle);
            if (outOCR == "?") outOCR = "";
            if (outOCR != output){
              // résultat OCR différent de l'analyse géométrique
              std::cout<<"!!! résultat OCR "<<outOCR << " géométrie " << output << std::endl;
              if (output == "") outOCR1 = outOCR;
            }
          }
          if (printoption  > 1 && output.size() > 0)
              std::cout << "V1 ==> " << output << std::endl;
          if (output == "V") return 11;
          else if (output == "D") return 12;
          else if (output == "R") return 13;
          if (printoption > 1 && !threadoption) {
              afficherImage("V1c", ima_car);
              afficherImage("V1", ima_CARV);
          }
          // cv::waitKey(0);
          if (output != "") break;
      } // caractère à gauche puis à droite

      if (output == "") { // analyse géométrique infructueuse
          // analyse OCR du caractère gauche
          std::string outOCR;
          if (maconf.tesOCR >= 1) outOCR = tesOCR(ima_CARG, true, &confiance, &angle);
          else                    outOCR = execOCR("SERVEUR", ima_CARG, &confiance, &angle);
          if (outOCR == "?") outOCR = "";
          if (outOCR == "") {
            if (outOCR1 != "") output = outOCR1;
          } else output = outOCR;
      }
      if (printoption > 1 && output.size() > 0)
          std::cout << "V1 " << output << " confiance " << confiance << " angle " << angle << std::endl;
      if (output == "V") return 11;
      else if (output == "D") return 12;
      else if (output == "R") return 13;
      return 0; // recherche infructueuse

  }

    //////////////////////////////////////////////////////////////////
    ///// traitement d'une petite carte   ///////////////////////////
    /////////////////////////////////////////////////////////////////
    // ce n'est pas un habillé ( R D V)
  int decoderCarte1a10 (cv::Mat imacarte, config& maconf, int& numcol){
    cv::Rect r, rr;
    cv::Mat lig, GS;
    cv::Scalar m, mbl, mbl2;
    int printoption = maconf.printoption;
    int waitoption = maconf.waitoption;
    cv::Mat imac = imacarte.clone();

    // si la carte est incomlète, décoder en utilisant l'OCR sur le chiffre de gauche
    //  puis sur le chiffre de droite si on ne trouve pas
    if (imacarte.rows < maconf.hauteurcarte / 2) { // carte très incomplète
      // déterminer la valeur du blanc : en haut de la carte à droite du chiffre, une petite ligne
      // déterminer la présence d'un gros symbole en haut à gauche
      rr.x = maconf.largeurchiffre + maconf.deltachiffre + 2;
      rr.width = maconf.largeurchiffre;
      rr.y = 2; rr.height=1;
      lig = imacarte(rr); mbl = cv::mean(lig);
      bool estGS = false;
      rr.x = maconf.deltagros; rr.width = maconf.largeurgros;
      rr.y = std::min(imacarte.rows -1, maconf.deltagroshaut); rr.height = maconf.taillegros;
      if (rr.height > imacarte.rows - rr.y) rr.height = imacarte.rows - rr.y;
      lig = imacarte(rr); m=cv::mean(lig);
      if (mbl[0] - m[0] > 10) estGS = true;
      
      for (int i = 0; i<2; i++){ // chiffre de gauche puis de droite
        // localiser le chiffre, l'extraire
        // chercher le haut du chiffre : commencer au milieu estimé du chiffre et remonter jusqu'au blanc
        if (i > 0) {
          r.x = imacarte.cols - 1 - r.width;
        } else {
          r.x = maconf.deltachiffre - 1; 
        }
        r.y = maconf.deltahaut + maconf.taillechiffre / 2;
        r.height = 1; 
        r.width = maconf.largeurchiffre + 3;
        while (r.y > 0 && r.y + r.height <= imacarte.rows){
          lig = imacarte(r); m = cv::mean(lig);
          if(m[0] > mbl[0] - 10) {
            r.y++; break;
          }
          r.y--;
        }
        // on a trouvé le haut du caractère
        r.width = maconf.largeurchiffre + 3;
        //r.y = maconf.deltahaut - 1;
        r.height = maconf.taillechiffre +1;
        if (r.height > imacarte.rows - r.y) r.height = imacarte.rows - r.y;
        if (printoption > 1) tracerRectangle(r, imac, "carte", cv::Scalar(255,0,0)); //cv::waitKey(0);
        cv::Mat ima_carW;
        cv::Mat ima_car = imacarte(r);
        cv::cvtColor(ima_car, ima_carW, cv::COLOR_BGR2GRAY);
        cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche
        cv::Mat image_bordee;
        int tb = 2;
        if (maconf.taillechiffre <= 6)
            tb = 1;
        cv::copyMakeBorder(ima_carW, image_bordee, tb, tb, tb, tb, cv::BORDER_CONSTANT, cv::Scalar(255));
        ima_carW = image_bordee;
        if(printoption > 1) {
          afficherImage("CAR", ima_carW); 
          afficherImage("CAR3", ima_car);
        }
        std::string output;
        std::string nomOCR = "SERVEUR";
        if (maconf.tesOCR != 0)
            nomOCR = "tesOCR";
        double confiance, angle;
        if (maconf.tesOCR >= 1){
            output = tesOCR(ima_carW, false, &confiance, &angle);
                if (output == "" && maconf.tesOCR == 1) {
                  output = execOCR("SERVEUR", ima_carW, &confiance, &angle);
          }
        }
        else  output = execOCR(nomOCR, ima_carW, &confiance, &angle);
        if (printoption  > 1 && output.size() > 0)
            std::cout << " caractere : " << output << " confiance " << confiance << " angle " << angle << std::endl;
        if (output == "10") return 10;
        if (output.size() == 1) {
          char c = output[0];
          if (c > '0' && c <= '9') {
            int val = c - '0';
            // valider selon la présence d'un gros symbole en haut à gauche
            if (estGS) {
              if (val  < 4) continue;
            } else {
              if (val  >= 4) continue;
            }
            return val;
          }
        }
      } // chiffre à gauche puis à droite
      if(maconf.waitoption) cv::waitKey(0);
      return -1; // décodage impossible
    } // carte très incomplète

    int exclure = maconf. deltachiffre + maconf.taillechiffre + maconf.taillesymbole
     + maconf.deltachsymb + 2;
    // recalculer la valeur du blanc
    if (maconf.deltagros > 5) {
        r.x = 2;
        r.width = maconf.deltagros - 4;
    } else {
        r.x = 1;
        r.width = maconf.deltagros - 2;
    }
    if (exclure < imacarte.rows) {
      r.y = exclure; // sous le chiffrec et le symbole
      r.height = std::min(imacarte.rows - r.y, maconf.hauteurcarte - 2*exclure);
      lig = imacarte(r);
      mbl2 = cv::mean(lig); // valeur du blanc
      if (mbl2[0] > mbl[0]) mbl = mbl2;
      if (printoption > 1 && !threadoption){tracerRectangle(r, imac, "carte",cv::Scalar(0,255,0));}
    }


    int retcode = 0;
    // Gros symbole en haut à gauche ?
    r.x = maconf.deltagros;
    r.width = maconf.largeurgros;
    r.y = maconf.deltagroshaut;
    r.height = std::min(imacarte.rows - r.y, maconf.taillegros);
    lig = imacarte(r); m = mean(lig);
    GS = lig;
    cv::Scalar  mGShg = m;
    if (m[0] > 9*mbl[0]/10) { // pas de GS en haut à gauche --> 1 2 ou 3
        // GS haut centre ?
        r.x = (maconf.largeurcarte - maconf.largeurgros)/ 2;
        r.y = maconf.deltagroshaut;
        lig = imacarte(r); m = mean(lig); // GS haut centre
        if (printoption > 2 ) {
            std::cout<<"m haut centre "<<m[0]<<" , blanc "<<mbl[0]<<std::endl;
            if (!threadoption) 
               tracerRectangle(r, imac, "carte", cv::Scalar(255,0,0)); // GS haut centre
        }
        GS = lig;
        r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;  // GS central 
        if (printoption > 1 && !threadoption) 
          tracerRectangle(r, imac, "carte", cv::Scalar(0,255,0)); // GS central
        lig = imacarte(r); // GS central
        if (m[0] > 9*mbl[0]/10) { // pas de GS haut centre --> 1
            m = cv::mean(lig); // GS central
            if (m[0] > 9*mbl[0]/10) { // pas de GS central
                retcode =  0; // aucun GS haut gauche, haut centre, central --> erreur
            } else { // c'est un As
              //        pour un As, le gros symbole peut être inversé
              //        tester si la partie supérieure est plus claire que la partie invérieure
              //        faire le calcul de couleur --> rouge ou noir
              //        si c'est rouge : inverser si le bas est plus rouge que le haut
              //              sans effet si carreau
              //        si c'est noir ????
              cv::Scalar m1, m2;
              int box[4];
              r.y -=4; r.height += 8;   // étendre largement
              r.x -=4; r.width += 8;
              GS = imacarte(r).clone();
              calculerBox(GS, maconf.taillegros, maconf.largeurgros, m1, box, m2, maconf);
              r.x += box[0]; r.y += box[2]; r.width = maconf.largeurgros; r.height = maconf.taillegros;
              GS = imacarte(r).clone();
              numcol = calculerCouleur(GS, false, maconf, mbl);
              if (numcol == 1 || numcol == 2){
                //   coeur         coeur       carreau
                //  ???? ****        *            *
                //  ????*****       ***          ***
                //   *******      *******      *******
                //     ***       ????*****       ***
                //      *        ???? ****        *
                // comparer les zones ???? en haut et en bas du symbole
                // s'il y a une différence d'intensité bleue (et/ou verte) c'est du coeur
                rr= r;
                rr.width = maconf.largeurgros / 2 - 1;
                rr.height = maconf.taillegros / 3;
                lig = imacarte(rr); m1 = cv::mean(lig);
                rr.y = r.y + r.height - rr.height;
                lig = imacarte(rr); m2 = cv::mean(lig);
                if (std::abs(m1[0] - m2[0]) > 30) numcol = 1;
                else numcol = 2;
                return 1; // on a trouvé un As de Coeur ou de carreau

              } else { // couleur noire pique ou trefle
                // tester la zone juste au dessus de la queue du symbole
                //    ***        *****
                //   *****        ***             <--- zone à tester 1
                //  *******    *********
                // *********    *******
                //    ***        *****            <--- zone à tester 2
                //   *****        ***
                rr = r;
                rr.x += maconf.largeurgros / 3; rr.width /= 3;
                rr.y += (maconf.taillegros +4) / 8; rr.height = (maconf.taillegros +4) / 8;
                lig = imacarte(rr); m1 = cv::mean(lig); // zone 1
                rr.y = r.y + maconf.taillegros - maconf.taillegros /4; 
                lig = imacarte(rr); m2 = cv::mean(lig);
                if (m1[0] > m2[0]) {
                  // inverser
                  cv::rotate(GS, GS,cv::ROTATE_180);
                  numcol = calculerCouleur(GS, false, maconf, mbl);
                }  
              }
                //r.y -=2; r.height += 4;
                //if (printoption && !threadoption) tracerRectangle(r, imac, "carte", cv::Scalar(255,0,0));
                // GS = imacarte(r).clone(); // GS central élargi
                retcode =  1; // on a trouvé un As et sa couleur 
                return 1; // on a trouvé un As de Pique ou Trefle
            }
            
        } else { // il y a un GS en haut au centre ( 2 ou 3 )
            int box[4];
            cv::Scalar m2;
            r.x -= 4;
            r.width += 8;
            r.y = maconf.deltagroshaut; // GS haut centre
            r.y -=4; r.height += 8;
            GS = imacarte(r).clone();
            calculerBox(GS, maconf.taillegros, maconf.largeurgros, m, box, m2, maconf);
            r.x += box[0];
            r.y += box[2];
            r.width = maconf.largeurgros;
            r.height = maconf.taillegros;
            GS = imacarte(r).clone();
            if (printoption > 1 && !threadoption) tracerRectangle(r, imac, "carte", cv::Scalar(255,0,0));
            m = cv::mean(lig); // GS central
            if (m[0] > 9*mbl[0]/10) { // pas de GS central --> 2
                retcode = 2;
            } else { // GS central présent --> 3
                retcode = 3;
            }
        }
    } else { // GS haut gauche présent (4 à 10)
        cv::Scalar mext;
        int Box[4];
        // extraire largement au dessus
        r.y = 1; // juste sous le bord de carte
        r.height = maconf.deltagroshaut + maconf.taillegros + 1;
        r.x -= 2; r.width += 2; // extraire  2 pixels à gauche et 0 à droite
        // extraire le rectangle de la taille du gros symbole optimale (la plus foncée)
        GS = imacarte(r).clone();
        calculerBox(GS, maconf.taillegros, maconf.largeurgros, m, Box, mext, maconf);
        r.x += Box[0]; r.width = maconf.largeurgros;
        r.y += Box[2]; r.height = maconf.taillegros;
        GS = imacarte(r).clone();
        if (printoption > 1 && !threadoption) tracerRectangle(r, imac, "carte", cv::Scalar(255,0,0));

        // GS haut gauche présent --> 4 à 10
        // considérer la zone centrale à gauche entre GS haut et bas
        // compter le nombre de GS hormis celui du haut ( 0 1 ou 2)
        // zone du GS 2 sur 4, limitée au haut du GS 3 sur 4
        int nbGSg=0; // nombre de GS dans la partie centrale de la colonne gauche
        r.x = maconf.deltagros;
        r.y = maconf.deltagroshaut + 4*maconf.taillegros/3;
        r.width = (maconf.largeurcarte - maconf.largeurgros)/2 - r.x;
        r.height = (maconf.hauteurcarte - maconf.taillegros)/2 - r.y;
        if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
        lig = imacarte(r); m = mean(lig);
        if (m[0] < 9*mbl[0]/10) nbGSg=2;  // GS symétrique en dessous (4 GS dans la colonne)
        else {
            r.y = (maconf.hauteurcarte - maconf.taillegros)/ 2;
            r.height = maconf.taillegros;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
            lig = imacarte(r); m = mean(lig);
            if (m[0] < 9*mbl[0]/10) nbGSg=1;  // GS central (3 GS dans la colonne)
        }
        if (nbGSg == 0){ // 4 ou 5-->
            // GS central ?
            r.x = (maconf.largeurcarte - maconf.largeurgros) / 2;
            r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;
            r.height = maconf.taillegros;
            r.width = maconf.largeurgros;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
            lig = imacarte(r);   m = cv::mean(lig);
            if (m[0] > 9*mbl[0]/10) { // pas de GS central --> 4
                retcode =4;
            }
            else retcode = 5;
        } else if (nbGSg == 1) { // un seul GS sur le bord gauche hormis haut et bas --> 6 ou 7 ou 8
            //__________ |________       |___________    |__________
            // H         | H             | H             | H
            //           |               |               |
            //    *      |    *          |     --> 7     |
            // *  --> 8  | *             | *             | *    --> 6
            //    *      |      --> 7    |    *          |
            //           |               |               | 
            //_B_________| B______       |_B__________   | B________
            int nbGSc = 0;
            // compter les GS dans la colonne centrale:
            //         0 --> 6   1 --> 7  2 --> 8   
            // GS central sup ou inf ?
            r.x = (maconf.largeurcarte - maconf.largeurgros) / 2;
            r.width = maconf.largeurgros;
            if (r.x < maconf.deltagros + maconf.largeurgros) { // le GS de gauche empiète sur cette colonne centrale
                r.width -= 2*(maconf.deltagros + maconf.largeurgros - r.x);
                r.x = maconf.deltagros + maconf.largeurgros;
            }
            r.height = maconf.taillegros;
            r.y = maconf.deltagroshaut + maconf.taillegros;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
            lig = imacarte(r); m = cv::mean(lig);
            if (m[0] < 95*mbl[0]/100) { // GS sup présent
                nbGSc++;
            }
            // GS central inf ?
            r.y = maconf.hauteurcarte - maconf.deltagroshaut - maconf.taillegros - r.height;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
            lig = imacarte(r); m = cv::mean(lig);
            if (m[0] < 95*mbl[0]/100) { // GS inf présent
                nbGSc++;
            }
            retcode = 6 + nbGSc;

        } else { // 4 GS à gauche--> 8 9 ou 10 ( 8 pour certains modèles de cartes))
            // compter les GS de la colonne centrale
            //    h1 h2 h3 
            // h1 et h3 --> 10
            // h1 ou h3 --> 9       trouvé pour certains jeux de cartes
            // h2 seul --> 9        trouvé pour d'autres jeux de cartes
            // limiter à gauche et à droite
            int nbGS = 0;
            r.width = maconf.largeurcarte - 2* (maconf.deltagros + maconf.largeurgros);
            r.x = (maconf.largeurcarte - r.width) / 2;
            r.y = maconf.deltagroshaut + maconf.taillegros / 2;
            r.height = maconf.taillegros;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
            lig = imacarte(r); m = cv::mean(lig); // GS haut
            if (m[0] < 9*mbl[0]/ 10) nbGS = 1;
            r.y = maconf.hauteurcarte - maconf.deltagroshaut - maconf.taillegros - r.height;
            if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
            lig = imacarte(r); m = cv::mean(lig); // GS bas
            if (m[0] < 9*mbl[0]/ 10) nbGS++;
            if (nbGS == 2) retcode = 10; // 8 sur les cotés et 2 au centre
            else if (nbGS == 1) retcode = 9;
            else {
                r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;
                if (printoption > 1 && !threadoption) tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
                lig = imacarte(r); m = cv::mean(lig); // GS central
                if (m[0] < 9*mbl[0]/ 10) nbGS++;
                if (nbGS == 1) retcode = 9; // 8 sur les cotés et 1 au centre ou en haut ou en bas
                else retcode = 8; // rien dans la colonne centrale
            }
        }
    }
    // calculer la valeur du blanc
    // considérer la colonne de gauche sous le chiffre et le symbole
    r.x = 2; // ignorer deux pixels
    r.width = std::max(1,maconf.deltagros - 3);
    r.y = maconf.deltahaut + maconf.taillechiffre + maconf.deltachsymb + maconf.taillesymbole + 2;
    r.height = 2*maconf.taillechiffre;
    if (r.height > imacarte.rows - r.y) r.height = imacarte.rows - r.y;
    lig = imacarte(r); mbl = cv::mean(lig);
    numcol = calculerCouleur(GS, false, maconf, mbl);

    if (retcode == 0) retcode = 2;   // jamais indéterminé !!!!!!
    return retcode;
}


void eclaircirfond(cv::Mat& image) {
      // éclaircir (en bleu) les pixels qui ont moins de rouge que de bleu
      for (int x = 0; x < image.cols; x++){
        for (int y=0; y < image.rows; y++){
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] - pixel[2] > -20) 
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 230);
        }
    }
}


void blanchircadre(cv::Mat& image, cv::Scalar moyblanc, int nb) {
    // blanchir les (nb) lignes en haut et en bas de l'image
    // et les (nb) colonnes de gauche et de droite

    // nouvelle méthode : trouver l'intensité bleue maximale
    // puis blanchir le cadre avec cette intensité
    cv::Scalar maxbgr, intbgr;
    int maxbleu = 0;
    for (int x=0; x < image.cols; x++) {
        for (int y=0; y < image.rows; y++){
            intbgr = image.at<cv::Vec3b>(y, x);
            if(intbgr[0] > maxbleu) {maxbleu = intbgr[0]; maxbgr = intbgr;}
        }
    }
    cv::Vec3b blanc = cv::Vec3b(maxbgr[0], maxbgr[1], maxbgr[2]);
    //cv::Vec3b blanc = cv::Vec3b(moyblanc[0], moyblanc[1], moyblanc[2]);
    int x = 0; int xx = image.cols - 1;
    for (int y=0; y < image.rows; y++){
        for (int i=0; i< nb;i++){
            image.at<cv::Vec3b>(y, x + i) = blanc;
            image.at<cv::Vec3b>(y, xx- i) = blanc;
        }
    }
    int y = 0; int yy = image.rows - 1;
    for (int x = 0; x < image.cols; x++){
        for (int i=0; i< nb; i++){
            image.at<cv::Vec3b>(y + i, x) = blanc;
            image.at<cv::Vec3b>(yy - i, x) = blanc;
        }
    }
}


void amplifyContrast(cv::Mat& image) {
  // déterminer le minimum et le maximum par couleur et global
  int min[3]={255,255,255};
  int max[3]= {0,0,0};
      for (int x = 0; x < image.cols; x++){
          for (int y=0; y < image.rows; y++){
              cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
              for (int i = 0; i<3; i++){
                  if (pixel[i] < min[i]) min[i]= pixel[i];
                  if (pixel[i] > max[i]) max[i]= pixel[i];
              }
          }
      }
  int ming, maxg;
  ming = min[0]; ming = std::min(ming, min[1]); ming = std::min(ming, min[2]); 
  maxg = max[0]; maxg = std::max(maxg, max[1]); maxg = std::max(maxg, max[2]); 
  if (maxg > ming){
      for (int x = 0; x < image.cols; x++){
          for (int y=0; y < image.rows; y++){
              cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
              pixel[0] = 255*(pixel[0]-ming)/ (maxg- ming);
              pixel[1] = 255*(pixel[1]-ming)/ (maxg- ming);
              pixel[2] = 255*(pixel[2]-ming)/ (maxg- ming);
              image.at<cv::Vec3b>(y, x) = pixel;
          }
      }
  }
}

// obtenir une courbe à partir d'un pixel d'une image des bords.
void followContour(const cv::Mat& edges, cv::Point2i start, cv::Point2i ref, std::vector<cv::Point2i>& contour, double tolerance) {
    cv::Point2i A(ref.x, ref.y); // origine de la droite à compléter
    cv::Point2i B(start.x, start.y); // dernier point de la ligne en cours de constitution
   // écarts des 8 voisins 
    int dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    int dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

    // TODO : optimiser en calculant la normale à la droite ref-start
    // 
    // start  : point de départ 
    // B      : dernier pixel du contour en cours de création
    // aller au dela de (start) pas dans la direction de A
    // M : nouveau pixel analysé, doit être à moins de 2 pixels de la droite ref-start
    // les voisins de B doivent etre dans la direction de A à B 

    // le point courant est B
    while (true) {
        double dmin = 123456; // grande valeur
        int imin = 0;
        cv::Point2i N;
        for (int i = 0; i < 8; ++i) {
            cv::Point2i M(B.x + dx[i], B.y + dy[i]);
            if (M.x < 0 || M.x >= edges.cols || M.y < 0 || M.y >= edges.rows) continue;  // pas dansdans l'image
            if (edges.at<uchar>(M) < 200) continue; // pixel noir (en fait 0)
            // sélectionner ce point (M) uniquement s'il est au dela de l'extrémité courante (B)
            // produit scalaire AB.BM > 0
            int ps = (B.x - A.x) * (M.x - B.x) + (B.y - A.y) * (M.y - B.y);
            if (ps <= 0) continue;
            // M doit être au plus à un pixel de la droite originale
            double d = calculerDistance(M, A, start);
            if (abs(d) < dmin) {
                imin = i;
                dmin = abs(d);
                N = M;
            }
        }
        if (dmin > 1.5)  break; // aucun pixel proche de la droite en cours de prolongement
        B = N; // nouvelle extrémité
        contour.push_back(N);
    }
    //      #
    //     #
    //   ##  point B et point N  la distance de N à AB est entre 1 et 2 (max 1.414?)      
    //  #
    // #    point A
}
