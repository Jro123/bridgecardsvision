#include <opencv2/opencv.hpp>
#include <vector>
#include <stack>
#include <cmath>
#include "config.h"


// tracer le rectangle r sur une copie de l'image et afficher la fenêtre dont le nom est s 
void tracerRectangle(cv::Rect r, cv::Mat copie, std::string s, cv::Scalar couleur) {
    if (r.width >= 1) cv::line(copie, cv::Point2i(r.x, r.y), cv::Point2i(r.x + r.width-1, r.y), couleur);
    if (r.height > 1){
        if (r.width > 1) cv::line(copie, cv::Point2i(r.x + r.width - 1, r.y + r.height-1), cv::Point2i(r.x, r.y + r.height-1), couleur);
        cv::line(copie, cv::Point2i(r.x, r.y), cv::Point2i(r.x, r.y + r.height-1), couleur);
        if (r.width> 1) cv::line(copie, cv::Point2i(r.x + r.width - 1, r.y + r.height-1), cv::Point2i(r.x + r.width - 1, r.y), couleur);
    }
    afficherImage(s, copie); cv::waitKey(1);
}

void afficherImage(std::string nom, cv::Mat image) {
#ifndef _WIN32
    //cv::namedWindow(nom, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(nom, cv::WINDOW_NORMAL);

    //cv::resizeWindow(nom, image.cols, image.rows);
    cv::imshow(nom, image);
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

void calculerBlanc(uncoin& moncoin, config& maconf) {
    cv::Mat lig;
    cv::Rect r;
    int db = 1; // d�calage de la zone de test du blanc par rapport au bord
    if (maconf.deltacadre > 5) db = 2;
    r.width = 2 * maconf.taillechiffre;
    r.height = maconf.deltacadre - 2 * db;

    if (moncoin.UU.x > moncoin.PP.x) { // � droite
        r.x = moncoin.PP.x + maconf.taillechiffre;
    }
    else {
        r.x = moncoin.PP.x - 3 * maconf.taillechiffre;
    }
    if (moncoin.U.y > moncoin.PP.y) { // au dessous
        r.y = moncoin.PP.y + db;
    }
    else {
        r.y = moncoin.PP.y - db - r.height;
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
    int *pBox, cv::Scalar& moyext, config& maconf){
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
        if (printoption) std::cout<<"BOX "<<pBox[0]<<","<<pBox[1]<<","<<pBox[2]<<","<<pBox[3]<<
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






void calculerOrientation(uncoin& moncoin, config& maconf) {
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
    cv::Mat lig;
    cv::Rect r;
    cv::Scalar moyblanc = moncoin.moyblanc;
    int bleulim = 9; // �cart de bleu entre le blanc et la zone test�e pour le petit symbole (*3 pour GS)
    bool estgrossymb = false;
    bool inverse(false);

    //TODO : petit symbole vertical? suivi d'une zone blanche de même taille
    //  sinon (zone du PS blanche ou zone suivante non blanche) PS horizontal
    // Gros symbole dans l'autre coté
    // ___________________
    //|            | 
    //| 9    GS    | 9  ps
    //| z1   GS    | GS
    //| z2   GS    | GS

    // U V UU VV mal recalculés si l'OCR n'a pas permis de déterminer l'orientation
    // dans ce cas , l'orientation a été fixée à ***inverse***
    cv::Mat z1, z2;
    cv::Scalar m1, m2;
    r.width = maconf.largeursymbole; r.height = maconf.taillesymbole;
    // petit symbole vertical (z1)
    if (moncoin.estunRDV){
        if (moncoin.UU.x < moncoin.QQ.x) r.x = moncoin.QQ.x - maconf.deltaVDR - r.width;
        else r.x = moncoin.QQ.x + maconf.deltaVDR;
        if (moncoin.U.y < moncoin.QQ.y) r.y = moncoin.QQ.y - maconf.deltaVDR
             - maconf.tailleVDR - maconf.deltachsymb - r.height;
        else r.y = moncoin.QQ.y + maconf.deltaVDR + maconf.tailleVDR + maconf.deltachsymb;

    } else {
        if(moncoin.caractere == 'X') r.width -= (maconf.largeursymbole / 3);
        if (moncoin.UU.x < moncoin.PP.x) r.x = moncoin.PP.x - maconf.deltasymbole - r.width;
        else r.x = moncoin.PP.x + maconf.deltasymbole;
        if (moncoin.U.y < moncoin.PP.y) r.y = moncoin.PP.y - maconf.deltahautsymbole - r.height;
        else r.y = moncoin.PP.y + maconf.deltahautsymbole;

    }
    cv::Mat imacop = moncoin.ima_coin.clone();
    if (printoption) cv::circle(imacop, moncoin.PP, 1, cv::Scalar(0,255, 0));
    if (printoption) tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,255)); // petit symbole vertical
    //TODO : ajouter 1 pixel sur les 4 cotés pour avoir la couleur extérieure
    z1 = moncoin.ima_coin(r).clone();
    cv:: Rect r1 = r;
    m1 = cv::mean(z1);  // zone du petit symbole vertical
    // amplifyContrast(z1);  // attention moyblanc pas amplifié
    if (moyblanc[0] - m1[0] > bleulim || moyblanc[1] - m1[1] > bleulim || moyblanc[2] - m1[2] > bleulim )
    {   // zone non blanche
        if (moncoin.estunRDV && moncoin.caractere == 'R') {
            if (moncoin.U.y > moncoin.PP.y) r.y += r.height;
            else r.y -= r.height/2;
            r.height /= 2;  // éviter le sceptre du roi (près du petit symbole)
        } else {
            if (moncoin.U.y > moncoin.PP.y) r.y += r.height;
            else r.y -= r.height;
        }
        
        if (printoption) tracerRectangle(r, imacop, "orient", cv::Scalar(255,0,0)); // sous symbole vertical
        z2 = moncoin.ima_coin(r).clone(); // zone sous le petit symbole vertical
        m2 = cv::mean(z2);
        r = r1;
        // gros symbole ?
        if (moncoin.estunRDV && (moyblanc[0] - m2[0]) > 2*bleulim ) // gros symbole présent sur le coté vertical
        {
            // nécéssairement un honneur (pas de gros symbole si près du bord pour les autres cartes)
            // carte COFALU : de plus, c'est un coeur, et la carte est horizontale
            inverse = true;
            estgrossymb = true;
            // le petit symbole est sur le bord horizontal
            // l'extraire pour obtenir la couleur 
        }
        else{ // petit symbole suivi de blanc ou morceau de GS suivi de blanc
            // analyser la zone du petit symbole horizontal
            r.width = maconf.taillesymbole; r.height = maconf.largeursymbole;
            if (moncoin.estunRDV) {
                if (moncoin.UU.x < moncoin.QQ.x) r.x = moncoin.QQ.x - maconf.deltaVDR 
                   - maconf.tailleVDR - maconf.deltachsymb - r.width;
                else r.x = moncoin.QQ.x + maconf.deltaVDR + maconf.tailleVDR + maconf.deltachsymb;
                if (moncoin.U.y < moncoin.QQ.y) r.y = moncoin.QQ.y - maconf.deltaVDR - r.height;
                else r.y = moncoin.QQ.y + maconf.deltaVDR;
            } else {
                /* if(moncoin.caractere == 'X') */ r.height -= (maconf.largeursymbole / 3);
                if (moncoin.UU.x < moncoin.PP.x) r.x = moncoin.PP.x - maconf.deltahautsymbole - r.width;
                else r.x = moncoin.PP.x + maconf.deltahautsymbole;
                if (moncoin.U.y < moncoin.PP.y) r.y = moncoin.PP.y - maconf.deltasymbole - r.height;
                else r.y = moncoin.PP.y + maconf.deltasymbole;

            }

            if (printoption) tracerRectangle(r, imacop, "orient", cv::Scalar(255,255,0)); // PS horizontal
            z2 = moncoin.ima_coin(r).clone();
            m2 = cv::mean(z2);
            if (moyblanc[0] - m2[0] < bleulim ){ // z2  blanche : z1 = petit symbole vertical
            // blanche -> ps vertical.   
            } else {
                //  autre -> indéterminé : cas possibles : honneur Pique Trefle ou carreau
                //  dans ce cas, utiliser la détection faite par l'OCR
                // selon l'orientation du coin, il y a un gros symbole à coté du caractère vertical
                // vérifier s'il est présent
                char car = moncoin.caractere;
                estgrossymb = false;
                if (car == 'V' || car == 'D' || car == 'R'){
                    // extraire le gros symbole éventuel
                    r.width = maconf.largeurgrosRDV; r.height = maconf.taillegrosRDV;
                    if(moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.deltagrosRDV;
                    else r.x = moncoin.QQ.x - maconf.deltagrosRDV - r.width;
                    if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.deltagroshautRDV;
                    else r.y = moncoin.QQ.y - maconf.deltagroshautRDV - r.height;
                    cv::Mat GS = moncoin.ima_coin(r).clone();
                    m2 = cv::mean(GS);
                    if (moyblanc[0] - m2[0] > bleulim) { // il y a un gros symbole
                        if (moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y) inverse = true;
                        if (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y) inverse = true;
                    } else { // il n'y a pas de GS
                        if (moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y) inverse = true;
                        if (moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y) inverse = true;
                    }
                    moncoin.inverse = inverse;
                }
                if (moncoin.inverse) { z1 = z2; r1 = r;}
            }
        }
    }
    else { // zone z1 blanche, le petit symbole est sur le coté horizontal
        inverse = true;
    }
    if(inverse) { // extraire le petit symbole pour déterminer la couleur
        r.x = moncoin.UU.x; r.y = moncoin.UU.y;
        r.width = moncoin.VV.x - moncoin.UU.x + 1;
        r.height = moncoin.VV.y -moncoin.UU.y + 1;
        if (printoption) tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,0));
        z1 = moncoin.ima_coin(r).clone();
        r1 = r;
    }
    moncoin.inverse = inverse;
//
// z1 = petit symbole à utiliser pour obtenir la couleur
//
{
    int Box[4];
    cv::Scalar moy, moyext;
    cv::Mat imaR;
    int ts, ls;
    if (inverse) {
        ls = maconf.taillesymbole;
        ts = maconf.largeursymbole;
    } else {
        ts = maconf.taillesymbole;
        ls = maconf.largeursymbole;
    }
    // pour déterminer la couleur, agrandir z1
    // TODO : !!!!!!!!!!!!!!!!!!!!!! r n'est plus z1 initial OK !!!!!
    r = r1;
    r.x--; r.y--; r.width += 2, r.height += 2;
    z1 = moncoin.ima_coin(r);
    amplifyContrast(z1);
    calculerBox(z1, ts, ls, moy, Box, moyext, maconf);
    if (ts >= z1.rows || ls >= z1.cols)
     moyext = moncoin.moyblanc;
    int ecartrelatif = 100.0*((double)moyext[2]/moyext[0]) / ((double)moy[2] / moy[0]);
    if ( ecartrelatif > 105 ){ // expérimental: rouge < 77   noir > 105
        moncoin.estRouge = false; if (printoption) std::cout << " Noir "<< ecartrelatif;
    } else if (ecartrelatif < 77) {
        moncoin.estRouge = true; if (printoption) std::cout << " Rouge "<<ecartrelatif;
    } else {
        // c'est rouge si l'écart rouge-bleu est significatif
        double ecart = (moy[2] - moy[0]) / (256-moy[0]);
        if (ecart > 0.1){
            moncoin.estRouge = true; if (printoption) std::cout << " Rouge! "<<ecartrelatif<<", "<<ecart;
        } else {
            moncoin.estRouge = false; if (printoption) std::cout << " Noir! "<<ecartrelatif<<", "<< ecart;
        }
    }
}
moncoin.inverse = inverse;
return;

// !!!!!!!!!!!!!!!! ce qui suit est désactivé
// TODO : à supprimer
#ifdef ACTIVER
    // position de la zone du quart de gros symbole
    if (moncoin.estunRDV) {
        r.width = r.height = maconf.largeurgrosRDV / 2;
        if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.largeurVDR + maconf.deltaVDR + 1;
        else r.x = moncoin.QQ.x - maconf.largeurVDR - maconf.deltaVDR - 1 - r.width;
        if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.largeurVDR + maconf.deltaVDR + 1;
        else r.y = moncoin.QQ.y - maconf.largeurVDR - maconf.deltaVDR - 1 - r.height;
    }
    else {
        r.width = r.height = maconf.largeurgros/2;
        if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.PP.x + maconf.largeurchiffre + 1 + maconf.deltachiffre;
        else r.x = moncoin.PP.x - maconf.largeurchiffre - 1 - maconf.deltachiffre - r.width;
        if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.largeurchiffre + 1 + maconf.deltachiffre;
        else r.y = moncoin.PP.y - maconf.largeurchiffre - 1 - maconf.deltachiffre - r.height;
    }
    cv::Mat copie = moncoin.ima_coin.clone();
    if (printoption) tracerRectangle(r, copie, "Extrait", cv::Scalar(255, 255, 0));
    cv::Mat zone = moncoin.ima_coin(r).clone();
    cv::resize(zone, zone, cv::Size(), 8.0, 8.0);
    // amplifyContrast(zone);   // !!!!! ne surtout pas activer 
    cv::Scalar moy = cv::mean(zone);
    // s'il est pr�sent, le symbole peut �tre rouge ou noir.
    // dans les deux cas, l'intensit� bleue (et verte) est inf�rieure � sa valeur dans une zone blanche
    if (moyblanc[0] - moy[0] > 2*bleulim) estgrossymb = true;
    // d�terminer la couleur Rouge ou noir :
    // rouge s'il y a significativement plus de rouge que de bleu
    // sinon noir s'il y a moins de rouge que dans la zone blanche de r�f�rence
    int dbleu(0); // �cart en bleu entre les deux zones du petit symbole
    int dRBref = moncoin.moyblanc[2] - moncoin.moyblanc[0];
    int dRB = moy[2] - moy[0]; // rouge - bleu : rouge si > m�me �cart sur le blanc
    int drouge = moyblanc[2] - moy[2];  // ecart rouge zone blanche - GS (> 0 ==> noir)

    // si on connait le caractère trouvé par OCR, on en déduit la présence d'un gros symbole
    if (moncoin.caractere == 'X' || (moncoin.caractere > '3' && moncoin.caractere <= '9')) {
        // c'est un chiffre accompagné d'un gros symbole
        estgrossymb = true;
        // extraire le gros symbole
        if (moncoin.inverse){
            r.width = maconf.taillegros;
            r.height = maconf.largeurgros;
            if(moncoin.UU.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.deltagroshaut;
            else r.x = moncoin.PP.x - maconf.deltagroshaut - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltagros;
            else r.y = moncoin.PP.y - maconf.deltagros - r.height;
        } else {
            r.height = maconf.taillegros;
            r.width = maconf.largeurgros;
            if(moncoin.UU.x > moncoin.PP.x ) r.x = moncoin.PP.x + maconf.deltagros;
            else r.x = moncoin.PP.x - maconf.deltagros - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.deltagroshaut;
            else r.y = moncoin.PP.y - maconf.deltagroshaut - r.height;
        }
        zone = moncoin.ima_coin(r).clone();
    } else if (estgrossymb){
        // si on a déterminé la présence du gros symbole par analyse du quart proche du coin
        // extraire le gros symbole complet,
        //  en fonction de l'orientation et en supposant que c'est un chiffre (10 cas / 13)
        // (sauf si on sait déjà que c'est un VDR)
        if (moncoin.estunRDV) {
            r.width = r.height = maconf.largeurgrosRDV;
            if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.QQ.x + maconf.largeurVDR + 1;
            else r.x = moncoin.QQ.x - maconf.largeurVDR - 1 - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.QQ.y + maconf.largeurVDR + 1;
            else r.y = moncoin.QQ.y - maconf.largeurVDR - 1 - r.height;
        }
        else {
            estgrossymb = false; // inexploitable, RDV ou chiffre ?
            r.width = r.height = maconf.largeurgros;
            if (moncoin.UU.x > moncoin.PP.x) r.x = moncoin.PP.x + maconf.largeurchiffre + 1 + maconf.deltachiffre;
            else r.x = moncoin.PP.x - maconf.largeurchiffre - 1 - maconf.deltachiffre - r.width;
            if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.PP.y + maconf.largeurchiffre + 1 + maconf.deltachiffre;
            else r.y = moncoin.PP.y - maconf.largeurchiffre - 1 - maconf.deltachiffre - r.height;
        }
        zone = moncoin.ima_coin(r).clone();
    }

    if (estgrossymb && moncoin.caractere == 'V') estgrossymb = false;
    if (estgrossymb) {
        int ts = r.height; int ls = r.width;
        r.x--; r.y--; r.width +=2; r.height += 2;
        zone = moncoin.ima_coin(r).clone();
        amplifyContrast(zone);
        int Box[4];
        cv::Mat imaR;
        cv::Scalar moyext; // moyennes hors du symbole
        calculerBox(zone, ts, ls, moy, Box, moyext);
        double ecartrelatif = (double)moy[2]/moy[0] / ((double) moyext[2] / moyext[0] );
        //calculerEncombrement(zone, moncoin.moyblanc, moy, Box, imaR, moyext);
        //if ((double)(moy[2]) / moy[0] > 1.5) {
        //    moncoin.estRouge = true; if (printoption) std::cout << moy << moyext << " GS Rouge ";
        //}
        //else 
        if (ecartrelatif > 1.15){  // valeur 1.15 à valider
            moncoin.estRouge = true; if (printoption) std::cout << moy << moyext<< " GS Rouge ";
        }
        else {
            moncoin.estRouge = false; if (printoption) std::cout << moy << moyext << " GS Noir ";
        }
    }
    // d�terminer l'orientation, selon l'orientation du coin et la pr�sence du gros symbole d'un RDV
    if (moncoin.estunRDV) {
        if (estgrossymb) {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y)
                || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y))
                inverse = true;
            else inverse = false;
        }
        else {
            if ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y)
                || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y))
                inverse = false;
            else inverse = true;
        }
    }
    // sinon on d�terminera l'orientation avec le petit symbole
    // d�terminer la couleur
    if (!estgrossymb || !moncoin.estunRDV) {
        // rechercher la position du petit symbole, sous le chiffre
        // deux zones (UV et UUVV) dont une est blanche et l'autre contient le symbole
        // on compare les intensit�s de bleu (ou de vert) qui varient aussi bien pour le symbole rouge que noir 
        cv::Mat z1, z2;
        cv::Scalar ect1, ect2, moy1, moy2;
        // limiter la partie loin du bord de carte, car il peut y avoir un gros symbole proche de la zone sans petit symbole
        // !!! inutile, on vient de tester qu'il n' y a pas de gros symbole ou c'est un chiffre et le gros symbole est loin du bord de carte
        // il peut y avoir un gros symbole d'une carte avec chiffre, proche d'une des deux zones
        // donc limiter la zone vers l'int�rieur du coin
        // zone 1 verticale
        r.width = moncoin.V.x - moncoin.U.x + 1;
        r.height = moncoin.V.y - moncoin.U.y + 1;
        r.x = moncoin.U.x;
        r.y = moncoin.U.y;
        //if (moncoin.UU.x > moncoin.PP.x) r.width -= 2; // devrait suffire
        //else { r.x += 2; r.width-=2; }
        z1 = moncoin.ima_coin(r).clone();
        amplifyContrast(z1);
        //cv::resize(z1, z1, cv::Size(), 16.0, 16.0);
        cv::meanStdDev(z1, moy1, ect1 );
        //moy1 = cv::mean(z1);
        // zone 2 horizontale
        r.width = moncoin.VV.x - moncoin.UU.x + 1;
        r.height = moncoin.VV.y - moncoin.UU.y + 1;
        r.x = moncoin.UU.x;
        r.y = moncoin.UU.y;
        //if (moncoin.U.y > moncoin.PP.y) r.height -=2; // devrait suffire
        //else { r.y +=2; r.height -=2; }
        z2 = moncoin.ima_coin(r).clone();
        amplifyContrast(z2);
        //cv::resize(z2, z2, cv::Size(), 16.0, 16.0);
        cv::meanStdDev(z2, moy2, ect2 );
        //moy2 = cv::mean(z2);
        if (ect2[0] > ect1[0]) if (!moncoin.estunRDV) inverse = true;
        //dbleu = moy1[0] - moy2[0];
        //if (dbleu > 0) if (!moncoin.estunRDV) inverse = true;

        // s'il n'y avait pas de gros symbole, d�terminer la couleur
        if (!estgrossymb) {
            // dans la zone du petit symbole, ne considérer que les pixels qui ne sont pas "blancs"
            // un pixel est considéré comme blanc si la différence d'intensité entre rouge et bleu 
            // est voisine de celle des pixels de la référence blanche ( écart à valider expérimentalement)
            // et si l'intensité globale est proche de celle du blanc
            // sinon, on prend en compte le pixel
            // on calcule la moyenne bleue et rouge des pixels pris en compte
            // rouge si la moyenne rouge est proche du rouge de la référence blanche
            if(inverse) z1 = z2;
            //amplifyContrast(z1);
            cv::imshow("Z1", z1); cv::waitKey(1);
            int Box[4];
            cv::Scalar moyext;
            cv::Mat imaR;
            int ts, ls;
            if (inverse) {
                ls = maconf.taillesymbole;
                ts = maconf.largeursymbole;
            } else {
                ts = maconf.taillesymbole;
                ls = maconf.largeursymbole;
            }
            calculerBox(z1, ts, ls, moy, Box, moyext);
            if(ts*ls == z1.rows*z1.cols) moyext = moncoin.moyblanc;
            //calculerEncombrement(z1,moncoin.moyblanc, moy, Box, imaR, moyext);
            //calculerMoyenneSymbole (z1, moncoin.moyblanc, moy);
            //cv::Scalar moyZ = cv::mean(z1);
            double ecartrelatif = ((double)moyext[2]/moyext[0]) / ((double)moy[2] / moy[0]);
            if ( ecartrelatif > 0.85 ){
            //if (moyext[2] - moy[2] > 20){ // symbole moins rouge que la zone encadrante
                moncoin.estRouge = false; if (printoption) std::cout << moy << moyext << " Noir ";
            } else {
                moncoin.estRouge = true; if (printoption) std::cout << moy << moyext<< " Rouge ";
            }
            /************** 
            if (moncoin.moyblanc[2] -  moy[2] <  30) {  // 30 expérimental
                moncoin.estRouge = true; if (printoption) std::cout << moy << "/"<< moncoin.moyblanc<< " Rouge ";
            } else {
                moncoin.estRouge = false; if (printoption) std::cout << moy << "/"<< moncoin.moyblanc << " Noir ";
            }
            */
        }
    }
    if (inverse) if (printoption) std::cout << " inverse "<<dbleu;
    else if (printoption) std::cout<<" est Droit "<<dbleu;

    if (moncoin.inverse == inverse || moncoin.caractere == ' ') if (printoption) std::cout << std::endl;
    else {
        if (printoption) std::cout << "!!! incoherence  calcul orientation"<<std::endl;
        // la recherche du gros symbole est discriminante, si il n'y est pas
        /****************** report� au retour
        if (moncoin.estunRDV) { 
            if (!estgrossymb) moncoin.inverse = inverse; 
        }
        else  if (!moncoin.inverse) moncoin.inverse = inverse;
        ********************/
    }
    moncoin.inverse = inverse;
    cv::waitKey(1);

    if (printoption) std::cout << std::endl;
    // on a obtenu l'orientation et la couleur
    // on peut obtenir la couleur de carte (P C K T) en analysant le symbole (gros ou petit)
#endif    
}


void eclaircirfond(cv::Mat& image) {
      // éclaircir (en bleu) les pixels qui ont moins de rouge que de bleu
      for (int x = 0; x < image.cols; x++){
        for (int y=0; y < image.rows; y++){
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] - pixel[2] > -10) 
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 230);
        }
    }
}


void blanchircadre(cv::Mat& image, cv::Scalar moyblanc, int nb) {
    // blanchir les (nb) lignes en haut et en bas de l'image
    // et les (nb) colonnes de gauche et de droite

    // nouvelle méthode : trouver l'intensité bleue maxnimale
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
    CV_Assert(image.type() == CV_8UC3);  // Assurez-vous que l'image est de type CV_8UC3

    // Convertir l'image en flottant pour le traitement
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // Calculer les valeurs min et max pour chaque canal
    double minVal[3], maxVal[3];
    double minV, maxV;
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);

    minV = 255; maxV = 0;
    for (int i = 0; i < 3; ++i) {
        cv::minMaxLoc(channels[i], &minVal[i], &maxVal[i]);
        if (minVal[i] < minV) minV = minVal[i];
        if (maxV < maxVal[i]) maxV = maxVal[i];
    }

    // Ajuster chaque canal pour que les valeurs aillent de 0 � 255
    for (int i = 0; i < 3; ++i) {
        channels[i] = 255 * (channels[i] - minV) / (maxV - minV);
    }

    // Fusionner les canaux ajust�s
    cv::merge(channels, floatImage);
    floatImage.convertTo(image, CV_8U);
}





/*******************************
// d�terminer si la courbe ABC est une ligne droite
bool isStraightLine(const cv::Point& a, const cv::Point& b, const cv::Point& c, double tolerance) {
    // produit scalaire AB.BC
    cv::Point ab = b - a;
    cv::Point bc = c - b;
    double dotProduct = ab.x * bc.x + ab.y * bc.y;
    double magnitudeAB = std::sqrt(ab.x * ab.x + ab.y * ab.y);
    double magnitudeBC = std::sqrt(bc.x * bc.x + bc.y * bc.y);
    double angleCosine = dotProduct / (magnitudeAB * magnitudeBC);
    //return  dotProduct >= 0 && std::abs(angleCosine - 1.0) < tolerance;
    //return  dotProduct > 0;
    return angleCosine > 1 - tolerance;
}
***********************************/

// obtenir une courbe à partir d'un pixel d'une image des bords.
void followContour(const cv::Mat& edges, cv::Point2i start, cv::Point2i ref, std::vector<cv::Point2i>& contour, double tolerance) {
    cv::Point2i A(ref.x, ref.y); // origine de la droite à compléter
    cv::Point2i B(start.x, start.y); // dernier point de la ligne en cours de constitution
   // écarts des 8 voisins 
    int dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    int dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

    // TODO : optimiser en calculant la normale à la droite ref-start
    // 
    // start  : point de d�part 
    // B      : dernier pixel du contour en cours de cr�ation
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
