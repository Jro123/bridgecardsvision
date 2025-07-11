#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <stack>
#include <cmath>
#include "config.h"

extern int threadoption;

// tracer le rectangle r sur une copie de l'image et afficher la fenêtre dont le nom est s 
void tracerRectangle(cv::Rect r, cv::Mat copie, std::string s, cv::Scalar couleur) {
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

void calculerBlanc(uncoin& moncoin, const config& maconf) {
    cv::Mat lig;
    cv::Rect r;
    int db = 1; // d�calage de la zone de test du blanc par rapport au bord
    if (maconf.deltacadre > 5) db = 2;
    r.width = 2 * maconf.taillechiffre;
    r.height = maconf.deltacadre - 2 * db;

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

// validation du caractère obtenu par OCR
std::string ValiderOCR(std::string output, bool estserveur, bool inverse, uncoin& moncoin, const config& maconf) {
    // valider le caractère obtenu.
    // chiffre 1 : absence de gros symbole à coté du chiffre ni au centre
    // chiffre 2 ou 3 : absence de gros symbole à coté du chiffre, présence au centre
    // chiffre 4 à 9 ou 10 : présence de gros symbole
    // VDR : dessus à gauche ou dessous à droite : présence de gros symbole
    // VDR : (sinon) pas de gros symbole
    bool estGS = false; // gros symbole présent dans le coin ?
    bool estGScent = false; // GS central haut présent ? 
    cv::Rect rr;
    cv::Mat GS;
    cv::Mat lig;
    cv::Mat extrait;
    cv::Scalar m, ect;
    extrait = moncoin.ima_coin.clone();

    if (moncoin.estunRDV && output == "0") output = "D";  // Dame anglaise (Q) confondue avec 0
    if (moncoin.estunRDV && output == "1") output = "V";  // Valet anglais (J) confondue avec 1

    if (output == "V" || output == "D" || output == "R")
    {
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
#ifdef ACTIVER        
        if (!moncoin.estunRDV) {  // chiffre attendu ou indéterminé
            rr.width = maconf.largeurgros;
            rr.height = maconf.taillegros;
            if (!inverse) { // OCR vertical ou OCR par le serveur (= orientation indéterminée)
                // Gros symbole à coté du caractère
                if (moncoin.UU.x > moncoin.PP.x)
                    rr.x = moncoin.PP.x  + maconf.deltagros;
                else
                    rr.x = moncoin.PP.x - maconf.deltagros - rr.width;
                if (moncoin.U.y > moncoin.PP.y)
                    rr.y = moncoin.PP.y + maconf.deltagroshaut;
                else
                    rr.y = moncoin.PP.y - maconf.deltagroshaut - rr.height;
            } else { // OCR inverse
                // gros symbole au dessus ou au dessous du caractère
                if (moncoin.UU.x > moncoin.PP.x)
                    rr.x = moncoin.PP.x + maconf.deltagroshaut;
                else
                    rr.x = moncoin.PP.x - maconf.deltagroshaut - rr.width;
                if (moncoin.U.y > moncoin.PP.y)
                    rr.y = moncoin.PP.y + maconf.deltagros;
                else
                    rr.y = moncoin.PP.y - maconf.deltagroshaut - rr.height;
            }
        } else // le caractère attendu est un Roi Dame ou Valet ( et trouvé)
#endif         
        { 
            if (!inverse) {
                if (moncoin.UU.x > moncoin.QQ.x)
                    rr.x = moncoin.QQ.x + maconf.deltagrosRDV;
                else
                    rr.x = moncoin.QQ.x - maconf.deltagrosRDV - rr.width;
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
                    rr.y = moncoin.QQ.y + maconf.deltagrosRDV + 1;
                else
                    rr.y = moncoin.QQ.y - maconf.deltagrosRDV - rr.height - 1;
            }
        }
        // extraire la zone du gros symbole attendu selon le caractère trouvé par OCR
        if (maconf.printoption && !maconf.threadoption){
            tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
        }
        GS = moncoin.ima_coin(rr).clone();
        // amplifyContrast(GS);
        cv::meanStdDev(GS, m, ect);
        if (ect[0] > 30 + (255 - m[0]) / 5)
            estGS = true;
        if ((!estGS && ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y < moncoin.PP.y)
                || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y > moncoin.PP.y)))
          || (estGS && ((moncoin.UU.x > moncoin.PP.x && moncoin.U.y > moncoin.PP.y)
                || (moncoin.UU.x < moncoin.PP.x && moncoin.U.y < moncoin.PP.y))))
        {
            if (!estserveur)
            {
                if (maconf.printoption)
                    std::cout << output << " !! caractere incompatible avec gros symbole " << std::endl;
                return "";
            }
            else
            { // détecté même si caractère tourné de 90 degrés (ex: par trocr)
                return output;
            }
        }
        return output;
    } else  { // le caractère trouvé par OCR est un chiffre ( ou autre sauf V D R)

        if (output != "10" && !(output >= "1" && output <= "9")) {
            if (output.size() == 2) {
                if (output[0] == '1'){  // 1x --> x 
                    if (output[1] > '1' && output[1] <= '9') output = output[1];
                } else output = "";
            } else output = "";
        }
        if (output == "") return "";
        // TODO : à reprogrammer !
        // GS du coin absent: valide 1 2 ou 3
        //     4 : --> 1
        //     8 : --> 3
        //     >4 : --> invalide
        //     GS central à coté :
        //          2 ou 3 --> OK, 1 --> invalide
        // GS du coin présent: valide 4 à 10
        //     1 : --> 4 (pas de GS sous le coin) ou 7 (GS central sous le coin) 
        //                ou 10 (2 GS sous le coin)
        //     2 : --> 7
        //     3 : --> 8
        //     >= 4 --> OK
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
        lig = moncoin.ima_coin(rr);
        m = cv::mean(lig);
        if (moncoin.moyblanc[0] - m[0] > 20 ) estGS = true;

        if (maconf.printoption && !maconf.threadoption){
            tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
        }

        //  GS central haut ?
        if (inverse){
            if(moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagroshaut;
            else rr.x = moncoin.PP.x - maconf.deltagroshaut - rr.width;
            if (moncoin.U.y  > moncoin.PP.y) rr.y = moncoin.PP.y + (maconf.largeurcarte - rr.height) / 2;
            else rr.y = moncoin.PP.y -(maconf.largeurcarte + rr.height) / 2;
        } else {
            if(moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagroshaut;
            else rr.y = moncoin.PP.y - maconf.deltagroshaut - rr.height;
            if (moncoin.UU.x  > moncoin.PP.x) rr.x = moncoin.PP.x + (maconf.largeurcarte - rr.width) / 2;
            else rr.x = moncoin.PP.x -(maconf.largeurcarte + rr.width) / 2;
        }
        if (rr.x < 0 || rr.y < 0 ||   rr.width > moncoin.ima_coin.cols - rr.x
            || rr.height > moncoin.ima_coin.rows - rr.y) {
            if (maconf.printoption) std::cout<<output<<" GS centre haut  inaccessible"<<std::endl;
            return output; // controle impossible
            }
        lig = moncoin.ima_coin(rr);
        m = cv::mean(lig);
        if (moncoin.moyblanc[0] - m[0] > 20 ) estGScent = true;
        if (maconf.printoption && !maconf.threadoption){
            tracerRectangle(rr, extrait, "valider", cv::Scalar(0, 255, 0));
        }

        if (!estGS){
            if (output == "1") {
                if (!estGScent) return output; // OK
                if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
                return "";
            } 
            if (output == "2" || output == "3") {
                if (estGScent) return output; // OK
                if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
                return ""; // devrait être "1"
            }
            if (output == "7") {
                if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
                if (!estGScent) return "1";
                else return "2";
            }
            if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
            return "";
        }
        // il y a un GS dans le coin --> 4 à 10
        // selon le caractère:
        // 1 : --> 7 ou 4    selon GS sous le caractère
        // 2 : --> invalide
        // 3 : --> 8 ou 5    selon GS au centre (présent --> 5)
        if (output == "2") return ""; 
        // présence d'un GS juste au dessous (position 2 sur 4)
        if (inverse) {
            rr.width = 2*maconf.deltagros/3 ; // eviter le GS dessous au centre (ex : 7)
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagros;
            else rr.y = moncoin.PP.y - maconf.deltagros - rr.height;
            if (moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagroshaut + rr.width + 1;
            else rr.x = moncoin.PP.x - maconf.deltagroshaut - 2*rr.width - 1;

        } else {
            rr.height = 2*maconf.taillegros/3;
            if (moncoin.UU.x > moncoin.PP.x) rr.x = moncoin.PP.x + maconf.deltagros;
            else rr.x = moncoin.PP.x - maconf.deltagros - rr.width;
            if (moncoin.U.y > moncoin.PP.y) rr.y = moncoin.PP.y + maconf.deltagroshaut + rr.height + 1;
            else rr.y = moncoin.PP.y - maconf.deltagroshaut - 2*rr.height - 1;
        }
        if (rr.x < 0 || rr.y < 0 ||   rr.width > moncoin.ima_coin.cols - rr.x
            || rr.height > moncoin.ima_coin.rows - rr.y) {
            if (maconf.printoption) std::cout<<output<<" GS gauche 2/4 inaccessible"<<std::endl;
            // return output; // controle impossible
            if (estGS && output == "1") output = "";  // ce pourrait être un 4 ou un 7
            if (estGS && output == "3") output = "";  // ce pourrait être un 8 ou un 5
        } else {
            // TODO : ne faire ce test que si c'est une nouvelle carte avec 4 cotés (donc complète)
            lig = moncoin.ima_coin(rr);
            m = cv::mean(lig);
            if (maconf.printoption && !maconf.threadoption){
                tracerRectangle(rr, extrait, "valider", cv::Scalar(0, 128, 128));
            }
            if (moncoin.moyblanc[0] - m[0] > 20 ) { // GS présent sous le coin
                if (output == "1") return "10";
                if (output == "3") return "8";
                if (output != "4" && output != "5") return output;
                if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
                return "";
            } else {
                if (output == "1") return "4";
                if (output == "4" || output == "5") return output;
                if (output == "3") return "5";
                if (maconf.printoption) std::cout<<output<<" invalide"<<std::endl;
                return "";
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
    cv::Scalar moyblanc = moncoin.moyblanc;
    int bleulim = 10; // �cart de bleu entre le blanc et la zone test�e pour le petit symbole (*3 pour GS)
    bool estgrossymb = false;
    bool inverse(false);
    bool estDroit(false);
    cv::Mat z1, z2, z3;  // PS vertical, horizontal, sous PS vertical
    cv::Scalar m1, m2, m3;
    cv::Rect r2;
    cv::Mat imacop = moncoin.ima_coin.clone();
    int ts,ls;

    // analyser la zone du petit symbole vertical (z1)
    r.width = maconf.largeursymbole; r.height = maconf.taillesymbole;
    r.height = 3*r.height/2; // grossir sous la zone du symbole vertical
    if (moncoin.estunRDV){
        r.x = moncoin.U.x;
    } else {
        /* if(moncoin.caractere == 'X') */ r.width -= (maconf.largeursymbole / 3);
        if (moncoin.UU.x < moncoin.PP.x) r.x = moncoin.PP.x - maconf.deltasymbole - r.width;
        else r.x = moncoin.PP.x + maconf.deltasymbole;
    }
    if (moncoin.U.y > moncoin.PP.y) r.y = moncoin.U.y;
    else r.y = moncoin.V.y - r.height;
    if(r.height > imacop.rows - r.y) r.height = imacop.rows - r.y;
    if (printoption && !threadoption) {
        cv::circle(imacop, moncoin.PP, 1, cv::Scalar(0,255, 0));
        if (moncoin.estunRDV) cv::circle(imacop, moncoin.QQ, 1, cv::Scalar(0,0, 255));
        tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,255)); // petit symbole vertical
    } 
    z1 = moncoin.ima_coin(r).clone();
    cv:: Rect r1 = r;
    m1 = cv::mean(z1);  // zone du petit symbole vertical

    // si on n'a pas encore déterminé l'orientation, analyser les deux zones du petit symbole
    if (!moncoin.inverse && !moncoin.estDroit){
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

        // analyser la zone du petit symbole horizontal (z2)
        r.width = maconf.taillesymbole; r.height = maconf.largeursymbole;
        if (moncoin.estunRDV) {
            r.x = moncoin.UU.x; r.y = moncoin.UU.y;
        } else {
            /* if(moncoin.caractere == 'X') */ r.height -= (maconf.largeursymbole / 3); // gros symbole!
            r.x = moncoin.UU.x;
            if (moncoin.U.y < moncoin.PP.y) r.y = moncoin.UU.y + maconf.largeursymbole/3;
            else r.y = moncoin.UU.y;
        }
        if (printoption && !threadoption) tracerRectangle(r, imacop, "orient", cv::Scalar(255,255,0)); // PS horizontal
        z2 = moncoin.ima_coin(r).clone();
        m2 = cv::mean(z2);
        r2 = r;

        // amplifyContrast(z1);  // attention moyblanc pas amplifié
        if (moyblanc[0] + moyblanc[1] + moyblanc[2] - m1[2] - m1[1] - m1[0 ] > 3*bleulim )
        {   // zone non blanche
            // vérifier maintenant la zone du symbole horizontal
            //         si elle est blanche, on valide la zone z1
            if (moyblanc[0] + moyblanc[1] + moyblanc[2] - m2[2] - m2[1] - m2[0 ] < 3*bleulim ){ 
                // z2  blanche : z1 = petit symbole vertical
                // z2 blanche : valider z1
            } else {
                // z1 non blanche, z2 non blanche. z1 ou z2 morceau de gros symbole
                // ne se produit que pour R D ou V
                // et (dessus à droite) ou (dessous à gauche) donc inverse
                // ou (dessus à gauche) ou (dessous à droite) donc droit
                // confirmer la présence du gros symbole au delà de la zone du petit symbole
                // se produit aussi pour un chiffre entre 4 et 9 ou 10
                //     la présence avérée du gros symbole ne permet pas de choisir
                cv::Rect rG;
                rG.width = maconf.largeurgrosRDV / 2;
                rG.height = maconf.taillegrosRDV / 2;
                if (moncoin.UU.x > moncoin.QQ.x) rG.x = moncoin.QQ.x + maconf.deltagrosRDV;
                else rG.x = moncoin.QQ.x - maconf.deltagrosRDV  -rG.width;
                if (moncoin.U.y > moncoin.PP.y) rG.y = moncoin.QQ.y + maconf.deltagrosRDV;
                else rG.y = moncoin.QQ.y - maconf.deltagrosRDV - rG.height;
                cv::Mat gros = moncoin.ima_coin(rG);
                if (printoption && !threadoption) tracerRectangle(rG, imacop, "orient", cv::Scalar(255,0,0));
                cv::Scalar mG = cv::mean(gros);
                if (moyblanc[0] - mG[0] > bleulim ) estgrossymb =true;
                if (moncoin.estunRDV) {
                    if (estgrossymb){
                        if (moncoin.U.y < moncoin.PP.y) {
                            if (moncoin.UU.x > moncoin.PP.x) inverse = true;
                        } else if (moncoin.UU.x < moncoin.PP.x) inverse = true;
                    } else {
                        if (moncoin.U.y < moncoin.PP.y) {
                            if (moncoin.UU.x < moncoin.PP.x) inverse = true;
                        } else if (moncoin.UU.x > moncoin.PP.x) inverse = true;
                    }
                } else {
                    // choizir la zone z1 ou z2 la plus foncée (gros symbole à peine présent dans z1 ou z2)
                    if (m1[0] > m2[0]) inverse = true;
                }
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
            if (printoption && !threadoption) tracerRectangle(r, imacop, "orient", cv::Scalar(0,255,0));
            z1 = moncoin.ima_coin(r).clone();
            r1 = r;
        }
        moncoin.inverse = inverse;
    } else { // orientation déjà obtenue
        inverse = moncoin.inverse;
        estDroit = moncoin.estDroit;
        if (estDroit) inverse = false;
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

    // pour déterminer la couleur, agrandir la zone
    // r = r1;
    r.x--; r.y--; r.width += 2, r.height += 2;
    tracerRectangle(r, imacop, "orient", cv::Scalar(0,0,255));
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
    if ( ecartW > 1.20) {   // préciser la limite après les tests
        moncoin.estRouge = true; if (printoption) std::cout << " Rouge! ecart = "<< ecartW;
    } else {
        moncoin.estRouge = false; if (printoption) std::cout << " Noir ecart = "<< ecartW;
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



// calcul de la couleur de bridge à partir du gros symbole
int calculerCouleur(cv::Mat GS, const config& maconf){
    bool printoption = maconf.printoption;
    bool threadoption = maconf.threadoption;
    cv::Mat symbgros = GS.clone();
    cv::Scalar moy, moyext, ect, m1;
    int moyb; // intensité moyenne bleue
    cv::Rect r, rr;
    int yBH, hBH; // position et taille bande horizontale la plus foncée
    cv::Mat bande, centre, lig;
    int ts, ls;
    bool estRouge(false);
    int numcol(-1); // 0=Pique, 2=Coeur, 3=carreau, 4=Trefle, -1=indéterminé
    if (GS.rows == maconf.taillegrosRDV + 4){
        // c'est un personnage
        ts = maconf.taillegrosRDV;
        ls = maconf.largeurgrosRDV;
    } else {
        ts = std::min(maconf.taillegros, GS.rows);
        ls = std::min(maconf.largeurgros, GS.cols);
    }
    // roi_image : petit ou gros symbole
    amplifyContrast(GS); // parfois contre productif
    if (printoption && !threadoption) {
        afficherImage("symbole", symbgros);
        afficherImage("gros", symbgros);
        cv::waitKey(1);
    }
    // déterminer la couleur : Rouge ou noir ?
{
    int Box[4];
    cv::Mat z1, imaR;
     // le symbole est soit rouge, soit noir
    // dans les deux cas, l'intensité bleue est plus faible que l'extérieur
    // déterminer la moyenne de bleu
    // ne considérer que les pixels moins bleus que cette moyenne
    // cumuler les intensités bleues et rouges de ces pixels
    // s'il y a plus de rouge que de bleu, c'est rouge

    z1 = GS.clone();
    amplifyContrast(z1);
    calculerBox(z1, ts, ls, moy, Box, moyext, maconf);
    if (ts >= GS.rows && ls >= GS.cols)
     moyext = cv::Scalar(255,255,255);
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
    if ( ecartW > 1.20) {   // préciser la limite après les tests
        estRouge = true; if (printoption) std::cout << " Rouge! ecart = "<< ecartW<<std::endl;
    } else {
        estRouge = false; if (printoption) std::cout << " Noir ecart = "<< ecartW<<std::endl;
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

    if (estRouge )
        eclaircirfond(GS);
    if (printoption && !threadoption) {
        afficherImage("symbole", symbgros);
        afficherImage("gros", symbgros);
        cv::waitKey(1);
    }
    // calculer l'encombrement du symbole et les moyennes d'intensite du symbole
    int Box[4]; // xmin xmax ymin ymax
    calculerBox(GS, ts, ls, moy, Box, moyext, maconf);
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
    if (printoption && !threadoption)
        tracerRectangle(r, symbgros, "gros", coulBH);
//
/////////////////////// traitement symbole rouge ////////////////////////
//
    if (estRouge && numcol < 0)
    {
        int xgBH = Box[0];
        int xmin, xmax;
        xmin = Box[0];
        xmax = Box[1];
        int xopt;
        // on a xmin et xmax du symbole
        r.x = xgBH; // gauche de la bande horizontale
        r.y = yBH;  // haut de cette bande
        r.width = std::min(Box[1] - r.x + 1, xmax - xmin + 1);
        //if (printoption && !threadoption)
        //    tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 0, 0)); // bande horizontale centrée
        //CS.x = (xmin + xmax) / 2;    // donc le centre horizontal du symbole
        // rechercher la position de la bande verticale optimale (avec le plus de rouge = le moins de bleu)
        //  uniquement dans le tiers inférieur du symbole (en forme de V pour coeur et carreau)
        // essayer avec deux largeurs (2 et 1 ou 3) et choisir le résultat le moins clair
        //
        r.y = yBH + (hBH + 1) / 2; // position au milieu la bande horizontale et au dessous
        if (r.y >= Box[3])  r.y = Box[3];
        r.height = Box[3] + 1 - r.y;
        if (r.height < 1) r.height = 1;
        r.x = Box[0] + ls / 3;
        xmax -= ls / 3;
        
        int largeurcol = 2;
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
            xmax -= ls / 3;
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
            if (printoption)
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
        if (printoption && !threadoption)
            tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0)); // bande vericale centrée
        if (moy[0] - mb > 30)
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
            if (printoption && !maconf.threadoption)
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
                if (printoption)
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
            if (printoption && !maconf.threadoption)
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
            if (printoption && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne en haut à gauche ou droite
            lig = GS(r);
            moy = cv::mean(lig);
            cv::Scalar moyHaut = moy;

            r.y = std::max(0, ybas + 1 - r.height);
            // r.x et r.width inchangés
            if (printoption && !maconf.threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne de test en haut à gauche
            lig = GS(r);
            moy = cv::mean(lig);
            // comparer l'intensité bleue entre le segment en haut et le segment en bas
            // coeur s'il y a significativement plus de bleu en bas, sinon carreau
            int ecartbleu = moy[0] - moyHaut[0];
            if (ecartbleu > 100) numcol = 1; // 100 : experimental
            else if (ecartbleu < 50) numcol = 2;
            // sinon : indéterminé
            if (maconf.printoption)
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
        if (printoption && !threadoption)
        {
            if (numcol == 1)  std::cout << " coeur" << std::endl;
            else              std::cout << " carreau " << std::endl;
            if (maconf.waitoption > 2) cv::waitKey(0); else cv::waitKey(1);
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
        if (ts >= 12){

            cv::Rect r;
            r.height = ts/2;   // 50% semble correct après tests
            if (r.height < 5) r.height = 5; // au moins 5 pixels
            r.width = (ls +2) /3;
            r.x = BoxW[0] + ls/3;
            r.y = BoxW[2]; // haut du symbole
            cv::Mat imsup= GS(r).clone(); // rectangle supérieur central

            r.height = 1; // ligne de 1 pixel
            r.y = 0;
            r.x = 0;
            r.width = imsup.cols;
            cv::Mat lig = imsup(r).clone();
            int mref = cv::mean(lig)[0]; // ref pour Pique limité à la composante bleue
            int mrefT = mref; // référence pour trefle : minimum (plus foncé) des 3 premières lignes
            int m;
            r.y=1;
            while(r.y < imsup.rows) {
                lig = imsup(r).clone();
                m = cv::mean(lig)[0];
                if (m > mrefT + 20) {numcol = 3; break;} // tige de la feuille de trefle
                else if (r.y > imsup.rows - 2  && m < mref - 20) {numcol = 0; break;} // pique
                if (r.y <= 1) mref = std::min(m,mref);
                if (r.y <= 2) mrefT = std::min(m, mrefT);
                r.y++;
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
            if (maconf.printoption && !maconf.threadoption)
             tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 255, 0));
            centre = GS(r);
            // amplifyContrast(centre);
            cv::meanStdDev(centre, moy, ect);
            if (maconf.printoption)
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

            if (maconf.printoption && !maconf.threadoption)
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
            if (maconf.printoption)
                std::cout << " intensite centre " << moycent << ", haut  " << moyhaut << ", ecr " << ecr << std::endl;
                if (ecr > 0) {
                    numcol = 0;
                    if (maconf.printoption)  std::cout << "Pique";
                } else {
                    numcol = 3;
                    if (maconf.printoption)  std::cout << "Trefle";
                }
        }
    }

    return numcol;
}



// obtenir la valeur et la couleur d'une carte
int decoderCarte(cv::Mat& image, int pts[4][2], config& maconf, int& numcol) {
    int printoption = maconf.printoption;
    int waitoption = maconf.waitoption;
    int threadoption = maconf.threadoption;
    // Étape 1 : Ordonner les points
    int ptsT[4][2];
    // selon la destination transformée
    // A(0,0) B (0, max) C (max,max) D (max, 0)
    //  A________D
    //  |       |
    //  |       |
    //  |       |
    //  |       |
    //  B_______C

    // 4 points UVWT
    // UV : les deux points les plus à gauche
    // si UV est la longueur de carte
    //      si U au dessus de V : point 1 = U point 2 = V
    //      sinon                 point 1 = V point 2 = U
    //      point 3 = le plus bas de W et T 
    //      point 4 = l'autre
    //      
    // sinon 
    //      si U est dessous V  : point 1 = U point 4 = V
    //      sinon                 point 1 = V point 4 = U
    //      point 2 = le plus bas de W T
    //      point 3 = l'autre
    
    int lg2moy = (maconf.hauteurcarte + maconf.largeurcarte)/2;
    lg2moy *=lg2moy;
    cv::Point2i U, V, W, T;
    U.x = pts[0][0]; U.y = pts[0][1];
    int i1 = 0; int i2, i3, i4;
    for (int i = 1; i < 4; i++){
        if (pts[i][0] < U.x) { U.x = pts[i][0]; U.y = pts[i][1]; i1 = i;}
    }
    V.x = image.cols; 
    for (int i = 1; i < 4; i++){
        if (i == i1) continue;
        if (pts[i][0] < V.x) { V.x = pts[i][0]; V.y = pts[i][1]; i2 = i;}
    }
    int lg2 = (V.x - U.x)*(V.x - U.x) + (V.y - U.y)*(V.y - U.y);
    if (lg2 > lg2moy) { // UV est la hauteur de carte
        if (U.y < V.y) {
            // point 1 = U
            // point 2 = V
            ptsT[0][0] = U.x; ptsT[0][1] = U.y;
            ptsT[1][0] = V.x; ptsT[1][1] = V.y;
        } else {
            // point 1 = V
            // point 2 = U
            ptsT[0][0] = V.x; ptsT[0][1] = V.y;
            ptsT[1][0] = U.x; ptsT[1][1] = U.y;
        }
        W.y = -1;
        for (int i=0; i <4; i++){
            if (i == i1 || i == i2) continue;
            if (pts[i][1] > W.y) {W.x = pts[i][0]; W.y = pts[i][1]; i3 = i;}
        }
        ptsT[2][0] = W.x; ptsT[2][1] = W.y;
        for (int i=0; i <4; i++){
            if (i == i1 || i == i2 || i == i3) continue;
            ptsT[3][0] = pts[i][0]; ptsT[3][1] = pts[i][1];
        }
    } else { // UV est la largeur de carte
        if (U.y > V.y){
            ptsT[0][0] = U.x; ptsT[0][1] = U.y;
            ptsT[3][0] = V.x; ptsT[3][1] = V.y;
        }else{
            ptsT[3][0] = U.x; ptsT[3][1] = U.y;
            ptsT[0][0] = V.x; ptsT[0][1] = V.y;
        }
        W.y = -1;
        for (int i=0; i <4; i++){
            if (i == i1 || i == i2) continue;
            if (pts[i][1] > W.y) {W.x = pts[i][0]; W.y = pts[i][1]; i3 = i;}
        }
        ptsT[1][0] = W.x; ptsT[1][1] = W.y;
        for (int i=0; i <4; i++){
            if (i == i1 || i == i2 || i == i3) continue;
            ptsT[2][0] = pts[i][0]; ptsT[2][1] = pts[i][1];
        }
    }
    cv::Mat imacarte;
    // calcul de la matrice de déplacement
    cv::Point2i AA(ptsT[0][0], ptsT[0][1]);
    cv::Point2i BB(ptsT[1][0], ptsT[1][1]);
    cv::Point2i CC(ptsT[2][0], ptsT[2][1]);
    cv::Point2i AB(BB-AA); //vecteur
    cv::Point2i BC(CC-BB);
    float lg = cv::norm(AB); // coté long 
    float la = cv::norm(BC);
    float angle = std::atan2(AB.x, AB.y);
    cv::Mat rot = cv::getRotationMatrix2D(AA, -angle * 180.0 / CV_PI, 1.0);
    // transformer AA en (0,0)
    cv::Mat A_mat = (cv::Mat_<double>(3,1) << (double)AA.x, (double)AA.y, 1.0);
    cv::Mat A_rotated = rot * A_mat;
    double dx = -A_rotated.at<double>(0);
    double dy = -A_rotated.at<double>(1);
    rot.at<double>(0, 2) += dx;
    rot.at<double>(1, 2) += dy;



    // image déplacée
    cv::warpAffine(image, imacarte, rot, image.size());
    // affichage après rotation
    //cv::imshow("imrot", imacarte); cv::waitKey(0);
    // image redimensionnée
    cv::Rect rW(0, 0, std::min((int)la, imacarte.cols), std::min((int)lg, imacarte.rows) );
    imacarte = imacarte(rW);
    //cv::imshow("imacarte", imacarte); cv::waitKey(0);

#ifdef ACTIVER
    std::vector<cv::Point2f> ordered = {
        cv::Point2f(ptsT[0][0], ptsT[0][1]),
        cv::Point2f(ptsT[1][0], ptsT[1][1]),
        cv::Point2f(ptsT[2][0], ptsT[2][1]),
        cv::Point2f(ptsT[3][0], ptsT[3][1])
    };
    // Étape 3 : Destination des points dans l’image redressée
    std::vector<cv::Point2f> destination = {
        cv::Point2f(0, 0),                     // A → (0, 0)
        cv::Point2f(0, maconf.hauteurcarte - 1),        // B → (0, h-1)
        cv::Point2f(maconf.largeurcarte - 1, maconf.hauteurcarte - 1), // C
        cv::Point2f(maconf.largeurcarte - 1, 0)          // D
    };

    // Étape 4 : Matrice de transformation
    cv::Mat M = cv::getPerspectiveTransform(ordered, destination);

    // Étape 5 : Appliquer la transformation
    cv::warpPerspective(image, imacarte, M, cv::Size(maconf.largeurcarte, maconf.hauteurcarte));
#endif
    // afficher l'image extraite :
    cv::imshow("imacarte", imacarte);
    // 
    // l'image obtenue correspond aux 4 points:
    // A (0,0)
    // B (0, imacarte.rows - 1)
    // C (imacarte.cols - 1, imacarte.rows-1)
    // D (imacarte.cols - 1, 0)
    // analyser l'image redressée
    //
    // déterminer si c'est un personnage (Roi Dame ou Valet)
    // obtenir la valeur du blanc dand une petite colonne à droite de AB
    // au milieu de la bordure entre le bord de carte (AB) et le cadre éventuel
    // en évitant le caractère et le petit symbole en A et B
    // 
    // considérer les 4 zones entre le bord (exclus) et la position d'un éventuel gros symbole
    // si elles ne sont pas toutes blanches, c'est un personnage, on retourne 0
    //

    int valcarte = -1;
    cv::Mat GS; // gros symbole pour déterminer la couleur 
    int exclure = maconf.taillechiffre + maconf.taillesymbole + maconf.deltachsymb + 2;
    cv::Mat lig;
    cv::Rect r;
    cv::Scalar m, mbl;
    r.x = 1;
    r.width = maconf.deltacadre - 2;
    r.y = exclure;
    r.height = maconf.hauteurcarte - 2*exclure;
    lig = imacarte(r);
    mbl = cv::mean(lig); // valeur du blanc

     r.x = 1;
    r.width = maconf.deltagros - 2;
    r.y = exclure;
    r.height = maconf.hauteurcarte - 2*exclure;
    try {
        lig = imacarte(r);
    } catch (...){
        return -1;
    }
    m = cv::mean(lig);
    if (m[0] < 9*mbl[0]/ 10){
        // c'est un personnage
        valcarte = 0;
    }
    r.x = maconf.deltacadre + maconf.deltaVDR + maconf.largeurVDR;
    r.width = maconf.largeurcarte - 2*r.x;
    r.y = maconf.deltacadre - 1;
    r.height = maconf.deltagroshaut - r.y - 1;
    lig = imacarte(r);
    m = cv::mean(lig);
    if (m[0] < 9*mbl[0]/ 10){
        // c'est un personnage
        valcarte = 0;
    }

    if (valcarte == 0){
        // extraire le gros symbole en haut à gauche (toujours présent pour un personnnage)
        //        et calculer la couleur
        r.x = maconf.deltacadre + maconf.deltagrosRDV ;
        r.y = maconf.deltacadrehaut + maconf.deltagroshautRDV -2;
        r.width = maconf.largeurgrosRDV;
        r.height = maconf.taillegrosRDV +4;
        GS = imacarte(r).clone();

        numcol = calculerCouleur(GS, maconf);

        // extraire le caractère en haut à gauche et le décoder par OCR
        r.x = maconf.deltacadre + maconf.deltaVDR;
        r.y = maconf.deltacadrehaut + maconf.deltahautVDR;
        r.width = maconf.largeurVDR;
        r.height = maconf.tailleVDR;
        cv::Mat ima_car = imacarte(r);
        cv::Mat ima_carW;
        double confiance, angle;
        std::string output;
        cv::cvtColor(ima_car, ima_carW, cv::COLOR_BGR2GRAY);
        cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche
        {
            cv::Mat image_bordee;
            int tb = 2;
            
            cv::copyMakeBorder(ima_carW, image_bordee, tb, tb, tb, tb, cv::BORDER_CONSTANT, cv::Scalar(255));
            ima_carW = image_bordee;
        }
        cv::Mat ima_CARV = ima_carW.clone();
        if (printoption >= 1 && !threadoption){
            afficherImage("V1c", ima_car);
            afficherImage("V1", ima_CARV);
        }
        if (printoption && !threadoption){
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
        }
        if (maconf.tesOCR >= 1) output = tesOCR(ima_carW, true, &confiance, &angle);
        else                    output = execOCR("SERVEUR", ima_carW, &confiance, &angle);

        if (printoption > 1 && output.size() > 0)
            std::cout << "V1 " << output << " confiance " << confiance << " angle " << angle << std::endl;
        if (output == "?")
            output = "";
        if (output == "V") return 11;
        else if (output == "D") return 12;
        else if (output == "R") return 13;

        return 0;
    }

    // Gros symbole en haut à gauche ?
    r.x = maconf.deltagros;
    r.y = maconf.deltagroshaut;
    r.width = maconf.largeurgros;
    r.height = maconf.taillegros;
    lig = imacarte(r);
    m = mean(lig);
    cv::Scalar  mGShg = m;
    if (m[0] > 9*mbl[0]/10) { // pas de GS en haut à gauche --> 1 2 ou 3
        // GS haut centre ?
        r.x = (maconf.largeurcarte - maconf.largeurgros)/ 2;
        r.y = maconf.deltagroshaut;
        lig = imacarte(r);
        m = mean(lig);
        r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;
        lig = imacarte(r); // GS central
        if (m[0] > 9*mbl[0]/10) { // pas de GS haut centre --> 1
            m = cv::mean(lig); // GS central
            if (m[0] > 9*mbl[0]/10) { // pas de GS central
                return 0; // aucun GS haut gauche, haut centre, central --> erreur
            }
            r.y -=2; r.height += 4;
            GS = imacarte(r).clone(); // GS central élargi
            numcol = calculerCouleur(GS, maconf);
            return 1; // on a trouvé
        }
        // il y a un GS en haut au centre
        r.y = maconf.deltagroshaut; // GS haut centre
        r.y -=2; r.height += 4;
        GS = imacarte(r).clone();
        numcol = calculerCouleur(GS, maconf);
        m = cv::mean(lig); // GS central
        if (m[0] > 9*mbl[0]/10) { // pas de GS central --> 2
            return 2;
        } else { // GS central présent --> 3
            return 3;
        }
        return 0; // protection. normalement jamais atteint 
    } else {
        // extraire 2 pixels de plus dessus et dessous
        r.y -= 2; r.height += 4;
        GS = imacarte(r).clone();
    }

    numcol = calculerCouleur(GS, maconf);
    // tracer les zones analysées sur une copie
    cv::Mat imac = imacarte.clone();

    // GS haut gauche présent --> 4 à 10
    // considérer la zone centrale à gauche entre GS haut et bas
    // compter le nombre de GS ( 0 1 ou 2)
    // zone du GS 2 sur 4, limitée au haut du GS 3 sur 4
    int nbGSg=0; // nombre de GS dans la partie centrale de la colonne gauche
    r.x = maconf.deltagros;
    r.y = maconf.deltagroshaut + maconf.taillegros;
    r.width = (maconf.largeurcarte - maconf.largeurgros)/2 - r.x;
    r.height = (maconf.hauteurcarte - maconf.taillegros)/2 - r.y;
    tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
    lig = imacarte(r);
    m = mean(lig);
    if (m[0] < 9*mbl[0]/10) nbGSg=2;  // GS symétrique en dessous
    else {
        r.y = (maconf.hauteurcarte - maconf.taillegros)/ 2;
        r.height = maconf.taillegros;
        tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
        lig = imacarte(r);
        m = mean(lig);
        if (m[0] < 9*mbl[0]/10) nbGSg=1;  // GS central
    }
    if (nbGSg == 0){ // 4 ou 5-->
        // GS central ?
        r.x = (maconf.largeurcarte - maconf.largeurgros) / 2;
        r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;
        r.height = maconf.taillegros;
        r.width = maconf.largeurgros;
        tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
        lig = imacarte(r);
        m = cv::mean(lig);
        if (m[0] > 9*mbl[0]/10) { // pas de GS central --> 4
            return 4;
        }
        return 5;
    } else if (nbGSg == 1) { // --> 6 ou 7
        // GS central sup ou inf ?
        r.x = (maconf.largeurcarte - maconf.largeurgros) / 2;
        r.width = maconf.largeurgros;
        if (r.x < maconf.deltagros + maconf.largeurgros) { // le GS de gauche empiète sur cette colonne centrale
            r.width -= 2*(maconf.deltagros + maconf.largeurgros - r.x);
            r.x = maconf.deltagros + maconf.largeurgros;
        }
        r.height = maconf.taillegros;
        r.y = maconf.deltagroshaut + maconf.taillegros;
        tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
        lig = imacarte(r);
        m = cv::mean(lig);
        if (m[0] < 95*mbl[0]/100) { // GS sup présent
            return 7;
        }
        r.y = maconf.hauteurcarte - maconf.deltagroshaut - maconf.taillegros - r.height;
                tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
        lig = imacarte(r);
        m = cv::mean(lig);
        if (m[0] < 95*mbl[0]/100) { // GS inf présent
            return 7;
        }
        return 6; // pas de GS dans la colonne centrale

    } else { // 4 GS à gauche--> 8 9 ou 10
        // compter les GS de la colonne centrale
        int nbGS = 0;
        r.width = (maconf.largeurcarte - maconf.deltagros - maconf.largeurgros) / 2;
        r.x = (maconf.largeurcarte - r.width) / 2;
        r.y = maconf.deltagroshaut + maconf.taillegros / 2;
        r.height = maconf.taillegros;
        tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
        lig = imacarte(r); // GS haut
        m = cv::mean(lig);
        if (m[0] < 9*mbl[0]/ 10) nbGS = 1;
        r.y = maconf.hauteurcarte - maconf.deltagroshaut - maconf.taillegros - r.height;
        tracerRectangle(r,imac,"carte", cv::Scalar(0,255,0));
        lig = imacarte(r); // GS bas
        m = cv::mean(lig);
        if (m[0] < 9*mbl[0]/ 10) nbGS++;
        if (nbGS == 2) return 10; // 8 sur les cotés et 2 au centre

        r.y = (maconf.hauteurcarte - maconf.taillegros) / 2;
        tracerRectangle(r,imac,"carte", cv::Scalar(255,0,0));
        lig = imacarte(r); // GS central
        m = cv::mean(lig);
        if (m[0] < 9*mbl[0]/ 10) nbGS++;
        if (nbGS == 1) return 9; // 8 sur les cotés et 1 au centre ou en haut ou en bas
        return 8; // rien dans la colonne centrale
    }
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


/*****************
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
    cv::Mat ima2;
    floatImage.convertTo(ima2, CV_8U);
    image = ima2;
    *******************/
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
