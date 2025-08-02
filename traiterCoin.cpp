#define _USE_MATH_DEFINES
#ifdef _WIN32
    #include <Windows.h>
    #include <tchar.h>
#else
    #include <algorithm>
#endif

#include <iostream>
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
#include "config.h"

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>


#ifndef _WIN32
#include <thread> // pour std::thread
// #include <atomic>  // pour std::atomic
// std::atomic<bool> is_window_open(true);
#endif


extern std::mutex mtx;
extern std::condition_variable cvar;
extern int activeThreads;
extern int threadoption;

std::mutex resultMutex;

void retourcoin(int n){
    if (threadoption) {
        //std::cout<<"+++ fin de tache du coin "<<n<<std::endl;
        std::lock_guard<std::mutex> lock(mtx); // protection d'accès à activeThreads
        --activeThreads;
        cvar.notify_one();
    }
     return;
}


void traiterCoin(int n, int coins[][12], cv::Mat image,  std::vector<std::string>& resultats, 
                        cv::Mat result, const int *l1, const int *l2, const config &maconf)
{

    int waitoption = maconf.waitoption;
    int printoption = maconf.printoption;
    if (printoption) std::cout<< "+++Traitercoin " <<n <<std::endl;
    // threadoption = maconf.threadoption;  // Attention ! variable globale
    if(threadoption)
        if (printoption) std::cout << "+++Thread " << std::this_thread::get_id() << " demarre..." << std::endl;
    
    int cecoin[12];
    for (int i = 0; i<12; i++) cecoin[i] = coins[n][i];
    std::string nomOCR = "tesOCR";
    if (maconf.tesOCR == 0)
        nomOCR = "SERVEUR";
    else
        nomOCR = "tesOCR";

    bool estunRDV = false;
    bool estunRDV0 = false; // ce qui a été déterminé par la comparaison des coins
    int i = cecoin[0]; // indice de ligne
    int j = cecoin[1];
    if (i < 0 || j < 0)
        {retourcoin(n); return;} // coin éliminé
    if (cecoin[6])
        estunRDV0 = cecoin[6];
    estunRDV = false; // détection douteuse, désactivée
    // cv::Vec4i l1 = lines[i];  // ligne AB
    // cv::Vec4i l2 = lines[j];  // ligne CD
    cv::Point2i P = cv::Point2i(cecoin[4], cecoin[5]); // intersection des deux lignes
    cv::Point2i Q;
    Q = P;              // initialiser un point valide
    int k = cecoin[2];  // coin= A si 0  ou B (=2)
    int kk = cecoin[3]; // coin = C ou D
    // déterminer le rectangle correspondant au coin selon les directions AB et CD, point diagonal Q
    //
    cv::Point2i R; // AB --> PR  R = A ou B
    cv::Point2i S; // CD --> PS  S = C ou D
    R.x = l1[2 - k];
    R.y = l1[3 - k];
    S.x = l2[2 - kk];
    S.y = l2[3 - kk];

    float lgPR = (R.x - P.x) * (R.x - P.x) + (R.y - P.y) * (R.y - P.y);
    float lgPS = (S.x - P.x) * (S.x - P.x) + (S.y - P.y) * (S.y - P.y);
    lgPR = sqrt(lgPR);
    lgPS = sqrt(lgPS);

    /////////////////////////////////// déterminer le carré de la zone d'intérêt dans le coin (là où il y a le chiffre et le symbole) /////
    // H sur PR     PH = taillechiffre + deltahaut
    // K sur PS     PK = taillechiffre + deltahaut
    // L autre sommet du carré PHLK
    // I centre du carré, milieu de HK
    // Q : cadre si on a identifié un cadre
    // calcul des coordonnées
    cv::Point2i H;
    cv::Point2i K;
    cv::Point2i L;
    cv::Point2i I;
    int taille = maconf.taillechiffre + maconf.deltahaut; // pour visualiser juste le numéro ou  R D V
    if (estunRDV)
        taille = maconf.tailleVDR + maconf.deltahautVDR;
    taille = maconf.hauteurcarte + 4; // extraire toute la carte

    double pr = (R.x - P.x) * (R.x - P.x) + (R.y - P.y) * (R.y - P.y);
    double ps = (S.x - P.x) * (S.x - P.x) + (S.y - P.y) * (S.y - P.y);
    pr = sqrt(pr);
    ps = sqrt(ps);
    H.x = P.x + ((double)taille / pr) * (R.x - P.x);
    H.y = P.y + ((double)taille / pr) * (R.y - P.y);

    K.x = P.x + ((double)taille / ps) * (S.x - P.x);
    K.y = P.y + ((double)taille / ps) * (S.y - P.y);

    I.x = (H.x + K.x) / 2;
    I.y = (H.y + K.y) / 2;

    // PL = 2*PI   L.x - P.x = 2(I.x - P.x)    L.x = 2I.x - P.x
    L.x = 2 * I.x - P.x;
    L.y = 2 * I.y - P.y;

    // encombrement du carré PHLK
    int xg, xd, yh, yb;
    xg = std::min(P.x, H.x);
    xg = std::min(xg, L.x);
    xg = std::min(xg, K.x);

    yh = std::min(P.y, H.y);
    yh = std::min(yh, L.y);
    yh = std::min(yh, K.y);

    xd = std::max(P.x, H.x);
    xd = std::max(xd, L.x);
    xd = std::max(xd, K.x);

    yb = std::max(P.x, H.y);
    yb = std::max(yb, L.y);
    yb = std::max(yb, K.y);

    // extraire un rectangle d'image plus grand pour conserver la zone d'intérêt après rotation
    //xg = xg - 3 * taille;
    //xd = xd + 3 * taille;
    xg--; xd++;
    xg--; xd++;
    if (xg < 0)
        xg = 0;
    if (xd >= image.cols)
        xd = image.cols - 1;
    //yh = yh - 3 * taille;
    //yb = yb + 3 * taille;
    yh--; yb++;
    yh--; yb++;
    if (yh < 0)
        yh = 0;
    if (yb >= image.rows)
        yb = image.rows - 1;

    // PR horizontal ? (|dx| > |dy|)
    int dx = R.x - P.x;
    int dy = R.y - P.y;
    int dxx = S.x - P.x;
    int dyy = S.y - P.y;

    // déterminer l'angle de rotation
    // déterminer le coté PS ou PR le plus horizontal
    // calculer l'angle de rotation, pour redresser le coin
    // le plus horizontal? PS ou PR
    //
    double angrad;
    if ((dx * dx + dy * dy) > (dxx * dxx + dyy * dyy))
    { // choisir PR
        // déterminer l'angle d'inclinaison
        angrad = std::atan2(dy, dx);
    }
    else
    {
        angrad = std::atan2(dyy, dxx);
    }
    double angdeg = angrad * 180.0 / M_PI;
    // effectuer une rotation inférieure à pi, meme à PI/2
    if (angdeg <= -45)
        angdeg += 180; // entre -45 et + ???
    if (angdeg > 180)
        angdeg -= 180;
    if (angdeg > 90)
        angdeg -= 90;
    if (angdeg > 45)
        angdeg -= 90; // rotation limitée à 45° dans un sens ou dans l'autre

    // type de coin :
    //   haut gauche :
    //      cas 1 : S.x > P.x  et S.y < R.y et P.y < R.y
    //      cas 2 : R.x > P.x  et R.y < S.y et P.y < S.y
    //
    //   haut droit
    //      cas 1 : S.x < P.x et S.y < R.y et P.y > R.y
    //      cas 2 : R.x < P.x et S.y > R.y et S.y > P.y
    //
    //   coin bas gauche
    //      cas 1 : S.x > P.x et S.y > R.y et P.y < R.y
    //      cas 2 : R.x > P.x et R.y < S.y et P.y < S.y
    //
    //   coin bas droit
    //      cas 1 : Q.x < P.x et S.y > R.y et P.y < R.y
    //      cas 2 : R.x < P.x et R.y < S.y et P.y < S.y
    //
    //  coins bas : il faudra tourner de 180 degrés
    //      min(S.y, R.y) < P.y

    // extraire le rectangle
    dx = std::max(maconf.hauteurcarte +6, xd - xg);
    dy = std::max(maconf.hauteurcarte +6 ,yb - yh);
    // le point P doit être dans le rectangle à extraire
    // par construction : P.x > xg  P.y > yh
    if (dx < P.x - xg) dx = P.x - xg + 2;
    if (dy < P.y - yh) dy = P.y - yh + 2;
    if (dx > image.cols - xg) dx = image.cols - xg;
    if (dy > image.rows - yh) dy = image.rows - yh;
    cv::Rect regionw(xg, yh, dx, dy);
    cv::Mat coinPetit = image(regionw).clone();
    amplifyContrast(coinPetit);

    // actualiser les coordonnées des points P H K I dans l'image extraite (translation)
    // translation (xg,yh) ---> 0,0)   translation (-xg, -yh)
    cv::Point2i PP(P.x - xg, P.y - yh);
    cv::Point2i HH(H.x - xg, H.y - yh);
    cv::Point2i KK(K.x - xg, K.y - yh);
    cv::Point2i II(I.x - xg, I.y - yh);
    cv::Point2i QQ(Q.x - xg, Q.y - yh);

    //////////////////////////////// redresser l'image //////////////////////////
    if (abs(angdeg) > 0.1)
    {
        if (printoption)
            std::cout << " rotation " << angdeg << " degres" << std::endl;
        // if (printoption && !threadoption) afficherImage("avant rot", coinPetit);
        cv::Point2f ctr(coinPetit.cols / 2, coinPetit.rows / 2);
        cv::Mat imarot;

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(ctr, angdeg, 1.0);
        // Déterminer la taille de l'image de sortie après rotation
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), coinPetit.size(), angdeg).boundingRect2f();
        // Ajuster la matrice de rotation pour tenir compte du changement de taille de l'image
        rotation_matrix.at<double>(0, 2) += bbox.width / 2.0 - coinPetit.cols / 2.0;
        rotation_matrix.at<double>(1, 2) += bbox.height / 2.0 - coinPetit.rows / 2.0;

        // Appliquer la transformation pour obtenir l'image tournée
        cv::warpAffine(coinPetit, imarot, rotation_matrix, bbox.size());
        coinPetit = imarot;
        // actualiser les coordonnées des points PP HH KK II QQ
        cv::Mat W = (cv::Mat_<double>(3, 1) << PP.x, PP.y, 1);
        cv::Mat WW = rotation_matrix * W;
        cv::Point2f PPP(WW.at<double>(0, 0), WW.at<double>(1, 0));
        PP = (PPP + cv::Point2f(0.5, 0.5)); // avec arrondi

        W = (cv::Mat_<double>(3, 1) << HH.x, HH.y, 1);
        WW = rotation_matrix * W;
        cv::Point2f HHH(WW.at<double>(0, 0), WW.at<double>(1, 0));
        HH = HHH;

        W = (cv::Mat_<double>(3, 1) << KK.x, KK.y, 1);
        WW = rotation_matrix * W;
        cv::Point2f KKK(WW.at<double>(0, 0), WW.at<double>(1, 0));
        KK = KKK;

        W = (cv::Mat_<double>(3, 1) << II.x, II.y, 1);
        WW = rotation_matrix * W;
        cv::Point2f III(WW.at<double>(0, 0), WW.at<double>(1, 0));
        II = III;
        if (estunRDV)
        {
            cv::Mat W = (cv::Mat_<double>(3, 1) << QQ.x, QQ.y, 1);
            cv::Mat WW = rotation_matrix * W;
            cv::Point2f QQQ(WW.at<double>(0, 0), WW.at<double>(1, 0));
            QQ = (QQQ + cv::Point2f(0.5, 0.5));
        }
    }
    cv::Mat extrait = coinPetit.clone();
    if (printoption && !threadoption) {
        afficherImage("Extrait", extrait);
        if (waitoption > 2)
            cv::waitKey(0);
        else
            cv::waitKey(1);
    }

    cv::Point2i HO; // point Horizontal HH ou KK
    cv::Point2i VE; // point vertical KK ou HH
    if (abs(PP.y - HH.y) < abs(PP.x - HH.x))
    {
        HO = HH;
        VE = KK;
    }
    else
    {
        HO = KK;
        VE = HH;
    }
    bool estArtefact = false;
    //
    // préciser la position du coin selon la présence des traits des bords
    // on recherche les traits à distance de l'arrondi du coin de carte
    // soit à l'extérieur ( à 1 ou 2 pixels), soit vers l'intérieur
    // une ligne ou colonne vers l'intérieur
    // si elle est plus claire, on était sur un trait de bord ou de cadre
    // sinon, le trait du bord est à l'intérieur
    // dans ce cas, rechercher le trait foncé à proximité vers l'intérieur
    {
        int dc = maconf.deltacadre;
        cv::Rect r;
        cv::Scalar m0, m1, m2, mbl;
        int mb3;
        int limr = 10; // écart entre clair et foncé à présiser par des tests
        cv::Mat lig;
        bool deplace = false;
        mbl = cv::Scalar(255,255,255);  // par défaut, si on ne trouve pas mieux
        // obtenir l'intensité du blanc
        // on utilise une ligne entre le bord horizontal du coin et le chiffre vertical
        // en évitant le début du chiffre horizontal
        r.height = 1;
        r.width = maconf.largeurchiffre;
        if (HO.x < PP.x)
            r.x = PP.x - maconf.deltachiffre - maconf.largeurchiffre - r.width;
        else
            r.x = PP.x + maconf.deltachiffre + maconf.largeurchiffre;
        if (VE.y < PP.y)
            r.y = PP.y - std::max(1, maconf.deltacadre / 2); // entre le bord et le cadre (si RDV)
        else
            r.y = PP.y + std::max(1,maconf.deltacadre / 2);
        if(r.x < 0) r.x = 0;
        if (r.width > coinPetit.cols - r.x) r.width = coinPetit.cols - r.x;
        lig = coinPetit(r);
        mbl = cv::mean(lig); // valeur de référence du blanc
        mb3 = mbl[0] + mbl[1] + mbl[2];
        cv::circle(extrait, PP, 7, cv::Scalar(255, 0, 0));
        if (printoption && !threadoption) {
            afficherImage("Ext", extrait);
            afficherImage("Extrait", extrait);
        }
        int dx = 1;
        if (HO.x < PP.x)
            dx = -1; // d'une colonne à la suivante à l'intérieur du coin
        int dy = 1;
        if (VE.y < PP.y)
            dy = -1; // d'une ligne à la suivante à l'intérieur du coin

        // trait horizontal:
        r.y = PP.y;
        r.height = 1;
        r.width = 2 * maconf.taillechiffre;
        if (HO.x > PP.x) {
            r.x = PP.x + maconf.deltacoin;
            if (r.width > coinPetit.cols - r.x) r.width = coinPetit.cols - r.x;
        }else {
            r.x = PP.x - maconf.deltacoin - r.width;
            if (r.x < 0) {r.width += r.x;  r.x = 0;}
        }
        int ypre = PP.y;
        if (r.y < 0) r.y = 0;
        if (r.y >= coinPetit.rows) r.y = coinPetit.rows - 1;
        lig = coinPetit(r); // ligne du bord horizontal du coin
        m0 = cv::mean(lig); // ligne supposée être le bord de carte
        if (m0[0] + m0[1] + m0[2] - mb3 > -3*limr) { // ligne blanche !
            // chercher une ligne foncée à l'extérieur à 1 ou 2 pixels
            r.y -= dy;
            if (r.y >= 0 && r.y < coinPetit.rows - r.height) {
                lig = coinPetit(r); m1 = cv::mean(lig);
                if (m1[0] + m1[1] + m1[2] - mb3 > -3*limr) { // m1 blanche
                    r.y -= dy;
                    if (r.y >= 0 && r.y < coinPetit.rows - r.height) {
                        lig = coinPetit(r); m1 = cv::mean(lig);
                        if (m1[0] + m1[1] + m1[2] - mb3 < -3*limr) { // m1 foncée
                            PP.y = r.y; deplace = true;
                        }
                    }
                } else {PP.y = r.y; deplace = true;}
            }
        }
        if (!deplace){ // rechercher la ligne foncée vers l'intérieur
            r.y += dy;
            lig = coinPetit(r);
            m1 = cv::mean(lig); // ligne suivante à l'intérieur
            if (m0[0] - m1[0] > limr || m0[0] - mbl[0] > -limr)
            { // m1 est plus foncée ou m0 est claire
                // rechercher une ligne foncée à l'intérieur
                for (int j = 0; i < 3; i++)
                {
                    lig = coinPetit(r); m1 = cv::mean(lig);
                    if (m1[0] - mbl[0] < -limr)
                    { // ligne foncée
                        // il peut y avoir plusieurs lignes foncées
                        // rechercher la plus à l'intérieur, suivie d'une ligne blanche
                        // rechercher la ligne blanche suivante à au plus 3 pixels
                        for (int j = 0; j < 3; j++)
                        {
                            r.y += dy; lig = coinPetit(r);  m2 = cv::mean(lig);
                            if (m2[0] - mbl[0] > -limr)
                            {              // ligne blanche
                                r.y -= dy; // ligne foncée juste avant
                                if (r.y != PP.y)
                                {
                                    PP.y = r.y; // nouveau bord de coin
                                    deplace = true;
                                }
                                break;
                            }
                        }
                    }
                    if (deplace)
                        break;
                    r.y += dy;
                }
            }

        }

        if (deplace && printoption)
            std::cout << "deplacement vertical du coin " << PP.y - ypre << std::endl;
        // trait vertical:
        int xpre = PP.x;
        deplace = false;
        r.x = PP.x;
        r.width = 1;
        r.height = 2 * maconf.taillechiffre;
        if (VE.y > PP.y) {
            r.y = PP.y + maconf.deltacoin;
            if ( r.height > coinPetit.rows - r.y) r.height = coinPetit.rows - r.y;
        }else {
            r.y = PP.y - maconf.deltacoin - r.height;
            if (r.y < 0) {r.height += r.y; r.y = 0;}
        }
        if (r.x < 0) r.x = 0;
        if (r.x >= coinPetit.cols) r.x = coinPetit.cols- 1;
        lig = coinPetit(r); // colonne du bord vertical du coin
        m0 = cv::mean(lig); // colonne supposée être le bord de carte
        if (m0[0] + m0[1] + m0[2] - mb3 > -3*limr) { // blanche !
            // chercher une ligne foncée à l'extérieur à 1 ou 2 pixels
            r.x -= dx;
            if (r.x >= 0 && r.x < coinPetit.cols - r.width) {
                lig = coinPetit(r); m1 = cv::mean(lig);
                if (m1[0] + m1[1] + m1[2] - mb3 > -3*limr) { // m1 blanche
                    r.x -= dx;
                    if (r.x >= 0 && r.x < coinPetit.cols - r.width) {
                        lig = coinPetit(r); m1 = cv::mean(lig);
                        if (r.x >= 0 && r.x < coinPetit.cols - r.width) {
                            if (m1[0] + m1[1] + m1[2] - mb3 < -3*limr) { // m1 foncée
                                PP.x = r.x; deplace = true;
                            }
                        }
                    }
                } else {PP.x = r.x; deplace = true;}
            }
        }
        if (!deplace) {
            r.x += dx;
            lig = coinPetit(r);
            m1 = cv::mean(lig); // colonne suivante à l'intérieur
            if (m0[0] - m1[0] > limr || m0[0] - mbl[0] > -limr)
            { // m1 est plus foncée ou m0 est claire
                // rechercher une ligne foncée à l'intérieur
                for (int j = 0; i < 3; i++)
                {
                    lig = coinPetit(r); m1 = cv::mean(lig);
                    if (m1[0] - mbl[0] < -limr)
                    { // ligne foncée
                        // il peut y avoir plusieurs lignes foncées
                        // rechercher la plus à l'intérieur, suivie d'une ligne blanche
                        // rechercher la ligne blanche suivante à au plus 3 pixels
                        for (int j = 0; j < 3; j++)
                        {
                            r.x += dx;
                            if (r.x >= 0 && r.x < coinPetit.cols - r.width) {
                                lig = coinPetit(r); m2 = cv::mean(lig);
                                if (m2[0] - mbl[0] > -limr)
                                {              // colonne blanche
                                    r.x -= dx; // colonne foncée juste avant
                                    if (r.x != PP.x)
                                    {
                                        PP.x = r.x; // nouveau bord de coin
                                        deplace = true;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (deplace)
                        break;
                    r.x += dx;
                }
            }
        }
        if (deplace && printoption)
            std::cout << "deplacement horizontal du coin " << PP.x - xpre << std::endl;

        cv::circle(extrait, PP, 1, cv::Scalar(0, 255, 255)); // position éventuellement décallée
        if (printoption && !threadoption) {
            afficherImage("Ext", extrait);
            afficherImage("Extrait", extrait);
        }
    }

    // si on a déjà déterminé que c'est un RDV, ne pas tester les décallages de position du coin
    //  préciser la position du coin en fonction des bordures dessus, dessous, à droite, à gauche
    // 4 cas pour la position de P et Q
    // P est le coin de carte : bordures blanches à l'intérieur du coin
    // P est décalé à l'intérieur en X : bordure horizontale blanche, bordure verticale contient un caractère
    //     et bordure blanche verticale à l'extérieur
    // P est décalé à l'intérieur en Y : bordure verticale blanche, b. Horiz contient un caractère
    //     et bordure blanche horizontale à l'extérieur
    // P est complètement décalé en X et Y : bordure horizontale et verticale contient un caractère
    //     et bordures blanches H et V à l'extérieur

    if (false && !estunRDV)
    { // désactivé, car les décallages viennent d'être calculés
        cv::circle(extrait, PP, 1, cv::Scalar(255, 0, 0));
        if (printoption && !threadoption) afficherImage("Extrait", extrait);
        cv::Rect r, rH, rV, rHX, rVX;
        cv::Mat bordH, bordV, bordHX, bordVX;
        cv::Scalar mH, mV, mHX, mVX, ectH, ectV, ectHX, ectVX;
        int dx(0), dy(0);
        int dc = maconf.deltacadre;
        int dw = 1;
        if (dc > 5)
            dw = 2;
        // bordure intérieure horizontale
        r.width = maconf.deltachiffre;
        r.height = dc - 2 * dw;
        if (HO.x > PP.x)
            r.x = PP.x + maconf.deltacoin;
        else
            r.x = PP.x - (maconf.deltacoin + r.width);
        r.y = PP.y;
        if (VE.y > PP.y)
            r.y += dw;
        else
            r.y = PP.y - dw - r.height;
        rH = r;
        bordH = coinPetit(r);
        meanStdDev(bordH, mH, ectH);
        // bordure intérieure verticale
        r.width = dc - 2 * dw;
        r.height = maconf.taillechiffre;
        r.x = PP.x;
        if (HO.x > PP.x)
            r.x += dw;
        else
            r.x = PP.x - dw - r.width;
        r.y = PP.y;
        if (VE.y > PP.y)
            r.y = PP.y + dc + maconf.deltacoin;
        else
            r.y = PP.y - (maconf.deltacoin + r.height);
        rV = r;
        bordV = coinPetit(r);
        cv::meanStdDev(bordV, mV, ectV);
        // bordure extérieure horizontale
        r.width = dc + maconf.deltachiffre;
        r.height = dc - 2 * dw;
        r.x = PP.x;
        if (HO.x > PP.x)
            r.x += maconf.deltacoin;
        else
            r.x = PP.x - (maconf.deltacoin + r.width);
        r.y = PP.y;
        if (VE.y > PP.y)
            r.y = PP.y - r.height - dw;
        else
            r.y = PP.y + dw;
        rHX = r;
        bordHX = coinPetit(r);
        meanStdDev(bordHX, mHX, ectHX);
        // bordure extérieure verticale
        r.width = dc - 2 * dw;
        r.height = dc + maconf.deltachiffre;
        r.x = PP.x;
        if (HO.x > PP.x)
            r.x = PP.x + dw;
        else
            r.x = PP.x - dw - r.width;
        r.y = PP.y;
        if (VE.y > PP.y)
            r.y = PP.y + maconf.deltacoin;
        else
            r.y = PP.y - maconf.deltacoin - r.height;
        rVX = r;
        bordVX = coinPetit(r);
        cv::meanStdDev(bordVX, mVX, ectVX);

        if (ectH[0] > 30)
        { // caractère à droite ou à gauche de PP
            if (printoption && !threadoption) tracerRectangle(rH, extrait, "Extrait", cv::Scalar(255, 0, 0));
            // vérifier que la bordure horizontale extérieure est uniforme, sinon artefact
            if (ectHX[0] > 20)
            {
                estArtefact = true;
                std::cout << " Artefact bord H" << std::endl;
                {retourcoin(n); return;}
            }
            dy = dc;
            if (VE.y > PP.y)
                dy = -dc; // noter décallage dessus ou dessous
            std::cout << "décallage coin Y " << dy << std::endl;
        }
        if (ectV[0] > 30)
        { // caractère dessus ou dessous  de PP
            if (printoption && !threadoption) tracerRectangle(rV, extrait, "Extrait", cv::Scalar(255, 0, 0));
            // vérifier que la bordure verticale extérieure est blanche, sinon artefact
            if (ectVX[0] > 20)
            {
                estArtefact = true;
                std::cout << " Artefact bord V" << std::endl;
                {retourcoin(n); return;} 
            }
            dx = dc;
            if (HO.x > PP.x)
                dx = -dc; // noter décallage à gauche ou à droite
            std::cout << "Décallage coin X " << dx << std::endl;
        }
        if (dx)
        {
            PP.x += dx;
            HO.x += dx;
            VE.x += dx;
            if (HO.x > PP.x)
                QQ.x = PP.x + dc;
            else
                QQ.x = PP.x - dc;
            estunRDV = true;
        }
        if (dy)
        {
            PP.y += dy;
            HO.y += dy;
            VE.y += dy;
            if (VE.y > PP.y)
                QQ.y = PP.y + dc;
            else
                QQ.y = PP.y - dc;
            estunRDV = true;
        }
    }

    // déterminer si c'est un artefact:
    // le coin ne doit pas être uniforme.
    // considérer la zone où doit se trouver le caractère, horizontal ou vertical
    // en s'éloignant des bords de la carte (2 pixels)
    {
        cv::Scalar m1, m2, ect1, ect2;
        int dcc = maconf.deltacadre;
        cv::Rect rr;
        int szcar = std::max(maconf.taillechiffre, maconf.largeurchiffre);
        rr.width = rr.height = szcar;
        if (II.x > PP.x ) rr.x = PP.x + 2;
        else rr.x = PP.x - 2 - rr.width;
        if (II.y > PP.y) rr.y = PP.y + 2;
        else rr.y = PP.y - 2 - rr.height;
        cv::Mat carre = coinPetit(rr); // carré centré sur le caractère
        cv::meanStdDev(carre, m1, ect1);
        if (printoption && !threadoption) std::cout << m1 << ect1 << std::endl;
        if ( m1[0] + m1[1] + m1[2] > 720 ||
            (ect1[0] < m1[0] / 10 && ect1[1] < m1[1] / 10 && ect1[2] < m1[2] / 10) )
        {
            if (printoption && !threadoption) {
                cv::circle(result, cv::Point2i(cecoin[4], cecoin[5]), 2, cv::Scalar(255, 0, 0), -1);
                afficherImage("result", result);
                tracerRectangle(rr, extrait, "Artefact", cv::Scalar(255, 0, 0));
                std::cout << " artefact caractere clair ou uniforme " << std::endl;
            }
            {retourcoin(n); return;} 
        }
        // les bandes entre le bord de carte et la position éventuelle du cadre ou du chiffre
        //    doivent être uniformes et plutot claires (surtout en rouge)
        //    attention au coin arondi
        // ne pas faire ce test si l'écart entre le bord de carte et le cadre éventuel est trop faible
        if (maconf.deltacadre > 3 ) { 
            dcc = 1;
            if (maconf.deltacadre > 5)
                dcc = maconf.deltacadre / 3;
            // bande horizontale:
            cv::Mat lig;
            // recherche du cadre horizontal:
            rr.width = maconf.taillechiffre + maconf.taillesymbole;
            rr.height = std::max(1, maconf.deltacadre - 2 * dcc - 1); // flou du bord de carte
            if (II.x > PP.x)
            {
                rr.x = PP.x + maconf.deltahautsymbole;
                rr.width = std::min(rr.width, coinPetit.cols - rr.x);
            }
            else
            {
                rr.x = PP.x - maconf.deltahautsymbole - rr.width;
                if (rr.x < 0)
                {
                    rr.width += rr.x;
                    rr.x = 0;
                }
            }
            if (II.y > PP.y)
                rr.y = PP.y + dcc;
            else
                rr.y = PP.y - maconf.deltacadre + dcc;
            lig = coinPetit(rr);
            cv::meanStdDev(lig, m1, ect1);
            if (ect1[0] > m1[0] / 10 || ect1[1] > m1[1] / 10 || ect1[2] > m1[2] / 10)
            {
                if (printoption && !threadoption) {
                    afficherImage("result", result);
                    tracerRectangle(rr, extrait, "Artefact", cv::Scalar(255, 0, 0));
                    std::cout << " artefact H " << m1 << ect1 << std::endl;
                }
                estArtefact = true;
                // return ; // uniquement si le caractère n'est pas 10
            }
            // bande verticale
            rr.height = maconf.taillechiffre + maconf.taillesymbole;
            rr.width = std::max(1, maconf.deltacadre - 2 * dcc - 1);
            if (II.y > PP.y)
            {
                rr.y = PP.y + maconf.deltahautsymbole;
                rr.height = std::min(rr.height, coinPetit.rows - rr.y);
            }
            else
            {
                rr.y = PP.y - maconf.deltahautsymbole - rr.height;
                if (rr.y < 0)
                {
                    rr.height += rr.y;
                    rr.y = 0;
                }
            }
            if (II.x > PP.x)
                rr.x = PP.x + dcc;
            else
                rr.x = PP.x - maconf.deltacadre + dcc;
            lig = coinPetit(rr);
            cv::meanStdDev(lig, m1, ect1);
            if (ect1[0] > m1[0] / 10 || ect1[1] > m1[1] / 10 || ect1[2] > m1[2] / 10)
            {
                if (printoption && !threadoption) {
                    afficherImage("result", result);
                    tracerRectangle(rr, extrait, "Artefact", cv::Scalar(255, 0, 0));
                    std::cout << " artefact V " << m1 << ect1 << std::endl;
                }
                estArtefact = true;
                // return ;
            }
        }
    }
    if (estunRDV != estunRDV0){
        if (printoption) std::cout<<"!!! incohérence RDV. avant "<<estunRDV0<<" après "<<estunRDV<<std::endl;
    }
    if (!estunRDV) {   // chiffre ou pas encore déterminé
        cv::Rect rr;
        cv::Scalar m1, m2, ect1, ect2;
        cv::Mat lig;
        //
        // déterminer si c'est un RDV en recherchant les traits du cadre
        // considérer une ligne à l'intérieur du coin, entre le bord et le cadre éventuel
        // analyser les lignes vers l'intérieur, jusqu'au cadre (+ un peu)
        // si on trouve une ligne plus foncée, c'est un cadre
        // ligne au dela du chiffre et du symbole, vers le milieu de la carte

        rr.height = 1;
        rr.width = maconf.taillechiffre + maconf.taillesymbole;
        if (II.x > PP.x)
        {
            rr.x = PP.x + maconf.deltachiffre + maconf.taillechiffre + maconf.taillesymbole;
            rr.width = std::min(rr.width, coinPetit.cols - rr.x);
            if (rr.x >= coinPetit.cols) rr.x = coinPetit.cols - 1;
        }
        else
        {
            rr.x = PP.x - maconf.deltachiffre - maconf.taillechiffre - maconf.taillesymbole - rr.width;
            if (rr.x < 0)
            {
                rr.width += rr.x;
                rr.x = 0;
            }
        }
        if (II.y > PP.y)
            rr.y = PP.y + std::max(1, maconf.deltacadre / 2);
        else
            rr.y = PP.y - std::max(1, maconf.deltacadre / 2);
        if (rr.width <= 0) rr.width = 1;
        lig = coinPetit(rr);
        cv::meanStdDev(lig, m1, ect1);

        m2 = m1;
        double mbleuref = m1[0];
        if (II.y > PP.y)
        {
            rr.y++;
            while (rr.y < PP.y + maconf.deltacadre + 2)
            {
                lig = coinPetit(rr);
                cv::meanStdDev(lig, m2, ect2);
                if (m2[0] < mbleuref - 8)
                    break;
                if (m2[0] > mbleuref)
                    mbleuref = m2[0];
                rr.y++;
            }
        }
        else
        {
            rr.y--;
            while (rr.y > PP.y - maconf.deltacadre - 2)
            {
                lig = coinPetit(rr);
                cv::meanStdDev(lig, m2, ect2);
                if (m2[0] < mbleuref - 8)
                    break;
                if (m2[0] > mbleuref)
                    mbleuref = m2[0];
                rr.y--;
            }
        }
        if (m2[0] < mbleuref - 8)
        {
            estunRDV = true;
            QQ.y = rr.y;
        }
        // recherche du cadre vertical:
        rr.height = maconf.taillechiffre + maconf.taillesymbole;
        rr.width = 1;
        if (II.y > PP.y)
        {
            rr.y = PP.y + maconf.deltachiffre + maconf.taillechiffre + maconf.taillesymbole;
            if (rr.y > coinPetit.rows - rr.height) rr.y = std::max(0,coinPetit.rows - rr.height);
            rr.height = std::max(1,std::min(rr.height, coinPetit.rows - rr.y));
        }
        else
        {
            rr.y = PP.y - maconf.deltachiffre - maconf.taillechiffre - maconf.taillesymbole - rr.height;
            if (rr.y < 0) { rr.height += rr.y; rr.y = 0; if (rr.height <= 0) rr.height = 1; }
        }
        if (II.x > PP.x)
            rr.x = PP.x + std::max(1,maconf.deltacadre / 2);
        else
            rr.x = PP.x - std::max(1, maconf.deltacadre / 2);
        lig = coinPetit(rr);
        cv::meanStdDev(lig, m1, ect1);
        m2 = m1;
        mbleuref = m1[0];
        if (II.x > PP.x)
        {
            rr.x++;
            while (rr.x < PP.x + maconf.deltacadre + 2)
            {
                lig = coinPetit(rr);
                cv::meanStdDev(lig, m2, ect2);
                if (m2[0] < mbleuref - 8)
                    break;
                if (m2[0] > mbleuref)
                    mbleuref = m2[0];
                rr.x++;
            }
        }
        else
        {
            rr.x--;
            while (rr.x > PP.x - maconf.deltacadre - 2)
            {
                lig = coinPetit(rr);
                cv::meanStdDev(lig, m2, ect2);
                if (m2[0] < mbleuref - 8)
                    break;
                if (m2[0] > mbleuref)
                    mbleuref = m2[0];
                rr.x--;
            }
        }
        if (estunRDV)
        {
            if (m2[0] < mbleuref - 8)
            {
                // estunRDV = true;
                QQ.x = rr.x;
            }
            else
                estunRDV = false;
        }
    }
    // afficher le coin redressé
    cv::circle(extrait, PP, 4, cv::Scalar(0, 255, 255), -1); // cercle jaune
    cv::circle(extrait, HH, 1, cv::Scalar(0, 255, 128), -1); // cercle vert
    cv::circle(extrait, KK, 1, cv::Scalar(0, 255, 255), -1); // cercle jaune
    // cv::circle(extrait, II, 4, cv::Scalar(0, 255, 255), -1);   // cercle jaune
    if (estunRDV)
        cv::circle(extrait, QQ, 1, cv::Scalar(0, 0, 255), -1); // cercle rouge au coin du cadre

    cv::Scalar moy, ect;
    cv::Scalar moyext;     // moyennes à l'extérieur du symbole ou chiffre
    bool nonvu = true;     // indique que l'on n'a pas identifié le caractère (chiffre, V D R ou 10)
    bool vuprec = false;   // indique que le caractère à été identifié en analysant la zone verticale (il peut être horizontal)
    cv::String output;     // résultat de l' OCR
    cv::String outprec;    // résultat de l'OCR vertical, puis après les deux OCR
    cv::String outRDV;     // résultat de l' OCR pour RDV
    double confiance = 0;  // indice de confiance OCR
    double confRDV = 0;    // indice de confiance OCR RDV
    cv::Mat ima_car;       // image du caractère
    cv::Rect r;            // rectangle d'extraction d'une image de l'image extraite coinPetit
    bool estVert = false;  // indique que le caractère est vertical
    bool estHoriz = false; // indique que le caractère est horizontal
    bool estDroit = false; // indique que le caractère est vertical
    bool inverse = false;  // indique que la carte (redressée) est horizontale
    bool estRouge = false; // indique que le coin comporte un caractère et symbole rouge
    bool estNoir = false;  // indique que le coin est noir
    bool etaitRDV = estunRDV; // mémoriser si le cadre a été trouvé
    double angle = 0;         // angle du caractère dans le coin : 0:indéterminé, 360: vertical, 90  
    double angV = 0;         // angle de détection verticale : 0(= indéterminé)  90 ou 360
    bool estServeur = false; // détection par serveur OCR (trOCR) (orientation non déterminée)

    std::string Vcar(""), Hcar(""); // caractère détecté par OCR

    //
    //  coinPetit : redressé si incliné
    //
    // commencer par rechercher le symbole pique coeur ... dans ce coin
    // deux zones possibles. l'une contient le symbole, l'autre est de couleur uniforme si c'est un chiffre
    //             si c'est un R D V : trouver un autre moyen d'identification
    //             carte verticale : un gros symbole est à droite ou à gauche
    //             carte horizontale : pas de symbole sous (ou sur) la lettre VDR et le symbole
    //
    // pivoter de 90 degrés si nécessaire
    // analyser coinPetit, qui contient le chiffre et le symbole
    // triangle PP HH KK
    // calculer la couleur moyenne des pixels dans la zone prévisible du symbole et dans la zone symétrique. comparer
    // rectangle de la zone 1 : U coin envisagé du symbole
    //                          V coin diagonal
    //                          A coin haut gauche du chiffre ou VDR
    //                          B coin diagonal
    // rectangle de la zone 2 : UU coin envisagé du symbole
    //                          VV coin diagonal
    //                          AA coin haut gauche du chiffre si inversé
    //                          BB coin diagonal

    // si PP-HH est horizontal U est à distance horizontale de P de taille (deltachiffre + taillechiffre déjà calculé)
    //                         U est à distance verticale de P de deltachiffre  du même coté que KK
    cv::Point2i U;
    cv::Point2i V;
    cv::Point2i A;
    cv::Point2i B;

    cv::Point2i UU;
    cv::Point2i VV;
    cv::Point2i AA;
    cv::Point2i BB;

    // agrandir la zone d'extraction. on incorpore le bas du chiffre
    // il faudra rechercher la ligne blanche dans le symbole extrait (en remontant à partir de la ligne médiane
    // à faire aussi sur la largeur du symbole

    int largeursymbole;
    if (estunRDV)
    {
        if(coins[n][10] >= 0) { // on connait la couleur de la carte identifiée comme un personnage
            // calculer les coordonnées de QQ
            if (VE.y < PP.y) QQ.y = PP.y - maconf.deltacadre;
            else QQ.y = PP.y + maconf.deltacadre;
            if (HO.x < PP.x) QQ.x = PP.x - maconf.deltacadre;
            else QQ.x = PP.x + maconf.deltacadre;
        }
        if (printoption)
            std::cout << "RDV  P =" << PP << " Q=" << QQ << std::endl;
        largeursymbole = maconf.largeursymbole;
    }
    else
    {
        largeursymbole = maconf.largeursymbole; // largeur identique pour honneurs et petites cartes
    }

    int taillecar = maconf.taillechiffre;
    int largeurcar = maconf.largeurchiffre;
    int deltahaut = maconf.deltahaut;
    if (estunRDV)
    {
        taillecar = maconf.tailleVDR;
        largeurcar = maconf.largeurVDR;
        deltahaut = maconf.deltahautVDR;
    }
    int deltaVDR = deltahaut - maconf.deltacadrehaut; // écart entre le cadre et le haut de la lettre VDR
    bool cadreY = estunRDV && abs(abs(QQ.y - PP.y) - maconf.deltacadre) < (maconf.deltacadre + 1) / 2;
    bool cadreX = estunRDV && abs(abs(QQ.x - PP.x) - maconf.deltacadre) < (maconf.deltacadre + 1) / 2;
    if (printoption > 1)
        std::cout << "cadreY? " << cadreY << " cadreX? " << cadreX << std::endl;

    // déterminer les points A et B du chiffre (ou de VDR) et les points U et V du symbole et AA BB UU VV

    if (VE.y < PP.y)
    { // dessus  __|  ou |__
        if (printoption > 1)
            std::cout << "Dessus ";
        if (cadreY)
        {
            B.y = QQ.y - maconf.deltaVDR; // ignorer le trait du cadre
            BB.y = B.y;
            VV.y = QQ.y - maconf.deltasymbcadre;
            A.y = B.y - maconf.tailleVDR;
            V.y = A.y - maconf.deltachsymb;
        }
        else
        {
            B.y = PP.y - deltahaut;
            BB.y = PP.y - maconf.deltachiffre;
            A.y = B.y - taillecar - 1;
            VV.y = PP.y - maconf.deltasymbole;
            V.y = PP.y - maconf.deltahautsymbole;
        }
        U.y = V.y - maconf.taillesymbole + 1;
        UU.y = VV.y - maconf.largeursymbole + 1;
        AA.y = BB.y - largeurcar + 1;
    }
    else
    { //  dessous   T
        if (printoption > 1)
            std::cout << "dessous ";
        if (cadreY)
        {
            A.y = QQ.y + maconf.deltaVDR;
            AA.y = A.y;
            UU.y = QQ.y + maconf.deltasymbcadre;
            B.y = A.y + maconf.tailleVDR;
            U.y = B.y + maconf.deltachsymb;
        }
        else
        {
            A.y = PP.y + deltahaut;
            AA.y = PP.y + maconf.deltachiffre;
            UU.y = PP.y + maconf.deltasymbole;
            B.y = A.y + taillecar + 1;
            U.y = PP.y + maconf.deltahautsymbole;
        }
        V.y = U.y + maconf.taillesymbole;
        VV.y = UU.y + maconf.largeursymbole;
        BB.y = AA.y + largeurcar - 1;
    }
    if (HO.x > PP.x)
    { // à droite  |__ ou |--
        if (printoption > 1)
            std::cout << "a droite" << std::endl;
        if (cadreX)
        {
            U.x = QQ.x + maconf.deltasymbcadre;
            A.x = QQ.x + maconf.deltaVDR;
            AA.x = A.x;
            BB.x = AA.x + maconf.tailleVDR;
            UU.x = BB.x + maconf.deltachsymb;
        }
        else
        {
            U.x = PP.x + maconf.deltasymbole;
            A.x = PP.x + maconf.deltachiffre;
            AA.x = PP.x + deltahaut;
            BB.x = AA.x + taillecar - 1;
            UU.x = PP.x + maconf.deltahautsymbole;
        }

        V.x = U.x + maconf.largeursymbole;
        VV.x = UU.x + maconf.taillesymbole - 1;
        B.x = A.x + largeurcar + 1;
    }
    else
    { // à gauche __| ou --|
        if (printoption > 1)
            std::cout << "a gauche" << std::endl;
        if (cadreX)
        {
            V.x = QQ.x - maconf.deltasymbcadre;
            B.x = QQ.x - maconf.deltaVDR;
            BB.x = B.x;
            AA.x = BB.x - maconf.tailleVDR + 1;
            VV.x = AA.x - maconf.deltachsymb;
        }
        else
        {
            V.x = PP.x - maconf.deltasymbole;
            B.x = PP.x - maconf.deltachiffre;
            BB.x = PP.x - deltahaut;
            AA.x = BB.x - taillecar + 1;
            VV.x = PP.x - maconf.deltahautsymbole;
        }
        U.x = V.x - maconf.largeursymbole;
        VV.x = PP.x - maconf.deltahautsymbole;
        UU.x = VV.x - maconf.taillesymbole + 1;
        A.x = B.x - largeurcar - 1;
    }
    // préciser la position du petit symbole par rapport au bord de carte
    // chercher une ligne (ou colonne) non blanche à l'intérieur dans la position verticale prévisible
    if (!estunRDV) {
        cv::Mat lig;
        cv::Scalar m0, m1;
        cv::Rect r;
        int dx(1), dy(1);
        if (HO.x < PP.x)
            dx = -1;
        if (VE.y < PP.y)
            dy = -1;
        r.x = PP.x + dx;
        r.height = 2 * maconf.taillesymbole;
        if (VE.y > PP.y) {
            r.y = PP.y + maconf.deltahautsymbole;
            if (r.height > coinPetit.rows - r.y) r.height = coinPetit.rows - r.y;
        }else{
            r.y = PP.y - maconf.deltahautsymbole - r.height;
            if (r.y < 0) { r.height += r.y; r.y = 0;}
        }
        r.width = 1;
        lig = coinPetit(r);
        m0 = cv::mean(lig); // ligne blanche à coté du bord
        for (int i = 0; i < maconf.deltasymbole - 1; i++)
        {
            r.x += dx;
            lig = coinPetit(r);
            m1 = cv::mean(lig);
            if (m1[0] - m0[0] < -10)
                break;
        }
        // on a obtenu la position du symbole vertical
        if (HO.x < PP.x) {V.x = r.x; U.x = V.x - maconf.largeursymbole + 1;}
        else {U.x = r.x; V.x = U.x + maconf.largeursymbole -1;}

        // position du petit symbole horizontal
        r.y = PP.y + dy;
        r.width = 2 * maconf.taillesymbole;
        if (HO.x > PP.x) {
            r.x = PP.x + maconf.deltahautsymbole;
            if (r.x > coinPetit.cols - r.width) r.x = coinPetit.cols - r.width;
        }else{
            r.x = PP.x - maconf.deltahautsymbole - r.width;
            if (r.x < 0) { r.width += r.x; r.x = 0; if(r.width <= 0) r.width = 1;}
        }
        r.height = 1;
        lig = coinPetit(r);
        m0 = cv::mean(lig); // ligne blanche à coté du bord
        for (int i = 0; i < maconf.deltasymbole - 1; i++)
        {
            r.y += dy;
            lig = coinPetit(r);
            m1 = cv::mean(lig);
            if (m1[0] - m0[0] < -10)
                break;
        }
        if (VE.y < PP.y) {VV.y = r.y; UU.y = VV.y - maconf.largeursymbole + 1;}
        else {UU.y = r.y; VV.y = UU.y + maconf.largeursymbole - 1;}
    }


    U.x = std::max(0, U.x);
    V.x = std::max(0, V.x);
    UU.x = std::max(0, UU.x);
    VV.x = std::max(0, VV.x);
    U.y = std::max(0, U.y);
    V.y = std::max(0, V.y);
    UU.y = std::max(0, UU.y);
    VV.y = std::max(0, VV.y);
    A.x = std::max(0, A.x);
    A.y = std::max(0, A.y);
    B.x = std::max(0, B.x);
    B.y = std::max(0, B.y);
    AA.x = std::max(0, AA.x);
    AA.y = std::max(0, AA.y);
    BB.x = std::max(0, BB.x);
    BB.y = std::max(0, BB.y);

    // mémoriser les caractéristiques du coin qu'on vient de calculer
    uncoin moncoin(coins);
    moncoin.A = A;
    moncoin.B = B;
    moncoin.AA = AA;
    moncoin.BB = BB;
    moncoin.U = U;
    moncoin.V = V;
    moncoin.UU = UU;
    moncoin.VV = VV;
    moncoin.PP = PP;
    moncoin.QQ = QQ;
    moncoin.inverse = inverse;
    moncoin.estDroit = estDroit;
    moncoin.numcoin = n;
    moncoin.estunRDV = estunRDV;
    moncoin.ima_coin = coinPetit;
    moncoin.moyblanc = cv::Scalar(255, 255, 255);
    moncoin.caractere = ' ';

    calculerBlanc(moncoin, maconf);

    // élimination des artefacts :
    // coin foncé ou deux zones de symbole foncées ou uniformes
    // coin uniforme (pas de chiffre) ou foncé
    cv::Mat z1, z2;
    cv::Scalar m1, m2, ect1, ect2;
    cv::Rect rz, rz2;
    int limsombre = 100;
    if (moncoin.moyblanc[2] < 200)
        limsombre = 64; // TODO : limites à préciser
    rz.width = std::min(maconf.largeurchiffre, maconf.taillechiffre);
    rz.width = std::min(rz.width, maconf.deltagros - 1);
    rz.width = std::min(rz.width, maconf.deltagroshaut - 1);
    rz.height = rz.width;
    if (UU.x < PP.x) {
        rz.x = PP.x - 1 - rz.width;
    } else {
        rz.x = PP.x +1;
    }
    rz.height = maconf.largeurchiffre;
    if (U.y < PP.y) {
        rz.y = PP.y - 1 - rz.height;
    }    else {
        rz.y = PP.y + 1;
    }
    z1 = coinPetit(rz); // zone minimale du caractère vertical ou horizontal
    cv::meanStdDev(z1, m1, ect1);
    // éliminer ce coin s'il est clair et uniforme
    // clair : 90% du blanc :
    // uniforme : écart type 10% <= 25    (25 à préciser)
    {
        int refb = (moncoin.moyblanc[0] + moncoin.moyblanc[1] + moncoin.moyblanc[2]) * 9 / 10;
        int mt = m1[0] + m1[1] + m1[2];
        int ect = ect1[0] + ect1[1] + ect1[2];
        if (printoption) std::cout << "intensite ecart type " << m1 << ect1 <<moncoin.moyblanc<< std::endl;
        if (mt > refb && ect < 75) {
            // coin clair et uniforme
            if(printoption) 
                std::cout << "coin clair uniforme " << std::endl;
            {retourcoin(n); return;} 
        }
    } 
    if (m1[2] < limsombre)
    {
        if (printoption)
            std::cout << "coin trop sombre " << m1[2] << std::endl;
        cv::circle(result, cv::Point2i(cecoin[4], cecoin[5]), 2, cv::Scalar(255, 0, 0), -1);
        if (printoption && !threadoption)
            tracerRectangle(rz, extrait, "Artefact", cv::Scalar(255, 0, 0));
        {retourcoin(n); return;} 
    }

    // deux zones foncées
    rz.x = U.x;
    rz.width = V.x - U.x + 1;
    rz.y = U.y;
    rz.height = V.y - U.y + 1;
    z1 = coinPetit(rz);
    m1 = cv::mean(z1);
    rz2 = rz;
    rz2.x = UU.x;
    rz2.width = VV.x - UU.x + 1;
    rz2.y = UU.y;
    rz2.height = VV.y - UU.y + 1;
    z2 = coinPetit(rz2);
    m2 = cv::mean(z2);
    // std::cout<<m1[2]<<" "<<m2[2]<<"..";
    if (m1[2] < limsombre && m2[2] < limsombre)
    {
        if (!estunRDV)
            cv::circle(result, cv::Point2i(cecoin[4], cecoin[5]), 2, cv::Scalar(255, 0, 0), -1);
        if (printoption && !threadoption) {
            std::cout << " coin artefact " << m1 << m2 << std::endl;
            tracerRectangle(rz, extrait, "Artefact", cv::Scalar(255, 0, 0));
            tracerRectangle(rz2, extrait, "Artefact", cv::Scalar(255, 0, 0));
            afficherImage("result", result);
        }
        // normal pour R D V avec gros et petit symbole
        if (!estunRDV)
            {retourcoin(n); return;} 
    }

    cv::circle(extrait, U, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
    cv::circle(extrait, V, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
    cv::circle(extrait, UU, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
    cv::circle(extrait, VV, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
    cv::circle(extrait, A, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
    cv::circle(extrait, B, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
    cv::circle(extrait, AA, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
    cv::circle(extrait, BB, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
    cv::circle(extrait, PP, 2, cv::Scalar(255, 255, 0), -1); // cercle jaune

    if (estunRDV)
        cv::circle(extrait, QQ, 2, cv::Scalar(0, 0, 128), -1); // cercle rouge foncé
    if (printoption && !threadoption) {
        afficherImage("coin", extrait);
        cv::waitKey(1);
    }

    int deltaect = 20; // 20 : valeur expérimentale
    int dc = maconf.deltacadre;

    cv::Mat zone;
    if (printoption)
        std::cout << std::endl;

    dc = maconf.deltacadre; // faciliter l'écriture

    bool recalcul = false;
    bool reafficher = false;
    cv::Mat ima_CARV, ima_CARH;

    // analyser les autres coins de la même carte
    //        on en déduit éventuellement l'orientation de ce coin
    //        et donc l'opportunité de la recherche OCR directe et/ou inverse
    // le premier coté du coin est la droite PP-HH
    // si on a trouvé un coin adjacent, on sait si ce coté est la longueur ou largeur de la carte
    cv::Point2i HP (HH - PP);
    if (coins[n][10] == -3 ) { // PP-HH est la longueur
        if (std::abs(HP.x) > std::abs(HP.y)){ // PP-HH horizontal longueur
            estDroit = false;
            inverse = true;
        } else {
            estDroit = true;
            inverse = false;
        }
    } else if (coins[n][10] == -2) { // PP-HH est la largeur
        if (std::abs(HP.x) > std::abs(HP.y)){ // PP-HH horizontal largeur
            estDroit = true;
            inverse = false;
        } else {
            estDroit = false;
            inverse = true;
        }
    }
    if (printoption > 1) {
        if (inverse)
            std::cout << "inverse ";
        if (estDroit)
            std::cout << "vertical ";
    }


    // calculer l'orientation et la couleur
    // TODO: vérifier
    moncoin.inverse = inverse;
    moncoin.estunRDV = estunRDV;
    moncoin.caractere = ' ';
    calculerOrientation(moncoin, maconf);
    estRouge = moncoin.estRouge;
    inverse = moncoin.inverse;
    estDroit = moncoin.estDroit;
    if (printoption > 1) {
        if (inverse)
            std::cout << "inverse ";
        if (estDroit)
            std::cout << "vertical ";
        if (estunRDV)
            std::cout << "Q=" << QQ;
        std::cout << "P=" << PP << std::endl;
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// identification du caractère par appel à l' OCR /////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int ls, ts;         // largeur et taille du caractère
    cv::Mat ima_carW;   // pour affichage éventuellement grossi
    double confs[8];    // indices de confiance des résultats des 8 cas (4 verticaux et 4 horizontaux)
    std::string out[8]; // 8 caractères lus
    for (int i = 0; i < 8; i++)
    {
        confs[i] = 0;
        out[i] = "";
    }
    if (!inverse)
    { // caractère vertical ou pas encore déterminé
        if (printoption > 1)
            std::cout << "V?" << std::endl;

        dx = B.x - A.x + 1;
        dy = B.y - A.y + 1;
        xg = A.x;
        if (xg == PP.x)
            xg++;
        if (xg + dx == PP.x)
            xg--;
        yh = A.y;
        if (yh == PP.y)
            yh++;
        if (yh + dy == PP.y)
            yh--;
        // ls = dx; ts = dy;
        ls = maconf.largeurchiffre;
        ts = maconf.taillechiffre;
        if (estunRDV)
        {
            ls = maconf.largeurVDR;
            ts = maconf.tailleVDR;
        }
        // provoque des erreurs désactivé:
#ifdef INACTIVE
        ////////////////////// !!!!!!!!!!!!!!!!! élargissement de la zone du chiffre !!!!!!!!!!!
        if (!estunRDV)
        {
            // zone large et haute, mais à au moins 2 pixels du bord du coin sur le coté
            // il y a toujours au moins 2 pixels blancs si la définition permet la lecture du chiffre
            int dc = maconf.deltachiffre / 2;
            dx = maconf.largeurchiffre + maconf.deltachiffre - 0; // avec un peu de symbole
            dy = maconf.deltahautsymbole - dc;
            if (UU.x > PP.x)
                xg = std::max(2 + PP.x, A.x - dc);
            else
                xg = std::min(PP.x - dx - 2, A.x - dc);
            if (U.y > PP.y)
                yh = std::max(1 + PP.y, A.y - dc);
            else
            {
                yh = V.y + 1;
                dy = PP.y - 2 - V.y;
            } // avant le symbole
        }
        else
        {
            dy = dy /*- 1 */ + maconf.deltaVDR;
            if (U.y > PP.y)
            {
                yh = yh + 1 - maconf.deltaVDR;
            }
        }
#endif
        cv::Scalar m1, m2, m3;
        cv::Rect rr;
        int dbleu = 10;
        r = cv::Rect(xg, yh, dx, dy);
        // le haut du symbole peut être en yh ou yh+dy selon orientation
        // on identifie le problème lorsque la ligne au dessus (ou au dessous) est plus claire
        // extraire les deux lignes et comparer
        //  _____
        //
        //  1       s
        //  1       s
        //  9       x
        //  _       _
        //  x       9
        //  s       1
        //  s       1
        //          ____
        rr.x = xg;
        rr.width = dx;
        rr.height = 1;
        if (U.y > PP.y)
        {                       // au dessous, lignes en  9 (yh+dy-1) et _ (yh+dy)
            rr.y = yh + dy - 1; // dernière ligne du chiffre (9)
            cv::Mat lig = coinPetit(rr);
            m1 = mean(lig);
            rr.y++; // ligne (_) entre chiffre et symbole
            lig = coinPetit(rr);
            m2 = mean(lig);
            if (m2[0] > m1[0] + dbleu)
                r.height++; // inclure la ligne blanche
        }
        else
        { // au dessus, lignes en _ ou 9  (yh) et x _  (yh-1)
            // rechercher la ligne blanche en déscendant
            rr.y = yh; // dans le symbole (ou blanc)
            cv::Mat lig = coinPetit(rr);
            m1 = mean(lig); // intensité de la ligne
            while (r.height > maconf.taillechiffre)
            {
                rr.y++; // ligne au dessous
                lig = coinPetit(rr);
                m2 = mean(lig);
                if (m2[0] > m1[0] + dbleu)
                { // est blanche
                    r.y = rr.y;
                    break;
                }
                else if (m2[0] < m1[0] - dbleu)
                {               // ligne du chiffre
                    r.y = rr.y; // inutile de garder une ligne blanche
                    break;
                }
                r.height--;
            }
            // r.height = dy - (r.y - yh);
        }

        // extraire le caractère
        if (r.width > coinPetit.cols - r.x) r.width = coinPetit.cols - r.x;
        if (r.height > coinPetit.rows - r.y) r.height = coinPetit.rows - r.y;
        ima_car = coinPetit(r).clone();
        int Box[4] = {0, 0, 0, 0};
        // déterminer l'encombrement
        calculerBox(ima_car, ts, ls, moy, Box, moyext, maconf);
        r.x += Box[0];
        r.width = Box[1] - Box[0] + 1;
        r.y += Box[2];
        r.height = Box[3] - Box[2] + 1;
        ima_car = coinPetit(r).clone();

        // si on sait que le caractère est vertical et si il est au dessus du coin, tourner de 180 degrés
        // de toutes façons, si le caractère est au dessus, s'il est vertical, il est à l'envers

        // si le caractère est dessus, le retourner de 180 degrés
        if (U.y < PP.y)
            cv::rotate(ima_car, ima_car, cv::ROTATE_180);
        // amplifyContrast(ima_car);
        cv::cvtColor(ima_car, ima_carW, cv::COLOR_BGR2GRAY);
        cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche
        {
            cv::Mat image_bordee;
            int tb = 2;
            if (ts <= 6)
                tb = 1;
            cv::copyMakeBorder(ima_carW, image_bordee, tb, tb, tb, tb, cv::BORDER_CONSTANT, cv::Scalar(255));
            ima_carW = image_bordee;
        }
        // ima_carW = ima_car.clone();
        ima_CARV = ima_carW.clone();
        if (printoption > 1 && !threadoption){
            afficherImage("V1c", ima_car);
            afficherImage("V1", ima_CARV);
        }
        if (printoption && !threadoption){
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
        }
        nonvu = true;
        std::string outRDV;
        if (maconf.tesOCR >= 1) output = tesOCR(ima_carW, estunRDV, &confiance, &angle);
        else                    output = execOCR(nomOCR, ima_carW, &confiance, &angle);
        if (printoption  && output.size() > 0)
            std::cout << "V1 " << output << " confiance " << confiance << " angle " << angle << std::endl;
        if (output == "?")
            output = "";
        // if (confiance < 0.7) output=""; // éviter les fausses détections
        if ((int)angle == 90 && confiance > 0.9)
        {
            inverse = true;
        }
        bool testerVDR = true;
        if (output.size() > 0)
        {
            // accepter 10  V et R si la confiance est suffisante
            if ((output == "10" || output == "R" || output == "V") && confiance > 0.9)
                testerVDR = false;
            if (confiance > 0.99 && output[0] >= '1' && output[0] <= '9')
                testerVDR = false;
            if (!testerVDR)
            {
                confs[0] = confiance;
                out[0] = output;
            }
        }
        if (!inverse && !estunRDV && testerVDR)
        { // tester la zone du caractère VDR éventuel
            double angRDV;
            r.width = maconf.largeurVDR + 2; // agrandir la zone
            r.height = maconf.tailleVDR + 2 * maconf.deltachsymb;
            if (UU.x > PP.x)
                r.x = PP.x + maconf.deltacadre + 1 /* +  maconf.deltaVDR */;
            else
                r.x = PP.x - maconf.deltacadre - 1 - r.width /* - maconf.deltaVDR*/;
            if (U.y > PP.y)
                r.y = PP.y + maconf.deltahautVDR;
            else
                r.y = PP.y - maconf.deltahautVDR - r.height;
            cv::Mat ima_RDV = coinPetit(r).clone();
            ls = maconf.largeurVDR;
            ts = maconf.tailleVDR;
            calculerBox(ima_RDV, ts, ls, moy, Box, moyext, maconf);
            r.x += Box[0];
            r.y += Box[2];
            r.width = ls;
            r.height = ts;
            ima_RDV = coinPetit(r).clone();
            if (U.y < PP.y)
                cv::rotate(ima_RDV, ima_RDV, cv::ROTATE_180);
            // if (dx < 20) cv::resize(ima_RDV, ima_RDV, cv::Size(), 8.0, 8.0);
            // amplifyContrast(ima_RDV);
            // eclaircirfond(ima_RDV);
            cv::cvtColor(ima_RDV, ima_carW, cv::COLOR_BGR2GRAY);
            cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
            if (printoption > 1 && !threadoption ) {
                afficherImage("V1X", ima_carW);
                if (waitoption > 2)
                    cv::waitKey(0);
                else
                    cv::waitKey(1);
                }
            if (maconf.tesOCR >= 1)
            {
                outRDV = tesOCR(ima_carW, estunRDV, &confRDV, &angRDV);
                //if (outRDV == "" && maconf.tesOCR == 1)
                //    outRDV = execOCR("SERVEUR", ima_RDV, &confRDV, &angRDV);
            }
            else
                outRDV = execOCR(nomOCR, ima_RDV, &confRDV, &angRDV);
            if (outRDV == "?")
                outRDV = "";
            if (printoption)
                std::cout << "   OCR V1 pour RDV " << outRDV << " confiance " << confRDV << " angle " << angRDV << std::endl;
            // if (confRDV < 0.7) outRDV="";
            if (confRDV < confiance + 0.2) outRDV = "";
            if ( !estunRDV && (outRDV == "ID" || outRDV == "1D" || outRDV == "|D")) outRDV = "10";
            if (outRDV.size() >= 2 && (outRDV[1] == 'R' || outRDV[1] == 'D' || outRDV[1] == 'V'))
                outRDV = outRDV[1];
            if (outRDV.size() > 0)
            {
                if (outRDV[0] == 'V' || outRDV[0] == 'D' || outRDV[0] == 'R'
                    /*|| outRDV[0] == '>' || outRDV[0] == '<'*/)
                {
                    outRDV = outRDV[0];
                    if (output[0] == 'R' && outRDV[0] == 'D')
                        outRDV = "R"; // R meilleur que D
                    if (confRDV > confiance)
                    {
                        output = outRDV;
                        confiance = confRDV;
                        angle = angRDV;
                    }
                }
                if (output[0] == 'V' || output[0] == 'D' || output[0] == 'R')
                {
                    if ((int)angle == 360 && confRDV > 0.98)
                    {
                        estDroit = true;
                    }
                }
            }
        } // OCR RDV
        // acepter V... D... R...  .V... .D... .R...  .v...
        if ( !estunRDV && (output == "ID" || output == "1D" || output == "|D")) output = "10";
        if (output.size() >= 2 && (output[1] == 'V' || output[1] == 'D' || output[1] == 'R'))
            output = output[1];
        if (output.size() >= 2 &&
            (output[0] == 'V' || output[0] == 'v' || output[0] == 'D' || output[0] == 'R'))
            output = output[0];
        if (output == "v")
            output = "V";
        if (output == "M")
            output = "V"; // bord du cadre V et un morceau de gros symbole
        if (output.size() > 0 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
        {
            if (estunRDV || confiance > 0.5) {
                out[0] = output[0];
                confs[0] = confiance;
                recalcul = true;
                if (confiance > 0.98)
                {
                    if (output[0] == 'V' || output[0] == 'R')
                    {
                        if ((int)angle == 360)
                            estDroit = true;
                        if ((int)angle == 90)
                            inverse = true;
                    }
                    nonvu = false; // inutile de rechercher une autre position du cadre
                    if (printoption > 1)
                        std::cout << "V1 " << output << " confiance " << confiance << std::endl;
                }
            } else { output = ""; confiance = 0;}
        }
        else if (!estunRDV && output.size() > 0)
        {
            if ((output.size() == 1 && (output[0] > '0' && output[0] <= '9')) || ((output.size() >= 2) && ((output[0] == '1' || output[0] == 'I' || output[0] == 'i') && (output[1] == '0' || output[1] == 'O' || output[1] == '9' || output[1] == 'C' || output[1] == 'U' || output[1] == 'Q' || output[1] == '6'))))
            {
                if (printoption)
                    std::cout << "V1===>" << output << " confiance " << confiance << " angle " << angle << std::endl;
                if (output.size() >= 2)
                {
                    output = "10";
                    if (confiance > 0.9)
                        nonvu = false;
                } // détermination fiable
                if (output == "10")
                    estDroit = true;
                Vcar = output;
                outprec = output;
                vuprec = true;
                confs[0] = confiance;
                out[0] = output; // à recalculer
                recalcul = true;
                // if(estDroit) nonvu = false; // on sait que le caractère est vertical, inutile de tester si on reconnait un caractère horizontal
            }
        }
        if (output.size() == 1 && (output[0] == '<' || output[0] == '>') && confiance > 0.95 && ((int)angle == 360))
        {
            output = "V";
            inverse = true;
            estDroit = false;
            out[4] = "V";
            confs[4] = confiance;
            out[0] = "";
            confs[0] = 0;
            output = "";
            recalcul = true;
            nonvu = false;
        }
        if ((int)angle == 90 && confiance > 0.8)
        {
            inverse = true;
            estDroit = false;
            out[4] = output;
            confs[4] = confiance;
            out[0] = "";
            confs[0] = 0;
            Vcar = "";
            Hcar = output;
            recalcul = true;
            nonvu = false;
        }
        angV = angle;


        // valider le caractère obtenu.
        // chiffre 1 : absence de gros symbole à coté du chiffre ni au centre
        // chiffre 2 ou 3 : absence de gros symbole à coté du chiffre, présence au centre
        // chiffre 4 à 9 ou 10 : présence de gros symbole
        // VDR : dessus à gauche ou dessous à droite : présence de gros symbole
        // VDR : (sinon) pas de gros symbole
        if (output.size() > 0)
        {
            std::string outW = ValiderOCR(output, estServeur, false, moncoin, maconf);
            if (output == "3" && outW != "3") output = "";
            else output = outW;
            outprec = Vcar = out[0] = output;
            if (output == ""){
                confs[0] = 0;
                nonvu = true;
            }
        }
#ifdef ACTIVER        
        if (output.size() > 0)
        {

            bool estGS = false;
            cv::Rect rr;
            cv::Mat GS;
            cv::Scalar m, ect;
            if (output == "V" || output == "D" || output == "R")
            {
                rr.width = 2 * maconf.largeurgrosRDV / 3; // couper la tête des honneurs
                rr.height = maconf.taillegrosRDV;
                if (output == "D")
                    rr.height /= 2; // couper la fleur tenue par la reine de carreau
                if (!estunRDV)
                {
                    if (UU.x > PP.x)
                        rr.x = PP.x + maconf.deltacadre + maconf.deltagrosRDV;
                    else
                        rr.x = PP.x - maconf.deltacadre - maconf.deltagrosRDV - rr.width;
                    if (U.y > PP.y)
                        rr.y = PP.y + maconf.deltacadrehaut + maconf.deltagroshautRDV;
                    else
                        rr.y = PP.y - maconf.deltacadrehaut - maconf.deltagroshautRDV - rr.height;
                }
                else
                {
                    if (UU.x > QQ.x)
                        rr.x = QQ.x + maconf.deltagrosRDV;
                    else
                        rr.x = QQ.x - maconf.deltagrosRDV - rr.width;
                    if (U.y > QQ.y)
                        rr.y = QQ.y + maconf.deltagroshautRDV + 1;
                    else
                        rr.y = QQ.y - maconf.deltagroshautRDV - rr.height - 1;
                }
                if (printoption && !threadoption)
                    tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
                GS = coinPetit(rr).clone();
                // amplifyContrast(GS);
                cv::meanStdDev(GS, m, ect);
                if (ect[0] > 5 + (255 - m[0]) / 5)
                    estGS = true;
                if ((!estGS && ((UU.x > PP.x && U.y > PP.y) || (UU.x < PP.x && U.y < PP.y))) || (estGS && ((UU.x > PP.x && U.y < PP.y) || (UU.x < PP.x && U.y > PP.y))))
                {
                    if ((int)angle == 360)
                    {
                        if (printoption)
                            std::cout << output << " !! caractere incompatible avec gros symbole " << std::endl;
                        out[0] = "";
                        confs[0] = 0;
                        output = "";
                        outprec = "";
                        Vcar = "";
                        nonvu = true;
                    }
                    else if ((int)angle == 0)
                    { // détecté même si caractère tourné de 90 degrés (ex: par trocr)
                        confs[0] = 0;
                        confs[4] = confiance;
                        Vcar = "";
                        Hcar = output;
                        out[4] = output;
                        nonvu = false;
                        inverse = true;
                    }
                }
            }
            else
            { // chiffre
                // 1 2 ou 3 : pas de gros symbole
                // 5 : le caractère 3 est parfois détecté comme un 5
                // 7 : le caractère 1 est parfois détecté comme un 7
                // autre (10 ou 4 à 9) : présence de gros symbole
                rr.width = maconf.largeurgros; // éviter le gros symbole central du 1 2 3
                rr.height = maconf.taillegros;
                if (output == "1" || output == "2" || output == "3" || output == "5" || output == "7")
                {
                    rr.width /= 2;
                    //rr.height /= 2;
                }
                if (UU.x > PP.x)
                    rr.x = PP.x + maconf.deltagros;
                else
                    rr.x = PP.x - maconf.deltagros - rr.width;
                if (U.y > PP.y)
                    rr.y = PP.y + maconf.deltagroshaut;
                else
                    rr.y = PP.y - maconf.deltagroshaut - rr.height;
                GS = coinPetit(rr).clone();
                // amplifyContrast(GS);
                cv::meanStdDev(GS, m, ect);
                if (ect[0] > 30 + (255 - m[0]) / 5)
                    estGS = true;
                if ((estGS && (output == "1" || output == "2" || output == "3")) || (!estGS && output > "3" && output <= "9"))
                {
                    if (printoption)
                        std::cout << output << " !! caractere incompatible avec gros symbole " << std::endl;
                    // if (output == "7"   ) output = "1";
                    //  else if (output == "3") output = "5"; // peut-être  8
                    // else
                    if (estGS && output == "1")
                        output = "4";               // pourrait etre 10 !!!
                    else if (!estGS && output == "4")
                        output = "1";
                    else if (!estGS && output == "5")
                        output = "3";
                    else if (estGS && output == "3")  // pourrait être 8
                        output = "5";
                    else if (!estGS && output == "8")
                        output = "3";
                    else
                    {
                        output = "";
                    }
                    out[0] = output;
                    outprec = output;
                    Vcar = output;
                    if (output == "" && (int)angle != 90)
                    {
                        confs[0] = 0;
                        nonvu = true;
                    }
                }
            }
        }
#endif        
    }
    //////////////////////////////////////////////////////////////////////////////
    // si l'orientation n'est pas déterminée, verticale ou horizontale , tester le caractère horizontal
    /////////////////////////////////////////////////////////////////////////////////
    if (!estDroit)
    { // on ne sait pas si le caractère est vertical ou horizontal
        if (printoption > 1)
            std::cout << "H?" << std::endl;
        // tester si le caractère est horizontal
        dx = BB.x - AA.x + 2; // test expérimental
        dy = BB.y - AA.y + 1;
        xg = AA.x;
        if (xg == PP.x)
            xg++;
        if (xg + dx == PP.x)
            xg--;
        yh = AA.y;
        if (yh == PP.y)
            yh++;
        if (yh + dy == PP.y)
            yh--;
        ls = dx;
        ts = dy;
        ts = maconf.largeurchiffre;
        ls = maconf.taillechiffre;
        if (estunRDV)
        {
            ts = maconf.largeurVDR;
            ls = maconf.tailleVDR;
        }
        // provoque des erreurs : désactivé :
#ifdef INACTIVE
        if (!estunRDV)
        {
            // zone large et haute, mais à au moins 2 pixels du bord de carte sur le coté
            int dc = maconf.deltachiffre / 2;
            dx = maconf.taillechiffre + dc;
            if (UU.x > PP.x)
                xg = std::max(1 + PP.x, AA.x - dc);
            else
                xg = std::min(PP.x - dx - 1, AA.x - dc);
            dy = maconf.largeurchiffre + maconf.deltachiffre;
            if (U.y > PP.y)
                yh = std::max(2 + PP.y, AA.y - dc);
            else
                yh = std::min(PP.y - dy - 2, AA.y - dc);
        }
        else
        {
            dx = dx - 1 + maconf.deltaVDR;
            if (UU.x > PP.x)
            {
                xg = xg + 1 - maconf.deltaVDR;
            }
        }
#endif
        if (estunRDV)
            dy++;
        if (estunRDV && UU.y < PP.y)
            yh--;
        // extraire le caractère
        r = cv::Rect(xg, yh, dx, dy);
        // le haut du symbole peut être en xg ou xg+dx selon orientation
        // on identifie le problème lorsque la ligne au dessus (ou au dessous) est plus claire
        // extraire les deux lignes et comparer
        // ssx_91111  |    ou |  11119_xss
        // ssx_91111  |       |  11119_xss
        cv::Rect rr;
        cv::Scalar m1, m2, m3;
        int dbleu = 10; // à déterminer : écart de bleu  entre ligne  rouge et blanche
        rr.y = yh;
        rr.height = dy;
        rr.width = 1;
        if (UU.x > PP.x)
        { // à droite, colonnes  en  9 (xg+dx-1) et _ (xg+dx)
            // |  11119_xss
            rr.x = xg + dx - 1; // dernière colonne (9)
            cv::Mat lig = coinPetit(rr);
            m1 = mean(lig);
            rr.x++; // ligne (_)
            lig = coinPetit(rr);
            m2 = mean(lig);
            if (m2[0] > m1[0] + dbleu)
                r.width++; // ajouter la ligne blanche entre chiffre et symbole
            else
            {           // la ligne (_) est la fin du chiffre ou le début du symbole
                rr.x++; // ligne (x)
                lig = coinPetit(rr);
                m3 = mean(lig);
                if (m3[0] > m2[0] + dbleu)
                    r.width += 2;
                else
                {
                    // TODO : ligne (9) précédée par une ligne claire?
                }
            }
        }
        else
        { // à gauche, colonnes en xg et xg+1
            // ssx_9111 |
            rr.x = xg; // première colonne (9)
            cv::Mat lig = coinPetit(rr);
            m1 = mean(lig);
            rr.x--; // colonne précédente (_)
            lig = coinPetit(rr);
            m2 = mean(lig);
            if (m2[0] > m1[0] + dbleu)
            {
                r.x--;
                r.width++;
            }
            else
            {
                // la colonne (_) est la fin  du caractère ou le début du symbole
                rr.x--; // colonne (x)
                lig = coinPetit(rr);
                m3 = mean(lig);
                if (m3[0] > m2[0] + dbleu)
                {
                    r.width++;
                    r.x -= 2;
                }
            }
        }
        if (r.x < 0) r.x = 0; if (r.y < 0) r.y = 0;
        if (r.width > coinPetit.cols - r.x) r.width = coinPetit.cols - r.x;
        if (r.height > coinPetit.rows - r.y) r.height = coinPetit.rows - r.y;
        // extraire le caractère
        ima_car = coinPetit(r).clone();
        int Box[4] = {0, 0, 0, 0};
        // déterminer l'encombrement
        calculerBox(ima_car, ts, ls, moy, Box, moyext, maconf);
        r.x += Box[0];
        r.width = Box[1] - Box[0] + 1;
        r.y += Box[2];
        r.height = Box[3] - Box[2] + 1;
        ima_car = coinPetit(r).clone();
        if (B.x > PP.x)
            cv::rotate(ima_car, ima_car, cv::ROTATE_90_CLOCKWISE);
        else
            cv::rotate(ima_car, ima_car, cv::ROTATE_90_COUNTERCLOCKWISE);
        // ima_carW = ima_car.clone();
        cv::cvtColor(ima_car, ima_carW, cv::COLOR_BGR2GRAY);
        cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche ( 1 ou 2 pixels, 1 seul si le caractère est petit)
        {
            cv::Mat image_bordee;
            int tb = 2;
            if (ts <= 6)
                tb = 1;
            cv::copyMakeBorder(ima_carW, image_bordee, tb, tb, tb, tb, cv::BORDER_CONSTANT, cv::Scalar(255));
            ima_carW = image_bordee;
        }
        ima_CARH = ima_carW.clone();
        // if (dx < 20) cv::resize(ima_carW, ima_carW, cv::Size(), 8.0, 8.0);
        // amplifyContrast(ima_carW);
        if (printoption > 1 && !threadoption){
            afficherImage("H1c", ima_car);
            afficherImage("H1", ima_CARH);
        }
        if (printoption && !threadoption){
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
        }
        std::string outRDV;
        if (maconf.tesOCR >= 1) output = tesOCR(ima_carW, estunRDV, &confiance, &angle);
        else                    output = execOCR(nomOCR, ima_carW, &confiance, &angle);
        if (output == "?")  output = "";
        if (output == "M")  output = "V"; // bord du cadre V et un morceau de gros symbole
        if (printoption && output.size() > 0)
            std::cout << "H1 " << output << " confiance " << confiance << " angle " << angle << std::endl;
        bool testerVDR = !estunRDV;
        if (output.size() > 0)
        {
            // accepter 10  V et R si la confiance est suffisante
            if ((output == "10" || output == "R" || output == "V") && confiance > 0.9)
                testerVDR = false;
            if (confiance > 0.99 && output[0] >= '1' && output[0] <= '9')
                testerVDR = false;
            if (!testerVDR)
            {
                if ((int)angle != 90)
                {
                    confs[4] = confiance;
                    out[4] = output;
                }
            }
        }

        if (testerVDR)
        {
            double confRDV;
            double angRDV;
            r.width = maconf.tailleVDR + maconf.deltachsymb + 2; // rapproche du symbole
            // agrandir la zone en hauteur sauf s'il y a un gros symbole
            if (((UU.x > PP.x) && (U.y < PP.y)) || ((UU.x < PP.x) && (U.y > PP.y)))
                 r.height = maconf.largeurVDR + maconf.deltachsymb; // !! gros symbole
            else r.height = maconf.largeurVDR + 2 * maconf.deltachsymb; // agrandir la zone

            if (UU.x > PP.x) r.x = PP.x + maconf.deltahautVDR /* + 1*/;
            else             r.x = PP.x - maconf.deltahautVDR - r.width;
            if (U.y > PP.y)  r.y = PP.y + maconf.deltacadre /* + maconf.deltaVDR */;
            else             r.y = PP.y - maconf.deltacadre - r.height /* - maconf.deltaVDR */;
            cv::Mat ima_RDV = coinPetit(r).clone();
            ts = maconf.largeurVDR;
            ls = maconf.tailleVDR;
            calculerBox(ima_RDV, ts, ls, moy, Box, moyext, maconf);
            r.x += Box[0];
            r.y += Box[2];
            r.width = ls;
            r.height = ts;
            ima_RDV = coinPetit(r).clone();
            // eclaircirfond(ima_RDV);
            // amplifyContrast(ima_RDV);
            if (UU.x > PP.x)  cv::rotate(ima_RDV, ima_RDV, cv::ROTATE_90_CLOCKWISE);
            else              cv::rotate(ima_RDV, ima_RDV, cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::cvtColor(ima_RDV, ima_carW, cv::COLOR_BGR2GRAY);
            cv::threshold(ima_carW, ima_carW, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
            
            if (printoption && !threadoption) {
                if (printoption > 1) afficherImage("H1X", ima_carW);
                if (waitoption > 2) cv::waitKey(0); else cv::waitKey(1);
            }
            if (maconf.tesOCR >= 1) outRDV = tesOCR(ima_carW, estunRDV, &confRDV, &angRDV);
            else                    outRDV = execOCR(nomOCR, ima_RDV, &confRDV, &angRDV);
            if (printoption)
                std::cout << "   OCR H1 pour RDV " << outRDV << " confiance " << confRDV << " angle " << angRDV << std::endl;
            if (outRDV == "?") outRDV = "";
            if (confRDV < confiance + 0.2) outRDV = "";
            if (outRDV == "i)" || outRDV == "1)")   outRDV = "D";
            if (outRDV == "v")    outRDV = "V";
            if ( !estunRDV && (outRDV == "ID" || outRDV == "1D" || outRDV == "|D")) outRDV = "10";
            if (outRDV.size() >= 2 && (outRDV[1] == 'R' || outRDV[1] == 'D' || outRDV[1] == 'V'))
                outRDV = outRDV[1];
            // ignorer si ce n'est pas un R D V
            if (outRDV.size() > 0)
            {
                if (outRDV == "M") outRDV = "V";
                if (outRDV[0] == 'V' || outRDV[0] == 'D' || outRDV[0] == 'R')
                {
                    if (output[0] == 'R' && outRDV[0] == 'D')
                        outRDV = "R"; // R reconnu est mieux que D
                    if (confRDV > confiance)
                    {
                        output = outRDV;
                        confiance = confRDV;
                        angle = angRDV;
                        if (angle == 360 && confiance > 0.8) { inverse = true; nonvu = false; }
                    }
                }
            }
        } // OCR RDV H1
        if ((int)angV == 360 && angle != 360)
        {
            output = "";
            confs[4] = 0;
            out[4] = "";
        }
        if ((int)angle == 360 && confiance > confs[0] && (int)angV != 360)
        {
            confs[0] = 0;
            out[0] = "";
        }
        if (output.size() == 0 && out[0] != "" && confs[0] > 0.95)
            nonvu = false; // inutile de tester les décalages
        if ( !estunRDV && (output == "ID" || output == "1D" || output == "|D")) output = "10";
        if (output.size() >= 2 && (output[1] == 'V' || output[1] == 'D' || output[1] == 'R'))
            output = output[1];
        if (output.size() >= 2 &&
            (output[0] == 'V' || output[0] == 'v' || output[0] == 'D' || output[0] == 'R'))
            output = output[0];
        if (output == "v") output = "V";
        if (output == "M") output = "V"; // bord du cadre V et un morceau de gros symbole
                          // normalement le caractère V a été détecté lors de l'analyse du caractère vertical
        /*if ((output == "<" || output == ">") && ((int)angle == 360)
                && out[0] != "R" && out[0] != "D" && confiance > confs[0]) {
            output = "V";
            out[0] = "V";
            out[4] = "";
            confs[0] = confiance;
            estDroit = true;
            inverse = false;
            confs[4] = 0;
        }*/

        if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
        {
            if (!estDroit && (int)angle != 90)
            {
                if (estunRDV || confiance > 0.5) {
                    out[4] = output[0];
                    Hcar = out[4];
                    confs[4] = confiance;
                    if ((Hcar == out[0]))
                        nonvu = false; // conforme à la détection verticale

                    recalcul = true; // inutile, rien ne changerait
                    if (confiance > 0.65 || (out[4] == out[0] && confs[4] > confs[0] && confs[0] > 0.8))
                    {
                        if (printoption > 1)
                            std::cout << "horiz 1 " << output << " confiance " << confiance << std::endl;
                        output = output[0];
                        recalcul = true;
                        if ((Vcar == "") || (Vcar == Hcar))
                            nonvu = false; // on vient de trouver ou conforme à la détection verticale
                        if (confiance > confs[0])
                        {
                            nonvu = false;
                            inverse = true;
                        }
                    }
                } // V D R attendu ou confiance assez bonne
            }
            if ((int)angle == 90) // ce cas ne se produit plus après adaptation tesseract c++
            {
                if (out[0] == "")
                {
                    out[0] = output;
                    confs[0] = confiance;
                    Vcar = output;
                    confs[4] = 0;
                }
            }
        } // lu V D R
        else if (!estunRDV)
        {
            if ((output.size() == 1 && (output[0] > '0' && output[0] <= '9'))
             || ((output.size() == 2 || output.size() == 3) 
                && (output[0] == '1' || output[0] == 'I' || output[0] == 'i' || output[0] == '4')
                && (output[1] == '0' || output[1] == 'O' 
                    || output[1] == 'C' || output[1] == '1' || output[1] == '9' 
                    || output[1] == 'U' || output[1] == 'Q' || output[1] == '6')))
            {
                if (output.size() >= 2)  output = "10";
                if (printoption > 1)
                    std::cout << "H1===>" << output << " confiance " << confiance << " angle " << angle << std::endl;
                if ((int)angle == 90)
                {
                    if (out[0] == "")
                    {
                        out[0] = output;
                        confs[0] = confiance;
                        Vcar = output;
                        confs[4] = 0;
                    }
                } else {
                    Hcar = output;
                    out[4] = output;
                    confs[4] = confiance;
                    recalcul = true;
                    if (output == "10") { inverse = true; nonvu = false; }
                    if ((out[0] == "V" || out[0] == "D" || out[0] == "R") && confs[0] > 0.8)
                        nonvu = false;
                    if ((!vuprec || Hcar == Vcar) && (confiance > 0.99))
                            nonvu = false; // chiffre fiable confirmé horizontal ou vertical ?
                    if (out[0] == "10" && confs[0] > 0.8 && confs[4] < 0.5)
                        nonvu = false; // inutile de chercher un décalage
                    if (out[0] == "10" && confs[0] > 0.99)
                        nonvu = false; // inutile de chercher un décalage
                }
            }
            else if (output[0] > '0' && output[0] <= '9')
            {
                Hcar = output = output[0];
                out[4] = output;
                confs[4] = confiance;
                recalcul = true;
            }
            else
            { // ni un chiffre ni un VDR
                if (out[0] != "" && confs[0] > 0.9)
                    nonvu = false;
            }
        }
        if (output.size() > 0) {
            std::string outW = ValiderOCR(output, false, true, moncoin, maconf);
            if (output == "3" && outW != "3") output = "";
            else output = outW;
        }
            Hcar = out[4] = output;
        if (output.size() == 0 ){
            confs[4] = 0;
            if (out[0] == "")
                nonvu = true;
        }
#ifdef ACTIVER
        // valider le caractère obtenu.
        // chiffre 1 2 ou 3 : absence de gros symbole à coté du chiffre
        // chiffre 4 à 9 ou 10 : présence de gros symbole
        // VDR : dessus à gauche ou dessous à droite : pas de gros symbole
        // VDR : présence de grosc symbole
        if (output.size() > 0 && (int)angle != 90)
        {
            bool estGS = false;
            cv::Rect rr;
            cv::Mat GS;
            cv::Scalar m, ect;
            if (output == "V" || output == "D" || output == "R")
            {
                rr.height = 2 * maconf.largeurgrosRDV / 3; // couper la tête du roi Dame Valet
                rr.width = maconf.taillegrosRDV;
                // if (output == "D" && ((UU.x > PP.x && U.y < PP.y) || (UU.x < PP.x && U.y > PP.y)))
                //    rr.width /= 2; // couper la fleur de la reine de carreau
                if (estunRDV)
                {
                    if (UU.x > QQ.x) rr.x = QQ.x + maconf.deltagroshautRDV;
                    else             rr.x = QQ.x - maconf.deltagroshautRDV - rr.width;
                    if (U.y > QQ.y)  rr.y = QQ.y + maconf.deltagrosRDV;
                    else             rr.y = QQ.y - maconf.deltagrosRDV - rr.height;
                }
                else
                {
                    if (UU.x > PP.x) rr.x = PP.x + maconf.deltacadrehaut + maconf.deltagroshautRDV + 1; // cadre
                    else             rr.x = PP.x - maconf.deltacadrehaut - maconf.deltagroshautRDV - rr.width - 1;
                    if (U.y > PP.y)  rr.y = PP.y + maconf.deltacadre + maconf.deltagrosRDV;
                    else             rr.y = PP.y - maconf.deltacadre - maconf.deltagrosRDV - rr.height;
                }
                if (printoption && !threadoption)
                    tracerRectangle(rr, extrait, "valider", cv::Scalar(255, 0, 0));
                GS = coinPetit(rr).clone();
                // amplifyContrast(GS);
                cv::meanStdDev(GS, m, ect);
                if (ect[0] > 5 + (255 - m[0]) / 10) estGS = true;
                if ((!estGS && ((UU.x > PP.x && U.y < PP.y) || (UU.x < PP.x && U.y > PP.y))) || (estGS && ((UU.x > PP.x && U.y > PP.y) || (UU.x < PP.x && U.y < PP.y))))
                {
                    if (printoption)
                        std::cout << output << " !! incompatible avec gros symbole " << std::endl;
                    out[4] = "";
                    confs[4] = 0;
                    output = "";
                    outprec = "";
                    Hcar = "";
                    if (out[0] == "")
                        nonvu = true;
                }
            }
            else
            { // chiffre
                // 1 2 ou 3 : pas de gros symbole
                // 5 7 : parfois détectés par erreur pour 3 ou 1
                // autre (10 ou 4 à 9) : présence de gros symbole
                rr.height = maconf.largeurgros;
                rr.width = maconf.taillegros;
                if (output == "2" || output == "3" || output == "5" || output == "7")
                {
                    rr.height /= 2;
                    rr.width /= 2;
                }
                // eviter gros symbole central pour 2 ou 3
                if (UU.x > PP.x) rr.x = PP.x + maconf.deltagroshaut + 1;
                else             rr.x = PP.x - maconf.deltagroshaut - rr.width - 1;
                if (U.y > PP.y)  rr.y = PP.y + maconf.deltagros;
                else             rr.y = PP.y - maconf.deltagros - rr.height;
                if (rr.x < 0) {rr.width += rr.x; rr.x = 0; if (rr.width < 1) rr.width = 1;}
                if (rr.y < 0) {rr.height += rr.y; rr.y = 0; if (rr.height < 1) rr.height = 1;}
                if (rr.width > coinPetit.cols - rr.x) rr.width = coinPetit.cols - rr.x;
                if (rr.height > coinPetit.rows - rr.y) rr.height = coinPetit.rows - rr.y;
                GS = coinPetit(rr).clone();
                // amplifyContrast(GS);
                cv::meanStdDev(GS, m, ect);
                if (ect[0] > 30 + (255 - m[0]) / 5)
                    estGS = true;
                if ((estGS && (output == "1" || output == "2" || output == "3")) || (!estGS && output > "3" && output <= "9"))
                {
                    if (printoption)
                        std::cout << "!! chiffre " << output << " incompatible avec gros symbole " << std::endl;
                    // if (output == "7"   ) output = "1"; // peut-être 3 ou 5
                    // else if (output == "3") output = "5"; // peut être 8
                    // else
                    if (estGS && output == "1") output = "4";  // peut être 10 !!
                    else if (!estGS && output == "4") output = "1";
                    else if (!estGS && output == "5") output = "3";
                    else if (!estGS && output == "8") output = "3";
                    else  output = "";

                    out[4] = output;
                    outprec = output;
                    Hcar = output;
                    
                    if (output == "")
                    {
                        confs[4] = 0;
                        if (out[0] == "")
                            nonvu = true;
                        if (printoption && !threadoption)
                            tracerRectangle(rr, extrait, "Extrait", cv::Scalar(255, 0, 0));
                    }
                }
            }
        }
#endif        
    }

    if (confs[4] > confs[0] || out[0] == "") output = out[4];
    else                                     output = out[0];
    if (output == "V" || output == "D" || output == "R")
        if (!estunRDV || confiance > 0.5) estunRDV = true;
    // si on a trouvé en vertical et/ou horizontal, avec une confiance très élevée,
    // on en déduit l'orientation et il est inutile de chercher les autres dispositions
    if (std::max(confs[0], confs[4]) > 0.9)
    {
        nonvu = false;
        if (confs[0] > confs[4]) estDroit = true;
        inverse = !estDroit;
    }

    // si on n'a rien trouvé ou avec faible confiance et si tesseract est sélectionné,
    //  essayer la recherche avec le serveur
    if (maconf.tesOCR == 1 && (output == "" || (confs[0] < 0.70 && confs[4] < 0.70)))
    {
        estServeur = true;
        output = "";
        if (!inverse) output = execOCR("SERVEUR", ima_CARV, &confiance, &angle);
        if (output == "")
            if (!estDroit) output = execOCR("SERVEUR", ima_CARH, &confiance, &angle);
        if ( !estunRDV && (output == "ID" || output == "1D" || output == "|D")) output = "10";
        if (output.size() > 0){
            if (output == "B") {
                if (estunRDV) output = "R";
                else output = "8";
            }
            std::string outW = ValiderOCR(output, estServeur, inverse, moncoin, maconf);
            if (output == "3" && outW != "3") output = "";
            else output = outW;
        }
        if (printoption) std::cout<<" serveur : "<< output<<" confiance "<< confiance<<std::endl;
#ifdef ACTIVER
        if (output.size() > 0)
        {
            char w = output[0];
            if (estunRDV && w != 'V' && w != 'D' && w != 'R')
                output = "";
            else if (w != 'V' && w != 'D' && w != 'R' && !(w > '0' && w <= '9') && output != "10")
                output = "";
            // 11 à 19 : on a confondu le bord de carte avec le chiffre 1
            if (output.size() == 2 && w == '1' && output[1] > '0' && output[1] <= '9') output = output[1];
        }
#endif        
        if (output != "")
        {
            if (output == out[0])
            { // confirmer la détection tesseract
                if (confiance > confs[0])   confs[0] = confiance;
                else                        confiance = confs[0];
                confs[4] = 0;
                estServeur = false;
            }
            else if (output == out[4])
            { // confirmer la détection inverse
                if (confiance > confs[4])   confs[4] = confiance;
                else                        confiance = confs[4];
                confs[0] = 0;
                estServeur = false;
            }
            else { // résultat du serveur différent
                // si Tesseract à trouvé une valeur, on l'a validée ou corrigée
                // le plus fréquent est la confusion entre 1 et 4
                if (inverse && out[4] == "4" && output == "1") output = "4";
                else if (inverse && out[4] == "1" && output == "4") output = "1";
                if (!inverse && out[0] == "4" && output == "1") output = "4";
                else if (!inverse && out[0] == "1" && output == "4") output = "1";

                if (out[0] != "" && confiance > confs[0])  {out[0] = output; confs[0] = confiance;}
                if (out[4] != "" && confiance > confs[4])  {out[4] = output; confs[4] = confiance;}
            }
        }
        if (output != "") {
            // TODO valider la valeur trouvée selon la présence d'un gros symbole
            if (printoption)
                std::cout << "==> serveur " << output << " confiance " << confiance << std::endl;
            std::string outW = ValiderOCR(output, estunRDV, inverse, moncoin, maconf);
            if (output == "3" && outW != "3") output = "";
            else output = outW;
            if (output.size() > 0){
                moncoin.caractere = output[0];
                if (output == "10") moncoin.caractere = 'X';
            }
            else std::cout<<" !! invalide!!"<<std::endl;
        }
        if (confiance < 0.5)
        {
            if (printoption)
                std::cout << "!!! " << output << " confiance trop faible " << confiance << std::endl;
            output = "";
            out[0] = out[4] = "";
        }
    }

    // tester les autres possibilités verticales décalées (par expérience,  ceci arrive rarement)
#ifdef ACTIVER
    // tester les autres possibilités verticales décalées (par expérience,  ceci arrive rarement)
    if (!inverse && !estDroit)
    { // caractère vertical ou pas encore déterminé

        if (nonvu && !cadreX)
        { // essayer le cas où le coin gauche ou droit est le cadre
            dx = B.x - A.x + 1;
            dy = B.y - A.y + 1;
            xg = A.x;
            yh = A.y;
            if (UU.x > PP.x) // à droite, décaler de deltacadre à gauche
                xg -= dc;
            else
                xg += dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            if (U.y < PP.y)
                cv::rotate(ima_car, ima_car, cv::ROTATE_180);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("V2", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "V2 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[1] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[1] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[1] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];

            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[1] = output[0];
                recalcul = true;
                if (confiance > 0.99)
                {
                    nonvu = false; // détection jugée fiable selon la confiance
                    // on a trouvé en décalant . donc décaler aussi U V A B et calculer QQ
                    Vcar = output;
                }
            }
        }
        if (nonvu && !cadreX)
        {
            // cas où les deux bords du coin sont le cadre
            if (U.y < PP.y) // au dessus, décaler de deltacadre vers le bas
                yh += dc;
            else
                yh -= dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            if (U.y < PP.y)
                cv::rotate(ima_car, ima_car, cv::ROTATE_180);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("V3", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCRVDR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "V3 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[2] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[2] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[2] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];

            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[2] = output[0];
                if (confiance > 0.99)
                {
                    nonvu = false; // on estime avoir trouvé avec ce niveau de confiance
                    recalcul = true;
                    Vcar = output;
                }
            }
        }
        if (nonvu && !cadreX)
        {
            // dernier cas le bord horizontal du coin est le cadre
            if (UU.x > PP.x) // à droite, décaler de deltacadre à gauche
                xg += dc;    // on revient à l'état initial
            else
                xg -= dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            if (U.y < PP.y)
                cv::rotate(ima_car, ima_car, cv::ROTATE_180);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("V4", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCRVDR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "V4 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[3] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[3] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[3] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];

            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[3] = output[0];
                if (confiance > 0.99)
                {
                    nonvu = false;
                    recalcul = true;
                    Vcar = output;
                }
            }
        }
        if (!nonvu)
        {
            if (printoption > 0)
                std::cout << output << " confiance " << std::max(confs[0], confs[4]) << std::endl;
            vuprec = true;
            outprec = output;
        }
    }
    //////////////////////////// tester les autres positions horizontales par appel de l' OCR /////////////////
    if ((nonvu || confiance < 0.7) && !estDroit && !inverse)
    {
        dx = abs(BB.x - AA.x);
        dy = abs(BB.y - AA.y);
        xg = std::min(AA.x, BB.x);
        yh = std::min(AA.y, BB.y);
        if (estunRDV)
            dy++;
        if (estunRDV && UU.y < PP.y)
            yh--;

        if (nonvu && !cadreX)
        {
            // essayer le cas où le coin gauche ou droit est le cadre
            if (UU.x > PP.x) // à droite, décaler de deltacadre à gauche
                xg -= dc;
            else
                xg += dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            // cv::imwrite(nomcoin, ima_car);
            if (B.x > PP.x)
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_CLOCKWISE);
            else
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_COUNTERCLOCKWISE);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("H2", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCRVDR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "H2 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[5] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[5] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[5] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];
            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[5] = output[0];
                recalcul = true;
                if (confiance > 0.99)
                {
                    nonvu = false; // détection jugée fiable
                    Hcar = output;
                }
            }
        }
        if ((nonvu || confiance < 0.7) && !cadreX && !cadreY)
        {
            // cas où les deux bords du coin sont le cadre
            if (U.y < PP.y) // au dessus, décaler de deltacadre vers le bas
                yh += dc;
            else
                yh -= dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            // cv::imwrite(nomcoin, ima_car);
            if (B.x > PP.x)
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_CLOCKWISE);
            else
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_COUNTERCLOCKWISE);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("H3", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCRVDR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "H3 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[6] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[6] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[6] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];

            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[6] = output[0];
                if (confiance > 0.99)
                {
                    nonvu = false;
                    recalcul = true;
                    Hcar = output;
                }
            }
        }
        if ((nonvu || confiance < 0.7) && !cadreY)
        {
            // dernier cas le bord horizontal du coin est le cadre
            if (UU.x > PP.x) // à droite, décaler de deltacadre à gauche
                xg += dc;    // on revient à l'état initial
            else
                xg -= dc;
            cv::Rect r(xg, yh, dx, dy);
            ima_car = coinPetit(r).clone();
            // cv::imwrite(nomcoin, ima_car);
            if (B.x > PP.x)
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_CLOCKWISE);
            else
                cv::rotate(ima_car, ima_car, cv::ROTATE_90_COUNTERCLOCKWISE);
            ima_carW = ima_car.clone();
            if (dx < 20)
                cv::resize(ima_carW, ima_carW, cv::Size(), 4.0, 4.0);
            if (printoption > 1 && !threadoption)
                afficherImage("H4", ima_carW);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
            output = execOCRVDR(nomcoin, ima_car, &confiance, &angle);
            if (printoption > 1)
                std::cout << "H4 " << output << " confiance " << confiance << std::endl;
            if (output.size() > 0)
                confs[7] = confiance;
            if (output.size() == 2 && (output == "IV" || output == "ID" || output == "IR"))
                out[7] = output[1];
            if (output.size() == 2 && (output == "VI" || output == "DI" || output == "RI"))
                out[7] = output[0];
            if (output.size() == 2 && (output == "VC" || output == "DC" || output == "RC"))
                output = output[0];
            if (output.size() == 2 && (output == "VA" || output == "DA" || output == "RA"))
                output = output[0];

            if (output.size() == 1 && (output[0] == 'V' || output[0] == 'D' || output[0] == 'R'))
            {
                out[7] = output[0];
                if (confiance > 0.99)
                {
                    Hcar = output;
                    recalcul = true;
                    nonvu = false;
                }
            }
        }
    }
#endif
    // si on a trouvé IR IV ID ou RI DI ou VI OU ...
    //   sélectionner le meilleur candidat ( meilleur indice de confiance )
    // if (nonvu)

    k = -1; // rechercher la meilleure détection
    // 1er OCR vertical ou horizontal
    if (!inverse)
        if ((out[0] == "V" || out[0] == "D" || out[0] == "R") && confs[0] > 0.4)
            k = 0;
    if (!estDroit)
        if ((out[4] == "V" || out[4] == "D" || out[4] == "R") && confs[4] > std::max(0.4, confs[0]))
            k = 4;
    if (!inverse)
    {
        if (out[0] != "" && confs[0] > 0.8)
            k = 0;
        if (out[0] != "")
            Vcar = out[0];
    }
    else
    {
        if (out[4] != "" && confs[4] > 0.8)
            k = 4;
        if (out[4] != "")
            Hcar = out[4];
    }
    if (!estDroit)
    {
        if (out[4] != "" && (confs[4] > std::max(0.4, confs[0]) || out[0] == ""))
            k = 4;
        if (out[4] != "")
            Hcar = out[4];
    }
    // parfois, on détecte un caractère D alors que c'est un R qui a été détecté avec une moins bonne confiance
    if (out[0] == "R" && confs[0] > 0.4 && out[4] == "D")
        k = 0;
    if (out[4] == "R" && confs[4] > 0.4 && out[0] == "D")
        k = 4;
    kk = k;
    // les 6 autres OCR:
    for (int i = 1; i < 8; i++)
    {
        if (i == 4)
            continue;
        if (out[i] == "")
            continue;
        if (i < 4 && Vcar == "" && out[i] != "")
            Vcar = out[i];
        else if (Hcar == "" && out[i] != "")
            Hcar = out[i];
        if (confs[i] > confs[k])
        {
            // privilégier le premier OCR Vertical ou Horizontal
            //    le coin est généralement formé par les bords de carte
            // si c'est un chiffre et i différent de 0 et 4
            // si out[0] = V D ou R et confs[0] > 0.5, ne pas modifier
            k = i;
            if (out[i] == "10" || (out[i] > "0" && out[i] <= "9"))
            {
                if (kk >= 0)
                {
                    if (out[kk] == "R" && confs[kk] > 0.4)
                        k = kk;
                    else if (out[i] == out[kk] && confs[kk] > 0.5)
                        k = kk;
                }
            }
            else if ((out[i] == "V" || out[i] == "D" || out[i] == "R"))
            {
                if (kk >= 0)
                {
                    if ((out[kk] == "V" || out[kk] == "D" || out[kk] == "R") && confs[kk] > 0.5)
                        k = kk;
                    if (out[kk] == out[i] && confs[kk] > 0.5)
                        k = kk;
                }
            }
        }
        else if (k < 0 && confs[i] > 0.4)
        {
            k = i;
        } // il faut une confiance minimale
    }
    if (k >= 0)
    { // on a trouvé un chiffre ou caractère V D R
        if (out[k] == "10" || (out[k] >= "1" && out[k] <= "9"))
            estunRDV = false;
        confiance = confs[k];
        output = out[k];
        outprec = output;
        nonvu = false;
        moncoin.caractere = output[0];
        if (output == "10")
            moncoin.caractere = 'X';
        if (printoption > 0)
            std::cout << "=======>" << output << " confiance " << confiance << std::endl;
        if (k < 4)
        {
            estDroit = true;
            inverse = false;
            Vcar = output;
            Hcar = "";
        }
        else
        {
            estDroit = false;
            inverse = true;
            Vcar = "";
            Hcar = output;
        }
        if (output == "V" || output == "D" || output == "R")
        {
            recalcul = true; // recalculer les positions de A B U V AA ...
            cadreX = true;
            cadreY = true;
            // recalculer les positions du coin (PP) et du cadre (QQ) selon la position détectée
            if (k == 1 || k == 5)
            {
                QQ.x = PP.x;
                if (UU.x > PP.x) PP.x -= dc;
                else             PP.x += dc;
            }
            else if (k == 2 || k == 6)
            {
                QQ = PP;
                if (UU.x > PP.x) PP.x -= dc;
                else       PP.x += dc;
                if (U.y > PP.y) PP.y -= dc;
                else            PP.y += dc;
            }
            else if (k == 3 || k == 7)
            {
                QQ.y = PP.y;
                if (U.y > PP.y) PP.y -= dc;
                else            PP.y += dc;
            }
            else if (!etaitRDV)
            {
                if (inverse)
                {
                    if (UU.x > PP.x) QQ.x = PP.x + maconf.deltacadrehaut;
                    else             QQ.x = PP.x - maconf.deltacadrehaut;
                    if (U.y > PP.y)  QQ.y = PP.y + maconf.deltacadre;
                    else             QQ.y = PP.y - maconf.deltacadre;
                }
                else
                {
                    if (UU.x > PP.x) QQ.x = PP.x + maconf.deltacadre;
                    else             QQ.x = PP.x - maconf.deltacadre;
                    if (U.y > PP.y)  QQ.y = PP.y + maconf.deltacadrehaut;
                    else             QQ.y = PP.y - maconf.deltacadrehaut;
                }
            }
            estunRDV = true;
        } // V D ou R
        else
        { // c'est un chiffre
            // recalculer seulement la position de PP
            if (k == 1 || k == 5)
            {
                if (UU.x > PP.x) PP.x -= dc;
                else             PP.x += dc;
            }
            else if (k == 2 || k == 6)
            {
                if (UU.x > PP.x) PP.x -= dc;
                else             PP.x += dc;
                if (U.y > PP.y)  PP.y -= dc;
                else             PP.y += dc;
            }
            else if (k == 3 || k == 7)
            {
                if (U.y > PP.y) PP.y -= dc;
                else            PP.y += dc;
            }
            if (k != 0 && k != 4) recalcul = true;
        }
        if (k != 0 && k != 4)
        { // la position du coin change
            // invalider l'orientation et la couleur
            if (printoption > 0)
                std::cout << " !! invalide  orientation et couleur" << std::endl;
            inverse = false;
            estDroit = false;
            estRouge = false;
            estNoir = false;
            recalcul = true;
        }
    }

    // artefact détecté près d'un bord, sauf chiffre 10 (près du bord)
    if (estArtefact && output != "10")
    {
        cv::circle(result, cv::Point2i(cecoin[4], cecoin[5]), 2, cv::Scalar(255, 0, 0), -1);
        if (printoption)
            std::cout << "Artefact " << std::endl;
        // if (maconf.deltacadre > 10 || confiance < 0.8)  // ne pas laisser de fausse détection
            {retourcoin(n); return;}  // sinon les bords peuvent être flous
    }

    if (recalcul)
    {

        if (VE.y < PP.y)
        { // dessus  __|  ou |__
            if (printoption > 1)
                std::cout << "Dessus ";
            if (cadreY)
            {
                B.y = QQ.y - maconf.deltaVDR; // ignorer le trait du cadre
                BB.y = B.y;
                VV.y = QQ.y - maconf.deltasymbcadre;
                A.y = B.y - maconf.tailleVDR;
                V.y = A.y - maconf.deltachsymb;
            }
            else
            {
                B.y = PP.y - deltahaut;
                BB.y = PP.y - maconf.deltachiffre;
                A.y = B.y - taillecar - 1;
                VV.y = PP.y - maconf.deltasymbole;
                V.y = PP.y - maconf.deltahautsymbole;
            }
            U.y = V.y - maconf.taillesymbole + 1;
            UU.y = VV.y - maconf.largeursymbole + 1;
            AA.y = BB.y - largeurcar + 1;
        }
        else
        { //  dessous   T
            if (printoption > 1)
                std::cout << "dessous ";
            if (cadreY)
            {
                A.y = QQ.y + maconf.deltaVDR;
                AA.y = A.y;
                UU.y = QQ.y + maconf.deltasymbcadre;
                B.y = A.y + maconf.tailleVDR;
                U.y = B.y + maconf.deltachsymb;
            }
            else
            {
                A.y = PP.y + deltahaut;
                AA.y = PP.y + maconf.deltachiffre;
                UU.y = PP.y + maconf.deltasymbole;
                B.y = A.y + taillecar + 1;
                U.y = PP.y + maconf.deltahautsymbole;
            }
            V.y = U.y + maconf.taillesymbole;
            VV.y = UU.y + maconf.largeursymbole;
            BB.y = AA.y + largeurcar - 1;
        }
        if (HO.x > PP.x)
        { // à droite  |__ ou |--
            if (printoption > 1)
                std::cout << "a droite" << std::endl;
            if (cadreX)
            {
                U.x = QQ.x + maconf.deltasymbcadre;
                A.x = QQ.x + maconf.deltaVDR;
                AA.x = A.x;
                BB.x = AA.x + maconf.tailleVDR;
                UU.x = BB.x + maconf.deltachsymb;
            }
            else
            {
                U.x = PP.x + maconf.deltasymbole;
                A.x = PP.x + maconf.deltachiffre;
                AA.x = PP.x + deltahaut;
                BB.x = AA.x + taillecar - 1;
                UU.x = PP.x + maconf.deltahautsymbole;
            }

            V.x = U.x + maconf.largeursymbole;
            VV.x = UU.x + maconf.taillesymbole - 1;
            B.x = A.x + largeurcar + 1;
        }
        else
        { // à gauche __| ou --|
            if (printoption > 1)
                std::cout << "a gauche" << std::endl;
            if (cadreX)
            {
                V.x = QQ.x - maconf.deltasymbcadre;
                B.x = QQ.x - maconf.deltaVDR;
                BB.x = B.x;
                AA.x = BB.x - maconf.tailleVDR + 1;
                VV.x = AA.x - maconf.deltachsymb;
            }
            else
            {
                V.x = PP.x - maconf.deltasymbole;
                B.x = PP.x - maconf.deltachiffre;
                BB.x = PP.x - deltahaut;
                AA.x = BB.x - taillecar + 1;
                VV.x = PP.x - maconf.deltahautsymbole;
            }
            U.x = V.x - maconf.largeursymbole;
            VV.x = PP.x - maconf.deltahautsymbole;
            UU.x = VV.x - maconf.taillesymbole + 1;
            A.x = B.x - largeurcar - 1;
        }

        U.x = std::max(0, U.x);
        V.x = std::max(0, V.x);
        UU.x = std::max(0, UU.x);
        VV.x = std::max(0, VV.x);
        U.y = std::max(0, U.y);
        V.y = std::max(0, V.y);
        UU.y = std::max(0, UU.y);
        VV.y = std::max(0, VV.y);
        A.x = std::max(0, A.x);
        A.y = std::max(0, A.y);
        B.x = std::max(0, B.x);
        B.y = std::max(0, B.y);
        AA.x = std::max(0, AA.x);
        AA.y = std::max(0, AA.y);
        BB.x = std::max(0, BB.x);
        BB.y = std::max(0, BB.y);

        // mémoriser les caractéristiques du coin qu'on vient de recalculer
        moncoin.A = A;
        moncoin.B = B;
        moncoin.AA = AA;
        moncoin.BB = BB;
        moncoin.U = U;
        moncoin.V = V;
        moncoin.UU = UU;
        moncoin.VV = VV;
        moncoin.PP = PP;
        moncoin.QQ = QQ;
        moncoin.inverse = inverse;
        moncoin.estDroit = estDroit;
        moncoin.numcoin = 0;
        moncoin.estunRDV = estunRDV;
        moncoin.ima_coin = coinPetit;
        moncoin.moyblanc = cv::Scalar(255, 255, 255);
        // moncoin.caractere = ' ';  // déjà déterminé

        calculerBlanc(moncoin, maconf);
        reafficher = true;
    }

    // déterminer l'orientation et la couleur
    calculerOrientation(moncoin, maconf);
    // priorité à cette détermination si OCR via le serveur
    // sinon choisir la détermination par OCR si différente et très fiable
    if (estServeur) // détection via le serveur qui n'indique pas l'orientation
    {
        inverse = moncoin.inverse;
        if (inverse) {
            Hcar = output;
            out[4] = output;
            Vcar = "";
            out[0] = "";
            confs[0] = 0;
            confs[4] = confiance;
        } else {
            Vcar = output;
            out[0] = output;
            Hcar = "";
            out[4] = "";
            confs[4] = 0;
            confs[0] = confiance;
        }
    }
    else if (moncoin.inverse != inverse && angle != 90)
    {
        if (confiance > 0.7) {
            if (moncoin.inverse) {
                if (nonvu || out[0] == "" || out[4] != "")
                {
                    inverse = true;
                    Hcar = out[4];
                    if (Hcar != "")
                        nonvu = false;
                }
            } else {
                if (out[4] == "" || out[0] != "") {
                    inverse = false;
                    Vcar = out[0];
                    if (Vcar != "")
                        nonvu = false;
                }
            }
        } else { // confiance faible : invalider la détection OCR
            nonvu = true;
            outprec = output = Hcar = Vcar = out[0] = out[4] = "";
            inverse = moncoin.inverse;
        }
        }
    moncoin.inverse = inverse;
    estDroit = !inverse;
    estRouge = moncoin.estRouge;
    estNoir = !estRouge;
    if (!threadoption) {
        std::cout << "==>";
        if (inverse)
            std::cout << "inverse";
        else
            std::cout << "est Droit";
        if (estRouge)
            std::cout << " Rouge";
        else
            std::cout << " Noir";
        std::cout << std::endl;
    }

    if (inverse) {
        output = outprec = Hcar;
        estDroit = false;
        if (Hcar != "")
            nonvu = false;
    } else {
        output = outprec = Vcar;
        estDroit = true;
        if (Vcar != "")
            nonvu = false;
    }
    if (printoption > 0) {
        if (inverse)
            std::cout << "inverse" << std::endl;
        if (!nonvu)
            std::cout << "==> " << outprec << std::endl;
    }

    // si c'est un Roi Dame ou Valet, on a trouvé sa valeur et son orientation
    // et on a déterminé que c'est un R D ou V
    // et on a recalculé les coordonnées des points U et V (ou UU et VV si inverse)
    // on a les deux points diagonaux de la zone 1 et de la zone 2

    // extraire la zone 1
    // calculer la couleur moyenne et l'écart type de ce rectangle
    cv::Scalar mean_color1, stddev_color1;
    cv::Scalar mean_color2, stddev_color2;
    dc = maconf.deltacadre;
    // décalage lié à la position du cadre ?
    if (reafficher)
    {
        if (printoption)
            std::cout << "P=" << PP << "Q=" << QQ << std::endl;
        extrait = coinPetit.clone();
        cv::circle(extrait, U, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
        cv::circle(extrait, V, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
        cv::circle(extrait, UU, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
        cv::circle(extrait, VV, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
        cv::circle(extrait, A, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
        cv::circle(extrait, B, 1, cv::Scalar(0, 0, 0), -1);      // cercle noir
        cv::circle(extrait, AA, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
        cv::circle(extrait, BB, 1, cv::Scalar(0, 128, 0), -1);   // cercle vert foncé
        cv::circle(extrait, PP, 2, cv::Scalar(0, 255, 255), -1); // cercle jaune

        cv::line(extrait, cv::Point2i(U.x, U.y), cv::Point2i(U.x, V.y), cv::Scalar(0, 0, 0), 1);
        cv::line(extrait, cv::Point2i(U.x, U.y), cv::Point2i(V.x, U.y), cv::Scalar(0, 0, 0), 1);
        cv::line(extrait, cv::Point2i(V.x, U.y), cv::Point2i(V.x, V.y), cv::Scalar(0, 0, 0), 1);
        cv::line(extrait, cv::Point2i(V.x, V.y), cv::Point2i(U.x, V.y), cv::Scalar(0, 0, 0), 1);

        cv::line(extrait, cv::Point2i(UU.x, UU.y), cv::Point2i(UU.x, VV.y), cv::Scalar(0, 255, 0), 1);
        cv::line(extrait, cv::Point2i(UU.x, UU.y), cv::Point2i(VV.x, UU.y), cv::Scalar(0, 255, 0), 1);
        cv::line(extrait, cv::Point2i(VV.x, UU.y), cv::Point2i(VV.x, VV.y), cv::Scalar(0, 255, 0), 1);
        cv::line(extrait, cv::Point2i(VV.x, VV.y), cv::Point2i(UU.x, VV.y), cv::Scalar(0, 255, 0), 1);

        if (estunRDV)
            cv::circle(extrait, QQ, 2, cv::Scalar(0, 0, 128), -1); // cercle rouge foncé
        if (printoption && !threadoption) {
            afficherImage("Extrait", extrait);
            if (waitoption > 2)
                cv::waitKey(0);
            else
                cv::waitKey(1);
        }
    }

    cv::Mat roi2_image;
    cv::Mat roi_image;
    // on a déterminé la zone d'intérêt
    // extraire maintenant une zone élargie de quelques pixels dans chaque sens
    // analyser la zone proche du haut du symbole, pour préciser la position du symbole
    // 4 cas pour haut du symbole : inverse UU.x ou VV.x  droit U.y ou V.y
    // position qui inclut le bas du chiffre (ligne suivie d'une ligne plus claire)
    // ou trop basse (remonter jusqu'à une ligne claire)

    if (inverse) // considérer la zone 2
    {
        if (printoption > 1)
            std::cout << "inverse (rappel)" << std::endl;
        xg = UU.x;
        yh = UU.y;
        dx = 1 + VV.x - UU.x; // inclure le haut et le bas du symbole
        dy = 1 + VV.y - UU.y;
        ls = maconf.taillesymbole;
        ts = maconf.largeursymbole;
    } else { // extraire la zone 1
        if (printoption > 1)
            std::cout << "est droit (rappel)" << std::endl;
        xg = U.x;
        dx = 1 + V.x - U.x;
        yh = U.y;
        dy = 1 + V.y - U.y;
        ts = maconf.taillesymbole;
        ls = maconf.largeursymbole;
    }
    // ajouter quelques pixels là ou on peut, pour tenir compte des variations de géométrie des cartes
    int ajout;

    if (estunRDV)
    {
        // ajouter quelques pixels sous le symbole et vers l'intérieur de la carte
        int dh = maconf.deltacadre / 2; // il y a beaucoup de blanc sous le symbole
        dx += dh; dy += dh;
        if (UU.x < PP.x) xg -= dh;
        if (U.y < PP.y) yh -= dh;
#ifdef ACTIVER
        if (UU.x < PP.x) xg++; // + 1 pixel vers le chiffre ou vers le cadre
        else             xg--;

        if (U.y < PP.y) yh++; // +1 pixel vers le chiffre ou le cadre
        else            yh--;
#endif
    } else  { // le symbole est sous un chiffre
        // élargir  la zone du symbole,  1 ou 2 pixels vers le chiffre, à enlever plus tard
        ajout = 1;
        if (maconf.taillesymbole > 7)  ajout = 2;
        // si le chiffre est 10, on aura peut-être décalé PP vers l'extérieur du coin, de deltacadre
        // car un des deux petits rectangles de test dans le coin rencontre le caractère 1 ou le caractère 0
        // dans ce cas, élargir la zone du coin de deltacadre vers l'intérieur du coin
        if (inverse) {
            if (maconf.deltasymbole > 3) {
                dy += 6 * ajout;
                yh -= 3 * ajout; // 3 pixels vers l'intérieur de la carte et 3 pixels vers le bord
            } else {
                dy += ajout;
                yh -= ajout; // 1 pixels vers l'intérieur de la carte et 1 pixels vers le bord
            }
            dx += 4 * ajout; // 3 pixels sous le symbole et 1 pixel vers le chiffre
            if (UU.x < PP.x ) xg -= 3 * ajout; // 1 pixel vers le chiffre
            else              xg -= ajout;
        } else {                    // vertical
            if (maconf.deltasymbole > 3) {
                dx += 6 * ajout; // 3 pixels vers l'intérieur 3 pixels vers le bord lateral
                xg -= 3 * ajout;
            } else {
                dx += ajout; // 1 pixels vers l'intérieur 1 pixels vers le bord lateral
                xg -= ajout;
            }
            dy += 4 * ajout; // 3 pixels sous le symbole et 1 pixel vers le chiffre
            if (U.y < PP.y)   yh -= 3 * ajout;
            else              yh -= ajout;
        }
    }
    if (xg < 0)  xg = 0;
    if (yh < 0)  yh = 0;
    if (dy > coinPetit.rows - yh) dy = coinPetit.rows - yh;
    if (dx > coinPetit.cols - xg) dx = coinPetit.cols - xg;
    if (dx <= 0 || dy <= 0) {
        if (printoption)
            std::cout << "!!!!! erreur extraction du symbole " << std::endl;
        // on conserve l'évaluation précédente, sans agrandissement
    } else {
        // extraire
        cv::Rect roi3(xg, yh, dx, dy);
        if (printoption && !threadoption)
            tracerRectangle(roi3, extrait, "Extrait", cv::Scalar(0, 255, 255));
        if (waitoption > 2)
            cv::waitKey(0);
        roi_image = coinPetit(roi3).clone();
    }

    // redresser le symbole extrait

    // selon l'orientation du coin
    // il faudra pivoter  de + ou - 90 degrés ou 180
    // coin haut gauche : si inversion : pivoter symbole -90
    // coin haut droit  : si inversion : pivoter +90
    // coin bas gauche  : si inversion : pivoter tout -90, sonon pivoter tout 180
    // coin bas droit   : si inversion  : pivoter tout +90 , sinon pivoter tout 180
    int rotation;
    rotation = 0;
    if (U.y < PP.y)
    { // coin bas gauche ou droite (de la carte)
        cv::Mat rotated_image;
        if (!inverse)
        { // tourner de 180 degrés
            cv::rotate(roi_image, rotated_image, cv::ROTATE_180);
            roi_image = rotated_image.clone();
            rotation = 2;
        }
    }
    if (inverse) { // tourner tout à droite (-90) ou à gauche
        cv::Mat rotated_image;
        if (UU.x > PP.x)
        { // à droite haut ou bas
            cv::rotate(roi_image, rotated_image, cv::ROTATE_90_CLOCKWISE);
            rotation = 1;
        } else { // à gauche
            cv::rotate(roi_image, rotated_image, cv::ROTATE_90_COUNTERCLOCKWISE);
            rotation = 3;
        }
        roi_image = rotated_image.clone();
    }
    // préciser la position du bas du symbole : un peu de noir ou de rouge (=moins de bleu)
    // partir du bas de l'image et remonter
    {
        ts = maconf.taillesymbole;
        ls = maconf.largeursymbole;
        int xg, Box[4];
        cv::Mat lig;
        if (estRouge) {
            eclaircirfond(roi_image); // indispensable pour trouver le haut du symbole
            // chercher la ligne blanche éventuelle (entre caractère et symbole) à partir du haut,
            // parmi les 3 premières lignes
            r.x = 0;
            r.width = roi_image.cols;
            r.height = 1;
            int ih = 0;
            double mpre = 255;
            for (int i = 0; i < 3; i++) {
                r.y = i;
                lig = roi_image(r);
                moy = mean(lig);
                if (moy[0] > 250) {
                    ih = i + 1;
                    break;
                } // sans la ligne blanche
                if (moy[0] > 230 && moy[0] - mpre > 10) {
                    ih = i + 1;
                    break;
                } // sans la ligne blanche
                mpre = moy[0];
            }
            r.y = ih; // haut du symbole
            r.height = roi_image.rows - r.y;
            roi_image = roi_image(r).clone();
        }
        calculerBox(roi_image, ts, ls, moy, Box, moyext, maconf);
        xg = Box[0];
        r.x = Box[0] + ls / 3;
        r.width = ls / 3; // recherche sur la partie centrale
        r.height = 1;
        r.y = roi_image.rows - 1;
        while (r.y >= ts) {
            lig = roi_image(r);  moy = cv::mean(lig);
            if (moy[0] < moyext[0] - 15) break; // ligne du symbole
            r.y--;
        }
        // on a une position dans le symbole
        // remonter jusqu'à la ligne blanche au dessus du symbole
        r.y -= ts;
        while (r.y >= 0)  {
            lig = roi_image(r).clone(); moy = cv::mean(lig);
            if (moy[0] > moyext[0] - 5) {
                r.y++;
                break;
            } // on élimine la ligne blanche
            r.y--;
        }
        if (r.y < 0) r.y = 0;
        r.height = ts + 1;
        // if(r.y > roi_image.rows - r.height) r.y = roi_image.rows - r.height;
        if (r.height > roi_image.rows - r.y) r.height = roi_image.rows - r.y;
        r.x = xg; r.width = ls;
        roi_image = roi_image(r).clone(); // en haut : haut du symbole
    }
#ifdef ACTIVER
    // essayer de déterminer le symbole par OCR
    // l'expérience prouve que ce n'est pas pertinent
    {
        cv::Mat ima;
        cv::cvtColor(roi_image, ima, cv::COLOR_BGR2GRAY);
        cv::threshold(ima, ima, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche
        cv::Mat image_bordee;
        cv::copyMakeBorder(ima, image_bordee, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
        ima = image_bordee;
        if (printoption && !threadoption)
            afficherImage("PS", ima);
        if (waitoption > 2)
            cv::waitKey(0);
        else
            cv::waitKey(1);
        double conf(0.0), ang(0.0);
        std::string val = execOCR("SERVEUR", ima, &conf, &ang);
        if (printoption) std::cout<<" valeur du symbole :"<<val<<", confiance "<<conf<<std::endl;
    } 
#endif
    // ajuster le haut du symbole, pour enlever le bas du chiffre
    // ligne 0 : bas du chiffre   ou  ligne blanche    ou  chiffre
    //       1 : ligne blanche    ou  haut du symbole  ou  bas du chiffre
    //       2 : haut du symbole                       ou  ligne blanche
    if (false)  { // devenu inutile
        cv::Rect rr;
        cv::Mat lig;
        cv::Scalar m1, m2, m3;
        int iy, jy;
        // analyser la ligne du haut du symbole (1) et la suivante (2)
        iy = 1; jy = 2; rr.x = 0; rr.y = 1;
        rr.height = 1;  rr.width = roi_image.cols;
        lig = roi_image(rr); m1 = cv::mean(lig);
        rr.y = jy; lig = roi_image(rr); m2 = cv::mean(lig);
        if (m2[0] - m1[0] > 5)
        {
            // éliminer la ligne supérieure(bas du caractère) et la ligne blanche
            rr.y = 2; // enlever 2 lignes du haut
            rr.height = roi_image.rows - 2;
        } else {
            // si la ligne  est plus foncée que la ligne 0, éliminer la ligne 0
            rr.y = 0;
            lig = roi_image(rr); m3 = cv::mean(lig);
            if (m1[0] > m3[0])
            {
                rr.y = 1; // enlever la ligne rajoutée lors de l'extraction (bas du chiffre)
                rr.height = roi_image.rows - 1;
            }
            else  rr.height = roi_image.rows; // laisser en l'état
        }
        roi_image = roi_image(rr).clone();
    }

    // analyser le symbole pour distinguer entre pique et trefle, si c'est une carte noire
    //   ou entre coeur et carreau, si c'est une carte rouge
    // trefle : le rectangle du tiers central est plus clair que les bords latéraux
    // Pique  : uniforme ou centre plus foncé

    // symbole rouge : voir plus loin

    int numcol = -1; // 0 : Pique, 1: Coeur, 2: carreau, 3: trefle

    cv::Mat symbgros; // image agrandie du symbole (petit ou gros)
    cv::Mat imaSymb = roi_image.clone();

    cv::Point2i CS((maconf.largeurVDR + 1) / 2, (maconf.taillesymbole + 1) / 2); // centre probable du symbole

    int deltableu(0);   // écart de bleu entre la partie infériere et la partie supérieure
    int deltableugs(0); // ... du gros symbole
    int deltableucent(0);
    double echelle = 1.0; // agrandissement du symbole, petit ou gros;
    // int ts(0), ls(0); // taille du symbole après agrandissement
    int diviseur = 3; // partie de la bande horizontale du symbole
    cv::Mat lig;
    cv::Mat bande;
    bool estgrossymb = false;
    echelle = 1;
    ts = maconf.taillesymbole;
    ls = maconf.largeursymbole;

    ////////////////////////////////////////////////////////////
    // ne pas utiliser le gros symbole si le petit symbole est assez grand ////////
    ///////////////////////////////////////////////////////////////////////////

    if (maconf.taillesymbole < maconf.ignorerGS) {

        cv::Rect rG; // rectangle pour extraire l'image du gros symbole

        // un gros symbole est présent à coté du caractère si la valeur de la carte est de 4 à 10
        // rechercher le gros symbole en haut de la carte 2 ou 3
        //  ou un honneur couché à droite au dessus ou à gauche au dessous
        //     ou droit à droite au dessous ou à gauche au dessus
        if (estunRDV)
        {
            if (inverse) {
                rG.width = maconf.taillegrosRDV;
                rG.height = maconf.largeurgrosRDV;
                if (UU.x > PP.x && U.y < PP.y) { // couché, à droite au dessus
                    estgrossymb = true;
                    rG.x = QQ.x + maconf.deltagroshautRDV;
                    rG.y = QQ.y - maconf.deltagrosRDV - rG.height;
                } else if (UU.x < PP.x && U.y > PP.y) { // couché à gauche au dessous
                    estgrossymb = true;
                    rG.x = QQ.x - maconf.deltagroshautRDV - maconf.taillegrosRDV;
                    rG.y = QQ.y + maconf.deltagrosRDV;
                }
            }
            else { // droit
                rG.height = maconf.taillegrosRDV;
                rG.width = maconf.largeurgrosRDV;
                if (UU.x > PP.x && U.y > PP.y) { // à droite dessous
                    estgrossymb = true;
                    rG.x = QQ.x + maconf.deltagrosRDV;
                    rG.y = QQ.y + maconf.deltagroshautRDV;
                }  else if (UU.x < PP.x && U.y < PP.y) { // à gauche dessus
                    estgrossymb = true;
                    rG.x = QQ.x - maconf.deltagrosRDV - maconf.largeurgrosRDV;
                    rG.y = QQ.y - maconf.deltagroshautRDV - rG.height;
                }
            }
        }
        else
        //TODO : rechercher s'il y a un GS à coté du caractère
        //       ou au milieu du haut de la carte
        //        même si le caractère n'a pas été lu
        //      alternativement : refaire la recherche OCR avant ces tests
        //if (!nonvu && (output == "10" || (output[0] > '3' && output[0] <= '9')))
        if (true)
        {
            //estgrossymb = true;
            if (inverse) {
                rG.width = maconf.taillegros + 4; // on ajoute 2 pixels en haut et bas du symbole
                rG.height = maconf.largeurgros;
                if (U.y < PP.y) rG.y = PP.y - maconf.deltagros - rG.height; // au dessus
                else rG.y = PP.y + maconf.deltagros;
                if (UU.x < PP.x) rG.x = PP.x - maconf.deltagroshaut - maconf.taillegros -2;
                else rG.x = PP.x + maconf.deltagroshaut -2;
            } else {
                rG.width = maconf.largeurgros;
                rG.height = maconf.taillegros +4; // ajout 2 pixels en haut et en bas
                if (UU.x > PP.x) rG.x = PP.x + maconf.deltagros;
                else rG.x = PP.x - maconf.deltagros - maconf.largeurgros;
                if (U.y > PP.y) rG.y = PP.y + maconf.deltagroshaut -2;
                else rG.y = PP.y - maconf.deltagroshaut - maconf.taillegros -2;
            }
            if (rG.x < 0 || rG.y < 0 || rG.x + rG.width > coinPetit.cols || rG.y + rG.height > coinPetit.rows)
                estgrossymb = false;
            else {
                cv::Mat imaGS = coinPetit(rG);
                cv::Scalar m = cv::mean(imaGS);
                if ((output == "10" || (output[0] > '3' && output[0] <= '9'))
                    || m[0] < 9*moncoin.moyblanc[0]/10 ){
                    estgrossymb = true;
                }
            }
        }
        if (!estgrossymb && !estunRDV) {
            // rechercher le gros symbole au milieu du haut de la carte
            // même position en hauteur que les GS des cartes 4 à 10
            // mais centré en largeur
            //estgrossymb = true;
            if (inverse) {
                rG.width = maconf.taillegros + maconf.taillesymbole; // ajout des 4 cotés
                rG.height = maconf.largeurgros + maconf.largeursymbole;
                if (U.y < PP.y) rG.y = PP.y - (maconf.largeurcarte + rG.height) / 2;
                else rG.y = PP.y + (maconf.largeurcarte - rG.height) / 2;
                if (UU.x < PP.x) rG.x = PP.x - maconf.deltagroshaut - maconf.taillegros - maconf.taillesymbole / 2;
                else rG.x = PP.x + maconf.deltagroshaut - maconf.taillesymbole / 2;
            } else {
                rG.width = maconf.largeurgros + maconf.largeursymbole;
                rG.height = maconf.taillegros + maconf.taillesymbole;
                if (UU.x > PP.x) rG.x = PP.x + (maconf.largeurcarte - rG.width) / 2;
                else rG.x = PP.x - (maconf.largeurcarte + rG.width) / 2;
                if (U.y > PP.y) rG.y = PP.y + maconf.deltagroshaut - maconf.taillesymbole / 2;
                else  rG.y = PP.y - maconf.deltagroshaut - maconf.taillegros - maconf.taillesymbole / 2;
            }
            if (rG.x < 0 || rG.y < 0 || rG.x + rG.width > coinPetit.cols || rG.y + rG.height > coinPetit.rows)
                estgrossymb = false;
            else {
                cv::Mat imaGS = coinPetit(rG);
                cv::Scalar m = cv::mean(imaGS);
                if ((output[0] == '2' || output[0] == '3')
                    || m[0] < 9*moncoin.moyblanc[0]/10 ){
                    estgrossymb = true;
                }
            }
        }
        if (estgrossymb) {
            r = rG;
            if (rG.x < 0 || rG.y < 0 || rG.x + rG.width > coinPetit.cols || rG.y + rG.height > coinPetit.rows)
                estgrossymb = false;
        }
        if (estgrossymb) {
            if (printoption  && !threadoption)
                tracerRectangle(r, extrait, "Extrait", cv::Scalar(255, 0, 0));
            echelle = 1; // agrandir n'améliore pas le résultat
            if (estunRDV) {
                ts = maconf.taillegrosRDV;
                ls = maconf.largeurgrosRDV;
            } else {
                ts = maconf.taillegros;
                ls = maconf.largeurgros;
            }
            if (waitoption > 2 && !threadoption)
                cv::waitKey(0);
            roi_image = coinPetit(r).clone(); // gros symbole
            // redresser
            if (inverse) {
                if (UU.x > PP.x) {
                    cv::rotate(roi_image, roi_image, cv::ROTATE_90_CLOCKWISE);
                } // à droite rotation + 90
                else {
                    cv::rotate(roi_image, roi_image, cv::ROTATE_90_COUNTERCLOCKWISE);
                } // à gauche rotation - 90
            }
            else if (U.y < PP.y) {
                cv::rotate(roi_image, roi_image, cv::ROTATE_180);
            } //  droit dessus rotation 180
            // else {} // dessous laisser tel quel
        }
    }

    if (printoption && estgrossymb){
        if (threadoption ) std::cout<<"coin "<< n << " ";
        std::cout<<"GS "<<std::endl;
    }
    numcol = -1;
    int Box[4]; // xmin xmax ymin ymax
    int yBH;
    int hBH;

    if (estgrossymb) numcol = calculerCouleur(roi_image, maconf, moncoin.moyblanc);
    if (numcol < 0) {  // pas trouvé (improbable sauf pour un As)
       
        // roi_image : petit ou gros symbole
        amplifyContrast(roi_image); // parfois contre productif
        // if (estRouge) eclaircirfond(roi_image); // déjà fait
        if (estRouge && estgrossymb)
            eclaircirfond(roi_image); // déjà fait pour petit symbole
        symbgros = roi_image.clone();
        if (printoption && !threadoption) {
            afficherImage("symbole", symbgros);
            afficherImage("gros", symbgros);
            cv::waitKey(1);
        }
        ts *= echelle;
        ls *= echelle;
        // calculer l'encombrement du symbole et les moyennes d'intensite du symbole
        calculerBox(roi_image, ts, ls, moy, Box, moyext, maconf);
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
        cv::Rect rr = r;
        rr.x = r.x + ls / 3;
        rr.width = ls / 3;
        int moyb;
        while (r.y <= Box[3] - r.height) {
            cv::Mat bande = roi_image(r);
            moy = cv::mean(bande);
            if (!estRouge) {
                cv::Mat centre = roi_image(rr);
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
    }
    //////////////////////////////////////////////////////////////////////////////
    ///////////////// recherche OCR complémentaire ///////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    // si on déjà reconnu le chiffre ou la lettre V D ou R il est inutile de relancer la recherche
    bool nonreconnu = false;
    int x, y;

    if (nonvu) { // on n'a pas encore trouvé le caractère
        if (inverse) { A = AA; B = BB; }
        // si c'est un Roi une Dame ou un Valet, la position du coin est mal définie
        // la position du coin est maintenant bien définie (à 1 pixel près)
        // la réalité peut être légèrement à l'extérieur du coin identifié
        // donc déplacer de "deltacadre"
        if (estunRDV)
        {
            cv::Point2i M((UU + VV) / 2); // un point à l'intérieur du coin
            if (!cadreY) {
                if (M.y > PP.y) A.y -= maconf.deltacadre;
                else            B.y += maconf.deltacadre;
            }
            // il y a un gros symbole juste à coté, génant pour le Valet de Pique
            if (inverse)
                if (U.y < PP.y)  {
                    A.y -= maconf.largeurVDR / 3; // risque d'absorber une partie du gros symbole
                    B.y += 1; // on risque d'absorber le trait du cadre
                } else {
                    B.y += maconf.largeurVDR / 3;
                    A.y -= 1;
                }
            else if (UU.x > PP.x) {
                B.x += maconf.largeurVDR / 3;
                A.x -= 1;
            } else {
                A.x -= maconf.largeurVDR / 3;
                B.x += 1;
            }
        } else { // probablement un chiffre, bords de carte mal définis, si c'est en réalité un R D V
            if (A.y < PP.y) B.y += maconf.deltahaut / 2;
            else            A.y -= maconf.deltahaut / 2;
            if (!inverse)
            {
                if (A.x > PP.x) {
                    A.x -= maconf.deltacadre / 2;
                    B.x += maconf.deltacadre / 2;
                } else {
                    B.x += maconf.deltacadre / 2;
                    A.x -= maconf.deltacadre / 2;
                }
            }
        }

        // rester dans les limites de coinPetit
        A.x = std::max(0, A.x);
        A.y = std::max(0, A.y);
        B.x = std::max(0, B.x);
        B.y = std::max(0, B.y);
        if (B.x > coinPetit.cols) B.x = coinPetit.cols;
        if (B.y > coinPetit.rows) B.y = coinPetit.rows;

        // limiter le rectangle AB : ne doit pas contenir les bords du coin ou du cadre:
        cv::Point2i RR = PP;
        if (estunRDV)  RR = QQ;
        if (UU.x < PP.x) if (B.x >= RR.x)  B.x = RR.x - 1;
        if (UU.x > PP.x) if (A.x <= RR.x)  A.x = RR.x + 1;
        if (U.y < PP.y)  if (B.y >= RR.y)  B.y = RR.y - 1;
        if (U.y > PP.y)  if (A.y <= RR.y)  A.y = RR.y + 1;
        // on a déterminé la zone du chiffre : rectangle de diagonale AB
        //
        if (inverse) cv::line(extrait, A, B, cv::Scalar(0, 255, 0), 1); // petit trait vert
        else         cv::line(extrait, A, B, cv::Scalar(0, 0, 0), 1); // petit trait noir
        if (estunRDV)  cv::circle(extrait, PP, 4, cv::Scalar(0, 0, 0), 2); // cercle noir
        if (printoption && !threadoption) afficherImage("Extrait", extrait);
        x = A.x;
        y = A.y;
        dx = std::max(1, 1 + B.x - A.x);
        dy = std::max(1, 1 + B.y - A.y);

        if (x + dx >= coinPetit.cols) dx = coinPetit.cols - x - 1;
        if (y + dy >= coinPetit.rows) dy = coinPetit.rows - y - 1;

        cv::Rect regionC(x, y, dx, dy);
        ima_car = coinPetit(regionC);

        // éventuelle rotation déjà déterminée lors de l'étude du symbole
        cv::Mat rotated_image;
        enum cv::RotateFlags rf;
        if (rotation == 1) rf = cv::ROTATE_90_CLOCKWISE;
        else if (rotation == 2) rf = cv::ROTATE_180;
        else if (rotation == 3) rf = cv::ROTATE_90_COUNTERCLOCKWISE;
        if(rotation){ cv::rotate(ima_car, rotated_image, rf); ima_car = rotated_image; }

        cv::Mat ima_ch = ima_car;
        cv::cvtColor(ima_car, ima_ch, cv::COLOR_BGR2GRAY);
        cv::threshold(ima_ch, ima_ch, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        // ajouter une bordure blanche
        cv::Mat image_bordee;
        cv::copyMakeBorder(ima_ch, image_bordee, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(255));
        ima_ch = image_bordee;
        if (printoption && !threadoption) {
            afficherImage("chiffre", ima_ch);
            if (waitoption > 2) cv::waitKey(0); else cv::waitKey(1);
        }

        if (maconf.tesOCR > 0) output = tesOCR(ima_ch, estunRDV, &confiance, &angle);
        if (maconf.tesOCR <= 1 && (output == "" || confiance < 0.30))
            output = execOCR("SERVEUR", ima_ch, &confiance, &angle);
        if (printoption) std::cout << output << " confiance " << confiance << std::endl;
        if (confiance < 0.5) output = "";
        if (output != ""){
            if (output[0] == 'R' || output[0] == 'D' || output[0] == 'V') output = output[0];
        }
        std::string outW = ValiderOCR(output, false, inverse, moncoin, maconf);
        if (output == "3" && outW != "3") output = "";
        else output = outW;
        if (printoption && !threadoption){
            if (waitoption > 2) cv::waitKey(0); else cv::waitKey(1);
        }
        nonreconnu = false;
        if (output == "" ) output = outprec;
        int sz = output.size();
        if (sz < 1 || sz > 2) nonreconnu = true;
        else {
            if (estunRDV) {
                // accepter IV ID IR et V* D* R*
                if (output == "IV" || output == "ID" || output == "IR") output = output[1];
                if (output[0] != 'V' && output[0] != 'D' && output[0] != 'R') nonreconnu = true;
            } else {
                if (sz != 1 || (output[0] < '1' || output[0] > '9')) nonreconnu = true;
                if (output[0] == 'V' || output[0] == 'D' || output[0] == 'R') nonreconnu = false;
                if (sz == 2) {
                    nonreconnu = true;
                    if ((output[0] == '1' || output[0] == 'I' || output[0] == 'i')
                     && (output[1] == '0' || output[1] == 'O' || output[1] == 'C'))
                    {
                        nonreconnu = false;
                        output = "10";
                    }
                }
            }
        }
    }
    // valider le caractère reconnu selon la présence d'un gros symbole
    // déjà vérifié si un caractère avait été reconnu par OCR
    // sinon, la présence du gros symbole n'est pas calculée
    if (false && !nonreconnu)
    {
        if (output > "3" && output <= "9" && !estgrossymb)  nonreconnu = true;
        if (output == "10" && !estgrossymb)  nonreconnu = true;
        if ((output == "1" || output == "2" || output == "3") && estgrossymb)  nonreconnu = true;
    }

    if (nonreconnu)
    {
        if (printoption)  std::cout << "non reconnu " << output << std::endl;
        retourcoin(n);
        return; 
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// traitement du symbole rouge //////////////////////////
    /////////////////////////////////traitement du symbole rouge //////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    // cas particulier du Valet rouge chapeau bleu pour coeur ou rouge pour carreau
    // sinon, analyser le gros symbole éventuel à coté du caractère
    // sinon analyser le petit symbole sous le caractère

    // Valet ? chapeau rouge--> carreau,  bleu --> coeur
    if (estRouge && numcol < 0)
    {
        // si c'est un Valet, tester la couleur du chapeau
        // le chapeau est au milieu du petit coté de la carte
        // on teste la couleur d'un petit rectangle
        // rouge --> carreau, bleu --> coeur
        numcol = -1; // couleur non déterminée
        // chapeau bleu ou rouge du valet ?
        if (output == "V")
        {
            int demilargeur = maconf.hauteurcarte / 3; // approximatif
            if (inverse)
            {
                r.height = maconf.largeurgrosRDV;
                r.width = maconf.tailleVDR;
                if (UU.x > PP.x)
                    r.x = QQ.x;
                else
                    r.x = QQ.x - r.width;
                if (U.y < PP.y)
                    r.y = PP.y - demilargeur - r.height / 2;
                else
                    r.y = PP.y + demilargeur - r.height / 2;
            }
            else
            {
                r.width = maconf.largeurgrosRDV;
                r.height = maconf.tailleVDR;
                if (UU.x < PP.x)
                    r.x = PP.x - demilargeur - r.width / 2;
                else
                    r.x = PP.x + demilargeur - r.width / 2;
                if (UU.y < PP.y)
                    r.y = QQ.y - r.height;
                else
                    r.y = QQ.y;
            }
            cv::Mat ima_gs = coinPetit(r);
            cv::line(extrait, cv::Point2i(r.x, r.y), cv::Point2i(r.x, r.y + r.height), cv::Scalar(0, 0, 0));
            cv::line(extrait, cv::Point2i(r.x, r.y), cv::Point2i(r.x + r.width, r.y), cv::Scalar(0, 0, 0));
            cv::line(extrait, cv::Point2i(r.x, r.y + r.height), cv::Point2i(r.x + r.width, r.y + r.height), cv::Scalar(0, 0, 0));
            cv::line(extrait, cv::Point2i(r.x + r.width, r.y), cv::Point2i(r.x + r.width, r.y + r.height), cv::Scalar(0, 0, 0));
            if (printoption && !threadoption) {
                afficherImage("Extrait", extrait);
                if (waitoption > 2)
                    cv::waitKey(0);
                else
                    cv::waitKey(1);
            }

            cv::meanStdDev(ima_gs, moy, ect);
            if (printoption > 1)
                std::cout << "chapeau moy et ect " << moy << "," << ect << std::endl;

            if (moy[0] < moy[2])
                numcol = 2; // carreau
            else
                numcol = 1; // coeur
        }
    }
    // traitement symbole (petit ou gros)
    if (estRouge && numcol < 0)
    {
        // roi_image : le haut peut être la fin du chiffre
        int xgBH = Box[0];
        int xmin, xmax;
        xmin = Box[0];
        xmax = Box[1];
        int xopt;
        // on a xmin et xmax du symbole
        r.x = xgBH; // gauche de la bande horizontale
        r.y = yBH;  // haut de cette bande
        r.width = std::min(Box[1] - r.x + 1, xmax - xmin + 1);
        if (printoption && !threadoption)
            tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 0, 0)); // bande horizontale centrée
        CS.x = (xmin + xmax) / 2;                                      // donc le centre horizontal du symbole
        // rechercher la position de la bande verticale optimale (avec le plus de rouge = le moins de bleu)
        //  uniquement dans le tiers inférieur du symbole (en forme de V pour coeur et carreau)
        // essayer avec deux largeurs (2 et 1 ou 3) et choisir le résultat le moins clair
        //
        r.y = yBH + (hBH + 1) / 2; // position au milieu la bande horizontale et au dessous
        if (r.y >= Box[3])
            r.y = Box[3];
        r.height = Box[3] + 1 - r.y;
        if (r.height < 1)
            r.height = 1;
        if (estgrossymb)
        {
            r.x = Box[0] + ls / 3;
            xmax -= ls / 3;
        }
        int largeurcol = 2;
        r.width = 2; // calcul avec largeur 2
        xopt = r.x;
        double minb3 = 255 * 2;
        double minb4 = minb3;
        while (r.x <= xmax - r.width)
        {
            bande = roi_image(r);
            moy = cv::mean(bande);
            double mbg = moy[0] + moy[1];
            if (mbg <= minb3)
            {
                minb3 = mbg;
                xopt = r.x;
            }
            r.x++;
        }
        int xopt3 = xopt;
        r.width = 1;
        if (ls > 8)
            r.width = 3; // calcul avec largeur 1 ou 3
        r.x = xgBH;      // gauche de la bande horizontale
        if (ls >= 18)
        {
            r.x += ls / 6;
            xmax -= ls / 3;
        }

        xopt = r.x;
        while (r.x <= xmax - r.width)
        {
            bande = roi_image(r);
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
        r.height = roi_image.rows;
        bande = roi_image(r);
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
        bande = roi_image(rr);
        moy = cv::mean(bande); // 1 pou 3 pixels
        rr.y++;                // ligne en dessous
        if (rr.x > 0)
            rr.x--;
        rr.width += 2; // zone centrale élargie ( Zone RWR)
        if (rr.width > roi_image.cols - rr.x)
            rr.width = roi_image.cols - rr.x;
        bande = roi_image(rr);
        cv::Scalar moy1 = cv::mean(bande); // milieu (zone RWR)

        rr.x = xmin + 1;
        rr.width = xmax - xmin - 1; // ligne en dessous moins pixels du bord
        bande = roi_image(rr);
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
            if (printoption && !threadoption)
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
                    bande = roi_image(rr);
                    m1 = mean(bande);
                }
                l2 = xmax - xaxe - largeurcol;
                if (l2 > 0)
                {
                    rr.x = xaxe + largeurcol;
                    rr.width = l2;
                    bande = roi_image(rr);
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
            bande = roi_image(rr);
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
            if (printoption && !threadoption)
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
            bool adroite = false;
            if (estgrossymb)
            {
                if (inverse)
                {
                    if (U.y < PP.y && UU.x < PP.x)
                        adroite = true;
                    if (U.y > PP.y && UU.x > PP.x)
                        adroite = true;
                }
                else
                {
                    if (U.y > PP.y && UU.x < PP.x)
                        adroite = true;
                    if (U.y < PP.y && UU.x > PP.x)
                        adroite = true;
                }
            }

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
                if (estgrossymb)
                    r.width = 2 * r.width / 3;
                if (r.width > roi_image.cols - r.x)
                    r.width = roi_image.cols - r.x;
            }
            else
            {
                // entre Box[0] et xaxe-1
                r.width = std::max(1, xaxe - Box[0]);
                if (estgrossymb)
                    r.width = 2 * r.width / 3;
                r.x = std::max(0, xaxe - r.width);
            }
            if (r.height > Box[3] + 1 - r.y)
                r.height = Box[3] + 1 - r.y;
            if (printoption && !threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne en haut à gauche ou droite
            lig = roi_image(r);
            moy = cv::mean(lig);
            cv::Scalar moyHaut = moy;

            r.y = std::max(0, ybas + 1 - r.height);
            // r.x et r.width inchangés
            if (printoption && !threadoption)
                tracerRectangle(r, symbgros, "gros", cv::Scalar(0, 255, 0)); // petite ligne de test en haut à gauche
            lig = roi_image(r);
            moy = cv::mean(lig);
            // comparer l'intensité bleue entre le segment en haut et le segment en bas
            // coeur s'il y a significativement plus de bleu en bas, sinon carreau
            int ecartbleu = moy[0] - moyHaut[0];
            if ((estgrossymb && ecartbleu > 100) || (!estgrossymb && ecartbleu > 22))
                numcol = 1; // 0 : experimental
            else if ((estgrossymb && ecartbleu < 50) || (!estgrossymb && ecartbleu < 15))
                numcol = 2;
            // sinon : indéterminé
            if (printoption)
            {
                if (numcol == 1)
                    std::cout << " coeur  ";
                if (numcol == 2)
                    std::cout << " carreau ";
                std::cout << " intensite bleu bas - haut " << ecartbleu << std::endl;
            }
            if (printoption > 1 && !threadoption)
                afficherImage("gros", symbgros);
        }
    }
    if (estRouge && numcol > 0)
    {
        if (printoption && !threadoption)
        {
            if (numcol == 1)  std::cout << " coeur" << std::endl;
            else              std::cout << " carreau " << std::endl;
            if (waitoption > 2) cv::waitKey(0); else cv::waitKey(1);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// traitement du symbole noir ///////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    if (estNoir && numcol < 0)  {
        int BoxW[4];
        calculerBox(roi_image, ts, ls, moy, BoxW, moyext, maconf);


        // si la définition est suffisante, on ne s'est pas intéressé au gros symbole
        if (estunRDV && !estgrossymb) { // GS pas encore déterminé ou réellement absent
            if (inverse) { 
                if ((U.x > PP.x && UU.y < PP.y) || (U.x < PP.x && UU.y > PP.y)) estgrossymb = true;
            } else {
                if ((U.x > PP.x && UU.y > PP.y) || (U.x < PP.x && UU.y < PP.y)) estgrossymb = true;
            }
        }
        numcol = -1;
        // lyre du roi
        if (moncoin.caractere == 'R' && !estgrossymb) {
            // déterminer la position du haut de la lyre du roi de pique
            // uniquement inverse (dessus gauche ou dessous droit)
            //           ou droit (dessous gauche ou dessus droit)
            bool testerlire = false; 
            if ((inverse && ( UU.x < PP.x && U.y < PP.y|| UU.x > PP.x && U.y > PP.y))
            || (!inverse && ( UU.x > PP.x && U.y < PP.y|| UU.x < PP.x && U.y > PP.y))
            ) {
                if (inverse){
                    r.width = maconf.taillesymbole;
                    if (UU.x < QQ.x) // à gauche
                        r.x = UU.x - r.width - 1;
                    else
                        r.x = VV.x + 1;
                    r.height = maconf.largeursymbole;
                    if (U.y > QQ.y)
                        r.y = (UU+VV).y / 2;
                    else
                        r.y = (UU+VV).y /2 - r.height;
                }
                else
                {
                    r.width = maconf.largeursymbole;
                    r.height = maconf.taillesymbole;
                    if (UU.x < QQ.x)
                        r.x = (U+V).x /2  - r.width;
                    else
                        r.x = (U+V).x /2;
                    if (U.y > QQ.y)
                        r.y = V.y + 1;
                    else
                        r.y = U.y - r.height - 1;
                }
                lig = coinPetit(r).clone();
                if (printoption && !threadoption) tracerRectangle(r, extrait, "Extrait", cv::Scalar(0, 0, 255));
                cv::meanStdDev(lig, moy, ect);
                if (printoption)
                    std::cout << "ecart type lyre du roi noir " << ect << std::endl;
                if (ect[0] > 30)
                    numcol = 0; // Pique
                else if (ect[0] < 10)
                    numcol = 3; // Trefle
                // sinon : indéterminé
            }
        }
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
        // si on trouve une ligne significativement plus claire (que la précédente?) : c'est du trefle 
        if (ts >= 12){

            cv::Rect r;
            r.height = (2*ts+2)/5;   // 40% semble correct après tests
            if (r.height < 5) r.height = 5;
            r.width = (ls +2) /3;
            r.x = BoxW[0] + ls/3;
            r.y = BoxW[2]; // haut du symbole
            cv::Mat imsup= roi_image(r).clone(); // rectangle supérieur central
            r.height = 1; // ligne de 1 pixel
            r.y = 0;
            r.x = 0;
            r.width = imsup.cols;
            cv::Mat lig = imsup(r).clone();
            int mref = cv::mean(lig)[0]; // limité à la composante bleue
            int m;
            r.y++;
            while(r.y < imsup.rows) {
                lig = imsup(r).clone();
                m = cv::mean(lig)[0];
                if (m > mref + 20) {numcol = 3; break;} // tige de la feuille de trefle
                else if (r.y > imsup.rows - 2  && m < mref - 20) {numcol = 0; break;} // pique
                r.y++; mref = m;
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
            if (printoption && !threadoption) tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 255, 0));
            centre = roi_image(r);
            // amplifyContrast(centre);
            cv::meanStdDev(centre, moy, ect);
            if (printoption)
                std::cout << " P/T ? écart type " << ect << std::endl;
            if (ect[0] < 10)
                numcol = 0;
            if (ect[0] > 60)
                numcol = 3;
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
                haut = roi_image(r);
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
                haut = roi_image(r);
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

            if (printoption && !threadoption)
            {
                tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
                tracerRectangle(rc, symbgros, "gros", cv::Scalar(0, 255, 0));
            }

            haut = roi_image(r);
            centre = roi_image(rc);
            moyh = cv::mean(haut);
            moyc = cv::mean(centre);
            int moycent = moyc[0] + moyc[1] + moyc[2];
            int moyhaut = moyh[0] + moyh[1] + moyh[2];
            if (moyhaut == 0) moyhaut = 1; // ne pas diviser par 0
            int ecr = 100 * (moyhaut - moycent) / moyhaut;
            if (printoption)
                std::cout << " intensite centre " << moycent << ", haut  " << moyhaut << ", ecr " << ecr << std::endl;
            if (estgrossymb) { // on estime que l'ecart type est discriminant
                if (ecr > 0) {
                    numcol = 0;
                    if (printoption)  std::cout << "Pique";
                } else {
                    numcol = 3;
                    if (printoption)  std::cout << "Trefle";
                }
            } else { // on analyse le petit symbole
                if (std::abs(ecr) <= 10) { // zones haute et centrale peu différentes
                    //  examiner les zones à gauche et à droite du rectangle central
                    cv::Mat gauche, droite;
                    cv::Scalar moyg, moyd;
                    r = rc;
                    r.x = Box[0];
                    gauche = roi_image(r);
                    moyg = cv::mean(gauche);
                    if (printoption && !threadoption)
                        tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
                    r.x = Box[1] + 1 - r.width;
                    droite = roi_image(r);
                    moyd = mean(droite);
                    if (printoption && !threadoption)
                        tracerRectangle(r, symbgros, "gros", cv::Scalar(255, 0, 0));
                    moyhaut = (moyg[0] + moyd[0] + moyg[1] + moyd[1] + moyg[2] + moyd[2]) / 2;
                    ecr = 100 * (moyhaut - moycent) / moyhaut;
                    if (printoption)
                        std::cout << " intensite centre " << moycent << ", bords  " << moyhaut << ", ecr " << ecr << std::endl;
                }
                if (ecr > 20) { // 20 expérimental, écart  net pour Pique
                    numcol = 0;
                    if (printoption)  std::cout << "Pique";
                } else if (ecr <= -10) { // centre plus clair
                    numcol = 3;
                    if (printoption)  std::cout << "Trefle";
                } else {
                    if (printoption)
                        std::cout << "couleur noire indéterminable Trefle ?" << ecr << std::endl;
                    cv::circle(result, cv::Point2i(cecoin[4], cecoin[5]), 4, cv::Scalar(0, 0, 255), -1);
                    {retourcoin(n); return;} 
                }
            }
            // si c'est un Valet, tester la couleur du chapeau
            // le chapeau est au milieu du petit coté de la carte
            // on teste la couleur d'un petit rectangle
            // rouge --> pique, bleu --> trefle
            // désactivé car ce n'est pas vrai pour tous les jeux de cartes
            // TODO : ajouter un indicateur dans la configuration
            //
            // DESACTIVE
            if (false && estunRDV && outprec == "V")
            {
                int demilargeur = maconf.largeurcarte / 2; // approximatif
                if (inverse)
                {
                    r.height = maconf.largeurgrosRDV;
                    r.width = maconf.tailleVDR;
                    if (UU.x > PP.x)
                        r.x = QQ.x;
                    else
                        r.x = QQ.x - r.width;
                    if (U.y < PP.y)
                        r.y = PP.y - demilargeur - r.height / 2;
                    else
                        r.y = PP.y + demilargeur - r.height / 2;
                }
                else
                {
                    r.width = maconf.largeurgrosRDV;
                    r.height = maconf.deltacadre;
                    if (UU.x < PP.x)
                        r.x = PP.x - demilargeur - r.width / 2;
                    else
                        r.x = PP.x + demilargeur - r.width / 2;
                    if (U.y < PP.y)
                        r.y = QQ.y - r.height - maconf.deltacadre / 2;
                    else
                        r.y = QQ.y + maconf.deltacadre / 2;
                }
                cv::Mat ima_gs = coinPetit(r);
                cv::line(extrait, cv::Point2i(r.x, r.y), cv::Point2i(r.x, r.y + r.height), cv::Scalar(0, 0, 0));
                cv::line(extrait, cv::Point2i(r.x, r.y), cv::Point2i(r.x + r.width, r.y), cv::Scalar(0, 0, 0));
                cv::line(extrait, cv::Point2i(r.x, r.y + r.height), cv::Point2i(r.x + r.width, r.y + r.height), cv::Scalar(0, 0, 0));
                cv::line(extrait, cv::Point2i(r.x + r.width, r.y), cv::Point2i(r.x + r.width, r.y + r.height), cv::Scalar(0, 0, 0));
                if(printoption && !threadoption) {
                    afficherImage("Extrait", extrait);
                    if (waitoption > 2)
                        cv::waitKey(0);
                    else
                        cv::waitKey(1);
                }

                cv::meanStdDev(ima_gs, moy, ect);
                if (printoption > 1)
                    std::cout << "chapeau moy et ect " << moy << "," << ect << std::endl;

                if (moy[0] < moy[2])
                    numcol = 0; // pique
                else
                    numcol = 3; // trefle
            }
        }
    }

    // imaSymb = roi_image.clone();

    if (numcol < 0 || output == "")  {retourcoin(n); return;}  // couleur indéterminée
    std::string texte = "";
    if (numcol == 0) { texte = "P"; if (printoption) std::cout << " Pique ";}
    if (numcol == 1) { texte = "C"; if (printoption) std::cout << " Coeur ";}
    if (numcol == 2) { texte = "K"; if (printoption)  std::cout << " Carreau ";}
    if (numcol == 3) { texte = "T"; if (printoption)  std::cout << " trefle ";}
    coins[n][10] = numcol;

    if (!nonvu && outprec != "" && outprec != output)
        if (printoption) {
            std::cout << "detection incoherente " << output << " <> " << outprec << std::endl;
            std::cout << output << " confiance " << confiance << std::endl << std::endl;
        }
    texte += output;
    char valcarte = output[0];
    if (output == "10") coins[n][11] = 10;
    else if(valcarte == 'V') coins[n][11] = 11;
    else if(valcarte == 'D') coins[n][11] = 12;
    else if(valcarte == 'R') coins[n][11] = 13;
    else if (valcarte > '0' && valcarte <= '9' ) coins[n][11] = valcarte - '0';


    if (printoption)
        std::cout << std::endl;
    // if (waitoption > 1) cv::waitKey(0);  else cv::waitKey(1);// attendre
    if(printoption) std::cout<< "=> fin traitercoin "<< n<<" texte lu:"<< texte <<std::endl;
       // Protection du vecteur de résultats
    
    if (false) { // n'est plus utile, resultats n'étant plus employé
        if (threadoption) std::lock_guard<std::mutex> verrou (resultMutex);
        std::string txtnum;
        txtnum = texte + "#" + std::to_string(n);
        resultats.push_back(txtnum);
    }
    
    if (threadoption) {
        //--activeThreads;
        retourcoin(n); // Signal que l'on peut démarrer une nouvelle tâche
        //std::cout<<"Apres notify_one"<<std::endl;
    }
    return ;
}
