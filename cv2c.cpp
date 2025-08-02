// Copyright Jacques ROSSILLOL 2024
//
//
#define _USE_MATH_DEFINES
// #include <tesseract/baseapi.h>
// #include <leptonica/allheaders.h>

#ifdef _WIN32
#include <Windows.h>
#include <tchar.h>
#else
#include <algorithm>
#endif

#include <iostream>
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

#ifndef _WIN32
#include <thread> // pour std::thread
#include <atomic> // pour std::atomic
std::atomic<bool> is_window_open(true);
#endif

#include <vector>
#include <mutex>
#include <condition_variable>

void afficherResultat(cv::Mat result, cv::Point2i PT, std::string res);

// using std::max;
// using std::min;

int waitoption = 1;   // 0 : pas d'attente après affichages
                      // 1 : attendre après le traitement d'une frame
                      // 2 : attendre après le traitement de chaque coin
                      // 3 :attendre après affichage du symbole et du chiffre
int printoption = 2;  // 0 : ne pas imprimer
                      // 1 : imprimer les lignes, coins détectés, OCR
                      // 2 : imprimer les calculs d'intensités et écarts types
int threadoption = 1; // 0 : monotache
                      // 1 : autant que de coeurs
                      // n : nombre de sous-taches
std::string nomOCR = "tesOCR";

cv::Point2f computeIntersection(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
{
    // Calculer les vecteurs directionnels
    cv::Point2f d1 = p2 - p1;
    cv::Point2f d2 = p4 - p3;

    // Résoudre les équations paramétriques
    float denom = d1.x * d2.y - d1.y * d2.x;
    if (denom == 0)
    {
        throw std::runtime_error("Les lignes sont parallèles et ne se croisent pas.");
    }

    float t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / denom;
    cv::Point2f intersection = p1 + t * d1;

    return intersection;
}
cv::Point2i calculerInter(cv::Vec4i l1, cv::Vec4i l2)
{
    cv::Point2f pt = computeIntersection(cv::Point2f(l1[0], l1[1]), cv::Point2f(l1[2], l1[3]), cv::Point2f(l2[0], l2[1]), cv::Point2f(l2[2], l2[3]));

    cv::Point2i pti = cv::Point2i(pt.x + 0.5, pt.y + 0.5);
    return pti;
}

bool PointEntreDeux(cv::Point2i M, cv::Point2i P, cv::Point2i Q)
{
    // déterminer si la projection de M sur PQ est entre P et Q
    // calculer PM.PQ et comparer à PQ.PQ
    // PM.PQ < 0 : M hors du segment PQ , du coté P
    long pmpq = (M.x - P.x) * (Q.x - P.x) + (M.y - P.y) * (Q.y - P.y);
    if (pmpq < 0)
        return false;
    long pqpq = (Q.x - P.x) * (Q.x - P.x) + (Q.y - P.y) * (Q.y - P.y);
    if (abs(pmpq) > abs(pqpq))
        return false;
    return true;
}

int MAX_THREADS = std::thread::hardware_concurrency(); // Limite du nombre de sous-tâches actives
std::mutex mtx;                                        // Protection des accès concurrents
std::condition_variable cvar;                          // Synchronisation des sous-tâches
int activeThreads = 0;                                 // Nombre de sous-tâches en cours

///////////////////// principe de détermination de la dernière carte posée ///////////////////
// 1- extraire une image monochromatique. soit grise,
//      soit une couleur qui n'est pas dans le tapis de jeu : bleu si le tapis est vert
//      le fond devient alors noir
// 2- déterminer les limites (edges)
// 3- déterminer les droites correspondant aux limites d'une carte
//    lignes formées de beaucoup de pixels
//    on trouve les limites des cartes et les cadres des Rois Dame Valet
//    et des lignes diverses à l'inérieur des cartes Roi Dame Valet
//
// 4- déterminer les coins des cartes
//    un coin est l'intersection de deux droites perpendicullaires proche d'une extrémité de chaque droite
//    (en réalité le coin est arrondi, les deux droites ne sont pas concourrantes)
// 4-- rectifier l'extrémité de ligne des lignes formant le coin
// 4-- éliminer les coins proches d'un autre et dans l'angle de cet autre
//
//     on peut trouver plusieurs coins pour chaque carte
//     on peut trouver des coins parasites (deux lignes de deux cartes différentes)
//
// 5- pour chaque coin retenu
//   - extraire la partie de l'image qui est dans ce coin
//     déterminer si c'est une carte rouge ou noire à partir de l'extrait de l'image originale
//   - utiliser un outil OCR  pour déterminer le chiffre ou la lettre
//   - si on a reconnu un chiffre ou 10, la carte comporte des gros dessins Pique Coeur Carreau ou Trefle
//      ce qui permettra de distinguer plus facilement entre Pique et Trefle
//   - la couleur des pixels dans la zone du symbole de couleur permet de distinguer les rouges (C K) des noirs (P T)
//   - distinguer coeur et carreau selon la parie supérieure du symbole
//     distinguer pique et trefle selon l'intensité de la partie centrale

int processFrame(config &maconf, cv::Mat frame, bool estvideo, int *nbcoins,  int lescoins[500][10]);
int processVideo(config &maconf, cv::String nomfichier)
{
    cv::Mat img = cv::imread(nomfichier);
    if (!img.empty())
    {
        int nbcoins= 0;
        int lescoins[1][10];
        processFrame(maconf, img, false, &nbcoins, lescoins);
        return 0;
    }

    // Ouvrir le fichier vidéo
    cv::VideoCapture cap(nomfichier);
    if (!cap.isOpened())
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier vidéo " << nomfichier << std::endl;
        return -1;
    }

    // Lire et afficher les frames 1 sur 25
    int nbf = 0;
    cv::Mat frame;
    cv::Mat frameW;
    cv::Mat framePrec;
    cv::Mat image;
    cv::Mat diff;
    bool bPremier = true;
    int lescoins[500][10];   // très largement suffisant
    int nbcoins = 0;
    // 0 : 1=valide, 0=non utilisé
    // 1 : X du coin
    // 2 : Y du coin
    // 3 : couleur 0=Pique, 1=Coeur, 2=Carreau, 3=Trefle, -1=indéterminé ou artefact
    // 4 : valeur  1=As, ... 9=9, 10=10, 11=Valet, 12=Dame, 13=Roi
    // 5 à 9 : inutilisé
    while (true)
    {
        cap >> frame; // Capture une frame
        if (frame.empty())
        {
            break; // Arrêter si aucune frame n'est capturée
        }
        if (nbf == 0)
        {
            nbf = 1;   // écart entre une frame et la suivante
            if (printoption)
                cv::imshow("Frame", frame); // Afficher la frame
            // comparer à la frame précédente
            // extraire la partie modifiée (la première fois : tout)
            // conserver le tableau des coins identifiés
            //   pour chaque coin : position, couleur et valeur carte
            // invalider les coins sur une zone modifiée
            // extraire l'image modifiée
            // traiter cette image en ajoutant les nouvaux coins
#ifdef ACTIVER
            cv::cvtColor(frame, frameW, cv::COLOR_BGR2GRAY);
            if (bPremier) { bPremier = false; image = frame.clone();}
            else {
                // extraire l'image modifiée
                cv::absdiff(framePrec, frameW, diff);
                cv::threshold(diff, diff, 30, 255, cv::THRESH_BINARY);
                // nettoyer le tableau des coins
                for (int n = 0; n <= nbcoins; n++){
                    if (lescoins[n][0] == 0) continue;
                    if (diff.at<uchar>(lescoins[n][2], lescoins[n][1]) >= 250){
                        lescoins[n][0] = 0;
                    }
                }
                // déterminer le rectangle modifié
                int xmin(diff.cols) , xmax(0), ymin(diff.rows), ymax(0);
                for (int y = 0; y< diff.rows; y++){
                    for (int x = 0; x < diff.cols; x++){
                        if (diff.at<uchar>(y,x) == 255)
                        xmin = std::min(xmin, x);
                        xmax = std::max(xmax, x);
                        ymin = std::min(ymin, y);
                        ymax = std::max(ymax, y);
                    }
                }
                cv::Rect r (xmin, ymin, xmax+1-xmin, ymax+1-ymin);
                image = frame(r).clone();
            }
            if (image.cols > 0 && image.rows > 0) {
                if (diff.cols > 0 && diff.rows > 0) {
                    cv::imshow("diff", diff);
                    if (waitoption) cv::waitKey(0);
                }
                // autre stratégie : on comparera les coins trouvés aux coins trouvés précédemment
                // processFrame(maconf, image, true, &nbcoins, lescoins);
                processFrame(maconf, frame, true, &nbcoins, lescoins);
                framePrec = frameW.clone();
                // Attendre 30 ms et quitter si 'q' est pressé
                if (cv::waitKey(30) == 'q')
                {
                    break;
                }
            }
#endif  
            processFrame(maconf, frame, true, &nbcoins, lescoins);
        }
        nbf--;
    }

    cap.release(); // Libérer la capture vidéo
    // cv::destroyAllWindows(); // Fermer toutes les fenêtres ouvertes
    return 0;
}

int main(int argc, char **argv)
{
    config maconf;
    std::cout << " arguments optionels :"
              << " nom du fichier image ou video, "
              << " nom du fichier de configuration, "
              << " taille de carte (en pixels) " << std::endl
              << std::endl;

    std::string nomfichier;
    nomfichier = setconfig(maconf); // initialisation par défaut

    if (argc > 1)
        nomfichier = argv[1];
    std::string nomconf; // nom du fichier de configuration
    if (argc > 3)
        maconf.hauteurcarte = std::stoi(argv[3]);
    else
    {
        size_t pos1 = nomfichier.find('_');
        size_t pos11 = nomfichier.find('_', pos1 + 1);
        if (pos11 != std::string::npos)
            pos1 = pos11;
        size_t pos2 = nomfichier.find('.', pos1);

        if (pos1 != std::string::npos && pos2 != std::string::npos)
        {
            // Extraire la sous-chaîne
            std::string extracted = nomfichier.substr(pos1 + 1, pos2 - pos1 - 1);
            maconf.hauteurcarte = std::stoi(extracted);
            // std::cout << "Sous-chaîne extraite : " << extracted << std::endl;
        }
        else
        {
            std::cout << "Délimiteurs non trouvés" << std::endl;
        }
    }
    if (argc > 2)
        nomconf = argv[2];
    else
        nomconf = "COFALU.txt";
    lireConfig(nomconf, maconf);
    // else  if (maconf.hauteurcarte != 0) resetconfig(maconf.hauteurcarte, maconf);
    waitoption = maconf.waitoption;
    printoption = maconf.printoption;
    threadoption = maconf.threadoption;
    if (maconf.tesOCR == 0)
        nomOCR = "SERVEUR";
    else
        nomOCR = "tesOCR";
    return processVideo(maconf, nomfichier);
}


// TODO : comparer l'image à l'image précédente, si on traite une vidéo
//    après le traitement d'une frame, conserver le résultat du décodage
//     qui se trouve dans le tableau des coins
//    traitement de la nouvelle frame:
//    comparer à la frame précédente. on obtient les pixels modifiés
//    invalider les résultats de chaque coin sur une zone modifiée
//    restreindre l'image à analyser à la partie modifiée  

int processFrame(config &maconf, cv::Mat image, bool estvideo, int *pnbcoins, int lescoins[500][10])
{
    activeThreads = 0;
    if (maconf.threadoption > 1)
        MAX_THREADS = maconf.threadoption;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::string> resultats; // vecteur des résultats
    std::vector<std::thread> threads;

#define NBCOULEURS 10
    cv::Scalar couleurs[10];
    couleurs[0] = cv::Scalar(255, 128, 128); // bleu
    couleurs[1] = cv::Scalar(128, 255, 128); // vert
    couleurs[2] = cv::Scalar(128, 128, 255); // rouge
    couleurs[3] = cv::Scalar(255, 255, 128); // turquoise
    couleurs[4] = cv::Scalar(255, 128, 255); // violet
    couleurs[5] = cv::Scalar(128, 255, 255); // jaune
    couleurs[6] = cv::Scalar(255, 255, 255); // blanc
    couleurs[7] = cv::Scalar(255,0,0); // bleu foncé
    couleurs[8] = cv::Scalar(0,255,0); // vert foncé
    couleurs[9] = cv::Scalar(0,0,255); // rouge foncé
    int c = 0;

    if (image.empty())
    {
        std::cerr << "Erreur de chargement de l'image" << std::endl;
        return -1;
    }
    cv::Mat result = image.clone();
    auto start = std::chrono::high_resolution_clock::now();

    // afficher l'image en couleurs
    if (printoption)
        cv::imshow("couleur", image);

    // Séparer les canaux Bleu, Vert, Rouge
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(image, bgrChannels);
    // Utiliser seulement le canal Vert (par exemple)
    cv::Mat greenChannel = bgrChannels[1];
    // cv::imshow("vert", greenChannel);
    cv::Mat blueChannel = bgrChannels[0];
    // cv::imshow("bleu", blueChannel);
    cv::Mat redChannel = bgrChannels[2];
    // cv::imshow("rouge", redChannel);

    // Convertir en niveaux de gris
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Appliquer le flou gaussien pour réduire le bruit
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    // cv::imshow("blur", blurred);

    ////////////////// utiliser une des images monochromatiques /////////////////
    cv::Mat grise;
    cv::cvtColor(gray, grise, cv::COLOR_GRAY2BGR);
    if (printoption)
        cv::imshow("grise", grise);

    std::vector<ligne> lignes;
    std::vector<cv::Vec4i> lines;
    //std::vector<cv::Vec4d> lneq; // équation de droite : abc lg ax+by+c = 0 
    int gmin = maconf.gradmin;
    int gmax = maconf.gradmax;

    int methode = 2; // 1 : canny et HoughLines,   2: ximgproc
    if (methode == 2)
    {
        // Appliquer le détecteur de segments de ligne LSD
        std::vector<cv::Vec4f> lines_f;

        // Paramètres du FastLineDetector : longueur minimale, écart entre lignes, etc.
        int length_threshold = maconf.nbpoints; // Longueur minimale d'une ligne
        float distance_threshold = 1.41421356f; // Distance maximale entre deux points formant une ligne
        // float distance_threshold = 1.5f; // Distance maximale entre deux points formant une ligne
        double canny_th1 = gmin;     // Seuil bas pour Canny
        double canny_th2 = gmax;     // Seuil haut pour Canny
        int canny_aperture_size = 3; // Taille de l'ouverture pour Canny
        bool do_merge = true;       // ne pas Fusionner les lignes adjacentes ( // )

        cv::Ptr<cv::ximgproc::FastLineDetector> lsd = cv::ximgproc::createFastLineDetector(
            length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, do_merge);

        lsd->detect(gray, lines_f);

        // Dessiner les segments de ligne détectés
        cv::Mat result;
        cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
        // lsd->drawSegments(result, lines_f); // plus loin pour plusieurs couleurs
        // Convertir les coordonnées des lignes en entiers
        // calculer la longueur et l'équation cartésienne

        int ic = 0;
        for (size_t i = 0; i < lines_f.size(); i++)
        {
            ic++; if (ic >= NBCOULEURS) ic = 0;
            ligne ln;
            cv::Vec4i l(cvRound(lines_f[i][0]), cvRound(lines_f[i][1]), cvRound(lines_f[i][2]), cvRound(lines_f[i][3]));
            ln.ln = l;

            cv::Point A(l[0], l[1]);
            cv::Point B(l[2], l[3]);
            // tracer la ligne sur l'image result
            if(printoption) cv::line(result, A, B, couleurs[ic], 1);
            float lg = std::sqrt((l[2] - l[0])*(l[2] - l[0]) + (l[3] - l[1])*(l[3] - l[1]));
            // vecteur normal (a,b) directeur (b, -a)  
            double a = -(B.y - A.y) / lg;
            double b = (B.x - A.x) / lg;
            float c = -a*A.x - b*A.y; // ax + by + c = 0
            ln.lg = lg;
            ln.a = a;
            ln.b = b;
            ln.c = c;
            lignes.push_back(ln);
        }
        // Afficher l'image avec les lignes détectées
        if (printoption)
            cv::imshow("ximgproc", result);
        auto t11 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree = t11 - t0;
        if(printoption)
         std::cout << "Duree de detection des lignes : " << duree.count() << " secondes" << std::endl;
        if (waitoption > 1)
            cv::waitKey(0);
    }
    auto t22 = std::chrono::high_resolution_clock::now();
    cv::Mat edges;
    int iwait = 1;
    cv::Mat ima2;
    ima2 = grise.clone();
    cv::Canny(ima2, edges, gmin, gmax, 3, false);
    // cv::Canny(gray, edges, gmin, gmax, 3, false);
    if (printoption)
        cv::imshow("bords", edges);
    cv::waitKey(iwait);

    if (methode == 1)
    {
        // Utiliser la détection de contours de Canny
        // grossir l'image (désactivé)
        // canny (image, gradiant mini, gradiant maxi, ouverture)
        // gradient : variation d'intensité entre 2 pixels voisins
        // gradient mini : si le gradient calculé est inférieur, ce n'est pas un bord
        // gradiant maxi : si le gradient calculé est supérieur, c'est un bord

        // agrandir l'image, pour le calcul des lignes
        double mult = 1.0; // meilleure valeur.
        // cv::resize(gray, gray, cv::Size(), mult, mult);

        ///////////////// identifier les lignes de bord des cartes (grandes) /////////////////

        // Utiliser la transformation de Hough pour détecter les segments de droite
        // https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

        //
        // résolution de la distance de la droite à l'origine. 1 pxel
        // résolution angulaire de la normale à la droite
        // nombre minimal de courbes qui déterminent la droite
        // nombre minimal de points sur la droite
        // écart maximal entre deux pixels sur la droite
        double theta = CV_PI / 360;
        int threshold = maconf.nbpoints;
        double gap = maconf.ecartmax;
        double minlg = maconf.nbpoints;
        cv::HoughLinesP(edges, lines, 1, theta, threshold, minlg, gap);
        // cv::HoughLinesP(gray, lines, 1, CV_PI / 360, maconf.nbvote, maconf.nbpoints, maconf.ecartmax); // ne fonctionne pas
        // on refait le calcul des bords pour la suite
        cv::Canny(ima2, edges, gmin, gmax, 3, false);
        if (printoption)
            cv::imshow("bords", edges);
        cv::waitKey(iwait);

        // les lignes sont agrandies
        //
        for (int i = 0; i < lines.size(); i++)
        {
            cv::Vec4i l = lines[i];
            lines[i][0] /= mult;
            lines[i][1] /= mult;
            lines[i][2] /= mult;
            lines[i][3] /= mult;
        }
    }

    // Dessiner les segments de droite et afficher leurs longueurs et extrémités
    //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < ima2.rows; y++)
        for (int x = 0; x < ima2.cols; x++) ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
    /***************/

    c = 0; // indice de couleur
    double maxlg = 0;
    int il1 = 0;
    int nblignes = lignes.size();
    for (int i = 0; i < nblignes; i++)
    {
        cv::Vec4i l = lignes[i].ln;

        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        cv::line(ima2, A, B, couleurs[c], 1);
        c++;
        if (c >= NBCOULEURS) c = 0;
        //double lg = lneq[i][3];
        float lg = lignes[i].lg;
        if (printoption)
            std::cout << "Ligne " << i <<" "<< A << "->" << B << " Longueur: " << lg << std::endl;
        if (lg > maxlg) {
            maxlg = lg;
            il1 = i; // indice de la ligne la plus longue
        }
    }

    // Afficher l'image avec les segments de droite
    if (printoption)
    {
        cv::imshow("Lignes toutes", ima2);
        cv::waitKey(1);
    }

    int lgmax = maconf.taillechiffre;
    // fusionner les lignes AB  et CD si // si C et D sont proches de la ligne AB
    //   et si C ou D est proche de A ou B : AB --> AC ou AD ou BC ou BD
    if (true) {
        double epsilon = 1.2; // à peine plus qu'un pixel d'écart entre les deux lignes #//
        double deltamax = 1;
        for (int k = 0; k < 5; k++) { 
            // fusionner des lignes fusionnées, de plus en plus distantes
            deltamax = k + 1;
            for (int i = 0; i < nblignes; i++)
            {
                ligne ln = lignes[i];
                cv::Vec4i l = ln.ln;
                if (l[0] < 0)   continue; // ligne invalidée
                cv::Point2i A(l[0], l[1]);
                cv::Point2i B(l[2], l[3]);
                float lg1 = ln.lg;
                float a = ln.a;
                float b = ln.b;
                float c = ln.c;
                for (int j = i + 1; j < nblignes; j++)
                {
                    // fusionner la ligne la plus courte sur la plus longue
                    ligne ln2 = lignes[j];
                    cv::Vec4i ll = ln2.ln;
                    if (ll[0] < 0)  continue; // ligne invalidée
                    cv::Point2i C(ll[0], ll[1]);
                    cv::Point2i D(ll[2], ll[3]);
                    float lg2 = ln2.lg;
                    if (lg1 > lg2)
                    {
                        // distances de C ou D à AB > epsilon à préciser --> ignorer la ligne j
                        float dC = a*C.x + b*C.y + c;
                        if (abs(dC) > epsilon)   continue;
                        float dD = a*D.x + b*D.y + c;
                        if (abs(dD) > epsilon)  continue;
                    }
                    else
                    {
                        float dA = A.x*ln2.a + A.y*ln2.b + ln2.c;
                        if (abs(dA) > epsilon)  continue;
                        float dB = B.x*ln2.a + B.y*ln2.b + ln2.c;
                        if (abs(dB) > epsilon) continue;

                    }
                    // 4 points A B C D alignés. ignorer si l'écart entre AB et CD est important
                    //
                    int xmin, xmax, ymin, ymax;
                    if (std::abs(A.x - B.x) > std::abs(A.y - B.y))
                    {
                        xmin = std::min(A.x, B.x);
                        if (xmin > C.x && xmin > D.x) { // AB à droite de CD
                            if ((xmin - C.x) > deltamax && (xmin - D.x) > deltamax) continue; // segments loins
                        } else {
                            xmax = std::max(A.x, B.x);
                            if (C.x - xmax > deltamax && D.x - xmax > deltamax) continue; // CD à gauche de AB
                        }
                    } else { // Y plus variable que X
                        ymin = std::min(A.y, B.y);
                        if (ymin > C.y && ymin > D.y)  { // AB sous CD
                            if ((ymin - C.y) > deltamax && (ymin - D.y) > deltamax) continue; // segments loins
                        } else {
                            ymax = std::max(A.y, B.y);
                            if (C.y - ymax > deltamax && D.y - ymax > deltamax) continue; // CD au dessus de AB
                        }
                    }
                    // déterminer les extrémités après fusion : abs mini - abs maxi  // ord mini - maxi
                    // utiliser x ou y
                    cv::Point2i U(A), V(A); // futures extrémités
                    if (std::abs(A.x - B.x) > std::abs(A.y - B.y)) { // X plus variable
                        if (U.x > B.x) U = B;
                        if (U.x > C.x) U = C;
                        if (U.x > D.x) U = D;
                        if (V.x < B.x) V = B;
                        if (V.x < C.x) V = C;
                        if (V.x < D.x) V = D;
                    } else {  // Y plus variable
                        if (U.y > B.y) U = B;
                        if (U.y > C.y) U = C;
                        if (U.y > D.y) U = D;
                        if (V.y < B.y) V = B;
                        if (V.y < C.y) V = C;
                        if (V.y < D.y) V = D;
                    }
                    // remplacer AB par UV
                    // ne rien faire si la nouvelle ligne serait plus grande que la hauteur de carte
                    int lg2uv = (V.x - U.x) * (V.x - U.x) + (V.y - U.y) * (V.y - U.y);
                    if (lg2uv < maconf.hauteurcarte * maconf.hauteurcarte)
                    {
                        // et invalider la ligne CD
                        lignes[i].ln[0] = U.x; lignes[i].ln[1] = U.y;
                        lignes[i].ln[2] = V.x; lignes[i].ln[3] = V.y;
                        if (printoption > 1){
                            std::cout<<" ligne "<< i << " "<<A<<B<<" --> "<<U<<V<<std::endl;
                            std::cout<<" ligne "<<j<<" supprimee"<<std::endl;
                            std::cout<<"verif "<<lines[i]<<std::endl;
                        }
                        A = U;
                        B = V;
                        // invalider la ligne j
                        lignes[j].ln[0] = -1;
                        ll[0] = -1;
                        // mettre à jour la longueur de la ligne i = AB
                        lg1 = std::sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
                        lignes[i].lg = lg1;
                    }
                } // next j
            } // next i
        } //k écart suivant

    }

    // prolonger les lignes
    if (methode == 1)  {
        // prolonger les lignes assez longues (au moins 1/6 de la hauteur de carte)
        // essayer de prolonger chaque ligne : regarder le pixel dans le prolongement de la ligne
        // ligne AB (B à droite de A) choisir une direction x ou y selon le maximum de |dx| et |dy|
        // AB selon X , prolongement en B : regarder le pixel blanc (dans edges) à droite (B.x +1, B.y)
        //   et le pixel blanc  à droite plus haut ou plus bas (B.x +1, B.y +- 1) (le plus proche de AB)
        //   à condition que les autres pixels proche de B soient noirs (dans edge)
        // choisir le plus proche de AB, à distance de moins de 2 pixels de AB,  qui remplace B
        // même principe du coté A
        // itérer tant qu'on trouve des pixels blancs dans l'image des bords et noirs dans l'affichage des lignes
        double tolerance = 0.4; // Ajustez la tolérance selon vos besoins. 0.4 entre 45 et 60 degrés
        cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8U);
        int maxlg = maconf.hauteurcarte / 6;
        maxlg *= maxlg;
        for (int i = 0; i < lines.size(); i++)
        {
            cv::Vec4i l = lines[i];
            if (l[0] < 0)
                continue; // ligne invalidée
            cv::Point2i A(l[0], l[1]);
            cv::Point2i B(l[2], l[3]);
            int lgAB = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
            if (lgAB < maxlg)
                continue;
            // prolonger la ligne en A

            std::vector<cv::Point2i> contour;
            // on commence par prolonger en A
            // puis en B
            followContour(edges, A, B, contour, tolerance);
            // Obtenir l'extrémité du contour
            if (!contour.empty())
            {
                cv::Point2i Z = contour.back();
                // std::cout << "L'extremite du contour est  (" << Z.x << ", " << Z.y << ")" << std::endl;
                //  remplacer A par Z si A est entre B et Z
                //  sinon, si B est entre Z et A, remplacer B par Z
                cv::Point2i ab = B - A;
                cv::Point2i az = Z - A;
                int ps = ab.x * az.x + ab.y * az.y;
                if (ps <= 0)
                { // A entre B et Z : remplacer A par Z
                    if (printoption > 1)
                        std::cout << i << " on remplace A " << A << " par " << Z << std::endl;
                    A = Z;
                }
            }
            else
            {
                if (printoption > 1)
                    std::cout << i << " Aucun contour trouve en A." << A << std::endl;
            }
            // prolonger en B
            int sz1 = contour.size();
            followContour(edges, B, A, contour, tolerance);
            // Obtenir l'extrémité du contour
            int sz2 = contour.size();
            if (sz2 > sz1)
            { // on a ajouté au moins un point
                cv::Point2i Z = contour.back();
                // std::cout << "L'extremite du contour est  (" << Z.x << ", " << Z.y << ")" << std::endl;
                //  remplacer B par Z si A est entre B et Z
                if (printoption > 1)
                    std::cout << i << " on remplace B " << B << " par " << Z << std::endl;
                B = Z;
            }
            else
            {
                if (printoption > 1)
                    std::cout << i << "Aucun contour trouve en B." << B << std::endl;
            }
            if (printoption > 1)
            {
                cv::imshow("Contour", contourImage);
                cv::waitKey(1);
            }
            lines[i][0] = A.x;
            lines[i][1] = A.y;
            lines[i][2] = B.x;
            lines[i][3] = B.y;
            for (const auto &P : contour)
            {
                contourImage.at<uchar>(P) = 255;
            }
        }

        // cv::waitKey(0);
    }

    // invalider les lignes dont la longueur est inférieure à la taille du chiffre + symbole
    // test :éliminer les ligne de longueur inférieure à la moitié de hauteur de carte
    // éliminer les lignes plus longues que la hauteur de carte
    // modif 2025/06/11 : on conserve les lignes longues à cause du mort
    lgmax = maconf.taillechiffre + maconf.taillesymbole; // limite inférieure
    // lgmax = maconf.hauteurcarte / 6;   // test à valider
    int lgmin = maconf.hauteurcarte + maconf.deltacadre;
    for (int i = 0; i < lignes.size(); i++)
    {
        ligne ln = lignes[i];
        cv::Vec4i l = ln.ln;
        if (l[0] < 0)
            continue; // ligne invalidée
        cv::Point2i A(l[0], l[1]);
        cv::Point2i B(l[2], l[3]);
        double lg1 = ln.lg;
        if ((lg1 < lgmax) /*/ ||  (lg1 > lgmin) */)
        {
            lignes[i].ln[0] = -1; // invalider la ligne
            if (printoption > 1)
                std::cout << "supprime la ligne " << i << " " << A << "-" << B << " longueur " << lg1 << std::endl;
        }
    }

    //
    // afficher les lignes qui restent
    for (int y = 0; y < ima2.rows; y++) for (int x = 0; x < ima2.cols; x++)
        ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
    c = 0;
    maxlg = 0;
    il1 = 0;
    for (int i = 0; i < lignes.size(); i++)
    {
        ligne ln = lignes[i];
        cv::Vec4i l = ln.ln;
        if (l[0] < 0)  continue; // ligne fusionnée ou ignorée car trop courte
        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        cv::line(ima2, A, B, couleurs[c], 1);
        c++;
        if (c >= NBCOULEURS)  c = 0;
        float length = ln.lg;
        if (printoption > 1)
         std::cout << "Ligne " << i <<" "<< A << "->" << B << " Longueur: " << length << std::endl;
        if (length > maxlg)
        {
            maxlg = length;
            il1 = i;
        }
    }
    if (printoption)
        std::cout << "longueur maximale " << maxlg << std::endl;
    // Afficher l'image avec les segments de droite
    if (printoption)
    {
        cv::imshow("Lignes", ima2);
        cv::waitKey(1);
    }

    //////////////////////////////// rechercher les coins des cartes ///////////////////
    //
    // 500 coins au maximum
    // indice de ligne 1, ligne2, indicateur d'extrémité de la ligne 1 (0 ou 2), ligne 2, x et y du point d'intersection
    int coins[500][12]; // mémoriser tous les coins trouvés
    // O : indice de la première ligne
    // 1  : indice de la deuxième ligne
    // 2  : indicateur de sommet commun de la ligne 1  = 0 ou 2  (indice dans la ligne pour x
    // 3  : indicateur ligne 2
    // 4  : x du point d'intersection
    // 5  : y
    // 6  : indicateur de coin d' un Roi Dame ou Valet  ( = 0 sinon)
    // 7  : x du cadre si R D V
    // 8  : y
    // 9  : numéro de carte
    // 10 : couleur 0=pique, 1=coeur, 2=carreau, 3=trefle
    // 11 : valeur 1=As, ... 10=10, 11=Valet, 12=Dame, 13=Roi
    //      utilisé pour indiquer si le premier coté est la longueur ou largeur de la carte
    //      -1 : longueur, -2 : largeur, autre(0) : pas de coin opposé 
    int nbcoins = 0;
    int nbcartes = 0;
    // pour chaque ligne AB
    for (int i = 0; i < nblignes; i++)
    {
        ligne ln = lignes[i];
        cv::Vec4i l1 = ln.ln;
        if (l1[0] < 0)   continue; // ligne fusionnée ou effacée
        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        float lg1 = ln.lg;
        if (printoption > 1)
          std::cout << i << " Ligne AB " << A << B << " Longueur: " << lg1 << std::endl;
        il1 = i;
        float a = ln.a; // vecteur normal de la droite AB
        float b = ln.b;
        //
        // chercher, parmi les autres lignes la ligne orthogonale CD dont une extremité (C ou D)  est proche de A ou B
        // TODO: ou proche de la ligne AB entre A et B. ou dont A ou B est proche de la ligne CD entre C et D
        
        // TODO : multithread 
        // initialiser une sous-tache pour calculer les coins sur la ligne AB
        // protéger l'ajout des coins
        
        
        for (int j = i + 1; j < nblignes; j++)
        {
            ligne ln2 = lignes[j];
            float psX;
            // ligne CD ortogonale à AB ?
            // calculer le produit scalaire des vecteurs normés AB x CD 
            cv::Vec4i l2 = ln2.ln;
            if (l2[0] < 0) continue; // ligne trop courte ou fusionnée à une autre
            cv::Point2i C(l2[0], l2[1]);
            cv::Point2i D(l2[2], l2[3]);
            float lg2 = ln2.lg;
            psX = a*ln2.a + b*ln2.b; // cosinus (AB, CD) = cosinus des normales
            if (std::abs(psX) > maconf.cosOrtho)  continue;  // lignes non approximativement orthogonales
            coins[nbcoins][0] = 0; // indice de ligne AB
            coins[nbcoins][1] = 0; // indice de ligne CD
            coins[nbcoins][2] = 0; // 0 si A proche du coin P = ABxCD     2 si B proche de P
            coins[nbcoins][3] = 0; // 0 si C proche de P    2 si D proche de P
            coins[nbcoins][4] = 0; // P.x
            coins[nbcoins][5] = 0; // P.y
            coins[nbcoins][6] = 0; //  1 c'est un Roi Dame ou Valet, sinon indéterminé
            coins[nbcoins][7] = 0; // Q.x   coin du cadre de R D ou V
            coins[nbcoins][8] = 0; // Q.y
            coins[nbcoins][9] = 0; // numéro de carte
            coins[nbcoins][10] = -1; // couleur 0=pique, 1=coeur, 2=carreau, 3=trefle, -1= indéterminé
            coins[nbcoins][11] = 0; // valeur 1=As, ... 11=Valet, 12=Dame, 13=Roi

            bool bCoin = false;
            // A proche de C ?
            if (std::abs(C.x - A.x) < maconf.deltacoin && std::abs(C.y - A.y) < maconf.deltacoin)
            { // A proche de C
                if (printoption)
                    std::cout << "  coin AC (" << A.x - C.x << "," << A.y - C.y << ") " << A << "," << C << std::endl;
                bCoin = true;
            }
            // A proche de D ?
            else if (std::abs(A.x - D.x) < maconf.deltacoin && std::abs(A.y - D.y) < maconf.deltacoin)
            { // A proche de D
                if (printoption)
                    std::cout << "  coin AD (" << A.x - D.x << "," << A.y - D.y << ") " << A << "," << D << std::endl;
                coins[nbcoins][3] = 2;
                bCoin = true;
            }
            // B proche de C ?
            else if (std::abs(B.x - C.x) < maconf.deltacoin && std::abs(B.y - C.y) < maconf.deltacoin)
            { // B proche de C
                if (printoption)
                    std::cout << "  coin BC (" << B.x - C.x << "," << B.y - C.y << ") " << B << "," << C << std::endl;
                bCoin = true;
                coins[nbcoins][2] = 2;
            }
            // B proche de D ?
            else if (std::abs(B.x - D.x) < maconf.deltacoin && std::abs(B.y - D.y) < maconf.deltacoin)
            { // B proche de D
                if (printoption)
                    std::cout << "  coin BD (" << B.x - D.x << "," << B.y - D.y << ") " << B << "," << D << std::endl;
                bCoin = true;
                coins[nbcoins][2] = 2;
                coins[nbcoins][3] = 2;
            }
            if (bCoin)
            {
                // calculer l'angle du complément # sinus = cosinus des normales
                double alfa = std::abs(psX) * 180.0 / 3.1416; // en degrés

                if (printoption) std::cout << "  angle " << alfa << " degres" << std::endl;
                //        mémoriser le coin : indices des deux droites et numéros des extrémités de chaque droite (0 ou 2)
                coins[nbcoins][0] = i; // indice première ligne
                coins[nbcoins][1] = j; // indice deuxième ligne
                cv::Point2i P = calculerInter(l1, l2);
                coins[nbcoins][4] = P.x;
                coins[nbcoins][5] = P.y;
                float length = lignes[j].lg;  // longueur CD
                if (printoption)
                    std::cout << " " << j << "  Ligne CD " << j << " " << C << "->" << D << " Longueur: " << length << std::endl;
                if (printoption)
                    std::cout << " ==> coin " << nbcoins << " en " << P << " " << i << " " << j << " k " << coins[nbcoins][2] << " kk " << coins[nbcoins][3] << std::endl;
                nbcoins++;
            }
        } // deuxième droite
    } // première droite
        auto t33 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duree33 = t33 - t22;
        if (printoption) std::cout << "Duree d'identification des coins : " << duree33.count() << " secondes" << std::endl;
    ////////////// on a déterminé les coins //////////////////////
    for (int i = 0; i < nbcoins; i++)
    {
        if (printoption)
            std::cout << "coin " << i << " : " << coins[i][4] << ", " << coins[i][5] << std::endl;
    }

    // déterminer la taille des cartes
    // déterminer les probables bords de carte
    // deux coins sur une même ligne (ou deux ligne // proches), à distance vraissemblable (paramètre général de configuration)
    // la plus grande distance serait la hauteur de carte, sauf si plusieurs cartes sont alignées
    // une des autres devrait être dans le rapport des cotés de carte ( 3 / 2 )
    //

    int htmax = 0;
    cv::Point2i P1, P2;
    for (int n = 0; n < nbcoins; n++)
    {
        int i = coins[n][0];
        int j = coins[n][1];

        if (i < 0 || j < 0) continue; // coin éliminé

        cv::Vec4i l1 = lignes[i].ln;
        cv::Vec4i l2 = lignes[j].ln;
        cv::Point2i A(coins[n][4], coins[n][5]);
        cv::Point2i H, K; // extremités non communes sur les deux lignes : coin AH,AK
        int k = coins[n][2];
        H.x = l1[2 - k];
        H.y = l1[3 - k];
        int kk = coins[n][3];
        K.x = l2[2 - kk];
        K.y = l2[3 - kk];
        // rechercher les coins opposés de la carte du coin n
        for (int m = n + 1; m < nbcoins; m++)
        {
            int ii = coins[m][0];
            int jj = coins[m][1];
            if (ii < 0 || jj < 0) continue; // coin éliminé

            cv::Vec4i l11 = lignes[ii].ln;
            cv::Vec4i l22 = lignes[jj].ln;
            cv::Point2i HH, KK; // coin A HH KK
            int k = coins[m][2];
            HH.x = l11[2 - k];
            HH.y = l11[3 - k];
            int kk = coins[m][3];
            KK.x = l22[2 - kk];
            KK.y = l22[3 - kk];

            cv::Point2i B(coins[m][4], coins[m][5]);
            // une des lignes commune avec une de l'autre coin?
            // le coin B doit être sur une des lignes du coin A
            // le coin A doit être sur une des lignes du coin B
            // les deux autres lignes doivent être // et de même sens
            // AB semble alors etre un bord de carte
            //
            bool estoppose = false;
            float epsilon = maconf.deltacadre / 2;
            // première ligne du coin n contient le sommet du coin m ?
            float dist = B.x * lignes[i].a + B.y + lignes[i].b + lignes[i].c;
            if (std::abs(dist) < epsilon) {
                // B proche de la ligne i
                // ligne ii ou jj // i ? produit vectoriel des normales
                float pv = lignes[i].a * lignes[ii].b  - lignes[i].b * lignes[ii].a;
                if (std::abs(pv) < maconf.deltaradian ) {
                    // i et ii confondus
                    // de sens opposé ? produit scalaire AH.BHH
                    float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
                    if (ps < 0) { // A et B opposés
                        // vérifier que K et KK sont du même coté de la droite i = ii
                        float d1 = lignes[i].a*K.x + lignes[i].b*K.y + lignes[i].c;
                        float d2 = lignes[i].a*KK.x + lignes[i].b*KK.y + lignes[i].c;
                        if (d1*d2 > 0){ //cotés non communs de même orientation
                            estoppose = true;
                        } 
                    }
                } else { // i et ii non //
                    // i et jj confondus ?
                    double pv = lignes[i].a * lignes[jj].a  - lignes[i].b * lignes[jj].b;
                    if (std::abs(pv) < maconf.deltaradian ) {
                        // i et jj confondus
                        // de sens opposé ? produit scalaire AH.BKK
                        float ps = (H.x - A.x)*(KK.x - B.x) + (H.y - A.y)*(KK.y - B.y);
                        if (ps < 0) { // A et B opposés
                            // vérifier que K et HH sont du même coté de la droite i = jj
                            float d1 = lignes[i].a*K.x + lignes[i].b*K.y + lignes[i].c;
                            float d2 = lignes[i].a*HH.x + lignes[i].b*HH.y + lignes[i].c;
                            if (d1*d2 > 0){ //cotés non communs de même orientation
                                estoppose = true;
                            } 
                        }
                    }
                }
            } else // B pas proche de la ligne i. proche de la ligne j ?
            if (std::abs(B.x * lignes[j].a + B.y + lignes[j].b + lignes[j].c < epsilon)) { 
                // B proche de la droite j
                // ligne ii ou jj // j ? produit vectoriel des normales
                float pv = lignes[j].a * lignes[ii].b  - lignes[j].b * lignes[ii].a;
                if (std::abs(pv) < maconf.deltaradian ) {
                    // j et ii confondus
                    // de sens opposé ? produit scalaire AK.BHH
                    float ps = (K.x - A.x)*(HH.x - B.x) + (K.y - A.y)*(HH.y - B.y);
                    if (ps < 0) { // A et B opposés
                        // vérifier que H et KK sont du même coté de la droite j = ii
                        float d1 = lignes[j].a*H.x + lignes[j].b*H.y + lignes[j].c;
                        float d2 = lignes[j].a*KK.x + lignes[j].b*KK.y + lignes[j].c;
                        if (d1*d2 > 0){ //cotés non communs de même orientation
                            estoppose = true;
                        } 
                    }
                } else { // j et ii non //
                    // j et jj confondus ?
                    double pv = lignes[j].a * lignes[jj].a  - lignes[j].b * lignes[jj].b;
                    if (std::abs(pv) < maconf.deltaradian ) {
                        // j et jj confondus
                        // de sens opposé ? produit scalaire AH.BH
                        float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
                        if (ps < 0) { // A et B opposés
                            // vérifier que K et KK sont du même coté de la droite j = jj
                            float d1 = lignes[j].a*H.x + lignes[j].b*H.y + lignes[j].c;
                            float d2 = lignes[j].a*HH.x + lignes[j].b*HH.y + lignes[j].c;
                            if (d1*d2 > 0){ //cotés non communs de même orientation
                                estoppose = true;
                            } 
                        }
                    }
                }
            }
            if (estoppose){
                float lg = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
                lg = std::sqrt(lg);
                if (lg > htmax && lg < maconf.hauteurcarte + maconf.deltacoin)
                {
                    P1 = A;
                    P2 = B;
                    htmax = lg;
                }
                continue;
            }
        } // for m
        // if (htmax < 8 * maconf.hauteurcarte / 10) htmax = maconf.hauteurcarte;
    }
    // recalculer les paramètres de position sur la carte
    double htcard;
    if (maconf.hauteurcarte == 0)
        htcard = htmax;
    else
        htcard = maconf.hauteurcarte;
    if (abs(htmax - htcard) < maconf.deltacoin)
        htcard = htmax;
    if (htcard)
    {
    } // resetconfig(htcard, maconf);
    else
    {
        if (printoption)
            std::cout << " !!!!! impossible d'estimer la taille des cartes" << std::endl;
        if (printoption)
            std::cout << " !!!!! poursuite avec la configuration " << std::endl;
    }
    //
    if (printoption)
        std::cout << "hauteur carte : " << htcard << std::endl;

    // TODO : pour chaque coin, rechercher les deux coins adjacents de la carte.
    //        créer les coins adjacents des lignes, même si une des deux est courte, correspondent

    ////////////////////////// éliminer les artefacts /////////////////////////////

    //

    // faire le tri parmi les coins détectés
    //    pour chaque coin et pour chaque droite
    //       rechercher parmi les autres coins une droite // proche
    //       si on trouve éliminer le coin (le point) situé dans l'autre
    // pour chaque coin AB x CD
    // éliminer les coins A'B' x C'D' qui vérifient
    //     la droite A'B' est // AB ou // CD  et à faible distance
    //         et le milieu de A'B' est dans l'angle AB x CD
    //  ou la droite C'D' est // AB ou // CD et proche
    //         et le milieu de C'D' est dans l'angle AB x CD
    c = 0;
    for (int n = 0; n < nbcoins; n++)
    {
        int i = coins[n][0]; // indice de la première ligne
        int j = coins[n][1]; // indice de la deuxième ligne
        // if (i < 0 || j < 0) continue; // coin déjà éliminé // !!! poursuivre pour éliminer ce que P contient

        cv::Vec4i l1 = lignes[abs(i)].ln;
        cv::Vec4i l2 = lignes[abs(j)].ln;
        // point d'intersection des droites l1 et l2
        cv::Point2i P(coins[n][4], coins[n][5]);
        // déterminer le triangle RPS du coin
        int k = coins[n][2];  // 0 pour origine A, 2 pour extrémité B de la première ligne
        int kk = coins[n][3]; // 0 pour origine C, 2 pour extrémité D de la deuxième ligne
        k = 2 - k;            // pour indexer l'extrémité loin de P
        kk = 2 - kk;

        cv::Point2i R(l1[k], l1[k + 1]);
        cv::Point2i S(l2[kk], l2[kk + 1]);

        if (printoption){
            std::cout << "Coin " << n << " " << P << " , " << R << " , " << S << std::endl;
            if (coins[n][9] > 0 ) std::cout<<" --> carte numero "<<coins[n][9]<<std::endl;

        }

        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        cv::Point2i C(l2[0], l2[1]);
        cv::Point2i D(l2[2], l2[3]);
        double pvPRS = (R.x - P.x) * (S.y - P.y) - (R.y - P.y) * (S.x - P.x); // produit vectoriel PR ^ PS inversé car repère inversé. négatif sens trigo
        // TODO
        //       !!! reporter ce test àprès l'élimination des coins internes
        //       !!! donc dans une nouvelle boucle sur les coins conservés
        //       déterminer si un des cotés est le cadre d'un honneur,
        //       en cherchant une ligne // à distance convenable (deltacadre) à l'extérieur
        //       repositionner le coin, associer les cotés, en créant des lignes au besoin
        //       èliminer éventuellement les coins redondants
        //
        //       déterminer si un coté est bordé à l'intérieur par une ligne // (à 1 pixel)
        //       choisir cette ligne pour le coin et recalculer R ou S

        bool trouveQ = false;
        bool QdansP = false;
        bool eliminerP; // éliminer P après recherche de tous les coins contenus dans P
        eliminerP = false;
        int dc = maconf.deltacadre;
        for (int m = n + 1; m < nbcoins; m++)
        {
            int ii = coins[m][0];
            int jj = coins[m][1];
            if (ii < 0 || jj < 0)
            { // déjà éliminé
                // std::cout << "  coin " << m << " deja elimine " << std::endl;
                ii = abs(ii);
                jj = abs(jj);
                // continue;
                //  continuer car le coin P est éventuellent dans le coin Q
                //  ou Q est le sommet du cadre du coin P
            }
            cv::Vec4i l11 = lignes[ii].ln;
            cv::Vec4i l22 = lignes[jj].ln;

            cv::Point2i Q(coins[m][4], coins[m][5]);

            int k = coins[m][2];  // 0 pour origine, 2 pour extrémité de la première ligne
            int kk = coins[m][3]; // 0 pour origine, 2 pour extrémité de la deuxième ligne
            k = 2 - k;
            kk = 2 - kk;

            cv::Point2i U(l11[k], l11[k + 1]);
            cv::Point2i V(l22[kk], l22[kk + 1]);

            // coin  UQV

            cv::Point2i AA(l11[0], l11[1]);
            cv::Point2i BB(l11[2], l11[3]);
            cv::Point2i CC(l22[0], l22[1]);
            cv::Point2i DD(l22[2], l22[3]);
            // ignorer ce coin Q s'il n'est pas // coin P
            //
            double pv;
            bool estl22 = false;
            if (i == ii)
                pv = 0;
            else
                pv = lignes[std::abs(i)].a*lignes[std::abs(ii)].b - lignes[std::abs(i)].b*lignes[std::abs(ii)].a;
                // produit vectoriel des normales des lignes i et ii
            if (abs(pv) > maconf.deltaradian)
            { // AB  non // A'B'
                if (i == jj)
                    pv = 0;
                else
                    pv = lignes[std::abs(i)].a*lignes[std::abs(jj)].b - lignes[std::abs(i)].b*lignes[std::abs(jj)].a;
                if (abs(pv) > maconf.deltaradian)
                    continue;  //  AB  non // C'D'
                estl22 = true; // AB // C'D'   et donc  CD // A'B'
            }
            // TODO :
            //       déterminer si Q est proche d'une des lignes i (AB) ou j (CD)
            //       puis calculer la distance de Q à l'autre ligne
            //        hauteur ou largeur de carte ?
            bool memecarte = false;
            if (i < 0 || j < 0) continue; // coin P éliminé
            if (ii < 0 || jj < 0) continue; // coin Q éliminé
            int dc = std::max(5, maconf.deltacadre);
            int dc2 = std::max(3, dc/2);
            float d1 = lignes[i].a * Q.x + lignes[i].b * Q.y + lignes[i].c;
            float d2 = lignes[j].a * Q.x + lignes[j].b * Q.y + lignes[j].c;
            float d3, d11, d22;
            // distance de P à A'B' (=QU)  ou C'D' (=QV)
            d11 = lignes[ii].a*P.x + lignes[ii].b*P.y + lignes[ii].c;
            d22 = lignes[jj].a*P.x + lignes[jj].b*P.y + lignes[jj].c;
            float lgPQ;
            bool bienoriente = true;
            if (std::abs(d1) > dc2 && std::abs(d2) > dc2 ) continue; // Q n'est pas proche d'un coté de P
            if (std::abs(d2) < dc2 ){ // Q proche du coté CD= PS du coin n (P)
                // vérifier que PS et (QU ou QV) orientés en sens inverse
                // et que PR et (QV ou QU) ont meme orientation
                // et PS et PQ ont meme orientation   PS.PQ > 0
                if ((S.x - P.x)*(Q.x - P.x) + (S.y - P.y)*(Q.y - P.y) < 0 ) bienoriente = false;
                if (std::abs(d11) < std::abs(d22)) { // P proche de A'B' = QU
                    // PS et QU orientés de sens contraire ? : PS.QU < 0    et PR.QV > 0 et PS.PQ > 0
                    if ((S.x - P.x)*(U.x - Q.x) + (S.y - P.y)*(U.y - Q.y) > 0 ) bienoriente = false;
                    if ((R.x - P.x)*(V.x - Q.x) + (R.y - P.y)*(V.y - Q.y) < 0 ) bienoriente = false;

                } else { // P proche de C'D' = QV
                    if ((S.x - P.x)*(V.x - Q.x) + (S.y - P.y)*(V.y - Q.y) > 0 ) bienoriente = false;
                    if ((R.x - P.x)*(U.x - Q.x) + (R.y - P.y)*(U.y - Q.y) < 0 ) bienoriente = false;
                }
                if (bienoriente) {
                    d3 = d1;
                    // d1  proche de hauteur ou largeur de carte ?
                    if (std::abs(d1) <= maconf.hauteurcarte + dc && std::abs(d1) > maconf.hauteurcarte - dc) {
                        // Q opposé à P
                        memecarte = true;
                    } else if (std::abs(d1) <= maconf.largeurcarte + dc && std::abs(d1) > maconf.largeurcarte - dc) {
                        // Q opposé à P
                        memecarte = true;
                    } 
                }
            } else if (std::abs(d1) < dc2) { // Q proche de A'B' = PR
                // vérifier que PR et (QU ou QV) orientés en sens inverse
                // et que PS et (QV ou QU) ont meme orientation
                if ((R.x - P.x)*(Q.x - P.x) + (R.y - P.y)*(Q.y - P.y) < 0 ) bienoriente = false;
                if (std::abs(d11) < std::abs(d22)) { // P proche de A'B' = QU
                    // PR et QU orientés de sens contraire ? : PR.QU < 0    et PS.QV > 0
                    if ((R.x - P.x)*(U.x - Q.x) + (R.y - P.y)*(U.y - Q.y) > 0 ) bienoriente = false;
                    if ((S.x - P.x)*(V.x - Q.x) + (S.y - P.y)*(V.y - Q.y) < 0 ) bienoriente = false;
                } else { // P proche de C'D' = QV
                    if ((R.x - P.x)*(V.x - Q.x) + (R.y - P.y)*(V.y - Q.y) > 0 ) bienoriente = false;
                    if ((S.x - P.x)*(U.x - Q.x) + (S.y - P.y)*(U.y - Q.y) < 0 ) bienoriente = false;
                }
                if (bienoriente) {
                    d3 = d2;
                    if (std::abs(d2) <= maconf.hauteurcarte + dc && std::abs(d2) >= maconf.hauteurcarte - dc) {
                        // Q opposé à P
                        memecarte = true;
                    } else if (std::abs(d2) <= maconf.largeurcarte + dc && std::abs(d2) >= maconf.largeurcarte - dc) {
                        // Q opposé à P
                        memecarte = true;
                    }
                }
            }
            if (memecarte){ // coins n et m sur la même carte
                lgPQ = std::abs(d3);
                if (printoption) std::cout << " coin "<< m << " opposé au coin "<< n << " ecart "<< d3<<std::endl;
                // indiquer aussi si le premier coté du coin m est long ou court
                // rechercher si P est proche de A'B' ou de C'D'
                float dist = P.x*lignes[ii].a + P.y*lignes[ii].b + lignes[ii].c;
                if (std::abs(dist) < dc2) { // P proche de A'B'
                    // noter que le premier coté du coin Q est la longueur ou la largeur
                    if (lgPQ > 5*maconf.hauteurcarte/6) { // coté long
                        coins[m][10] = -3; 
                    } else coins[m][10] = -2; // coté largeur
                } else {
                    dist = P.x*lignes[jj].a + P.y*lignes[jj].b + lignes[jj].c;
                    if (std::abs(dist) < dc2) { // P proche de C'D'
                        if (lgPQ > 5*maconf.hauteurcarte/6) // coté largeur pour le premier coté du coin
                            coins[m][10] = -2; 
                        else coins[m][10] = -3; // coté longeur pour le premier coté
                    }
                }

                // si le coin m est déjà associé à un coin (<n) le coin n appartient à la même carte
                if (coins[m][9] != 0) { // carte du coin m déjà fixée
                    coins[n][9] = coins[m][9];
                } else if(coins[n][9] != 0) { // carte du coin n déjà fixée
                    coins[m][9] = coins[n][9];
                } else{ // nouvelle carte, commune aux deux coins n et m
                    nbcartes++;
                    coins[n][9] = coins[m][9] = nbcartes;
                }
                if (printoption) std::cout<<" --> carte numero "<< coins[n][9]<<std::endl;
            } else { // PQ n'est pas un bord de carte
                lgPQ = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
                if (lgPQ > 3*maconf.deltacadre) continue;
            }


            // coins P et Q proches //
            // tenir compte de l'imprécision de position des bords de carte (*2)
            // il peut y avoir trois droites // : le cadre d'un RDV et deux droites proches du bord de carte
            /* if (abs(Q.x - P.x) > 3 * maconf.deltacadre / 2 + 1)
                continue; // *3/2 si oblique
            if (abs(Q.y - P.y) > 3 * maconf.deltacadre / 2 + 1)
                continue;
            */
            // ignorer le coin Q  (QU QV) s'il n'est pas orienté comme le coin P (PR PS)
            if (estl22)
            { //  PR // QV
                // calcule le produit scalaire PR.QV    négatif si orientés en sens inverse
                int ps = (R.x - P.x) * (V.x - Q.x) + (R.y - P.y) * (V.y - Q.y);
                if (ps < 0)
                    continue; // aucun ne peut être cadre de l'autre
                // calcule le produit scalaire PS.QU    négatif si orientés en sens inverse
                ps = (S.x - P.x) * (U.x - Q.x) + (S.y - P.y) * (U.y - Q.y);
                if (ps < 0)
                    continue; // aucun ne peut être cadre de l'autre
            }
            else
            {
                // calcule le produit scalaire PR.QU   négatif si orientés en sens inverse
                int ps = (R.x - P.x) * (U.x - Q.x) + (R.y - P.y) * (U.y - Q.y);
                if (ps < 0)
                    continue; // aucun ne peut être cadre de l'autre
                // calcule le produit scalaire PS.QV    négatif si orientés en sens inverse
                ps = (S.x - P.x) * (V.x - Q.x) + (S.y - P.y) * (V.y - Q.y);
                if (ps < 0)
                    continue; // aucun ne peut être cadre de l'autre
            }

            // éliminer ce coin Q s'il est dans le coin P
            //
            double d, dd; // distances algébriques de Q à PS et PR
            double epsilon = dc - 1;
            // tenir compte de l'orthogonalité PR xPS
            // projections de Q : H sur PR   K sur PS
            // PH = PQ.PR / ||PR||
            double lgpr = (R.x - P.x) * (R.x - P.x) + (R.y - P.y) * (R.y - P.y);
            lgpr = sqrt(lgpr); // longueur de PR
            double lgps = (S.x - P.x) * (S.x - P.x) + (S.y - P.y) * (S.y - P.y);
            lgps = sqrt(lgps); // longueur de PS
            dd = ((Q.x - P.x) * (R.x - P.x) + (Q.y - P.y) * (R.y - P.y)) / lgpr;
            d = ((Q.x - P.x) * (S.x - P.x) + (Q.y - P.y) * (S.y - P.y)) / lgps;

            if (std::max(abs(d), abs(dd)) > dc + 2*epsilon)
                continue; // Q loin de PR ou de PS donc n'est pas le cadre

            // les deux coins n et m appartiennent à la même carte
            if (coins[m][9] != 0) {
                coins[n][9] = coins[m][9];
            } else if (coins[n][9] != 0){
                coins[m][9] = coins[n][9];
            } else {
                nbcartes++;
                coins[n][9] = coins[m][9] = nbcartes;
            }
            if (coins[m][0] >= 0 && coins[m][1] >= 0)
            { // Q pas encore éliminé
                bool elimQ = false;
                if (d >= 0 && d < dc + 2 * epsilon && dd >= 0 && dd < dc + 2 * epsilon)
                    elimQ = true;

                // if ((d >= -epsilon && dd >= dc / 2)
                //|| (dd >= -epsilon && d >= dc / 2) ) elimQ = true;
                if ((d >= -epsilon / 4 && dd >= epsilon / 4 && dd < dc + 2 * epsilon) 
                || (dd >= -epsilon / 4 && d >= epsilon && d < dc + 2 * epsilon))
                    elimQ = true;
                if (elimQ)
                {
                    // Q à l'intérieur du coin P
                    // marquer le coin Q "éliminé"
                    coins[m][0] = -ii;
                    coins[m][1] = -jj;
                    if (printoption)
                        std::cout << " --> elimination du coin " << m
                                  << " dans le coin " << n << std::endl;
                }
            }

            //  P est-il dans le coin Q
            // dans ce cas, d et dd sont négatifs, de valeur absolue inférieure à deltacadre
            if ((d < 1 && dd < -epsilon && dd + dc >= -epsilon)
             || (dd < 1 && d < -epsilon && d + dc >= -epsilon))
            {
                eliminerP = true;
                if (printoption)
                    std::cout << " --> elimination du coin " << n
                              << " dans le coin " << m << std::endl;
            }
            if (coins[m][0] >= 0)
            { // Q pas encore éliminé
                // P est-il le sommet du cadre de Q ?
                // a distance deltacadre du coté négatif des droites du coin Q
                if ((dd < 0 && std::abs(dd + dc) <= epsilon) && (d < 0 && std::abs(d + dc) <= epsilon))
                {
                    // P est le sommet du cadre du coin Q
                    coins[m][6] = 1; //  Q estunRDV
                    coins[m][7] = P.x;
                    coins[m][8] = P.y;
                    eliminerP = true;
                    if (printoption)
                        std::cout << " --> elimination du coin " << n
                                  << " dans le coin " << m << std::endl;
                }
                continue;
            }

            // Q est-il le cadre de la carte du coin P
            // comme les coins on des angles droits, on sait que CD est  // l'autre droite A'B' ou C'D'
            //
            // si Q est à l'intérieur du coin P (éventuellement sur un des cotés PR ou PS)
            //    éliminer le coin Q
            //    si Q est proche du cadre si P est un R D ou V : noter que P est un Roi Dame ou Valet
            //
            // sinon, P est à l'intérieur de Q
            //    différer l'élimination de P après la poursuite de la recherche des autres coins Q dans P
            //    si P est proche du cadre possible de Q : noter Q est un Roi Dame ou Valet

            // calculer la distance entre ces deux (presque) //
            // en fait, c'est la distance du point Q à chacune des droites PR et PS

            // il faut qu'au moins une des distances soit proche de deltacadre pour que le coin P soit un Roi Dame ou valet
            // fausses détections ==> deux cas acceptables:
            //   Q est à l'intérieur et à distance deltacadre des deux cotés du coin P
            //   Q et à distance deltacadre d'un coté du coin P, à l'intérieur et à distance très faible de l'autre coté
            // MODIF 2025/04/22 : Q doit être le sommet du cadre

            if (d > 0 && dd > 0 && std::abs(d - dc) <= epsilon && std::abs(dd - dc) <= epsilon)
            {
                // Q est cadre à l'intérieur de PR et PS)
                coins[n][6] = 1; // estunRDV
                coins[n][7] = Q.x;
                coins[n][8] = Q.y;
                coins[m][0] = -ii; // éliminer Q
                coins[m][1] = -jj;
                continue;
            }

            if (coins[m][0] < 0 || coins[m][1] < 0)
                continue; // Q éliminé

            // Q n'est pas sur le cadre de P
        } // for m

        // élimination différée de P ?
        if (eliminerP)
        { // c'est peut-être déjà fait
            if (coins[n][0] >= 0)
            {
                if (printoption)
                    std::cout << "elimination coin " << n << std::endl;
                coins[n][0] = -i;
                coins[n][1] = -j;
            }
        } else if (coins[n][9] == 0) { // pas encore affecté à une carte
            nbcartes++;
            coins[n][9] = nbcartes; // nouvelle carte
            if (printoption) std::cout<<" --> nouvelle carte "<<nbcartes<<" pour le coin "<<n<<std::endl;
        }
        c++;
        c--; // pour pouvoir mettre un point d'arrêt
    } // for n

    // on a obtenu tous les coins et les cartes.

    //TODO : si on traite une video,
    // un coin qui n'était pas présent avant désigne une nouvelle carte
    // analyser la nouvelle carte. vérifier qu'il y a une seule nouvelle carte
    // extraire et redresser la carte
    // déterminer la valeur du blanc 
    // analyser les zones normalement blanches d'une carte autre que R D ou V
    // --> on sait si c'est un honneur
    // si ce n'est pas un honneur, analyser les zones où sont les gros symboles
    // --> valeur de la carte, entre 1 et 10
    // extraire un gros symbole (y compris pour V D R)
    // analyser ce gros symbole --> rouge ou noir, couleur Pique Trefle Coeur ou carreau 

    // on a alors identifié la nouvelle carte et les nouveaux coins
    // il sera inutile de traiter ces coins, sauf si la carte est un honneur
    const std::string nomval[14] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "V", "D", "R"};
    if (estvideo) {
        if (printoption) std::cout<< std::endl<<"================== recherche des nouvelles cartes ======"<<std::endl;
        int epsilon = 2;
        int nca = 0; // numéro de carte complète à analyser
        int nc = 0; // numéro de carte cherchée
        for (int n = 0; n < nbcoins; n++){
            if (coins[n][9] <= nc) continue; // coin d'une carte déjà recherchée
            if (coins[n][0] < 0 || coins[n][1] < 0) continue; // coin éliminé
            // nouvelle carte
            nc = coins[n][9]; // numéro de carte
                // noter que par construction des numéros de cartes, 
                //  les numéros de cartes sont en ordre croissant des numéros de coins
            // rechercher dans lescoins mémorisés
            cv::Point2i PT(coins[n][4], coins[n][5]);
            bool trouve(false);
            bool trouvecarte(false);
            std::string nomcarte="";
            int numcol, valcarte;
            for (int h=0; h < *pnbcoins; h++){
                if (std::abs(PT.x - lescoins[h][1]) >= epsilon ) continue;
                if (std::abs(PT.y - lescoins[h][2]) >= epsilon ) continue;
                trouve = true;
                trouvecarte = true;
                numcol = lescoins[h][3];
                valcarte = lescoins[h][4];
                if (lescoins[h][3] == 0) nomcarte = "Pique";
                else if (numcol == 1) nomcarte = "Coeur";
                else if (numcol == 2) nomcarte = "Carreau";
                else if (numcol == 3) nomcarte = "Trefle";
                if (lescoins[h][4] > 0 && lescoins[h][4] < 14)
                    nomcarte += " " + nomval[lescoins[h][4]];
                break;
            }
            if (trouve){
                if(printoption)  std::cout<< " carte "<< nc << " ("<< nomcarte 
                 << ") déjà dans la frame précédente. coin " << n << std::endl;
                // noter qu'il est inutile d'analyser ce coin
                coins[n][10] = numcol;
                coins[n][11] = valcarte;
                // ainsi que tous les coins de cette carte
                for (int m= 0; m < nbcoins; m++){
                    if (coins[m][9] != nc) continue;
                    coins[m][10] = numcol;
                    coins[m][11] = valcarte;
                } 
                continue;
            } 
            // nouveau coin, donc nouvelle carte
            // vérifier que c'est la seule nouvelle carte
            // analyser les coins à partir du coin n+1 nouveaux 
            //   de la même carte --> les angles de la carte
            //   d'une autre carte --> erreur : plusieurs nouvelles cartes
            if (printoption) std::cout<< "nouvelle carte "<< nc<<" nouveau sommet "<< n <<PT<<std::endl;
            bool estcarte(false);
            for (int m =n+1; m< nbcoins; m++){
                if (coins[m][9] != nc) continue; // pas de cette carte
                if (coins[m][0] < 0 || coins[m][1] < 0) continue; // coin éliminé
                cv::Point2i P2(coins[m][4], coins[m][5]);
                // nouveau coin ?
                bool trouve(false);
                for (int h=0; h < *pnbcoins; h++){
                    if (std::abs(P2.x - lescoins[h][1]) > epsilon ) continue;
                    if (std::abs(P2.y - lescoins[h][2]) > epsilon ) continue;
                    trouve = true;
                    trouvecarte = true;
                    numcol = lescoins[h][3];
                    valcarte = lescoins[h][4];
                    break;
                }
                //if (trouve) continue;
                // coin de la même carte ?
                if (nc != coins[m][9]) { // une autre nouvelle carte
                    //std::cout<< " Erreur : coin " << n<< " autre nouvelle carte "
                    //<< coins[m][9]<<" , coin  "<< m <<std::endl;
                } else { // autre coin de la nouvelle carte
                    // mémoriser
                    if (printoption) std::cout << " autre sommet "<< m<< " "<< P2<<std::endl;
                    // distance entre les deux points proche de longueur ou largeur de carte
                    if (!estcarte) {
                        double dist = std::sqrt((PT.x - P2.x)*(PT.x - P2.x) + (PT.y - P2.y)*(PT.y - P2.y));
                        if (std::abs(dist - maconf.hauteurcarte) < 4) estcarte =true;
                        if (std::abs(dist - maconf.largeurcarte) < 4) estcarte =true;
                    }
                }               
            } // for m
            if (trouvecarte) { // carte dont au moins un coin était dans la frame précédente
                // indiquer la valeur et la couleur de carte sur tous les coins de cette carte (nc)
                for (int m = 0; m<nbcoins; m++){
                    if (coins[m][9] != nc) continue;
                    coins[m][10] = numcol;
                    coins[m][11] = valcarte;
                }
            }
            else if (estcarte) { // carte avec au moins deux coins et absente de la frame précédente
                if (printoption) std::cout<<" carte complète "<< nc<<std::endl;
                // analyser la nouvelle carte
                if (nca != 0){
                    if (printoption) std::cout<< " plusieurs cartes complètes "<< nca << ","<<nc<<std::endl;
                    nca = 0;
                    break;
                } 
                nca = nc;
            } else // coin (n) absent de la frame précédente et non opposé à un autre coin
             if(printoption) std::cout<< " carte "<<nc<< " a un seul coin: "<< n<<std::endl;
        } // for n
        if (nca){
            if (printoption) std::cout<<"==> analyse de la carte "<<nca<<std::endl;
            // obtenir les 4 sommets du rectangle de la carte
            // trouver l'encombrement des coins de la carte
            // partir d'un coin le plus  à gauche, le plus haut s'il y en a plusieurs, (P)
            // rechercher un coin opposé (Q) à distance proche de la longueur ou largeur de carte
            // Q doit être proche d'une des lignes du coin P
            // et la distance PQ doit être la plus proche possible de la longueur ou largeur de carte
            // si la distance n'est pas exactement la longueurou largeeur
            // repartir du coin Q (--> P) et chercher un coin Q à distance convenable.
            // on a alors deux coins P et Q opposés PQ étant un coté de la carte
            // chercher ou calculer les deux autres sommets
            int xmin(image.cols), ymin(image.rows);
            cv::Point2i P, Q, U, V;
            int n1, n2, n3, n4;
            int nbpts;
            int i, j;
            int dc = std::max(4, maconf.deltacadre);
            // rechercher le coin le plus à gauche (et le plus haut s'il y en a plusieurs)
            for (int n = 0; n < nbcoins; n++) {
                if (coins[n][9] == nca && coins[n][0] >= 0 && coins[n][1] >= 0){
                    if (coins[n][4] <= xmin ) {
                        xmin = coins[n][4];
                        if (coins[n][5] < ymin ) {
                            n1 = n;
                            ymin = coins[n][5];
                        }
                    }
                }
                P = cv::Point2i(coins[n1][4], coins[n1][5]);
            }
            if(printoption) std::cout<<" coin "<< n1 <<" "<<P<<std::endl;
            // rechercher un coin opposé sur le premier coté du coin P
            i = coins[n1][0]; // indice de la 1ère ligne
            j = coins[n1][1];
            cv::Vec4i ln = lignes[i].ln;
            float a = lignes[i].a; float b = lignes[i].b; float c = lignes[i].c;
            n2 = -1;
            float ecart, ecartmin(dc);
            int lgref, lgref2;
            for (int n = 0; n < nbcoins; n++) {
                if (coins[n][9] != nca || coins[n][0] < 0 || coins[n][1] < 0) continue;
                if (n == n1) continue;
                Q = cv::Point2i(coins[n][4], coins[n][5]);
                float dist = a*Q.x + b*Q.y + c;
                float lg = std::abs(dist);
                if (lg > 2) continue; // Q n'est pas opposé à P
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
            // si on n'a pas trouvé de coin opposé, rechercher sur l'autre ligne du coin P
            if (n2 < 0) {
                j = i;
                i = coins[n1][1]; // indice de la 2ème ligne
                ln = lignes[i].ln;
                a = lignes[i].a; b = lignes[i].b; c = lignes[i].c;
                float ecart, ecartmin(dc);
                for (int n = 0; n < nbcoins; n++) {
                    if (coins[n][9] != nca || coins[n][0] < 0 || coins[n][1] < 0) continue;
                    if (n == n1) continue;
                    Q = cv::Point2i(coins[n][4], coins[n][5]);
                    float dist = a*Q.x + b*Q.y + c;
                    float lg = std::abs(dist);
                    if (lg > 2) continue; // Q n'est pas opposé à P
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
            }
            if (n2 < 0) {
                // aucun coin opposé au coin P
                nbpts = 1;
            } else {
                Q = cv::Point2i(coins[n2][4], coins[n2][5]);
                if(printoption) std::cout<<" coin opposé "<< n2 <<" "<<Q<<std::endl;
                // on a deux sommets de la carte P (n1) et Q (n2)
                // rechercher ou calculer les deux autres sommets U et V
                // la ligne commune est : coin n1 indice i, ligne ln, coefficient a b c
                // rechercher le coin U sur l'autre ligne (j) du coin P
                cv::Vec4i ln = lignes[j].ln;
                float aa = lignes[j].a; float bb = lignes[j].b; float cc = lignes[j].c;
                float ecmin = 3;
                n3 = -1;
                for (int n = 0; n < nbcoins; n++) {
                    if (coins[n][9] != nca || coins[n][0] < 0 || coins[n][1] < 0) continue;
                    if (n == n1 || n == n2) continue;
                    U = cv::Point2i(coins[n][4], coins[n][5]);
                    float dist = aa*U.x + bb*U.y + cc;
                    if (std::abs(dist) > 1) continue;
                    float lg = std::sqrt((U.x - P.x)*(U.x - P.x) + (U.y - P.y)*(U.y - P.y));
                    if (std::abs(lg - lgref2) > ecmin) continue;
                        ecmin = std::abs(lg - lgref2);
                        n3 = n;
                }
                if(n3 >= 0) {
                    U = cv::Point2i(coins[n3][4], coins[n3][5]);
                    if(printoption) std::cout<<" coin trois "<< n3 <<" "<<U<<std::endl;
                }

                // rechercher le sommet V sur le coté orthogonal à PQ du coin Q (n2)
                int ii = coins[n2][0]; int jj = coins[n2][1];
                float ps = a*lignes[ii].a + b*lignes[ii].b;
                if(std::abs(ps) < 0.5) {
                    ii = jj; jj = coins[n2][0];
                }
                // ii est la ligne // PQ  jj est la ligne ortogonale à PQ
                // rechercher un coin V de cette carte tel que QV = lgref2 (à deltacadre près)
                // distance de V à la droite PQ = lgref2   et QV orthogonal à PQ
                n4 = -1;
                ecmin = 3;
                 for (int n = 0; n < nbcoins; n++) {
                    if (coins[n][9] != nca || coins[n][0] < 0 || coins[n][1] < 0) continue;
                    if (n == n1 || n == n2 || n == n3) continue;
                    V = cv::Point2i(coins[n][4], coins[n][5]);
                    // distance de V à cette ligne jj orthogonale à PQ:
                    float d = V.x * lignes[jj].a + V.y * lignes[jj].b + lignes[jj].c;
                    if (std::abs(d) > 3) continue;

                    float dist = std::abs(a*V.x + b*V.y + c);
                    if (std::abs(dist -lgref2) > ecmin) continue;
                        ecmin = std::abs(dist - lgref2);
                        n4 = n;
                }
                if(n4 >= 0) {
                    V = cv::Point2i(coins[n4][4], coins[n4][5]);
                    if(printoption) std::cout<<" coin quatre "<< n4 <<" "<<V<<std::endl;
                }


                if (n3 < 0) {
                    // pas de coin sur l'autre ligne du coin P
                    U = V;
                    n3 = n4;
                    n4 = -1;
                }
                nbpts = 2;
            }
            if (n4 >= 0) nbpts = 4;
            else if (n3 >= 0) nbpts = 3;
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
            if (nbpts == 2 || nbpts == 3){
                // les deux premiers points sont la longueur ou la largeur

                // on dispose du vecteur normal des lignes des coins
                cv::Point2i P(coins[n1][4], coins[n1][5]);
                cv::Point2i Q(coins[n2][4], coins[n2][5]);
                // vérifier que PQ est la hauteur ou la largeur de carte
                cv::Point2i C; // autre sommet opposé au point P
                cv::Point2i D; // autre sommet opposé au point Q
                // PQ hauteur ou largeur de carte ?
                double lgW; // longueur de l'autre coté
                double lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
                double lg = std::sqrt(lg2);
                if (lg > 5*maconf.hauteurcarte/4 && nbpts == 3){
                    // c'est la diagonale.
                    // s'il y a un 3ème sommet, remplacer le 2ème sommet
                    Q.x = coins[n3][4]; Q.y = coins[n3][5];
                    pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1]; 
                    lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
                    lg = std::sqrt(lg2);
                }
                if (std::abs (lg - maconf.hauteurcarte) <= dc ) lgW = maconf.largeurcarte;
                else if (std::abs (lg - maconf.largeurcarte) <= dc ) lgW = maconf.hauteurcarte;
                else lgW = 0;
                if (lgW > 0) {
                    int i = coins[n1][0]; // un coté du coin P
                    int j = coins[n1][1]; // autre coté
                    int k = 2 - coins[n1][3];
                    double a = lignes[i].a; // vecteur normal
                    double b = lignes[i].b;
                    double c = lignes[i].c;
                    double dist = a*Q.x + b*Q.y +c;
                    if (abs(dist) > dc) { // PQ // autre ligne du coin P
                        i = j;
                        j = coins[n1][0];
                        k = 2 - coins[n1][2];
                        a = lignes[i].a; // vecteur normal
                        b = lignes[i].b;
                        c = lignes[i].c;
                    }
                    cv::Point2i R(lignes[j].ln[k], lignes[j].ln[k+1]); // extrémité loin de P
                    // distance de R à la droite (i)
                    double dR = a*R.x + b*R.y + c;
                    // dR > 0 : vecteur normal dirigé vers l'intérieur 
                    if (dR < 0) lgW = -lgW;
                    C.x = Q.x + a*lgW;
                    C.y = Q.y + b*lgW;
                    D.x = P.x + a*lgW;
                    D.y = P.y + b*lgW;
                    if (C.x >= 0 && C.x < image.cols && C.y >= 0 && C.y < image.rows 
                    && D.x >= 0 && D.x < image.cols && D.y >= 0 && D.y < image.rows ) {
                        pts[2][0] = C.x;
                        pts[2][1] = C.y;
                        pts[3][0] = D.x;
                        pts[3][1] = D.y;
                        nbpts = 4;   // on vient de compléter
                    } else {
                        if(printoption) std::cout<<" hors de l'ecran"<<std::endl;
                    }
                }
            }

            if (nbpts == 4){
                // TODO : vérifier qu'aucun autre coin d'une autre carte n'est dans celle-ci
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
                for (int n = 0; n < nbcoins; n++){
                    if (coins[n][9] == nca) continue; // un coin de cette carte
                    cv::Point2i P(coins[n][4], coins[n][5]);
                    // TODO : ignorer ce coin s'il n'était déjà dans la frame précédente
                    if (estvideo) {
                        bool trouve = false;
                        for (int h = 0; h < *pnbcoins; h++){
                            if (lescoins[h][0] == 0) continue; // slot vide
                            if (std::abs(P.x - lescoins[h][1]) <= 1 
                            && std::abs(P.y - lescoins[h][2]) <= 1 ) {
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
                        ps = (P.x - U.x)*(V.x - U.x) + (P.y - U.y)*(V.y - U.y);
                        if (ps < 3*maconf.hauteurcarte) {
                            estdehors = true; break;}
                        if (ps > (V.x - U.x)*(V.x - U.x) + (V.y - U.y)*(V.y - U.y) - 2*maconf.hauteurcarte) {
                            estdehors = true; break;
                        }
                        U = V;
                    }
                    if (!estdehors ){
                        nbpts = 0;
                        if(printoption) std::cout<< " coin "<< n << P <<" dans la carte "<<std::endl; 
                    }
                }
                if (nbpts == 4) {
                    int numcol= -1;
                    int valcarte = decoderCarte(image, pts, maconf, numcol);
                    if (printoption) {
                        std::cout<<"valeur carte "<<valcarte<<std::endl;
                        std::cout<<"couleur "<<numcol<<std::endl;
                    }
                    // mémoriser la valeur obtenue sur tous les coins de la carte
                    // uniquement si on a trouvé couleur et valeur
                    if (numcol >= 0 && valcarte > 0 && valcarte <= 13) {
                        for (int n = 0; n < nbcoins; n++){
                            if (coins[n][9] != nca) continue; // pas un coin de cette carte
                            coins[n][10] = numcol;
                            coins[n][11] = valcarte;
                            if (valcarte > 10) coins[n][6] = 1;
                        }
                    }
                }
            }
        } // traitement carte nca
    }

    // afficher ce qui reste selectionné
    cv::Mat imaC = ima2.clone();
    //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < imaC.rows; y++)
    {
        for (int x = 0; x < imaC.cols; x++)
        {
            imaC.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
        }
    }
    /***************/
    if (estvideo){
        // éliminer les coins de la frame précédente qui ne sont pas dans celle-ci
        int dc = std::max(2, maconf.deltacadre);
        for (int h = 0; h < *pnbcoins; h++){
            if (lescoins[h][0] == 0) continue;
            bool trouve = false;
            cv::Point2i Q(lescoins[h][1], lescoins[h][2]);
            for (int n = 0; n < nbcoins; n++){
                //if (coins[n][0] < 0 || coins[n][1] < 0) continue; // coin éliminé
                cv::Point2i P (coins[n][4], coins[n][5]);
                // proche ?
                if (std::abs(P.x - Q.x) <= dc && std::abs(P.y - Q.y) <= dc ) {
                    trouve = true;
                    break;
                }
            }
            if (!trouve) {
                lescoins[h][0] = 0; // le coin n'est plus présent
            }
        }
    }

    // afficher les coins
    c = 0;
    for (int n = 0; n < nbcoins; n++)
    {
        int cc = coins[n][9]; // numéro de carte
        while (cc >= NBCOULEURS) cc -= NBCOULEURS;
        int i = coins[n][0];
        int j = coins[n][1];
        cv::Point P(coins[n][4], coins[n][5]);

        cv::Vec4i l1 = lignes[abs(i)].ln;
        cv::Vec4i l2 = lignes[abs(j)].ln;

        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        cv::Point2i C(l2[0], l2[1]);
        cv::Point2i D(l2[2], l2[3]);
        cv::Vec4i nl1(A.x, A.y, B.x, B.y);
        cv::Vec4i nl2(C.x, C.y, D.x, D.y);

        // !!!! uniquement sur les copies

        // remplacer l'extremité qui convient par l'intersection
        int k = coins[n][2]; // quelle extrémité de la ligne 1?
        if (k != 0 && k != 2) k = 0; // protection
        nl1[k] = coins[n][4];
        nl1[k + 1] = coins[n][5];
        k = coins[n][3];
        if (k != 0 && k != 2) k = 0; // protection
        nl2[k] = coins[n][4];
        nl2[k + 1] = coins[n][5];

        // pour chacune des lignes l1 et l2, remplacer l'extrémité loin du coin par le milieu de la ligne
        // uniquement si la ligne est plus longue que la demi-largeur de carte et pour l'affichage : faire une copie

        k = coins[n][2]; // vaut 0 ou 2
        if (std::max(abs(B.x - A.x), abs(B.y - A.y)) > maconf.hauteurcarte / 3)
        {
            nl1[2 - k] = (A.x + B.x) / 2;
            nl1[3 - k] = (A.y + B.y) / 2;
        }
        k = coins[n][3];
        if (std::max(abs(D.x - C.x), abs(D.y - C.y)) > maconf.hauteurcarte / 3)
        {
            nl2[2 - k] = (C.x + D.x) / 2;
            nl2[3 - k] = (C.y + D.y) / 2;
        }
        if (estvideo){
            // si ce coin était trouvé dans la frame précédente, inutile de le considérer
            if (coins[n][10] > 0 && coins[n][11] > 0) {
                if (printoption) std::cout<<"coin "<< n << " dans une frame précédente carte couleur:"
                     <<coins[n][10]<<", valeur:"<<coins[n][11]<<std::endl;
                    i = coins[n][0] = -std::abs(i);
                    j = coins[n][1] = -std::abs(j);
            } else
             for (int h = 0; h < *pnbcoins; h++){
                if (lescoins[h][0] == 0) continue; // slot vide
                if (std::abs (P.x - lescoins[h][1]) <= maconf.deltacadre 
                && std::abs (P.y - lescoins[h][2]) <= maconf.deltacadre ) {
                    // déjà trouvé dans la précédente frame
                    coins[n][10] = lescoins[h][3];
                    coins[n][11] = lescoins[h][4];
                    i = coins[n][0] = -std::abs(i);
                    j = coins[n][1] = -std::abs(j);
                    if (printoption) std::cout<<"coin "<< n << " dans une frame précédente carte couleur:"
                     <<lescoins[h][3]<<", valeur:"<<lescoins[h][4]<<std::endl;
                    break;
                }
            }
        }

        if (i < 0 || j < 0 || (estvideo && coins[n][10] >= 0 && coins[n][11]> 0) )
        {                                                          // coin éliminé précédemment
            cv::circle(imaC, P, 2, cv::Scalar(255, 255, 255), -2); //  cercle blanc au sommet du coin
            cv::circle(grise, P, 2, cv::Scalar(0, 0, 255), -2);    //  cercle rouge au sommet du coin
            // si ce coin ressemble à un cadre, afficher les lignes en trait fin gris
            cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), cv::Scalar(128, 128, 128), 1); // petit trait
            cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), cv::Scalar(128, 128, 128), 1); // petit trait
            continue;                                                                                           // coin éliminé
        }

        // TODO : pour chaque coté, rechercher une ligne // vers l'extérieur à distance deltacadre
        //        rechercher une ligne // à l'intérieur à 1 pixel

        cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), couleurs[cc], 1); // petit trait
        cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), couleurs[cc], 1); // petit trait
        if (coins[n][6] > 0)
            cv::circle(imaC, P, 5, couleurs[cc], 3); //  cercle au sommet du coin
        else
        {
            cv::circle(imaC, P, 5, couleurs[cc], 1); //  cercle épais (RDV) au sommet du coin
            cv::circle(grise, P, 5, couleurs[cc], 1);
        }
        // afficher le numéro du coin
        std::string texte = std::to_string(n);
        cv::putText(imaC, texte, P, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    couleurs[cc], 1);
        c++;
        if (c >= NBCOULEURS)
            c = 0;
    } // for n

    if (htmax > 4 * maconf.hauteurcarte / 5)
    {
        if (printoption)
            std::cout << "probable hauteur de carte : " << htmax << std::endl;
        cv::circle(imaC, P1, 6, cv::Scalar(0, 128, 128), 4);
        cv::circle(imaC, P2, 6, cv::Scalar(0, 128, 128), 4);
    }
    if (printoption)
    {
        cv::imshow("coins", imaC);
        cv::imshow("grise", grise);
        cv::waitKey(1);
    }

    // cv::waitKey(0);
    //  extraire les coins
    //
    bool estunRDV;
    estunRDV = false;       // le coin contient-il un cadre ?
    cv::Point2i Q;          // point du cadre
    std::string cartes[50]; // cartes trouvées
    //nbcartes = 0;  // 

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = t1 - t22;
    if (printoption) std::cout << "Temps préparatoire : " << duree.count() << " secondes" << std::endl
      << "============================"<< std::endl;

    result = image.clone();
    cv::imshow("result", result);
    int szPrec = 0;


// TODO : vérifier que l'on obtient le même résultat pour tous les coins d'une même carte
//        a posteriori après traitement multithread


    std::string valeurcarte[14] = {" ", "1","2", "3", "4", "5", "6", "7", "8", "9", "10", "V", "D", "R"};
    std::string couleurcarte[4] = {"P", "C", "K", "T"}; 


    for (int n = 0; n < nbcoins; n++)
    {
        int i = coins[n][0]; // indice de ligne
        int j = coins[n][1];
        if (i < 0 || j < 0 || (estvideo && coins[n][10] >= 0 && coins[n][11]> 0) )
            continue; // coin éliminé ou déjà analysé dans une carte
        int l1W[4], l2W[4];

        cv::Vec4i l1 = lignes[i].ln; // ligne AB
        cv::Vec4i l2 = lignes[j].ln; // ligne CD
        for (int i = 0; i < 4; i++)
        {
            l1W[i] = l1[i];
            l2W[i] = l2[i];
        }
        if (printoption)
            std::cout << std::endl
                      << "coin " << n << "   ";
        std::string cartelue;

        const int *p = &coins[0][0];
        if (threadoption == 0) { // pas de sous-tache
            if (coins[n][10] < 0 || coins[n][11] <=0) // valeur non encore déterminée
                traiterCoin(n, coins, image, resultats,
                        result, &l1W[0], &l2W[0], maconf);
            if (coins[n][11] != 0 && coins[n][10] >= 0) // valeur trouvée
            {
                if (!estvideo) cv::imshow("result", result);
                cv::Point2i PT(coins[n][4], coins[n][5]);
                std::string resW = couleurcarte[coins[n][10]] + valeurcarte[coins[n][11]];
                std::string res = resW + "#";

                afficherResultat(result, PT, res);
                if (waitoption > 1)
                    cv::waitKey(0);
                else
                    cv::waitKey(1);
            }
        }
        else // sous-taches
        if (coins[n][10] < 0 || coins[n][11] <=0) // valeur non encore déterminée
        { // démarrer une sous-tache
            std::unique_lock<std::mutex> lock(mtx);
            // std::cout << "Avant attente cvar..." << std::endl;
            cvar.wait(lock, []
                      { return activeThreads < MAX_THREADS; });
            // std::cout << "Débloqué !" << std::endl;

            ++activeThreads;
            threads.emplace_back([n, &coins, image, &resultats, result, l1W, l2W, maconf]()
                                 { traiterCoin(n, std::ref(coins), image, std::ref(resultats), result, l1W, l2W, maconf); });

            // std::cout<< activeThreads<< " theads actives "<< " coin "<<n <<std::endl;
            // threads.emplace_back(traiterCoin, n, coins, std::ref(image),
            //     std::ref(resultats), std::ref(result), l1W, l2W, std::ref(maconf));
        }
    } // boucle sur les coins

    if (threadoption > 0)
    {
        // Attente de toutes les sous-tâches
        for (auto &t : threads)
        {
            t.join();
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    if (printoption) std::cout << "Temps écoulé : " << elapsed.count() << " secondes" << std::endl;
    // cv::imshow("result", result);
    // Affichage des résultats après synchronisation
    // les résultats sont dans le tableau des coins 
    // afficher un résultat pour chaque carte
    bool nouveaucoin = false;
    for (int nc = 1; nc <= nbcartes; nc++) {
        bool premier = true;
        int cc1, vc1;
        for (int n = 0; n < nbcoins; n++){
            if (nc != coins[n][9]) continue; // pas la carte nc
            int i = coins[n][0];
            int j = coins[n][1];
            cv::Point2i PT(coins[n][4], coins[n][5]);
            if (!estvideo) { // on ne traite pas une video
                if (i < 0 || j < 0) continue; //coin éliminé
                if (coins[n][11] == 0) continue; // valeur de carte non trouvée
            
                // pas vidéo et coin non éliminé et valeur de carte trouvée et carte en cours
                if (i>=0 && j >= 0 && coins[n][11] > 0 && nc == coins[n][9]){
                    if (premier) {cc1 = coins[n][10]; vc1 = coins[n][11];}
                    std::string resW = couleurcarte[coins[n][10]] + valeurcarte[coins[n][11]];
                    std::string res = resW + "#";
                    afficherResultat(result, PT, res);
                    if (premier) cartes[nc - 1] = resW;
                    else if (cc1 != coins[n][10] || vc1 != coins[n][11]) {
                        // incohérence. quelle détection est fausse?
                        std::cout<< "détection incohérente " << resW << " carte "<< cartes[nc - 1] <<std::endl; 
                    }
                    premier = false;
                }
                //
                // si on traite une vidéo, ajouter les coins détectés ou analysés
            } else { // on traite une video
                // rechercher si le coin est déjà dans lescoins
                bool trouve(false);
                int hvide = -1;
                for(int h = 0; h<*pnbcoins; h++){
                    if (hvide < 0 && lescoins[h][0] == 0) hvide = h;
                    if (std::abs(PT.x - lescoins[h][1]) <= 2 
                        && std::abs(PT.y - lescoins[h][2]) <= 2 ) {
                            trouve = true;
                            break;
                    }
                }
                if (!trouve) {
                    if (hvide < 0) {hvide = *pnbcoins; *pnbcoins += 1;}
                    int i = hvide;
                    lescoins[i][0] = 1; 
                    lescoins[i][1] = PT.x;
                    lescoins[i][2] = PT.y;
                    lescoins[i][3] = coins[n][10]; // couleur
                    lescoins[i][4] = coins[n][11]; // valeur 
                    nouveaucoin = true;
                }
            }
        } // for(n) coins
        // valoriser tous les coins de cette carte si la valeur est connue
        if (cc1 >= 0 && vc1 > 0 ){
            for (int n = 0; n < nbcoins; n++){
                if (coins[n][9] != nc) continue;
                coins[n][10] = cc1;
                coins[n][11] = vc1;
            }
        }
    } // for(nc) cartes
    // cv::imshow("result", result); // désactivé en multitache
    if (!estvideo) { cv::imshow("result", result); cv::waitKey(1);}

    // si on traite une vidéo, les coins trouvés précédemment ou maintenant
    //    sont dans le tableau lescoins[]
    // on affiche les valeurs trouvées
    // on reconstitue alors le tableau des cartes
    bool nouvellecarte = false;
    if (estvideo){
        nbcartes = 0;
        for (int h = 0; h < *pnbcoins; h++){
            if (lescoins[h][0] == 0) continue; // slot vide
            cv::Point2i PT(lescoins[h][1], lescoins[h][2]);
            if ((lescoins[h][3] < 0) // coin non identifié (couleur)
            || (lescoins[h][4] < 1 || lescoins[h][4] > 13)){ // coin non identifié (valeur)
                cv::circle(result, PT, 2, cv::Scalar(255,0,0), -1);
                continue;
            }
            char nomcol = (couleurcarte[lescoins[h][3]])[0];
            std::string val = valeurcarte[lescoins[h][4]];
            std::string res = nomcol + val; 
            afficherResultat(result, PT, res);
            int i;
            for (i=0; i < nbcartes; i++){
                if (nomcol == cartes[i][0] && val == cartes[i].substr(1)) break;
            }
            if (i == nbcartes){
                cartes[i] = nomcol + val;
                nbcartes++;
                nouvellecarte = true;
            }
        }
        // vider la fin du tableau lescoins
        int h = *pnbcoins - 1;
        while (h >= 0 && lescoins[h][0] == 0) h--;
        *pnbcoins = h + 1;
        cv::imshow("result", result); cv::waitKey(1);
    }
    std::cout<<"===> cartes trouvées :"<<std::endl;
    for (int i = 0; i < nbcartes; i++)
    {
        if(cartes[i].size() < 2) continue;
        char nomcol = cartes[i][0];
        std::string valeur = cartes[i].substr(1);
        std::string cartecouleur;
        if (nomcol == 'P')
            cartecouleur = "Pique ";
        else if (nomcol == 'C')
            cartecouleur = "Coeur ";
        else if (nomcol == 'K')
            cartecouleur = "Carreau ";
        else
            cartecouleur = "Trefle ";
        std::cout << cartecouleur << valeur << std::endl;
    }

    std::cout << "====== fin de frame ======" << std::endl;
    if (waitoption)
        if (!estvideo || nouveaucoin)
            cv::waitKey(0);
    {
        double val;
        val = cv::getWindowProperty("symbole", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("symbole");
        val = cv::getWindowProperty("orient", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("orient");
        val = cv::getWindowProperty("coin", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("coin");
        val = cv::getWindowProperty("Artefact", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("Artefact");
        val = cv::getWindowProperty("coins", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("coins");
        val = cv::getWindowProperty("Ext", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("Ext");
        val = cv::getWindowProperty("bords", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("bords");
        val = cv::getWindowProperty("lignes ximgproc", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("lignes ximgproc");
        val = cv::getWindowProperty("Lignes", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("Lignes");
        val = cv::getWindowProperty("Lignes toutes", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("Lignes toutes");
        val = cv::getWindowProperty("Extrait", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("Extrait");
        val = cv::getWindowProperty("chiffre", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("chiffre");
        val = cv::getWindowProperty("gros", cv::WND_PROP_VISIBLE);
        if (val > 0)
            cv::destroyWindow("gros");
        val = cv::getWindowProperty("droit", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("droit");
        val = cv::getWindowProperty("avant rot", cv::WND_PROP_VISIBLE);
        if (val != 0)
            cv::destroyWindow("avant rot");
    }

    return 0;
}

// affichage du résultat sur un coin
void afficherResultat(cv::Mat result, cv::Point2i PT, std::string res)
{
    int pos = res.find('#');
    std::string texte = res.substr(0, pos);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    cv::Scalar colt(0, 0, 0);          // texte noir
    cv::Scalar rectColor(0, 255, 255); // sur fond jaune
    int epais = 1;

    // Obtenir la taille du texte
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(texte, font, scale, epais, &baseline);
    // baseline += epais;
    // Définir le coin inférieur gauche du rectangle
    cv::Point rectOrigin = PT + cv::Point(0, baseline);
    // Définir le coin supérieur droit du rectangle
    cv::Point rectCorner = rectOrigin + cv::Point(textSize.width, -3 * textSize.height / 2);
    // Dessiner le rectangle rempli avec la couleur rectColor
    int numcol;
    // Définition des symboles Unicode
    std::vector<std::string> symbols = {"♠", "♥", "♦", "♣"};
    std::string symcol;
    cv::Scalar coulsymb = cv::Scalar(0, 0, 255); // rouge
    if (texte[0] == 'C')
        numcol = 1;
    if (texte[0] == 'K')
        numcol = 2;
    if (texte[0] == 'P')
    {
        numcol = 0;
        coulsymb = cv::Scalar(0, 0, 0);
    } // noir sur fond jaune
    if (texte[0] == 'T')
    {
        numcol = 3;
        coulsymb = cv::Scalar(0, 128, 0);
    } // vert foncé sur fond jaune
    symcol = symbols[numcol];
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
#ifndef _WIN32
    ft2->loadFontData("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0);
#else
    ft2->loadFontData("C:/windows/fonts/arial.ttf", 0);
#endif

    std::string texteW = texte;
    texteW[0] = ' ';
    cv::rectangle(result, rectOrigin, rectCorner, rectColor, cv::FILLED);
    cv::putText(result, texteW, PT, font, scale, colt, epais);
    ft2->putText(result, symcol, PT, 10, coulsymb, -1, cv::LINE_AA, true);

    //if (printoption)
    //    cv::imshow("result", result);
}
