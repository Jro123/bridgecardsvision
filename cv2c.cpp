// Copyright Jacques ROSSILLOL 2024
//
// 
#define _USE_MATH_DEFINES
//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

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
#include "config.h"


#ifndef _WIN32
#include <thread>  // pour std::thread
#include <atomic>  // pour std::atomic
std::atomic<bool> is_window_open(true);
#endif
using std::max;
using std::min;



int waitoption = 1;   // 0 : pas d'attente après affichages
                      // 1 : attendre après le traitement d'une frame
                      // 2 : attendre après le traitement de chaque coin
                      // 3 :attendre après affichage du symbole et du chiffre
int printoption = 2;  // 0 : ne pas imprimer
                      // 1 : imprimer les lignes, coins détectés, OCR
                      // 2 : imprimer les calculs d'intensités et écarts types
std::string nomOCR = "tesOCR";                      

cv::Point2f computeIntersection(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4) {
    // Calculer les vecteurs directionnels
    cv::Point2f d1 = p2 - p1;
    cv::Point2f d2 = p4 - p3;

    // Résoudre les équations paramétriques
    float denom = d1.x * d2.y - d1.y * d2.x;
    if (denom == 0) {
        throw std::runtime_error("Les lignes sont parallèles et ne se croisent pas.");
    }

    float t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / denom;
    cv::Point2f intersection = p1 + t * d1;

    return intersection;
}
cv::Point2i calculerInter(cv::Vec4i l1, cv::Vec4i l2) {
    cv::Point2f pt = computeIntersection(cv::Point2f(l1[0], l1[1]), cv::Point2f(l1[2], l1[3]), cv::Point2f(l2[0], l2[1]), cv::Point2f(l2[2], l2[3]));

    cv::Point2i pti =  cv::Point2i(pt.x + 0.5, pt.y + 0.5);
    return pti;
}

bool PointEntreDeux(cv::Point2i M, cv::Point2i P, cv::Point2i Q) {
    // déterminer si la projection de M sur PQ est entre P et Q
    // calculer PM.PQ et comparer à PQ.PQ
    // PM.PQ < 0 : M hors du segment PQ , du coté P
    long pmpq = (M.x - P.x) * (Q.x - P.x) + (M.y - P.y) * (Q.y - P.y);
    if (pmpq < 0) return false;
    long pqpq = (Q.x - P.x) * (Q.x - P.x) + (Q.y - P.y) * (Q.y - P.y);
    if (abs(pmpq) > abs(pqpq)) return false;
    return true;
}


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

int processFrame(config& maconf, cv::Mat frame);
int processVideo(config& maconf, cv::String nomfichier) {
            cv::Mat img = cv::imread(nomfichier);
            if (!img.empty()) {
                processFrame(maconf, img);
                return 0;
            }

        // Ouvrir le fichier vidéo
        cv::VideoCapture cap(nomfichier);
        if (!cap.isOpened()) {
                std::cerr << "Erreur : Impossible d'ouvrir le fichier vidéo " << nomfichier << std::endl;
                return -1;
        }

        // Lire et afficher les frames 1 sur 25
        int nbf = 25;
        cv::Mat frame;
        while (true) {
            cap >> frame; // Capture une frame
            if (frame.empty()) {
                break; // Arrêter si aucune frame n'est capturée
            }
            if (nbf == 0) {
                nbf = 25;
                if (printoption) afficherImage("Frame", frame); // Afficher la frame
                processFrame(maconf, frame);

                // Attendre 30 ms et quitter si 'q' est pressé
                if (cv::waitKey(30) == 'q') {
                    break;
                }
            }
            nbf--;
        }

        cap.release(); // Libérer la capture vidéo
        //cv::destroyAllWindows(); // Fermer toutes les fenêtres ouvertes
        return 0;
    }

int main(int argc, char** argv) {
    config maconf;
    std::cout << " arguments optionels :" 
        <<" nom du fichier image ou video, " 
        <<" nom du fichier de configuration, "
        << " taille de carte (en pixels) " << std::endl<< std::endl;

    std::string nomfichier;
    nomfichier = setconfig(maconf); // initialisation par défaut

    if (argc > 1) nomfichier = argv[1];
    std::string nomconf; // nom du fichier de configuration
    if (argc > 3) maconf.hauteurcarte = std::stoi(argv[3]);
    else {
        size_t pos1 = nomfichier.find('_');
        size_t pos11 = nomfichier.find('_', pos1 + 1);
        if (pos11 != std::string::npos) pos1 = pos11;
        size_t pos2 = nomfichier.find('.', pos1);
    
        if (pos1 != std::string::npos && pos2 != std::string::npos) {
            // Extraire la sous-chaîne
            std::string extracted = nomfichier.substr(pos1 + 1, pos2 - pos1 - 1);
            maconf.hauteurcarte = std::stoi(extracted);
            //std::cout << "Sous-chaîne extraite : " << extracted << std::endl;
        } else {
            std::cout << "Délimiteurs non trouvés" << std::endl;
        }    
    }
    if (argc > 2) nomconf = argv[2];
    else nomconf = "COFALU.txt";
    lireConfig(nomconf, maconf);
    // else  if (maconf.hauteurcarte != 0) resetconfig(maconf.hauteurcarte, maconf);
    waitoption = maconf.waitoption;
    printoption = maconf.printoption;
    if(maconf.tesOCR == 0) nomOCR = "SERVEUR"; else nomOCR = "tesOCR";
    return processVideo(maconf, nomfichier);
}

int processFrame(config& maconf, cv::Mat image) {
    auto t0 = std::chrono::high_resolution_clock::now();

#define NBCOULEURS 7
    cv::Scalar couleurs[7];
    couleurs[0] = cv::Scalar(255, 128, 128); // bleu
    couleurs[1] = cv::Scalar(128, 255, 128); // vert 
    couleurs[2] = cv::Scalar(128, 128, 255); // rouge
    couleurs[3] = cv::Scalar(255, 255, 128); // turquoise
    couleurs[4] = cv::Scalar(255, 128, 255); // violet
    couleurs[5] = cv::Scalar(128, 255, 255); // jaune
    couleurs[6] = cv::Scalar(255, 255, 255); // blanc
    int c = 0;

    if (image.empty()) {
        std::cerr << "Erreur de chargement de l'image" << std::endl;
        return -1;
    }
    cv::Mat result = image.clone();
    auto start = std::chrono::high_resolution_clock::now();

    // afficher l'image en couleurs
    if (printoption) afficherImage("couleur", image);

    // Séparer les canaux Bleu, Vert, Rouge 
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(image, bgrChannels);
    // Utiliser seulement le canal Vert (par exemple) 
    cv::Mat greenChannel = bgrChannels[1];
    //afficherImage("vert", greenChannel);
    cv::Mat blueChannel = bgrChannels[0];
    //afficherImage("bleu", blueChannel);
    cv::Mat redChannel = bgrChannels[2];
    //afficherImage("rouge", redChannel);

    // Convertir en niveaux de gris
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);


    // Appliquer le flou gaussien pour réduire le bruit
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    //afficherImage("blur", blurred);


    ////////////////// utiliser une des images monochromatiques /////////////////
    cv::Mat grise;
    cv::cvtColor(gray, grise, cv::COLOR_GRAY2BGR);
    if (printoption) afficherImage("grise", grise);
    std::vector<cv::Vec4i> lines;
        int gmin = maconf.gradmin;
        int gmax = maconf.gradmax;

    int methode = 2; // 1 : canny et HoughLines,   2: ximgproc
    if (methode == 2) {
        // Appliquer le détecteur de segments de ligne LSD
        std::vector<cv::Vec4f> lines_f;

        // Paramètres du FastLineDetector : longueur minimale, écart entre lignes, etc.
        int length_threshold = maconf.nbpoints;     // Longueur minimale d'une ligne
        float distance_threshold = 1.41421356f; // Distance maximale entre deux points formant une ligne
        //float distance_threshold = 1.5f; // Distance maximale entre deux points formant une ligne
        double canny_th1 = gmin;       // Seuil bas pour Canny
        double canny_th2 = gmax;       // Seuil haut pour Canny
        int canny_aperture_size = 3;   // Taille de l'ouverture pour Canny
        bool do_merge = false;         // ne pas Fusionner les lignes adjacentes ( // )

        cv::Ptr<cv::ximgproc::FastLineDetector> lsd = cv::ximgproc::createFastLineDetector(
            length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, do_merge);

        lsd->detect(gray, lines_f);

        // Dessiner les segments de ligne détectés
        cv::Mat result;
        cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
        lsd->drawSegments(result, lines_f);
        // Convertir les coordonnées des lignes en entiers
        for (size_t i = 0; i < lines_f.size(); i++) {
            cv::Vec4i line_i(cvRound(lines_f[i][0]), cvRound(lines_f[i][1]), cvRound(lines_f[i][2]), cvRound(lines_f[i][3]));
            lines.push_back(line_i);
        }
        // Afficher l'image avec les lignes détectées
        afficherImage("lignes ximgproc", result);
        auto t11 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = t11 - t0;
    std::cout << "Temps initial : " << duree.count() << " secondes" << std::endl;
        cv::waitKey(0);
    }
        auto t22 = std::chrono::high_resolution_clock::now();
        cv::Mat edges;
        int iwait = 1;
        cv::Mat ima2;
        ima2 = grise.clone();
        cv::Canny(ima2, edges, gmin, gmax, 3, false);
        //cv::Canny(gray, edges, gmin, gmax, 3, false);
        if (printoption) afficherImage("bords", edges); cv::waitKey(iwait);

    if (methode == 1){
        // Utiliser la détection de contours de Canny
        // grossir l'image (désactivé)
        // canny (image, gradiant mini, gradiant maxi, ouverture)
        // gradient : variation d'intensité entre 2 pixels voisins
        // gradient mini : si le gradient calculé est inférieur, ce n'est pas un bord
        // gradiant maxi : si le gradient calculé est supérieur, c'est un bord

        // agrandir l'image, pour le calcul des lignes
        double mult = 1.0;  // meilleure valeur.
        //cv::resize(gray, gray, cv::Size(), mult, mult);


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
        if (printoption) afficherImage("bords", edges); cv::waitKey(iwait);

        // les lignes sont agrandies
        //
        for (int i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            lines[i][0] /= mult;
            lines[i][1] /= mult;
            lines[i][2] /= mult;
            lines[i][3] /= mult;
        }
    }

    // Dessiner les segments de droite et afficher leurs longueurs et extrémités
        //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < ima2.rows; y++) {
        for (int x = 0; x < ima2.cols; x++) {
            ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
        }
    }
    /***************/

    c = 0;
    double maxlg = 0;
    int il1 = 0;
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        cv::line(ima2, A,B, couleurs[c], 1);
        //cv::circle(ima2, A, 2, couleurs[c], -1);
        //cv::circle(ima2, B, 2, couleurs[c], -1);
        c++; if (c >= NBCOULEURS) c = 0;
        double length = std::sqrt(std::pow(l[2] - l[0], 2) + std::pow(l[3] - l[1], 2));
        if(printoption) std::cout << "Ligne " << i << ": (" << l[0] << ", " << l[1] << ") -> (" << l[2] << ", " << l[3] << "), Longueur: " << length << std::endl;
        if (length > maxlg) { maxlg = length; il1 = i; }
    }

    // Afficher l'image avec les segments de droite
    if (printoption) afficherImage("Lignes toutes", ima2);
    cv::waitKey(1);

    int lgmax = maconf.taillechiffre; lgmax *= lgmax;
    if (false) {
        // fusionner les lignes AB (la plus grande)  et CD si // si C et D sont proches de la ligne AB
        //   et si C ou D est proche de A ou B : AB --> AC ou AD ou BC ou BD
        double epsilon = max(1, maconf.deltacadre / 2);
        epsilon = min(1.5, epsilon);
        int deltamax = maconf.deltacoin;
        for (int k = 0; k < 5; k++) { // fusionner des lignes fusionnées, de plus en plus distantes

            for (int i = 0; i < lines.size(); i++) {
                cv::Vec4i l = lines[i];
                if (l[0] < 0) continue; // ligne invalidée
                cv::Point2i A(l[0], l[1]);
                cv::Point2i B(l[2], l[3]);
                double lg1 = ((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y)); // longueur**2
                for (int j = i + 1; j < lines.size(); j++) {
                    cv::Vec4i ll = lines[j];
                    if (ll[0] < 0) continue; // ligne invalidée
                    cv::Point2i C(ll[0], ll[1]);
                    cv::Point2i D(ll[2], ll[3]);
                    double lg2 = ((D.x - C.x) * (D.x - C.x) + (D.y - C.y) * (D.y - C.y));
                    if (lg1 > lg2) {
                        // distances de C ou D à AB > epsilon à préciser --> ignorer la ligne j
                        double dC = calculerDistance(C, A, B);
                        if (abs(dC) > epsilon) continue;
                        double dD = calculerDistance(D, A, B);
                        if (abs(dD) > epsilon) continue;
                    }
                    else {
                        double dA = calculerDistance(A, C, D);
                        if (abs(dA) > epsilon) continue;
                        double dB = calculerDistance(B, C, D);
                        if (abs(dB) > epsilon) continue;
                    }
                    // 4 points A B C D alignés. ignorer si l'écart entr AB et CD est important
                    // 
                    int xmin, xmax, ymin, ymax;
                    if (abs(A.x - B.x) > abs(A.y - B.y)) {
                        xmin = min(A.x, B.x);
                        if (xmin > C.x && xmin > D.x) { // AB à droite de CD
                            if ((xmin - C.x) > deltamax && (xmin - D.x) > deltamax) continue; // segments loins
                        }
                        else {
                            xmax = max(A.x, B.x);
                            if (C.x - xmax > deltamax && D.x - xmax > deltamax) continue;
                        }
                    }
                    else {
                        ymin = min(A.y, B.y);
                        if (ymin > C.y && ymin > D.y) { // AB à droite de CD
                            if ((ymin - C.y) > deltamax && (ymin - D.y) > deltamax) continue; // segments loins
                        }
                        else {
                            ymax = max(A.y, B.y);
                            if (C.y - ymax > deltamax && D.y - ymax > deltamax) continue;
                        }
                    }
                    // déterminer les extrémités après fusion : abs mini - abs maxi  // ord mini - maxi
                // utiliser x ou y 
                    cv::Point2i U(A), V(A); // futures extrémités
                    if (abs(A.x - B.x) > abs(A.y - B.y)) {
                        if (U.x > B.x) U = B;
                        if (U.x > C.x) U = C;
                        if (U.x > D.x) U = D;
                        if (V.x < B.x) V = B;
                        if (V.x < C.x) V = C;
                        if (V.x < D.x) V = D;
                    }
                    else {
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
                    if (lg2uv < maconf.hauteurcarte * maconf.hauteurcarte) {
                        // et invalider la ligne CD
                        lines[i][0] = U.x;
                        lines[i][1] = U.y;
                        lines[i][2] = V.x;
                        lines[i][3] = V.y;
                        A = U; B = V;
                        //invalider la ligne j
                        lines[j][0] = -1;
                        ll[0] = -1;
                    }
                } // next j
            }
            deltamax = 5 * deltamax / 4;
        }

        // invalider les lignes dont la longueur est inférieure à la taille du chiffre
        for (int i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            if (l[0] < 0) continue; // ligne invalidée
            cv::Point2i A(l[0], l[1]);
            cv::Point2i B(l[2], l[3]);
            double lg1 = ((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y)); // longueur**2
            if (lg1 < lgmax) lines[i][0] = -1; // invalider la ligne
        }
    }
    
    /* if (methode == 1) */ {
        // prolonger les lignes assez longues (au moins 1/6 de la hauteur de carte)
        // essayer de prolonger chaque ligne : regarder le pixel dans le prolongement de la ligne
        // ligne AB (B à droite de A) choisir une direction x ou y selon le maximum de |dx| et |dy|
        // AB selon X , prolongement en B : regarder le pixel blanc (dans edges) à droite (B.x +1, B.y)
        //   et le pixel blanc  à droite plus haut ou plus bas (B.x +1, B.y +- 1) (le plus proche de AB)
        //   à condition que les autres pixels proche de B soient noirs (dans edge) 
        // choisir le plus proche de AB, à distance de moins de 2 pixels de AB,  qui remplace B
        // même principe du coté A
        // itérer tant qu'on trouve des pixels blancs dans l'image des bords et noirs dans l'affichage des lignes
        double tolerance = 0.4;  // Ajustez la tolérance selon vos besoins. 0.4 entre 45 et 60 degrés
        cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8U);
        int maxlg = maconf.hauteurcarte / 6;
        maxlg *= maxlg;
        for (int i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            if (l[0] < 0) continue; // ligne invalidée
            cv::Point2i A(l[0], l[1]);
            cv::Point2i B(l[2], l[3]);
            int lgAB = (B.x - A.x)* (B.x - A.x) + (B.y - A.y)*(B.y - A.y);
            if (lgAB < maxlg) continue;
            // prolonger la ligne en A

            std::vector<cv::Point2i> contour;
            // on commence par prolonger en A
            // puis en B
            followContour(edges, A, B, contour, tolerance);
            // Obtenir l'extrémité du contour
            if (!contour.empty()) {
                cv::Point2i Z = contour.back();
                //std::cout << "L'extremite du contour est  (" << Z.x << ", " << Z.y << ")" << std::endl;
                // remplacer A par Z si A est entre B et Z
                // sinon, si B est entre Z et A, remplacer B par Z
                cv::Point2i  ab = B - A;
                cv::Point2i az = Z - A;
                int ps = ab.x * az.x + ab.y * az.y;
                if (ps <= 0) { // A entre B et Z : remplacer A par Z
                    if (printoption)  std::cout << i << " on remplace A " << A << " par " << Z << std::endl;
                    A = Z;
                }
            }
            else {
                if (printoption) std::cout << i << " Aucun contour trouve en A." << A << std::endl;
            }
            // prolonger en B
            int sz1 = contour.size();
            followContour(edges, B, A, contour, tolerance);
            // Obtenir l'extrémité du contour
            int sz2 = contour.size();
            if (sz2 > sz1) { // on a ajouté au moins un point
                cv::Point2i Z = contour.back();
                //std::cout << "L'extremite du contour est  (" << Z.x << ", " << Z.y << ")" << std::endl;
                // remplacer B par Z si A est entre B et Z
                if (printoption) std::cout << i << " on remplace B " << B << " par " << Z << std::endl;
                B = Z;
            }
            else {
                if (printoption) std::cout << i << "Aucun contour trouve en B." << B << std::endl;
            }
            if (printoption > 1) afficherImage("Contour", contourImage);
            cv::waitKey(1);
            lines[i][0] = A.x;
            lines[i][1] = A.y;
            lines[i][2] = B.x;
            lines[i][3] = B.y;
            for (const auto& P : contour) {
                contourImage.at<uchar>(P) = 255;
            }
        }
        
        // cv::waitKey(0);
    }


    // invalider les lignes dont la longueur est inférieure à la taille du chiffre + symbole
    // test :éliminer les ligne de longueur inférieure à la moitié de hauteur de carte
    // éliminer les lignes plus longues que la hauteur de carte
    lgmax = maconf.taillechiffre + maconf.taillesymbole;
    //lgmax = maconf.hauteurcarte / 6;   // test à valider
    lgmax *= lgmax;
    int lgmin = maconf.hauteurcarte + maconf.deltacadre; lgmin *=lgmin;
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        if (l[0] < 0) continue; // ligne invalidée
        cv::Point2i A(l[0], l[1]);
        cv::Point2i B(l[2], l[3]);
        double lg1 = ((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y)); // longueur**2
        if ((lg1 < lgmax) ||  (lg1 > lgmin) ){
            lines[i][0] = -1; // invalider la ligne
            if (printoption) std::cout << "supprime la ligne " << i << " " << A << "-" << B <<" longueur "<<std::sqrt(lg1)<< std::endl;
        } 
    }



    // 
    // afficher les lignes qui restent
    for (int y = 0; y < ima2.rows; y++) {
        for (int x = 0; x < ima2.cols; x++) {
            ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
        }
    }
    /***************/

    c = 0;
    maxlg = 0;
    il1 = 0;
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        if (l[0] < 0) continue; // ligne fusionnée ou ignorée car trop courte
        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        cv::line(ima2, A, B, couleurs[c], 1);
        //cv::circle(ima2, A, 2, couleurs[c], -1);
        //cv::circle(ima2, B, 2, couleurs[c], -1);
        c++; if (c >= NBCOULEURS) c = 0;
        double length = std::sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
        if (printoption) std::cout << "Ligne " << i << A<< " -> " << B<< ", Longueur: " << length << std::endl;
        if (length > maxlg) { maxlg = length; il1 = i; }
    }
    if (printoption) std::cout << "longueur maximale " << maxlg << std::endl;
    // Afficher l'image avec les segments de droite
    if (printoption) afficherImage("Lignes", ima2);
    cv::waitKey(1);



    //////////////////////////////// rechercher les coins des cartes ///////////////////
    //
    // 500 coins au maximum
    // indice de ligne 1, ligne2, indicateur d'extrémité de la ligne 1 (0 ou 2), ligne 2, x et y du point d'intersection 
    int coins[500][10]; // mémoriser tous les coins trouvés
    // O : indice de la première ligne
    // 1  : indice de la deuxième ligne
    // 2  : indicateur de sommet commun de la ligne 1  = 0 ou 2  (indice dans la ligne pour x
    // 3  : indicateur ligne 2 
    // 4  : x du point d'intersection
    // 5  : y
    // 6  : indicateur de coin d' un Roi Dame ou Valet  ( = 0 sinon)
    // 7  : x du cadre si R D V
    // 8  : y
    // 9  : non utilisé
    int nbcoins = 0;
    // pour chaque ligne AB
    for (int i = 0; i < lines.size(); i++) {
        cv::Vec4i l1 = lines[i];
        if (l1[0] < 0) continue; // ligne fusionnée ou effacée
        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        double lg1 = std::sqrt( (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
        if(printoption) std::cout <<i << " Ligne AB " << A << B<< " Longueur: " << lg1 << std::endl;
        il1 = i;

        // 
        // chercher, parmi les autres lignes la ligne orthogonale CD dont une extremité (C ou D)  est proche de A ou B
        // TODO: ou proche de la ligne AB entre A et B. ou dont A ou B est proche de la ligne CD entre C et D
        for (int j = i+1; j < lines.size(); j++) {
            double psX;
            // ligne CD ortogonale à AB ?
            // calculer le produit scalaire des vecteurs normés AB x CD / (maxlg * length)
            cv::Vec4i l2 = lines[j];
            if (l2[0] < 0) continue; // ligne fusionnée à une autre
            cv::Point2i C(l2[0], l2[1]);
            cv::Point2i D(l2[2], l2[3]);
            double lg2 = sqrt((D.x - C.x) * (D.x - C.x) + (D.y - C.y) * (D.y - C.y));
            psX = (B.x - A.x) * (D.x - C.x) + (B.y - A.y) * (D.y - C.y);   // AB.CD
            psX = psX / (lg1 * lg2);   // cosinus AB CD
            if (abs(psX) > maconf.cosOrtho) continue; // lignes non approximativement orthogonales
            coins[nbcoins][0] = 0;  // indice de ligne AB
            coins[nbcoins][1] = 0;  // indice de ligne CD
            coins[nbcoins][2] = 0;  // 0 si A proche du coin P = ABxCD     2 si proche de D
            coins[nbcoins][3] = 0;  // 0 si C proche de P    2 si D proche de P 
            coins[nbcoins][4] = 0;  // P.x
            coins[nbcoins][5] = 0;  // P.y
            coins[nbcoins][6] = 0;  //  1 c'est un Roi Dame ou Valet, sinon indéterminé
            coins[nbcoins][7] = 0;  // Q.x   coin du cadre de R D ou V
            coins[nbcoins][8] = 0;  // Q.y
            coins[nbcoins][9] = 0;  // inutilisé


            bool bCoin = false;
            // A proche de C ?
            if (abs(C.x - A.x) < maconf.deltacoin && abs(C.y - A.y) <  maconf.deltacoin) { // A proche de C
                if (printoption) std::cout << "  coin AC (" << A.x - C.x << "," << A.y - C.y << ") " << A<< "," << C  << std::endl;
                bCoin = true;
            }
            // A proche de D ?
            else if (abs(A.x - D.x) < maconf.deltacoin && abs(A.y - D.y) < maconf.deltacoin) { // A proche de D
                if (printoption) std::cout << "  coin AD (" << A.x - D.x << "," << A.y - D.y << ") " << A << "," << D << std::endl;
                coins[nbcoins][3] = 2;
                bCoin = true;
            }
            // B proche de C ?
            else if (abs(B.x - C.x) < maconf.deltacoin && abs(B.y - C.y) < maconf.deltacoin) { // B proche de C
                if (printoption) std::cout << "  coin BC (" << B.x - C.x << "," << B.y - C.y << ") " << B << "," << C << std::endl;
                bCoin = true;
                coins[nbcoins][2] = 2;
            }
            // B proche de D ?
            else if (abs(B.x - D.x) < maconf.deltacoin && abs(B.y - D.y) < maconf.deltacoin) { // B proche de D
                if (printoption) std::cout << "  coin BD (" << B.x - D.x << "," << B.y - D.y << ") " << B << "," << D << std::endl;
                bCoin = true;
                coins[nbcoins][2] = 2;
                coins[nbcoins][3] = 2;
            }
            if (bCoin) {
                // calculer le sinus de l'angle entre les deux droites
                double alfa = abs(psX) * 180.0/3.1416;

                if (printoption) std::cout << "  angle " << alfa <<" degres" << std::endl;
                //        mémoriser le coin : indices des deux droites et numéros des extrémités de chaque droite (0 ou 2)
                coins[nbcoins][0] = i; // indice première ligne
                coins[nbcoins][1] = j;   // indice deuxième ligne
                cv::Point2i P = calculerInter(l1, l2);
                coins[nbcoins][4] = P.x;
                coins[nbcoins][5] = P.y;
                double length = std::sqrt((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y));
                if (printoption) std::cout <<" "<< j << "  Ligne CD " << j << ": (" << C << ") -> (" << D << "), Longueur: " << length << std::endl;
                if (printoption) std::cout << " ==> coin "<<nbcoins<< " en " << P << " " << i << " " << j << " k " << coins[nbcoins][2] << " kk " << coins[nbcoins][3] << std::endl;
                nbcoins++;

            }
        } // deuxième droite
    } // première droite

    ////////////// on a déterminé les coins //////////////////////
    for (int i=0; i< nbcoins; i++){
        if (printoption) std::cout<<"coin "<< i<< " : "<<coins[i][4] << ", "<<coins[i][5]<<std::endl;
    }


    // déterminer la taille des cartes
    // déterminer les probables bords de carte 
    // deux coins sur une même ligne (ou deux ligne // proches), à distance vraissemblable (paramètre général de configuration)
    // la plus grande distance serait la hauteur de carte, sauf si plusieurs cartes sont alignées
    // une des autres devrait être dans le rapport des cotés de carte ( 3 / 2 )
    // 


    int htmax = 0;
    cv::Point2i P1, P2;
    for (int n = 0; n < nbcoins; n++) {
        int i = coins[n][0];
        int j = coins[n][1];

        if (i < 0 || j < 0) continue; // coin éliminé

        cv::Vec4i l1 = lines[i];
        cv::Vec4i l2 = lines[j];
        cv::Point2i A(coins[n][4], coins[n][5]);
        cv::Point2i H, K;  // extremités non communes sur les deux lignes : coin AH,AK
        int k = coins[n][2];
        H.x = l1[2 - k]; H.y = l1[3 - k];
        int kk = coins[n][3];
        K.x = l2[2 - kk]; K.y = l2[3 - kk];

        for (int m = n + 1; m < nbcoins; m++) {
            int ii = coins[m][0];
            int jj = coins[m][1];
            if (ii < 0 || jj < 0) continue;

            cv::Vec4i l11 = lines[ii];
            cv::Vec4i l22 = lines[jj];
            cv::Point2i HH, KK;     // coin A HH KK
            int k = coins[m][2];
            HH.x = l11[2 - k]; HH.y = l11[3 - k];
            int kk = coins[m][3];
            KK.x = l22[2 - kk]; KK.y = l22[3 - kk];

            cv::Point2i B(coins[m][4], coins[m][5]);
            // une des lignes commune avec une de l'autre coin?
            // TODO :
            // le coin B doit être sur une des lignes du coin A
            // le coin A doit être sur une des lignes du coin B
            // les deux autres lignes doivent être // et de même sens
            // AB semble alors etre un bord de carte
            //
            if ((ii == i) || (ii == j) || (jj == i) || (jj == j)) {
                cv::Point2i U, V;
                if (ii == i) { U = K; V = KK; }
                else if (ii == j) { U = H; V = KK; }
                else if (jj == i) { U = K; V = HH; }
                else if (jj == j) { U = H; V = HH; }
                else continue; // pas de ligne commune


                // lesdeux coins doivent être opposés : les deux autres lignes sont //  mais doivent être  de même sens
                // ligne commune A? = B?   
                // autres cotés AU et BV
                // calculer le produit vectoriel des segments des lignes AU et BV : doit être petit(en valeur absolue)
                // calculer le produit scalaire : doit être positif (origine = point d'intersection)
                int lg = (U.x - A.x) * (U.x - A.x) + (U.y - A.y) * (U.y - A.y);
                int lk = (V.x - B.x) * (V.x - B.x) + (V.y - B.y) * (V.y - B.y);

                int pv = (U.x - A.x) * (V.y - B.y) - (U.y - A.y) * (V.x - B.x);
                double sinalfa = (double)pv / sqrt((double)lg * (double)lk);
                // if (abs(sinalfa) > 0.02) continue; // autres cotés non parallèles (au moins 2 degrés) // test inutile
                int ps = (U.x - A.x) * (V.x - B.x) + (U.y - A.y) * (V.y - B.y);
                if (ps < 0) continue; // pas dans le même sens

                lg = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
                lg = std::sqrt(lg);
                if (lg > htmax && lg < maconf.hauteurcarte + maconf.deltacoin) {
                    P1 = A;
                    P2 = B;
                    htmax = lg;
                }
            }
            // sinon, il faut vérifier si la ligne qui joint les coins est // une ligne de chacun des coins
        }
        //if (htmax < 8 * maconf.hauteurcarte / 10) htmax = maconf.hauteurcarte;
    }
    // recalculer les paramètres de position sur la carte
    double htcard;
    if (maconf.hauteurcarte == 0) htcard = htmax;
    else htcard = maconf.hauteurcarte;
    if (abs(htmax - htcard) < maconf.deltacoin) htcard = htmax;
    if (htcard)
        {} //resetconfig(htcard, maconf);
    else {
        if (printoption) std::cout << " !!!!! impossible d'estimer la taille des cartes" << std::endl;
        if (printoption) std::cout << " !!!!! poursuite avec les caractéristiques standard " << std::endl;
    }
    // 
    if (printoption) std::cout << "hauteur carte : " << htcard << std::endl;


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
    for (int n = 0; n < nbcoins; n++) {
        int i = coins[n][0]; // indice de la première ligne
        int j = coins[n][1]; // indice de la deuxième ligne
        // if (i < 0 || j < 0) continue; // coin déjà éliminé // !!! poursuivre pour éliminer ce que P contient

        cv::Vec4i l1 = lines[abs(i)];
        cv::Vec4i l2 = lines[abs(j)];
        // point d'intersection des droites l1 et l2
        cv::Point2i P(coins[n][4], coins[n][5]);
        // déterminer le triangle RPS du coin
        int k = coins[n][2];  // 0 pour origine A, 2 pour extrémité B de la première ligne
        int kk = coins[n][3];  // 0 pour origine C, 2 pour extrémité D de la deuxième ligne
        k = 2 - k;  // pour indexer l'extrémité loin de P
        kk = 2 - kk;

        cv::Point2i R(l1[k], l1[k + 1]);
        cv::Point2i S(l2[kk], l2[kk + 1]);

        if (printoption) std::cout << "Coin " << n << " " << P << " , " << R << " , " << S << std::endl;

        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        cv::Point2i C(l2[0], l2[1]);
        cv::Point2i D(l2[2], l2[3]);
        double pvPRS = (R.x - P.x) * (S.y - P.y) - (R.y - P.y) * (S.x - P.x); // produit vectoriel PR ^ PS inversé car repère inversé. négatif sens trigo


        bool trouveQ = false;
        bool QdansP = false;
        bool eliminerP; // éliminer P après recherche de tous les coins contenus dans P
        eliminerP = false;
        for (int m = n + 1; m < nbcoins; m++) {
            int ii = coins[m][0];
            int jj = coins[m][1];
            if (ii < 0 || jj < 0) { // déjà éliminé
                // std::cout << "  coin " << m << " deja elimine " << std::endl;
                ii = abs(ii); jj = abs(jj);
                //continue;
                // continuer car le coin P est éventuellent dans le coin Q
                // ou Q est le sommet du cadre du coin P
            }
            cv::Vec4i l11 = lines[ii];
            cv::Vec4i l22 = lines[jj];

            cv::Point2i Q(coins[m][4], coins[m][5]);
            // ne pas éliminer le coin n ou m s'ils ne sont pas proches
            // tenir compte de l'imprécision de position des bords de carte (*2)
            // il peut y avoir trois droites // : le cadre d'un RDV et deux droites proches du bord de carte
            if (abs(Q.x - P.x) > 3*maconf.deltacadre/2 + 1) continue;  // *3/2 si oblique
            if (abs(Q.y - P.y) > 3*maconf.deltacadre/2 + 1) continue;

            int k = coins[m][2];  // 0 pour origine, 2 pour extrémité de la première ligne
            int kk = coins[m][3];  // 0 pour origine, 2 pour extrémité de la deuxième ligne
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
            if(i == j) pv = 0; else pv = calculerSinus(l1, l11);
            if (abs(pv) > maconf.deltaradian) { // AB  non // A'B'
                if (i == jj) pv = 0; else pv = calculerSinus(l1, l22);
                if (abs(pv) > maconf.deltaradian)   continue;  //  AB  non // C'D'
                estl22 = true;
            }

            // ignorer le coin Q  (QU QV) s'il n'est pas orienté come le coin P (PR PS)
            if (estl22) {    //  PR // QV
                // calcule le produit scalaire PR.QV    négatif si orientés en sens inverse
                int ps = (R.x - P.x) * (V.x - Q.x) + (R.y - P.y) * (V.y - Q.y);
                if (ps < 0) continue;   // aucun ne peut être cadre de l'autre
                // calcule le produit scalaire PS.QU    négatif si orientés en sens inverse
                ps = (S.x - P.x) * (U.x - Q.x) + (S.y - P.y) * (U.y - Q.y);
                if (ps < 0) continue;   // aucun ne peut être cadre de l'autre
            }
            else {
                // calcule le produit scalaire PR.QU   négatif si orientés en sens inverse
                int ps = (R.x - P.x) * (U.x - Q.x) + (R.y - P.y) * (U.y - Q.y);
                if (ps < 0) continue;   // aucun ne peut être cadre de l'autre
                // calcule le produit scalaire PS.QV    négatif si orientés en sens inverse
                ps = (S.x - P.x) * (V.x - Q.x) + (S.y - P.y) * (V.y - Q.y);
                if (ps < 0) continue;   // aucun ne peut être cadre de l'autre
            }

            // éliminer ce coin Q s'il est dans le coin P
            // 
            int dc = maconf.deltacadre;
            double d, dd; // distances algébriques de Q à PS et PR
            double epsilon = std::max(1, dc / 2);
            // tenir compte de l'orthogonalité PR xPS
            // projections de Q : H sur PR   K sur PS
            // PH = PQ.PR / ||PR||
            double lgpr = (R.x - P.x) * (R.x - P.x) + (R.y - P.y) * (R.y - P.y);
            lgpr = sqrt(lgpr); // longueur de PR
            double lgps = (S.x - P.x) * (S.x - P.x) + (S.y - P.y) * (S.y - P.y);
            lgps = sqrt(lgps); // longueur de PS
            dd = ((Q.x - P.x) * (R.x - P.x) + (Q.y - P.y) * (R.y - P.y)) / lgpr;
            d = ((Q.x - P.x) * (S.x - P.x) + (Q.y - P.y) * (S.y - P.y)) / lgps;

            if (std::max(abs(d), abs(dd)) > dc + epsilon) continue; // Q loin de PR ou de PS donc n'est pas le cadre
            if (coins[m][0] >= 0 ) { // Q pas encore éliminé
                bool elimQ= false;
                if(d>= 0 && d < dc + 2*epsilon && dd >= 0 && dd < dc + 2*epsilon) elimQ = true;

                //if ((d >= -epsilon && dd >= dc / 2) 
                //|| (dd >= -epsilon && d >= dc / 2) ) elimQ = true;
                if ((d >= -epsilon/4 && dd >= epsilon/4 && dd < dc + 2*epsilon) 
                || (dd >= -epsilon/4 && d >= epsilon && d < dc + 2*epsilon)  ) elimQ = true;
                if (elimQ) {
                    // Q à l'intérieur du coin P
                    // marquer le coin Q "éliminé"
                    coins[m][0] = -ii;
                    coins[m][1] = -jj;
                    if (printoption) std::cout<<" --> elimination du coin "<< m 
                         << " dans le coin "<< n <<std::endl;
                }
            }

            //  P est-il dans le coin Q
            // dans ce cas, d et dd sont négatifs, de valeur absolue inférieure à deltacadre
            if ((d<1 && dd < -epsilon && dd + dc >= -epsilon )
                || (dd< 1 && d < -epsilon && d + dc >= -epsilon)) {
                    eliminerP = true;
                    if (printoption) std::cout<<" --> elimination du coin "<< n 
                         << " dans le coin "<< m <<std::endl;
            }
            if (coins[m][0] >= 0) { // Q pas encore éliminé
                // P est-il le sommet du cadre de Q ?
                // a distance deltacadre du coté négatif des droites du coin Q
                if ((dd < 0 && std::abs(dd + dc) <= epsilon) 
                    && (d < 0 && std::abs(d + dc) <= epsilon))
                {
                // P est le sommet du cadre du coin Q
                    coins[m][6] = 1; //  Q estunRDV
                    coins[m][7] = P.x;
                    coins[m][8] = P.y;
                    eliminerP = true;
                    if (printoption) std::cout<<" --> elimination du coin "<< n 
                         << " dans le coin "<< m <<std::endl;
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

            if ( d > 0 && dd > 0 &&  std::abs(d - dc) <= epsilon && std::abs(dd - dc) <= epsilon)  {
                // Q est cadre à l'intérieur de PR et PS)
                coins[n][6] = 1; // estunRDV
                coins[n][7] = Q.x;
                coins[n][8] = Q.y;
                coins[m][0] = -ii; // éliminer Q
                coins[m][1] = -jj;
                continue;
            }


            if (coins[m][0] < 0) continue;   // Q éliminé 

            // Q n'est pas sur le cadre de P
        } // for m

        // éliminatio différée de P ?
        if (eliminerP) {  // c'est peut-être déjà fait
            if(coins[n][0] >= 0) {
                if (printoption) std::cout << "elimination coin " << n << std::endl;
                coins[n][0] = -i;
                coins[n][1] = -j;
            }
        }
        c++; c--; // pour pouvoir mettre un point d'arrêt
    } // for n

        // afficher ce qui reste selectionné
    cv::Mat imaC = ima2.clone();
    //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < imaC.rows; y++) {
        for (int x = 0; x < imaC.cols; x++) {
            imaC.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
        }
    }
    /***************/

    // afficher les coins  
    c = 0;
    for (int n = 0; n < nbcoins; n++) {
        int i = coins[n][0];
        int j = coins[n][1];
        cv::Point P(coins[n][4], coins[n][5]);

        cv::Vec4i l1 = lines[abs(i)];
        cv::Vec4i l2 = lines[abs(j)];

        cv::Point2i A(l1[0], l1[1]);
        cv::Point2i B(l1[2], l1[3]);
        cv::Point2i C(l2[0], l2[1]);
        cv::Point2i D(l2[2], l2[3]);
        cv::Vec4i nl1(A.x, A.y, B.x, B.y);
        cv::Vec4i nl2(C.x, C.y, D.x, D.y);

        // !!!! uniquement sur les copies

        // remplacer l'extremité qui convient par l'intersection
        int k = coins[n][2]; // quelle extrémité de la ligne 1?
        if (k != 0 && k != 2) {
            k = 0; // protection 
        }
        nl1[k] = coins[n][4];
        nl1[k + 1] = coins[n][5];
        k = coins[n][3];
        nl2[k] = coins[n][4];
        nl2[k + 1] = coins[n][5];

        // pour chacune des lignes l1 et l2, remplacer l'extrémité loin du coin par le milieu de la ligne
        // uniquement si la ligne est plus logue que la demi-largeur de carte et pour l'affichage : faire une copie

        k = coins[n][2]; // vaut 0 ou 2
        if (max(abs(B.x - A.x), abs(B.y - A.y)) > maconf.hauteurcarte / 3) {
            nl1[2 - k] = (A.x + B.x) / 2;
            nl1[3 - k] = (A.y + B.y) / 2;
        }
        k = coins[n][3];
        if (max(abs(D.x - C.x), abs(D.y - C.y)) > maconf.hauteurcarte / 3) {
            nl2[2 - k] = (C.x + D.x) / 2;
            nl2[3 - k] = (C.y + D.y) / 2;
        }

        if (i < 0 || j < 0) { // coin éliminé précédemment
            cv::circle(imaC, P, 2, cv::Scalar(255,255,255), -2); //  cercle blanc au sommet du coin
            cv::circle(grise, P, 2, cv::Scalar(0,0,255), -2); //  cercle rouge au sommet du coin
            // si ce coin ressemble à un cadre, afficher les lignes en trait fin gris
            cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), cv::Scalar(128, 128, 128), 1); // petit trait
            cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), cv::Scalar(128, 128, 128), 1); // petit trait
            continue; // coin éliminé
        }

        cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), couleurs[c], 1); // petit trait
        cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), couleurs[c], 1); // petit trait
        if (coins[n][6] > 0) cv::circle(imaC, P, 5, couleurs[c], 3); //  cercle au sommet du coin
        else {  
            cv::circle(imaC, P, 5, couleurs[c], 1); //  cercle épais (RDV) au sommet du coin
            cv::circle(grise, P, 5, couleurs[c], 1);
        }
        // TODO : afficher le numéro du coin
        std::string texte = std::to_string(n);
        cv::putText(imaC, texte, P, cv::FONT_HERSHEY_SIMPLEX, 1.0,
             couleurs[c], 1);
        c++; if (c >= NBCOULEURS) c = 0;

    }
    if (htmax > 4*maconf.hauteurcarte/5) {
        if (printoption) std::cout << "probable hauteur de carte : " << htmax << std::endl;
        cv::circle(imaC, P1, 6, cv::Scalar(0, 128, 128), 4);
        cv::circle(imaC, P2, 6, cv::Scalar(0, 128, 128), 4);
    }
    if (printoption) { 
        afficherImage("coins détectés", imaC);
        afficherImage("grise", grise);
    }
    cv::waitKey(1);



    //cv::waitKey(0);
    // extraire les coins
    // 
    bool estunRDV;
    estunRDV = false; // le coin contient-il un cadre ?
    cv::Point2i Q; // point du cadre
    std::string cartes[50];  // cartes trouvées
    int nbcartes = 0;


    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duree = t1 - t22;
    std::cout << "Temps préparatoire : " << duree.count() << " secondes" << std::endl;
    
    for (int n = 0; n < nbcoins; n++) {

        int cecoin[10];
        for (int i = 0; i < 10; i++) cecoin[i] = coins[n][i];
        int i = cecoin[0];  // indice de ligne
        int j = cecoin[1];
        if (i < 0 || j < 0) continue; // coin éliminé
        int l1W[4], l2W[4];

        cv::Vec4i l1 = lines[i];  // ligne AB
        cv::Vec4i l2 = lines[j];  // ligne CD
        for (int i = 0; i < 4; i++){
            l1W[i] = l1[i];
            l2W[i] = l2[i];
        }
        if (printoption) std::cout<<std::endl<<"coin "<<n<<"   ";
        std::string cartelue;
        cartelue = traiterCoin(cecoin, image,
            result, &l1W[0], &l2W[0], maconf); 
        if (waitoption > 1) cv::waitKey(0);  else cv::waitKey(1);// attendre 
        bool trouvee = false;
        if (cartelue != ""){
            for (int i=0; i < nbcartes; i++) {
                if (cartelue == cartes[i]) { trouvee = true; break;}
            }
            if (!trouvee) {
                cartes[nbcartes] = cartelue;
                nbcartes++;
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "Temps écoulé : " << elapsed.count() << " secondes" << std::endl;

    afficherImage("result", result);
    for(int i = 0; i < nbcartes; i++){
        char nomcol = cartes[i][0];
        std::string valeur = cartes[i].substr(1);
        std::string cartecouleur;
        if(nomcol == 'P') cartecouleur = "Pique ";
        else if(nomcol == 'C') cartecouleur = "Coeur ";
        else if(nomcol == 'K') cartecouleur = "Carreau ";
        else cartecouleur = "Trefle ";
        std::cout<<cartecouleur<<valeur<<std::endl;
    }
    std::cout << "====== fini ======" << std::endl;
    if (waitoption) cv::waitKey(0);
    {
    double val;
    val = cv::getWindowProperty("symbole", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("symbole");
    val = cv::getWindowProperty("orient", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("orient");
    val = cv::getWindowProperty("coin", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("coin");
    val = cv::getWindowProperty("Artefact", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("Artefact");
    val = cv::getWindowProperty("coins détectés", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("coins détectés");
    val = cv::getWindowProperty("Ext", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("Ext");
    val = cv::getWindowProperty("bords", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("bords");
    val = cv::getWindowProperty("lignes ximgproc", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("lignes ximgproc");
    val = cv::getWindowProperty("Lignes", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("Lignes");
    val = cv::getWindowProperty("Lignes toutes", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("Lignes toutes");
    val = cv::getWindowProperty("Extrait", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("Extrait");
    val = cv::getWindowProperty("chiffre", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("chiffre");
    val = cv::getWindowProperty("gros", cv::WND_PROP_VISIBLE);
    if (val > 0) cv::destroyWindow("gros");
    val = cv::getWindowProperty("droit", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("droit");
    val = cv::getWindowProperty("avant rot", cv::WND_PROP_VISIBLE);
    if(val > 0) cv::destroyWindow("avant rot");
    }

    return 0;
}
