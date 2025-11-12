// Copyright Jacques ROSSILLOL 2024
//
//
#define _USE_MATH_DEFINES
// #include <tesseract/baseapi.h>
// #include <leptonica/allheaders.h>

#define POSTGRESQL
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

#ifdef POSTGRESQL
#include <pqxx/pqxx>
#else
#include <sqlite3.h>
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp> // Inclure le module ximgproc pour LSD
#include <opencv2/freetype.hpp>
#include "config.h"


#include <vector>
#include <mutex>
#include <condition_variable>

#ifndef _WIN32
  #include <thread> // pour std::thread
  #include <atomic> // pour std::atomic
  std::atomic<bool> is_window_open(true);
#endif
// constantes 
//const char* couleurs[] = {"P", "C", "K", "T"}; // ♠, ♥, ♦, ♣
const char* nomval[14] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9","10", "V", "D", "R"};
const char* valeurcarte[14]  = {" ", "1","2", "3", "4", "5", "6", "7", "8", "9", "10", "V", "D", "R"};
const char* nomcouleur[4]  = {"Pique", "Coeur", "Carreau", "Trefle"}; 
const char* couleurcarte[4]  = {"P", "C", "K", "T"}; 
#define NBCOULEURS 10
cv::Scalar couleurs[10]; // initialisées dans processvideo()

// déclarations de fonctions locales
void afficherResultat(cv::Mat result, cv::Point2i PT, std::string res, cv::Scalar color = cv::Scalar(0,255,255));
int processFrame(config &maconf, cv::Mat frame, bool estvideo,
  std::vector<unecartePrec>& cartessPrec, unpli &monpli);

using namespace cv;
using namespace std;

int distribution[4][13][2]; // les 4 mains en cours de décodage
unecarte carteMort[13]; // la main du mort

// pour calibration:
static std::vector<Point2f> selectedPoints;

int waitoption = 1;   // 0 : pas d'attente après affichages
                      // 1 : attendre après le traitement d'une frame ou d'un pli
                      // 2 : attendre après le traitement de chaque coin ou frame
                      // 3 :attendre après affichage du symbole et du chiffre
int printoption = 2;  // 0 : ne pas imprimer
                      // 1 : résultats pour chaque frame
                      // 2 : imprimer les lignes, coins détectés, OCR
                      // 3 : imprimer les calculs d'intensités et écarts types
int threadoption = 1; // 0 : monotache
                      // 1 : autant que de coeurs
                      // n : nombre de sous-taches
std::string nomOCR = "tesOCR";

double Durees[5];

// Callback pour sélectionner les points
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN && selectedPoints.size() < 4) {
        selectedPoints.emplace_back(x, y);
        cout << "Point sélectionné : " << x << ", " << y << endl;
    }
}

// Fonction d'étalonnage
// retour : 0= passer à la frame suivante, 1=étalonnage effectué, 2=terminer
int calibratePerspective(const Mat& frame, const string& outputFile) {
    selectedPoints.clear();

    cv::Mat H;
    // Calcul du facteur d’échelle
    int maxDim = std::max(frame.cols, frame.rows);
    int scale = 1;
    for (int s : {8, 4, 2}) {
        if (s * maxDim <= 800) {
            scale = s;
            break;
        }
    }

    // Agrandissement de l’image
    cv::Mat display;
    resize(frame, display, Size(), scale, scale, INTER_LINEAR);

    cout << "Sélectionnez les 4 coins du rectangle dans l’image agrandie (clic gauche)." << endl;
    cv::namedWindow("Calibration", WINDOW_AUTOSIZE);
    cv::setMouseCallback("Calibration", mouseCallback, nullptr);

    while (selectedPoints.size() < 4) {
        cv::Mat temp = display.clone();
        for (const auto& pt : selectedPoints)
            cv::circle(temp, pt, 5, Scalar(0, 0, 255), -1);
        cv::imshow("Calibration", temp);
        if (cv::waitKey(30) == 27) return 2; // Échap pour annuler
        if (cv::waitKey(30) == 32) return 0; // Espace pour frame suivante
    }

    cv::destroyWindow("Calibration");

    // Conversion des points vers l’échelle originale
    std::vector<Point2f> originalPoints;
    for (const auto& pt : selectedPoints)
        originalPoints.emplace_back(pt.x / static_cast<float>(scale), pt.y / static_cast<float>(scale));


    // les deux premiers points A B sont la largeur de carte, dans le sens trigo, conservés
    // calculer la normale puis la position des points CC et DD du rectangle

    // Points cibles : rectangle redressé
    unpoint A(originalPoints[0].x, originalPoints[0].y);
    unpoint B(originalPoints[1].x, originalPoints[1].y);
    unpoint C(originalPoints[2].x, originalPoints[2].y);
    unpoint D(originalPoints[3].x, originalPoints[3].y);
    unvecteur AB(A,B);
    unvecteur BC(B,C);

      float lgl = AB.lg();
      float lgh = BC.lg();
      unvecteur normale = BC.normale();
      unpoint AA(B.x - lgl*normale.x, B.y - lgl*normale.y); 
      unpoint DD(C.x - lgl*normale.x, C.y - lgl*normale.y); 
      
      std::vector<Point2f> targetPoints = {
          cv::Point2f(AA.x, AA.y),
          originalPoints[1],
          originalPoints[2],
          cv::Point2f(DD.x, DD.y)
      };

      H = getPerspectiveTransform(originalPoints, targetPoints);

 // si l'angle ABC est presque droit, enregistrer une transformation identité
    unvecteur ab = AB.normale();
    unvecteur bc = BC.normale();
    float ps = ab*bc;
    if (abs(ps) < 0.02 ) { // environ 1 degré
      H = cv::Mat::eye(3, 3, CV_64F);
    }
    cv::FileStorage fs(outputFile, FileStorage::WRITE);
    fs << "homography" << H;
    fs.release();

    cout << "Homographie enregistrée dans " << outputFile << endl;
    return 1;
}

void applyCalibration(const cv::Mat& frame, cv::Mat& frameW, const cv::Mat& H, cv::Size rectSize) {

    cv::Mat warped;
    warpPerspective(frame, warped, H, rectSize);
    frameW = warped.clone();
    return;
}

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

bool enregistrerContratEtPli(const std::string& nomTable, int numeroDonne,
    const std::string& contratTexte, const std::string joueurContrat,
    int numpli, const Pli& cepli) {

#ifdef POSTGRESQL
    try {
        pqxx::connection conn("dbname=bridge user=jro password=jro");
        if (!conn.is_open()) {
            std::cerr << "❌ Connexion PostgreSQL échouée.\n";
            return false;
        }
        pqxx::work txn(conn);

        int table_id = -1, donne_id = -1, contrat_id = -1;

        // 🔍 Table
        auto r1 = txn.exec_params("SELECT id FROM tables WHERE nom = $1", nomTable);
        if (r1.empty()) {
            std::cerr << "Table '" << nomTable << "' introuvable.\n";
            return false;
        }
        table_id = r1[0][0].as<int>();

        // 🔍 Donne
        auto r2 = txn.exec_params("SELECT id FROM donnes WHERE numero = $1", numeroDonne);
        if (r2.empty()) {
            std::cerr << "Donne numéro " << numeroDonne << " introuvable.\n";
            return false;
        }
        donne_id = r2[0][0].as<int>();

        // 🔍 Contrat
        auto r3 = txn.exec_params(
            "SELECT id FROM contrats WHERE table_id = $1 AND donne_id = $2",
            table_id, donne_id);

        if (!r3.empty()) {
            contrat_id = r3[0][0].as<int>();
        } else {
            txn.exec_params(
                "INSERT INTO contrats (table_id, donne_id, joueur, contrat) VALUES ($1, $2, $3, $4)",
                table_id, donne_id, joueurContrat, contratTexte);
            auto r4 = txn.exec("SELECT lastval();");
            contrat_id = r4[0][0].as<int>();
        }

        // 🃏 Pli
        txn.exec_params(
            "INSERT INTO plis (contrat_id, numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "ON CONFLICT (contrat_id, numero) DO UPDATE SET "
            "carte_nord = EXCLUDED.carte_nord, "
            "carte_est = EXCLUDED.carte_est, "
            "carte_sud = EXCLUDED.carte_sud, "
            "carte_ouest = EXCLUDED.carte_ouest, "
            "joueur = EXCLUDED.joueur",
             contrat_id, numpli,
            carteToString(cepli.carte[0].couleur, cepli.carte[0].valeur),
            carteToString(cepli.carte[1].couleur, cepli.carte[1].valeur),
            carteToString(cepli.carte[2].couleur, cepli.carte[2].valeur),
            carteToString(cepli.carte[3].couleur, cepli.carte[3].valeur),
            joueurToString(cepli.joueur));

        txn.commit();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur PostgreSQL : " << e.what() << std::endl;
        return false;
    }

#else
    sqlite3* db;
    sqlite3_stmt* stmt;
    int rc = sqlite3_open("bridge.db", &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Erreur ouverture base: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    int table_id = -1, donne_id = -1, contrat_id = -1;

    // 🔍 Table
    rc = sqlite3_prepare_v2(db, "SELECT id FROM tables WHERE nom = ?", -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, nomTable.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW)
            table_id = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    if (table_id == -1) {
        std::cerr << "Table '" << nomTable << "' introuvable.\n";
        sqlite3_close(db);
        return false;
    }

    // 🔍 Donne
    rc = sqlite3_prepare_v2(db, "SELECT id FROM donnes WHERE numero = ?", -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, numeroDonne);
        if (sqlite3_step(stmt) == SQLITE_ROW)
            donne_id = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    if (donne_id == -1) {
        std::cerr << "Donne numéro " << numeroDonne << " introuvable.\n";
        sqlite3_close(db);
        return false;
    }

    // 🔍 Contrat
    rc = sqlite3_prepare_v2(db,
        "SELECT id FROM contrats WHERE table_id = ? AND donne_id = ?;",
        -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, table_id);
        sqlite3_bind_int(stmt, 2, donne_id);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            contrat_id = sqlite3_column_int(stmt, 0);
        } else {
            sqlite3_finalize(stmt);
            rc = sqlite3_prepare_v2(db,
                "INSERT INTO contrats (table_id, donne_id, joueur, contrat) VALUES (?, ?, ?, ?);",
                -1, &stmt, nullptr);
            if (rc == SQLITE_OK) {
                sqlite3_bind_int(stmt, 1, table_id);
                sqlite3_bind_int(stmt, 2, donne_id);
                sqlite3_bind_text(stmt, 3, joueurContrat.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(stmt, 4, contratTexte.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    std::cerr << "Erreur insertion contrat: " << sqlite3_errmsg(db) << std::endl;
                    sqlite3_finalize(stmt);
                    sqlite3_close(db);
                    return false;
                }
            }
            sqlite3_finalize(stmt);
            rc = sqlite3_prepare_v2(db, "SELECT last_insert_rowid();", -1, &stmt, nullptr);
            if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW)
                contrat_id = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
        }
    } else {
        std::cerr << "Erreur vérification contrat: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return false;
    }

    // 🃏 Pli
    rc = sqlite3_prepare_v2(db,
        "INSERT INTO plis (contrat_id, numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur) VALUES (?, ?, ?, ?, ?, ?, ?);",
        -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, contrat_id);
        sqlite3_bind_int(stmt, 2, numpli);
        sqlite3_bind_text(stmt, 3, carteToString(cepli.carte[0].couleur, cepli.carte[0].valeur).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, carteToString(cepli.carte[1].couleur, cepli.carte[1].valeur).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, carteToString(cepli.carte[2].couleur, cepli.carte[2].valeur).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, carteToString(cepli.carte[3].couleur, cepli.carte[3].valeur).c_str(), -1, SQLITE_TRANSIENT);

        sqlite3_bind_text(stmt, 7, joueurToString(cepli.joueur).c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Erreur insertion pli: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            return false;
        }
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return true;
#endif
}

int processVideo(config &maconf, cv::String nomfichier)
{
    cv::Size rectSize(500, 500); // Exemple : rectangle 3:2
    std::string calibFile = "calibration.yml";
    bool isTransform = false;  // transformation homographique ?  
    cv::Mat Htrans; // matrice de la transformation homographique

    std::chrono::duration<double> duree;
    int numeroframe = 0;
    unpli monpli;   // pli en cours de décodage
    std::vector<unecartePrec> cartesPrec; // cartes de la frame précédente
    //std::vector<uncoinPrec> coinsPrec;

    couleurs[0] = cv::Scalar(255, 128, 128); // bleu clair
    couleurs[1] = cv::Scalar(128, 255, 128); // vert clair
    couleurs[2] = cv::Scalar(128, 128, 255); // rouge vif
    couleurs[3] = cv::Scalar(255, 255, 128); // turquoise
    couleurs[4] = cv::Scalar(255, 128, 255); // violet
    couleurs[5] = cv::Scalar(128, 255, 255); // jaune
    couleurs[6] = cv::Scalar(0, 128, 128); // marron
    couleurs[7] = cv::Scalar(255,0,0); // bleu foncé
    couleurs[8] = cv::Scalar(0,255,0); // vert foncé
    couleurs[9] = cv::Scalar(0,0,255); // rouge foncé

    cv::Mat img = cv::imread(nomfichier);
    if (!img.empty()) // c'est une image fixe
    {
        processFrame(maconf, img, false, cartesPrec, monpli);
        return 0;
    }

    // Ouvrir le fichier vidéo
    cv::VideoCapture cap(nomfichier);
    if (!cap.isOpened())
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier vidéo " << nomfichier << std::endl;
        return -1;
    }


    FileStorage fs(calibFile, FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Fichier de calibration introuvable. Image non transformée." << endl;
    } else {
      fs["homography"] >> Htrans;
      fs.release();
      isTransform = true;
    }
    auto t0 = std::chrono::high_resolution_clock::now();


    // Lire et afficher les frames
    int nbf = 0;
    cv::Mat frame;
    cv::Mat frameW;
    cv::Mat framePrec;
    cv::Mat framePli;  // image contenant un pli complet
    cv::Mat result;
    cv::Mat frameTotale;
    cv::Mat image;
    cv::Mat diff;
    bool bPremier = true;

    // int distribution[4][13][2];  // 4 joueurs (NSEO) 13 cartes couleur (0 1 2 3) valeur (1 à 13)
    // NSEO 13 cartes couleur et valeur
    for (int i=0; i < 4; i++)
      for (int j=0; j < 13;j++){
        distribution[i][j][0] = -1; // couleur inconnue
        distribution[i][j][1] = 0; // valeur inconnue
      }
        
    unpli monpliprec; // pli précédent, cartes dans l'ordre d'apparition
    Pli cepli;  // pli en cours dans l'odre Nord Est Sud Ouest
    Pli pliprec; // pli précédent
    int j1 = maconf.declarant + 1; // entame par le joueur qui suit le déclarant et précède le mort
    j1 = j1%4; 
    cepli.joueur = j1;
    int numpli = 0;
    int nbcartes = 0;  // nombre de cartes dans le pli en cours
    bool mortAnalyse = false; // indique si l'analyse du mort a été faite
    int joueurMort = 0; // numéro de joueur du mort (0=Nord, 1=Est, 2=Sud, 3=Ouest). calculé plus tard
    joueurMort = (maconf.declarant +2 ) % 4;

    int nbCartesDuPli = 0;
    while (true) { // boucle sur les frames
      cap >> frame; // Capture une frame
      if (!frame.empty() && printoption > 1 ) afficherImage("Frame", frame); // Afficher la frame
      
      // étalonner la prise de vue ?
      if (maconf.calibrationoption) {
        int rc = calibratePerspective(frame, calibFile);
          if (rc == 0)  continue; // frame suivante
          else if (rc == 2) break; // calibration validée, fin du programme
      }
      // redresser l'image
      cv::Size rectSize;
      rectSize.height = frame.rows;
      rectSize.width = frame.cols;
      cv::Mat frameW = frame.clone(); 
      if (!frame.empty()) {
        if (isTransform && frame.rows > 0 && frame.cols > 0)
          applyCalibration(frame, frameW, Htrans, rectSize);
          frame = frameW;
      }

      // extraire la partie de l'image où sont posées les cartes jouées
      if (!frame.empty()) {
        frameTotale = frame.clone();
        cv::Rect r;
        r.x = maconf.xjeu;
        r.y = maconf.yjeu;
        r.width = maconf.wjeu;
        r.height = maconf.hjeu;
        frame = frameTotale(r);
        if (framePli.cols == 0) framePli = frame.clone();
      }

#ifdef ACTIVER
      // comparer à la frame précédente
      // extraire la partie modifiée (la première fois : tout)
      // conserver le tableau des coins identifiés
      //   pour chaque coin : position, couleur et valeur carte
      // invalider les coins sur une zone modifiée
      // extraire l'image modifiée
      // traiter cette image en ajoutant les nouvaux coins
      cv::cvtColor(frame, frameW, cv::COLOR_BGR2GRAY);
      if (bPremier) { bPremier = false; image = frame.clone();}
      else {
          // extraire l'image modifiée
          cv::absdiff(framePrec, frameW, diff);
          cv::threshold(diff, diff, 30, 255, cv::THRESH_BINARY);
          // nettoyer le vecteur coinsPrec
          coinsPrec.clear();
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
          processFrame(maconf, frame, true, cartesPrec, monpli);
          framePrec = frameW.clone();
          // Attendre 30 ms et quitter si 'q' est pressé
          if (cv::waitKey(30) == 'q')
          {
              break;
          }
      }
#endif 
      //        définir un tableau des cartes jouées par chacun des 4 joueurs
      //        définir un tableau de 4 cartes jouées à chaque pli
      //        après le traitement de chaque frame
      //             vérifier que chaque carte détectée n'a pas déjà été jouée
      //                  (à faire seulement ou également lors du décodage d'une carte)
      //             comparer au pli en cours. 
      //             pas de retrait de carte du pli en cours, mais signaler
      //             ajout d'une nouvelle carte uniquement si le pli n'est pas déjà complet
      //                  sinon, signaler le problème 

      if (! frame.empty())  processFrame(maconf, frame, true, cartesPrec, monpli);

      // s'il n'y a aucune carte dans cette trame et si il y a 4 cartes dans le pli en cours:
      //        enregistrer le pli en tenant compte du joueur qui a entamé le pli
      //        vérifier en considérant la position des 4 cartes du pli (carte Nord est en haut de l'image)
      //        déterminer le joueur (N E S O) qui remporte le pli en fonction du contrat
      //            --> joueur qui entame le pli suivant
      //        initialiser le pli
      //        pour chaque carte du vecteur coinsPrec :
      ///           si elle n'est pas dans le pli en cours : l'ajouter au pli
      //            si c'est une autre nouvelle carte du pli : erreur
      //
      bool estvide = true;

      // rechercher dans le vecteur des cartes précédentes 

      // cas particulier de la vidéo FUNBRIDGE :
      // si le pli en cours est complet et s'il y a une seule carte dans cette frame,
      //    enregistrer le pli en cours et ignorer la carte détectée
      int nbcf(0); // comptage des cartes de cette frame (0, 1 ou 2)
      int cc(-1), vv(0);
      // frame vide ? aucune carte (même avec un seul coin) détectée
      for (auto& ucp : cartesPrec){
        int c = ucp.couleur;
        int v = ucp.valeur;
        if (c < 0) continue; // couleur non déterminée
        if (v <= 0) continue; // valeur non déterminée
        if (v > 13) continue; // valeur invalide
        // actualiser les positions si cette carte est dans le pli
        for (auto& uc : monpli.cartes) {
          if (uc.couleur != c || uc.valeur != v) continue;
          for (int i=0; i< ucp.coinsPrec.size(); i++){
            auto& cp = ucp.coinsPrec[i];
            if (i < 4) {
              uc.sommet[i].x = cp.x;
              uc.sommet[i].y = cp.y;
            }
            uc.xmin = std::min(uc.xmin, cp.x);
            uc.ymin = std::min(uc.ymin, cp.y);
            uc.xmax = std::max(uc.xmax, cp.x);
            uc.ymax = std::max(uc.ymax, cp.y);
          }
          break;
        }

        if (cc < 0) {
            nbcf = 1;
        } else {
          if (c != cc || v != vv) {nbcf = 2;}
        }
      }
      if (nbcf > 0) estvide = false;

      nbcartes = monpli.nbcartes;

      if (nbcartes < 2) estvide = false; // une seule carte trouvée dans les frames précédentes


      //  si c'est le premier pli et la troisième carte jouée, analyser la zone du mort
      //        la frame à analyser ne comporte aucune carte jouée
      //        la position de cette zone est indiquée dans la configuration (pour N E S et O)
      //        extraire l'image correspondante, la redresser (rotation de 0 1 2 ou 3 angles droits)
      //        composée de 1 à 4 colonnes de cartes
      //        chaque colonne est d'une seule couleur (P C K T)
      //        il reste 12 cartes (le mort à joué le premier pli)
      //        dans chaque colonne, seule la dernière carte (la plus petite) est complète
      //          on y trouve un gros symbole, qui permet d'obtenir la couleur
      //          les autres cartes ne sont visibles que pour les 2 coins du haut de carte
      //        mémoriser les cartes du mort
      //
      //        déterminer qui est le mort (N S E O) en fonction de la zone contenant des cartes
      //        extraire et redresser l'image de cette zone
      //        
      //
      // si c'est un autre pli : vérifier qu'une des 4 cartes a été jouée par le mort
      //                         on en déduit le premier joueur (N E S O) du pli
      //                         valider avec le calcul selon les règles du bridge
      if (maconf.wmort> 0 && numpli == 0 && monpli.nbcartes == 3 && !mortAnalyse){
        // extraire la zone du mort
        // redresser de 90 180 ou 270 degrés si le déclarant est Ouest, Nord ou Est
        cv::Mat imaMort;
        cv::Rect r;
        cv::Mat lig;
        cv::Scalar m, m0, m1, m2, m3; // couleur moyenne des 4 zones
        int d0, d1, d2, d3; // écarts de couleur entre les deux moitiés
        // zone Nord :
        r.x = maconf.xmort;
        r.y = maconf.ymort;
        r.width = maconf.wmort /2;
        r.height = maconf.hmort;
        lig = frameTotale(r); m0 = cv::mean(lig); // moitié gauche
        r.x += r.width;
        lig = frameTotale(r); m = cv::mean(lig); // moitié droite
        d0 = std::abs(m0[0] - m[0]) + std::abs(m0[1] - m[1]) +std::abs(m0[2] - m[2]);

        // zone Est
        r.width = maconf.hmort;
        r.x = frameTotale.cols - r.width;
        r.y = maconf.xmort;
        r.height = maconf.wmort / 2; // moitié haute
        lig = frameTotale(r); m1 = cv::mean(lig); // moitié gauche
        r.y += r.height;
        lig = frameTotale(r); m = cv::mean(lig); // moitié droite
        d1 = std::abs(m1[0] - m[0]) + std::abs(m1[1] - m[1]) +std::abs(m1[2] - m[2]);

        // zone Sud
        r.x = maconf.xmort;
        r.width = maconf.wmort /2; // moitié gauche
        r.height = maconf.hmort;
        r.y = frameTotale.rows - r.height;
        lig = frameTotale(r); m2 = cv::mean(lig); // moitié gauche
        r.x += r.width;
        lig = frameTotale(r); m = cv::mean(lig); // moitié droite
        d2 = std::abs(m2[0] - m[0]) + std::abs(m2[1] - m[1]) +std::abs(m2[2] - m[2]);

        // zone Ouest
        r.width = maconf.hmort;
        r.x = 0;
        r.y = maconf.xmort;
        r.height = maconf.wmort / 2; // moitié 
        lig = frameTotale(r); m3 = cv::mean(lig); // moitié haute
        r.y += r.height;
        lig = frameTotale(r); m = cv::mean(lig); // moitié basse
        d3 = std::abs(m3[0] - m[0]) + std::abs(m3[1] - m[1]) +std::abs(m3[2] - m[2]);

        // choisir la zone où l'écart entre les deux moitiés est maximal
        cv::Mat rotated;
        if (d0 > d1 && d0 > d2 && d0 > d3) {
          // extraire la zone Nord
          r.x = maconf.xmort;
          r.y = maconf.ymort;
          r.width = maconf.wmort;
          r.height = maconf.hmort;
          imaMort = frameTotale(r).clone();
          joueurMort = 0;
        }
        else if (d1> d0 && d1 > d2 && d1 > d3){
          // extraire la zone Est et tourner de 90 degrés 
          r.width = maconf.hmort;
          r.x = frameTotale.cols - r.width;
          r.y = maconf.xmort;
          r.height = maconf.wmort;
          imaMort = frameTotale(r).clone();
          cv::rotate(imaMort, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
          imaMort = rotated;
          joueurMort = 1;
        }
        else if (d2> d0 && d2 > d1 && d2 > d3){
          // extraire la zone Sud et tourner de 180 degrés
          r.x = maconf.xmort;
          r.width = maconf.wmort;
          r.height = maconf.hmort;
          r.y = frameTotale.rows - r.height;
          imaMort = frameTotale(r).clone();
          cv::rotate(imaMort, rotated, cv::ROTATE_180);
          imaMort = rotated;
          joueurMort = 2;
        }
        else if (d3> d0 && d3 > d1 && d3 > d2){
          // extraire la zone Ouest et tourner de -90 degrés 
          r.width = maconf.hmort;
          r.x = 0;
          r.y = maconf.xmort;
          r.height = maconf.wmort; 
          imaMort = frameTotale(r).clone();
          cv::rotate(imaMort, rotated, cv::ROTATE_90_CLOCKWISE);
          imaMort = rotated;
          joueurMort = 3;
        }

        // extraire les colonnes de carte (1 à 4) de chaque couleur
        // traiter chaque colonne
        //   déterminer la couleur sur la carte la plus basse (le coin ayant x maximal)
        //   décoder cette carte
        //   itérer sur les autres cartes :
        //      extraire la zone qui ne contient pas (lehaut de) cette carte
        //      déterminer les deux coins supérieurs de la carte
        //      décoder la carte
        std::cout << "traitement des cartes du mort"<<std::endl;
        for (int i = 0; i<13; i++){
          carteMort[i].couleur = -1;
          carteMort[i].valeur = 0;
        }
        carteMort[0].couleur = cepli.carte[joueurMort].couleur;
        carteMort[0].valeur = cepli.carte[joueurMort].valeur;
        if(printoption > 0) afficherImage("Mort", imaMort);
        // spécifique FUNBRIDGE
        int saveLargeur = maconf.largeurcarte;
        maconf.largeurcarte *= float(0.95);
        traiterMort(maconf, imaMort, carteMort);
        maconf.largeurcarte = saveLargeur;

        mortAnalyse = true;
      } // décodage du mort

      if ((estvide   || frame.empty()) && numpli < 13) {
        if (monpli.nbcartes > 0)
        { // un pli en cours (au moins une carte jouée)
          // on complète le pli avec le 2 de Trefle (ou carreau si trefle est atout)
          // déterminer le gagnant
          //
          int couldef = 3;
          if (maconf.contratcouleur == 3) couldef = 2;
          unecarte uc;
          uc.couleur = couldef;
          uc.valeur = 2;
          for (int i = monpli.nbcartes; i < 4; i++){
            monpli.cartes[i] = uc;
            std::cout<<" pli incomplet"<<std::endl;
          }
          monpli.nbcartes = 4; // le pli est complet ou vient d'être complété
          bool estincomplet = false;

          
          // dans le pli en cours (monpli), on a les 4 cartes jouées (dans l'ordre du jeu)
          // déterminer les joueurs à partir de la position des 4 cartes
          // vérifier que c'est compatible avec cepli.joueur de la première carte
          // trouver la carte jouée par Nord : la plus haute (y minimal)
          // vérifier que la carte jouée par Est est la plus à droite
          //        que la carte jouée par Sud est la plus basse
          //        que la carte jouée par Ouest  est la plus à gauche
          int indiceNord = 0; // indice de la carte jouée par Nord dans ce pli 
          int indiceOuest = 0; 
          int indiceSud = 0; 
          int indiceEst = 0; 
          int yNord = 12345;
          int xOuest = 12345;
          int xEst = 0;
          int ySud = 0;
          for (int i = 0; i< 4; i++){ // les 4 cartes du pli
            auto& uc = monpli.cartes[i];
              if (uc.ymin < yNord) {
                yNord = uc.ymin;
                indiceNord = i;
              }
              if (uc.xmin < xOuest) {
                xOuest = uc.xmin;
                indiceOuest = i;
              }
              if (uc.xmax > xEst) {
                xEst = uc.xmax;
                indiceEst = i;
              }
              if (uc.ymax > ySud) {
                ySud = uc.ymax;
                indiceSud = i;
              }
          }
          // constituer le pli des cartes Nord, Est, Sud et Ouest
          cepli.carte[0] = monpli.cartes[indiceNord];
          cepli.carte[1] = monpli.cartes[indiceEst];
          cepli.carte[2] = monpli.cartes[indiceSud];
          cepli.carte[3] = monpli.cartes[indiceOuest];

          int joueur1 = 4 - indiceNord; if (joueur1 == 4) joueur1 = 0;
          // vérifier la compatibilité avec le joueur qui a entamé ce pli
          if (joueur1 != cepli.joueur) {
            std::cout<<"!!! la position des cartes est incompatible avec le premier joueur du pli"<<std::endl;
          }
          // on a la carte jouée par Nord : monpli.cartes[indiceNord]
          // cepli.carte[0] : carte jouée par Nord
          // monpli.carte[0] : premiére carte jouée, celle jouée par cepli.joueur


          // on affiche les valeurs trouvées
          if (maconf.printoption >= 0) {
            cv::Scalar colorFond;
            result = framePli.clone();
            for (int k=0; k<4; k++){ // 4 cartes du pli
              if (k==0) colorFond = cv::Scalar(128,255,255);
              else if (k==1) colorFond = cv::Scalar(255,255,192);
              else if (k==2) colorFond = cv::Scalar(255,192,255);
              else colorFond = cv::Scalar(192,192,192);
              unecarte& uc = cepli.carte[k];
              int numcol = uc.couleur;
              char nomcol = '?';
              if (numcol >= 0 && numcol <= 3) nomcol = couleurcarte[numcol][0];
              std::string val = valeurcarte[uc.valeur];
              std::string res = nomcol + val;
              for (int i= 0; i<4; i++){
                afficherResultat(result, uc.sommet[i], res, colorFond);
              }
            }   
            afficherImage("result", result); cv::waitKey(1);
          }
          
          int j1 = cepli.joueur;
          int jgagnant = j1; // a priori le même joueur emporte le pli
          int coul = cepli.carte[j1].couleur;
          int val = cepli.carte[j1].valeur; if (val == 1) val = 14; // As > R
          int n = 0;
          for (n=0; n < 4; n++){
            int c = cepli.carte[n].couleur;
            if (c < 0) {
              std::cout<<" pli incomplet"<<std::endl;
              c = 3;
              if (maconf.contratcouleur == 3) c = 2;
              cepli.carte[n].couleur = c; // compléter avec le 2 de trefle (ou carreau si trefle est atout)
            }
            if (cepli.carte[n].valeur <= 0) cepli.carte[n].valeur = 2;

            if (c != coul && c != maconf.contratcouleur) continue;
            if (c == maconf.contratcouleur && coul != maconf.contratcouleur) {
              coul = maconf.contratcouleur;
              val = cepli.carte[n].valeur;
              jgagnant = n;
            } else {
              int v = cepli.carte[n].valeur; if (v == 1) v = 14;
              if (v >= val){
                val = v; jgagnant = n;
              }
            }
          }
          
          // mémoriser les 4 cartes dans la distribution
          for (int k = 0; k < 4; k++){  // N E S O
            int c = cepli.carte[k].couleur;
            int v = cepli.carte[k].valeur;
            distribution[k][numpli][0] = c; // couleur
            distribution[k][numpli][1] = v; // valeur
            // vérifier que la carte jouée par le mort est dans les cartes du mort
            // et que la carte jouée par un autre n'est pas dans les cartes du mort
            if (numpli > 0){
              int i;
              if (k == joueurMort){ // carte du mort
                for (i = 0; i < 13; i++){
                  if (c == carteMort[i].couleur && v == carteMort[i].valeur) break;
                }
                if (i == 13) std::cout<<"!!! carte couleur "<<c<< " valeur "<<v
                <<" n'est pas dans le jeu du mort"<<std::endl;
              } else {
                for (i = 0; i < 13; i++){
                  if (c == carteMort[i].couleur && v == carteMort[i].valeur){
                    std::cout<<"!!! carte couleur "<<c<< " valeur "<<v
                      <<" est dans le jeu du mort"<<std::endl;
                    break;
                  }
                }
              }
            }
          } // for k
          cepli.joueurgagnant = jgagnant;
          pliprec = cepli;
          cepli = Pli(); cepli.joueur = jgagnant;
          // afficher les cartes du pli
          numpli++;
          std::cout<<"==> pli "<<numpli<< " joueur " << joueurToString(pliprec.joueur) << "  frame "<< numeroframe <<std::endl;
          for(int i=0; i< 4; i++){
            std::string s = carteToString(pliprec.carte[i].couleur, pliprec.carte[i].valeur);
            if (pliprec.joueur == i) s = "-->" +s; else s = "   " + s;
            if (pliprec.joueurgagnant == i) s += "-->";
            s = "        " + s;
            std::cout<<"      "<<s<<std::endl;
          }
          // enregistrer le pli complet dans la base de données
          std::string contrat;
          contrat = maconf.contratvaleur;
          if (maconf.contratcouleur < 0) contrat += "SA";
          //else contrat += lettrecouleur[maconf.contratcouleur];
          else contrat += couleurcarte[maconf.contratcouleur];
          std::string strDeclarant = joueurToString(maconf.declarant);
          enregistrerContratEtPli ("test", maconf.numeroDonne, contrat, strDeclarant , numpli, pliprec);
          nbcartes = 0; // noter qu'il n'y a aucune carte dans le pli en cours
          monpli = unpli(); // nouveau pli vide, initialisé
          if (waitoption) cv::waitKey(0);          
        }
        else {
          if (printoption > 0)
           std::cout<<" frame vide, aucune carte jouée du pli en cours"<<std::endl;
        }
      } // frame vide, aucune carte trouvée

      if (numpli >= 13) break; // on a décodé les 13 plis
      if (frame.empty()) break; // fin du fichier vidéo

      numeroframe++;
      if (printoption > 1) std::cout << "====== fin de frame "<< numeroframe <<" ======" << std::endl;
      if (monpli.nbcartes == 4) { // pli complet
        if (nbCartesDuPli != 4) {
          framePli = frame.clone(); // première frame avec un pli complet
          nbCartesDuPli = 4;
        }
      } else nbCartesDuPli = monpli.nbcartes;
  } // while(true) boucle sur les frames

    auto t1 = std::chrono::high_resolution_clock::now();
    duree = t1 - t0;
    std::cout << "Temps total video : " << duree.count() << " secondes" << std::endl
      << "============================"<< std::endl;

    cap.release(); // Libérer la capture vidéo
    // cv::destroyAllWindows(); // Fermer toutes les fenêtres ouvertes
    return 0;
}

int main(int argc, char **argv)
{
#ifndef _WIN32
cv::startWindowThread(); cv::waitKey(1);
#endif

  config maconf;
    std::cout << " arguments optionels :"
              << " nom du fichier image ou video, "
              << " nom du fichier de configuration, "
              << " hauteur de carte (en pixels) " << std::endl
              << std::endl;

    for(int i=0; i<sizeof(Durees); i++) Durees[i] = 0;

    std::string nomfichier;
    nomfichier = setconfig(maconf); // initialisation par défaut

    if (argc > 1)
        nomfichier = argv[1];
    std::string nomconf; // nom du fichier de configuration
    if (argc > 3) {
        maconf.hauteurcarte = std::stoi(argv[3]);
        maconf.largeurcarte = 2 * maconf.hauteurcarte / 3;
    } else {
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
            maconf.largeurcarte = 2 * maconf.hauteurcarte / 3;
        }
        else
        {
            std::cout << "Délimiteurs non trouvés" << std::endl;
        }
    }
    if (argc > 2)
        nomconf = argv[2];
    else
        nomconf = "FUNBRIDGE.txt";
    lireConfig(nomconf, maconf);
    waitoption = maconf.waitoption;
    printoption = maconf.printoption;
    threadoption = maconf.threadoption;
    if (maconf.tesOCR == 0)
        nomOCR = "SERVEUR";
    else
        nomOCR = "tesOCR";
    
    int ret = processVideo(maconf, nomfichier);
    std::cout<<" Durées de traitements "<<Durees[0]<<" , "<<Durees[1]<<" , "<<Durees[2]<<std::endl;
    std::cout<<"Appuyer sur une touche quelconque pour quitter"<<std::endl;
    cv::waitKey(0);
    return ret;
}



// trouver les lignes droites dans une image
void trouverLignes(config &maconf, cv::Mat gray, std::vector<ligne>& lignes, bool estMort){
  std::vector<cv::Vec4i> lines; // segments détectés par opencv
  int gmin = maconf.gradmin;
  int gmax = maconf.gradmax;

  cv::Mat edges;
  int iwait = 1;
  cv::Mat ima2;
  int methode = 2; // 1 : canny et HoughLines,   2: ximgproc
  methode = maconf.linesoption;
  if (methode == 2)
  {
    // Appliquer le détecteur de segments de ligne LSD

    // Paramètres du FastLineDetector : longueur minimale, écart entre lignes, etc.
    int length_threshold = maconf.nbpoints; // Longueur minimale d'une ligne
    float distance_threshold = 1.41421356f; // Distance maximale entre deux points formant une ligne
    // float distance_threshold = 1.5f; // Distance maximale entre deux points formant une ligne
    double canny_th1 = gmin;     // Seuil bas pour Canny
    double canny_th2 = gmax;     // Seuil haut pour Canny
    int canny_aperture_size = 3; // Taille de l'ouverture pour Canny
    bool do_merge = true;       //  Fusionner les lignes adjacentes ( // )

    cv::Ptr<cv::ximgproc::FastLineDetector> lsd = cv::ximgproc::createFastLineDetector(
        length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, do_merge);

    lsd->detect(gray, lines);
  } else  if (methode == 1) {
    // Utiliser la détection de contours de Canny
    // grossir l'image (désactivé)
    // canny (image, gradiant mini, gradiant maxi, ouverture)
    // gradient : variation d'intensité entre 2 pixels voisins
    // gradient mini : si le gradient calculé est inférieur, ce n'est pas un bord
    // gradiant maxi : si le gradient calculé est supérieur, c'est un bord

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
    ima2 = gray.clone();
    cv::Canny(ima2, edges, gmin, gmax, 3, false);
    cv::HoughLinesP(edges, lines, 1, theta, threshold, minlg, gap);
    if (printoption > 1)
        afficherImage("bords", edges);
    // cv::waitKey(0);
  } // methode 1

  // créer les lignes, avec équation cartésienne
  for (auto l:lines)
  {
    ligne ln;
    ln.ln = l;

    cv::Point A(l[0], l[1]);
    cv::Point B(l[2], l[3]);
    // tracer la ligne sur l'image result
    float lg = std::sqrt((B-A).x * (B-A).x + (B-A).y * (B-A).y);
    // vecteur normal (a,b) directeur (b, -a)  
    float a = -float(B.y - A.y) / lg;
    float b = float(B.x - A.x) / lg;
    float c = -a*A.x - b*A.y; // ax + by + c = 0
    ln.lg = lg;
    ln.a = a;
    ln.b = b;
    ln.c = c;
    lignes.push_back(ln);
  }

    cv::Mat grise;
    if (printoption > 1 || maconf.linesoption == 1){
      cv::cvtColor(gray, grise, cv::COLOR_GRAY2BGR); // pour affichage en rouge les lignes
      ima2 = grise.clone();
    } 
    if (printoption > 1) afficherImage("grise", grise);
    
    // Dessiner les segments de ligne détectés
    cv::Mat result;
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    if (printoption > 1){
      int ic = 0;
      for (auto ln:lignes) {
          ic++; ic %= NBCOULEURS;
          cv::Vec4i l = ln.ln;
          cv::Point A(l[0], l[1]);
          cv::Point B(l[2], l[3]);
          cv::line(result, A, B, couleurs[ic], 1);
      }
      afficherImage("ximgproc", result);
    }

    int nblignes = lignes.size();
    if (printoption > 1 || maconf.linesoption == 1){ 
      cv::Canny(ima2, edges, gmin, gmax, 3, false);
      afficherImage("bords", edges);
      // Dessiner les segments de droite et afficher leurs longueurs et extrémités
      //********************** fond noir pour ne voir que les lignes des coins
      for (int y = 0; y < ima2.rows; y++)
          for (int x = 0; x < ima2.cols; x++) ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir

      int c = 0; // indice de couleur
      float maxlg = 0;
      for (int i=0; i<lignes.size(); i++)
      {
        ligne& ligne=lignes[i];
          maxlg = std::max(maxlg, ligne.lg);
          if (printoption <= 1) continue;
          cv::Vec4i l = ligne.ln;
          cv::Point A(l[0], l[1]);
          cv::Point B(l[2], l[3]);
          cv::line(ima2, A, B, couleurs[c], 1);
          c++; c = c%NBCOULEURS;
          std::cout << "Ligne "<<i<<" " << A << "->" << B << " Longueur: " << ligne.lg << std::endl;
      }
      if (printoption > 1)
      afficherImage("Lignes toutes", ima2); // Afficher l'image avec les segments de droite
    }

    int lgmax = maconf.taillechiffre;
    // fusionner les lignes AB  et CD si // si C et D sont proches de la ligne AB
    //   et si C ou D est proche de A ou B : AB --> AC ou AD ou BC ou BD
    if (maconf.fusionoption) {
      double epsilon = 1.2; // à peine plus qu'un pixel d'écart entre les deux lignes #//
      float deltamax = 1;
      for (int k = 0; k < 5; k++) { 
        // fusionner des lignes fusionnées, de plus en plus distantes
        deltamax = k + 1;
        for (int i = 0; i < lignes.size(); i++)
        {
          ligne& ln = lignes[i];
          cv::Vec4i l = ln.ln;
          if (l[0] < 0)   continue; // ligne invalidée
          cv::Point2i A(l[0], l[1]);
          cv::Point2i B(l[2], l[3]);
          float lg1 = ln.lg;
          float a = ln.a;
          float b = ln.b;
          float c = ln.c;
          for (int j = i + 1; j < lignes.size(); j++)
          {
            // fusionner la ligne la plus courte sur la plus longue
            ligne& ln2 = lignes[j];
            cv::Vec4i ll = ln2.ln;
            if (ll[0] < 0)  continue; // ligne invalidée
            cv::Point2i C(ll[0], ll[1]);
            cv::Point2i D(ll[2], ll[3]);
            float lg2 = ln2.lg;
            if (lg1 > lg2)
            {
              // distances de C ou D à AB > epsilon à préciser --> ignorer la ligne j
              float dC = ln.dist(C); // a*C.x + b*C.y + c;
              if (abs(dC) > epsilon)   continue;
              float dD = ln.dist(D); //a*D.x + b*D.y + c;
              if (abs(dD) > epsilon)  continue;
            }
            else
            {
              float dA =ln2.dist(A); //A.x*ln2.a + A.y*ln2.b + ln2.c;
              if (abs(dA) > epsilon)  continue;
              float dB = ln2.dist(B); //B.x*ln2.a + B.y*ln2.b + ln2.c;
              if (abs(dB) > epsilon) continue;
            }
            // 4 points A B C D alignés. ignorer si l'écart entre AB et CD est important
            //
            int xmin, xmax, ymin, ymax;
            if (std::abs(A.x - B.x) > std::abs(A.y - B.y))
            {
              xmin = std::min(A.x, B.x);
              if (xmin > C.x && xmin > D.x) { // AB à droite de CD
                  if ((xmin - C.x) > deltamax && (xmin - D.x) > deltamax) continue; // segments loin
              } else {
                  xmax = std::max(A.x, B.x);
                  if (C.x - xmax > deltamax && D.x - xmax > deltamax) continue; // CD à gauche de AB
              }
            } else { // Y plus variable que X
              ymin = std::min(A.y, B.y);
              if (ymin > C.y && ymin > D.y)  { // AB sous CD
                  if ((ymin - C.y) > deltamax && (ymin - D.y) > deltamax) continue; // segments loin
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
              if (printoption > 2){
                std::cout<<" ligne "<< i << " "<<A<<B<<" --> "<<U<<V<<std::endl;
                std::cout<<" ligne "<<j<<" supprimee"<<std::endl;
                std::cout<<"verif "<<lignes[i].ln<<std::endl;
              }
              A = U;
              B = V;
              // invalider la ligne j
              lignes[j].ln[0] *= -1;
              ll[0] *= -1;
              // mettre à jour la longueur de la ligne i = AB
              lg1 = std::sqrt((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y));
              lignes[i].lg = lg1;
            }
          } // next j
        } // next i
      } //k écart suivant
    }
    // prolonger les lignes
    if (maconf.linesoption == 1)  {
      // prolonger les lignes assez longues (au moins 1/6 de la hauteur de carte)
      // essayer de prolonger chaque ligne : regarder le pixel dans le prolongement de la ligne
      // ligne AB (B à droite de A) choisir une direction x ou y selon le maximum de |dx| et |dy|
      // AB selon X , prolongement en B : regarder le pixel blanc (dans edges) à droite (B.x +1, B.y)
      //   et le pixel blanc  à droite plus haut ou plus bas (B.x +1, B.y +- 1) (le plus proche de AB)
      //   à condition que les autres pixels proche de B soient noirs (dans edge)
      // choisir le plus proche de AB, à distance de moins de 2 pixels de AB,  qui remplace B
      // même principe du coté A
      // itérer tant qu'on trouve des pixels blancs dans l'image des bords et noirs dans l'affichage des lignes
      int maxlg;
      double tolerance = 0.4; // Ajustez la tolérance selon vos besoins. 0.4 entre 45 et 60 degrés
      cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8U);
      maxlg = maconf.hauteurcarte / 6;
      maxlg *= maxlg;
      for (int i = 0; i < lignes.size(); i++)
      {
        cv::Vec4i l = lignes[i].ln;
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
              if (printoption > 2)
                  std::cout << i << " on remplace A " << A << " par " << Z << std::endl;
              A = Z;
          }
        }
        else
        {
            if (printoption > 2)
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
          if (printoption > 2)
              std::cout << i << " on remplace B " << B << " par " << Z << std::endl;
          B = Z;
        }
        else
        {
          if (printoption > 2)
              std::cout << i << "Aucun contour trouve en B." << B << std::endl;
        }
        if (printoption > 2)
        {
          afficherImage("Contour", contourImage);
          // cv::waitKey(1);
        }
        lignes[i].ln[0] = A.x;
        lignes[i].ln[1] = A.y;
        lignes[i].ln[2] = B.x;
        lignes[i].ln[3] = B.y;
        for (const auto &P : contour)
        {
            contourImage.at<uchar>(P) = 255;
        }
      }
      // cv::waitKey(0);
    }


    // TODO : éliminer les droites qui contiennent un segment dans une liste spécifique
    //        concerne les vidéos où il y a un fond commun sur la table ou entre la table et la caméra

    // invalider les lignes dont la longueur est inférieure à la taille du chiffre + symbole
    // test :éliminer les ligne de longueur inférieure à la moitié de hauteur de carte
    // éliminer les lignes plus longues que la hauteur de carte
    // modif 2025/06/11 : on conserve les lignes longues à cause du mort
    // conserver les petites lignes verticales si on analyse les cartes du mort
    lgmax = maconf.taillechiffre + maconf.taillesymbole; // limite inférieure
    int lgmin = maconf.hauteurcarte + maconf.deltacadre;
    for ( auto& ln : lignes)
    {
      cv::Vec4i l = ln.ln;
      if (l[0] < 0)  continue; // ligne déjà invalidée
      if ( ln.lg < maconf.taillechiffre || ( (!estMort || std::abs(ln.b) > 0.2) && (ln.lg < lgmax)) )
      {
        ln.ln[0] *= -1; // invalider la ligne
        if (printoption > 2){
          cv::Point2i A(l[0], l[1]);
          cv::Point2i B(l[2], l[3]);
          std::cout << "supprime la ligne " << A << "->" << B << " longueur " << ln.lg << std::endl;
        }
      }
    }

    float maxlg = 0;
    // afficher les lignes qui restent
    if (printoption > 1) {
      for (int y = 0; y < ima2.rows; y++) for (int x = 0; x < ima2.cols; x++)
          ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
      int c = 0;
      for (auto& ln : lignes)
      {
        cv::Vec4i l = ln.ln;
        if (l[0] < 0)  continue; // ligne fusionnée ou ignorée car trop courte
        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        cv::line(ima2, A, B, couleurs[c], 1);
        c++; c = c % NBCOULEURS;
        if (printoption > 2)
          std::cout << "Ligne "<< A << "->" << B << " Longueur: " << ln.lg << std::endl;
        if (ln.lg > maxlg) maxlg = ln.lg;
      }
      // Afficher l'image avec les segments de droite
      std::cout << "longueur maximale " << maxlg << std::endl;
      afficherImage("Lignes", ima2);
    }




}

void validerCoin(config& maconf, std::vector<ligne>& lignes, std::vector<uncoin>& Coins, uncoin& cn);
void validerCoin(config& maconf, std::vector<ligne>& lignes, std::vector<uncoin>& Coins, uncoin& cn){
  // comparer aux coins déjà enregistrés sommet Q lignes l1 et l2 extrémités U et V
  // ignorer si l1 n'est pas parallèle à AB (ou CD)
  // ignorer si Q est loin de P (loin de AB et de CD)
  // ignorer si l1 (QU) n'est pas de même sens que PH (ou PK)    
  // ignorer si l2 (QV) n'est pas de même sens que PK (ou PH)
  // si Q est à l'intérieur de cn ( Q et H du même coté de CD)  ignorer cn  
  //                              ( Q et K du même coté de AB)  ignorer cn
  // si Q est dans le quart intérieur du coin : actualiser ce coin : cadre= Q, estunRDV
  // si Q est dans le quart hors du coin
  //   actualiser le coin (Q)  : cadre = P, estunRDV 
  //   et ignorer  le coin cn
  //
  
  int dc = maconf.deltacadre;
  int dcRDV = std::max(2*dc, dc + maconf.deltaVDR + 1);
  cv::Point2i P = cn.l1->intersect(cn.l2); // calculer le point d'intersection des deux lignes de cn
  cn.sommet = P;
  cn.R = cn.H; cn.S = cn.K;
  cv::Point2i H(cn.H), K(cn.K);
  if (maconf.printoption > 0) std::cout<<"              validercoin " << P<<H<<K<<std::endl;
  bool eliminer(false);
  for (int i=0; i<Coins.size(); i++){
    uncoin& coin = Coins[i];
    cv::Point2i U(coin.H), V(coin.K), Q(coin.sommet);
    // Q loin des cotés de cn ?
    // en tenant compte de la ligne parasite qui borde le caractère R
    if (std::abs(cn.l1->dist(Q)) > dcRDV) continue;
    if (std::abs(cn.l2->dist(Q)) > dcRDV) continue;
    if(std::abs(cn.l1->a*coin.l2->a + cn.l1->b*coin.l2->b ) > maconf.cosOrtho) { // l1 non // l1
      if(std::abs(cn.l1->a*coin.l1->a + cn.l1->b*coin.l1->b ) > maconf.cosOrtho) continue; // l1 non // l2
      // l1 // l2  (et l2 // l1)
      U = coin.K; V = coin.H;
    }
    // dans le même sens ? PH.QU   et pour l2//l2 PK.QV
    if ((H.x - P.x)*(U.x - Q.x) + (H.y - P.y)*(U.y - Q.y) < 0) continue; //sens opposé
    if ((K.x - P.x)*(V.x - Q.x) + (K.y - P.y)*(V.y - Q.y) < 0) continue; //sens opposé

    cv::Point2i HH(coin.H), KK(coin.K);
    // Q à l'intérieur du coin cn ? PQ.PH > 0 et PQ.PK> 0
    if ((Q.x - P.x)*(H.x - P.x) + (Q.y - P.y)*(H.y - P.y) > 0
        && (Q.x - P.x)*(K.x - P.x) + (Q.y - P.y)*(K.y - P.y) > 0) {
          // Q à l'intérieur : actualiser le coin Q = coin cn, et éliminer cn
          eliminer = true;
          // c'est le cadre sauf si on a déjà un cadre meilleur
          if ( std::abs(std::abs(cn.l1->dist(Q)) - dc) < std::abs(std::abs(cn.l1->dist(coin.cadre)) - dc)
          && std::abs(std::abs(cn.l1->dist(Q)) - dc) < std::abs(std::abs(cn.l1->dist(coin.cadre)) - dc) ) {
            coin.cadre = Q;
            coin.estunRDV = true;
          }
          
          // actualiser coin (=cn)
          coin.l1 = cn.l1; coin.l2 = cn.l2;
          coin.H = coin.R = cn.H; coin.K = coin.S = cn.K;
          coin.sommet = cn.sommet;
          if (maconf.printoption)
            std::cout<<"==> actualisation coin "<< i<<Q<<HH<<KK<<" --> "
            <<coin.sommet<<coin.H<<coin.K<<"cadre"<<coin.cadre<<std::endl;
          
    } else if ((Q.x - P.x)*(H.x - P.x) + (Q.y - P.y)*(H.y - P.y) >= 0
        || (Q.x - P.x)*(K.x - P.x) + (Q.y - P.y)*(K.y - P.y) >= 0) {
          // Q à l'intérieur d'un coté, à l'extérieur de l'autre (ou sur une ligne)
          // si Q est à distance deltacadre (+- 1) d'une ligne et sur l'autre (+-1)
          //   repositionner Q à deltacadre de la ligne vers l'intérieur
          //   puis actualiser le coin avec cn 
          cv::Point2i W;
          if (std::abs(cn.l1->dist(Q)) <= 1 && std::abs(std::abs(cn.l2->dist(Q)) - dc) <=1) {
            // Q sur l1
            //   H-------P--Q     ou H------Q--P
            if (cn.l2->dist(H) * cn.l2->dist(Q) < 0 ) { // P entre H etQ
              //coin.sommet = Q; // inchangé
              if (cn.l1->dist(cn.K) > 0) {W.x = P.x + dc*cn.l1->a; W.y = P.y + dc*cn.l1->b; }
              else {W.x = P.x - dc*cn.l1->a; W.y = P.y - dc*cn.l1->b; }
            } else { // Q entre H et P
              coin.sommet = P;
              if (cn.l1->dist(cn.K) > 0) {W.x = Q.x + dc*cn.l1->a; W.y = Q.y + dc*cn.l1->b; }
              else {W.x = Q.x - dc*cn.l1->a; W.y = Q.y - dc*cn.l1->b; }
              // remplacer les lignes du coin par celles de cn
              coin.l1 = cn.l1; coin.l2 = cn.l2; coin.H = coin.R = cn.H; coin.K = coin.S = cn.K;
            }
            eliminer=true;
            coin.cadre  = W; coin.estunRDV = true;
            if (maconf.printoption)
              std::cout<<"==> actualisation coin "<<i<<Q<<HH<<KK<<" --> "
              <<coin.sommet<<coin.H<<coin.K<<"cadre"<<coin.cadre<<" RDV"<<std::endl;
          } else if (std::abs(cn.l2->dist(Q)) <= 1 && std::abs(std::abs(cn.l1->dist(Q)) - dc) <=1) {
            // Q sur l2
            // l2: K-------P--Q   ou  K-------Q--P
            if (cn.l1->dist(K) * cn.l1->dist(Q) < 0 ) { // P entre K etQ
              //coin.sommet = Q; // inchangé
              if (cn.l2->dist(cn.H) > 0) {W.x = P.x + dc*cn.l2->a; W.y = P.y + dc*cn.l2->b; }
              else {W.x = P.x - dc*cn.l2->a; W.y = P.y - dc*cn.l2->b; }
            } else { // Q entre H et P
              coin.sommet = P;
              if (cn.l2->dist(cn.K) > 0) {W.x = Q.x + dc*cn.l2->a; W.y = Q.y + dc*cn.l2->b; }
              else {W.x = Q.x - dc*cn.l2->a; W.y = Q.y - dc*cn.l2->b; }
              // remplacer les lignes du coin par celles de cn
              coin.l1 = cn.l1; coin.l2 = cn.l2; coin.H = coin.R = cn.H; coin.K = coin.S = cn.K;
            }
            eliminer=true;
            coin.cadre  = W; coin.estunRDV = true;
            if (maconf.printoption)
              std::cout<<"==> actualisation coin "<<i<<Q<<HH<<KK<<" --> "
              <<coin.sommet<<coin.H<<coin.K<<"cadre"<<coin.cadre<<" RDV"<<std::endl;
          } else {
            // Q ni dedans ni dehors
            // remplacer une ligne (1 ou 2 ? à préciser) par la ligne parallèle de cn
            //         et recalculer l'intersection
            // conserver le coin. actualiser
            // si Q et K sont du même coté de l1 : remplacer la ligne //l1 du coin par l1
            // si Q et H sont du même coté de l2 : remplacer la ligne //l2 du coin par l2
            // recalculer le sommet du coin
            ligne la= *(cn.l1), lb = *(cn.l2);
            if (cn.l1->dist(Q) * cn.l1->dist(K) > 0) {
              if (std::abs( coin.l1->a * cn.l1->a + coin.l1->b * cn.l1->b) < 0.5){ //coin.l1 orthogonale à cn.l1
                la = *(coin.l2);
                coin.l2 = cn.l1; coin.K = coin.R = H;
              }else{
                la = *(coin.l1);
                 coin.l1 = cn.l1; coin.H = coin.R = H;
              }
            }
            if (cn.l2->dist(Q) * cn.l2->dist(H) > 0) {
              if (std::abs( coin.l1->a * cn.l1->a + coin.l1->b * cn.l1->b) < 0.5){ //coin.l1 orthogonale à cn.l1
                lb = *(coin.l1);
                coin.l1 = cn.l2; coin.H= coin.R = K;
              }else{
                lb = *(coin.l2); 
                coin.l2 = cn.l2; coin.K= coin.S = K;
              }
            }
            coin.R = coin.H; coin.S = coin.K;
            coin.cadre = la.intersect(lb);
            coin.sommet = coin.l1->intersect(coin.l2);
            if (maconf.printoption)
              std::cout<<"==> actualisation coin "<<i<<Q<<HH<<KK<<" --> "
              <<coin.sommet<<cn.H<<cn.K<<"cadre"<<coin.cadre<<" RDV"<<std::endl;
            coin.estunRDV = true;
            eliminer = true;
        }
    } else {
      // Q est dans le quart hors du coin. y compris le sommet P
      // donc P est à l'intérieur du coin Q
      // donc c'est un RDV et il faut obtenir ou calculer le cadre
      // si Q est à distance  dc+-1 des deux lignes P est le cadre
      if ( std::abs(std::abs(cn.l1->dist(Q))  - dc) <= 2
        && std::abs(std::abs(cn.l2->dist(Q))  - dc) <= 2 ) {
            // sauf si on a déjà un cadre meilleur
          if ( std::abs(std::abs(cn.l1->dist(Q)) - dc) < std::abs(std::abs(cn.l1->dist(coin.cadre)) - dc)
          && std::abs(std::abs(cn.l1->dist(Q)) - dc) < std::abs(std::abs(cn.l1->dist(coin.cadre)) - dc) ) {
            coin.cadre = P; coin.estunRDV = true;
          }

      } else {
        // calculer la position du cadre à deltacadre des deux cotés du coin Q
        // partir de W=Q, déplacer de deltacadre sur coin.l1 (= selon la normale à coin.l2) en direction de H
        //  puis déplacer de deltacadre, selon la normale de coin.l1 en direction de K
        cv::Point2i W(Q);
        if (coin.l2->dist(coin.H) > 0) {W.x += dc*coin.l2->a; W.y += dc*coin.l2->b; }
        else {W.x -= dc*coin.l2->a; W.y -= dc*coin.l2->b; }
        if (coin.l1->dist(coin.K) > 0) {W.x += dc*coin.l1->a; W.y += dc*coin.l1->b; }
        else {W.x -= dc*coin.l1->a; W.y -= dc*coin.l1->b; }
        coin.cadre = W; coin.estunRDV = true;
      }
      if (maconf.printoption)
        std::cout<<"==> actualisation coin "<<i<<coin.sommet<<coin.H<<coin.K
          <<" --> cadre"<<coin.cadre<<" RDV"<<std::endl;
      eliminer = true;
    }
  }
  if (!eliminer) { 
    // ne pas créer un coin avec un coté trop court
    if (std::abs(cn.l1->dist(cn.K) ) < 2*maconf.deltacoin) return;
    if (std::abs(cn.l2->dist(cn.H) ) < 2*maconf.deltacoin) return;
    if (maconf.printoption) std::cout<<"==> ajout coin "<<Coins.size()<<P<<H<<K<<std::endl;
    Coins.push_back(cn);
  }
}

void trouverCoins(config& maconf, std::vector<ligne>& lignes, std::vector<uncoin>& Coins){
  int printoption = maconf.printoption;
  int nbcoins = 0;
  int nbcartes = 0;
  // pour chaque ligne AB
  // rechercher les lignes CD orthogonales à AB
  // ignorer CD trop courte (2*deltacoin)
  // ignorer les lignes CD non orthogonales à AB
  // ignorer les lignes CD loins de AB (A et B loins du même coté de CD)
  //    ou (C et D loins de AB du même coté)
  // pour une ligne CD :
  //    proche d'une extrémité M = A ou B (N = B ou A)
  //      U= C ou D  (V = D ou C) proche de AB : préparer le coin cn sur AB et CD avec H=N K=V P=ABxCD
  //    sinon, si U=C ou D proche de AB :
  //           préparer un coin sur AB et CD avec H=A K=V P=ABxCD
  //           préparer un coin              avec H=B 
  //    sinon (donc C et D loins séparés par AB):
  //           préparer 4 coins avec H=A ou H=B et K=U ou K=V
  //
  // valider chaque coin calculé
  //
  int dc = maconf.deltacadre;
  int dcoin = maconf.deltacoin;
  uncoin cn;
  for (int i = 0; i < lignes.size(); i++) { // ligne AB
    ligne& ln = lignes[i];
    cv::Vec4i l1 = ln.ln;
    if (l1[0] < 0)   continue; // ligne fusionnée ou effacée
    if(ln.lg < 2*maconf.deltacoin) continue; // trop courte
    cv::Point2i A(l1[0], l1[1]);
    cv::Point2i B(l1[2], l1[3]);
    cn.l1 = &ln;
    for (int j = i+1; j < lignes.size(); j++) { // ligne CD
      ligne& ln2 = lignes[j];
      cv::Vec4i l2 = ln2.ln;
      if (l2[0] < 0)   continue; // ligne fusionnée ou effacée
      if(ln.lg < 2*maconf.deltacoin) continue; // trop courte
      if(std::abs(ln.a*ln2.a + ln.b*ln2.b) > maconf.cosOrtho) continue; // pas un angle droit
      if (ln2.dist(A) > dcoin && ln2.dist(B) > dcoin) continue; // AB loin de CD, coté positif
      if (ln2.dist(A) < -dcoin && ln2.dist(B) < -dcoin) continue; // AB loin de CD coté négatif

      cv::Point2i C(l2[0], l2[1]);
      cv::Point2i D(l2[2], l2[3]);
      if (ln.dist(C) > dcoin && ln.dist(D) > dcoin) continue; // CD loin de AB, coté positif
      if (ln.dist(C) < -dcoin && ln.dist(D) < -dcoin) continue; // CD loin de AB coté négatif

      cv::Point2i M(A), N(B); // M proche de CD, N loin
      cv::Point2i U(C), V(D); // U proche de AB, V loin 
      if (std::abs(ln2.dist(A)) > std::abs(ln2.dist(B))) { M=B; N=A; }

      if (std::abs(ln2.dist(M)) < dcoin) { // M proche de la droite CD. proche de C ou D ?
        if (printoption > 0) std::cout<<"                lignes "<<i<<" "<<j<<std::endl;
        if (std::abs(ln.dist(C)) > std::abs(ln.dist(D))) { U=D; V=C; }
        if (std::abs(ln.dist(U)) <= dcoin) { // U proche de AB
          // préparer un nouveau coin
          cn.H = N; cn.K=V; cn.l1= &ln; cn.l2=&ln2;
          validerCoin (maconf, lignes, Coins, cn);
        } else {
          // C et D séparés par la droite AB
          // deux coins : un du coté C et un du coté D
          cn.H = N; cn.K=C; cn.l2=&ln2;
          validerCoin (maconf, lignes, Coins, cn);
          cn.K = D;
          validerCoin (maconf, lignes, Coins, cn);
        }
      } else { // A et B loin de CD (déjà éliminé si du même coté)
          if (printoption > 0) std::cout<<"                lignes "<<i<<" "<<j<<std::endl;
        // ligne CD au milieu du segment AB
        // C ou D proche de AB ?
        U=C; V=D;
        if (std::abs(ln.dist(C)) >  dcoin) {U=D; V=C;}

        if (std::abs(ln.dist(U)) < dcoin) { // U proche de AB. donc deux coins 
          cn.l2=&ln2; cn.H=A; cn.K=V;
          validerCoin (maconf, lignes, Coins, cn);
          cn.H=B;
          validerCoin (maconf, lignes, Coins, cn);
        } else { // CD coupe AB au milieu. donc 4 coins;
          cn.l2=&ln2; cn.H=A; cn.K=C;
          validerCoin (maconf, lignes, Coins, cn);
          cn.K=D;
          validerCoin (maconf, lignes, Coins, cn);
          cn.H=B;
          validerCoin (maconf, lignes, Coins, cn);
          cn.K=C;
          validerCoin (maconf, lignes, Coins, cn);
        }
      }
    } // CD
  } // AB
}


// TODO : comparer l'image à l'image précédente, si on traite une vidéo
//    après le traitement d'une frame, conserver le résultat du décodage
//     qui se trouve dans le tableau des coins
//    traitement de la nouvelle frame:
//    comparer à la frame précédente. on obtient les pixels modifiés
//    invalider les résultats de chaque coin sur une zone modifiée
//    restreindre l'image à analyser à la partie modifiée  

int processFrame(config &maconf, cv::Mat image, bool estvideo,
   std::vector<unecartePrec>& cartesPrec, unpli &monpli)
{
    std::chrono::duration<double> duree;
    activeThreads = 0;
    if (maconf.threadoption > 1)
        MAX_THREADS = maconf.threadoption;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::string> resultats; // vecteur des résultats
    std::vector<std::thread> threads;

    int c = 0;
    std::vector<ligne> lignes;   // segments complétés par l'équation de droite
    std::vector<uncoin> Coins;   // coins entre lignes orthogonales
    std::vector<unecarte> cartes; // cartes 

    std::string cards[50]; // cartes trouvées
    int nbcards;

    if (image.empty())
    {
        std::cerr << "Erreur de chargement de l'image" << std::endl;
        return -1;
    }
    cv::Mat result = image.clone();
    if (printoption > 1)  afficherImage("couleur", image); // afficher l'image en couleurs
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); // Convertir en niveaux de gris
    // obtenir les lignes droites dans l'image monochrome

    trouverLignes(maconf, gray, lignes);


    auto t11 = std::chrono::high_resolution_clock::now();
    duree = t11 - t0;
    Durees[0] += duree.count();
    auto t22 = t11;
    if(printoption > 1)
      std::cout << "Duree de detection des lignes : " << duree.count() << " secondes" << std::endl;
    if (waitoption > 1) cv::waitKey(0);

    //////////////////////////////// rechercher les coins des cartes ///////////////////
    //

    int nbcartes = 0;
    trouverCoins(maconf, lignes, Coins);

    auto t33 = std::chrono::high_resolution_clock::now();
    duree = t33 - t22;
    if (printoption > 1) std::cout << "Duree d'identification des coins : " << duree.count() << " secondes" << std::endl;
    Durees[1] += duree.count();
    ////////////// on a déterminé les coins //////////////////////
    if (printoption > 1) {
      int i = 0;
      for (auto &moncoin : Coins) std::cout <<"coin "<<i++ << moncoin.sommet<<std::endl;
    }

    // déterminer la taille des cartes, proche de la taille indiquée dans la configuration
    // déterminer les probables bords de carte
    // deux coins sur une même ligne (ou deux ligne // proches), à distance vraissemblable (paramètre général de configuration)
    // la plus grande distance serait la hauteur de carte, sauf si plusieurs cartes sont alignées
    // une des autres devrait être dans le rapport des cotés de carte ( 3 / 2 )
    // 

    float epsilon = std::max(1,maconf.deltacadre / 2);
    int htmax = 0; // hauteur maximale de carte, proche de la valeur dans la configuration
    int lamax = 0; // largeur maximale ....
    int ecartHt = maconf.hauteurcarte;
    int ecartLa = maconf.hauteurcarte;
    cv::Point2i P1, P2;
    for (int n = 0; n < Coins.size(); n++){
      const auto& cn = Coins[n];
      if (cn.elimine) continue;

      cv::Vec4i l1 = cn.l1->ln;
      cv::Vec4i l2 = cn.l2->ln;
      cv::Point2i A = cn.sommet;
      cv::Point2i H, K; // extremités non communes sur les deux lignes : coin AH,AK
      H = cn.H;
      K = cn.K;
      // rechercher les coins opposés de la carte du coin n
      for (int m = n + 1; m < Coins.size() ; m++) {
        const auto& cm = Coins[m];
        if (cm.elimine) continue; // coin éliminé
        cv::Vec4i l11 = cm.l1->ln;
        cv::Vec4i l22 = cm.l2->ln;
        cv::Point2i B(cm.sommet);
        cv::Point2i HH(cm.H);
        cv::Point2i KK(cm.K);
        // une des lignes commune avec une de l'autre coin?
        // le coin B doit être sur une des lignes du coin A
        // le coin A doit être sur une des lignes du coin B
        // les deux autres lignes doivent être // et de même sens
        // AB semble alors etre un bord de carte
        //
        bool estoppose = false;
        // première ligne du coin n contient le sommet B du coin m ?
        float dist = cn.l1->dist(B);
        if (std::abs(dist) < epsilon) {
          // B proche de la première ligne du coin A
          // ligne de B // l1 ? produit vectoriel des normales
          float pv = cn.l1->a * cm.l1->b  - cn.l1->b * cm.l1->a;
          if (std::abs(pv) < maconf.deltaradian ) {
            // l1(n) et l1(m) parallèles
            // de sens opposé ? produit scalaire AH.BHH
            float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
            if (ps < 0) { // A et B opposés
              // vérifier que K et KK sont du même coté de la droite l1
              float d1 = cn.l1->dist(K);
              float d2 = cn.l1->dist(KK);
              if (d1*d2 > 0){ //cotés non communs de même orientation
                  estoppose = true;
              } 
            }
          } else { // l1(n) et l1(m) non //
            // l1(n) et l2(m) confondus ?
            float pv = cn.l1->a * cm.l2->b  - cn.l1->b * cm.l2->a;
            if (std::abs(pv) < maconf.deltaradian ) {
              // l1 et l2 confondus
              // de sens opposé ? produit scalaire AH.BKK
              float ps = (H.x - A.x)*(KK.x - B.x) + (H.y - A.y)*(KK.y - B.y);
              if (ps < 0) { // A et B opposés
                // vérifier que K et HH sont du même coté de la droite l1 (= l2 de m)
                float d1 = cn.l1->dist(K);
                float d2 = cn.l1->dist(HH);
                if (d1*d2 > 0){ //cotés non communs de même orientation
                    estoppose = true;
                } 
              }
            }
          }
        } else // B pas proche de la ligne i. proche de la ligne j ?
        if (std::abs(B.x * cn.l2->a + B.y + cn.l2->b + cn.l2->c < epsilon)) { 
         // B proche de la droite l2
          // ligne l1 ou l2 (m) // l2 (n) ? produit vectoriel des normales
          float pv = cn.l2->a * cm.l1->b  - cn.l2->b * cm.l1->a;
          if (std::abs(pv) < maconf.deltaradian ) {
            // l2(n) et l1(m) confondus
            // de sens opposé ? produit scalaire AK.BHH
            float ps = (K.x - A.x)*(HH.x - B.x) + (K.y - A.y)*(HH.y - B.y);
            if (ps < 0) { // A et B opposés
              // vérifier que H et KK sont du même coté de la droite j = ii
              float d1 = cn.l2->dist(H);
              float d2 = cn.l2->dist(KK);
              if (d1*d2 > 0){ //cotés non communs de même orientation
                estoppose = true;
              } 
            }
          } else { // l2(n) et l1(m) non //
            // l2(n) et l2(m) confondus ?
            float pv = cn.l2->a * cm.l2->a  - cn.l2->b * cm.l2->b;
            if (std::abs(pv) < maconf.deltaradian ) {
              // de sens opposé ? produit scalaire AH.BH
              float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
              if (ps < 0) { // A et B opposés
                // vérifier que H et HH sont du même coté de la droite j = jj
                float d1 = cn.l2->dist(H);
                float d2 = cn.l2->dist(HH);
                if (d1*d2 > 0){ //cotés non communs de même orientation
                    estoppose = true;
                } 
              }
            }
          }
        }
        // déterminer précisément la hauteur de carte, proche de la valeur dans la configuration
        if (estoppose) { // coins n et m sur un bord de carte (probablement)
          float lg = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
          lg = std::sqrt(lg);
          // AB proche de la hauteur de carte ?
          int dl = std::abs(lg - maconf.hauteurcarte);
          if (dl < maconf.deltacadre) {
            if (dl < ecartHt) {
              ecartHt = dl;
              htmax = lg;
              P1 = A; P2 = B;
            }
          }
          else {
            // AB proche de la largeur de carte ?
            int dl = std::abs(lg - maconf.largeurcarte);
            if (dl < maconf.deltacadre) {
              if (dl < ecartLa) {
                ecartLa = dl;
                lamax = lg;
                //P1 = A; P2 = B;
              }
            }
          }
          continue; // inutile
        }
      } // for m
      if (htmax < 8 * maconf.hauteurcarte / 10) htmax = maconf.hauteurcarte;
    }
    /****************** ne fonctionne pas !!!!!!!!!!!!!!! 
    // recalculer les paramètres de position sur la carte
    if (htmax != 0) {
      maconf.hauteurcarte = htmax;
      maconf.largeurcarte = 2*htmax / 3;
    } else if (lamax != 0) {
      maconf.largeurcarte = lamax;
      maconf.hauteurcarte = 3*lamax / 2;
    }
    else {
        if (printoption > 0) {
            std::cout << " !!!!! impossible d'estimer la taille des cartes" << std::endl;
            std::cout << " !!!!! poursuite avec la configuration " << std::endl;
        }
    }
    **************/
    //
     if (printoption > 1)
        std::cout << "hauteur carte : " << maconf.hauteurcarte << std::endl;

    // TODO : pour chaque coin, rechercher les deux coins adjacents de la carte.
    //        créer les coins adjacents des lignes, même si une des deux est courte, correspondent

    ////////////////////////// éliminer les artefacts /////////////////////////////

    // faire le tri parmi les coins détectés
    // pour chaque couple de coins P (n) et Q (m)
    //    éliminer le coin contenu dans l'autre, proche et //


    bool bwait = false;
    if (bwait) cv::waitKey(0);

    c = 0;
    for (int n = 0; n < Coins.size(); n++) {
      auto& cn = Coins[n];
      if (cn.elimine) continue;
      cv::Vec4i l1 = cn.l1->ln;
      cv::Vec4i l2 = cn.l2->ln;
      cv::Point2i P(cn.sommet);
      cv::Point2i R(cn.H);
      cv::Point2i S(cn.K);

      if (printoption > 1){
        std::cout << "Coin " << n << " " << P << " , " << R << " , " << S << std::endl;
        if (cn.numCarte > 0 ) std::cout<<" --> carte numero "<<cn.numCarte<<std::endl;
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
      // comparer aux coins suivants
      for (int m = n + 1; m < Coins.size(); m++) {
        auto& cm = Coins[m];
        cv::Point2i Q = cm.sommet;
        cv::Vec4i l11 = cm.l1->ln;
        cv::Vec4i l22 = cm.l2->ln;

        cv::Point2i U = cm.H;
        cv::Point2i V = cm.K;

        // coin  UQV

        cv::Point2i AA(l11[0], l11[1]);
        cv::Point2i BB(l11[2], l11[3]);
        cv::Point2i CC(l22[0], l22[1]);
        cv::Point2i DD(l22[2], l22[3]);
        // ignorer ce coin Q s'il n'est pas // coin P
        //
        double pv;
        bool estl22 = false;
        if (cn.l1 == cm.l1)
            pv = 0;
        else
            pv = cn.l1->a * cm.l1->b - cn.l1->b * cm.l1->a;
            // produit vectoriel des normales des lignes l1 des deux coins
        if (std::abs(pv) > maconf.deltaradian)
        { // AB  non // A'B'
          if (cn.l1 == cm.l2)
              pv = 0;
          else
              pv = cn.l1->a * cm.l2->b - cn.l1->b * cm.l2->a;
          if (std::abs(pv) > maconf.deltaradian)  continue;  //  AB  non // C'D'
          estl22 = true; // AB // C'D'   et donc  CD // A'B' (orthogonaux)
        }
        //       déterminer si Q est proche d'une des lignes l1 (AB) ou l2 (CD)
        //       puis calculer la distance de Q à l'autre ligne
        bool memecarte = false;

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (cn.elimine && cm.elimine) continue; // deux coins éliminés
        // il faut continuer pour déterminer si l'un est le cadre de l'autre
        {
          float d1, d2, d3, d11, d22;
          int dc = std::max(5, maconf.deltacadre);
          int dc2 = 2*maconf.deltacadre;
          d1 = cn.l1->dist(Q);
          d2 = cn.l2->dist(Q);

          // distance de P à A'B' (=QU)  ou C'D' (=QV)
          d11 = cm.l1->dist(P);
          d22 = cm.l2->dist(P);
          float lgPQ;
          bool bienoriente = true;
          if (std::abs(d1) > dc2 && std::abs(d2) > dc2 ) continue; // Q n'est pas proche d'un coté de P
          if (std::abs(d1) > dc ){ // donc Q proche du coté CD= PS du coin n (P)
            // vérifier que PS et (QU ou QV) orientés en sens inverse
            // et que PR et (QV ou QU) ont meme orientation
            // et PS et PQ ont meme orientation   PS.PQ > 0
            if ((S.x - P.x)*(Q.x - P.x) + (S.y - P.y)*(Q.y - P.y) < 0 ) bienoriente = false;
            else {
              if (std::abs(d11) < std::abs(d22)) { // P proche de A'B' = QU
                // PS et QU orientés de sens contraire ? : PS.QU < 0    et PR.QV > 0 et PS.PQ > 0
                if ((S.x - P.x)*(U.x - Q.x) + (S.y - P.y)*(U.y - Q.y) > 0 ) bienoriente = false;
                if ((R.x - P.x)*(V.x - Q.x) + (R.y - P.y)*(V.y - Q.y) < 0 ) bienoriente = false;
              } else { // P proche de C'D' = QV
                if ((S.x - P.x)*(V.x - Q.x) + (S.y - P.y)*(V.y - Q.y) > 0 ) bienoriente = false;
                if ((R.x - P.x)*(U.x - Q.x) + (R.y - P.y)*(U.y - Q.y) < 0 ) bienoriente = false;
              }
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
          } else  { // Q proche de AB = PR
            // vérifier que PR et (QU ou QV) orientés en sens inverse
            // et que PS et (QV ou QU) ont meme orientation
            if ((R.x - P.x)*(Q.x - P.x) + (R.y - P.y)*(Q.y - P.y) < 0 ) bienoriente = false;
            else {
              if (std::abs(d11) < std::abs(d22)) { // P proche de A'B' = QU
                // PR et QU orientés de sens contraire ? : PR.QU < 0    et PS.QV > 0
                if ((R.x - P.x)*(U.x - Q.x) + (R.y - P.y)*(U.y - Q.y) > 0 ) bienoriente = false;
                if ((S.x - P.x)*(V.x - Q.x) + (S.y - P.y)*(V.y - Q.y) < 0 ) bienoriente = false;
              } else { // P proche de C'D' = QV
                if ((R.x - P.x)*(V.x - Q.x) + (R.y - P.y)*(V.y - Q.y) > 0 ) bienoriente = false;
                if ((S.x - P.x)*(U.x - Q.x) + (S.y - P.y)*(U.y - Q.y) < 0 ) bienoriente = false;
              }
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
            if (printoption > 1) std::cout << " coin "<< m << Q<< " opposé au coin "<< n 
             << P << " ecart "<< d3<<std::endl;
            // indiquer aussi si le premier coté du coin m est long ou court
            // rechercher si P est proche de A'B' ou de C'D'
            float dist = cm.l1->dist(P);
            if (std::abs(dist) < dc2) { // P proche de A'B'
              // noter que le premier coté du coin Q est la longueur ou la largeur
              /**************************** ADAPTER 
              if (lgPQ > 5*maconf.hauteurcarte/6) { // coté long
                coins[m][10] = -3; 
              } else coins[m][10] = -2; // coté largeur
              ***********************/
            } else {
              dist = cm.l2->dist(P);
              if (std::abs(dist) < dc2) { // P proche de C'D'
                /***************************ADAPTER
                if (lgPQ > 5*maconf.hauteurcarte/6) // coté largeur pour le premier coté du coin
                    coins[m][10] = -2; 
                else coins[m][10] = -3; // coté longeur pour le premier coté
                **************************/
              }
            }

            // si le coin m est déjà associé à un coin (<n) le coin n appartient à la même carte
            if (cm.numCarte != 0) {
              if (cn.numCarte != cm.numCarte) {
                unecarte& uc = cartes[cm.numCarte -1];
                uc.coins.push_back(&cn);
                cn.numCarte = cm.numCarte;
              }
            } else if (cn.numCarte != 0) {
              unecarte& uc = cartes[cn.numCarte -1];
              uc.coins.push_back(&cm);
              cm.numCarte = cn.numCarte;
            } else {
              unecarte uc;
              nbcartes = cartes.size() + 1;
              cn.numCarte = cm.numCarte = nbcartes;
              uc.coins.push_back(&cn);
              uc.coins.push_back(&cm);
              cartes.push_back(uc);
            }
            if (printoption > 1) std::cout<<" --> carte numero "<< cn.numCarte<<std::endl;
          } else { // PQ n'est pas un bord de carte
            lgPQ = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
            if (lgPQ > 3*maconf.deltacadre) continue;
          }

        }
        continue;
      } // for m

      // élimination différée de P ?
      if (eliminerP)
      { // c'est peut-être déjà fait
        if (!cn.elimine)
        {
          if (printoption > 2)
              std::cout << "elimination coin " << n << std::endl;
          cn.elimine = true;
        }
      } else if (cn.numCarte == 0) { // pas encore affecté à une carte
        unecarte uc;
        uc.coins.push_back(&cn);
        cartes.push_back(uc);
        nbcartes = cartes.size();
        cn.numCarte = nbcartes; // nouvelle carte
        if (printoption > 1) std::cout<<" --> nouvelle carte "<<nbcartes<<" pour le seul coin "<<n<<std::endl;
      }
      c++;
      c--; // pour pouvoir mettre un point d'arrêt
    } // for n

    // on a obtenu tous les coins et les cartes.
    // certains coins sont identifiés comme personnages (R D V) car contenant un cadre

    if (estvideo){
    // si on traite une video,
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
    // il sera inutile de traiter ces coins, même si la carte est un personnage

      traiterCartes(image, maconf,cartes, Coins, cartesPrec, lignes, monpli);
      // TODO : éliminer chaque carte de la frame précédente dont aucun coin n'est dans celle-ci
      // éliminer les coins des cartes de la frame précédente qui ne sont pas dans une carte de celle-ci
      int dc = std::max(6, 2*maconf.deltacadre);
      for (auto it=cartesPrec.begin() ; it != cartesPrec.end(); ){
        bool trouve = false;
        unecartePrec& ucp = *it;
        std::string couleurinconnue("??");
        std::string couleurcarte = couleurinconnue;
        if (ucp.couleur >= 0 && ucp.couleur < 4) couleurcarte = nomcouleur[ucp.couleur];
        for (auto& up : ucp.coinsPrec) {
          cv::Point2i Q (up.x, up.y);
          for (auto& carte : cartes) {
            for (auto cn : carte.coins) {
              if (cn->elimine) continue; // coin éliminé
              cv::Point2i P (cn->sommet);
              // proche ?
              if (std::abs(P.x - Q.x) <= dc && std::abs(P.y - Q.y) <= dc ) {
                trouve = true;
                // récupérer la couleur et valeur de carte
                if (cn->couleur < 0 && up.couleur >= 0){
                  cn->couleur = up.couleur;
                  cn->valeur = up.valeur;
                }
                break;
              }
              if (trouve) break;
            }
          }
        }
        if (!trouve) { // carte précédente non trouvée dans la frame analysée
          // supprimer les coins correspondants (et de coinsPrec)
          for (auto& up : ucp.coinsPrec) {
            if (printoption > 1 ) std::cout<<" retrait coin ("
            <<up.x<<","<<up.y<<") "<<up.couleur<< " "<<up.valeur<<std::endl;
            // retirer aussi le coin précédent (up) des coins précédents
          }
          // supprimer la carte précédente de cartesPrec 
          if (printoption > 1 ) 
            std::cout<<" retrait carte précédente "<< couleurcarte <<" "<<ucp.valeur <<std::endl;
          it = cartesPrec.erase(it);
        } else it++;
      } 
    }

    // afficher les coins
  if (printoption > 1) {
    // afficher ce qui reste selectionné
    cv::Mat imaC = image.clone();
    //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < imaC.rows; y++)
      for (int x = 0; x < imaC.cols; x++)
          imaC.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir

    c = 0;
    for (int n = 0; n < Coins.size(); n++) {
      auto& cn = Coins[n];
      int cc = cn.numCarte; // numéro de carte
      while (cc >= NBCOULEURS) cc -= NBCOULEURS;
      cv::Point P(cn.sommet);

      cv::Vec4i l1 = cn.l1->ln;
      cv::Vec4i l2 = cn.l2->ln;

      cv::Point2i A(l1[0], l1[1]);
      cv::Point2i B(l1[2], l1[3]);
      cv::Point2i C(l2[0], l2[1]);
      cv::Point2i D(l2[2], l2[3]);
      cv::Vec4i nl1(A.x, A.y, B.x, B.y);
      cv::Vec4i nl2(C.x, C.y, D.x, D.y);

      // !!!! uniquement sur les copies

      if (estvideo){
        int proche = maconf.deltacoin; // valeur à préciser
        bool trouvePrec(false);
        // si ce coin était trouvé dans la frame précédente, inutile de le considérer
        if (cn.couleur >= 0 && cn.valeur > 0) {
          if (printoption > 1) std::cout<<"coin "<< n <<cn.sommet<< " identifié: "
                <<nomcouleur[cn.couleur]<<" "<<cn.valeur<<std::endl;
          //cn.elimine = true;
        } else for (auto ucp : cartesPrec){
          for (const auto& up : ucp.coinsPrec){
            if (std::abs (P.x - up.x) <= proche 
            && std::abs (P.y - up.y) <= proche ) {
              // déjà trouvé dans la précédente frame
              cn.couleur = ucp.couleur;
              cn.valeur = ucp.valeur;
              cn.elimine = true;
              if (printoption > 1){
                std::string s;
                if (ucp.couleur >=0 && ucp.couleur <=3 && ucp.valeur >= 0 && ucp.valeur <=13) {
                  s = couleurcarte[ucp.couleur];
                }
                std::cout<<"coin "<< n << " dans une frame précédente :"
                <<couleurcarte[ucp.couleur] <<" "<<ucp.valeur<<std::endl;
              }
              trouvePrec = true;
              break;
            }
          }
          if (trouvePrec) break;
        }
      }

      if (cn.elimine ) { // coin éliminé précédemment
        cv::circle(imaC, P, 2, cv::Scalar(255, 255, 255), -2); //  cercle blanc au sommet du coin
        //cv::circle(grise, P, 2, cv::Scalar(0, 0, 255), -2);    //  cercle rouge au sommet du coin
        // si ce coin ressemble à un cadre, afficher les lignes en trait fin gris
        cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), cv::Scalar(128, 128, 128), 1); // petit trait
        cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), cv::Scalar(128, 128, 128), 1); // petit trait
        continue;                                                                                           // coin éliminé
      }

      // TODO : pour chaque coté, rechercher une ligne // vers l'extérieur à distance deltacadre
      //        rechercher une ligne // à l'intérieur à 1 pixel

      cv::line(imaC, cv::Point(nl1[0], nl1[1]), cv::Point(nl1[2], nl1[3]), couleurs[cc], 1); // petit trait
      cv::line(imaC, cv::Point(nl2[0], nl2[1]), cv::Point(nl2[2], nl2[3]), couleurs[cc], 1); // petit trait
      if (cn.estunRDV){
        cv::circle(imaC, P, 2, couleurs[cc], 3); //  cercle au sommet du coin
        cv::circle(imaC, cn.cadre, 1, couleurs[cc], 1); //  point sur le cadre
      }
      else
      {
        cv::circle(imaC, P, 3, couleurs[cc], 1); //  cercle épais (RDV) au sommet du coin
        //cv::circle(grise, P, 3, couleurs[cc], 1);
      }
      // afficher le numéro du coin
      std::string texte = std::to_string(n);
      cv::putText(imaC, texte, P, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  couleurs[cc], 1);
      c = (c+1)% NBCOULEURS;
    } // for n

    if (htmax > 4 * maconf.hauteurcarte / 5)
    {
      std::cout << "probable hauteur de carte : " << htmax << std::endl;
      cv::circle(imaC, P1, 6, cv::Scalar(0, 128, 128), 4);
      cv::circle(imaC, P2, 6, cv::Scalar(0, 128, 128), 4);
    }
    afficherImage("coins", imaC);
  } //if(printoption > 0)

  bool estunRDV(false); // le coin contient-il un cadre ?
  cv::Point2i Q;          // point du cadre

  auto t1 = std::chrono::high_resolution_clock::now();
  duree = t1 - t33;
  if (printoption > 1) std::cout << "Temps préparatoire : " << duree.count() << " secondes" << std::endl
    << "============================"<< std::endl;
  Durees[2] += duree.count();
  result = image.clone();
  if (printoption > 0) afficherImage("result", result);
  int szPrec = 0;


// TODO : vérifier que l'on obtient le même résultat pour tous les coins d'une même carte
//        a posteriori après traitement multithread

  if (!estvideo ||  maconf.coinsoption > 0) {    // traiter aussi les coins isolés
    int numcarte = 1;
    //while(numcarte != 0){
      //bool plusdecarte = true;
      //int coul = -1; int val = 0; // couleur et valeur de la carte
      for (int n = 0; n < Coins.size(); n++) {
        const auto& cn = Coins[n];
        //if(cn.numCarte != numcarte) continue;
        //plusdecarte = false;
        if (cn.elimine || (estvideo && cn.couleur >= 0 && cn.valeur> 0) )
            continue; // coin éliminé ou déjà analysé dans une carte
        int l1W[4], l2W[4];
        // TODO : éliminer le coin s'il est sur un bord d'une carte déjà analysée
        cv::Vec4i l1 = cn.l1->ln;
        cv::Vec4i l2 = cn.l2->ln;
        for (int i = 0; i < 4; i++)
        {
          l1W[i] = l1[i];
          l2W[i] = l2[i];
        }
        if (printoption > 1)
            std::cout << std::endl
                      << "coin " << n << "   ";
        std::string cartelue;

        if (threadoption == 0) { // pas de sous-tache
          if (cn.couleur < 0 || cn.valeur <=0) // valeur non encore déterminée
              traiterCoin(n, Coins, image, resultats,
                      result, &l1W[0], &l2W[0], maconf);
          if (cn.valeur != 0 && cn.couleur >= 0) // valeur trouvée
          {
            //if (coul < 0) {coul = cn.couleur; val =cn.valeur;}
            //else if (coul != cn.couleur || val != cn.valeur) {
            //  std::cout<<"!! carte "<< numcarte<<" coin "<<n<< " couleur ou valeur incohérentes "<<std::endl;
            //}
            if (!estvideo) afficherImage("result", result);
            cv::Point2i PT(cn.sommet);
            std::string resW = couleurcarte[cn.couleur];
            resW += valeurcarte[cn.valeur];
            std::string res = resW + "#";

            afficherResultat(result, PT, res);
            if (waitoption > 1)
                cv::waitKey(0);
            else cv::waitKey(1);
          }
        }
        else // sous-taches
        if (cn.couleur < 0 || cn.valeur <=0) // valeur non encore déterminée
        { // démarrer une sous-tache
          if (threadoption == 1) MAX_THREADS = std::thread::hardware_concurrency(); // Limite du nombre de sous-tâches actives
          else MAX_THREADS = threadoption;
          if (MAX_THREADS == 0) MAX_THREADS = 8;
          std::unique_lock<std::mutex> lock(mtx);
          // std::cout << "Avant attente cvar..." << std::endl;
          cvar.wait(lock, []
                    { return activeThreads < MAX_THREADS; });
          // std::cout << "Débloqué !" << std::endl;

          ++activeThreads;
          threads.emplace_back([n, &Coins, image, &resultats, result, l1W, l2W, maconf]()
                                { traiterCoin(n, std::ref(Coins), image, std::ref(resultats), result, l1W, l2W, maconf); });

          // std::cout<< activeThreads<< " theads actives "<< " coin "<<n <<std::endl;
          // threads.emplace_back(traiterCoin, n, coins, std::ref(image),
          //     std::ref(resultats), std::ref(result), l1W, l2W, std::ref(maconf));
        }
      } // boucle sur les coins
      //if (plusdecarte) numcarte = 0;
      //numcarte++;
    //} //boucle sur les cartes

    if (threadoption > 0) {
      // Attente de toutes les sous-tâches
      for (auto &t : threads)
      {
          t.join();
      }
    }
  }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    if (printoption > 1) std::cout << "Temps écoulé : " << elapsed.count() << " secondes" << std::endl;
    Durees[3] += elapsed.count();

    // si on traite une vidéo, ajouter les  coins de chaque carte 
    //   aux coins des frames précédentes (du pli en cours depuis une frame vide de cartes)
    int numcarte = cartesPrec.size(); // numéro de carte dans les frames précédentes
    //if (estvideo) for (const auto& up : coinsPrec) {numcarte = std::max(numcarte, up.numcarte);}

    int dc = std::max(4, maconf.deltacadre/2); // tolérance d'égalité
    bool coinNouveau(false);
    for (int ic = 0; ic < cartes.size(); ic++) {  // les cartes de cette frame
      int nc = ic+1; // cartes numérotées à partir de 1
      bool nouveaucoin = false;
      int cc1(-1), vc1(0); // couleur et valeur de carte
      int ncp(-1); // numéro de carte dans la frame précédente
      bool trouve(false);

      unecartePrec ucp0;
      unecartePrec& carteP = ucp0;
      for (const auto coin : cartes[ic].coins){ // les coins de cette carte de cette frame

        cv::Point2i PT(coin->sommet);
        if (!estvideo) { // on ne traite pas une video
          // Affichage des résultats après synchronisation multithread
          // les résultats sont dans le tableau des coins 
          // afficher un résultat pour chaque carte
          bool premier(true);
          if (coin->elimine) continue; //coin éliminé
          if (coin->valeur == 0) continue; // valeur de carte non trouvée
      
          // pas vidéo et coin non éliminé et valeur de carte trouvée et carte en cours
          if (!coin->elimine && coin->valeur > 0 ){
              if (premier) {cc1 = coin->couleur; vc1 = coin->valeur;}
              std::string resW = couleurcarte[coin->couleur];
              resW += valeurcarte[coin->valeur];
              std::string res = resW + "#";
              afficherResultat(result, PT, res);
              if (premier) cards[nc - 1] = resW;
              else if (cc1 != coin->couleur || vc1 != coin->valeur) {
                  // incohérence. quelle détection est fausse?
                  std::cout<< "détection incohérente " << resW << " carte "<< cards[nc - 1] <<std::endl; 
              }
              premier = false;
          }
          //
          // si on traite une vidéo, ajouter les coins détectés ou analysés
        } else { // on traite une video
          // rechercher si le coin est proche d'un coin d' une carte de la frame précédente
          // 
          // si on le trouve, on obtient la carte précédente
          //     et si cette carte est identifiée, on obtient couleur et valeur du nouveau coin
          //     et de tous les coins de cette carte de la frame
          // attendre la fin d'analyse de tous les coins de cette carte de la frame
          for (auto& ucp : cartesPrec){
            for (auto& up : ucp.coinsPrec) {
              if (std::abs(PT.x - up.x) > dc || std::abs(PT.y - up.y) > dc )continue;
              if (coin->couleur >= 0 && coin->valeur > 0 
                && (ucp.couleur != coin->couleur || ucp.valeur != coin->valeur)) continue;
              // on vient de trouver le coin précédent de la carte précédente
              if (!trouve){
                if (ucp.couleur >= 0) {
                  cc1 = ucp.couleur;
                  vc1 = ucp.valeur;
                } else {                    // ne devrait jamais se produire
                  cc1 = coin->couleur;
                  vc1 = coin->valeur;
                  ucp.couleur = cc1;
                  ucp.valeur = vc1;
                }
                carteP = ucp; 
              }
              trouve = true;
              up.x = coin->sommet.x; up.y = coin->sommet.y; // actualiser la position
              break;
            } // coins de la carte précédente
          } // cartes de la frame précédente
          // si on a trouvé la carte précédente dont un coin correspond au coin analysé
          // et si la carte précédente est identifiée, la carte analysée devient identifiée avec tous ses coins
        } // on traite une video
      } //pour chaque coin de la carte

      //    si on a trouvé la carte (un des coins) dans la frame précédente
      //    rechercher chaque coin de cette carte dans les coins précédents
      //     trouvé : actualiser couleur et valeur (précédente)
      //     non trouvé : ajouter avec couleur, valeur et numéro de carte
      if (trouve){ // on a trouvé la carte (carteP) dans la frame précédente de la carte de la frame analysée
        cartes[ic].couleur = cc1;
        cartes[ic].valeur = vc1;

        // ajouter les coins de la carte qui ne sont pas dans la carte (carteP) des coins précédents
        for (auto coin : cartes[ic].coins) {
          cv::Point2i P = coin->sommet;
          bool ajoutercoin(true);
          for (auto& up : carteP.coinsPrec) {
            if (std::abs(P.x - up.x) > dc || std::abs(P.y - up.y) > dc ) continue;
            ajoutercoin = false;
            break;
          }
          if (ajoutercoin){
            uncoinPrec up;
            up.couleur = cc1;
            up.valeur = vc1;
            up.x = P.x;
            up.y = P.y;
            carteP.coinsPrec.push_back(up);
          }
        }
      }

      // valoriser tous les coins de cette carte si la valeur est connue
      if (cc1 >= 0 && vc1 > 0 ){
        for (auto coin : cartes[ic].coins){
          coin->couleur = cc1;
          coin->valeur = vc1;
        }
      }
      if (!trouve) {
        // ajouter une nouvelle carte de la frame précédente 
        // ajouter les nouveaux coins de la nouvelle carte
        //    dans le vecteur des coins de cette nouvelle carte précédente
        unecartePrec ucp;
        auto& carte = cartes[ic];
        ucp.couleur = carte.couleur;
        ucp.valeur = carte.valeur;
        for (auto coin: cartes[ic].coins){
          if (coin->couleur < 0 || coin->valeur <= 0) continue;
          if (carte.couleur < 0) {
            carte.couleur = coin->couleur;
            carte.valeur = coin->valeur;
            ucp.couleur = carte.couleur;
            ucp.valeur = carte.valeur;
          }
          uncoinPrec up;
          up.couleur = ucp.couleur; // redondant. à suprimer de la classe si ce n'est plus référencé
          up.valeur = ucp.valeur;
          up.x = coin->sommet.x;
          up.y = coin->sommet.y;
          ucp.coinsPrec.push_back(up);
        }
        if (ucp.coinsPrec.size() > 0 ) cartesPrec.push_back(ucp);
      }
    } // for(ic) cartes

    // cv::imshow("result", result); // désactivé en multitache
    if (!estvideo) { afficherImage("result", result); cv::waitKey(1);}

    // si on traite une vidéo, les cartes trouvées précédemment ou maintenant
    //    sont dans le vecteur cartesPrec
    // on affiche les valeurs trouvées
    // on reconstitue alors le tableau d'affichage des noms de cartes
    bool nouvellecarte = false;
    if (estvideo){
      nbcards = 0; // nombre de "cartes" dans le tableau d'affichage
      for (const auto ucp : cartesPrec) { 
        for (const auto& up : ucp.coinsPrec){
          cv::Point2i PT(up.x, up.y);
          if ((up.couleur < 0) // coin non identifié (couleur)
          || (up.valeur < 1 || up.valeur > 13)){ // coin non identifié (valeur)
            cv::circle(result, PT, 2, cv::Scalar(255,0,0), -1);
            continue;
          }
          int numcol = up.couleur;
          char nomcol = '?';
          if (numcol >= 0 && numcol <= 3) nomcol = couleurcarte[numcol][0];
          std::string val = valeurcarte[up.valeur];
          std::string res = nomcol + val; 
          afficherResultat(result, PT, res);
          int i;
          for (i=0; i < nbcards; i++){
            if (nomcol == cards[i][0] && val == cards[i].substr(1)) break;
          }
          if (i == nbcards){  // nouvelle carte du pli en cours
            cards[nbcards] = nomcol + val;
            nbcards++;
            nouvellecarte = true;
          }
        }
      }
      if (printoption > 0) {afficherImage("result", result); cv::waitKey(1);}
    } // estvideo

    bool aUneCarte = false;
    for (int i = 0; i < nbcards; i++)
    {
      if(cards[i].size() < 2) continue;
      aUneCarte = true;
      break;
    }
    // afficher le résultat de la dernière image contenant au moins une carte
    if (aUneCarte && (monpli.nbcartes >= 4 || printoption > 0)) {
      //cv::imshow("complet", result); cv::waitKey(1);
    }
    if (printoption > 0 && nbcards > 0) {
      std::cout<<"===> cartes trouvées :"<<std::endl;
      for (int i = 0; i < nbcards; i++)
      {
        if(cards[i].size() < 2) continue;
        char nomcol = cards[i][0];
        std::string valeur = cards[i].substr(1);
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
    }

    if (waitoption && !estvideo) cv::waitKey(0);
    if (waitoption > 1 && estvideo && coinNouveau ) cv::waitKey(0);
    if (false) {
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
void afficherResultat(cv::Mat result, cv::Point2i PT, std::string res, cv::Scalar coulFond)
{
    int pos = res.find('#');
    std::string texte = res.substr(0, pos);
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    cv::Scalar colt(0, 0, 0);          // texte noir
    cv::Scalar rectColor(0, 255, 255); // sur fond jaune
    rectColor = coulFond;
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

    //if (printoption > 0)
    //    cv::imshow("result", result);
}
