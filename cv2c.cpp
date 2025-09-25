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

void afficherResultat(cv::Mat result, cv::Point2i PT, std::string res);

using namespace cv;
using namespace std;

int distribution[4][13][2]; // les 4 mains en cours de décodage

// pour calibration:
static std::vector<Point2f> selectedPoints;

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

// Convertit couleur/valeur en chaîne lisible
std::string carteToString(int couleur, int valeur) {
  const char* couleurs[] = {"P", "C", "K", "T"}; // ♠, ♥, ♦, ♣
  if (couleur < 0 || couleur > 3 || valeur < 1 || valeur > 13) return "??";
  std::string val;
  if (valeur == 1)  val = "A";
  else if (valeur <= 10) val = std::to_string(valeur);
  else if (valeur == 11) val = "V";
  else if (valeur == 12) val = "D";
  else if (valeur == 13) val = "R";
  
  return std::string(couleurs[couleur]) + val;
}

// Convertit numéro joueur en texte
std::string joueurToString(int j) {
  const char* noms[] = {"Nord", "Est", "Sud", "Ouest"};
  return (j >= 0 && j < 4) ? noms[j] : "Inconnu";
}

bool enregistrerContratEtPli(const std::string& nomTable, int numeroDonne,
    const std::string& contratTexte, const std::string& joueurContrat,
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



int processFrame(config &maconf, cv::Mat frame, bool estvideo,std::vector<uncoinPrec>& coinsPrec,  int *nbcoins,  int (&lescoins)[500][10], unpli &monpli);
int processVideo(config &maconf, cv::String nomfichier)
{
    cv::Size rectSize(500, 500); // Exemple : rectangle 3:2
    std::string calibFile = "calibration.yml";
    bool isTransform = false;  // transformation homographique ?  
    cv::Mat Htrans; // matrice de la transformation homographique

    std::chrono::duration<double> duree;
    int numeroframe = 0;
    unpli monpli;   // pli en cours de décodage
    monpli.nbcartes = 0;
    std::vector<uncoinPrec> coinsPrec;

    int lescoins[500][10];   // très largement suffisant
    int nbcoins = 0;
    for(int h = 0; h < 500; h++)
     for (int i= 0; i<10; i++) lescoins[h][i] = 0;

    cv::Mat img = cv::imread(nomfichier);
    if (!img.empty())
    {
        processFrame(maconf, img, false, coinsPrec,  &nbcoins, lescoins, monpli);
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
    cv::Mat frameTotale;
    cv::Mat image;
    cv::Mat diff;
    bool bPremier = true;
    // 0 : 1=valide, 0=non utilisé
    // 1 : X du coin
    // 2 : Y du coin
    // 3 : couleur 0=Pique, 1=Coeur, 2=Carreau, 3=Trefle, -1=indéterminé ou artefact
    // 4 : valeur  1=As, ... 9=9, 10=10, 11=Valet, 12=Dame, 13=Roi
    // 5 : numéro de carte
    // 6 à 9 : inutilisé

    // int distribution[4][13][2];  // reporté en variable globale
    // NSEO 13 cartes couleur et valeur
    for (int i=0; i < 4; i++)
      for (int j=0; j < 13;j++){
        distribution[i][j][0] = -1; // couleur inconnue
        distribution[i][j][1] = 0; // valeur inconnue
      }
        
    unpli monpliprec; // pli précédent
    Pli cepli;  // pli en cours
    Pli pliprec; // pli précédent
    int j1 = maconf.declarant + 1;
    if (j1 > 3) j1 -= 3; 
    cepli.joueur = j1;
    int numpli = 0;
    int nbcartes = 0;  // nombre de cartes dans le pli en cours

    while (true) {
      cap >> frame; // Capture une frame
      if (frame.empty())
      {
          // continuer pour traiter le dernier pli
      } else  if (printoption){
          cv::imshow("Frame", frame); // Afficher la frame
      }
      // étalonner la prise de vue ?
      if (maconf.calibrationoption) {
        cv::Size rectSize;
        rectSize.height = maconf.hauteurcarte;
        rectSize.width = maconf.largeurcarte;
        int rc = calibratePerspective(frame, calibFile);
          if (rc == 0) {
              continue;
          }
          else if (rc == 2) break; // calibration validée, fin du programme
          else { // étalonnage effectué
          }
      }
      // redresser l'image
      rectSize.height = frame.rows;
      rectSize.width = frame.cols;
      cv::Mat frameW = frame.clone(); 
      if (!frame.empty()) {
        if (isTransform && frame.rows > 0 && frame.cols > 0)
          applyCalibration(frame, frameW, Htrans, rectSize);
          frame = frameW;
      }

      // TODO : extraire la partie de l'image où sont posées les cartes jouées
      if (!frame.empty()) {
        frameTotale = frame.clone();
        cv::Rect r;
        r.x = 0; r.y = 0;
        r.width = frame.cols;
        r.height = frame.rows;
        r.x = maconf.xjeu;
        r.y = maconf.yjeu;
        r.width = maconf.wjeu;
        r.height = maconf.hjeu;
        frame = frameTotale(r);
      }

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
          processFrame(maconf, frame, true,coinsPrec, &nbcoins, lescoins);
          framePrec = frameW.clone();
          // Attendre 30 ms et quitter si 'q' est pressé
          if (cv::waitKey(30) == 'q')
          {
              break;
          }
      }
#endif 
      // TODO :
      //        définir un tableau des cartes jouées par chacun des 4 joueurs
      //        définir un tableau de 4 cartes jouées à chaque pli
      //        après le traitement de chaque frame
      //             vérifier que chaque carte détectée n'a pas déjà été jouée
      //                  (à faire seulement ou également lors du décodage d'une carte)
      //             comparer au pli en cours. 
      //             pas de retrait de carte du pli en cours, mais signaler
      //             ajout d'une nouvelle carte uniquement si le pli n'est pas déjà complet
      //                  sinon, signaler le problème 
      if (! frame.empty())  processFrame(maconf, frame, true, coinsPrec,  &nbcoins, lescoins, monpli);
      // s'il n'y a aucune carte dans cette trame et si il y a 4 cartes dans le pli en cours:
      //        enregistrer le pli en tenant compte du joueur qui a entamé le pli
      //        déterminer le joueur (N E S O) qui remporte le pli en fonction du contrat
      //            --> joueur qui entame le pli suivant
      //        effacer le pli
      //        pour chaque carte du vecteur coinsPrec :
      ///           si elle n'est pas dans le pli en cours : l'ajouter au pli
      //            si c'est une autre nouvelle carte du pli : erreur
      //
      bool estvide = true; // aucune carte détectée

      for (auto up : coinsPrec){
        cv::Point2i PT(up.x, up.y);
        int c = up.couleur;
        int v = up.valeur;
        if (c < 0) continue; // couleur non déterminée
        if (v <= 0) continue; // valeur non déterminée
        if (v > 13) continue; // valeur invalide
        estvide = false;
        // carte déjà dans le pli en cours (même couleur, même valeur) ?
        if (c == cepli.carte[0].couleur && v == cepli.carte[0].valeur ) continue;
        if (c == cepli.carte[1].couleur && v == cepli.carte[1].valeur ) continue;
        if (c == cepli.carte[2].couleur && v == cepli.carte[2].valeur ) continue;
        if (c == cepli.carte[3].couleur && v == cepli.carte[3].valeur ) continue;
        // nouvelle carte
        int n = cepli.joueur;
        if (cepli.carte[n].couleur  == -1) {
          cepli.carte[n].couleur= c; cepli.carte[n].valeur= v;
          if (printoption) std::cout<<"nouvelle carte joueur "<<n<< " couleur "<< c
              << " valeur "<< v<<std::endl;
          nbcartes++;
        }
        else {
          n++;
          if (n == 4) n = 0;
          while (n != cepli.joueur) {
            if (cepli.carte[n].couleur == -1) {
              cepli.carte[n].couleur= c; cepli.carte[n].valeur= v;
              if (printoption) std::cout<<"nouvelle carte joueur "<<n<< " couleur "<< c
                  << " valeur "<< v<<std::endl;
              nbcartes++;
              break;
            }
            n++; if (n == 4) n = 0;
          }
        }
      }






      if ((estvide   || frame.empty()) && numpli < 13) {
        if (cepli.carte[0].couleur >= 0
        || cepli.carte[1].couleur >= 0
        || cepli.carte[2].couleur >= 0
        || cepli.carte[3].couleur >= 0) { // un pli en cours (au moins une carte jouée)
          // vérifier que le pli est complet et déterminer le gagnant
          bool estincomplet = false;
          int j1 = cepli.joueur;
          int j = j1; // a priori le même joueur emporte le pli
          int coul = cepli.carte[j1].couleur;
          int val = cepli.carte[j1].valeur; if (val == 1) val = 14; // As > R
          int n = 0;
          for (n=0; n < 4; n++){
            int c = cepli.carte[n].couleur;
            if (c < 0) {
              std::cout<<" pli incomplet"<<std::endl;
              // estincomplet = true; break;
              c = 3;
              cepli.carte[n].couleur = 3; // compléter avec le 2 de trefle
            }
            if (cepli.carte[n].valeur <= 0) cepli.carte[n].valeur = 2;

            if (c != coul && c != maconf.contratcouleur) continue;
            if (c == maconf.contratcouleur && coul != maconf.contratcouleur) {
              coul = maconf.contratcouleur;
              val = cepli.carte[n].valeur;
              j = n;
            } else {
              int v = cepli.carte[n].valeur; if (v == 1) v = 14;
              if (v >= val){
                val = v; j = n;
              }
            }
          }
          if (estincomplet) {
            // TODO : signaler l'anomalie
            //       compléter le pli avec des cartes inconnues (afficher un dos de carte)
            //       enregistrer
            std::cout<<" pli incomplet"<<std::endl;
          } else {
            // mémoriser les 4 cartes dans la distribution
            int joueur = cepli.joueur;
            for (int k = 0; k < 4; k++){
              distribution[joueur][numpli-1][0] = cepli.carte[k].couleur; // couleur
              distribution[joueur][numpli-1][1] = cepli.carte[k].valeur; // valeur
              joueur++; if (joueur > 3) joueur -= 4;
            }
            const std::string NSEO[4] = {"Nord", "Est", "Sud", "Ouest"};
            cepli.joueurgagnant = j;
            pliprec = cepli;
            cepli = Pli(); cepli.joueur = j;
            // enregistrer le pli complet 
            numpli++;
            std::cout<<"==> pli "<<numpli<< " joueur " << NSEO[pliprec.joueur] << "  frame "<< numeroframe <<std::endl;
            for(int i=0; i< 4; i++){
              std::string s = carteToString(pliprec.carte[i].couleur, pliprec.carte[i].valeur);
              if (pliprec.joueur == i) s = "*" +s; else s = " " + s;
              if (pliprec.joueurgagnant == i) s += "*";
              std::cout<<"      "<<s<<std::endl;
            }
            enregistrerContratEtPli ("test", 1, "3SA", "nord", numpli, pliprec);
            // vider le tableau des coins :
            //for(int h = 0; h < 500; h++) for (int i = 0; i < 10; i++) lescoins[h][i] = 0;
            //nbcoins=0;
            coinsPrec.clear();
            // noter qu'il n'y a aucune carte dans le pli en cours
            nbcartes = 0;
            //std::cout<<" nouveau pli enregistre frame "<<numeroframe<<std::endl;
            monpli.nbcartes=0;
            for (int i= 0; i<4; i++) {
              monpli.cartes[i].couleur = -1; 
              monpli.cartes[i].valeur = 0;
              for (int j = 0; j < 4; j++) {
                monpli.cartes[i].sommet[j].x = 0;
                monpli.cartes[i].sommet[j].y = 0;
              }
            } 
            if (waitoption) cv::waitKey(0);          
          } // pli complet
          // TODO : si c'est le premier pli, analyser la zone du mort
          //        la frame à analyser ne comporte aucune carte jouée
          //        la position de cette zone est indiquée dans la configuration (pour N E S et O)
          //        extraire l'image correspondante, la redresser (rotation de 0 1 2 ou 3 angles droits)
          //        composée de 1 à 4 colonnes 
          //        chaque colonne est d'une seule couleur (P C K T)
          //        il reste 12 cartes (le mort à joué le premier pli)
          //        dans chaque colonne, seule la dernière carte (la plus petite) est complète
          //          on y trouve un gros symbole, qui permet d'obtenir la couleur
          //          les autres cartes ne sont visibles que pour 2 coins de largeur de carte
          //        vérifier si on peut utiliser le code d'analyse des cartes jouées
          //        sinon : développer une analyse spécifique
          //        mémoriser les cartes du mort
          //
          // si c'est un autre pli : vérifier qu'une des 4 cartes a été jouée par le mort
          //                         on en déduit le premier joueur (N E S O) du pli
          //                         valider avec le calcul selon les règles du bridge
        }
        else {
          if (nbcoins && printoption)
           std::cout<<" frame vide, aucune carte jouée du pli en cours"<<std::endl;
        }
        nbcoins = 0; // aucun coin pour la frame précédente
      } // frame vide, aucune carte trouvée

      if (frame.empty()) break;

      numeroframe++;
      if (printoption) std::cout << "====== fin de frame "<< numeroframe <<" ======" << std::endl;
  }
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
    config maconf;
    std::cout << " arguments optionels :"
              << " nom du fichier image ou video, "
              << " nom du fichier de configuration, "
              << " taille de carte (en pixels) " << std::endl
              << std::endl;

    for(int i=0; i<sizeof(Durees); i++) Durees[i] = 0;
    //setenv("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/5/", 1);

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
        nomconf = "FUNBRIDGE.txt";
    lireConfig(nomconf, maconf);
    // else  if (maconf.hauteurcarte != 0) resetconfig(maconf.hauteurcarte, maconf);
    waitoption = maconf.waitoption;
    printoption = maconf.printoption;
    threadoption = maconf.threadoption;
    if (maconf.tesOCR == 0)
        nomOCR = "SERVEUR";
    else
        nomOCR = "tesOCR";
    
    int ret = processVideo(maconf, nomfichier);
    std::cout<<" Durées de traitements "<<Durees[0]<<" , "<<Durees[1]<<" , "<<Durees[2]<<std::endl;
    return ret;
}


// analyser les cartes
// 
void traiterCartes(cv::Mat image, config& maconf, std::vector<uncoin>& Coins, std::vector<uncoinPrec>& coinsPrec,
   int *pnbcoins, int (&lescoins)[500][10], const std::vector<ligne>&  lignes, unpli& monpli) {
  if (printoption) std::cout<< std::endl<<"================== recherche des nouvelles cartes ======"<<std::endl;
  const std::string nomval[14] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "V", "D", "R"};
  bool estvideo = true;
  int epsilon = std::max(4,2*maconf.deltacadre);
  int nca = 0; // numéro de carte complète à analyser
  int nc = 0; // numéro de carte cherchée
  int numcol, valcarte;
  int h0 = -1; // coin de la frame précédente
  int h1 = -1;
  int attendre = 0; // pour pouvoir actualiser l'affichage en debug 
  if(attendre) cv::waitKey(0);
  // déterminer si la carte était déjà dans la frame précédente
  // TODO : la distance entre un sommet de la carte et un point de la frame précédente
  //         doit être commune à tous les sommets de la carte
  // un des coins de cette carte était-il déjà dans la frame précédente
  //for (int n = 0; n < nbcoins; n++){
  for (int n = 0; n < Coins.size(); n++){
    // nouvelle carte
    if (Coins[n].numCarte <= nc) continue; // coin d'une carte déjà recherchée
    if (Coins[n].elimine) continue;
    if (Coins[n].couleur >= 0 && Coins[n].valeur > 0) continue; // coin déjà analysé 
    nc = Coins[n].numCarte; // numéro de carte suivante de cette frame

        // noter que par construction des numéros de cartes, 
        //  les numéros de cartes sont en ordre croissant des numéros de coins
    // considérer tous les coins de cette carte nc
    h0 = -1;  // a priori pas présente dans la frame précédente
    h1 = -1;
    bool estNouveau = false; // nouveau sommet dans cette frame ?
    cv::Point2i  P1(Coins[n].sommet);
    int dmin(100), dmax(0);
    std::string nomcarte="";
    int numcol(-1), valcarte(0);
    for(int  m= n; m< Coins.size(); m++ ){
      if (Coins[m].numCarte != nc) continue; // coin d'une autre carte
      cv::Point2i PT(Coins[m].sommet);
      // rechercher dans le vecteur coinsPrecdes coins  mémorisés des frames précédentes
      bool trouve(false);
      h0 = -1;
      dmin = 100; // distance minimale entre le coin PT de la carte et la frame précédente
      
      for (int h = 0; h < coinsPrec.size(); h++){
        uncoinPrec up = coinsPrec[h];
        cv::Point2i QT(up.x, up.y);
        if (std::abs(PT.x - QT.x) > epsilon ) continue;
        if (std::abs(PT.y - QT.y) > epsilon ) continue;
        if (h1 >= 0) { // on a trouvé un coin précédent proche de cette carte
          if (numcol >= 0 && valcarte > 0) {
            if (numcol != up.couleur) continue; // couleur différente
            if (valcarte != up.valeur) continue; // valeur différente
          }
        }
        int d = std::max(std::abs(PT.x - QT.x), std::abs(PT.y - QT.y));
        if (dmin > d ) { 
          dmin = d;
          h0 = h;
        }
      }
      if (h0 >= 0){ // on a trouvé un coin précédent proche du coin PT de cette carte
        if (h1 < 0 || numcol < 0 || valcarte <= 0){ // pas encore trouvé un coin proche dans la frame précédente
          // mémoriser couleur et valeur du coin précédent
          numcol = coinsPrec[h0].couleur;
          valcarte = coinsPrec[h0].valeur;
          h1 = h0;
        }
      }
      if (h0 < 0){ // le coin PT n'est pas trouvé dans la frame précédente
        // mémoriser qu'il y a un nouveau sommet de la carte
        // et attendre la recherche de tous les sommets de la carte
        estNouveau = true;
      }
      else if (dmax < dmin) dmax = dmin; // maximum entre les coins de la carte et de la carte de la frame précédente
      if (h1 >= 0 && dmax > maconf.hauteurcarte / 10) break;
    } // for m
    // si un des sommets de la carte est nouveau dans cette frame
    // et si on a trouvé une carte correspondant à celle-ci dans la frame précédente
    //  qui n'a pas de couleur et valeur
    //  ==> ne pas reconnaitre que c'est la même carte
    if (estNouveau && h1 >= 0) {
      if (numcol < 0 || valcarte <= 0) dmax = 100;
    }
    if (h1 >= 0 && dmax < maconf.hauteurcarte / 10) { // carte proche d'une carte de la frame précédente
      if (numcol == 0) nomcarte = "Pique";
      else if (numcol == 1) nomcarte = "Coeur";
      else if (numcol == 2) nomcarte = "Carreau";
      else if (numcol == 3) nomcarte = "Trefle";
      if (coinsPrec[h1].valeur > 0 && coinsPrec[h1].valeur < 14)
        nomcarte += " " + nomval[coinsPrec[h1].valeur];
      if(printoption)  std::cout<< " carte "<< nc << " ("<< nomcarte 
        << ") déjà dans la frame précédente." << std::endl;
      // noter qu'il est inutile d'analyser ce coin
      Coins[n].couleur = numcol;
      Coins[n].valeur = valcarte;
      // ainsi que tous les coins de cette carte
      for (int i= 0; i < Coins.size(); i++){
        if (Coins[i].numCarte != nc) continue;
        Coins[i].couleur = numcol;
        Coins[i].valeur = valcarte;
      } 
      continue; // for n
    }

    // nouveau coin, nouvelle carte dans cette frame
    // vérifier que c'est la seule nouvelle carte
    // analyser les coins à partir du coin n+1 nouveaux 
    //   de la même carte --> les angles de la carte
    //   d'une autre carte --> erreur : plusieurs nouvelles cartes
    if (printoption) std::cout<< "nouvelle carte "<< nc<<" nouveau sommet "<< n <<P1<<std::endl;
    bool estcarte(false);
    for (int m =n+1; m< Coins.size(); m++){
      if (Coins[m].numCarte != nc) continue;
      if (Coins[m].elimine) continue;
      cv::Point2i P2(Coins[m].sommet);
      // coin de la même carte ?
      if (nc == Coins[m].numCarte) { // même carte
        // mémoriser
        if (printoption) std::cout << " autre sommet "<< m<< " "<< P2<<std::endl;
        // distance entre les deux points proche de longueur ou largeur de carte
        if (!estcarte) {
          double dist = std::sqrt((P1.x - P2.x)*(P1.x - P2.x) + (P1.y - P2.y)*(P1.y - P2.y));
          if (std::abs(dist - maconf.hauteurcarte) < maconf.hauteurcarte / 10) estcarte =true;
          if (std::abs(dist - maconf.largeurcarte) < maconf.hauteurcarte / 10) estcarte =true;
        }
      }               
    } // for m
    if (estcarte) { // carte avec au moins deux coins et absente de la frame précédente
      if (printoption) std::cout<<" carte complète "<< nc<<std::endl;
      // analyser la nouvelle carte
      if (nca != 0){
        if (printoption) std::cout<< " plusieurs cartes complètes "<< nca << ","<<nc<<std::endl;
        // TODO : ignorer la carte dont un sommet est sur un bord de carte de la frame précédente
        // parcourir les coins de la carte nc
        // comparer chaque coin aux bords des cartes du pli en cours
        // vérifier les deux cartes NC puis NCA
        // vérifier les sommets de la carte nc
        bool ignorerNC = false;
        for (int i = 0; i < Coins.size(); i++){
          if (Coins[i].numCarte != nc) continue;
          cv::Point2i P(Coins[i].sommet);
          cv::Point2i U, V;
          // comparer aux bords de cartes du pli en cours
          for (int j = 0; j < monpli.nbcartes; j++ ) { // chaque carte du pli
            numcol = monpli.cartes[j].couleur;
            numcol = monpli.cartes[j].valeur;
            for (int k = 0; k < 4; k++){ // chaque sommet
              //U = cv::Point2i(monpli.cartes[j].sommet[k].x, monpli.cartes[j].sommet[k].y);
              U = monpli.cartes[j].sommet[k];
              for (int l=k+1; l < 4; l++) { // chaque autre sommet de la même carte du pli
                //V = cv::Point2i(monpli.cartes[j].sommet[l].x, monpli.cartes[j].sommet[l].y);
                V = monpli.cartes[j].sommet[l];
                // on a un bord UV
                double dist = calculerDistance(P, U, V);
                if (std::abs(dist) > maconf.deltacadre) continue;
                ignorerNC = true;
                break;
              } // for l
              if (ignorerNC) break;
            } // for k
            if (ignorerNC) break;
          } // for j
          if (ignorerNC) break;
        } // for i
        if (ignorerNC) {
          nc = 0;
          if (printoption) std::cout<<" carte "<<nc<< " ignorée car dans la frame précédente"<<std::endl;
        }
        else {
          // vérifier la carte nca
          for (int i = 0; i < Coins.size(); i++){
            if (Coins[i].numCarte != nca) continue;
            cv::Point2i P(Coins[i].sommet);
            // comparer aux bords de cartes du pli en cours
            for (int j = 0; j < monpli.nbcartes; j++ ) { // chaque carte du pli
              numcol = monpli.cartes[j].couleur;
              numcol = monpli.cartes[j].valeur;
              for (int k = 0; k < 4; k++){ // chaque sommet
                //cv::Point2i U(monpli.cartes[j].sommet[k].x, monpli.cartes[j].sommet[k].y);
                cv::Point2i U(monpli.cartes[j].sommet[k]);
                for (int l=k+1; l < 4; l++) { // chaque autre sommet de la même carte du pli
                  //cv::Point2i V(monpli.cartes[j].sommet[l].x, monpli.cartes[j].sommet[l].y);
                  cv::Point2i V(monpli.cartes[j].sommet[l]);
                  // on a un bord UV
                  double dist = calculerDistance(P, U, V);
                  if (std::abs(dist) > maconf.deltacadre) continue;
                  ignorerNC = true;
                  break;
                } // for l
                if (ignorerNC) break;
              } // for k
              if (ignorerNC) break;
            } // for j
            if (ignorerNC) break;
          } // for i
          if (ignorerNC) {
            if (printoption) std::cout<<" carte "<<nca<< " ignorée car dans la frame précédente"<<std::endl;
            nca = nc;
            nc = 0;
          }
        } // else
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
    cv::Point2i P, Q, U, V;
    int n1(-1), n2(-1), n3(-1), n4(-1);
    int nbpts;
    //int i, j;
    int dc = std::max(4, maconf.deltacadre);
    // rechercher le coin le plus à gauche (et le plus haut s'il y en a plusieurs)
    int xmin(12345); // une grande valeur
    for (int n = 0; n < Coins.size(); n++) {
      if (Coins[n].numCarte != nca) continue; // autre carte
      if (Coins[n].elimine) continue; // coin éliminé
      if (Coins[n].sommet.x <= xmin ) {
        xmin = Coins[n].sommet.x;  //limite gauche de la carte
      }
    }
    int ymin(12345); // une grande valeur
    for (int n = 0; n < Coins.size(); n++) {
      if (Coins[n].numCarte != nca) continue; // autre carte
      if (Coins[n].elimine) continue; // coin éliminé
      if (Coins[n].sommet.x <= xmin ) {    // un des coins les plus à gauche
        if (Coins[n].sommet.y < ymin ) {
          n1 = n;
          ymin = Coins[n].sommet.y;
        }
      }
    }
    P = cv::Point2i(Coins[n1].sommet);
    if(printoption) std::cout<<" coin "<< n1 <<" "<<P<<std::endl;
    // rechercher un coin opposé sur le premier coté du coin P puis sur le deuxième
    n2 = -1;
    int lgref, lgref2;
    float a,b,c, aa,bb,cc;
    cv::Vec4i ln = Coins[n1].l1->ln;
    cv::Vec4i ln2 = Coins[n1].l2->ln;
    a = Coins[n1].l1->a; b = Coins[n1].l1->b; c = Coins[n1].l1->c;
    aa = Coins[n1].l2->a; bb = Coins[n1].l2->b; cc = Coins[n1].l2->c;
    for (int i=0; n2< 0 && i < 2; i++) {
      n2 = -1;
      float ecart, ecartmin(dc);
      for (int n = 0; n < Coins.size(); n++) {
        if (Coins[n].numCarte != nca || Coins[n].elimine) continue;
        if (n == n1) continue;
        Q = cv::Point2i(Coins[n].sommet);
        float dist = a*Q.x + b*Q.y + c;
        float lg = std::abs(dist);
        if (lg > dc) continue; // Q n'est pas opposé à P
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
      if (n2 < 0) {
        ln2 = ln;
        ln = Coins[n1].l2->ln;
        aa = a; bb=b; cc=c;
        a = Coins[n1].l2->a; b = Coins[n1].l2->b; c = Coins[n1].l2->c;
      }
    } // while n2


    if (n2 < 0) {
      // aucun coin opposé au coin P
      nbpts = 1;
    } else {
      Q = cv::Point2i(Coins[n2].sommet);
      if(printoption) std::cout<<" coin opposé "<< n2 <<" "<<Q<<std::endl;
      // on a deux sommets de la carte P (n1) et Q (n2)
      // rechercher ou calculer les deux autres sommets U et V
      // la ligne commune est : coin n1 indice i, ligne ln, coefficient a b c
      // rechercher le coin U sur l'autre ligne (j) du coin P
      cv::Vec4i ln = ln2;
      float ecmin = 3;
      n3 = -1;
      for (int n = 0; n < Coins.size(); n++) {
        if (Coins[n].numCarte != nca || Coins[n].elimine) continue;
        if (n == n1 || n == n2) continue;
        U = cv::Point2i(Coins[n].sommet);
        float dist = aa*U.x + bb*U.y + cc;
        if (std::abs(dist) > 1) continue; // U n'est pas sur l'autre ligne (que PQ) j du coin P
        float lg = std::sqrt((U.x - P.x)*(U.x - P.x) + (U.y - P.y)*(U.y - P.y));
        if (std::abs(lg - lgref2) > ecmin) continue; // U n'est pas à distance convenable de P (hauteur ou largeur de carte)
        if(std::abs(lg - lgref2) < ecmin) {
          ecmin = std::abs(lg - lgref2);
          n3 = n;
        }
      }
      if(n3 >= 0) {
        U = cv::Point2i(Coins[n3].sommet);
        if(printoption) std::cout<<" coin trois "<< n3 <<" "<<U<<std::endl;
      }
      ligne *l1 = Coins[n2].l1;
      ligne *l2 = Coins[n2].l2;
      // rechercher le sommet V sur le coté orthogonal à PQ du coin Q (n2)
      float ps = a*Coins[n2].l1->a + b*Coins[n2].l1->b;
      if(std::abs(ps) < 0.5) {
        l2 = l1;
        l1 =Coins[n2].l2;
      }
      // ii est la ligne // PQ  jj est la ligne ortogonale à PQ
      // rechercher un coin V de cette carte tel que QV = lgref2 (à deltacadre près)
      // distance de V à la droite PQ = lgref2   et QV orthogonal à PQ
      n4 = -1;
      ecmin = 3;
        for (int n = 0; n < Coins.size(); n++) {
        if (Coins[n].numCarte != nca || Coins[n].elimine) continue;
        if (n == n1 || n == n2 || n == n3) continue;
        V = cv::Point2i(Coins[n].sommet);
        // distance de V à cette ligne jj orthogonale à PQ:
        float d = V.x * l2->a + V.y * l2->b + l2->c;
        if (std::abs(d) > dc) continue;

        float dist = std::abs(a*V.x + b*V.y + c);
        if (std::abs(dist - lgref2) > ecmin) continue;
        if (std::abs(dist - lgref2) < ecmin) {
          ecmin = std::abs(dist - lgref2);
          n4 = n;
        }
      }
      if(n4 >= 0) {
        V = cv::Point2i(Coins[n4].sommet);
        if(printoption) std::cout<<" coin quatre "<< n4 <<" "<<V<<std::endl;
      }


      if (n3 < 0) {
        // pas de coin sur l'autre ligne du coin P
        U = V;
        n3 = n4;
        n4 = -1;
      }
    }
    nbpts = 1;
    if (n4 >= 0) nbpts = 4;
    else if (n3 >= 0) nbpts = 3;
    else if (n2 >= 0) nbpts = 2;
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
    double lgW; // longueur de l'autre coté
    double lg, lg2;
    int nbptsW = nbpts;
    if ((nbpts == 2 || nbpts == 3)){
      // les deux premiers points sont la longueur ou la largeur ou la diagonale

      // on dispose du vecteur normal des lignes des coins
      cv::Point2i P(Coins[n1].sommet);
      cv::Point2i Q(Coins[n2].sommet);
      // vérifier que PQ est la hauteur ou la largeur de carte
      cv::Point2i C; // autre sommet opposé au point P
      cv::Point2i D; // autre sommet opposé au point Q
      // PQ hauteur ou largeur de carte ?
      lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
      lg = std::sqrt(lg2);
      if (lg > 5*maconf.hauteurcarte/4 && nbpts == 3){
        // c'est la diagonale.
        // s'il y a un 3ème sommet, remplacer le 2ème sommet
        Q = Coins[n3].sommet;
        pts[1][0] = pts[2][0]; pts[1][1] = pts[2][1]; 
        lg2 = (Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y);
        lg = std::sqrt(lg2);
        int w = n2; n2 = n3; n3 = w;
      }
      if (std::abs (lg - maconf.hauteurcarte) <= dc ) lgW = maconf.largeurcarte;
      else if (std::abs (lg - maconf.largeurcarte) <= dc ) lgW = maconf.hauteurcarte;
      else lgW = 0;
      if (lgW > dc) {

        double a = Coins[n1].l1->a;
        double b = Coins[n1].l1->b;
        double c = Coins[n1].l1->c;
        double dist = a*Q.x + b*Q.y +c;
        cv::Point2i R(Coins[n1].K);
        if (abs(dist) > dc) { // PQ // autre ligne du coin P
          a = Coins[n1].l2->a;
          b = Coins[n1].l2->b;
          c = Coins[n1].l2->c;
          R = Coins[n1].H;
        }
        // distance de R à la droite (i)
        double dR = a*R.x + b*R.y + c;
        // dR > 0 : vecteur normal dirigé vers l'intérieur 
        if (dR < 0) lgW = -lgW;
        C.x = Q.x + a*lgW;
        C.y = Q.y + b*lgW;
        D.x = P.x + a*lgW;
        D.y = P.y + b*lgW;
        if (nbpts == 3) {
          // remplacer C ou D par le point n3
          cv::Point2i K(Coins[n3].sommet);
          if (std::abs(C.x - K.x) < dc && std::abs(C.y - K.y) < dc ) {
            C = K;
          } else if (std::abs(D.x - K.x) < dc && std::abs(D.y - K.y) < dc ) {
            D = K;
          }
        }
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
      } //if lgW > dc
    }

    if (nbpts == 4){
      // vérifier qu'aucun autre coin d'une autre carte n'est dans celle-ci
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
      for (int n = 0; n < Coins.size(); n++){
        if (Coins[n].numCarte == nca) continue; // un coin de cette carte
        cv::Point2i P(Coins[n].sommet);
        // TODO : ignorer ce coin s'il n'était déjà dans la frame précédente
        if (estvideo) {
          bool trouve = false;
          for (int h = 0; h < coinsPrec.size(); h++){
            uncoinPrec up = coinsPrec[h];
            if (std::abs(P.x - coinsPrec[h].x) <= 1 
            && std::abs(P.y - coinsPrec[h].y) <= 1 ) {
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
          //ps = (P.x - U.x)*(V.x - U.x) + (P.y - U.y)*(V.y - U.y);
          ps = unvecteur(U,P)*unvecteur(U,V);
          if (ps < 3*maconf.hauteurcarte) {
            estdehors = true; break;}
          //if (ps > (V.x - U.x)*(V.x - U.x) + (V.y - U.y)*(V.y - U.y) - 2*maconf.hauteurcarte) {
          if (ps > unvecteur(U,V)*unvecteur(U,V) - 2*maconf.hauteurcarte) {
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
        // TODO : comparer aux cartes du pli en cours
        //        comparer à une carte (C)du pli en cours
        //        comparer chaque sommet aux 4 sommets de la carte (C)
        //        obtenir l'écart minimum
        //        obtenir pour les 4 sommets de (C) : ecart minimum et maximum des écarts minimaux
        //        obtenir l'écart global mini et maxi des minima 
        //        si l'écart maxi est faible, ignorer cette nouvelle carte 
        //        sinon : décoder et ajouter la carte au pli en cours
        float dmin2 = image.cols + image.rows;
        int iproche;
        for (int i = 0; i < monpli.nbcartes; i++){ // chaque carte du pli en cours
          float dmax1 = 0; // distance de la carte analysée à la carte du pli
          for (int k = 0; k < nbpts; k++){ // chaque sommet de la carte analysée
            unpoint M(pts[k][0], pts[k][1]); // un point de la carte en cours
            float dmin1 = image.cols + image.rows;
            for (int j = 0; j< 4; j++){ // chaque sommet de la carte du pli en cours
              unpoint S(monpli.cartes[i].sommet[j].x, monpli.cartes[i].sommet[j].y) ;
              // float d = std::sqrt((M.x - S.x)*(M.x - S.x) + (M.y - S.y)*(M.y - S.y));
              float d = unvecteur(M,S).lg();
              if (d < dmin1) dmin1 = d;
            }
            if (dmax1 < dmin1) dmax1 = dmin1;
          }
          // dmax1 = distance de la carte analysée à la carte i du pli
          if (dmin2 > dmax1) {dmin2 = dmax1; iproche = i;}
        }
        int numcol= -1;
        int valcarte = 0;
        bool ajouteraupli = false;
        // carte analysée proche d'une carte du pli ?
        if (dmin2 < maconf.hauteurcarte/4) {
          numcol = monpli.cartes[iproche].couleur;
          valcarte = monpli.cartes[iproche].valeur;
          if (printoption){
            std::cout<<"carte proche d'une carte du pli en cours, valeur "<<valcarte<<" couleur "<<numcol<<std::endl;
          }
        }
        else {  // Décoder cette nouvelle carte
          valcarte = decoderCarte(image, pts, maconf, numcol);
          // vérifier qu'elle n'est pas déjà dans la distribution
          // puis l'ajouter dans la distribution

        if (numcol >= 0 && valcarte > 0 && valcarte <= 13) { // on a trouvé
          bool duplique = false;
          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 13; j++) {
              if ( numcol == distribution[i][j][0] && valcarte == distribution[i][j][1]) {
                std::cout<<" cette carte ("<<numcol<<","<<valcarte<<")"<<" est deja dans le pli "
                  <<j<<" du joueur "<<i<<std::endl;
                duplique = true;
              }
            }
        }
          ajouteraupli = true;
          if (printoption) {
              std::cout<<"==> valeur carte "<<valcarte<<" couleur "<<numcol<<std::endl;
          }
        }
        // mémoriser la valeur obtenue sur tous les coins de la carte
        // uniquement si on a trouvé couleur et valeur
        if (numcol >= 0 && valcarte > 0 && valcarte <= 13) {
          for (int n = 0; n < Coins.size(); n++){
            if (Coins[n].numCarte != nca) continue; // pas un coin de cette carte
            Coins[n].couleur = numcol;
            Coins[n].valeur = valcarte;
            if (valcarte > 10) Coins[n].estunRDV = true;
          }
        }
        // TODO : si l'option d'analyse des coins isolés est active
        //        noter : ne pas analyser les coins pas encore analysés et sur un bord de cette carte
        if (maconf.coinsoption){
          cv::Point2i P(pts[3][0], pts[3][1]);
          for (int i = 0; i < 4; i++){
            cv::Point2i Q(pts[i][0], pts[i][1]);
            float a, b, c; // normale de la droite PQ et equation de PQ
            float lg = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
            a = (Q.y - P.y) /lg;
            b = -(Q.x - P.x) / lg;
            c = -a*P.x - b*P.y; 
            for (int n = 0; n < Coins.size(); n++){
              if (Coins[n].elimine) continue;
              if (Coins[n].couleur >= 0 && Coins[n].valeur > 0) continue; // déjà décodé
              cv::Point2i M(Coins[n].sommet);
              // sommet (M) du coin n sur le segment PQ ?
              if (std::abs(a*M.x + b*M.y + c) < maconf.deltacadre ) {
                // MP.MQ < 0 --> M entre P et Q ?
                if ((P.x - M.x)*(Q.x - M.x) + (P.y - M.y)*(Q.y - M.y) < 0) {
                  Coins[n].elimine = true;
                  break;
                }
              }
              P = Q;
            }
          }
        }

        //        
        if (ajouteraupli){
          // ajouter la carte au pli en cours
          if (monpli.nbcartes < 4) {
            int nc = monpli.nbcartes;
            monpli.cartes[nc].couleur = numcol;
            monpli.cartes[nc].valeur = valcarte;
            for (int i=0; i < 4;i++) {
              int x = pts[i][0]; int y = pts[i][1];
              monpli.cartes[nc].sommet[i].x = x;
              monpli.cartes[nc].sommet[i].y = y;
            }
            monpli.nbcartes++;
          }
        }
      }
    }
  } // traitement carte nca
return;
}

// TODO : comparer l'image à l'image précédente, si on traite une vidéo
//    après le traitement d'une frame, conserver le résultat du décodage
//     qui se trouve dans le tableau des coins
//    traitement de la nouvelle frame:
//    comparer à la frame précédente. on obtient les pixels modifiés
//    invalider les résultats de chaque coin sur une zone modifiée
//    restreindre l'image à analyser à la partie modifiée  

int processFrame(config &maconf, cv::Mat image, bool estvideo, std::vector<uncoinPrec>& coinsPrec, int *pnbcoins, int (&lescoins)[500][10], unpli &monpli)
{
    std::chrono::duration<double> duree;
    activeThreads = 0;
    if (maconf.threadoption > 1)
        MAX_THREADS = maconf.threadoption;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::string> resultats; // vecteur des résultats
    std::vector<std::thread> threads;

#define NBCOULEURS 10
    cv::Scalar couleurs[10];
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
    int c = 0;

    if (image.empty())
    {
        std::cerr << "Erreur de chargement de l'image" << std::endl;
        return -1;
    }
    cv::Mat result = image.clone();
    // auto start = std::chrono::high_resolution_clock::now();

    // afficher l'image en couleurs
    if (printoption)
        cv::imshow("couleur", image);
    /*****************************
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
    **********************/
    // Convertir en niveaux de gris
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Appliquer le flou gaussien pour réduire le bruit
    //cv::Mat blurred;
    //cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    // cv::imshow("blur", blurred);

    ////////////////// utiliser une des images monochromatiques /////////////////
    cv::Mat grise;
    cv::cvtColor(gray, grise, cv::COLOR_GRAY2BGR);
    if (printoption)
        cv::imshow("grise", grise);

    std::vector<ligne> lignes;   // segments complétés par l'équation de droite
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
        //std::vector<cv::Vec4f> lines_f;

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

        //lsd->detect(gray, lines_f);
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
      ima2 = image.clone();
      cv::Canny(ima2, edges, gmin, gmax, 3, false);
      cv::HoughLinesP(edges, lines, 1, theta, threshold, minlg, gap);
      if (printoption)
          cv::imshow("bords", edges);
      // cv::waitKey(0);
    } // methode 1

    // mémoriser les segments de ligne détectés avec l'équation de droite
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    // calculer la longueur et l'équation cartésienne

    for (size_t i = 0; i < lines.size(); i++)
    {
        ligne ln;
        cv::Vec4i l(lines[i][0], lines[i][1], lines[i][2], lines[i][3]);
        ln.ln = l;

        cv::Point A(l[0], l[1]);
        cv::Point B(l[2], l[3]);
        // tracer la ligne sur l'image result
        float lg = std::sqrt((l[2] - l[0])*(l[2] - l[0]) + (l[3] - l[1])*(l[3] - l[1]));
        // vecteur normal (a,b) directeur (b, -a)  
        float a = -(B.y - A.y) / lg;
        float b = (B.x - A.x) / lg;
        float c = -a*A.x - b*A.y; // ax + by + c = 0
        ln.lg = lg;
        ln.a = a;
        ln.b = b;
        ln.c = c;
        lignes.push_back(ln);
    }

    auto t11 = std::chrono::high_resolution_clock::now();
    duree = t11 - t0;
    if(printoption)
      std::cout << "Duree de detection des lignes : " << duree.count() << " secondes" << std::endl;
      Durees[0] += duree.count();
    if (waitoption > 1)
        cv::waitKey(0);
    
    // Dessiner les segments de ligne détectés
    cv::cvtColor(gray, result, cv::COLOR_GRAY2BGR);
    // lsd->drawSegments(result, lines_f); // plus loin pour plusieurs couleurs
    // Convertir les coordonnées des lignes en entiers
    // calculer la longueur et l'équation cartésienne

    if (printoption){
      int ic = 0;
      for (size_t i = 0; i < lignes.size() ; i++)
      {
          ic++; if (ic >= NBCOULEURS) ic = 0;
          ligne ln = lignes[i];
          cv::Vec4i l = ln.ln;

          cv::Point A(l[0], l[1]);
          cv::Point B(l[2], l[3]);
          // tracer la ligne sur l'image result
          cv::line(result, A, B, couleurs[ic], 1);
      }
      // Afficher l'image avec les lignes détectées
      cv::imshow("ximgproc", result);
    }

    auto t22 = std::chrono::high_resolution_clock::now();
    ima2 = grise.clone();
    cv::Canny(ima2, edges, gmin, gmax, 3, false);
    if (printoption)
        cv::imshow("bords", edges);
    int il1 = 0;
    int nblignes = lignes.size();
    if (printoption) {
      // Dessiner les segments de droite et afficher leurs longueurs et extrémités
      //********************** fond noir pour ne voir que les lignes des coins
      for (int y = 0; y < ima2.rows; y++)
          for (int x = 0; x < ima2.cols; x++) ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
      /***************/

      c = 0; // indice de couleur
      double maxlg = 0;
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
          std::cout << "Ligne " << i <<" "<< A << "->" << B << " Longueur: " << lg << std::endl;
          if (lg > maxlg) {
              maxlg = lg;
              il1 = i; // indice de la ligne la plus longue
          }
      }
      // Afficher l'image avec les segments de droite
      cv::imshow("Lignes toutes", ima2);
    }

    int lgmax = maconf.taillechiffre;
    // fusionner les lignes AB  et CD si // si C et D sont proches de la ligne AB
    //   et si C ou D est proche de A ou B : AB --> AC ou AD ou BC ou BD
    if (maconf.fusionoption) {
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
      int maxlg;
      double tolerance = 0.4; // Ajustez la tolérance selon vos besoins. 0.4 entre 45 et 60 degrés
      cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8U);
      maxlg = maconf.hauteurcarte / 6;
      maxlg *= maxlg;
      for (int i = 0; i < lignes.size(); i++)
      {
        //cv::Vec4i l = lines[i];
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
    //        concerne les vidéos où il y a une poutre entre la table et la caméra

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

    double maxlg = 0;
    // afficher les lignes qui restent
    if (printoption) {
      for (int y = 0; y < ima2.rows; y++) for (int x = 0; x < ima2.cols; x++)
          ima2.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
      c = 0;
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
      // Afficher l'image avec les segments de droite
      std::cout << "longueur maximale " << maxlg << std::endl;
      cv::imshow("Lignes", ima2);
        // cv::waitKey(1);
    }

    //////////////////////////////// rechercher les coins des cartes ///////////////////
    //

    std::vector<uncoin> Coins;
    int nbcoins = 0;
    int nbcartes = 0;
    // pour chaque ligne AB
    for (int i = 0; i < nblignes; i++) {
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
      
      
      for (int j = i + 1; j < nblignes; j++) {
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

        bool bCoin = false;
        cv::Point2i H, K; // H sur la jigne i, loin du sommet, K sur la ligne j
        // A proche de C ?
        if (std::abs(C.x - A.x) < maconf.deltacoin && std::abs(C.y - A.y) < maconf.deltacoin)
        { // A proche de C
          if (printoption)
              std::cout << "  coin AC (" << A.x - C.x << "," << A.y - C.y << ") " << A << "," << C << std::endl;
          bCoin = true;
          H = B; K = D;
        }
        // A proche de D ?
        else if (std::abs(A.x - D.x) < maconf.deltacoin && std::abs(A.y - D.y) < maconf.deltacoin)
        { // A proche de D
          if (printoption)
              std::cout << "  coin AD (" << A.x - D.x << "," << A.y - D.y << ") " << A << "," << D << std::endl;
          bCoin = true;
          H = B; K = C;
        }
        // B proche de C ?
        else if (std::abs(B.x - C.x) < maconf.deltacoin && std::abs(B.y - C.y) < maconf.deltacoin)
        { // B proche de C
          if (printoption)
              std::cout << "  coin BC (" << B.x - C.x << "," << B.y - C.y << ") " << B << "," << C << std::endl;
          bCoin = true;
          H = A; K = D;
        }
        // B proche de D ?
        else if (std::abs(B.x - D.x) < maconf.deltacoin && std::abs(B.y - D.y) < maconf.deltacoin)
        { // B proche de D
          if (printoption)
              std::cout << "  coin BD (" << B.x - D.x << "," << B.y - D.y << ") " << B << "," << D << std::endl;
          bCoin = true;
          H = A; K = C;
        }
        if (bCoin)
        {
          // calculer l'angle du complément # sinus = cosinus des normales
          double alfa = std::abs(psX) * 180.0 / 3.1416; // en degrés

          if (printoption) std::cout << "  angle " << alfa << " degres" << std::endl;
          //        mémoriser le coin : indices des deux droites et numéros des extrémités de chaque droite (0 ou 2)
          cv::Point2i P = calculerInter(l1, l2);
          float length = lignes[j].lg;  // longueur CD
          if (printoption) {
              std::cout << "    " << j << "  Ligne CD " << j << " " << C << "->" << D << " Longueur: " << length << std::endl;
              std::cout << " ==> coin " << nbcoins << " en " << P << " " << i << " " << j << std::endl;
          }
          Coins.push_back(uncoin(lignes[i], lignes[j]));
          int n = Coins.size() - 1;
          //Coins[n].pcoins = coins;
          Coins[n].numcoin = nbcoins;
          Coins[n].sommet = P;
          Coins[n].H = H;
          Coins[n].K = K;
          Coins[n].R = H;
          Coins[n].S = K;
          nbcoins++;
        }
      } // deuxième droite
    } // première droite

    auto t33 = std::chrono::high_resolution_clock::now();
    duree = t33 - t22;
    if (printoption) std::cout << "Duree d'identification des coins : " << duree.count() << " secondes" << std::endl;
    Durees[1] += duree.count();
    ////////////// on a déterminé les coins //////////////////////
    if (printoption) {
      for (auto &moncoin : Coins) std::cout <<"coin " << moncoin.sommet<<std::endl;
      for (int i = 0; i < Coins.size(); i++)
        {
          std::cout << "coin " << i << " : " << Coins[i].sommet << std::endl;
        }
    }

    // déterminer la taille des cartes, proche de la taille indiquée dans la configuration
    // déterminer les probables bords de carte
    // deux coins sur une même ligne (ou deux ligne // proches), à distance vraissemblable (paramètre général de configuration)
    // la plus grande distance serait la hauteur de carte, sauf si plusieurs cartes sont alignées
    // une des autres devrait être dans le rapport des cotés de carte ( 3 / 2 )
    // 

    int htmax = 0; // hauteur maximale de carte, proche de la valeur dans la configuration
    int lamax = 0; // largeur maximale ....
    int ecartHt = maconf.hauteurcarte;
    int ecartLa = maconf.hauteurcarte;
    cv::Point2i P1, P2;
    //for (int n = 0; n < nbcoins; n++)
    for (int n = 0; n < Coins.size(); n++)
    {
      if (Coins[n].elimine) continue;

      cv::Vec4i l1 = Coins[n].l1->ln;
      cv::Vec4i l2 = Coins[n].l2->ln;
      cv::Point2i A = Coins[n].sommet;
      cv::Point2i H, K; // extremités non communes sur les deux lignes : coin AH,AK
      H = Coins[n].H;
      K = Coins[n].K;
      // rechercher les coins opposés de la carte du coin n
      for (int m = n + 1; m < Coins.size() ; m++) {
        if (Coins[m].elimine) continue; // coin éliminé
        cv::Vec4i l11 = Coins[m].l1->ln;
        cv::Vec4i l22 = Coins[m].l2->ln;
        cv::Point2i B(Coins[m].sommet);
        cv::Point2i HH(Coins[m].H);
        cv::Point2i KK(Coins[m].K);
        // une des lignes commune avec une de l'autre coin?
        // le coin B doit être sur une des lignes du coin A
        // le coin A doit être sur une des lignes du coin B
        // les deux autres lignes doivent être // et de même sens
        // AB semble alors etre un bord de carte
        //
        bool estoppose = false;
        float epsilon = maconf.deltacadre / 2;
        // première ligne du coin n contient le sommet B du coin m ?
        //float dist = B.x * lignes[i].a + B.y + lignes[i].b + lignes[i].c;
        float dist = B.x * Coins[n].l1->a + B.y + Coins[n].l1->b + Coins[n].l1->c;
        if (std::abs(dist) < epsilon) {
          // B proche de la première ligne du coin A
          // ligne de B // i ? produit vectoriel des normales
          float pv = Coins[n].l1->a * Coins[m].l1->b  - Coins[n].l1->b * Coins[m].l1->a;
          if (std::abs(pv) < maconf.deltaradian ) {
            // i et ii confondus
            // de sens opposé ? produit scalaire AH.BHH
            float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
            if (ps < 0) { // A et B opposés
              // vérifier que K et KK sont du même coté de la droite i = ii
              float d1 = Coins[n].l1->a * K.x + Coins[n].l1->b * K.y + Coins[n].l1->c;
              float d2 = Coins[n].l1->a * KK.x + Coins[n].l1->b * KK.y + Coins[n].l1->c;
              if (d1*d2 > 0){ //cotés non communs de même orientation
                  estoppose = true;
              } 
            }
          } else { // i et ii non //
            // i et jj confondus ?
            //float pv = lignes[i].a * lignes[jj].b  - lignes[i].b * lignes[jj].a;
            float pv = Coins[n].l1->a * Coins[m].l2->b  - Coins[n].l1->b * Coins[m].l2->a;
            if (std::abs(pv) < maconf.deltaradian ) {
              // i et jj confondus
              // de sens opposé ? produit scalaire AH.BKK
              float ps = (H.x - A.x)*(KK.x - B.x) + (H.y - A.y)*(KK.y - B.y);
              if (ps < 0) { // A et B opposés
                // vérifier que K et HH sont du même coté de la droite i = jj
                float d1 = Coins[n].l1->a * K.x + Coins[n].l1->b * K.y + Coins[n].l1->c;
                float d2 = Coins[n].l1->a * HH.x + Coins[n].l1->b * HH.y + Coins[n].l1->c;
                if (d1*d2 > 0){ //cotés non communs de même orientation
                    estoppose = true;
                } 
              }
            }
          }
        } else // B pas proche de la ligne i. proche de la ligne j ?
        // if (std::abs(B.x * lignes[j].a + B.y + lignes[j].b + lignes[j].c < epsilon)) { 
        if (std::abs(B.x * Coins[n].l2->a + B.y + Coins[n].l2->b + Coins[n].l2->c < epsilon)) { 
         // B proche de la droite j
          // ligne ii ou jj // j ? produit vectoriel des normales
          float pv = Coins[n].l2->a * Coins[m].l1->b  - Coins[n].l2->b * Coins[m].l1->a;
          if (std::abs(pv) < maconf.deltaradian ) {
            // j et ii confondus
            // de sens opposé ? produit scalaire AK.BHH
            float ps = (K.x - A.x)*(HH.x - B.x) + (K.y - A.y)*(HH.y - B.y);
            if (ps < 0) { // A et B opposés
              // vérifier que H et KK sont du même coté de la droite j = ii
              float d1 = Coins[n].l2->a*H.x + Coins[n].l2->b*H.y + Coins[n].l2->c;
              float d2 = Coins[n].l2->a*KK.x + Coins[n].l2->b*KK.y + Coins[n].l2->c;
              if (d1*d2 > 0){ //cotés non communs de même orientation
                estoppose = true;
              } 
            }
          } else { // j et ii non //
            // j et jj confondus ?
            float pv = Coins[n].l2->a * Coins[m].l2->a  - Coins[n].l2->b * Coins[m].l2->b;
            if (std::abs(pv) < maconf.deltaradian ) {
              // j et jj confondus
              // de sens opposé ? produit scalaire AH.BH
              float ps = (H.x - A.x)*(HH.x - B.x) + (H.y - A.y)*(HH.y - B.y);
              if (ps < 0) { // A et B opposés
                // vérifier que H et HH sont du même coté de la droite j = jj
                float d1 = Coins[n].l2->a*H.x + Coins[n].l2->b*H.y + Coins[n].l2->c;
                float d2 = Coins[n].l2->a*HH.x + Coins[n].l2->b*HH.y + Coins[n].l2->c;
                if (d1*d2 > 0){ //cotés non communs de même orientation
                    estoppose = true;
                } 
              }
            }
          }
        }
        // déterminer précisément la hauteur de carte, proche de la valeur dans la configuration
        if (estoppose) {
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
            // AB proche de la hauteur de carte ?
            int dl = std::abs(lg - maconf.largeurcarte);
            if (dl < maconf.deltacadre) {
              if (dl < ecartLa) {
                ecartLa = dl;
                lamax = lg;
                //P1 = A; P2 = B;
              }
            }
          }
          continue;
        }
      } // for m
      // if (htmax < 8 * maconf.hauteurcarte / 10) htmax = maconf.hauteurcarte;
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
        if (printoption) {
            std::cout << " !!!!! impossible d'estimer la taille des cartes" << std::endl;
            std::cout << " !!!!! poursuite avec la configuration " << std::endl;
        }
    }
    **************/
    //
     if (printoption)
        std::cout << "hauteur carte : " << maconf.hauteurcarte << std::endl;

    // TODO : pour chaque coin, rechercher les deux coins adjacents de la carte.
    //        créer les coins adjacents des lignes, même si une des deux est courte, correspondent

    ////////////////////////// éliminer les artefacts /////////////////////////////

    //

    // faire le tri parmi les coins détectés
    // pour chaque couple de coins P (n) et Q (m)
    //    éliminer le coin contenu dans l'autre, proche et //


    bool bwait = false;
    if (bwait) cv::waitKey(0);

    c = 0;
    for (int n = 0; n < Coins.size(); n++) {
      if (Coins[n].elimine) continue;
      cv::Vec4i l1 = Coins[n].l1->ln;
      cv::Vec4i l2 = Coins[n].l2->ln;
      cv::Point2i P(Coins[n].sommet);
      cv::Point2i R(Coins[n].H);
      cv::Point2i S(Coins[n].K);

      if (printoption){
        std::cout << "Coin " << n << " " << P << " , " << R << " , " << S << std::endl;
        if (Coins[n].numCarte > 0 ) std::cout<<" --> carte numero "<<Coins[n].numCarte<<std::endl;
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

        cv::Point2i Q = Coins[m].sommet;
        cv::Vec4i l11 = Coins[m].l1->ln;
        cv::Vec4i l22 = Coins[m].l2->ln;


        cv::Point2i U = Coins[m].H;
        cv::Point2i V = Coins[m].K;

        // coin  UQV

        cv::Point2i AA(l11[0], l11[1]);
        cv::Point2i BB(l11[2], l11[3]);
        cv::Point2i CC(l22[0], l22[1]);
        cv::Point2i DD(l22[2], l22[3]);
        // ignorer ce coin Q s'il n'est pas // coin P
        //
        double pv;
        bool estl22 = false;
        //if (i == ii)
        if (Coins[n].l1 == Coins[m].l1)
            pv = 0;
        else
            pv = Coins[n].l1->a * Coins[m].l1->b - Coins[n].l1->b * Coins[m].l1->a;
            // produit vectoriel des normales des lignes l1 des deux coins
        if (std::abs(pv) > maconf.deltaradian)
        { // AB  non // A'B'
          if (Coins[n].l1 == Coins[m].l2)
              pv = 0;
          else
              pv = Coins[n].l1->a * Coins[m].l2->b - Coins[n].l1->b * Coins[m].l2->a;
          if (std::abs(pv) > maconf.deltaradian)
              continue;  //  AB  non // C'D'
          estl22 = true; // AB // C'D'   et donc  CD // A'B'
        }
        //       déterminer si Q est proche d'une des lignes l1 (AB) ou l2 (CD)
        //       puis calculer la distance de Q à l'autre ligne
        //        hauteur ou largeur de carte ?
        bool memecarte = false;

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (Coins[n].elimine && Coins[m].elimine) continue;
        // TODO : erreur, il faut continuer pour déterminer si l'un est le cadre de l'autre
        //if (!Coins[n].elimine && !Coins[m].elimine)
        {
          float d1, d2, d3, d11, d22;
          int dc = std::max(5, maconf.deltacadre);
          int dc2 = 2*maconf.deltacadre;
          d1 = Coins[n].l1->dist(Q);
          d2 = Coins[n].l2->dist(Q);

          // distance de P à A'B' (=QU)  ou C'D' (=QV)
          d11 = Coins[m].l1->dist(P);
          d22 = Coins[m].l2->dist(P);
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
            if (printoption) std::cout << " coin "<< m << Q<< " opposé au coin "<< n 
             << P << " ecart "<< d3<<std::endl;
            // indiquer aussi si le premier coté du coin m est long ou court
            // rechercher si P est proche de A'B' ou de C'D'
            float dist = Coins[m].l1->dist(P);
            if (std::abs(dist) < dc2) { // P proche de A'B'
              // noter que le premier coté du coin Q est la longueur ou la largeur
              /**************************** ADAPTER 
              if (lgPQ > 5*maconf.hauteurcarte/6) { // coté long
                coins[m][10] = -3; 
              } else coins[m][10] = -2; // coté largeur
              ***********************/
            } else {
              dist = Coins[m].l2->dist(P);
              if (std::abs(dist) < dc2) { // P proche de C'D'
                /***************************ADAPTER
                if (lgPQ > 5*maconf.hauteurcarte/6) // coté largeur pour le premier coté du coin
                    coins[m][10] = -2; 
                else coins[m][10] = -3; // coté longeur pour le premier coté
                **************************/
              }
            }

            // si le coin m est déjà associé à un coin (<n) le coin n appartient à la même carte
            if (Coins[m].numCarte != 0) {
              Coins[n].numCarte = Coins[m].numCarte;
            } else if (Coins[n].numCarte != 0) {
              Coins[m].numCarte = Coins[n].numCarte;
            } else {
              nbcartes++;
              Coins[n].numCarte = Coins[m].numCarte = nbcartes;
            }
            if (printoption) std::cout<<" --> carte numero "<< Coins[n].numCarte<<std::endl;
          } else { // PQ n'est pas un bord de carte
            lgPQ = std::sqrt((Q.x - P.x)*(Q.x - P.x) + (Q.y - P.y)*(Q.y - P.y));
            if (lgPQ > 3*maconf.deltacadre) continue;
          }

        }

        // if (ii < 0 || jj < 0) continue; // coin Q éliminé


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
        if (Coins[m].numCarte != 0) {
          Coins[n].numCarte = Coins[m].numCarte;
        } else if (Coins[n].numCarte != 0) {
          Coins[m].numCarte = Coins[n].numCarte;
        } else {
          nbcartes++;
          Coins[n].numCarte = Coins[m].numCarte = nbcartes;
        }

        if (!Coins[m].elimine) { // Q pas encore éliminé
          bool elimQ = false;
          if (d >= 0 && d < dc + 2 * epsilon && dd >= 0 && dd < dc + 2 * epsilon)
              elimQ = true;

          if ((d >= -epsilon / 4 && dd >= epsilon / 4 && dd < dc + 2 * epsilon) 
          || (dd >= -epsilon / 4 && d >= epsilon && d < dc + 2 * epsilon))
              elimQ = true;
          if (elimQ)
          {
            // Q à l'intérieur du coin P
            // marquer le coin Q "éliminé"
            Coins[m].elimine = true;
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
          if (printoption && !eliminerP)
              std::cout << " --> elimination du coin " << n
                        << " dans le coin " << m << std::endl;
          eliminerP = true; // élimination différée
        }
        if (!Coins[m].elimine)
        { // Q pas encore éliminé
          // P est-il le sommet du cadre de Q ?
          // a distance deltacadre du coté négatif des droites du coin Q
          if ((dd < 0 && std::abs(dd + dc) <= epsilon) && (d < 0 && std::abs(d + dc) <= epsilon))
          {
            // P est le sommet du cadre du coin Q
            Coins[m].estunRDV = true;
            Coins[m].cadre = P;
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
          Coins[n].estunRDV = true;
          Coins[n].cadre = Q;
          Coins[m].elimine = true; 

          continue;
        }

        if (Coins[m].elimine) continue;

        // Q n'est pas sur le cadre de P
      } // for m

      // élimination différée de P ?
      if (eliminerP)
      { // c'est peut-être déjà fait
        if (!Coins[n].elimine)
        {
          if (printoption)
              std::cout << "elimination coin " << n << std::endl;
          Coins[n].elimine = true;
        }
      } else if (Coins[n].numCarte == 0) { // pas encore affecté à une carte
        nbcartes++;
        Coins[n].numCarte = nbcartes; // nouvelle carte
        if (printoption) std::cout<<" --> nouvelle carte "<<nbcartes<<" pour le coin "<<n<<std::endl;
      }
      c++;
      c--; // pour pouvoir mettre un point d'arrêt
    } // for n

    // on a obtenu tous les coins et les cartes.
    // certains coins sont identifiés comme personnages (R D V) car contenant un cadre

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
    const std::string nomval[14] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "V", "D", "R"};

    if (estvideo) {
      traiterCartes(image, maconf, Coins, coinsPrec,
         pnbcoins, lescoins, lignes, monpli);
      }


    // afficher ce qui reste selectionné
    cv::Mat imaC = ima2.clone();
    //********************** fond noir pour ne voir que les lignes des coins
    for (int y = 0; y < imaC.rows; y++)
      for (int x = 0; x < imaC.cols; x++)
          imaC.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // fond noir
    /***************/
    if (estvideo){
      // éliminer les coins de la frame précédente qui ne sont pas dans celle-ci
      int dc = std::max(2, maconf.deltacadre);
      // TODO : éliminer les coins du vecteur coinsPrec qui ne sont pas dans cette frame
      for (auto it=coinsPrec.begin() ; it != coinsPrec.end();  ) {
        bool trouve = false;
        uncoinPrec up = *it;
        cv::Point2i Q (up.x, up.y);
        for (int n = 0; n < Coins.size(); n++){
          if (Coins[n].elimine) continue; // coin éliminé
          cv::Point2i P (Coins[n].sommet);
          // proche ?
          if (std::abs(P.x - Q.x) <= dc && std::abs(P.y - Q.y) <= dc ) {
            trouve = true;
            break;
          }
        }
        if (!trouve) {
          it = coinsPrec.erase(it);
        } else it++;
      }
    }

    // afficher les coins
  if (printoption) {
    c = 0;
    for (int n = 0; n < Coins.size(); n++) {
      int cc = Coins[n].numCarte; // numéro de carte
      while (cc >= NBCOULEURS) cc -= NBCOULEURS;
      cv::Point P(Coins[n].sommet);

      cv::Vec4i l1 = Coins[n].l1->ln;
      cv::Vec4i l2 = Coins[n].l2->ln;

      cv::Point2i A(l1[0], l1[1]);
      cv::Point2i B(l1[2], l1[3]);
      cv::Point2i C(l2[0], l2[1]);
      cv::Point2i D(l2[2], l2[3]);
      cv::Vec4i nl1(A.x, A.y, B.x, B.y);
      cv::Vec4i nl2(C.x, C.y, D.x, D.y);

      // !!!! uniquement sur les copies

      // remplacer l'extremité qui convient par l'intersection
      // TODO : adapter
      /**********************************
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
      **************************/
      if (estvideo){
        // si ce coin était trouvé dans la frame précédente, inutile de le considérer
        if (Coins[n].couleur >= 0 && Coins[n].valeur > 0) {
          if (printoption) std::cout<<"coin "<< n << " identifié, couleur:"
                <<Coins[n].couleur<<", valeur:"<<Coins[n].valeur<<std::endl;
          Coins[n].elimine = true;
        } else for (auto up : coinsPrec){
          if (std::abs (P.x - up.x) <= maconf.deltacadre 
          && std::abs (P.y - up.y) <= maconf.deltacadre ) {
            // déjà trouvé dans la précédente frame
            Coins[n].couleur = up.couleur;
            Coins[n].valeur = up.valeur;
            Coins[n].elimine = true;
            if (printoption) std::cout<<"coin "<< n << " dans une frame précédente carte couleur:"
              <<up.couleur<<", valeur:"<<up.valeur<<std::endl;
            break;
          }
        }
      }

      if (Coins[n].elimine   /*|| (estvideo && coins[n][10] >= 0 && coins[n][11]> 0)*/ )
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
      if (Coins[n].estunRDV)
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
      std::cout << "probable hauteur de carte : " << htmax << std::endl;
      cv::circle(imaC, P1, 6, cv::Scalar(0, 128, 128), 4);
      cv::circle(imaC, P2, 6, cv::Scalar(0, 128, 128), 4);
    }
    cv::imshow("coins", imaC);
    cv::imshow("grise", grise);
  } //if(printoption)

    // cv::waitKey(0);
    //  extraire les coins
    //
    bool estunRDV;
    estunRDV = false;       // le coin contient-il un cadre ?
    cv::Point2i Q;          // point du cadre
    std::string cartes[50]; // cartes trouvées
    //nbcartes = 0;  // 

    auto t1 = std::chrono::high_resolution_clock::now();
    duree = t1 - t33;
    if (printoption) std::cout << "Temps préparatoire : " << duree.count() << " secondes" << std::endl
      << "============================"<< std::endl;
    Durees[2] += duree.count();
    result = image.clone();
    cv::imshow("result", result);
    int szPrec = 0;


// TODO : vérifier que l'on obtient le même résultat pour tous les coins d'une même carte
//        a posteriori après traitement multithread


    const std::string valeurcarte[14]  = {" ", "1","2", "3", "4", "5", "6", "7", "8", "9", "10", "V", "D", "R"};
    const std::string couleurcarte[4]  = {"P", "C", "K", "T"}; 

  if (!estvideo ||  maconf.coinsoption > 0) {    // traiter aussi les coins isolés
    for (int n = 0; n < nbcoins; n++) {
      if (Coins[n].elimine || (estvideo && Coins[n].couleur >= 0 && Coins[n].valeur> 0) )
          continue; // coin éliminé ou déjà analysé dans une carte
      int l1W[4], l2W[4];
      // TODO : éliminer le coin s'il est sur un bord d'une carte déjà analysée



      cv::Vec4i l1 = Coins[n].l1->ln;
      cv::Vec4i l2 = Coins[n].l2->ln;
      for (int i = 0; i < 4; i++)
      {
        l1W[i] = l1[i];
        l2W[i] = l2[i];
      }
      if (printoption)
          std::cout << std::endl
                    << "coin " << n << "   ";
      std::string cartelue;

      const int *p = 0; // &coins[0][0];
      if (threadoption == 0) { // pas de sous-tache
        if (Coins[n].couleur < 0 || Coins[n].valeur <=0) // valeur non encore déterminée
            traiterCoin(n, Coins, image, resultats,
                    result, &l1W[0], &l2W[0], maconf);
        if (Coins[n].valeur != 0 && Coins[n].couleur >= 0) // valeur trouvée
        {
          if (!estvideo) cv::imshow("result", result);
          cv::Point2i PT(Coins[n].sommet);
          std::string resW = couleurcarte[Coins[n].couleur] + valeurcarte[Coins[n].valeur];
          std::string res = resW + "#";

          afficherResultat(result, PT, res);
          if (waitoption > 1)
              cv::waitKey(0);
          else cv::waitKey(1);
        }
      }
      else // sous-taches
      if (Coins[n].couleur < 0 || Coins[n].valeur <=0) // valeur non encore déterminée
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
    if (printoption) std::cout << "Temps écoulé : " << elapsed.count() << " secondes" << std::endl;
    Durees[3] += elapsed.count();

    // TODO:
    // si on traite une vidéo, ajouter les 4 sommet de la dernière carte ajoutée au pli en cours
    //   aux coins de la frame précédente (donc celle qu'on est en train de construire) 
    for (int i=0; i<4;i++){
      int x = monpli.cartes[nbcartes-1].sommet[i].x;
      int y = monpli.cartes[nbcartes-1].sommet[i].y;
      // rechercher s'il est présent dans les coins précédents (de la frame précédente)
      bool trouve = false;
      for (auto up : coinsPrec){
        if (x == up.x && y == up.y) { trouve = true; break;}
      }
      if (!trouve) {
        uncoinPrec up;
        up.couleur = monpli.cartes[nbcartes-1].couleur;
        up.valeur = monpli.cartes[nbcartes-1].valeur;
        up.x = x;
        up.y = y;
        coinsPrec.push_back(up);
      }
    }


    // cv::imshow("result", result);
    // Affichage des résultats après synchronisation
    // les résultats sont dans le tableau des coins 
    // afficher un résultat pour chaque carte
    bool nouveaucoin = false;
    for (int nc = 1; nc <= nbcartes; nc++) {
      bool premier = true;
      int cc1(-1), vc1(0);
      // TODO : si on traite une vidéo, trouver tous les coins de la carte nc
      //    compléter avec les sommets manquants
      //    mémoriser la carte avec ses 4 sommets dans la frame précédente
      for (int n = 0; n < Coins.size(); n++){
        if (nc != Coins[n].numCarte) continue; // pas la carte nc

        cv::Point2i PT(Coins[n].sommet);
        if (!estvideo) { // on ne traite pas une video
            if (Coins[n].elimine) continue; //coin éliminé
            if (Coins[n].valeur == 0) continue; // valeur de carte non trouvée
        
            // pas vidéo et coin non éliminé et valeur de carte trouvée et carte en cours
            if (!Coins[n].elimine && Coins[n].valeur > 0 && nc == Coins[n].numCarte){
                if (premier) {cc1 = Coins[n].couleur; vc1 = Coins[n].valeur;}
                std::string resW = couleurcarte[Coins[n].couleur] + valeurcarte[Coins[n].valeur];
                std::string res = resW + "#";
                afficherResultat(result, PT, res);
                if (premier) cartes[nc - 1] = resW;
                else if (cc1 != Coins[n].couleur || vc1 != Coins[n].valeur) {
                    // incohérence. quelle détection est fausse?
                    std::cout<< "détection incohérente " << resW << " carte "<< cartes[nc - 1] <<std::endl; 
                }
                premier = false;
            }
            //
            // si on traite une vidéo, ajouter les coins détectés ou analysés
        } else { // on traite une video
          // rechercher si le coin est déjà dans le vecteur coinsPrec
          bool trouve(false);
          
          for (auto up:coinsPrec){
            if (std::abs(PT.x - up.x) < 2 && std::abs(PT.y - up.y) < 2 ){
              trouve = true;
              cc1 = up.couleur;
              vc1 = up.valeur;
              break;
            }
          }
          if (!trouve) {
            int c = Coins[n].couleur;
            int v = Coins[n].valeur;
            nouveaucoin = true;
            cc1 = c; vc1 = v;
            uncoinPrec up;
            up.couleur = c;
            up.valeur = v;
            up.x = PT.x;
            up.y = PT.y;
            coinsPrec.push_back(up);
          }
        }
      } // for(n) coins
      // valoriser tous les coins de cette carte si la valeur est connue
      if (cc1 >= 0 && vc1 > 0 ){
        for (int n = 0; n < Coins.size(); n++){
          if (Coins[n].numCarte != nc) continue;
          Coins[n].couleur = cc1;
          Coins[n].valeur = vc1;
        }
      }
    } // for(nc) cartes
    // cv::imshow("result", result); // désactivé en multitache
    if (!estvideo) { cv::imshow("result", result); cv::waitKey(1);}

    // si on traite une vidéo, les coins trouvés précédemment ou maintenant
    //    sont dans le vecteur coinsPrec
    // on affiche les valeurs trouvées
    // on reconstitue alors le tableau des cartes
    bool nouvellecarte = false;
    if (estvideo){
      nbcartes = 0;
      for (auto up : coinsPrec){
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
        for (i=0; i < nbcartes; i++){
          if (nomcol == cartes[i][0] && val == cartes[i].substr(1)) break;
        }
        if (i == nbcartes){  // nouvelle carte du pli en cours
          cartes[i] = nomcol + val;
          nbcartes++;
          nouvellecarte = true;
        }
      }

      cv::imshow("result", result); cv::waitKey(1);
    } // estvideo

    // afficher le résultat de la dernière image contenant au moins une carte
    bool aUneCarte = false;
    for (int i = 0; i < nbcartes; i++)
    {
      if(cartes[i].size() < 2) continue;
      aUneCarte = true;
      break;
    }
    if (aUneCarte) {
      cv::imshow("complet", result); cv::waitKey(1);
    }
    if (printoption) {
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
    }

    if (waitoption && !estvideo) cv::waitKey(0);
    if (waitoption > 1 && estvideo && nouveaucoin) cv::waitKey(0);
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
