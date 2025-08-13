#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <sqlite3.h>

using namespace std;
using namespace cv;

// Dimensions des cartes
const int CARD_WIDTH = 200;   // à adapter selon l'image
const int CARD_HEIGHT = 300;  // à adapter selon l'image

// Ordre des couleurs dans l'image
map<char, int> suitIndex = {
    {'T', 0}, // Trèfle
    {'K', 1}, // Carreau
    {'C', 2}, // Cœur
    {'P', 3}  // Pique
};

// Ordre des valeurs dans l'image
vector<string> ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "V", "D", "R", "A"};

int getRankIndex(const string& rank) {
    for (size_t i = 0; i < ranks.size(); ++i)
        if (ranks[i] == rank) return i;
    return -1;
}

Mat extractCard(const Mat& sprite, const string& code) {
    if (code.size() < 2) return Mat();

    char suit = code[0];
    string rank = code.substr(1);
    int suitIdx = suitIndex[suit];
    int rankIdx = getRankIndex(rank);

    if (suitIdx < 0 || rankIdx < 0) return Mat();

    //Rect roi(rankIdx * CARD_WIDTH, suitIdx * CARD_HEIGHT, CARD_WIDTH, CARD_HEIGHT);
    Rect roi((rankIdx + 13*suitIdx) * CARD_WIDTH, 0, CARD_WIDTH, CARD_HEIGHT);
    return sprite(roi).clone();
}

void showPli(const Mat& sprite, const vector<pair<string, Point>>& cartes) {
    Mat canvas(800, 1200, CV_8UC3, Scalar(0, 128, 0)); // fond vert

    for (const auto& [code, pos] : cartes) {
        Mat card = extractCard(sprite, code);
        if (card.empty()) continue;

        if (pos.x == 0 || pos.x == canvas.cols - CARD_HEIGHT) {
            // Est/Ouest : rotation 90°
            rotate(card, card, ROTATE_90_CLOCKWISE);
        }

        Rect roi(pos.x, pos.y, card.cols, card.rows);
        card.copyTo(canvas(roi));
        imshow("Pli", canvas);
        waitKey(2000); // 2 secondes
    }

    cout << "Appuyez sur une touche pour continuer..." << endl;
    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: ./bridge_viewer <nom_table> <numero_donne>" << endl;
        return 1;
    }

    string nomTable = argv[1];
    int numeroDonne = stoi(argv[2]);

    // Charger sprite
    Mat sprite = imread("cards_2_fr.png");
    if (sprite.empty()) {
        cerr << "Erreur chargement image des cartes." << endl;
        return 1;
    }

    // Connexion DB
    sqlite3* db;
    if (sqlite3_open("bridge.db", &db) != SQLITE_OK) {
        cerr << "Erreur ouverture DB." << endl;
        return 1;
    }

    // Récupérer ID table
    int tableId = -1;
    string sqlTable = "SELECT id FROM tables WHERE nom = ?";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, sqlTable.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, nomTable.c_str(), -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        tableId = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    if (tableId == -1) {
        cerr << "Table non trouvée." << endl;
        sqlite3_close(db);
        return 1;
    }

    // Récupérer ID donne
    int donneId = -1;
    string sqlDonne = "SELECT id FROM donnes WHERE numero = ?";
    sqlite3_prepare_v2(db, sqlDonne.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, numeroDonne);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        donneId = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    if (donneId == -1) {
        cerr << "Donne non trouvée." << endl;
        sqlite3_close(db);
        return 1;
    }

    // Récupérer ID contrat
    int contratId = -1;
    string sqlContrat = "SELECT id FROM contrats WHERE table_id = ? AND donne_id = ?";
    sqlite3_prepare_v2(db, sqlContrat.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, tableId);
    sqlite3_bind_int(stmt, 2, donneId);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        contratId = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    if (contratId == -1) {
        cerr << "Contrat non trouvé." << endl;
        sqlite3_close(db);
        return 1;
    }

    // Récupérer les plis
    string sqlPlis = R"(
        SELECT numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur
        FROM plis WHERE contrat_id = ? ORDER BY numero ASC
    )";
    sqlite3_prepare_v2(db, sqlPlis.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, contratId);
    





    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* joueurCStr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        std::string joueur = joueurCStr ? joueurCStr : "";
        std::pair<const char *, cv::Point2i> cart[4];
        cart[0] = std::pair<const char *, cv::Point2i> ({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)), Point(250, 20)});
        cart[1] = std::pair<const char *, cv::Point2i> ({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)), Point(500, 150)});
        cart[2] = std::pair<const char *, cv::Point2i> ({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)), Point(250, 280)});
        cart[3] = std::pair<const char *, cv::Point2i> ({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)), Point(20, 150)});
        int j = 0;
        if (joueur == "Est") j = 1;
        else if (joueur == "Sud") j = 2;
        else if (joueur == "Ouest") j = 3;
        vector<pair<string, Point>> cartes;
        cartes.push_back(cart[j]);
        j++; if (j>3) j = 0; cartes.push_back(cart[j]);
        j++; if (j>3) j = 0; cartes.push_back(cart[j]);
        j++; if (j>3) j = 0; cartes.push_back(cart[j]);

        //cartes.push_back({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)), Point(250, 20)});   // Nord
        //cartes.push_back({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)), Point(500, 150)}); // Est
        //cartes.push_back({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)), Point(250, 280)}); // Sud
        //cartes.push_back({reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)), Point(20, 150)});  // Ouest

        showPli(sprite, cartes);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return 0;
}

