#define POSTGRESQL
#include <opencv2/opencv.hpp>

#ifdef POSTGRESQL
#include <pqxx/pqxx>
#else
#include <sqlite3.h>
#endif

#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

const int CARD_WIDTH = 200;
const int CARD_HEIGHT = 300;

std::string joueurs[4] = {"Nord", "Est", "Sud", "Ouest"};

#ifdef POSTGRESQL
    pqxx::connection openDatabase(string conninfo){
    try {
        pqxx::connection conn("dbname=bridge user=jro password=jro");
        if (!conn.is_open()) {
            cerr << "Erreur ouverture base PostgreSQL\n";
            exit(1);
        }
        return conn;
    } catch (const std::exception& e) {
        cerr << "Exception PostgreSQL : " << e.what() << endl;
        exit(1);
    }
  }
#else
sqlite3* openDatabase(const string& filename) {
    sqlite3* db;
    if (sqlite3_open(filename.c_str(), &db) != SQLITE_OK) {
        cerr << "Erreur ouverture base : " << sqlite3_errmsg(db) << endl;
        exit(1);
    }
    return db;
}
#endif

// obtenir une carte à partir du bandeau qui contient les 52 cartes de bridge
Mat extractCard(const Mat& sprite, const string& code) {
     if (code.size() < 2) return Mat();
    string couleurs = "TKCP";
    string valeurs = "23456789XVDRA";
    char c = code[1]; if (code.substr(1) == "10") c = 'X';
    int col = valeurs.find(c);
    int row = couleurs.find(code[0]);
    if (row == -1 || col == -1) return Mat();
    Rect roi((13*row +col) * CARD_WIDTH, 0, CARD_WIDTH, CARD_HEIGHT);
    return sprite(roi).clone();
}

class CarteModifiable {
public:
    string code; // couleur (PCKT) et valeur (2 à A)
    Point position;
    Rect zone;
    bool modifiee = false;

    CarteModifiable(string c, Point p) : code(c), position(p) {}
};

class Pli {
public:
    int numero;
    string joueur; // le joueur qui entame le pli
    vector<CarteModifiable> cartes; // les 4 cartes du pli

    Pli(int num, string j, vector<CarteModifiable> cs)
        : numero(num), joueur(j), cartes(cs) {}
};

class ChargeurPlis {
public:
#ifdef POSTGRESQL
    static vector<Pli> charger(pqxx::connection& conn, int contratId) {
        vector<Pli> plis;
        pqxx::work txn(conn);
        pqxx::result r = txn.exec_params(
            "SELECT numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur "
            "FROM plis WHERE contrat_id = $1 ORDER BY numero ASC", contratId);

        for (const auto& row : r) {
            int num = row["numero"].as<int>();
            string joueur = row["joueur"].as<string>();
            vector<CarteModifiable> cartes = {
                {row["carte_nord"].as<string>(), Point(250, 20)},
                {row["carte_est"].as<string>(), Point(500, 150)},
                {row["carte_sud"].as<string>(), Point(250, 280)},
                {row["carte_ouest"].as<string>(), Point(20, 150)}
            };
            plis.emplace_back(num, joueur, cartes);
        }
        txn.commit();
        return plis;
    }
#else
    static vector<Pli> charger(sqlite3* db, int contratId) {
        vector<Pli> plis;
        string sql = R"(SELECT numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur
                        FROM plis WHERE contrat_id = ? ORDER BY numero ASC)";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, contratId);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int num = sqlite3_column_int(stmt, 0);
            string joueur = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
            vector<CarteModifiable> cartes = {
                {reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)), Point(250, 20)},
                {reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)), Point(500, 150)},
                {reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)), Point(250, 280)},
                {reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)), Point(20, 150)}
            };
            plis.emplace_back(num, joueur, cartes);
        }
        sqlite3_finalize(stmt);
        return plis;
    }
#endif
};



class AfficheurPli {
    Mat sprite;
    vector<CarteModifiable>* cartesAffichees;
    CarteModifiable* carteSelectionnee = nullptr;
    vector<CarteModifiable> bandeauCartes;
    std::vector<Pli> *plis;
    bool estModifie() {
      bool modif = false;
      for( auto& pli : *plis){
        for (auto& carte : pli.cartes){
          if (carte.modifiee ) {modif = true; break;}
        }
        if (modif) break;
      }
      return modif;
    }
public:
    Rect zoneQuitter;
    Rect zoneMaj;
    bool quitterDemande = false;
    bool majDemande = false;
    CarteModifiable* carteBandeauSelectionnee = nullptr;
    CarteModifiable* cartePliSelectionnee = nullptr;

    AfficheurPli(const Mat& s) : sprite(s) {
      initialiserBandeau();
    }
    void initialiserBandeau();
    void afficher(std::vector<Pli> plis, Pli& pli); // afficher un pli
    void gererClic(int x, int y);
};

void AfficheurPli::afficher(std::vector<Pli> plis, Pli& pli) {
    this->plis = &plis;  // vecteur de tous les plis
    Mat canvas(800, 1296, CV_8UC3, Scalar(0, 128, 0));
    putText(canvas, "Pli " + to_string(pli.numero), Point(20, 40),
      FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    // 🔳 Bouton Quitter
    zoneQuitter = Rect(20, 60, 120, 40);
    rectangle(canvas, zoneQuitter, Scalar(200, 200, 200), FILLED);
    putText(canvas, "Quitter", Point(30, 90),
    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    // 🔳 Bouton Enregistrer
    if(estModifie()) {
      zoneMaj = Rect(20, 110, 150, 40);
      rectangle(canvas, zoneMaj, Scalar(0, 0, 255), FILLED);
      putText(canvas, "Enregistrer", Point(30, 140),
        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    }
    cartesAffichees = const_cast<vector<CarteModifiable>*>(&pli.cartes);

    //for (int i = 0; i < 4; i++) {
    //  CarteModifiable carte = pli.cartes[i];
    int i=0;
    for (auto& carte : pli.cartes) {
        Mat card = extractCard(sprite, carte.code);
        if (card.empty()) continue;
        carte.zone = Rect(carte.position.x, carte.position.y, card.cols, card.rows);
        card.copyTo(canvas(carte.zone));
        if (&carte == cartePliSelectionnee)
            rectangle(canvas, carte.zone, Scalar(255, 0, 0), 3); // bleue
        else if (carte.modifiee)
            rectangle(canvas, carte.zone, Scalar(0, 0, 255), 3); // rouge
        else if (pli.joueur == joueurs[i])
            rectangle(canvas, carte.zone, Scalar(0, 255, 0), 2); // vert
        i++;
    }        
    for (auto& carte : bandeauCartes) {
        Mat card = extractCard(sprite, carte.code);
        if (card.empty()) continue;
        resize(card, card, Size(CARD_WIDTH / 5, CARD_HEIGHT / 5));
        carte.zone = Rect(carte.position.x, carte.position.y, card.cols, card.rows);
        card.copyTo(canvas(carte.zone));
        if (&carte == carteBandeauSelectionnee)
            rectangle(canvas, carte.zone, Scalar(255, 0, 0), 2); // bordure bleue
    }

    imshow("Pli", canvas);
}

void AfficheurPli::gererClic(int x, int y) {
    if (zoneQuitter.contains(Point(x, y))) { quitterDemande = true; return; }      
    if (zoneMaj.contains(Point(x, y))) {
      majDemande = true;
      return; 
    }      
    for (auto& carte : *cartesAffichees) {
        if (carte.zone.contains(Point(x, y))) {
            cartePliSelectionnee = &carte;
            carte.modifiee = false;
            carteBandeauSelectionnee = nullptr;
            return;
          }
    }
    // Clic sur carte du bandeau
    for (auto& carte : bandeauCartes) {
        if (carte.zone.contains(Point(x, y))) {
            carteBandeauSelectionnee = &carte;
            if (cartePliSelectionnee) {
                cartePliSelectionnee->code = carte.code;
                cartePliSelectionnee->modifiee = true;
            }
            return;
        }
    }        
}

void AfficheurPli::initialiserBandeau() {
    bandeauCartes.clear();
    vector<string> valeurs = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "V", "D", "R", "A"};
    int scale = 5;
    int w = CARD_WIDTH / scale;
    int h = CARD_HEIGHT / scale;
    int xStart = 50;
    int yTop = 600;
    int yBottom = yTop + h + 10;

    // Ligne du haut : Trèfle et Carreau
    for (int i = 0; i < valeurs.size(); ++i) {
        bandeauCartes.emplace_back("T" + valeurs[i], Point(xStart + i * (w + 5), yTop));
        bandeauCartes.emplace_back("K" + valeurs[i], Point(xStart + (i + 13) * (w + 5), yTop));
    }

    // Ligne du bas : Cœur et Pique
    for (int i = 0; i < valeurs.size(); ++i) {
        bandeauCartes.emplace_back("C" + valeurs[i], Point(xStart + i * (w + 5), yBottom));
        bandeauCartes.emplace_back("P" + valeurs[i], Point(xStart + (i + 13) * (w + 5), yBottom));
    }
}

class ApplicationPli {
    std::vector<Pli> plis;
    int pliCourant = 0;
    AfficheurPli afficheur;
    int contratId;
    pqxx::connection* pconn;
    void enregistrerPlisModifiés();

public:
    void run();
#ifdef POSTGRESQL
    ApplicationPli(pqxx::connection& conn, int contratId, const Mat& sprite)
        : afficheur(sprite) , contratId(contratId), pconn(&conn) {
        plis = ChargeurPlis::charger(conn, contratId);
    }
#else
    ApplicationPli(sqlite3* db, int contratId, const Mat& sprite)
        : afficheur(sprite) {
        plis = ChargeurPlis::charger(db, contratId);
    }
#endif

    static void onMouseStatic(int event, int x, int y, int, void* userdata) {
        if (event != EVENT_LBUTTONDOWN) return;
        auto* app = static_cast<ApplicationPli*>(userdata);
        app->afficheur.gererClic(x, y);
        if(app->afficheur.majDemande) {
          app->enregistrerPlisModifiés();
          app->afficheur.majDemande = false;
        }
        app->afficheur.afficher(app->plis, app->plis[app->pliCourant]);
    }
  };

void ApplicationPli::run() {
      namedWindow("Pli", WINDOW_AUTOSIZE); // ✅ Crée la fenêtre avant le callback
    setMouseCallback("Pli", onMouseStatic, this);
    while (true) {
        if (afficheur.majDemande) { afficheur.majDemande = false; enregistrerPlisModifiés();}
        if (afficheur.quitterDemande) break;
        afficheur.afficher(plis, plis[pliCourant]);
        int key = waitKey(0);
        if (key == 27) break;
        else if (key == 32) {
          pliCourant = (pliCourant + 1) % plis.size();
          afficheur.cartePliSelectionnee = nullptr;
          afficheur.carteBandeauSelectionnee = nullptr;
        }
        else if (key == '-' || key == 81)
            pliCourant = (pliCourant - 1 + plis.size()) % plis.size();
    }
}
void ApplicationPli::enregistrerPlisModifiés() {
#ifdef POSTGRESQL
try {
  pqxx::work txn(*pconn);

  for (Pli& pli : plis) {
      bool modif = false;
      for (auto& carte : pli.cartes) {
          if (carte.modifiee) {
            carte.modifiee = false;
              modif = true;
              // break;
          }
      }
      if (!modif) continue;

      txn.exec_params(
          "UPDATE plis SET carte_nord = $1, carte_est = $2, carte_sud = $3, carte_ouest = $4, joueur = $5 "
          "WHERE contrat_id = $6 AND numero = $7",
          pli.cartes[0].code,
          pli.cartes[1].code,
          pli.cartes[2].code,
          pli.cartes[3].code,
          pli.joueur,
          this->contratId,
          pli.numero
      );
  }

  txn.commit();
} catch (const std::exception& e) {
  std::cerr << "Erreur PostgreSQL : " << e.what() << std::endl;
}

#else
sqlite3_stmt* stmt;
  for (const Pli& pli : plis) {
    bool modif = false;
    for (const auto& carte : pli.cartes) {
        if (carte.modifiee) {
            modif = true;
            break;
        }
    }
    if (!modif) continue;

    const char* sql = "UPDATE plis SET carte_nord = ?, carte_est = ?, carte_sud = ?, carte_ouest = ?, joueur = ? "
                      "WHERE contrat_id = ? AND numero = ?";

    if (sqlite3_prepare_v2(afficheur.db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, pli.cartes[0].code.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, pli.cartes[1].code.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, pli.cartes[2].code.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, pli.cartes[3].code.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, pli.joueur.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 6, contratId);
        sqlite3_bind_int(stmt, 7, pli.numero);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Erreur SQLite : " << sqlite3_errmsg(afficheur.db) << std::endl;
        }
    }
    sqlite3_finalize(stmt);
  }
#endif
}


int main(int argc, char** argv) {

    string nomTable = "test";
    int numeroDonne = 1;

    if (argc >= 2) nomTable = argv[1];
    if (argc >= 3) numeroDonne = stoi(argv[2]);

    Mat sprite = imread("cards_2_fr.png");
    if (sprite.empty()) {
        cerr << "Erreur chargement cartes.png" << endl;
        return 1;
    }

#ifdef POSTGRESQL
    string conninfo = "dbname=bridge user=jro";
    try {
      pqxx::connection conn = openDatabase(conninfo);

      pqxx::work txn(conn);
      // 🔍 Récupérer ID table
      int tableId = -1;
      pqxx::result r1 = txn.exec_params("SELECT id FROM tables WHERE nom = $1", nomTable);
      if (!r1.empty()) tableId = r1[0][0].as<int>();
      else {
          cerr << "Table non trouvée : " << nomTable << endl;
          return 1;
      }

      // 🔍 Récupérer ID donne
      int donneId = -1;
      pqxx::result r2 = txn.exec_params("SELECT id FROM donnes WHERE numero = $1", numeroDonne);
      if (!r2.empty()) donneId = r2[0][0].as<int>();
      else {
          cerr << "Donne non trouvée : " << numeroDonne << endl;
          return 1;
      }

      // 🔍 Récupérer ID contrat
      int contratId = -1;
      pqxx::result r3 = txn.exec_params(
          "SELECT id FROM contrats WHERE table_id = $1 AND donne_id = $2", tableId, donneId);
      if (!r3.empty()) contratId = r3[0][0].as<int>();
      else {
          cerr << "Contrat non trouvé." << endl;
          return 1;
      }
          txn.commit();
          ApplicationPli app(conn, contratId, sprite);
          app.run();
    } catch (const std::exception& e) {
        std::cerr << "Erreur PostgreSQL : " << e.what() << std::endl;
        return 1;
    }    
#else
    sqlite3* db = openDatabase("bridge.db");
    // 🔍 Récupérer ID table
    int tableId = -1;
    sqlite3_stmt* stmt;
    string sqlTable = "SELECT id FROM tables WHERE nom = ?";
    sqlite3_prepare_v2(db, sqlTable.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, nomTable.c_str(), -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        tableId = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    if (tableId == -1) {
        cerr << "Table non trouvée : " << nomTable << endl;
        sqlite3_close(db);
        return 1;
    }

    // 🔍 Récupérer ID donne
    int donneId = -1;
    string sqlDonne = "SELECT id FROM donnes WHERE numero = ?";
    sqlite3_prepare_v2(db, sqlDonne.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, numeroDonne);
    if (sqlite3_step(stmt) == SQLITE_ROW)
        donneId = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    if (donneId == -1) {
        cerr << "Donne non trouvée : " << numeroDonne << endl;
        sqlite3_close(db);
        return 1;
    }

    // 🔍 Récupérer ID contrat
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
    ApplicationPli app(db, contratId, sprite);
    app.run();
    sqlite3_close(db);
#endif
    return 0;
}
