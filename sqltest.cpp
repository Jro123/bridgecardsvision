#include <iostream>
#include <sqlite3.h>
#include <string>

// Fonction pour exécuter une requête SQL simple
bool executeSQL(sqlite3* db, const std::string& sql) {
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Erreur SQL: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}

// Fonction pour insérer un pli
void insererPli(sqlite3* db, int contrat_id, int numero,
                const std::string& nord, const std::string& est,
                const std::string& sud, const std::string& ouest,
                const std::string& joueur) {
    std::string sql = "INSERT INTO plis (contrat_id, numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur) VALUES (" +
                      std::to_string(contrat_id) + ", " + std::to_string(numero) + ", '" +
                      nord + "', '" + est + "', '" + sud + "', '" + ouest + "', '" + joueur + "');";
    executeSQL(db, sql);
}

// Fonction pour modifier une carte dans un pli
void modifierCarte(sqlite3* db, int pli_id, const std::string& direction, const std::string& nouvelle_carte) {
    std::string colonne = "carte_" + direction;
    std::string sql = "UPDATE plis SET " + colonne + " = '" + nouvelle_carte + "' WHERE id = " + std::to_string(pli_id) + ";";
    executeSQL(db, sql);
}

// Fonction pour lire les plis d’un contrat
void lirePlis(sqlite3* db, int contrat_id) {
    std::string sql = "SELECT numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur FROM plis WHERE contrat_id = " +
                      std::to_string(contrat_id) + " ORDER BY numero ASC;";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Erreur de préparation: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    std::cout << "Plis du contrat " << contrat_id << " :\n";
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int numero = sqlite3_column_int(stmt, 0);
        std::string nord = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        std::string est = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        std::string sud = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        std::string ouest = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        std::string joueur = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));

        std::cout << "Pli " << numero << " : N=" << nord << " E=" << est << " S=" << sud << " O=" << ouest << " | Joueur: " << joueur << "\n";
    }

    sqlite3_finalize(stmt);
}

int main() {
    sqlite3* db;
    if (sqlite3_open("bridge.db", &db) != SQLITE_OK) {
        std::cerr << "Impossible d’ouvrir la base de données.\n";
        return 1;
    }

    // Création de la table si elle n’existe pas
    std::string createTableSQL = R"(
        CREATE TABLE IF NOT EXISTS plis (
            id INTEGER PRIMARY KEY,
            contrat_id INTEGER,
            numero INTEGER,
            carte_nord TEXT,
            carte_est TEXT,
            carte_sud TEXT,
            carte_ouest TEXT,
            joueur TEXT,
            FOREIGN KEY(contrat_id) REFERENCES contrats(id)
        );
    )";
    executeSQL(db, createTableSQL);

    // Exemple d’utilisation
    insererPli(db, 1, 1, "AS", "KD", "7H", "9C", "nord");
    insererPli(db, 1, 2, "2S", "3D", "5H", "JC", "est");

    modifierCarte(db, 1, "sud", "8H");

    lirePlis(db, 1);

    sqlite3_close(db);
    return 0;
}

