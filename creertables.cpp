#include <sqlite3.h>
#include <iostream>

int main() {
    sqlite3* db;
    char* errMsg = nullptr;

    if (sqlite3_open("bridge.db", &db) != SQLITE_OK) {
        std::cerr << "Erreur ouverture DB\n";
        return 1;
    }

    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY,
            nom TEXT
        );
        CREATE TABLE IF NOT EXISTS donnes (
            id INTEGER PRIMARY KEY,
            numero INTEGER,
            distribution TEXT,
            vuln TEXT
        );
        CREATE TABLE IF NOT EXISTS contrats (
            id INTEGER PRIMARY KEY,
            table_id INTEGER,
            donne_id INTEGER,
            joueur TEXT,
            contrat TEXT,
            FOREIGN KEY(table_id) REFERENCES tables(id),
            FOREIGN KEY(donne_id) REFERENCES donnes(id)
        );
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

    if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Erreur SQL: " << errMsg << "\n";
        sqlite3_free(errMsg);
    } else {
        std::cout << "Tables créées avec succès.\n";
    }

    sqlite3_close(db);
    return 0;
}

