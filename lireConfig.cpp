#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "config.h"

#define ASSIGN_OPTION(option) \
    if (data.find(#option) != data.end()) maconf.option = data[#option];
#define ASSIGN_VAL(v) \
    {if (data.find(#v) != data.end())  maconf.v  = htcard*data[#v]/1000;}

int lireConfig(std::string nomfichier, config& maconf) {
    std::ifstream file(nomfichier);
    std::map<std::string, int> data;
    std::string line;
    int htcard = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string name;
        std::string value_str;
        std::getline(ss, name, '='); // Utilisez ',' pour CSV ou ':' pour cl�-valeur
        std::getline(ss, value_str, '#');
        if (value_str != "") {
            int value = std::stoi(value_str);
            data[name] = value;
        }
    }
    if (maconf.hauteurcarte == 0)  if (data.find("htcard") != data.end()) 
      maconf.hauteurcarte = data["htcard"];
    htcard = maconf.hauteurcarte;
    maconf.largeurcarte = htcard * 2 / 3;
    if (data.find("largeurcarte") != data.end()) 
      maconf.largeurcarte = data["largeurcarte"];
    ASSIGN_OPTION(largeurcarte)

    ASSIGN_VAL(deltahaut)
    //if (data.find("deltahaut") != data.end()) maconf.deltahaut = htcard*data["deltahaut"] / 1000;
    ASSIGN_VAL(deltacadre)
    ASSIGN_VAL(deltacadrehaut)
    ASSIGN_VAL(deltacoin)
    ASSIGN_VAL(taillechiffre)
    ASSIGN_VAL(tailleVDR)
    ASSIGN_VAL(largeurchiffre)
    ASSIGN_VAL(largeurVDR)
    ASSIGN_VAL(largeursymbole)
    ASSIGN_VAL(taillesymbole)
    ASSIGN_VAL(deltachiffre)
    ASSIGN_VAL(deltaVDR)
    ASSIGN_VAL(deltahaut)
    ASSIGN_VAL(deltahautVDR)
    ASSIGN_VAL(deltahautsymbole)
    ASSIGN_VAL(deltasymbole)
    ASSIGN_VAL(deltasymbcadre)
    ASSIGN_VAL(deltachsymb)
    ASSIGN_VAL(deltagros)
    ASSIGN_VAL(deltagroshaut)
    ASSIGN_VAL(largeurgros)
    ASSIGN_VAL(taillegros)
    ASSIGN_VAL(deltagrosRDV)
    ASSIGN_VAL(deltagroshaut)
    ASSIGN_VAL(deltagroshautRDV)
    ASSIGN_VAL(largeurgrosRDV)
    ASSIGN_VAL(taillegrosRDV)

    ASSIGN_OPTION(waitoption)
    ASSIGN_OPTION(printoption)
    ASSIGN_OPTION(threadoption)
    ASSIGN_OPTION(tesOCR)
    ASSIGN_OPTION(ignorerGS)
    ASSIGN_OPTION(coinsoption)
    ASSIGN_OPTION(linesoption)
    ASSIGN_OPTION(fusionoption)
    ASSIGN_OPTION(calibrationoption)
    ASSIGN_OPTION(gradmin)
    ASSIGN_OPTION(gradmax)
    ASSIGN_OPTION(nbvote)
    ASSIGN_OPTION(nbpoints)
    ASSIGN_OPTION(ecartmax)

    ASSIGN_OPTION(xjeu)
    ASSIGN_OPTION(yjeu)
    ASSIGN_OPTION(wjeu)
    ASSIGN_OPTION(hjeu)
    ASSIGN_OPTION(xmort)
    ASSIGN_OPTION(ymort)
    ASSIGN_OPTION(wmort)
    ASSIGN_OPTION(hmort)

    ASSIGN_OPTION(contratcouleur)
    ASSIGN_OPTION(contratvaleur)
    ASSIGN_OPTION(declarant)
    ASSIGN_OPTION(numeroDonne)


    return 0;
}
