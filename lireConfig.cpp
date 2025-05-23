#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "config.h"

#define ASSIGN_OPTION(option) \
    if (data.find(#option) != data.end()) maconf.option = data[#option];
#define ASSIGN_VAL(option) \
    if (data.find(#option) != data.end()) maconf.option = data[#option] / 1000;

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
    if (maconf.hauteurcarte == 0)  if (data.find("htcard") != data.end()) maconf.hauteurcarte = data["htcard"];
    htcard = maconf.hauteurcarte;

    ASSIGN_VAL(deltahaut)
    //if (data.find("deltahaut") != data.end()) maconf.deltahaut = htcard*data["deltahaut"] / 1000;
    if (data.find("deltacadre") != data.end()) maconf.deltacadre = htcard * data["deltacadre"] / 1000;
    if (data.find("deltacadrehaut") != data.end()) maconf.deltacadrehaut = htcard * data["deltacadrehaut"] / 1000;
    if (data.find("deltacoin") != data.end()) maconf.deltacoin = htcard * data["deltacoin"] / 1000;
    if (data.find("taillechiffre") != data.end()) maconf.taillechiffre = htcard * data["taillechiffre"] / 1000;
    if (data.find("tailleVDR") != data.end()) maconf.tailleVDR = htcard * data["tailleVDR"] / 1000;
    if (data.find("largeurchiffre") != data.end()) maconf.largeurchiffre = htcard * data["largeurchiffre"] / 1000;
    if (data.find("largeurVDR") != data.end()) maconf.largeurVDR = htcard * data["largeurVDR"] / 1000;
    if (data.find("largeursymbole") != data.end()) maconf.largeursymbole = htcard * data["largeursymbole"] / 1000;
    if (data.find("taillesymbole") != data.end()) maconf.taillesymbole = htcard * data["taillesymbole"] / 1000;
    if (data.find("deltachiffre") != data.end()) maconf.deltachiffre = htcard * data["deltachiffre"] / 1000;
    if (data.find("deltaVDR") != data.end()) maconf.deltaVDR = htcard * data["deltaVDR"] / 1000;
    if (data.find("deltahaut") != data.end()) maconf.deltahaut = htcard * data["deltahaut"] / 1000;
    if (data.find("deltahautVDR") != data.end()) maconf.deltahautVDR = htcard * data["deltahautVDR"] / 1000;
    if (data.find("deltahautsymbole") != data.end()) maconf.deltahautsymbole = htcard * data["deltahautsymbole"] / 1000;
    if (data.find("deltasymbole") != data.end()) maconf.deltasymbole = htcard * data["deltasymbole"] / 1000;
    if (data.find("deltasymbcadre") != data.end()) maconf.deltasymbcadre = htcard * data["deltasymbcadre"] / 1000;
    if (data.find("deltachsymb") != data.end()) maconf.deltachsymb = htcard * data["deltachsymb"] / 1000;
    if (data.find("deltagros") != data.end()) maconf.deltagros = htcard * data["deltagros"] / 1000;
    if (data.find("deltagroshaut") != data.end()) maconf.deltagroshaut = htcard * data["deltagroshaut"] / 1000;
    if (data.find("largeurgros") != data.end()) maconf.largeurgros = htcard * data["largeurgros"] / 1000;
    if (data.find("taillegros") != data.end()) maconf.taillegros = htcard * data["taillegros"] / 1000;
    if (data.find("deltagrosRDV") != data.end()) maconf.deltagrosRDV = htcard * data["deltagrosRDV"] / 1000;
    if (data.find("deltagroshautRDV") != data.end()) maconf.deltagroshautRDV = htcard * data["deltagroshautRDV"] / 1000;
    if (data.find("largeurgrosRDV") != data.end()) maconf.largeurgrosRDV = htcard * data["largeurgrosRDV"] / 1000;
    if (data.find("taillegrosRDV") != data.end()) maconf.taillegrosRDV = htcard * data["taillegrosRDV"] / 1000;

    if (data.find("waitoption") != data.end()) maconf.waitoption = data["waitoption"];
    if (data.find("printoption") != data.end()) maconf.printoption = data["printoption"];
    if (data.find("tesOCR") != data.end()) maconf.tesOCR = data["tesOCR"];
    //if (data.find("ignorerGR") != data.end()) maconf.ignorerGS = data["ignorerGS"];
    ASSIGN_OPTION(ignorerGS)
    ASSIGN_OPTION(gradmin)
    ASSIGN_OPTION(gradmax)
    ASSIGN_OPTION(nbvote)
    ASSIGN_OPTION(nbpoints)
    ASSIGN_OPTION(ecartmax)

    return 0;
}
