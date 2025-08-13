#include "config.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#ifdef _WIN32
    #include <Windows.h>
    #include <tchar.h>
    #define popen _popen
    #define pclose _pclose
#endif
#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <cmath>

#include <curl/curl.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include "json.hpp"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


// #pragma comment(lib, "python312.lib")

//std::string sendFileToServer(const std::string& filepath);

int numconfig = 0; // numéro de jeu de cartes : 0= GRIMAUD, 1= funbridge, 2= funbridge ancien, 3= CEFALU

cv::String setconfig(config& maconf) {
    maconf.hauteurcarte = 0;
    maconf.cosOrtho = 0.09; // 0.035 = 2 degr�s 0.09 = 5 degr�s
    maconf.cosOrthoOrtho = 0.017; // 1 degr�
    maconf.deltaradian = 0.1; // 6 degr�s

    maconf.deltacoin = 60;   // au moins taillechiffre
    maconf.deltachiffre = 6;
    maconf.deltahaut = 8;
    maconf.deltahautVDR = 8;
    maconf.deltahautsymbole = 30;
    maconf.deltasymbcadre = 2;
    maconf.taillechiffre = 22;
    maconf.largeurchiffre = 17;

    maconf.gradmin = 100;     // 200 : détecte seulement les meilleure lignes
    maconf.gradmax = 250;    // 250 : ?
    maconf.nbvote = 30;      // 30 : valeur plutot faible : beaucoup de lignes détectées 
    maconf.nbpoints = 10;    // 20 : nombre de pixels pour une ligne
    maconf.ecartmax = 3;    //  3 : plutot faible. 5: moyen. 20 : fouillis de lignes dans Roi Dame Valet. écart maximal entre segments détectés 

    maconf.deltacadre = maconf.deltachiffre / 2;
    maconf.deltacadrehaut = maconf.deltacadre;
    maconf.deltasymbole = maconf.deltachiffre + maconf.taillechiffre + 2;
    maconf.taillesymbole = maconf.taillechiffre * 2/3;
    maconf.tailleVDR = maconf.taillechiffre;
    maconf.largeurVDR = maconf.largeurchiffre;
    maconf.deltachsymb = 1;      // �cart entre le bas du caractère et le symbole

#ifdef _WIN32
    //numconfig = 1;return "D:\\10vvr.png";   // funbridge ne fonctionne pas : pas de détection de hauteur de carte
    //numconfig = 1; return "D:\\pli5.png";     // funbridge

    //numconfig = 2;  maconf.deltacoin = 50;  return "D:\\5honneurs.png";
    //numconfig = 2;  maconf.deltacoin = 50; maconf.hauteurcarte = 300; return "D:\\8cartesFB.png";
    //numconfig = 2;  maconf.deltacoin = 50;  return "D:\\8cartesFBinv.png";
    //numconfig = 2;  maconf.deltacoin = 25; maconf.hauteurcarte = 150; return "D:\\8cartesFBmini.png";
    //numconfig = 2;  maconf.deltacoin = 15; maconf.hauteurcarte = 110; return "D:\\8cartesFBminimini.png";
    //numconfig = 2;  maconf.deltacoin = 15; maconf.hauteurcarte = 110; return "D:\\8cartesFBminiminiInv.png";
    //numconfig = 2;  maconf.deltacoin = 25; maconf.hauteurcarte = 150; return "D:\\8cartesFBminiInv.png";
    // 
    // ///////////////////////////// hauteur carte < 100 : mauvaises reconnaissances de chiffres et symboles
    //numconfig = 2;  maconf.deltacoin = 12; maconf.hauteurcarte = 75; return "D:\\8cartesFBmicro.png";
    //numconfig = 2;  maconf.deltacoin = 12; maconf.hauteurcarte = 75; return "D:\\8cartesFBmicroInv.png";
    //
    // /////////////////////// mauvaise reconnaissance taille variable
    //numconfig = 0;  maconf.deltacoin = 25; maconf.hauteurcarte = 127; return "D:\\test.mp4";


    ////////////////// cartes plastifiées COFALU ////////////////////

    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 96; return "D:\\1a9P_96.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 112; return "D:\\9aRP_112.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 96; return "D:\\1a9P_96.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 108; return "D:\\9arP_108.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 112; return "D:\\1a9k_112.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 120; return "D:\\1a9k_120.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 120; return "D:\\10vdrKC_120.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 105; return "D:\\1a9T_105.png";
    //numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 108; return "D:\\9arP_108.png";
    numconfig = 3;  maconf.deltacoin = 15; maconf.hauteurcarte = 103; return "D:\\10vdrPT_103.png";
    //numconfig = 3;  maconf.deltacoin = 30; maconf.hauteurcarte = 253; return "D:\\10vdrPT_253.png";


    // 
    //numconfig = 0; maconf.deltacoin = 20; maconf.hauteurcarte = 127; return "D:\\10vdrpckt_0_121.png";
    // numconfig = 0; return "D:\\vdr1.png";
    //numconfig = 0; maconf.deltacoin = 50; maconf.hauteurcarte = 213; return "D:\\2a10_P.png";
    //numconfig = 0; return "D:\\4coeurs.png";
    //numconfig = 0; return "D:\\6cartes.png";
    //numconfig = 0; maconf.deltacoin = 50; maconf.hauteurcarte = 240;  return "D:\\6honneurs.png";
#else
    numconfig = 2;  maconf.deltacoin = 25; maconf.hauteurcarte = 150; return "../8cartesFBmini.png";

#endif
}
void resetconfig(double htcard, config& maconf) {

    // spécifique d'un jeu de cartes : jeu personnel superfine B.P. GRIMAUD
    if (numconfig == 0) { // GRIMAUD
        maconf.deltahaut = htcard * 0.036;      // écart en haut ou en bas pour V D R
        // 0.031 pour chiffre
        // en fait on peut choisir moind de 0.031, il y a du blanc au dessus
        maconf.deltacadre = htcard * 0.029;   // écart entre le bord de carte et le cadre de Roi Dame Valet 
        maconf.deltacadrehaut = htcard * 0.029;   // écart entre le bord supérieurde carte et le cadre de Roi Dame Valet 
        maconf.deltacoin = htcard * 0.120;   // rayon du bord arrondi de la carte. un peu plus 
        maconf.taillechiffre = htcard * 0.084;  // hauteur du chiffre
        maconf.tailleVDR = htcard * 0.094;  // hauteur du ccaractère V D ou R
        maconf.largeurchiffre = htcard * 0.070; // = largeur chiffre
        maconf.largeurVDR = htcard * 0.047; // = largeur du caractère V D ou R
        maconf.largeursymbole = htcard * 0.035; // = largeur du caractère V D ou R
        maconf.taillesymbole = htcard * 0.047;  // à peine plus que la taille estimée 0.035
        maconf.deltachiffre = htcard * 0.029;   // écart à gauche ou à droite du chiffre
        maconf.deltaVDR = (0.5 + htcard * 0.012);
        maconf.deltahaut = htcard * 0.035;      // écart entre le bord supérieur ou inférieur de la carte et le cadre si RDV
        maconf.deltahautVDR = htcard * 0.048;   // écart entre le bord supérieur et le caractère V D R
        maconf.deltahautsymbole = htcard * 0.120; // écart entre le bord supérieur et le symbole
        maconf.deltasymbole = htcard * 0.042;     // écart entre le bord et le symbole Pique Coeur ...
        maconf.deltasymbcadre = htcard * 0.012;  // écart entre le cadre et le symbole
        maconf.deltachsymb = (0.5 + htcard * 0.010);      // �cart entre le bas du caractère et le symbole
        maconf.deltagros = htcard * 0.106;
        maconf.deltagroshaut = htcard * 0.106;
        maconf.largeurgros = htcard * 0.153;
        maconf.taillegros = htcard * 0.165;
        maconf.deltagrosRDV = htcard * 0.070;
        maconf.deltagroshautRDV = htcard * 0.012;
        maconf.largeurgrosRDV = htcard * 0.106;
        maconf.taillegrosRDV = htcard * 0.152;

    } else
        if (numconfig == 1) { // funbridge
            maconf.deltahaut = htcard * 0.021;      // écart en haut ou en bas pour V D R
            // 0.033 pour chiffre
            // en fait on peut choisir moind de 0.033, il y a du blanc au dessus
            maconf.deltacadre = htcard * 0.029;   // écart entre le bord de carte et le cadre de Roi Dame Valet
            maconf.deltacadrehaut = htcard * 0.034;   // écart entre le bord de carte et le cadre de Roi Dame Valet 
            maconf.deltacoin = htcard * 0.060;   // rayon du bord arrondi de la carte. un peu plus
            maconf.taillechiffre = htcard * 0.083;  // hauteur du chiffre
            maconf.tailleVDR = htcard * 0.083;  // hauteur du ccaractère V D ou R
            maconf.largeurchiffre = htcard * 0.075; // = largeur symbole
            maconf.largeurVDR = htcard * 0.050; // = largeur du caractère V D ou R
            maconf.largeursymbole = htcard * 0.050; // = largeur du caractère V D ou R
            maconf.taillesymbole = maconf.taillechiffre;
            maconf.deltachiffre = htcard * 0.021;   // écart à gauche ou à droite du chiffre
            maconf.deltaVDR = htcard * 0.012;
            maconf.deltahaut = htcard * 0.021;   // écart au dessus du chiffre
            maconf.deltahautVDR = htcard * 0.021;   // écart au dessus du chiffre
            maconf.deltahautsymbole = htcard*0.124;
            maconf.deltasymbole = maconf.deltahaut + maconf.taillechiffre + 1;
            maconf.deltasymbcadre = htcard * 0.023;  // écart entre le cadre et le symbole
        }
        else
            if (numconfig == 2) { // funbrige ancien 
                maconf.deltahaut = htcard * 0.060;      // écart en haut ou en bas pour V D R
                // 0.031 pour chiffre
                // en fait on peut choisir moind de 0.031, il y a du blanc au dessus
                maconf.deltacadre = ( 0.5 +htcard * 0.033);   // écart entre le bord de carte et le cadre de Roi Dame Valet 
                maconf.deltacadrehaut = (0.5 + htcard * 0.046);   // écart entre le bord de carte et le cadre de Roi Dame Valet 
                maconf.deltacoin = (0.5 + htcard * 0.060);   // rayon du bord arrondi de la carte. un peu plus 
                maconf.taillechiffre = (0.5 +htcard * 0.098);  // hauteur du chiffre
                maconf.tailleVDR = ( 0.5 +htcard * 0.080);  // hauteur du ccaractère V D ou R
                maconf.largeurchiffre = (0.5 + htcard * 0.071); // = largeur symbole
                maconf.largeurVDR = (0.5 +htcard * 0.060); // = largeur du caractère V D ou R
                maconf.largeursymbole = (0.5 + htcard * 0.060); // = largeur du symbole
                maconf.taillesymbole = (0.5 +  htcard * 0.065);  // à peine plus que la taille estimée
                maconf.deltachiffre = ( 0.5 + htcard * 0.033);   // écart à gauche ou à droite du chiffre
                maconf.deltaVDR = (0.5 + htcard * 0.012);
                maconf.deltahaut = (0.5 + htcard * 0.035);      // écart entre le bord supérieur ou inférieur de la carte et le chiffre
                maconf.deltahautVDR = (0.5 + htcard * 0.060);   // écart au dessus du VDR
                maconf.deltahautsymbole = (0.5 + htcard*0.130);
                maconf.deltasymbole = (0.5 + htcard * 0.033);     // écart entre le bord et le symbole Pique Coeur ...
                maconf.deltasymbcadre = (0.5 + htcard * 0.000);  // écart entre le cadre et le symbole
                maconf.deltagros = htcard * 0.113;
                maconf.deltagroshaut = htcard * 0.107;
                maconf.largeurgros = htcard * 0.147;
                maconf.taillegros = htcard * 0.133;
                maconf.deltagrosRDV = htcard * 0.100;
                maconf.deltagroshautRDV = htcard * 0.053;
                maconf.largeurgrosRDV = htcard * 0.120;
                maconf.taillegrosRDV = htcard * 0.167;
            }
            else
                if (numconfig == 3) { // cartes plastifiées COFALU 
                    maconf.deltahaut = htcard * 0.046;      // écart en haut ou en bas pour V D R
                    // 0.031 pour chiffre
                    // en fait on peut choisir moind de 0.031, il y a du blanc au dessus
                    maconf.deltacadre = (0.5 + htcard * 0.054);   // écart entre le bord de carte et le cadre de Roi Dame Valet 
                    maconf.deltacadrehaut = (0.5 + htcard * 0.046);   // écart entre le bord de carte et le cadre de Roi Dame Valet 
                    maconf.deltacoin = (0.5 + htcard * 0.150);   // rayon du bord arrondi de la carte. un peu plus 
                    maconf.taillechiffre = (0.5 + htcard * 0.094);  // hauteur du chiffre
                    maconf.tailleVDR = (0.5 + htcard * 0.073);  // hauteur du ccaractère V D ou R
                    maconf.largeurchiffre = (0.5 + htcard * 0.117); // largeur du chiffre 10
                    maconf.largeurVDR = (0.5 + htcard * 0.059); // = largeur du caractère V D ou R
                    maconf.largeursymbole = (0.5 + htcard * 0.070); // = largeur du (petit) symbole
                    maconf.taillesymbole = (0.5 + htcard * 0.047);  // 1 mm de plus. cf deltahautsymbole
                    maconf.deltachiffre = (0.5 + htcard * 0.023);   // �cart à gauche ou à droite du chiffre
                    maconf.deltaVDR = (0.5 + htcard * 0.012);
                    maconf.deltahaut = (0.5 + htcard * 0.046);      // �cart entre le bord supérieur ou inférieur de la carte et le chiffre
                    maconf.deltahautVDR = (0.5 + htcard * 0.057);   // �cart au dessus du VDR
                    maconf.deltahautsymbole = (0.5 + htcard * 0.165); // �cart entre le bord haut et le symbole
                    maconf.deltasymbole = (0.5 + htcard * 0.048);     // �cart entre le bord et le symbole Pique Coeur ...
                    maconf.deltasymbcadre = (0.5 + htcard * 0.012);  // �cart entre le cadre et le symbole
                    maconf.deltachsymb = (0.5 + htcard * 0.020);      // �cart entre le bas du caractère et le symbole
                    maconf.deltagros = htcard * 0.130;
                    maconf.deltagroshaut = htcard * 0.095;            // entre 95 et 117
                    maconf.largeurgros = htcard * 0.153;             // r�el 176, r�duit pour �viter un gros symbole � cot� 
                    maconf.taillegros = htcard * 0.165;
                    maconf.deltagrosRDV = htcard * 0.062;           //�cart entre le cadre et le gros symbole
                    maconf.deltagroshautRDV = htcard * 0.018;       //�cart entre le cadre et le haut du gros symbole
                    maconf.largeurgrosRDV = htcard * 0.130;
                    maconf.taillegrosRDV = htcard * 0.117;
                }


}
   

std::string  tesOCR(cv::Mat image, bool estunRDV, double *pconfiance, double *pangle) {

cv::Mat thresh_image = image;
// Convertir en niveaux de gris
//cv::Mat gray_image;
//cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

// Appliquer un seuil
//cv::threshold(gray_image, thresh_image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
//cv::imshow("thresh", thresh_image); cv::waitKey(1);
//thresh_image = gray_image;
// Initialiser Tesseract
     // Initialiser Tesseract
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();

    if (ocr->Init(NULL, "eng")) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        delete ocr;
        return "";
    }
    char listecar[]="0123456789VDRKQJAv<>IUOQ";  // I U et O Q pour reconnaitre 10
    //ocr->SetImage(image.data, image.cols, image.rows, 3, image.step);
    //ocr->SetVariable("tessedit_char_whitelist", "0123456789DRVv<>IOUQ");
    if (estunRDV) ocr->SetVariable("tessedit_char_whitelist", "DRVKQJO0"); // ajout K Q J
    // ajout O et 0 à cause dela Queen 
    else ocr->SetVariable("tessedit_char_whitelist", "0123456789DRVKQJA");

    ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    ocr->SetVariable("debug_level", "0");
    ocr->SetImage(thresh_image.data, thresh_image.cols, thresh_image.rows, 1, thresh_image.step);
     // Obtenir le texte, le niveau de confiance et les positions
    ocr->Recognize(0);
    tesseract::ResultIterator* ri = ocr->GetIterator();
    // chercher un mot : 10 ou caractères
    tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL; // RIL_SYMBOL  _WORD
    double confmax = 0;
    std::string texte = "";
    std::string texteorig = "";
 
    if (ri != 0) {
        do {
            std::string motousymbole;
            std::string s = "";
            float conf(0.);
            char* symbol = 0;
            int x1, y1, x2, y2;
            char* mot = ri->GetUTF8Text(tesseract::RIL_WORD);
            if(mot){
                s = std::string(mot);
                if (s == "K") s = "R";
                if (s == "Q") s = "D";
                if (s == "O") s = "D";
                if (s == "0") s = "D";
                if (s == "J") s = "V";
                if (s == "A") s = "1";
                level = tesseract::RIL_WORD;
                texte = s;
                conf = ri->Confidence(level);
                motousymbole = "mot: ";
            }
            if (s.size() < 2 || conf < 70 ) {
                symbol = ri->GetUTF8Text(tesseract::RIL_SYMBOL);
                if(!symbol && ! mot ) continue;
                level = tesseract::RIL_SYMBOL;
                if (symbol) {
                    if(ri->Confidence(level) > conf) { 
                        conf = ri->Confidence(level);
                        s = std::string(symbol);
                        motousymbole = "symbole: ";
                    }
                }
            }
            if(mot) confmax=conf;
            ri->BoundingBox(level, &x1, &y1, &x2, &y2);

            //std::cout << motousymbole<< s << " Confidence: " << conf << 
            // " | BoundingBox: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" 
            // << thresh_image.cols <<"x"<<thresh_image.rows<< std::endl;

            if(!mot &&  symbol) {
                char car = *symbol;
                s = symbol;
                s = s[0];
                texteorig += s;
                if (texte == "") {
                    float dconf = 0;
                    if (car == 'K') { car = 'R'; s = "R"; dconf = 0;}
                    if (car == 'Q') { car = 'D'; s = "D"; dconf =0;}
                    if (car == 'J') { car = 'V'; s = "V"; dconf =0;}
                    if (car == 'A') { car = '1'; s = "1"; dconf =0;}
                    if (conf > 90 && car == 'O') { car = 'D'; s = "D"; dconf =20;}
                    if (conf > 90 && car == '0') { car = 'D'; s = "D"; dconf =20;}

                    if (conf > 90 && car == 'W') { car = 'V'; s = "V"; dconf =20;}
                    if (conf > 90 && car == 'y') { car = 'V'; s = "V"; dconf =20;}
                    if (conf > 80 && car == 'K') { car = 'R'; s = "R"; dconf = 20;}
                    if (conf > 80 && car == 'L') { car = '1'; s = "1"; dconf = 20;}
                    if (conf > 80 && car == 'I') { car = '1'; s = "1"; dconf =20;}
                    if (conf > 90 && car == 'i') { car = '1'; s = "1"; dconf =20;} // expérimental !
                    if (conf > 95 && car == 'z') { car = '2'; s = "2"; dconf = 20;}
                    if (conf > 95 && car == 'Z') { car = '2'; s = "2"; dconf = 20;}
                    if (conf > 95 && car == 'v') { car = 'V'; s = "V";}
                    if (conf > 95 && car == 'g') { car = '9'; s = "9"; dconf =20;}
                    if (conf > 90 && car == 'S') { car = '5'; s = "5"; dconf =20;}
                    if (conf > 80 && car == 's') { car = '5'; s = "5"; dconf =20;}
                    //if (conf > 90 && car == 'B') { car = 'R'; s = "R"; dconf =20;} // ou D
                    //if (conf > 90 && car == 'b') { car = 'D'; s = "D";} // ou 6 !
                    if (image.rows <= 10 ) { // image petite
                        if (conf > 80 && car == 'z') { car = '2'; s = "2"; dconf = 20;} // expérimental !
                        if (conf > 80 && car == 'Z') { car = '2'; s = "2"; dconf = 20;} // expérimental !
                    }
                    conf -= dconf;
                }
                for(int i = 0; listecar[i]; i++ ){
                    if (car == listecar[i]) {
                        texte += s;
                        if (conf > confmax) {confmax = conf;}
                        break;
                    }
                }
            }
            if (symbol) delete[] symbol;
            if (mot) delete[] mot; 
        } while (ri->Next(level) || ri->Next(tesseract::RIL_WORD));
    }
    if (texteorig == "i)" || texteorig == "i}") texte = "D";
    if (texte == "1U") texte = "10";
    if(texte.size() == 2){
        if (texte[0] == '|') 
           if(texte[1] >= '1' && texte[1] <= '9') texte = texte[1];
    }
    if(pangle) *pangle = 0;
    if(texte != "") {
        if(pangle) *pangle = 360;
    }
    if(pconfiance) *pconfiance = confmax/ 100;

    // Nettoyer et libérer les ressources
    ocr->End();
    delete ocr;
    return texte;
}

#define SERVEUR
#define TESOCR
//#define PYOCR
std::string execOCR(cv::String nom, cv::Mat ima_ch, double *pconfiance, double *pangle) {
    //
    std::string response = "";
    /**********/
    #ifdef TESOCR
    if (nom != "SERVEUR" ) response = tesOCR(ima_ch,false ,pconfiance, pangle); 
    #endif 
    #ifdef PYOCR
    if (response == "")
    response = pyOCR(ima_ch, pconfiance, pangle);   // ne fonctionne pas en mode debug 
     #endif
    // test d'utilisation de easyOCR. non concluant
    //response = sendImageToServer( ima_ch, pconfiance, "5001");
    //std::cout <<" easyocr==>"<< response<<std::endl;
    #ifdef SERVEUR
if (response.size() == 0) {
    response = sendImageToServer(ima_ch, pconfiance, pangle);
    if (response == "K") response = "R";
    if (response == "Q") response = "D";
    if (response == "J") response = "V";
}
#endif
    return response;

    // std::string tescmd = "tesseract.exe " + nom + "  stdout --psm 10 -l eng ffb";  // -l fra ne fonctionne pas
    // const char *cmd = tescmd.c_str();
    // return execcmd(cmd);
}

std::string execOCRVDR(cv::String nom, cv::Mat ima_ch, double *pconfiance, double *pangle) {
    return execOCR(nom, ima_ch, pconfiance, pangle);
}

std::string decodejson(std::string jsonString, double* pconfidence, double* pangle) {

    std::string js;
    js = jsonString;

#ifdef _WIN32
    // il n'y a pas de symples quotes sous windows
    // on �vite un bug recognized text "'"
#else
    // Remplacer les guillemets simples par des guillemets doubles pour un format JSON valide.
    //for (char& c : jsonString) {
    //    if (c == '\'') c = '"';
    //}
#endif
    // Analyse de la chaîne JSON
    auto jsonObject = nlohmann::json::parse(jsonString);

    // Extraction des valeurs
    std::string recognized_text = jsonObject["recognized_text"];
    double confidence = jsonObject["confidence"];
    std::vector<std::vector<std::vector<int>>> bbox = jsonObject["bbox"].get<std::vector<std::vector<std::vector<int>>>>();

    // Afficher les valeurs de bbox
    for (const auto& outer : bbox) {
        for (const auto& inner : outer) {
            for (const auto& val : inner) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    int orientation = jsonObject["orientation"];

    // Affichage des valeurs
    //std::cout << "json: texte " << recognized_text << " confiance: " << confidence <<" angle "<< orientation << std::endl;

    *pconfidence = confidence;
    if (pangle) *pangle = orientation;
    if (confidence > 0.10)    return recognized_text;
    else return "?";
}

// Fonction de rappel pour recevoir la réponse du serveur 
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}
// static CURL* curl = nullptr;

std::string sendImageToServer(cv::Mat image, double *pconfiance, double *pangle, std::string port) {
    std::string serv_url = "http://127.0.0.1:" + port + "/predict";
    std::vector<uchar> buf;
    cv::imencode(".bmp", image, buf);
    CURL* curl = nullptr;
    CURLcode res;
    std::string readBuffer = "";
    if (!curl) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
    }
    struct curl_slist* headers = NULL;
    if (curl) {
        //curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/predict");
        curl_easy_setopt(curl, CURLOPT_URL, serv_url.c_str()) ;
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buf.data());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(buf.size()));
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
        headers = curl_slist_append(headers, "Expect:");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);


        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            //std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        //curl_slist_free_all(headers);
    }
    int sz = readBuffer.size();
    std::string texte = "?";
    double confiance= 0;
    double angle = 0;
    if (sz > 0 && sz < 200){
        texte = decodejson(readBuffer, &confiance, &angle);
        if(pconfiance) *pconfiance = confiance;
        if (pangle) *pangle = angle;
    }
    if (curl) {
        //curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        //curl_global_cleanup();
    }
    return texte;
}






















std::string send_image(const std::vector<uchar>& buffer) {
    CURL* curl;
    CURLcode res;
    std::string response_str = "";

    curl = curl_easy_init();
    if (curl) {
        std::string json = "{\"image\":\"" + std::string(buffer.begin(), buffer.end()) + "\"}";
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/predict");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Response from server: " << response_str << std::endl;
        }
    }
    return response_str;
}





















std::string sendFileToServer(const std::string& filepath) {
    CURL* curl;
    CURLcode res;
    std::string response;
    std::string response_string;
    std::string header_string;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        curl_mime* form = curl_mime_init(curl);
        curl_mimepart* field = curl_mime_addpart(form);
        curl_mime_name(field, "file");
        curl_mime_filedata(field, filepath.c_str());

        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/predict");
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
        
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &header_string);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            fprintf(stderr, "File sent successfully!\n");
            fprintf(stderr, "Response: %s\n", response_string.c_str());
        }

        curl_easy_cleanup(curl);
        curl_mime_free(form);
    }

    curl_global_cleanup();
    return response_string;
}



/******
int main() {
    std::string filepath = "D:/coin9.png";
    std::string response = sendFileToServer(filepath);
    std::cout << "Response from server: " << response << std::endl;
    return 0;
}
****/
