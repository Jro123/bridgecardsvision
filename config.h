#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <thread>

class config {
public:
    int hauteurcarte;  // hauteur de carte en pixels
    int largeurcarte;   // largeur de la carte en pixels
    double cosOrtho; // cosinus pour consid�rer que deux droites sont orthogonales
    double cosOrthoOrtho; // tr�s orthogonales
    double deltaradian;  // 
    int deltacoin; // taille de la partie arrondie d'un coin de carte (rayon de l'arc de cercle)
    int deltacadre; // ecart entre le bord de carte et le cadre de Roi Dame Valet, en haut et sur le cot�
    int deltacadrehaut; // �cart entre le bord sup�rieur ou inf�rieur et le cadre
    int deltachiffre; // �cart entre le bord gauche ou droit et le chiffre
    int deltaVDR;     // �cart entre le cadre et le caract�re V D R
    int deltahaut;    // �cart entre le bord haut et le chiffre
    int deltahautVDR;  // �cart entre le bord haut et le caract�re V D ou R
    int deltahautsymbole;  // �cart entre le bord haut et le symbole
    int deltasymbcadre;  // �cart entre le cadre de cot�  d'un R D V et le symbole
    int deltachsymb;     // �cart entrele caract�re et le symbole
    int taillechiffre; // hauteur  du rectangle contenant le chiffre (ou 10 R D V)
    int tailleVDR;     // hauteur du caract�re V D ou R
    int largeurchiffre; // largeur du rectangle contenant le chiffre (ou 10 R D V)
    int largeurVDR;     // largeur du caract�re V D ou R
    int largeursymbole; // largeur du symbole
    int deltasymbole;  // �cart entre le bord  et le symbole
    //                    �cart entre bord vertical et symbole = deltachiffre
    int deltagros;    // �cart entre le cot� de carte et le gros symbole
    int deltagroshaut; // �cart entre le haut de carte et le gros symbole
    int taillegros;     // hauteur du gros symbole
    int largeurgros;    // largeur du gros symbole
    int deltagrosRDV;   // �cart entre le cot� et le gros symbole
    int deltagroshautRDV;  // �cart entre le haut de carte et le gros symbole
    int largeurgrosRDV;    // largeur du gros symbole
    int taillegrosRDV;     // taille du gros symbole
    int taillesymbole;  // hauteur et largeur du symbole
    int gradmin; // gradient mini pour canny
    int gradmax;
    int nbvote; // nombre de votes pour obtenir une droite
    int nbpoints; // nombre minimum de points sur une droite
    int ecartmax; // ecart maxi entre deux points d'une droite
    int waitoption;
    int printoption;
    int threadoption;
    int tesOCR;  // 1: utiliser tesseract, 0 : uniquement le serveur
    int ignorerGS; // utiliser le gros symbole si la taille du petit est plus perite
    int coinsoption; // 0: analyser uniquement les cartes,  1: analyser aussi les coins isolés
    int linesoption; // 1 : Canny et houghlines, 2 : ximgproc
    int fusionoption; // 0: pas de fusion, 1: fusionner les segments proches ayant même support

    int contratcouleur; // 0:Pique, 1:Coeur, 2:Carreau, 3:Trefle, -1:SA
    int contratvaleur; // 1 à 7
    int declarant; // numéro de joueur déclarant 0:nord, 1:Est, 2:Sud, 3:Ouest  
    config() {}
};

// carct�ristique d'un coin de carte
class uncoin {
public:
    int (*pcoins)[12];     // tableau des coins 
    int numcoin;     // num�ro de coin
    cv::Point2i A;   // haut gauche du chiffre suppos� vertical
    cv::Point2i B;   // bas droit
    cv::Point2i AA;  // haut gauche du chiffre suppos� horizontal
    cv::Point2i BB;
    cv::Point2i U;   // haut gauche du symbole suppos� vertical
    cv::Point2i V;
    cv::Point2i UU;  // haut gauche du symbole suppos� horizontal
    cv::Point2i VV;
    cv::Point2i PP;  // position du coin
    cv::Point2i QQ;  // position du cadre
    bool estunRDV;   // true si on a �tabli que c'est un Roi une Dame ou un Valet
    bool inverse;    // true si on a �tabli que la carte est horizontale
    bool estDroit;   // true si on sait que la carte est verticale
    bool estRouge;   // true si le coin a un symbole rouge 
    char caractere;
    cv::Scalar moyblanc;  // couleur du "blanc"
    cv::Mat ima_coin;     // image du coin
    uncoin() {}
    uncoin(int (*tab)[12]) : pcoins(tab) {}  // constructeur d'initialisation
};

class ligne{
public:
    cv::Vec4i ln; // x1, y1, x2, y2
    float lg; // longueur du segment
    float a; // équation de la droite
    float b;
    float c;
    ligne(){
        ln = {0,0,0,0};
        a = b = c = 0;
        lg = 0;
    }
};

class Pli {
public:
  int joueur;  // numéro du joueur qui entame le pli. 0=Nord, 1=Est, 2=Sud, 3=Ouest
  int cartes[4][2]; // Nord,est,sud,ouest    couleur et valeur
  int joueurgagnant;  // numéro du joueur qui remporte ce pli 
  Pli() { joueur=-1; joueurgagnant = -1; for(int i=0; i<4; i++) {cartes[i][0] = -1; cartes[i][1]= 0;}}
};

// une carte avec ses 4 sommets et sa valeur éventuelle
class unecarte {
public:
  int couleur;
  int valeur;
  int sommets[4][2];  
};

// un pli en cours avec les cartes déjà trouvées
class unpli {
public:
  int nbcartes;
  unecarte cartes[4];
};

int lireConfig(std::string nomfichier, config& maconf);

void tracerRectangle(cv::Rect r, cv::Mat copie, std::string s, cv::Scalar couleur);

void afficherImage(std::string nom, cv::Mat image);
int calculerCouleur(cv::Mat GS, const config& maconf, cv::Scalar mbl);

void calculerOrientation(uncoin& moncoin, const config& maconf);
void calculerBlanc(uncoin& moncoin, const config& maconf);  // d�terminer la composition du blanc
void calculerMinimum(const cv::Mat& image, cv::Scalar& minimum);

void calculerMoyenneSymbole(const cv::Mat& image, const cv::Scalar ref, cv::Scalar& moy);
void calculerBox(const cv::Mat& image, const int ts, const int ls, cv::Scalar& moy,
    int *pBox, cv::Scalar& moyext, const config& maconf);

void calculerEncombrement(const cv::Mat& image, const cv::Scalar ref, cv::Scalar& moy,
     int *pBox, cv::Mat& imaR, cv::Scalar& moyext);
double calculerDistance(cv::Vec4i& l1, cv::Vec4i& l2);

double calculerDistance(cv::Point2i& Q, cv::Point2i P, cv::Point2i R);
double calculerSinus(cv::Vec4i& l1, cv::Vec4i& l2);
int decoderCarte(cv::Mat& image, int pts[4][2], config& maconf, int& numcol);
std::string ValiderOCR(std::string output, bool estserveur, bool inverse,
             uncoin& moncoin, const config& maconf);

void eclaircirfond(cv::Mat& image);
void blanchircadre(cv::Mat& image, cv::Scalar moyblanc, int nb);
void amplifyContrast(cv::Mat& image);

void followContour(const cv::Mat& edges, cv::Point2i start, cv::Point2i ref, std::vector<cv::Point2i>& contour, double tolerance);
std::string sendImageToServer(cv::Mat image, double *pconfiance,double *pangle, std::string port="5000");

cv::String  setconfig(config& maconf);    // retourne le nom du fichier image 

void resetconfig(double htcarte, config& maconf);

cv::String pyOCR(cv::Mat image, double *pconfiance, double* pangle);

std::string tesOCR(cv::Mat ima_ch, bool estunRDV,  double *pconfiance= 0, double *pangle= 0);
std::string execOCR(cv::String nom, cv::Mat ima_ch, double *pconfiance= 0, double *pangle= 0);
std::string execOCRVDR(cv::String nom, cv::Mat ima_ch, double *pconfiance = 0, double *pangle=0);
//tescmd = "tesseract.exe " + nomcoin + "  stdout --psm 10 -l fra ffb ";

void traiterCoin(int n, int coins[][12], cv::Mat image,  std::vector<std::string>& resultats, 
    cv::Mat result, const int *l1, const int *l2, const config& maconf);

//void traiterCointhread(std::vector<std::thread> *pthreads,int n, int coins[500][10], cv::Mat image,  std::vector<std::string>& resultats, 
//    cv::Mat result, int *l1, int *l2, config& maconf);


// TODO:
// d�finir tous les param�tres � partir d'un seul : dimension de la carte
// hauteur de la carte exprim�e en mm
// et le facteur d'�chelle : nombre de pixels correspondant � la longueur de carte
//
// la position du chiffre ou de la lettre de valeur de la carte n'est pas la m�me dans les deux dimensions
// le rapport hauteur / largeur est de 5/4

// mesures pour une carte :
// hauteur carte : 1000      valeur de r�f�rence
// largeur         675
// �cart en haut    54
// �cart de cot�    42
// ht chiffre       80
// largeur          65
// ht symbole       65
