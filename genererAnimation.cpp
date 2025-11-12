#include <opencv2/opencv.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

struct Frame {
    int frame;
    int x, y;
    float angle;
};
struct FrameData {
    int frame;
    int x, y, angle;
};

struct Animation {
    int couleur, valeur;
    vector<FrameData> frames;
};

int valeurCarte(char v) {
    if (v == 'A') return 13;
    if (v == '1') return 13;
    if (v == 'X') return 9;
    if (v == 'V') return 10;
    if (v == 'D') return 11;
    if (v == 'R') return 12;
    return (v - '0') - 1; // pour 2 à 9
}
int couleurCarte(char c) {
    switch (c) {
        case 'T': return 0;
        case 'K': return 1;
        case 'C': return 2;
        case 'P': return 3;
        default: return -1;
    }
}
Mat extractCard(Mat& sheet, int couleur, int valeur, int cardWidth, int cardHeight) {
  cv::Mat ima;
  //ima = sheet(Rect((couleur*13 + valeur - 1)*cardWidth, 0, cardWidth, cardHeight)).clone();
  // une bordure sépare les cartes :
  // ècart à gauche 4 pixels, 4 pixels entre chaque carte, 4 pixels en haut et 4 pixels en bas  
  ima = sheet(Rect((couleur*13 + valeur - 1)*cardWidth +4, 2, cardWidth - 4, cardHeight - 8)).clone();
  cv::resize(ima, ima,cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
  return ima;
}

Mat rotateImageFull(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0, src.rows / 2.0);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);

    // Calculer la taille du canevas après rotation
    double abs_cos = abs(rot.at<double>(0, 0));
    double abs_sin = abs(rot.at<double>(0, 1));
    int bound_w = int(src.rows * abs_sin + src.cols * abs_cos);
    int bound_h = int(src.rows * abs_cos + src.cols * abs_sin);

    // Ajuster la matrice de rotation pour centrer
    rot.at<double>(0, 2) += bound_w / 2.0 - center.x;
    rot.at<double>(1, 2) += bound_h / 2.0 - center.y;

    Mat dst;
    Scalar fond(255, 0, 255); // rose fluo    
    warpAffine(src, dst, rot, Size(bound_w, bound_h), INTER_LINEAR, BORDER_CONSTANT, fond);
    return dst;
}

Mat rotateImage(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0, src.rows / 2.0);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    Mat dst;
    Scalar borderColor(255, 0, 255); // rose fluo    
    warpAffine(src, dst, rot, src.size(), INTER_LINEAR, BORDER_CONSTANT, borderColor);
    return dst;
}

FrameData interpolate(const FrameData& a, const FrameData& b, int frame) {
    float alpha = float(frame - a.frame) / (b.frame - a.frame);
    FrameData result;
    result.frame = frame;
    result.x = int(a.x + alpha * (b.x - a.x));
    result.y = int(a.y + alpha * (b.y - a.y));
    result.angle = int(a.angle + alpha * (b.angle - a.angle));
    return result;
}

int main(int argc, char** argv) {
    char *fichierTexte = (char*)"animation.txt";
    string fichiervideo = "output.avi";
    string line;
    map<string, Animation> animations;
    int totalFrames;

    if (argc >= 3) fichiervideo = argv[2];
    if (argc >= 2) fichierTexte = argv[1];
    else {
        cerr << "Usage: " << argv[0] << " <animation.txt> <output.avi>" << endl;
    }

    ifstream file(fichierTexte);

    while (getline(file, line)) {
      istringstream ss(line);
      int frame, x, y;
      int angle;
      char couleur, valeur;

      ss >> frame >> couleur >> valeur >> x >> y >> angle;

      string id = string(1, couleur) + string(1, valeur);
      animations[id].couleur = couleur;
      animations[id].valeur = valeur;
      animations[id].frames.push_back({frame, x, y, angle});
      if (frame > totalFrames) totalFrames = frame;
    }

    // Load card sheet
    Mat sheet = imread("cards_2_fr.png");
    int cardWidth = sheet.cols / 59; // 4*13 + 7 (dos de cartes)
    int cardHeight = sheet.rows;

    // Load animation data

    Scalar bg(0, 128, 0);

    totalFrames += 25; // une seconde de plus, image vide 
    Size frameSize(1920, 1080);
    VideoWriter writer(fichiervideo, VideoWriter::fourcc('M','J','P','G'), 25, frameSize);

    for (int frame = 0; frame < totalFrames; ++frame) {
      Mat canvas(frameSize, CV_8UC3, bg);
      int frame1 = -1;
      while(true) {
        int framemin = 2<<15;
        Animation anim;
        std::string id;
        for (auto& [idx, animx] : animations) {
          int nf = animx.frames.front().frame;
          if (nf <= frame1) continue;
          if (nf < framemin) {
            framemin = nf;
            anim = animx; id = idx;
          } 
        }
        if (framemin == 2<<15) break;
        frame1 = framemin;
          if (frame < anim.frames.front().frame || frame > anim.frames.back().frame) continue;

          FrameData current;
          for (size_t i = 1; i < anim.frames.size(); ++i) {
              if (frame <= anim.frames[i].frame) {
                  current = interpolate(anim.frames[i - 1], anim.frames[i], frame);
                  break;
              }
          }
          int couleurIndex = couleurCarte(anim.couleur); // P→3, C→2, etc.
          int valeurIndex = valeurCarte(anim.valeur);    // A→1, V→11, etc.
          Mat card = extractCard(sheet, couleurIndex, valeurIndex, cardWidth, cardHeight);
          Mat rotated = rotateImageFull(card, current.angle);

          // Créer un masque à partir des pixels non noirs
          Mat mask;
          Scalar fond(255, 0, 255); // rose fluo    
          inRange(rotated, fond , fond, mask); // exclut le rose fluo
          bitwise_not(mask, mask);            // inverse : garde tout sauf le fond

          // Définir la position
          int x = current.x;
          int y = current.y;
          Rect roi(x, y, rotated.cols, rotated.rows);

          // Vérifier les limites
          if (roi.x >= 0 && roi.y >= 0 &&
              roi.x + roi.width <= canvas.cols &&
              roi.y + roi.height <= canvas.rows) {
              rotated.copyTo(canvas(roi), mask);
          }

        } // while

        writer.write(canvas);
    } // frames

    writer.release();
    cout << "Video generated: "<<fichiervideo << endl;
    return 0;
}
