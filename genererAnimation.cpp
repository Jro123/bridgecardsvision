#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;
using namespace cv;
using namespace std;

struct FrameData {
    int frame;
    int x, y, angle;
};

struct Animation {
    int couleur, valeur;
    vector<FrameData> frames;
};

Mat extractCard(Mat& sheet, int couleur, int valeur, int cardWidth, int cardHeight) {
    return sheet(Rect((couleur*13 + valeur - 1)*cardWidth, 0, cardWidth, cardHeight)).clone();
    //return sheet(Rect(valeur * cardWidth, couleur * cardHeight, cardWidth, cardHeight)).clone();
}

Mat rotateImage(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0, src.rows / 2.0);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    Mat dst;
    warpAffine(src, dst, rot, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
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
    string jsonFile = "animation.json";
    string fichiervideo = "output.avi";
    if (argc >= 3) fichiervideo = argv[2];
    if (argc >= 2) jsonFile = argv[1];
    else {
        cerr << "Usage: " << argv[0] << " <animation.json> <output.avi>" << endl;
    }

    ifstream f(jsonFile);
    if (!f.is_open()) {
        cerr << "Erreur : impossible d’ouvrir le fichier " << jsonFile << endl;
        return 1;
    }

    json data = json::parse(f);

    // Load card sheet
    Mat sheet = imread("cards_2_fr.png");
    int cardWidth = sheet.cols / 59; // 4*13 + 7 (dos de cartes)
    int cardHeight = sheet.rows;

    // Load animation data

    Scalar bg(data["background_color"][2], data["background_color"][1], data["background_color"][0]);

    vector<Animation> animations;
    for (auto& anim : data["animations"]) {
        Animation a;
        a.couleur = anim["id"][0];
        a.valeur = anim["id"][1];
        for (auto& fr : anim["frames"]) {
            a.frames.push_back({fr["frame"], fr["x"], fr["y"], fr["angle"]});
        }
        animations.push_back(a);
    }

    int totalFrames = 75;
    Size frameSize(1920, 1080);
    VideoWriter writer(fichiervideo, VideoWriter::fourcc('M','J','P','G'), 25, frameSize);

    for (int frame = 0; frame < totalFrames; ++frame) {
        Mat canvas(frameSize, CV_8UC3, bg);

        for (auto& anim : animations) {
            if (frame < anim.frames.front().frame || frame > anim.frames.back().frame) continue;

            FrameData current;
            for (size_t i = 1; i < anim.frames.size(); ++i) {
                if (frame <= anim.frames[i].frame) {
                    current = interpolate(anim.frames[i - 1], anim.frames[i], frame);
                    break;
                }
            }

            Mat card = extractCard(sheet, anim.couleur, anim.valeur, cardWidth, cardHeight);
            Mat rotated = rotateImage(card, current.angle);

            Rect roi(current.x, current.y, card.cols, card.rows);
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= canvas.cols && roi.y + roi.height <= canvas.rows) {
                rotated.copyTo(canvas(roi));
            }
        }

        writer.write(canvas);
    }

    writer.release();
    cout << "Video generated: "<<fichiervideo << endl;
    return 0;
}
