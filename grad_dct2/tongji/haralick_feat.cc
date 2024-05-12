
#include "haralick_feat.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

//Marginal probabilities as in px = sum on j(p(i, j))
//                             py = sum on i(p(i, j))
vector<double> MargProbx(Mat cooc) {
    vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[i] += cooc.at<double>(i, j);
    return result;
}

vector<double> MargProby(Mat cooc) {
    vector<double> result(cooc.cols, 0.0);
    for (int j = 0; j < cooc.cols; j++)
        for (int i = 0; i < cooc.rows; i++)
            result[j] += cooc.at<double>(i, j);
    return result;
}

//probsum  := Px+y(k) = sum(p(i,j)) given that i + j = k
vector<double> ProbSum(Mat cooc) {
    vector<double> result(cooc.rows * 2, 0.0);
    for (int i = 0; i < cooc.rows; i++) 
        for (int j = 0; j < cooc.cols; j++)
            result[i + j] += cooc.at<double>(i, j);
    return result;
}

//probdiff := Px-y(k) = sum(p(i,j)) given that |i - j| = k
vector<double> ProbDiff(Mat cooc) {
    vector<double> result(cooc.rows, 0.0);
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            result[abs(i - j)] += cooc.at<double>(i, j);
    return result;
}


/*Features from coocurrence matrix*/
double HaralickEnergy(Mat cooc) {
    double energy = 0;
    for (int i = 0; i < cooc.rows; i++) {
        for (int j = 0; j < cooc.cols; j++) {
            energy += cooc.at<double>(i,j) * cooc.at<double>(i,j);
        }
    }
    return energy;
}

double HaralickEntropy(Mat cooc) {
    double entrop = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            entrop += cooc.at<double>(i,j) * log(cooc.at<double>(i,j) + EPS);
    return -1 * entrop;
}

double HaralickInverseDifference(Mat cooc) {
    double res = 0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            res += cooc.at<double>(i, j) * (1 / (1 + (i - j) * (i - j)));
    return res;
}

/*Features from MargProbs */
double HaralickCorrelation(Mat cooc, vector<double> probx, vector<double> proby) {
    double corr;
    double meanx, meany, stddevx, stddevy;
    meanStd(probx, meanx, stddevx);
    meanStd(proby, meany, stddevy);
    for (int i = 0; i < cooc.rows; i++) 
        for (int j = 0; j < cooc.cols; j++)
            corr += (i * j * cooc.at<double>(i, j)) - meanx * meany;
    return corr / (stddevx * stddevy);
}

//InfoMeasure1 = HaralickEntropy - HXY1 / max(HX, HY)
//HXY1 = sum(sum(p(i, j) * log(px(i) * py(j))
double HaralickInfoMeasure1(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
    double hx = Entropy(probx);
    double hy = Entropy(proby);
    double hxy1 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy1 += cooc.at<double>(i, j) * log(probx[i] * proby[j] + EPS);
    hxy1 = -1 * hxy1;

    return (ent - hxy1) / max(hx, hy);

}

//InfoMeasure2 = sqrt(1 - exp(-2(HXY2 - HaralickEntropy)))
//HX2 = sum(sum(px(i) * py(j) * log(px(i) * py(j))
double HaralickInfoMeasure2(Mat cooc, double ent, vector<double> probx, vector<double> proby) {
    double hxy2 = 0.0;
    for (int i = 0; i < cooc.rows; i++)
        for (int j = 0; j < cooc.cols; j++)
            hxy2 += probx[i] * proby[j] * log(probx[i] * proby[j] + EPS);
    hxy2 = -1 * hxy2;

    return sqrt(1 - exp(-2 * (hxy2 - ent)));
}

/*Features from ProbDiff*/
double HaralickContrast(Mat cooc, vector<double> diff) {
    double contrast = 0.0;
    for (int i = 0; i < diff.size(); i++) 
        contrast += i * i * diff[i];
    return contrast;
}

double HaralickDiffEntropy(Mat cooc, vector<double> diff) {
    double diffent = 0.0;
    for (int i = 0; i < diff.size(); i++) 
        diffent += diff[i] * log(diff[i] + EPS);
    return -1 * diffent;
}

double HaralickDiffVariance(Mat cooc, vector<double> diff) {
    double diffvar = 0.0;
    double diffent = HaralickDiffEntropy(cooc, diff);
    for (int i = 0; i < diff.size(); i++)
        diffvar += (i - diffent) * (i - diffent) * diff[i];
    return diffvar;
}

/*Features from Probsum*/
double HaralickSumAverage(Mat cooc, vector<double> sumprob) {
    double sumav = 0.0;
    for (int i = 0; i < sumprob.size(); i++)
        sumav += i * sumprob[i];
    return sumav;
}

double HaralickSumEntropy(Mat cooc, vector<double> sumprob) {
    double sument = 0.0;
    for (int i = 0; i < sumprob.size(); i++) 
        sument += sumprob[i] * log(sumprob[i] + EPS);
    return -1 * sument;
}

double HaralickSumVariance(Mat cooc, vector<double> sumprob) {
    double sumvar = 0.0;
    double sument = HaralickSumEntropy(cooc, sumprob);
    for (int i = 0; i < sumprob.size(); i++)
        sumvar += (i - sument) * (i - sument) * sumprob[i];
    return sumvar;
}


Mat MatCooc(Mat img, int N, int deltax, int deltay) 
{
    int atual, vizinho;
    int newi, newj;
    Mat ans = Mat::zeros(N + 1, N + 1, CV_64F);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            newi = i + deltay;
            newj = j + deltax;
            if (newi < img.rows && newj < img.cols && newj >= 0 && newi >= 0) {
                atual = (int) img.at<uchar>(i, j);
                vizinho = (int) img.at<uchar>(newi, newj);
                ans.at<double>(atual, vizinho) += 1.0;
            }
        }
    }
    return ans / (img.rows * img.cols);
}

//Assume tamanho deltax == tamanho deltay 
Mat MatCoocAdd(Mat img, int N, std::vector<int> deltax, std::vector<int> deltay)
{
    Mat ans, nextans;
    ans = MatCooc(img, N, deltax[0], deltay[0]);
    for (int i = 1; i < deltax.size(); i++) {
        nextans = MatCooc(img, N, deltax[i], deltay[i]);
        add(ans, nextans, ans);
    }
    return ans;
}

void printMat(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++)
            printf("%lf ", (double) img.at<double>(i, j));
        printf("\n");
    }
}


int main() {
    vector<string> categories = {"car", "cat", "chair", "horse"};
    vector<string> types = {"0_real", "1_fake"};
    ofstream realFile, fakeFile;
    realFile.open("real.txt");
    fakeFile.open("fake.txt");

    for (const auto& category : categories) {
        for (const auto& type : types) {
            string basePath = "/opt/data/private/wangjuntong/datasets/progan_train/" + category + "/" + type + "/";
            for (const auto& entry : fs::directory_iterator(basePath)) {
                if (entry.path().extension() == ".png") {
                    Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
                    if (img.empty()) continue; // Skip if the image is not successfully loaded

                    HaralickExtractor extractor;
                    vector<int> deltaX = {1}; // You may adjust these parameters
                    vector<int> deltaY = {0}; // You may adjust these parameters
                    vector<double> features = extractor.getFeaturesFromImage(img, deltaX, deltaY);

                    // Constructing the output line
                    stringstream line;
                    line << entry.path().filename().string() << ",";
                    line << "Energy:" << features[0] << ",";
                    line << "Entropy:" << features[1] << ",";
                    line << "InverseDifference:" << features[2] << ",";
                    line << "Correlation:" << features[3] << ",";
                    line << "InfoMeasureCorrelation1:" << features[4] << ",";
                    line << "InfoMeasureCorrelation2:" << features[5] << ",";
                    line << "Contrast:" << features[6] << ",";
                    line << "DifferenceEntropy:" << features[7] << ",";
                    line << "DifferenceVariance:" << features[8] << ",";
                    line << "SumAverage:" << features[9] << ",";
                    line << "SumEntropy:" << features[10] << ",";
                    line << "SumVariance:" << features[11];

                    // Writing the output line to the correct file
                    if (type == "1_fake") {
                        fakeFile << line.str() << endl;
                    } else {
                        realFile << line.str() << endl;
                    }
                }
            }
        }
    }

    realFile.close();
    fakeFile.close();

    return 0;
}
    
  
