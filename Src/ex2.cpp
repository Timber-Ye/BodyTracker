#include <iostream>
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#define PC 1
#define img_debug 1

using namespace cv;

//////////滤波///////////

/*!
 *
 * @param input : 输入图像
 * @param output ： 输出图像
 * @param sigma ： 标准差
 * @param kSize : 卷积核尺寸 只能取1， 3， 5.。。
 */
void Gaussian(Mat& input,  Mat& output, double sigma, int kSize);

Mat GaussKernel(int kSize, double sigma);

//////////形态学///////////

/*!
 *
 * @param src : 输入图像
 * @param dst ： 输出图像
 * @param sigma ： 标准差
 */
void Dilate(const Mat& src, Mat& dst, const Mat& kernel);
void Erode(const Mat& src, Mat& dst, const Mat& kernel);

bool hit(const Mat& src, const Mat& kernel, int location_x, int location_y);
bool fit(const Mat& src, const Mat& kernel, int location_x, int location_y);



Mat GaussKernel(int kSize, double sigma){
    Mat Mask = Mat::zeros(kSize, kSize, CV_64F);
    double sum = 0.0;
    if(kSize != (kSize/2)*2 + 1 ) throw kSize;
    else{
        int center = (kSize - 1)/2;
        double x, y;

        for(int i = 0; i < kSize; i++){
            y = pow(i - center, 2);
            for(int j = 0; j < kSize; j++){
                x = pow(j - center, 2);
                double z = exp(-(x+y) / (2 * sigma * sigma));

                Mask.at<double>(j, i) = z;
                sum += z;

            }
        }
        Mask = Mask / sum;
    }
    return Mask;
}

void Gaussian(Mat &input, Mat &output, double sigma, int kSize) {
    // 空域高斯滤波器函数

    Mat mask = Mat::zeros(kSize, kSize, CV_8UC1);
    try {
        mask = GaussKernel(kSize, sigma);
    }
    catch (int) {
        std::cerr << "error of kSize. It must be an odd.\n";
        exit(1);
    }

    output = input.clone();
    int border = (kSize - 1) / 2;
    Mat dst;
    copyMakeBorder(input, dst, border, border, border, border, BORDER_REFLECT_101);

    for(int i = border; i < border + input.cols; i++){
        for(int j = border; j < border + input.rows; j++){
            double sum[3] = {0};

            for(int k = -border; k <= border; k++){
                for(int l = -border; l <= border; l++){
                    if(input.channels() == 1) sum[0] += input.at<uchar>(j + l, i + k) *
                                                        mask.at<double>(l + border, k + border);
                    else if(input.channels() == 3){
                        sum[0] += input.at<Vec3b>(j + l, i + k)[0] *
                                  mask.at<double>(l + border, k + border);
                        sum[1] += input.at<Vec3b>(j + l, i + k)[1] *
                                  mask.at<double>(l + border, k + border);
                        sum[2] += input.at<Vec3b>(j + l, i + k)[2] *
                                  mask.at<double>(l + border, k + border);
                    }
                }
            }

            for(int m = 0; m < input.channels(); m++){
                if(sum[m] < 0) sum[m] = 0;
                else if (sum[m] > 255) sum[m] = 255;
            }

            if(input.channels() == 1) output.at<uchar>(j - border, i - border) = sum[0];
            else if(input.channels() == 3){
                output.at<Vec3b>(j - border, i - border)[0] = sum[0];
                output.at<Vec3b>(j - border, i - border)[1] = sum[1];
                output.at<Vec3b>(j - border, i - border)[2] = sum[2];
            }
        }
    }
}

/////////形态学//////////
bool hit(const Mat& src, const Mat& kernel, int location_x, int location_y){
    int h = kernel.rows;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < h; j++){
//            std::cout << kernel.type() << std::endl;
            if(kernel.at<uchar>(h-i-1, h-j-1) == 1 && src.at<uchar>(location_x - i, location_y - j) == 255){
                return true;
            }
            else continue;
        }
    }
    return false;
}

bool fit(const Mat& src, const Mat& kernel, int location_x, int location_y) {
    int h = kernel.rows;
    for(int i = 0; i < h; i++){
        for(int j = 0; j < h; j++){
            if(kernel.at<uchar>(i, j) == 1 && src.at<uchar>(location_x - i, location_y - j) != 255)
                return false;
            else continue;
        }
    }
    return true;
}

void Dilate(const Mat& src, Mat& dst, const Mat& kernel){
    //膨胀函数
    if(src.channels() != 1){
        std::cerr<<"Error of image format. Grayscale ONLY."<<std::endl;
        exit(-1);
    }
    else if(kernel.rows != kernel.cols){
        std::cerr<<"Error of kernel format."<<std::endl;
        exit(-1);
    }

    int border = kernel.rows - 1;
    threshold(src, dst, 127, 255, THRESH_BINARY);

    Mat tem;
    copyMakeBorder(dst, tem, border, 0, border, 0, BORDER_CONSTANT, 0);

    for(int i = border; i < border + dst.cols; i++){
        for(int j = border; j < border + dst.rows; j++){
            if(dst.at<uchar>(j-border, i-border) == 255) continue;
            else if(hit(tem, kernel, j, i))
                dst.at<uchar>(j-border, i-border) = 255;
        }
    }

}

void Erode(const Mat& src, Mat& dst, const Mat& kernel){
    //腐蚀函数
    if(src.channels() != 1){
        std::cerr<<"Error of image format. Only grayscale."<<std::endl;
        exit(-1);
    }
    else if(kernel.rows != kernel.cols){
        std::cerr<<"Error of kernel format."<<std::endl;
        exit(-1);
    }

    int border = 1;
    threshold(src, dst, 127, 255, THRESH_BINARY);
    Mat tem;
    copyMakeBorder(dst, tem, border, border, border, border, BORDER_CONSTANT, 0);

    for(int i = border; i < border + dst.cols; i++){
        for(int j = border; j < border + dst.rows; j++){
            if(dst.at<uchar>(j-border, i-border) == 0) continue;
            else if(!fit(tem, kernel, j, i))
                dst.at<uchar>(j-border, i-border) = 0;
        }
    }
}

int main() {
    Mat frame, frIn, img_grey;
    VideoCapture capture;

    frIn = imread("../text-broken.tif", -1);

    if(frIn.channels() != 1) cvtColor(frIn, img_grey, COLOR_BGR2GRAY);
    else img_grey = frIn;

    Mat kernel_1 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));


/*    // 空域高斯滤波函数
    Mat img_Gau;
    Gaussian(frIn, img_Gau, 1, 5);*/

/*    Mat img_Gau2;
    GaussianBlur(frIn, img_Gau2, cv::Size(5, 5), 1, 1);*/



    // 膨胀函数
    Mat img_dil;

    Dilate(img_grey, img_dil, kernel_1);

    Mat img_dil2;
    dilate(img_grey, img_dil2, kernel_1, Point(-1, -1), 1, BORDER_CONSTANT, 0);
    Mat compared = img_dil - img_grey;




    /*// 腐蚀函数
    Mat img_ero;
    Erode(img_grey, img_ero, kernel_1);

    Mat img_ero2;
    erode(img_grey, img_ero2, kernel_1, Point(-1, -1), 1, BORDER_CONSTANT, 0);

    Mat compared = img_grey - img_ero;*/

    imshow("Orgimg", frIn);
//    imshow("img_Gau", img_Gau);
//    imshow("img_Gau2", img_Gau2);

    imshow("img_dil", img_dil);
    imshow("img_dil2", img_dil2);
    imshow("img_dil compared", compared);

//    imshow("img_ero", img_ero);
//    imshow("img_ero2", img_ero2);
//    imshow("img_dil compared", compared);

    waitKey(0);
}