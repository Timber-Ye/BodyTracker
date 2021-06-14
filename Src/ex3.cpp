//
// Created by 翰樵 on 2020/10/25.
//
#include <iostream>
//#include <cv.h>
//#include <highgui.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#define PC 1
#define img_debug 1

bool weak_edge[1000000] = {0};

using namespace cv;
void myCanny(const Mat & src, Mat & dst, uint8_t threshold1, uint8_t threshold2);
void SobelOpt(const Mat & src, Mat & gradient_x, Mat & gradient_y);
void NoneMaximumSuppression(const Mat & gradient_x, const Mat & gradient_y, Mat & dst);
void HysteresisRecursion(Mat & input, int x, int y, uint8_t lowThreshold, uint8_t highThreshold);
void myHoughLine(Mat & input, Mat & output, int threshold);
void myHoughCircle(Mat & input, Mat & output, int threshold);

void SobelOpt(const Mat & src, Mat & gradient_x, Mat & gradient_y){
    int n = 2;
    int S_x[3][3] = {{-1*n, 0, 1*n},
                     {-2*n, 0, 2*n},
                     {-1*n, 0, 1*n}};
    int S_y[3][3] = {{1*n, 2*n, 1*n},
                     {0, 0, 0},
                     {-1*n, -2*n, -1*n}};

    int img_row = src.rows;
    int img_col = src.cols;

    for(int i = 1; i < img_row-1; i++)
    {
        for(int j = 1; j < img_col-1; j++)
        {
            int temp = 0;
            for(int m = 0; m < 3; m++)
            {
                for(int n = 0; n < 3; n++)
                {
                    temp += src.at<uchar>(i + m - 1, j + n - 1) * S_x[m][n];
                }
            }
            gradient_x.at<int>(i, j) = abs(temp);
        }
    }
    for(int i = 1; i < img_row-1; i++)
    {
        for(int j = 1; j < img_col-1; j++)
        {
            int temp = 0;
            for(int m = 0; m < 3; m++)
            {
                for(int n = 0; n < 3; n++)
                {
                    temp += src.at<uchar>(i + m - 1, j + n - 1) * S_y[m][n];
                }
            }
            gradient_y.at<int>(i, j) = abs(temp);
        }
    }

    Mat gradient(img_row, img_col, CV_32S, Scalar(0));
    for(int i = 1; i < img_row-1; i++) {
        for (int j = 1; j < img_col - 1; j++) {
            gradient.at<int>(i, j) = gradient_x.at<int>(i, j) + gradient_y.at<int>(i, j);
        }
    }
    Mat dst;
    gradient.convertTo(dst, CV_8UC1);
    imshow("grdn", dst);
    waitKey(0);
}

void NoneMaximumSuppression(const Mat & gradient_x, const Mat & gradient_y, Mat & dst){
    float neighborPixel1 = 0;
    float neighborPixel2 = 0;
    float Pixel;

    int img_rows = gradient_x.rows;
    int img_cols = gradient_x.cols;

    Mat G(img_rows, img_cols, CV_32S, Scalar(0));
    Mat Theta(img_rows, img_cols, CV_8UC1, Scalar(0));

    for(int i=0; i<img_rows; i++){
        for(int j = 0; j<img_cols; j++){
            int G_x = gradient_x.at<int>(i, j);
            int G_y = gradient_y.at<int>(i, j);
            float angle;

            G.at<int>(i, j) = G_x + G_y;

            if ((G_x != 0) || (G_y != 0)) {
                angle = std::atan2((float)G_y, (float)G_x) * 180.0 / 3.14159;
            }
            else {
                angle = 0.0;
            }

            if (((angle > -22.5) && (angle <= 22.5)) ||
                ((angle > 157.5) || (angle <= -157.5))) {
                Theta.at<uchar>(i, j) = 0;
            }
            else if (((angle > 22.5) && (angle <= 67.5)) ||
                     ((angle > -157.5) && (angle <= -112.5))) {
                Theta.at<uchar>(i, j) = 45;
            }
            else if (((angle > 67.5) && (angle <= 112.5)) ||
                     ((angle > -112.5) && (angle <= -67.5))) {
                Theta.at<uchar>(i, j) = 90;
            }
            else if (((angle > 112.5) && (angle <= 157.5)) ||
                     ((angle > -67.5) && (angle <= -22.5))) {
                Theta.at<uchar>(i, j) = 135;
            }
        }
    }

    for(int i = 1; i<img_rows-1; i++){
        for(int j =1; j < img_cols-1; j++){
            if (Theta.at<uchar>(i, j) == 0) {
                neighborPixel1 = G.at<int>(i, j+1);
                neighborPixel2 = G.at<int>(i, j-1);
            } else if(Theta.at<uchar>(i, j) == 45){
                neighborPixel1 = G.at<int>(i-1, j-1);
                neighborPixel2 = G.at<int>(i+1, j+1);
            }else if(Theta.at<uchar>(i, j) == 90){
                neighborPixel1 = G.at<int>(i+1, j);
                neighborPixel2 = G.at<int>(i-1, j);
            } else if(Theta.at<uchar>(i, j) == 135){
                neighborPixel1 = G.at<int>(i-1, j+1);
                neighborPixel2 = G.at<int>(i+1, j-1);
            }

            Pixel = G.at<int>(i, j);

            if(Pixel >= neighborPixel1 && Pixel >= neighborPixel2){
                if(Pixel > 255) dst.at<uchar>(i, j) = 255;
                else dst.at<uchar>(i, j) = Pixel;
            }else{
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

}

void HysteresisRecursion(Mat & src, int x, int y, uint8_t lowThreshold, uint8_t highThreshold) {
    int img_row = src.rows;
    int img_col = src.cols;
    bool flag = false;
    int size = 1;

//    std::cout << 1;
    uint8_t value = src.at<uchar>(x, y);
    if (value > highThreshold) src.at<uchar>(x, y) = 255;
    else if(value < lowThreshold) src.at<uchar>(x, y) = 0;
    else {
        src.at<uchar>(x, y) = 0;
        for (long x1 = x - size; x1 <= x + size; x1++)
        {
            for (long y1 = y - size; y1 <= y + size; y1++)
            {
                if ((x1 < img_row) && (y1 < img_col) && (x1 >= 0) && (y1 >= 0 ) && (x1 != x) && (y1 != y))
                {
                    value = src.at<uchar>(x1, y1);
                    if(value > highThreshold){
                        weak_edge[x*img_col + y] = true;
                        flag = true;
                        break;
                    }
                }
                else continue;
            }
            if(flag) break;
        }
    }
}

void myCanny(const Mat & src, Mat & dst, uint8_t threshold1, uint8_t threshold2){
    int img_row = src.rows;
    int img_col = src.cols;
    Mat src_blur = src.clone();
    GaussianBlur(src, src_blur, cv::Size(5, 5), 1);

    Mat gradient_x(img_row, img_col, CV_32S, Scalar(0));
    Mat gradient_y(img_row, img_col, CV_32S, Scalar(0));
    SobelOpt(src_blur, gradient_x, gradient_y);

    Mat non_max_sup(img_row, img_col, CV_8UC1, Scalar(0));
    NoneMaximumSuppression(gradient_x, gradient_y, non_max_sup);

    Mat edge = non_max_sup.clone();
    for (int i = 0; i < img_row; i++)
    {
        for (int j = 0; j < img_col; j++){
            HysteresisRecursion(edge, i, j, threshold1, threshold2);
        }
    }

    for (int i = 0; i < img_row; i++)
    {
        for (int j = 0; j < img_col; j++){
            if(weak_edge[i*img_col+j]) edge.at<uchar>(i, j) = 255;
        }
    }
    dst = edge;


}

void myHoughLine(Mat & input, Mat & output, int threshold){
    Mat img_grey2;
    if(input.channels() != 1) cvtColor(input, img_grey2, COLOR_BGR2GRAY);
    else img_grey2 = input;
    int R_min = 10;
    int R_max = cvRound(sqrt(pow(input.rows, 2) + pow(input.cols, 2)));
    float stepSize_r = 1;
    float stepSize_theta = 1;

    int RArrayLenth = cvRound((R_max-R_min)/stepSize_r);
    int ThetaArrayLenth = cvRound(180/stepSize_theta);


    long **R_Theta=new long*[ThetaArrayLenth];
    for(int i=0;i<ThetaArrayLenth;i++)
        R_Theta[i]=new long[RArrayLenth]();

    Mat frame_canny;
    Canny(img_grey2, frame_canny,  100, 200);
    imshow("canny", frame_canny);
    waitKey(0);

    for(int i = 0; i < frame_canny.rows; i++){
        for(int j = 0; j<frame_canny.cols;j++){
            if(frame_canny.at<uchar>(i, j)==255){
                for(int theta = 0; theta < ThetaArrayLenth; theta ++){
                    float r = cos((theta*stepSize_theta)/180*3.14159)*i+sin((theta*stepSize_theta)/180*3.14159)*j;
//                    std::cout << "r:"<< r << std::endl;
                    if(r>R_min&&r<R_max){
//                        std::cout << "0" <<std::endl;
                        R_Theta[theta][cvRound((r-R_min)/stepSize_r)] +=1;
                    }
                }
            }
            else continue;
        }
    }

/*    for(int i = 0; i < ThetaArrayLenth; i++) {
        for (int j = 0; j < RArrayLenth; j++) {
            std::cout << R_Theta[i][j] <<std::endl;
        }
    }*/
    Mat canny_Img;
    cvtColor(input, canny_Img, COLOR_GRAY2BGR);

    for(int i = 0; i < ThetaArrayLenth; i++){
        for(int j = 0; j < RArrayLenth; j++){
            if(R_Theta[i][j] > threshold){
                std::cout << R_Theta[i][j] <<std::endl;

                float theta = i * stepSize_theta*3.14159/180;
                float r = R_min + j * stepSize_r;

                int x1 = 0;
                int y1 = cvRound(r/sin(theta));
                int x2 = cvRound(r/cos(theta));
                int y2 = 0;
                line(canny_Img, Point(y1, x1), Point(y2, x2), Scalar(0, 0, 255), 2, 4, 0);
            }
        }
    }
    output = canny_Img;

}

void myHoughCircle(Mat & input, Mat & output, int threshold) {
    Mat img_grey2;
    if (input.channels() != 1) cvtColor(input, img_grey2, COLOR_BGR2GRAY);
    else img_grey2 = input;
    int R_min = 10;
    int R_max = cvRound(sqrt(pow(input.rows, 2) + pow(input.cols, 2))/2);
    float stepSize_r = 1;
    float stepSize_theta = 1;

    int RArrayLenth = cvRound((R_max - R_min) / stepSize_r);
    int ThetaArrayLenth = cvRound(360 / stepSize_theta);


    long ***X_Y_R = new long **[input.rows];
    for (int i = 0; i < input.rows; i++) {
        X_Y_R[i] = new long *[input.cols];
        for (int j = 0; j < input.cols; j++) {
            X_Y_R[i][j] = new long[RArrayLenth]();
        }
    }


    Mat frame_canny;
    Canny(img_grey2, frame_canny, 50, 100);
/*    imshow("canny", frame_canny);
    waitKey(0);*/

    for (int i = 0; i < frame_canny.rows; i++) {
        for (int j = 0; j < frame_canny.cols; j++) {

            if (frame_canny.at<uchar>(i, j) == 255) {
//                std::cout <<  "1" <<std::endl;
                for (int tht = 0; tht < ThetaArrayLenth; tht++) {
                    for (int R = 0; R < RArrayLenth; R++) {
                        float r = R_min + R * stepSize_r;
                        float theta = tht * stepSize_theta * 3.1415926 / 180;

                        int center_x = cvRound(i - r * cos(theta));
                        int center_y = cvRound(j - r * sin(theta));

                        if (center_x > 0 && center_x < frame_canny.rows && center_y > 0 &&
                            center_y < frame_canny.cols) {
//                            std::cout<<X_Y_R[center_x][center_y][cvRound(r)]<<std::endl;
                            X_Y_R[center_x][center_y][cvRound(r)] += 1;
                        }
                    }
                }
            }
            else continue;
//            std::cout <<  "frame rows:"<< frame_canny.rows<<"i="<<i<< "  frame cols:"<< frame_canny.cols<<"j="<< j <<std::endl;
        }
    }

/*    for(int i = 0; i < ThetaArrayLenth; i++) {
        for (int j = 0; j < RArrayLenth; j++) {
            std::cout << R_Theta[i][j] <<std::endl;
        }
    }*/
    Mat canny_Img;
    cvtColor(input, canny_Img, COLOR_GRAY2BGR);
//    imshow("canny", canny_Img);
//    waitKey(0);


    for(int i = 0; i < canny_Img.rows; i++){
        for(int j = 0; j < canny_Img.cols; j++){
            for(int k = 0; k < RArrayLenth; k++)
                if(X_Y_R[i][j][k] > threshold){
                    std::cout << X_Y_R[i][j][k] <<std::endl;
                    float r = R_min + k * stepSize_r;
                    circle(canny_Img, Point(j, i), cvRound(r), Scalar(0, 0, 255), 2, 4, 0);
                }
        }
    }
    output = canny_Img;
}

int main(){

    Mat frIn1, frIn2, frIn3;
    Mat img_grey1, img_grey2, img_grey3, frame_canny, frame_line, frame_circle;
    frIn1 = imread("../lena.tif", -1);
    frIn2 = imread("../airport.tif", -1);
    frIn3 = imread("../polymercell.tif", -1);
    if(frIn1.channels() != 1) cvtColor(frIn1, img_grey1, COLOR_BGR2GRAY);
    else img_grey1 = frIn1;

    if(frIn2.channels() != 1) cvtColor(frIn2, img_grey2, COLOR_BGR2GRAY);
    else img_grey2 = frIn2;

    if(frIn3.channels() != 1) cvtColor(frIn1, img_grey3, COLOR_BGR2GRAY);
    else img_grey3 = frIn3;

/*    myCanny(img_grey1, frame_canny, 50, 100);
    imshow("myCanny result", frame_canny);
    waitKey(0);*/

    /*myHoughLine(frIn2, frame_line, 195);
    imshow("Houghline result", frame_line);*/

    /*imshow("Houghcircle-", frIn3);
    waitKey(0);*/
    myHoughCircle(frIn3, frame_circle, 200);
    imshow("Houghcircle result", frame_circle);
    waitKey(0);

}
