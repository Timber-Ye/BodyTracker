//
// Created by 翰樵 on 2020/10/25.
//
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#define RED 0
#define YELLOW 1
#define  BLUE 2
#define GREEN 3

using namespace cv;
using namespace std;



Mat frIn, img_blur, img_gray, img_bi, ROI;
Mat kernel_1 = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
Mat Image = Mat::zeros(1, 660, CV_8UC3);

void myCvtColor(const Mat & src, Mat & dst);
float findMax(const float & x, const float & y, const float & z);
float findMin(const float & x, const float & y, const float & z);
void myInRange(const Mat & src, Mat & dst, int * upperb, int * lowerb);
void myBitWiseAnd(const Mat & srcC3, const Mat & srcC1, Mat & dst);
void onChangeTrackBar(int, void* usrdata);
int myCalhist(const Mat & src);
void threshCallBack(int, void* usrdata);
bool myJudge(const int a[], const int Array[][3]);
void drawHist(Mat & OutputArray, int count, int rank, int max);


void myCvtColor(const Mat & src, Mat & dst){
//////////转换到HSV空间///////////

/*!
 *
 * @param src : 输入三通道图像
 * @param dst ： 输出三通道图像
 */
    int width = src.cols;
    int height = src.rows;
    Mat hsv_img (height, width, CV_8UC3, Scalar::all(0));
    if(src.channels() != 3){
        cerr << "Format error. Three Channels Expected!";
    }

    else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float B = src.at<Vec3b>(i, j)[0];
                float G = src.at<Vec3b>(i, j)[1];
                float R = src.at<Vec3b>(i, j)[2];

                float B_ = B / 255.0;
                float G_ = G / 255.0;
                float R_ = R / 255.0;

                float max = findMax(B, G, R);
                float min = findMin(B, G, R);
                float max_ = findMax(B_, G_, R_);

                float v = max;
                float s;
                float h;

                if (v != 0) s = cvRound((max - min) / max_);
                else s = 0;
                if (abs(max - min) < 1e-3) h = 0;
                else if (abs(R - max) < 1e-3) h = 60 * (G - B) / (max - min);
                else if (abs(G - max) < 1e-3) h = 60 * (B - R) / (max - min) + 120;
                else if (abs(B - max) < 1e-3) h = 60 * (R - G) / (max - min) + 240;

                if (h < 0) h += 360;

                hsv_img.at<Vec3b>(i, j)[0] = cvRound(h/2);
                hsv_img.at<Vec3b>(i, j)[1] = cvRound(s);
                hsv_img.at<Vec3b>(i, j)[2] = cvRound(v);
            }
        }
    }
    dst = hsv_img;
}

float findMax(const float & x, const float & y, const float & z){
    float m = 0;
    m = x > m? x:m;
    m = y > m? y:m;
    m = z > m? z:m;
    return  m;
}

float findMin(const float & x, const float & y, const float & z){
    float m = 255;
    m = x < m? x:m;
    m = y < m? y:m;
    m = z < m? z:m;
    return  m;
}



void myInRange(const Mat & src, Mat & dst, int * upperb, int * lowerb) {
//////////基于HSV数值的颜色分割///////////

/*!
 *
 * @param src : 输入三通道图像
 * @param dst ： 输出二值图像
 * @param upperb ： 上限阈值(长度为3的数组)
 * @param lowerb : 下限阈值(长度为3的数组)
 */
    int width = src.cols;
    int height = src.rows;

    int U_h = upperb[0];
    int U_s = upperb[1];
    int U_v = upperb[2];
    int L_h = lowerb[0];
    int L_s = lowerb[1];
    int L_v = lowerb[2];

    Mat Mask_img(height, width, CV_8U, Scalar::all(0));
    if (src.channels() != 3) {
        cerr << "Format error. Three Channels Expected!";
    } else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int H = src.at<Vec3b>(i, j)[0];
                int S = src.at<Vec3b>(i, j)[1];
                int V = src.at<Vec3b>(i, j)[2];

                if (H <= U_h && H >= L_h && S <= U_s && S >= L_s && V <= U_v && V >= L_v)
                    Mask_img.at<uchar>(i, j) = 255;

                else Mask_img.at<uchar>(i, j) = 0;
            }
        }

        dst = Mask_img;
    }
}



void myBitWiseAnd(const Mat & srcC3, const Mat & srcC1, Mat & dst){
//////////按位与运算///////////

/*!
 *
 * @param srcC3 : 输入三通道图像
 * @param srcC1 ： 输入二值图像
 * @param dst ： 输出三通道图像
 */
    int width1 = srcC3.cols;
    int height1 = srcC3.rows;
    int width2 = srcC1.cols;
    int height2 = srcC1.rows;
    Mat masked_img(height1, width1, CV_8UC3, Scalar::all(0));

    if (srcC3.channels() != 3 || srcC1.channels() != 1) {
        cerr << "Format error. Three Channels as well as Single Channel Expected!";
    } else if(width1 != width2 || height1 != height2){
        cerr << "Format error. Same Size Expected!";
    }

    else {
        for (int i = 0; i < height1; i++) {
            for (int j = 0; j < width1; j++) {
                if (srcC1.at<uchar>(i, j) == 255) {
                    masked_img.at<Vec3b>(i, j)[0] = srcC3.at<Vec3b>(i, j)[0];
                    masked_img.at<Vec3b>(i, j)[1] = srcC3.at<Vec3b>(i, j)[1];
                    masked_img.at<Vec3b>(i, j)[2] = srcC3.at<Vec3b>(i, j)[2];
                }
            }
        }
    }
    dst = masked_img;
}



void onChangeTrackBar(int, void* usrdata) {
    /////////createTrackbar的回调函数////////////
    Mat hsv_img;
    imshow("strawberries-RGB", Image);
    myCvtColor(frIn, hsv_img);
    imshow("Org_hsv", hsv_img);
    int u_h = getTrackbarPos("u_h", "strawberries-RGB");
    int u_s = getTrackbarPos("u_s", "strawberries-RGB");
    int u_v = getTrackbarPos("u_v", "strawberries-RGB");
    int l_h = getTrackbarPos("l_h", "strawberries-RGB");
    int l_s = getTrackbarPos("l_s", "strawberries-RGB");
    int l_v = getTrackbarPos("l_v", "strawberries-RGB");

    int upper[3] = {u_h, u_s, u_v};
    int lower[3] = {l_h, l_s, l_v};

    myInRange(hsv_img, img_bi, upper, lower);
    /*cout<<"upper=["<<upper[0]<<" "<<upper[1]<<" "<<upper[2]<<"]    "
            <<"lower=["<<lower[0]<<" "<<lower[1]<<" "<<lower[2]<<"]"<<endl;*/
    dilate(img_bi, img_bi, kernel_1);

    myBitWiseAnd(hsv_img, img_bi, ROI);

    Mat dst;
    myBitWiseAnd(frIn, img_bi, dst);
    imshow("result", dst);
}




int myCalhist(const Mat & src){
//////////直方图统计///////////

/*!
 *
 * @param src : 输入图像
 * @return 传回主色编号
 */
    int width = src.cols;
    int height = src.rows;
    int red_count = 0;
    int green_count = 0;
    int yellow_count = 0;
    int blue_count = 0;
    int m = 0;
    int signal;

    int Red_1[2][3] = {{10, 255, 255},{0, 43, 46}};
    int Red_2[2][3] = {{180, 255, 255},{156, 43, 46}};
    int Blue[2][3] = {{124, 255, 255},{100, 43, 46}};
    int Green[2][3] = {{77, 255, 255},{35, 43, 46}};
    int Yellow[2][3] = {{34, 255, 255},{26, 43, 46}};

    Mat dstHistImage = Mat::zeros(256, 256, CV_8UC3);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int a[3] = {};

            a[0] = src.at<Vec3b>(i, j)[0];
            a[1] = src.at<Vec3b>(i, j)[1];
            a[2] = src.at<Vec3b>(i, j)[2];

            if(myJudge(a, Red_1)||myJudge(a, Red_2)){
                red_count++;
                m = m > red_count? m:red_count;
            }
            else if(myJudge(a, Blue)){
                blue_count++;
                m = m > blue_count? m:blue_count;
            }
            else if(myJudge(a, Yellow)){
                yellow_count++;
                m = m > yellow_count? m:yellow_count;
            }
            else if(myJudge(a, Green)){
                green_count++;
                m = m > green_count? m:green_count;
            }
        }
    }

    drawHist(dstHistImage, red_count, 0, m);
    drawHist(dstHistImage, blue_count, 1, m);
    drawHist(dstHistImage, green_count, 2, m);
    drawHist(dstHistImage, yellow_count, 3, m);

    imshow("Hist", dstHistImage);

    if(m == red_count){
        cout<<"RED LIGHT!"<<endl;
        signal = RED;
    }
    else if(m == blue_count){
        cout<<"BLUE LIGHT!"<<endl;
        signal = BLUE;
    }
    else if(m == yellow_count){
        cout<<"YELLOW LIGHT!"<<endl;
        signal = YELLOW;
    }
    else if(m == green_count){
        cout<<"GREEN LIGHT!"<<endl;
        signal = GREEN;
    }
    return signal;
}

bool myJudge(const int a[], const int Array[][3]){
    return a[0] <= Array[0][0] && a[0] >= Array[1][0]
           && a[1] <= Array[0][1] && a[1] >= Array[1][1]
           && a[2] <= Array[0][2] && a[2] >= Array[1][2];
}

void drawHist(Mat & OutputArray, int count, int rank, int max){
    int histHeight = OutputArray.rows;
    int scale = 64;

    int h = cvRound(histHeight - count/(float)max*histHeight);

    Point rookPoints[1][4];
    rookPoints[0][0] = Point(rank*scale, histHeight);
    rookPoints[0][1] = Point((rank + 1)*scale, histHeight);
    rookPoints[0][2] = Point((rank + 1)*scale, h);
    rookPoints[0][3] = Point(rank*scale, h);

    const Point* ppt[1] = {rookPoints[0]};
    int npt[]={4};
    if (rank ==0)
        fillPoly(OutputArray, ppt, npt, 1, Scalar(0, 0, 255));
    else if(rank == 1)
        fillPoly(OutputArray, ppt, npt, 1, Scalar(255, 0, 0));
    else if(rank == 2)
        fillPoly(OutputArray, ppt, npt, 1, Scalar(0, 255, 0));
    else if(rank == 3)
        fillPoly(OutputArray, ppt, npt, 1, Scalar(0, 255, 255));

}



void threshCallBack(int, void* usrdata){

    cvtColor(frIn, img_gray, COLOR_BGR2GRAY);
    GaussianBlur(img_gray, img_blur, Size(5, 5), 10);
    int T = getTrackbarPos("threshold", "strawberries-RGB");
    threshold(img_gray, img_bi, T, 255, THRESH_BINARY_INV);
    imshow("img_bi", img_bi);
    myBitWiseAnd(frIn, img_bi, ROI);
    imshow("ROI", ROI);
}



int main(int argc, char ** argv){
    Mat frIn_HSV, Mask, result;
    int u_h = 0;
    int u_s = 0;
    int u_v = 0;
    int l_h = 0;
    int l_s = 0;
    int l_v = 0;
    int thresh = 0;


    frIn = imread("../RGB.jpg", -1);
    namedWindow("strawberries-RGB", WINDOW_AUTOSIZE);


   /* myCvtColor(frIn, frIn_HSV);
    imshow("strawberries-HSV", frIn_HSV);
    waitKey(0);

    myInRange(frIn_HSV, Mask, upper, lower);
    imshow("strawberries-Mask", Mask);
    waitKey(0);

    myBitWiseAnd(frIn, Mask, result);
    imshow("strawberries-result", result);
    waitKey(0);

    cvtColor(frIn, frIn_HSV, COLOR_BGR2HSV);
    imshow("openCV_strawberries-HSV", frIn_HSV);
    waitKey(0);
    cout<<"openCV_cvtColor_Space"<<endl;
    cout<<frIn_HSV.at<Vec3b>(20, 20)<<endl;*/

    createTrackbar("u_h", "strawberries-RGB", &u_h, 180, onChangeTrackBar);
    createTrackbar("u_s", "strawberries-RGB", &u_s, 255, onChangeTrackBar);
    createTrackbar("u_v", "strawberries-RGB", &u_v, 255, onChangeTrackBar);
    createTrackbar("l_h", "strawberries-RGB", &l_h, 180, onChangeTrackBar);
    createTrackbar("l_s", "strawberries-RGB", &l_s, 255, onChangeTrackBar);
    createTrackbar("l_v", "strawberries-RGB", &l_v, 255, onChangeTrackBar);

    onChangeTrackBar(u_h, 0);
/*
    myCvtColor(frIn, frIn_HSV);
    myCalhist(frIn_HSV);*/


/*    adaptiveThreshold(img_gray, img_bi, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 41, 7);*/

/*    createTrackbar("threshold", "strawberries-RGB", &thresh, 255, threshBar);
    threshBar(thresh, 0);*/
    waitKey(0);

    vector<vector<Point>> contours;

    int largest_area = 0;
    int largest_contour_index = 0;

    cv::findContours(img_bi, contours, cv::noArray(), RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++)
    {
        float a = contourArea(contours[i]);
        if (a > largest_area){
            largest_area = a;
            largest_contour_index = i;
        }
    }

    Rect box = boundingRect(contours[largest_contour_index]);
    Mat roi;
    frIn(box).copyTo(roi);
    imshow("ROI", roi);
    waitKey(0);

    myCalhist(ROI);
    waitKey(0);
}