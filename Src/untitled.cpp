#include <iostream>
#include "../Inc/MyObjDetect.h"
bool Select = false;

using namespace std;

bool img_debug = true; // 找到目标之后的绘图
bool picture_debug = false; // 使用图片还是视频作为输入

int main(){
    cv::Mat frame, frame_blurred, frame_bi, frame_gray;
    cv::VideoCapture capture;
    cv::Rect target;
    if(picture_debug) frame=cv::imread("../img/leg2.png");
    else{
        capture.open(0);
        if(!capture.isOpened())
        {
            std::cout << "Camera not open.\n";
            exit(EXIT_FAILURE);
        }
    }
    Detectbody detector('C','f');
    KCF_BP_Kalman_Tracker tracker(0.6);
    while(true){
        if(!picture_debug) capture>>frame;
        if(!Select){
            cout << "press q to select current Image" << endl;
            cv::imshow("first", frame);
            char key = cv::waitKey(1);
            if (key != 'q')  // 按c键跳帧
            {
                continue;
            }
            if (key == 'q') {
                cv::destroyWindow("first");
                tracker.Select_target(frame);
                if (tracker.selectObject) Select=true;
            }
        }
        else{
            target = tracker.camshift_Track(frame);
            cv::rectangle(frame, target, cv::Scalar(255, 0, 0), 2);
            cv::imshow("Result", frame);
        }

        char k;
        if(picture_debug) k = cv::waitKey(1);
        else {
            k = cv::waitKey(1);
        }
        if(k==27) break;
    }
}