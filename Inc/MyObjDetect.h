//
// Created by 翰樵 on 2020/12/3.
//

#ifndef OPENCV_MYOBJDETECT_H
#define OPENCV_MYOBJDETECT_H

#include "KCF_BP_Kalman_Tracker.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
class Detectbody{
    cv::Rect target;
    int h;
    int w;
    int area;
    std::string option;
    bool flag;
    double angle;
    double target_ratio;
    bool selectObject;
    char method;
    cv::Mat result;
    KCF_BP_Kalman_Tracker* tracker;


    cv::Point Leg_bottom;
    cv::Point Bottom_center;
    cv::Point2f target_center;
    cv::Size Original_size;

    void getLowerBody_CascadeClassifier(cv::Mat src);
    void getLowerBody_simpleDIP(cv::Mat frame);
    void getLowerBody_KCF_BPKalmanFusion(const cv::Mat& frame);

public:

    Detectbody()= default;;
    explicit Detectbody(char Plan, char opt2);
    cv::Rect getTarget(cv::Mat src);
    void img_debug();

    ~Detectbody()= default;;
};


#endif //OPENCV_MYOBJDETECT_H
