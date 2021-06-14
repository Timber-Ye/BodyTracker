//
// Created by 翰樵 on 2020/12/13.
//

#ifndef OPENCV_KCF_BP_KALMAN_TRACKER_H
#define OPENCV_KCF_BP_KALMAN_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>


class KCF_BP_Kalman_Tracker{
    cv::Mat image;
    bool backprojMode;
    int trackObject;

    double dSimilarity; //!<用巴氏系数计算两直方图的相似性
    double Threshold; //!<KCF是否有效识别的阈值
    int point_num;
    std::vector<cv::Point2f> vec;
    cv::Mat trainingData_In, trainingData_Out;
    cv::Mat layerSizes; //!<构建神经网络
    cv::Mat hist1, hist2;
    cv::Mat backproj;

    cv::Point Origin;
    cv::Rect Selected_Rect;
    cv::RotatedRect trackBox;
    cv::Rect trackWindow, pre_trackWindow;

    cv::Ptr<cv::ml::ANN_MLP> ann;
    cv::Mat BP_result;
    cv::Point2f  BP_point;

    cv::RNG rng;
    float T=0.9; //!<采样周期
    float v_x, v_y, a_x, a_y;
    int stateNum; //!<状态数， 包括(x,y,dx,dy,d(dx),d(dy))
    int measureNum; //!<测量值2×1向量(x,y)
    cv::KalmanFilter KF;
    cv::Mat measurement;
    cv::Point2f pre_point;

public:
    bool showHist;
    bool selectObject;
    void diff_Mat(const cv::Mat& in_Image, const cv::Mat& out_Image); //!<构建差值矩阵
    cv::Point2f neural_networks( cv::Mat in_trainData, cv::Mat out_trainData ); //!<神经网络训练及预测
    cv::Point2f kalman_filter( const cv::Point2f& measure_point ); //!<卡尔曼预测
    cv::Mat drawHist(cv::Mat& src, const cv::Rect& selected_Rect,const std::string& WINDOW_NAME); //!<绘制直方图
    inline bool KCF_WORKS(double d, double t); //!<用巴氏系数计算两直方图的相似性.当巴氏系数小于阈值t时，未发生遮挡
    cv::Rect KCF_Track(cv::Mat src);
    cv::Rect camshift_Track(cv::Mat src);
    void Select_target(cv::Mat src);


    explicit KCF_BP_Kalman_Tracker(double t);

};
#endif //OPENCV_KCF_BP_KALMAN_TRACKER_H
