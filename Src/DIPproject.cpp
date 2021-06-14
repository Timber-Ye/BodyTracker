#include <stdlib.h>
#include <stddef.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ros/ros.h"

#include <math.h>

//#define READIMAGE_ONLY
#ifndef READIMAGE_ONLY
#include <geometry_msgs/Twist.h>
#endif

#define min_area 10 

using namespace cv;
using namespace std;

bool flag=1;
bool img_debug = true;
bool picture_debug = false;
int h_max=124;
int h_min=100;
int s_max=255;
int s_min=43;
int v_max=255;
int v_min=46;
int lowthreshold[3]={h_min,s_min,v_min};//定义范围
int highthreshold[3]={h_max,s_max,v_max};

//机器人速度
void Speed(int cx,Mat& src,geometry_msgs::Twist& cmd_red)
{  
    int centerL=ceil(src.cols/4)-round(src.cols/30);
    int centerR=ceil(src.cols/4)+round(src.cols/30);
    int centerM=ceil(src.cols/4);
    cout<<"src.cols"<<" "<<src.cols/4<<endl;
    cout<<"cs"<<" "<<cx<<endl;
    if(cx<centerL)
    {     
		cmd_red.angular.z = 0.1;		
    }
    else if(cx>centerR)
    {
        cmd_red.angular.z = -0.1;
    }
    else if(cx>=centerL && cx<=centerR)
    //else if(cx==centerM)
    {
        cmd_red.angular.z = 0;
        cmd_red.linear.x = 0.2;
    }
    //return cmd_red;
}

//基础图像处理
int Mymain(Mat& frame)
{
    int w = frame.cols;
    int h = frame.rows;
//    cv::imshow("frame", frame);
    cv::GaussianBlur(frame, frame_blurred,cv::Size(3,3),0.3);
//    cv::threshold(frame, frame_bi, 170, 255, cv::THRESH_BINARY);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Rect poi_1(0, h/2, w, h/2);
    int debug_x = 220;
    int debug_y = 150;
    cv::Rect poi_2(debug_x, debug_y, w-2*debug_x, h/2-debug_y);
    mask(poi_1).setTo(255);
    mask(poi_2).setTo(255);

    cv::Mat frame_Msked, frame_bi_Msked;
    frame.copyTo(frame_Msked, mask);
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(frame_gray, frame_bi, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY, 71, 10);
    frame_bi.copyTo(frame_bi_Msked, mask);
    cv::imshow("frame_bi", frame_bi_Msked);

    cv::Mat Morphed;
    cv::Mat element1(7,7,CV_8U,cv::Scalar(1));
    cv::dilate(frame_bi_Msked, Morphed, element1);
    cv::Mat element2(7,7,CV_8U,cv::Scalar(1));
    cv::morphologyEx(Morphed, Morphed, cv::MORPH_CLOSE, element2);

    cv::Rect poi_3(0, 0, w, debug_y);
    cv::Rect poi_4(0, debug_y, debug_x, h/2-debug_y);
    cv::Rect poi_5(w-debug_x, debug_y, debug_x, h/2-debug_y);
    Morphed(poi_3).setTo(255);
    Morphed(poi_4).setTo(255);
    Morphed(poi_5).setTo(255);
    cv::threshold(Morphed, Morphed, 127, 255, cv::THRESH_BINARY_INV);
    cv::imshow("frame_Morphed", Morphed);

    vector<vector<cv::Point> >contours;
    cv::findContours(Morphed,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    int target_index = 0;
    int target_ratio = 1;
    cv::RotatedRect target;
    for(int i=0;i<contours.size();i++){
        float area = cv::contourArea(contours[i]);
        if(area<200 || area>h*w*0.3) continue;
        else{
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            float ratio = rect.size.height/rect.size.width;
            std::cout<<"ratio:"<<ratio<<std::endl;
            if(ratio > target_ratio){
                target_ratio = ratio;
                target = rect;
                std::cout<<"!"<<std::endl;
            }
            else continue;
        }
    }
    cv::Point2f target_center = target.center;

    cv::Point Bottom_center(w/2, h);
    float angle = atan2(abs(target_center.x-Bottom_center.x), h-target_center.y);

    if(img_debug){
        stringstream strStream1;
        strStream1<< "Center: "<<target_center;
        string txt = strStream1.str();

        stringstream strStream2;
        strStream2<< "Angle: "<<angle;
        string txt2 = strStream2.str();

        cv::circle(frame_Msked, target_center, 5, cv::Scalar(0,127,0), 2);
        cv::line(frame_Msked, Bottom_center, target_center, cv::Scalar(0,255,127), 2);
        cv::putText(frame_Msked, txt, cv::Point(0, h-20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
        cv::putText(frame_Msked, txt2, cv::Point(0, h-50), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
    }
    cv::imshow("POI", frame_Msked);
    return target_center.x;
}

int main(int argc, char **argv)
{
	ROS_WARN("*****START*****");
	ros::init(argc, argv, "trafficLaneTrack"); //初始化ROS节点
	ros::NodeHandle n;

	//Before the use of camera, you can test ur program with images first: imread()
	VideoCapture capture;
	capture.open(1); //打开zed相机，如果要打开笔记本上的摄像头，需要改为0
	waitKey(100);
	if (!capture.isOpened())
	{
		printf("fail to open!\n");
		return 0;
	}
	waitKey(10);
#ifndef READIMAGE_ONLY
	//ros::Rate loop_rate(10);//定义速度发布频率
	ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/smoother_cmd_vel", 5); //定义dashgo机器人的速度发布器
#endif
	Mat frame_l, Orgframe, frame_r;
    int h, w;
    //ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/smoother_cmd_vel", 5); //定义dashgo机器人的速度发布器
	while (ros::ok())
	{
		capture.read(Orgframe);
		//Mat frame=imread("/home/wby/diptwo_ws/src/image_pkg/src/blue.jpg");//当前帧图片
		imshow("src", Orgframe);
        if(Orgframe.empty())           
        {
            cout<<"fail to open the picture"<<endl;
            return 0;
        }
        h = Orgframe.rows;
        w = Orgframe.cols;
        frame_l = Orgframe(Rect(0,0,w/2,h));
        //imshow("frame", frame);
        frame_r = Orgframe(Rect(w/2, 0, w/2, h));
        int left,right;
        namedWindow("window",WINDOW_AUTOSIZE);//窗口

        createTrackbar("h_max","window",&h_max,180);
        createTrackbar("h_min","window",&h_min,180);
        createTrackbar("s_max","window",&s_max,255);
        createTrackbar("s_min","window",&s_min,255);
        createTrackbar("v_max","window",&v_max,255);
        createTrackbar("v_min","window",&v_min,255);

        left=Mymain(frame_l);
        imshow("frame_l",frame_l);
        if(left==-1) continue;        
        right=Mymain(frame_r); 
        imshow("frame_r",frame_r);
        if(right==-1) continue;
        int center=round((left+right)/2);
        cout<<"frame_l.cols"<<" "<<frame_l.cols<<endl;
        cout<<"center:"<<center<<endl;
        cout<<"left"<<" "<<left<<" "<<"right"<<" "<<right<<endl;

#ifndef READIMAGE_ONLY
		//以下代码可设置机器人的速度值，从而控制机器人运动
		geometry_msgs::Twist cmd_red;
		cmd_red.linear.x = 0;
		cmd_red.linear.y = 0;
		cmd_red.linear.z = 0;
		cmd_red.angular.x = 0;
		cmd_red.angular.y = 0;
		cmd_red.angular.z = 0;	
        //Speed(rect.center.y,frame,cmd_red); //控制机器人速度
        Speed(center,Orgframe,cmd_red);
        cout<<"speedz"<<cmd_red.angular.z<<endl;
        pub.publish(cmd_red);
#endif
		ros::spinOnce();
		waitKey(5);
	}
	return 0;
}