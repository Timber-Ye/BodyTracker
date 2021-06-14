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
#include "MyObjDetect.h"

#include <math.h>

//#define READIMAGE_ONLY
#ifndef READIMAGE_ONLY
#include <geometry_msgs/Twist.h>
#endif

#define min_area 10 

using namespace cv;
using namespace std;

//定义全局变量
char plan ='B';
bool flag=0;
bool img_debug = true;
bool img_debug_x = false;
bool picture_debug = false;

int img_bottom_mask=0;
int debug_x = 220;
int debug_y = 150;
int element1_size=7;
int element2_size=9;
int BS=101;

int h_max=124;
int h_min=100;
int s_max=255;
int s_min=43;
int v_max=255;
int v_min=46;
int lowthreshold[3]={h_min,s_min,v_min};
int highthreshold[3]={h_max,s_max,v_max};
float target_area=0.0;
//int last_center=-1;
float last_z=0.0;
int center=0;
int debug=0;
int leg_bottom_y=0;


//控制机器人速度
void Speed(int cx,Mat& src,geometry_msgs::Twist& cmd_red)
{  
    //旋转
    int centerM=ceil(src.cols/4);
    int centerL=ceil(src.cols/4)-round(src.cols/30);
    int centerR=ceil(src.cols/4)+round(src.cols/30);
    int centerLL=ceil(src.cols/8);
    int centerRR=ceil(src.cols/8)+centerM;
    
    if(!flag)
    {
        cmd_red.angular.z=last_z;
    }
    else if(flag)
    {
        if(cx<centerLL)
        {
            cmd_red.angular.z = 0.3;	
        }
        else if(cx<centerL && cx>=centerLL)
        {     
            cmd_red.angular.z = 0.2;	
        }
        else if(cx>=centerL && cx<=centerR)
        {
            cmd_red.angular.z = 0;  
        }
        else if(cx>centerR && cx<=centerRR)
        {
            cmd_red.angular.z = -0.2;
        }
        else if(cx>centerRR)
        {
            cmd_red.angular.z = -0.3;	
        }    
    }

    //前进后退
    int delta_y=src.rows-leg_bottom_y;
    if(!flag)
    {
        cmd_red.linear.x=0;
    }
    else if(flag)
    {
        if(delta_y<80 && delta_y>=30)
        {
            cmd_red.linear.x=0.2;
        }
        else if(delta_y>=80)
        {
            cmd_red.linear.x=0.4;
        }
        else if(delta_y<30 && delta_y>0)
        {
            cmd_red.linear.x=0;
        }
        else if(delta_y==0)
        {
            cmd_red.linear.x=-0.2;
        }
    }  
    last_z=cmd_red.angular.z;
    cout<<"last_z: "<<last_z<<endl;
    cout<<"angular.z: "<<cmd_red.angular.z<<endl;
}

//单图像处理
int MysingleDIP(Mat& frame)
{
    int w = frame.cols;
    int h = frame.rows;
    cv::Mat frame_blurred;
    cv::Mat frame_gray;
    cv::Mat frame_bi;
//    cv::imshow("frame", frame);
    cv::GaussianBlur(frame, frame_blurred,cv::Size(3,3),0.3);
//    cv::threshold(frame, frame_bi, 170, 255, cv::THRESH_BINARY);

    cv::Mat mask_1 = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Mat mask_2;
    cv::Rect poi_1(0, h/2, w, h/2-img_bottom_mask);
    cv::Rect poi_2(debug_x, debug_y, w-2*debug_x, h/2-debug_y);
    mask_1(poi_1).setTo(255);
    mask_1(poi_2).setTo(255);
    cv::bitwise_not(mask_1, mask_2);

    cv::Mat frame_Msked, frame_bi_Msked;
    frame.copyTo(frame_Msked, mask_1);
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(frame_gray, frame_bi, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY, BS, 10);
    frame_bi.copyTo(frame_bi_Msked, mask_1);
    // cv::imshow("frame_bi", frame_bi_Msked);

    cv::Mat Morphed;
    cv::Mat element1(element1_size,element1_size,CV_8U,cv::Scalar(1));
    cv::dilate(frame_bi_Msked, Morphed, element1);
    cv::Mat element2(element2_size,element2_size,CV_8U,cv::Scalar(1));
    cv::morphologyEx(Morphed, Morphed, cv::MORPH_CLOSE, element2);
    cv::bitwise_or(Morphed, mask_2, Morphed);

    cv::threshold(Morphed, Morphed, 127, 255, cv::THRESH_BINARY_INV);
    cv::imshow("frame_Morphed", Morphed);

    vector<vector<cv::Point> >contours;
    cv::findContours(Morphed,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    cv::Rect target;
    cv::Rect candidate1, candidate2;
    float target_ratio = 0.0;
    float candidate1_ratio = 1.0;
    float candidate2_ratio = 1.0;
    float candidate1_area, candidate2_area;
    for(int i=0;i<contours.size();i++){
        float area = cv::contourArea(contours[i]);
        if(area < 1200 || area > h * w * 0.15) continue;
        else{
            cv::Rect rect = cv::boundingRect(contours[i]);
            float ratio = (float)rect.height/rect.width;
            if(img_debug_x){
                stringstream stringStream;
                stringStream << i+1 <<":";
                string debug_txt = stringStream.str();
                cv::rectangle(frame_Msked, rect, cv::Scalar(255, 0, 0), 2);
                cout<<"ratio["<<i+1<<"]:  " <<ratio<<"  Area["<<i+1<<"]:   "<<area<<endl;
                cv::putText(frame_Msked, debug_txt, cv::Point(rect.x-13, rect.y-5), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
                cv::waitKey(0);
            }

            if(ratio > candidate2_ratio){
                if(ratio > candidate1_ratio){
                    candidate2 = candidate1;
                    candidate2_ratio = candidate1_ratio;
                    candidate2_area = candidate1_area;
                    candidate1_ratio = ratio;
                    candidate1 = rect;
                    candidate1_area = area;
                }
                else{
                    candidate2 = rect;
                    candidate2_ratio = ratio;
                    candidate2_area = area;
                }
            }
            else continue;
        }
    }





    if(candidate1_ratio > 1 && candidate2_ratio > 1){
        if(candidate1.x >= candidate2.x){
            target = candidate1;
            target_ratio = candidate1_ratio;
            target_area = candidate1_area;
            if(img_debug){
                stringstream stringStream;
                stringStream << "R";
                string debug_txt = stringStream.str();
                cv::rectangle(frame_Msked, candidate2, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame_Msked, debug_txt, cv::Point(candidate2.x, candidate2.y-5),
                cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

                stringstream stringStream_2;
                stringStream_2 << "L";
                debug_txt = stringStream_2.str();
                cv::putText(frame_Msked, debug_txt, cv::Point(target.x, target.y-5),
                            cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

            }
        }
        else{
            target =candidate2;
            target_ratio = candidate2_ratio;
            target_area = candidate2_area;

            if(img_debug){
                stringstream stringStream;
                stringStream << "R";
                string debug_txt = stringStream.str();
                cv::rectangle(frame_Msked, candidate1, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame_Msked, debug_txt, cv::Point(candidate2.x, candidate2.y-5),
                            cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

                stringstream stringStream_2;
                stringStream_2 << "L";
                debug_txt = stringStream_2.str();
                cv::putText(frame_Msked, debug_txt, cv::Point(target.x, target.y-5),
                            cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

            }
        }
    }
    else if(candidate1_ratio > 1){
        target = candidate1;
        target_ratio = candidate1_ratio;
        target_area = candidate1_area;

        stringstream stringStream_2;
        stringStream_2 << "Single Leg";
        string debug_txt = stringStream_2.str();
        cv::putText(frame_Msked, debug_txt, cv::Point(target.x, target.y-5),
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

    }

    float angle=0.0;
    float target_center_x = target.x+target.width/2;
    float target_center_y = target.y+target.height/2;
    leg_bottom_y = target_center_y + target.height/2;

    cv::Point2f target_center(target_center_x, target_center_y);
    cv::Point Bottom_center(w/2, h);
    cv::Point Leg_bottom(target_center_x, leg_bottom_y);

    if(target_center_y < h/2-50) flag = false;
    else{
        flag = true;
        angle = atan2(abs(target_center.x-Bottom_center.x), h-target_center.y);
    }

    if(img_debug){
        stringstream strStream1;
        stringstream strStream2;
        stringstream strStream3;
        stringstream strStream4;
        stringstream strStream5;
        string txt, txt2, txt3, txt4, txt5;
        if(flag){
            strStream1<< "Center: "<<target_center;
            txt = strStream1.str();

            strStream2<< "Angle: "<<angle<<"rad";
            txt2 = strStream2.str();

            strStream3<< "Area: "<<target_area;
            txt3 = strStream3.str();

            strStream4<< "Ratio: "<<target_ratio;
            txt4 = strStream4.str();

            strStream5<< "Leg_bottom_y: "<<h-leg_bottom_y;
            txt5 = strStream5.str();

            cv::line(frame_Msked, Bottom_center, target_center, cv::Scalar(0,255,127), 2);
            cv::rectangle(frame_Msked, target, cv::Scalar(255, 0, 0), 2);
            cv::circle(frame_Msked, target_center, 5, cv::Scalar(0,127,0), 2);
            cv::circle(frame_Msked, Leg_bottom, 5, cv::Scalar(0,127,0), 2);
        }

        else{
            strStream1<< "Center: "<<"Not Found!";
            txt = strStream1.str();

            strStream2<< "Angle: "<<"Not Found!";
            txt2 = strStream2.str();

            strStream3<< "Area: "<<"Not Found!";
            txt3 = strStream3.str();

            strStream4<< "Ratio: "<<"Not Found!";
            txt4 = strStream4.str();

            strStream5<< "Leg_bottom_y: "<<"Not Found!";
            txt5 = strStream5.str();

        }
        cv::putText(frame_Msked, txt, cv::Point(0, h-20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
        cv::putText(frame_Msked, txt2, cv::Point(0, h-40), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
        cv::putText(frame_Msked, txt3, cv::Point(0, h-60), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
        cv::putText(frame_Msked, txt4, cv::Point(0, h-80), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
        cv::putText(frame_Msked, txt5, cv::Point(0, h-100), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
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

        //分割左右目
        frame_l = Orgframe(Rect(0,0,w/2,h));
        //imshow("frame", frame);
        frame_r = Orgframe(Rect(w/2, 0, w/2, h));

        if(plan=='A'){
            int left,right;

            //分别处理
            left=MysingleDIP(frame_l);
            //imshow("frame_l",frame_l);
            right=MysingleDIP(frame_r);
            //imshow("frame_r",frame_r);

            //汇总中心纵坐标
            center=round((left+right)/2);
            /*if(!flag)
            {
                center=last_center;
            }*/
        }
        else if(plan=='B'){
            Detectbody detector(frame_l, 'l');
            cv::Rect target = detector.getTarget();
            if(img_debug) detector.img_debug();
            else{
                cv::rectangle(frame_l, target, cv::Scalar(255, 0, 0), 2);
                cv::imshow("Result", frame_l);
            }
            center = target.x;
        }

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
        cout<<"speedz: "<<cmd_red.angular.z<<endl;
        cout<<"speedx: "<<cmd_red.linear.x<<endl;
        if(debug==1)
        {
            cmd_red.linear.x=0;
        }
        pub.publish(cmd_red);
#endif
		ros::spinOnce();
        //last_center=center;
        cout<<" "<<endl;
		waitKey(0);
		char k;
		k = waitKey(1);
	}
	return 0;
}