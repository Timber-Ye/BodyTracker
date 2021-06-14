//
// Created by 翰樵 on 2020/12/3.
//

#include "../Inc/MyObjDetect.h"
using namespace std;

void Detectbody::getLowerBody_CascadeClassifier(cv::Mat src){
/*    if(src.cols>1000||src.rows>1000){
        cv::resize(src, this->result, cv::Size2i(src.cols/3, src.rows/3));
    }*/

    cv::Mat frame_gray;
    cv::cvtColor(src, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    //加载分类器
    cv::CascadeClassifier lowerBodyDetector = cv::CascadeClassifier(
            "../haarcascade_lowerbody.xml");
    Original_size = lowerBodyDetector.getOriginalWindowSize();
    std::vector<cv::Rect> lowerBodies; // 输出检测到的目标( cv::Rect 类 )
    lowerBodyDetector.detectMultiScale(frame_gray, lowerBodies, 1.1,
            3,0|cv::CASCADE_SCALE_IMAGE,
            cv::Size(), cv::Size()); //多尺度检测

    if(lowerBodies.size()>0){
        int target_index = 0;
        for(size_t i=0; i < lowerBodies.size(); i++){
            if(lowerBodies[i].area() > this->area){
                target_index = i;
            }
        }
        target = lowerBodies[target_index];
        flag = true;
        area = target.area();
    }
    else{
        std::cout << "Not found!" << std::endl;
        flag = false;
    }
}

void Detectbody::getLowerBody_simpleDIP(cv::Mat frame){
    int img_bottom_mask = 0; // 图片下方掩膜宽度
    int debug_x = 220; // 感兴趣区域拓展
    int debug_y = 150; // 感兴趣区域拓展
    int BS = 101; // 自适应阈值BlockSize
    int element1_size = 7; // 腐蚀操作核尺寸
    int element2_size = 9; // 开运算核尺寸

    bool img_debug = true; // 找到目标之后的绘图
    bool img_debug_x = true; // 寻找目标过程中的绘图
    cv::Mat frame_blurred;
    cv::Mat frame_gray;
    cv::Mat frame_bi;


    cv::GaussianBlur(frame, frame_blurred,cv::Size(3,3),0.3);

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
//    cv::imshow("frame_Morphed", Morphed);

    std::vector<std::vector<cv::Point> >contours;
    cv::findContours(Morphed,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
//    cv::Rect target;
    cv::Rect candidate1, candidate2;
//    float target_ratio = 0.0;
    float candidate1_ratio = 1.0;
    float candidate2_ratio = 1.0;
    float candidate1_area, candidate2_area;
    for(int i=0;i<contours.size();i++){
        double area = cv::contourArea(contours[i]);
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
            area = candidate1_area;
            flag = true;
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
            area = candidate2_area;
            flag = true;
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
        area = candidate1_area;
        flag = true;

        stringstream stringStream_2;
        stringStream_2 << "Single Leg";
        string debug_txt = stringStream_2.str();
        cv::putText(frame_Msked, debug_txt, cv::Point(target.x, target.y-5),
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

    }
    else flag = false;

    if(img_debug){
        cv::imshow("frm_masked", frame_Msked);
    }
}


void Detectbody::getLowerBody_KCF_BPKalmanFusion(const cv::Mat& frame){
    if(!selectObject){
        cout << "press q to select current Image" << endl;
        cv::imshow("first", frame);
        char key = cv::waitKey(1);
        if (key != 'q')  // 按c键跳帧
        {
            flag = false;
            return;
        }
        if (key == 'q'){
            cv::destroyWindow("first");
            tracker->Select_target(frame);
            if(!target.empty()) flag = true;
            return;
        }
    }
    else{
        target = tracker->camshift_Track(frame);
    }
}


void Detectbody::img_debug(){

    std::stringstream strStream1;
    std::stringstream strStream2;
    std::stringstream strStream3;
    std::stringstream strStream4;
    std::stringstream strStream5;
    std::string txt, txt2, txt3, txt4, txt5;

    if(flag){
        target_center.x = target.x+target.width/2;
        target_center.y = target.y+target.height/2;
        Leg_bottom.x = target_center.x;
        Leg_bottom.y = target_center.y+target.height/2;
        target_ratio = target.height/target.width;
        angle = atan2(abs(target_center.x-Bottom_center.x), h-target_center.y)*180/CV_PI;

        strStream1 << "Center: "<<target_center;
        txt = strStream1.str();

        strStream2 << "Angle: "<<angle<<"degree";
        txt2 = strStream2.str();

        strStream3 << "Area: "<<area;
        txt3 = strStream3.str();

        strStream4 << "Leg_bottom_y: "<< Leg_bottom.y;
        txt4 = strStream4.str();

        strStream5 << "Ratio: "<< target_ratio;
        txt5 = strStream5.str();

        cv::line(result, Bottom_center, target_center, cv::Scalar(0,255,127), 2);
        cv::rectangle(result, target, cv::Scalar(255, 0, 0), 2);
        cv::circle(result, target_center, 5, cv::Scalar(0,127,0), 2);
        cv::circle(result, Leg_bottom, 5, cv::Scalar(0,127,0), 2);
    }
    else{
        strStream1<< "Center: "<<"Not Found!";
        txt = strStream1.str();

        strStream2<< "Angle: "<<"Not Found!";
        txt2 = strStream2.str();

        strStream3<< "Area: "<<"Not Found!";
        txt3 = strStream3.str();

        strStream4<< "Leg_bottom_y: "<<"Not Found!";
        txt4 = strStream4.str();

        strStream5 << "Ratio: "<< "Not Found!";
        txt5 = strStream5.str();
    }
    cv::putText(result, txt, cv::Point(0, h-20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
    cv::putText(result, txt2, cv::Point(0, h-40), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
    cv::putText(result, txt3, cv::Point(0, h-60), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
    cv::putText(result, txt4, cv::Point(0, h-80), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);
    cv::putText(result, txt5, cv::Point(0, h-100), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,127,255),1,8);

    cv::imshow("Result", result);
//    cv::waitKey(0);
}

cv::Rect Detectbody::getTarget(cv::Mat src) {
    result = src.clone();
    h = src.rows;
    w = src.cols;
    if(method=='A') this->getLowerBody_CascadeClassifier(result);
    else if(method=='B') this->getLowerBody_singleDIP(result);
    else if(method=='C') this->getLowerBody_KCF_BPKalmanFusion(result);
    return target;
}

Detectbody::Detectbody(char Plan, char opt2)
{
    flag = false;
    area = 0;
    angle = 0;
    target_ratio = 0.0;
    selectObject = false;

    Bottom_center.x = w/2;
    Bottom_center.y = h;
    method = Plan;

    if(Plan=='A'){
        switch (opt2) {
            case 'l':
                option = "../haarcascades/haarcascade_lowerbody.xml";
                break;
            case 'f':
                option = "../haarcascades/haarcascade_frontalface_alt.xml";
                break;
            case 'e':
                option = "../haarcascades/haarcascade_eye.xml";
                break;
            default:
                std::cout<<"Error!Wrong Option"<<std::endl;
                break;
        }
    }
}
