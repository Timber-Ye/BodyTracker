//
// Created by 翰樵 on 2020/12/13.
//

#include "../Inc/KCF_BP_Kalman_Tracker.h"
float hranges[] = { 0, 180 };
const float* phranges = hranges;

KCF_BP_Kalman_Tracker::KCF_BP_Kalman_Tracker(double t)
{
    trackObject = 0;
    showHist = true;
    selectObject = false;
    point_num = 0;
    backprojMode = false;
    dSimilarity = 0.0;
    stateNum = 6;
    measureNum = 2;
    cv::KalmanFilter m(stateNum, measureNum, 0);
    KF = m;
    measurement = cv::Mat::zeros( measureNum, 1, CV_32F );
    v_x = 1.0;
    v_y = 1.0;
    a_x = 0.1;
    a_y = 0.1;
    T = 0.9;
    Threshold = t;

    std::printf("Constructed!");
    trainingData_In = cv::Mat( 3, 4, CV_32FC1 );
    trainingData_Out = cv::Mat( 3, 2, CV_32FC1 );

    // 构建神经网络
    layerSizes = ( cv::Mat_<int>( 1, 3 ) << 4, 9, 2 );
    ann = cv::ml::ANN_MLP::create( );
    ann->setLayerSizes( layerSizes );
    ann->setActivationFunction( cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1 );
    ann->setTermCriteria( cv::TermCriteria( cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, FLT_EPSILON ) );
    ann->setTrainMethod( cv::ml::ANN_MLP::BACKPROP, 0.1, 0.1 );

    // 卡尔曼滤波器初始化
    KF.transitionMatrix = ( cv::Mat_<float>( 6, 6 ) <<
            1, 0, T, 0, ( T*T ) / 2, 0,
            0, 1, 0, T, 0, ( T*T ) / 2,
            0, 0, 1, 0, T, 0,
            0, 0, 0, 1, 0, T,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1 ); // 利用牛顿运动定理建立状态转移矩阵 六个状态：二维+(位置， 速度， 加速度)

    cv::setIdentity( KF.measurementMatrix );						//测量矩阵H
    cv::setIdentity( KF.processNoiseCov, cv::Scalar::all( 1e-5 ) );		//过程噪声方差矩阵Q
    cv::setIdentity( KF.measurementNoiseCov, cv::Scalar::all( 1e-1 ) );	//测量噪声方差矩阵R
    cv::setIdentity( KF.errorCovPost, cv::Scalar::all( 1 ) );			//(状态预估值的)误差协方差：P(k)

    // 状态初始值
    KF.statePost = ( cv::Mat_<float>( 6, 1 ) << 0, 0, v_x, v_y, a_x, a_y ); //在初始条件不确切时依旧能够收敛
}

void KCF_BP_Kalman_Tracker::Select_target(cv::Mat src){
    Threshold = 0.6;
    while(!selectObject){
        cv::imshow("Original_frame", src);
        Selected_Rect = cv::selectROI("Original_frame",src, false, false);
        if(Selected_Rect.empty()){
            std::cout<<"Wrong selection!"<<std::endl;
        }
        else{
            selectObject = true;
            trackObject = -1;
        }
    }

}

cv::Mat KCF_BP_Kalman_Tracker::drawHist(cv::Mat& src, const cv::Rect& selected_Rect, const std::string& WINDOW_NAME){
    cv::Mat hist;
    int hsize = 16;
    cv::Mat histImg = cv::Mat::zeros(200, 320, CV_8UC3);
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);

    mask(selected_Rect).setTo(255);

    cv::calcHist(&src, 1, 0, mask, hist, 1, &hsize, &phranges);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    histImg = cv::Scalar::all(0);
    int binW = histImg.cols / hsize;
    cv::Mat buf(1, hsize, CV_8UC3);
    for (int i = 0; i < hsize; i++)
        buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize), 255, 255);
    cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);

    for (int i = 0; i < hsize; i++){
        int val = cv::saturate_cast<int>(hist.at<float>(i)*histImg.rows / 255);
        cv::rectangle(histImg, cv::Point(i*binW, histImg.rows), cv::Point((i + 1)*binW, histImg.rows - val), cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
    }
    if(showHist) cv::imshow(WINDOW_NAME, histImg);

    return hist;
}

inline bool KCF_BP_Kalman_Tracker::KCF_WORKS(double d, double t){
    return d < t;
}

void KCF_BP_Kalman_Tracker::diff_Mat(const cv::Mat& in_Image, const cv::Mat& out_Image){
    int count = 0;
    std::vector<cv::Point2f> in_diff;
    std::vector<cv::Point2f> out_diff;

    // vec中存储的点坐标（六对）
    //for ( vector<Point2f>::iterator it = vec.begin( ); it != vec.end( ); it++ ) {
    //	cout << *it << endl;
    //}

    // 构建输入样本差值矩阵(三行四列，共四对差值，由五对点坐标产生，其中每行两对差值)
    for ( auto it = vec.begin( ); it != vec.end( ) - 2; it++ ) {
        float dete_x = ( *( it + 1 ) ).x - ( *it ).x;
        float dete_y = ( *( it + 1 ) ).y - ( *it ).y;
        in_diff.emplace_back( dete_x, dete_y );
        if ( count > 0 && count < 4 )
            in_diff.emplace_back( dete_x, dete_y );
        count++;
    }
    // vector转换到Mat---浅拷贝
    //Mat test_Image = Mat( 3, 4, CV_32FC1, (float*)in_diff.data( ) );
    // vector转换到Mat---深拷贝
    std::memcpy( in_Image.data, in_diff.data( ), 2 * in_diff.size( ) * sizeof( float ) );
    //cout << endl << "输入差值矩阵：" << endl << in_Image << endl << endl;

    //构建输出样本差值矩阵(三行两列，共三对差值，由四对点坐标产生，其中每行一对差值)
    for ( auto it = vec.begin( ) + 2; it != vec.end( ) - 1; it++ ) {
        float dete_x = ( *( it + 1 ) ).x - ( *it ).x;
        float dete_y = ( *( it + 1 ) ).y - ( *it ).y;
        out_diff.emplace_back( dete_x, dete_y );
    }
    //out_Image = Mat( 3, 2, CV_32FC1, (float*)out_diff.data( ) );
    std::memcpy( out_Image.data, out_diff.data( ), 2 * out_diff.size( ) * sizeof( float ) );
    //cout << endl << "输出差值矩阵：" << endl << out_Image << endl << endl;
}

cv::Point2f KCF_BP_Kalman_Tracker::neural_networks( cv::Mat in_trainData, cv::Mat out_trainData ){
// 创建训练数据，由构建的差值矩阵按行训练，也就是每次输入两对差值，输出一对差值，即
    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create( in_trainData, cv::ml::ROW_SAMPLE, out_trainData );
    ann->train( tData );

    // 构建当前预测差值矩阵
    std::vector<cv::Point2f> pre_diff;
    for ( auto it = vec.begin( ) + 3; it != vec.end( ) - 1; it++ ) {
        //cout << *it << endl;
        float dete_x = ( *( it + 1 ) ).x - ( *it ).x;
        float dete_y = ( *( it + 1 ) ).y - ( *it ).y;
        pre_diff.emplace_back( dete_x, dete_y );
    }
    cv::Mat sampleMat = cv::Mat( 1, 4, CV_32FC1, (float*)pre_diff.data( ) );
    cv::Mat predictMat = cv::Mat( 1, 2, CV_32FC1 );
    ann->predict( sampleMat, predictMat );
    //cout << endl << "神经网络预测差值矩阵：" << endl << sampleMat << endl;
    //cout << endl << "神经网络预测结果矩阵：" << endl << predictMat << endl;
    cv::Point2f pre_point4;
    cv::Point2f pre_point1 = vec.at( 3 );
    cv::Point2f pre_point2 = vec.at( 4 );
    cv::Point2f pre_point3 = vec.at( 5 );
    // 神经网络预测的下一时刻的点坐标，保留到小数点后一位
    pre_point4.x = vec.back( ).x + floor( predictMat.at<float>( 0 ) * 10 + 0.5 ) / 10;
    pre_point4.y = vec.back( ).y + floor( predictMat.at<float>( 0 ) * 10 + 0.5 ) / 10;
    //cout << endl << "pre_point1: " << pre_point1 << endl;
    //cout << "pre_point2: " << pre_point2 << endl;
    //cout << "pre_point3: " << pre_point3 << endl;
    //circle( image, pre_point4, 3, Scalar( 255, 0, 0 ), 3 );
    std::cout << "pre_point = " << pre_point4 << std::endl;

    return pre_point4;
}

cv::Point2f KCF_BP_Kalman_Tracker::kalman_filter( const cv::Point2f& measure_point ){
    // 更新上一时刻状态
    KF.statePost.at<float>( 0 ) = vec.back().x;
    KF.statePost.at<float>( 1 ) = vec.back().y;

    // 状态方程计算，由系统模型及上一时刻状态预测当前时刻位置。
    cv::Point2f klm_point; //预估位置坐标点
    cv::Mat prediction = KF.predict();
    klm_point.x = floor( prediction.at<float>( 0 ) * 10 + 0.5 ) / 10;
    klm_point.y = floor( prediction.at<float>( 1 ) * 10 + 0.5 ) / 10;

    //更新测量值。(采用神经网络的预测值来充当测量值)
    measurement.at<float>( 0 ) = (float)measure_point.x;
    measurement.at<float>( 1 ) = (float)measure_point.y;

    //更新---最优估计值
    cv::Point2f cor_point;//位置的最优估计坐标点
    KF.correct(measurement);
    cor_point.x = floor( KF.statePost.at<float>( 0 ) * 10 + 0.5 ) / 10;
    cor_point.y = floor( KF.statePost.at<float>( 1 ) * 10 + 0.5 ) / 10;

    //输出结果
    cv::circle( image, cor_point, 3, cv::Scalar( 255, 255, 255 ), 3 );
    std::cout << "cor_point = " << cor_point << std::endl << std::endl;

    return cor_point;
}

/*cv::Rect KCF_BP_Kalman_Tracker::KCF_Track(cv::Mat src){

}*/

cv::Rect KCF_BP_Kalman_Tracker::camshift_Track(cv::Mat src){
    cv::Mat hsv, hue, mask;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    int vmin = 10, vmax = 256, smin = 30;
    image = src.clone();

    if (trackObject){

        int _vmin = vmin, _vmax = vmax;

        cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin, _vmax)), cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);

        int ch[] = { 0, 0 };
        hue.create(hsv.size(), hsv.depth());
        cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

        // 构建目标模型的颜色直方图
        if (trackObject < 0){
//            hist1 = drawHist(hue, Selected_Rect, "目标模型的颜色直方图");
            trackObject = 1;
            trackWindow = Selected_Rect;
        }
        hist2 = drawHist(hue, trackWindow, "搜索窗口的颜色直方图");// 构建搜索窗的颜色直方图

        dSimilarity = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
        //---------------------------------------------------------------------
        //当巴氏系数小于阈值S时，未发生遮挡
        if (dSimilarity < Threshold) {
            cv::calcBackProject( &hue, 1, 0, hist1, backproj, &phranges );
            backproj &= mask; //滤除光线等情况的干扰。
            trackBox = cv::CamShift(backproj, trackWindow,
                    cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                            10, 1 ) );
            if ( trackWindow.area() <= 100 ) {
                int cols = backproj.cols, rows = backproj.rows, r = ( MIN( cols, rows ) + 5 ) / 6;
                trackWindow = cv::Rect( trackWindow.x - r, trackWindow.y - r,
                        trackWindow.x + r, trackWindow.y + r ) & cv::Rect( 0, 0, cols, rows );
            }
        }
            //---------------------------------------------------------------------
            //当巴氏系数于阈值S时，目标发生遮挡，以预测位置代替真实位置进行跟踪
        else {
            int x = pre_point.x;
            int y = pre_point.y;
            int w = Selected_Rect.width;
            int h = Selected_Rect.height;

            pre_trackWindow = cv::Rect( x - w / 2, y - h / 2, w, h );
            cv::calcBackProject( &hue, 1, 0, hist2, backproj, &phranges );
            backproj &= mask;

            trackBox = cv::CamShift( backproj, pre_trackWindow,
                    cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 1 ));
            // 更新搜索窗
            trackWindow = pre_trackWindow;
        }
        //神经网络+Kalman预测
        point_num++;
        vec.push_back(trackBox.center);
        if ( point_num >= 6 ) {
            diff_Mat( trainingData_In, trainingData_Out );
            BP_point = neural_networks( trainingData_In, trainingData_Out );
            pre_point = kalman_filter( BP_point);
            vec.erase( vec.begin( ) );  // 删除容器中的第一个点
        }
        cv::ellipse( image, trackBox, cv::Scalar( 0, 0, 255 ), 3);
        cv::circle( image, trackBox.center, 3, cv::Scalar( 0, 0, 255 ), 3 );
    }

    if (selectObject && Selected_Rect.width > 0 && Selected_Rect.height > 0){
        cv::Mat roi(image, Selected_Rect);
        cv::bitwise_not(roi, roi);
    }
    /*cv::imshow("CamShift Demo", image);*/
    return trackBox.boundingRect();
}

cv::Rect KCF_BP_Kalman_Tracker::KCF_Track(cv::Mat src){
    cv::Mat hsv, hue, mask;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    int vmin = 10, vmax = 256, smin = 30;
    image = src.clone();


    cv::namedWindow("CamShift Demo", 0);
    cv::createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
    cv::createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
    cv::createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

    cv::imshow("CamShift Demo", src);

    if (trackObject){

        int _vmin = vmin, _vmax = vmax;

        cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin, _vmax)), cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);

        int ch[] = { 0, 0 };
        hue.create(hsv.size(), hsv.depth());
        cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

        // 构建目标模型的颜色直方图
        if (trackObject < 0){
            hist1 = drawHist(hue, Selected_Rect, "目标模型的颜色直方图");
            trackObject = 1;
            trackWindow = Selected_Rect;
        }

        hist2 = drawHist(hue, trackWindow, "搜索窗口的颜色直方图");// 构建搜索窗的颜色直方图

        dSimilarity = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
        //---------------------------------------------------------------------
        //当巴氏系数小于阈值S时，未发生遮挡
        //---------------------------------------------------------------------
        if (dSimilarity < Threshold) {
            cv::calcBackProject( &hue, 1, 0, hist1, backproj, &phranges );
            backproj &= mask;
            trackBox = cv::CamShift(backproj, trackWindow, cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 1 ) );
            if ( trackWindow.area() <= 1 ) {
                int cols = backproj.cols, rows = backproj.rows, r = ( MIN( cols, rows ) + 5 ) / 6;
                trackWindow = cv::Rect( trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r ) & cv::Rect( 0, 0, cols, rows );
            }
            if ( backprojMode )
                cv::cvtColor(backproj, image, cv::COLOR_GRAY2BGR );

            // 跟踪的时候以椭圆为代表目标
            /*ellipse( image, trackBox, cv::Scalar( 0, 0, 255 ), 3);
            circle( image, trackBox.center, 3, cv::Scalar( 0, 0, 255 ), 3 );*/

            // 跟踪的时候以矩形框为代表
            cv::Point2f vertex[4];
            trackBox.points(vertex);
            for (int i = 0; i < 4; i++)
                line(image, vertex[i], vertex[(i + 1) % 4], cv::Scalar(0, 0, 255), 2, 8, 0);

        }
            //---------------------------------------------------------------------
            //当巴氏系数于阈值S时，目标发生遮挡，以预测位置代替真实位置进行跟踪
            //---------------------------------------------------------------------
        else {

            int x = pre_point.x;
            int y = pre_point.y;
            int w = Selected_Rect.width;
            int h = Selected_Rect.height;

            pre_trackWindow = cv::Rect( x - w / 2, y - h / 2, w, h );
            cv::calcBackProject( &hue, 1, 0, hist2, backproj, &phranges );
            backproj &= mask;

            trackBox = cv::CamShift( backproj, pre_trackWindow, cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 1 ) );
//            cv::ellipse( image, trackBox, cv::Scalar( 0, 255, 255 ), 3);

            // 更新搜索窗
            trackWindow = pre_trackWindow;
        }

        //神经网络+Kalman预测
        point_num++;
        vec.push_back(trackBox.center);
        if ( point_num >= 6 ) {
            diff_Mat( trainingData_In, trainingData_Out );
            BP_point = neural_networks( trainingData_In, trainingData_Out );
            pre_point = kalman_filter( BP_point);

            vec.erase( vec.begin( ) );  // 删除容器中的第一个点
        }
    }

    if (selectObject && Selected_Rect.width > 0 && Selected_Rect.height > 0){
        cv::Mat roi(image, Selected_Rect);
        cv::bitwise_not(roi, roi);
    }
    /*cv::imshow("CamShift Demo", image);*/
    return trackBox.boundingRect();
}


