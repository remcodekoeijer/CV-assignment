#ifndef CALIBRATECAM_H_
#define CALIBRATECAM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;





class CalibrateCam
{
	
public:
	CalibrateCam();
	CalibrateCam(const float, Size, vector<Vec2f>, Mat, Mat, vector<Mat>, vector<Mat>, Mat);
	void CreateKnownBoardPositions(Size, float, vector<Point3f>&);
	void getChessboardCorners(vector<Mat>, vector<vector<Point2f>>&, bool);
	void cameraCalibration(vector<Mat>, Size, float, Mat&, Mat&);
	void saveCameraCalibrationX(string, Mat, Mat);
	void getCameraCalibrationX(string,Mat&, Mat&);

	//variables
	const float calibrationSquareDimensions = 0.115f; //change this according to your image dimensions
	const Size chessboardDimensions = Size(6, 8); //corners size
	vector<Vec2f> foundPoints;
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;
	vector<Mat> savedImages;
	vector<Mat> savedImages2;
	Mat frameVid;
};

#endif /* CALIBRATE_H_ */#pragma once
