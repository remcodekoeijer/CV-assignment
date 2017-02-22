#include "CalibrateCam.h"


#include <iostream>
#include <fstream>
#include <inttypes.h>
#include <time.h>

using namespace std;
using namespace cv;

CalibrateCam::CalibrateCam(){}

CalibrateCam::CalibrateCam(const float calibrationSquareDimensions, Size chessboardDimensions, vector<Vec2f> foundPoints, Mat cameraMatrix, Mat distanceCoefficients, vector<Mat> savedImages, vector<Mat> savedImages2, Mat frameVid) {
	//...
}

void CalibrateCam::CreateKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (size_t i = 0; i < boardSize.height; i++)
	{
		for (size_t j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
		}
	}
}

void CalibrateCam::getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allfoundcorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(8, 6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found) {
			allfoundcorners.push_back(pointBuf);
		}
		if (showResults)
		{
			drawChessboardCorners(*iter, Size(8, 6), pointBuf, found);
			imshow("lookingforcorners", *iter);
			waitKey(0);
		}
	}
}

void CalibrateCam::cameraCalibration(vector<Mat> callibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	//get allpoints in images
	getChessboardCorners(callibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	CreateKnownBoardPositions(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	//radial vectors ,tangenial vectors
	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);

}

void CalibrateCam::saveCameraCalibrationX(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	FileStorage fs(name, FileStorage::WRITE);

	/*fs << "frameCount" << 5;
	time_t rawtime; time(&rawtime);
	fs << "calibrationDate" << asctime(localtime(&rawtime));*/
	
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distanceCoefficients;

	fs.release();

}

void CalibrateCam::getCameraCalibrationX(string name ,Mat &cameraMatrix, Mat &distanceCoefficients)
{
	FileStorage fs2(name, FileStorage::READ);

	//// first method: use (type) operator on FileNode.
	//int frameCount = (int)fs2["frameCount"];

	//string date;
	//// second method: use FileNode::operator >>
	//fs2["calibrationDate"] >> date;

	Mat cameraMatrix2, distCoeffs2;
	fs2["cameraMatrix"] >> cameraMatrix;
	fs2["distCoeffs"] >> distanceCoefficients;

	cout /*<< "frameCount: " << frameCount << endl
		<< "calibration date: " << date << endl*/
		<< "camera matrix: " << cameraMatrix << endl
		<< "distortion coeffs: " << distanceCoefficients << endl;


	fs2.release();
}



