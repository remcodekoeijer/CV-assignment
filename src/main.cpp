#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const int calibrationmode = 1; //0 is list of images, 1 is camera

const int boardHeight = 6;
const int boardWidth = 9;
const float squareSize = 50; //millimeters
const Size boardSize = cvSize(boardWidth, boardHeight);


int main()
{
	/*
	if (calibrationmode == 0)
	{
		vector<Mat> images;
		vector<vector<Point2f>> allCorners;
		
		images.push_back(imread("data/img1.jpg"));
		images.push_back(imread("data/img2.jpg"));
		images.push_back(imread("data/img3.jpg"));
		images.push_back(imread("data/img4.jpg"));
		images.push_back(imread("data/img5.jpg"));
		images.push_back(imread("data/img6.jpg"));
		images.push_back(imread("data/img7.jpg"));
		images.push_back(imread("data/img8.jpg"));

		getChessboardCorners(images, allCorners, true);
	}
	*/

	if (calibrationmode == 1)
	{
		Mat frame;
		Mat drawToFrame;
		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
		Mat distanceCoefficients;
		vector<Mat> savedImages;
		vector<vector<Point2f>> markerCorners, rejected;

		VideoCapture vid(0); //source of the camera

		if (!vid.isOpened()) //check if it has opened
			return 0;

		int fps = 20;

		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

		while (true)
		{
			if (!vid.read(frame))
				break;

			vector<Vec2f> foundPoints;
			bool found = false;

			found = findChessboardCorners(frame, boardSize, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH);
			frame.copyTo(drawToFrame);
			drawChessboardCorners(drawToFrame, boardSize, foundPoints, found);

			if (found)
				imshow("Webcam", drawToFrame);
			else
				imshow("Webcam", frame);

			char character = waitKey(1000 / fps);
		}
	}


	//waitKey(0);
	return 0;
}

/*
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin; iter != images.end; iter++)
	{
		vector<Point2f> pointBuffer;
		bool found = findChessboardCorners(*iter, boardSize, pointBuffer, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuffer);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, boardSize, pointBuffer, found);
			imshow("Looking for corners", *iter);
			waitKey(0);
		}
	}
}
*/