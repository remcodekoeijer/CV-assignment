#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <inttypes.h>
using namespace std;
using namespace cv;

const int calibrationmode = 1; //0 is list of images, 1 is camera

const float calibrationSquareDimensions = 0.016f;
const Size chessboardDimensions = Size(6, 9);

//functions
//create known positions for chessboard, object points
void  CreateKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (size_t i = 0; i < boardSize.height; i++)
	{
		for (size_t j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
		}
	}
}

//extract from an image the corners, input all images, double array that stores info
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allfoundcorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found) {
			allfoundcorners.push_back(pointBuf);
		}
		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("lookingforcorners", *iter);
			waitKey(0);
		}
	}
}

void cameraCalibration(vector<Mat> callibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
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

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		for (size_t r = 0; r < rows; r++)
		{
			for (size_t c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		for (size_t r = 0; r < rows; r++)
		{
			for (size_t c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}

	return false;
}

//getting matrix, not sure if correct way
void getCalibrationMatrix(Mat &cameraMatrix, Mat &distanceCoefficients)
{
	ifstream reader("CamCalib.txt");
	double nvalue;
	if (reader.is_open())
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		for (size_t r = 0; r < rows; r++)
		{
			for (size_t c = 0; c < columns; c++)
			{
				reader >> nvalue;
				cameraMatrix.at<double>(r, c) = nvalue;

			}
		}
		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;
		cout << rows << columns << endl;
		for (size_t r = 0; r < rows; r++)
		{
			for (size_t c = 0; c < columns; c++)
			{
				cout << nvalue << endl;
				reader >> nvalue;
				distanceCoefficients.at<double>(r, c) = nvalue;
			}
		}

		reader.close();
	}
}

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
		int count = 0;
		Mat newImg;
		vector<Vec2f> foundPoints;
		Mat image;
		//camera capture
		Mat frame;
		Mat drawToframe;

		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

		Mat distanceCoefficients;
		Mat distanceCoefficients2 = Mat::eye(5, 1, CV_64F);

		vector<Mat> savedImages;

		vector<vector<Point2f>> markerCorners, rejectedCandidates;

		VideoCapture vid(0);

		if (!vid.isOpened()) {
			return -1;
		}

		int framePerSecond = 20;


		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
		cout << "Press space bar for capture and Enter to create matrix" << endl;
		while (true)
		{
			if (!vid.read(frame))
				break;
			//vector<Vec2f> foundPoints;
			bool found = false;

			found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

			frame.copyTo(drawToframe);

			drawChessboardCorners(drawToframe, chessboardDimensions, foundPoints, found);
			if (found)
				imshow("Webcam", drawToframe);
			else
				imshow("Webcam", frame);
			char character = waitKey(1000 / framePerSecond);

			switch (character)
			{
				//space
			case 32:
				//save image
				if (found)
				{
					cout << "Image captured, num " << count++ << endl;
					Mat temp;
					frame.copyTo(temp);
					savedImages.push_back(temp);

					//print vector numbers
					for (vector<Vec2f>::const_iterator i = foundPoints.begin(); i != foundPoints.end(); ++i)
						cout << *i << ' ';
					cout << cameraMatrix << endl;
					cout << distanceCoefficients << endl;
				}
				break;
				//enter
			case 13:
				if (savedImages.size() > 3)
				{
					cout << "Matrix created with " << count << " images" << endl;
					cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimensions, cameraMatrix, distanceCoefficients);
					saveCameraCalibration("CamCalib.txt", cameraMatrix, distanceCoefficients);
				}

				break;
				//get matrix, do that every time
			case 'g':
			{
				getCalibrationMatrix(cameraMatrix, distanceCoefficients2);
				cout << cameraMatrix << endl;
				cout << distanceCoefficients2 << endl;
			}
			break;
			//project and draw stuff
			case 'x': {
				//new caption
				imwrite("outputImg.jpg", drawToframe);
				newImg = imread("outputImg.jpg");
				imshow("windowbefore", newImg);
				//make array for drawing image
				IplImage tmp = newImg;
				//vectors and stuff
				Mat rotation_vector; // Rotation in axis-angle form
				Mat translation_vector;
				cout << cameraMatrix << endl;
				cout << distanceCoefficients << endl;
				//create wannabe points
				vector<Point3f> worldCorners;
				CreateKnownBoardPositions(chessboardDimensions, calibrationSquareDimensions, worldCorners);
				//solve pnp
				solvePnP(worldCorners, foundPoints, cameraMatrix, distanceCoefficients2, rotation_vector, translation_vector);
				//project points
				Point2d corner = foundPoints[0];
				vector<Point2d> projectedPoints, projectedPointsAxis;
				vector<Point3d> pointsToDraw, axisPointsToDraw;
				//the 3 axis
				axisPointsToDraw.push_back(Point3d(0.1, 0, 0));
				axisPointsToDraw.push_back(Point3d(0, 0.1, 0));
				axisPointsToDraw.push_back(Point3d(0, 0, 0.1));
				//the points for the cube
				pointsToDraw.push_back(Point3d(1, 0, 0));
				pointsToDraw.push_back(Point3d(0, 1, 0));
				pointsToDraw.push_back(Point3d(0, 0, 1));
				pointsToDraw.push_back(Point3d(1, 1, 0));
				pointsToDraw.push_back(Point3d(0, 1, 1));
				pointsToDraw.push_back(Point3d(1, 0, 1));
				pointsToDraw.push_back(Point3d(1, 1, 1));

				for (int i = 0; i < pointsToDraw.size(); i++)
				{
					pointsToDraw[i] *= 0.02;
				}
				
				projectPoints(axisPointsToDraw, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients2, projectedPointsAxis);
				projectPoints(pointsToDraw, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients2, projectedPoints);
				cout << foundPoints.size() << endl;

				//cube lines
				cvLine(&tmp, corner, projectedPoints[0], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, corner, projectedPoints[1], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, corner, projectedPoints[2], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[0], projectedPoints[3], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[0], projectedPoints[5], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[1], projectedPoints[3], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[1], projectedPoints[4], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[2], projectedPoints[4], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[2], projectedPoints[5], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[3], projectedPoints[6], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[4], projectedPoints[6], Scalar(255, 255, 0), 2, 8);
				cvLine(&tmp, projectedPoints[5], projectedPoints[6], Scalar(255, 255, 0), 2, 8);
				//axis lines
				cvLine(&tmp, corner, projectedPointsAxis[0], Scalar(255, 0, 0), 2, 8);
				cvLine(&tmp, corner, projectedPointsAxis[1], Scalar(0, 225, 0), 2, 8);
				cvLine(&tmp, corner, projectedPointsAxis[2], Scalar(0, 0, 255), 2, 8);

				imshow("windowafter", newImg);
				imwrite("outputImgAxis.jpg", newImg);
			}
					  break;
			case 27:
				return -1;
				break;
			}
		}

	}
		return 0;
	
}

