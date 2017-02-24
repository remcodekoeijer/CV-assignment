#include <cstdlib>
#include <string>
#include <iostream>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CalibrateCam.h"

using namespace nl_uu_science_gmt;
using namespace std;


const int numOfIm = 20;


bool calibrateCameras2(string name, int numOfCam, CalibrateCam cam, int &numOfImages) {

	
	bool found2;
	string showValue;
	string ja = to_string(numOfCam);
	VideoCapture inputVid("data/cam" + ja + "/intrinsics.avi");

	int numOfFrames = inputVid.get(CV_CAP_PROP_FRAME_COUNT);
	int framesToDo = (int)(numOfFrames / numOfImages);

	for (size_t i = 0; i < numOfFrames; i += framesToDo) {
		inputVid.read(cam.frameVid);
		Mat temp2;
		cam.frameVid.copyTo(temp2);
		found2 = findChessboardCorners(temp2, cam.chessboardDimensions, cam.foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		showValue = found2 ? "used" : "not used";
		cout << "frame " << i << " " << showValue << endl;
		if (found2) {
			cam.savedImages2.push_back(temp2);
		}
		inputVid.set(CAP_PROP_POS_FRAMES, i);
	}
	if (cam.savedImages2.size() < 15) {
		numOfImages *= 2;
		cout << "run again" << endl;
		return false;
	}
		
	cout << cam.savedImages2.size() << endl;
	cout << "creating matrix..." << endl;
	cam.cameraCalibration(cam.savedImages2, cam.chessboardDimensions, cam.calibrationSquareDimensions, cam.cameraMatrix, cam.distanceCoefficients);
	cam.saveCameraCalibrationX(name, cam.cameraMatrix, cam.distanceCoefficients);
	cout << "Matrix created with " << cam.savedImages2.size() << "  images, press " << endl;
	cout << cam.cameraMatrix << endl;
	numOfImages = numOfIm;
	return true;

}

int main(int argc, char** argv)
{

	char type;
	bool enough = false;
	int numOfImages= numOfIm;
	do
	{
		cout << "Do you want to calibrate the cameras? [y/n]" << endl;
		cin >> type;
	} while (!cin.fail() && type != 'y' && type != 'n');

	if (type == 'y')
	{
		CalibrateCam Cam1,Cam2,Cam3,Cam4;
		do {
 			enough = calibrateCameras2("data/cam1/intrinsics.xml", 1, Cam1, numOfImages);
		} while (!enough);
		do {
			enough = calibrateCameras2("data/cam2/intrinsics.xml", 2, Cam2, numOfImages);
		} while (!enough);
		do {
			enough = calibrateCameras2("data/cam3/intrinsics.xml", 3, Cam3, numOfImages);
		} while (!enough);
		do {
			enough = calibrateCameras2("data/cam4/intrinsics.xml", 4, Cam4, numOfImages);
		} while (!enough);

	}
	
	//code for creating background image

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
