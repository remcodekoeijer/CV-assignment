#include <cstdlib>
#include <string>
#include <iostream>



#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CalibrateCam.h"

using namespace nl_uu_science_gmt;
using namespace std;

void CalibrateCameras(string name,int numOfCam,CalibrateCam cam) 
{

	int numOfImages = 30;
	bool found2;
	string ja = to_string(numOfCam);
	VideoCapture inputVid("data/cam"+ ja +"/intrinsics.avi");
	if (!inputVid.isOpened())
	{
		cout << "Could not open the input video: " << endl;
	}
	cout << "reading frames..." << endl;
	while (inputVid.read(cam.frameVid)) {
		
		Mat temp2;
		cam.frameVid.copyTo(temp2);
		cam.savedImages.push_back(temp2);
	}
	inputVid.release();
	cout << cam.savedImages.size() << endl;

	int savedImagessize = cam.savedImages.size();
	savedImagessize = (int)(savedImagessize / numOfImages);
	cout << savedImagessize << endl;

	for (size_t i = 0; i < cam.savedImages.size(); i += savedImagessize)
	{
		found2 = findChessboardCorners(cam.savedImages[i], cam.chessboardDimensions, cam.foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		cout << "frame "<<i << " " << found2 << endl;
		if (found2) {
			cam.savedImages2.push_back(cam.savedImages[i]);
		}
	}
	cout << cam.savedImages2.size() << endl;
	cout << "creating matrix..." << endl;
	cam.cameraCalibration(cam.savedImages2, cam.chessboardDimensions, cam.calibrationSquareDimensions, cam.cameraMatrix, cam.distanceCoefficients);
	cam.saveCameraCalibrationX(name, cam.cameraMatrix, cam.distanceCoefficients);
	cout << "Matrix created with " << cam.savedImages2.size() << "  images, press " << endl;
	cout << cam.cameraMatrix << endl;
}

int main(int argc, char** argv)
{
	char type;
	do
	{
		cout << "Do you want to calibrate the cameras? [y/n]" << endl;
		cin >> type;
	} while (!cin.fail() && type != 'y' && type != 'n');

	if (type == 'y')
	{
		CalibrateCam Cam;
		CalibrateCameras("data/cam1/intrinsics.xml", 1, Cam);
		CalibrateCameras("data/cam2/intrinsics.xml", 2, Cam);
		CalibrateCameras("data/cam3/intrinsics.xml", 3, Cam);
		CalibrateCameras("data/cam4/intrinsics.xml", 4, Cam);
	}
	
	//code for creating background image

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
