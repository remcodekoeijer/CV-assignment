/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>
#include <iostream>
#include <sstream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 20;
	const int S = 50;
	const int V = 25;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);

	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}
	return true;
}

/**
 * Separate the background from the foreground for the current frame
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
		Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	//================================================================================================================
	// Background subtraction H
	vector<Mat> means;
	Mat hMean(channels[0].rows, channels[0].cols, channels[0].type());
	split(hsv_image, means);
	
	//The problem is creating the mat with the means. for some reason it crashes in debug mode, since in release it doesnt consider out of range errors.
	//below are a few tests that give an error, since not all .at are crashing, but the first one does work. it's strange
	uchar test = camera->getHSVMeans(300 + 300 * hMean.cols)[0];
	uchar test2 = camera->getHSVMeans(400 + 100 * hMean.cols)[0];
	uchar test3 = camera->getHSVMeans(643 + 485 * hMean.cols)[0];
	hMean.at<Vec3b>(10, 10) = test;
	hMean.at<Vec3b>(485, 643) = test3;
	hMean.at<Vec3b>(300, 300) = test;
	hMean.at<Vec3b>(100, 400) = test2;

	for (int y = 0; y < hMean.rows; y++)
	{
		for (int x = 0; x < hMean.cols; x++)
		{
			hMean.at<Vec3b>(y, x) = camera->getHSVMeans(x + y * hMean.cols)[0]; //Mat with mean h-values
		}
	}
	
	cout << (float)hMean.at<Vec3b>(300, 300)[0] << '\n';
	cout << (float)hMean.at<Vec3b>(100, 400)[0] << '\n';

	cout << (float)hMean.at<Vec3b>(485, 643)[0] << '\n';
	Mat tmp, foreground, background;
	
	absdiff(channels[0], hMean, tmp); //get the difference


	threshold(tmp, foreground, camera->getHSVVarss(10000)[0], 255, CV_THRESH_BINARY); //now foreground is not empty, bit ugly...

	//threshold per pixel, since each pixel has a different variance
	/*
	for (int y = 0; y < tmp.rows; y++)
	{
		for (int x = 0; x < tmp.cols; x++)
		{
			if (tmp.at<Vec3b>(y, x)[0] < camera->getHSVVarss(x + y * tmp.cols)[0])
				foreground.at<Vec3b>(y, x)[0] = 255;
			else
				foreground.at<Vec3b>(y, x)[0] = 0;
		}
	}
	*/
	//================================================================================================================
	// Background subtraction S
	for (int y = 0; y < means[1].rows; y++)
	{
		for (int x = 0; x < means[1].cols; x++)
		{
			means[1].at<Vec3b>(y, x)[1] = camera->getHSVMeans(x + y * means[1].cols)[1]; 
		}
	}

	absdiff(channels[1], means[1], tmp);
	threshold(tmp, background, camera->getHSVVarss(10000)[1], 255, CV_THRESH_BINARY); //now background is not empty, bit ugly...
	
	//threshold per pixel, since each pixel has a different variance
	/*
	for (int y = 0; y < tmp.rows; y++)
	{
		for (int x = 0; x < tmp.cols; x++)
		{
			if (tmp.at<Vec3b>(y, x)[1] < camera->getHSVVarss(x + y * tmp.cols)[1])
				background.at<Vec3b>(y, x)[1] = 255;
			else
				background.at<Vec3b>(y, x)[1] = 0;
		}
	}
	*/
	bitwise_and(foreground, background, foreground);

	//================================================================================================================
	// Background subtraction V
	for (int y = 0; y < means[2].rows; y++)
	{
		for (int x = 0; x < means[2].cols; x++)
		{
			means[2].at<Vec3b>(y, x)[0] = camera->getHSVMeans(x + y * means[2].cols)[2]; 
		}
	}

	absdiff(channels[2], means[2], tmp);
	threshold(tmp, background, camera->getHSVVarss(10000)[2], 255, CV_THRESH_BINARY);
	//threshold per pixel
	/*
	for (int y = 0; y < tmp.rows; y++)
	{
		for (int x = 0; x < tmp.cols; x++)
		{
			if (tmp.at<Vec3b>(y, x)[2] < camera->getHSVVarss(x + y * tmp.cols)[2])
				background.at<Vec3b>(y, x)[2] = 255;
			else
				background.at<Vec3b>(y, x)[2] = 0;
		}
	}
	*/
	bitwise_or(foreground, background, foreground);

	//=============================================================
	// Improve the foreground image
	Mat element = getStructuringElement(0, Size(3, 3), Point(-1, -1));
	erode(foreground, foreground, element);
	dilate(foreground, foreground, element);

	camera->setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
