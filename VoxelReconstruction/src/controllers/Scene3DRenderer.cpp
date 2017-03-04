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
		//===============================================================================================================
		//voxel stuff
		//get the visible voxels from frame 10 ,from 1st camera ,maybe save for all cameras?
		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (m_current_frame == 10 && m_camera_view && c==1) //not sure how to access cameras points it just takes the first whatever...?
			{

				//voxel
				vector<Reconstructor::Voxel*> visVoxels(m_reconstructor.getVisibleVoxels());
				int clusterCount = 4;
				int sizeOfVisVoxels = visVoxels.size();
				Mat positions(sizeOfVisVoxels, 2, CV_32F);
				Mat center, bestlabels;

				//get points of voxels (x,y)
				for (int r = 0; r < sizeOfVisVoxels; r++)
				{
					positions.at<float>(r, 0) = visVoxels[r]->x;
					positions.at<float>(r, 1) = visVoxels[r]->y;
				}

				kmeans(positions, clusterCount, bestlabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, center);


				//project voxel points into image
				vector<Point2d> points;
				for (int i = 0; i < positions.rows; i++)
				{
					points.push_back(Point(positions.at<float>(i, 0), positions.at<float>(i, 1)));
				}

				vector<Point2d> centerPoints;
				for (int i = 0; i < center.rows; i++)
				{
					centerPoints.push_back(Point(center.at<float>(i, 1), center.at<float>(i, 0)));
				}
				cout << centerPoints << endl;
				//since origin is in the middle on voxel space move the centers to project on image
				for (int i = 0; i < center.rows; i++)
				{
					centerPoints[i].y = (int)abs(centerPoints[i].y / 2);
					centerPoints[i].x = (int)abs(centerPoints[i].x / 2);
				}
				cout << "corrected centerpoints; " << centerPoints << endl;

				//Image show test
				Mat testImage;
				m_cameras[1]->getFrame().copyTo(testImage);
				
				//draw a pixel at the center points.. not really visible
				for (int i = 0; i < centerPoints.size(); i++)
				{
					Vec3b pixel = testImage.at<Vec3b>(centerPoints[i].y, centerPoints[i].x);

					pixel = (0, 255, 255);

					testImage.at<Vec3b>(centerPoints[i].y, centerPoints[i].x) = pixel;

				}
				imshow("test", testImage);
				//draw circles at the center points
				circle(testImage, centerPoints[0], 50, Scalar(255, 255, 255), CV_FILLED, 8, 0);
				circle(testImage, centerPoints[1], 50, Scalar(255, 255, 255), CV_FILLED, 8, 0);
				circle(testImage, centerPoints[2], 50, Scalar(255, 255, 255), CV_FILLED, 8, 0);
				circle(testImage, centerPoints[3], 50, Scalar(255, 255, 255), CV_FILLED, 8, 0);
				imshow("test2", testImage);
				

			}
		}
		//======================================================================================

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

		Mat hsv_image,hsv_image2;
		cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

		vector<Mat> channels;
		split(hsv_image, channels);  // Split the HSV-channels for further analysis
		
		vector<Mat> means; //3 Matrices for the h, v and s means
		split(hsv_image, means); //Get the 3 channels

								
		//================================================================================================================
		// Background subtraction H

		//Create the h mean mat
		for (int y = 0; y < means[0].rows; y++)
		{
			for (int x = 0; x < means[0].cols; x++)
			{
				Scalar pixel = camera->getHSVMeans(x + y * means[0].cols);
				means[0].at<uchar>(y, x) = roundf(pixel[0]);
			}
		}
		Mat tmp;

		int tr = channels[0].rows;
		int tc = channels[0].cols;
		int tt = channels[0].type();
		Mat  foreground(tr, tc, tt), background(tr, tc, tt);
		
		absdiff(channels[0], means.at(0), tmp);
		//threshold(tmp, foreground, camera->getHSVVarss(10000)[0] / 2, 255, CV_THRESH_BINARY); 

		//threshold per pixel, since each pixel has a different variance
		for (int y = 0; y < tmp.rows; y++)
		{
			for (int x = 0; x < tmp.cols; x++)
			{
				if (tmp.at<uchar>(y, x) < 1 * camera->getHSVVarss(x + y * tmp.cols)[0])
				{
					uchar pixel = 0;
					foreground.at<uchar>(y, x) = pixel;
				}
				else
				{
					uchar pixel = 255;
					foreground.at<uchar>(y, x) = pixel;
				}
			//LOOP:;
			}
		}

		//================================================================================================================
		// Background subtraction S

		//Create the s mean mat
		for (int y = 0; y <means[1].rows; y++)
		{
			for (int x = 0; x < means[1].cols; x++)
			{
				Scalar pixel = camera->getHSVMeans(x + y * means[1].cols);
				means[1].at<uchar>(y, x) = roundf(pixel[1]);
			}
		}

		absdiff(channels[1], means.at(1), tmp);

		for (int y = 0; y < tmp.rows; y++)
		{
			for (int x = 0; x < tmp.cols; x++)
			{
				if (tmp.at<uchar>(y, x) < 1 * camera->getHSVVarss(x + y * tmp.cols)[1])
				{
					uchar pixel = 0;
					background.at<uchar>(y, x) = pixel;
				}
				else
				{
					uchar pixel = 255;
					background.at<uchar>(y, x) = pixel;
				}
			}
		}
		
		bitwise_and(foreground, background, foreground);
		//================================================================================================================
		// Background subtraction V

		//Create the h mean mat
		for (int y = 0; y <means[2].rows; y++)
		{
			for (int x = 0; x < means[2].cols; x++)
			{
				Scalar pixel = camera->getHSVMeans(x + y * means[2].cols);
				means[2].at<uchar>(y, x) = roundf(pixel[2]);
			}
		}

		absdiff(channels[2], means.at(2), tmp);

		for (int y = 0; y < tmp.rows; y++)
		{
			for (int x = 0; x < tmp.cols; x++)
			{
				
				if (tmp.at<uchar>(y, x) < 2.5 * camera->getHSVVarss(x + y * tmp.cols)[2])
				{
					uchar pixel = 0;
					background.at<uchar>(y, x) = pixel;
				}
				else
				{
					uchar pixel = 255;
					background.at<uchar>(y, x) = pixel;
				}
			}
		}

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
