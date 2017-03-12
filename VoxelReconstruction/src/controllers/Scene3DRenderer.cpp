/*
* Scene3DRenderer.cpp
*
*  Created on: Nov 15, 2013
*      Author: coert
*/

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>
#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <algorithm>


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
		histogramsCreated = false;
		halfWidth = m_reconstructor.getSize();
		step = m_reconstructor.getStep();
		Mat trackerImg(halfWidth * 2 / step * 5, halfWidth * 2 / step * 5, CV_8UC3, Scalar(0,0,0));
		setTrackerImage(trackerImg);
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
		
		//create offline histograms
		if (m_current_frame > 1 && !histogramsCreated)
		{
			getHistoOff(offH1, offH2, offH3, offH4);
			histogramsCreated = true;
			offHistos.push_back(offH1);
			offHistos.push_back(offH2);
			offHistos.push_back(offH3);
			offHistos.push_back(offH4);
			//cout << "offline created" << endl;
			numOfView = 1;
		}
		
		if (m_current_frame > 2 && histogramsCreated)
		{
			Mat complVoxR1;
			vector<Point3d> centerPoints;
			
			getHisto(h1, h2, h3, h4, numOfView, complVoxR1, centerPoints); //create online maybe merge functions practically the same
			//cout << "processframe " << numOfView << endl;
			//cout << "centerpoints: " << centerPoints << endl;
			onHistos.push_back(h1);
			onHistos.push_back(h2);
			onHistos.push_back(h3);
			onHistos.push_back(h4);

			//compare
			double chi = 0; //chi distance
			vector<double> chis; //chi array
			for (int f = 0; f<4; f++) //loop tghrough off histos
			{
				for (int o = 0; o<4; o++) //loop tghrough on histos
				{
					for (int r = 0; r < 16; r++)
					{
						for (int g = 0; g < 16; g++)
						{
							for (int b = 0; b < 16; b++)
							{
						    //chi square thing,check each online vs each offline
						
								if ((onHistos[o][r][g][b] + offHistos[f][r][g][b]) != 0) //division by zero check
									chi += (onHistos[o][r][g][b] - offHistos[f][r][g][b])*(onHistos[o][r][g][b] - offHistos[f][r][g][b]) / (onHistos[o][r][g][b] + offHistos[f][r][g][b]);
							}
						}
					}
					chis.push_back(chi/2);
					chi = 0;//next hist compare
				}
			}
			//now you have 16 piece chi distances in the first 4 you have the smallest of onhistos vs off1 so you know which one is off1
			double daMin1 = min(chis[0], min(chis[1], min(chis[2], chis[3])));
			double daMin2 = min(chis[4], min(chis[5], min(chis[6], chis[7])));
			double daMin3 = min(chis[8], min(chis[9], min(chis[10], chis[11])));
			double daMin4 = min(chis[12], min(chis[13], min(chis[14], chis[15])));
			//cout << daMin1 << endl; //this is online histogram as the coresponding offline 1 ,paint it the same
			//depending on the ordering of labels
			vector<Reconstructor::Voxel*> visVoxels(m_reconstructor.getVisibleVoxels());
			int count = 0;// this is the label coresponding  number
			for (int i = 0; i < 4; i++)//for 1st online
			{
				if (daMin1 == chis[i])
				{
						for (int j = 0; j < complVoxR1.rows; j++)
						{
							int label = complVoxR1.at<float>(j, 3);//get label of voxel

							if (label == count) {
								visVoxels[j]->color = Scalar(1.0f, 0.0f, 0.0f, 1.0f);//paint it red
								trackerImage.at<Vec3b>((halfWidth + centerPoints[label].x) / step * 5, (halfWidth + centerPoints[label].y) / step * 5) = Vec3b(0, 0, 255);
							}
								
						}	
				}
				count += 1;//increase label
			}
			count = 0;
			for (int i = 4; i < 8; i++)//for second online
			{
				
				if (daMin2 == chis[i])
				{
					for (int j = 0; j < complVoxR1.rows; j++)
					{
						int label = complVoxR1.at<float>(j, 3);

						if (label == count)
						{
							visVoxels[j]->color = Scalar(0.0f, 1.0f, 0.0f, 1.0f);
							trackerImage.at<Vec3b>((halfWidth + centerPoints[label].x) / step * 5, (halfWidth + centerPoints[label].y) / step * 5) = Vec3b(0, 255, 0);
						}
							

					}
				}
				count += 1;
			}
			count = 0;
			for (int i = 8; i < 12; i++)//for third online
			{
				
				if (daMin3 == chis[i])
				{
					for (int j = 0; j < complVoxR1.rows; j++)
					{
						int label = complVoxR1.at<float>(j, 3);

						if (label == count)
						{
							visVoxels[j]->color = Scalar(0.0f, 0.0f, 1.0f, 1.0f);
							trackerImage.at<Vec3b>((halfWidth + centerPoints[label].x) / step * 5, (halfWidth + centerPoints[label].y) / step * 5) = Vec3b(255, 0, 0);
							
						}
							

					}
				}
				count += 1;
			}
			count = 0;
			for (int i = 12; i < 16; i++)//for forth online
			{
				if (daMin4 == chis[i])
				{
					for (int j = 0; j < complVoxR1.rows; j++)
					{
						int label = complVoxR1.at<float>(j, 3);

						if (label == count)
						{
							visVoxels[j]->color = Scalar(1.0f, 0.0f, 1.0f, 1.0f);
							trackerImage.at<Vec3b>((halfWidth + centerPoints[label].x) / step * 5, (halfWidth + centerPoints[label].y) / step * 5) = Vec3b(255, 0, 255);
						}
							
					}
					
				}
				count += 1;
			}

			//drawing on the tracker image
			//trackerImage.at<float>((halfWidth + centerPoints[0].x) / step * 5, (halfWidth + centerPoints[0].y) / step * 5) = 255;
		//	trackerImage.at<float>((halfWidth + centerPoints[1].x) / step * 5, (halfWidth + centerPoints[1].y) / step * 5) = 255;
		//	trackerImage.at<float>((halfWidth + centerPoints[2].x) / step * 5, (halfWidth + centerPoints[2].y) / step * 5) = 255;
		//	trackerImage.at<float>((halfWidth + centerPoints[3].x) / step * 5, (halfWidth + centerPoints[3].y) / step * 5) = 255;
			imshow("trackerimage", trackerImage);
			
			//cout << "current frame " << m_current_frame << " chi square 1; "  <<chi/2<< endl;
			onHistos.clear();
			visVoxels.clear();
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

		Mat hsv_image, hsv_image2;
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

	void Scene3DRenderer::getHisto(vector<vector<vector<int>>> &h1, vector<vector<vector<int>>> &h2, vector<vector<vector<int>>> &h3, vector<vector<vector<int>>> &h4, int &numOfView, Mat &complVoxR, vector<Point3d> &centers)
	{
		//voxel
		vector<Reconstructor::Voxel*> visVoxels(m_reconstructor.getVisibleVoxels());
		
		int clusterCount = 4;
		int sizeOfVisVoxels = visVoxels.size();
		Mat positions(sizeOfVisVoxels, 2, CV_32F);
		Mat center, bestlabels;

		//get points of voxels (x,y)
		//TODO: ignore height of the voxels, so we dont need all visvoxels. only add the ones from the upper body. 
		for (int r = 0; r < sizeOfVisVoxels; r++)
		{
			positions.at<float>(r, 0) = visVoxels[r]->x;
			positions.at<float>(r, 1) = visVoxels[r]->y;
		}
		//cluster. using KMEANS_PP_CENTERS gives uniformly distributed initial centers. this makes the chance of a local minimum very slim. 
		kmeans(positions, clusterCount, bestlabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, center);

		//project cluster center into image,
		vector<Point3d> centerPoints;
		for (int i = 0; i < center.rows; i++)
		{
			centerPoints.push_back(Point3d(center.at<float>(i, 0), center.at<float>(i, 1), 0));
		}
		vector<Point2d> projectedCenters;

		projectPoints(centerPoints, m_cameras[numOfView]->getRvec(), m_cameras[numOfView]->getTvec(), m_cameras[numOfView]->getCamMatrix(), m_cameras[numOfView]->getDistCoeff(), projectedCenters);

		Mat complVox(sizeOfVisVoxels, 4, CV_32F);
		for (int i = 0; i < bestlabels.rows; i++)
		{
			complVox.at<float>(i, 0) = visVoxels[i]->x;
			complVox.at<float>(i, 1) = visVoxels[i]->y;
			complVox.at<float>(i, 2) = visVoxels[i]->z;

			int label = bestlabels.at<int>(i, 0);
			complVox.at<float>(i, 3) = label;

		}
		vector<Point3d> pointsWithHeight;
		for (int i = 0; i < complVox.rows; i++)
		{
			pointsWithHeight.push_back(Point3d(complVox.at<float>(i, 0), complVox.at<float>(i, 1), complVox.at<float>(i, 2)));
		}
		//projected points with height
		vector<Point2d> projectedPoints;
		projectPoints(pointsWithHeight, m_cameras[numOfView]->getRvec(), m_cameras[numOfView]->getTvec(), m_cameras[numOfView]->getCamMatrix(), m_cameras[numOfView]->getDistCoeff(), projectedPoints);

		//check to see if there is occlusion
		//==================================================

		//check if the projected points are nicely divided in 4 clusters. if not, there is occlusion and you switch camera view
		Mat projectedPointsMat(projectedPoints.size(), 1, CV_32F);
		Mat projectedPointLabels;
		Scalar labelCounts(0, 0, 0, 0);

		for (int r = 0; r < projectedPoints.size(); r++)
		{
			projectedPointsMat.at<float>(r, 0) = projectedPoints[r].x;
		}

		kmeans(projectedPointsMat, clusterCount, projectedPointLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1), 3, KMEANS_PP_CENTERS, noArray());

		for (int i = 0; i < projectedPointLabels.rows; i++)
		{
			int label = projectedPointLabels.at<int>(i, 0);

			//get the size of the clusters by counting the labels
			switch (label)
			{
			case 0: labelCounts[0]++;
				break;

			case 1: labelCounts[1]++;
				break;

			case 2: labelCounts[2]++;
				break;

			case 3: labelCounts[3]++;
				break;
			}
		}

		//cout << "labelcounts scalar: " << labelCounts << endl;
		int maxLabelCount = max(labelCounts[0], max(labelCounts[1], max(labelCounts[2], labelCounts[3])));
		int diff = maxLabelCount * 0.5; //max difference between clustersizes is 20%
		//cout << "diff: " << diff << endl; 

		if (std::abs(labelCounts[0] - labelCounts[1]) > diff || std::abs(labelCounts[0] - labelCounts[2]) > diff || std::abs(labelCounts[0] - labelCounts[3]) > diff)
		{
			//clustersizes differ to much, so there is occlusion in the current cameraview. Continue with the next view
			numOfView = (numOfView + 1) % 4;
			getHisto(h1, h2, h3, h4, numOfView, complVoxR, centers);
			return;
		}
		//=========================================================
		

		//create matrix with all projected points , their labels and RGB colors,
		Mat allProjectedPointsRGB(projectedPoints.size(), 6, CV_32F);
		for (int i = 0; i < projectedPoints.size(); i++)
		{
			allProjectedPointsRGB.at<float>(i, 0) = projectedPoints[i].x;
			allProjectedPointsRGB.at<float>(i, 1) = projectedPoints[i].y;
			allProjectedPointsRGB.at<float>(i, 2) = bestlabels.at<int>(i, 0);
			allProjectedPointsRGB.at<float>(i, 3) = 0;
			allProjectedPointsRGB.at<float>(i, 4) = 0;
			allProjectedPointsRGB.at<float>(i, 5) = 0;
		}
		
		Mat test2;
		m_cameras[numOfView]->getFrame().copyTo(test2);

		// fill allProjectedPointsRGB with the RGB values
		for (int r = 0; r < test2.rows; r++)
		{
			for (int c = 0; c < test2.cols; c++)
			{
				for (int i = 0; i < projectedPoints.size(); i++)
				{
					if (c == (int)(projectedPoints[i].x) && r == (int)(projectedPoints[i].y))
					{
						allProjectedPointsRGB.at<float>(i, 3) = test2.at<Vec3b>(r, c)[0];
						allProjectedPointsRGB.at<float>(i, 4) = test2.at<Vec3b>(r, c)[1];
						allProjectedPointsRGB.at<float>(i, 5) = test2.at<Vec3b>(r, c)[2];

					}
				}
			}
		}
		
		int binSize = 16;
		vector<vector<vector<int>>> histPerson1(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
		vector<vector<vector<int>>> histPerson2(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
		vector<vector<vector<int>>> histPerson3(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
		vector<vector<vector<int>>> histPerson4(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));

		for (int i = 0; i < projectedPoints.size(); i++)
		{
			//the bin number for the color values
			int r = allProjectedPointsRGB.at<float>(i, 3) / 16;
			int g = allProjectedPointsRGB.at<float>(i, 4) / 16;
			int b = allProjectedPointsRGB.at<float>(i, 5) / 16;

			int label = allProjectedPointsRGB.at<float>(i, 2);

			switch (label)
			{
			case 0: histPerson1[r][g][b] += 1;
				//	cout << r << " " << g << " " << b << " " << histPerson1[r][g][b] << endl;
				break;
			case 1: histPerson2[r][g][b] += 1;
				//	cout << r << " " << g << " " << b << " " << histPerson2[r][g][b] << endl;
				break;
			case 2: histPerson3[r][g][b] += 1;
				//	cout << r << " " << g << " " << b << " " << histPerson3[r][g][b] << endl;
				break;
			case 3: histPerson4[r][g][b] += 1;
				//	cout << r << " " << g << " " << b << " " << histPerson4[r][g][b] << endl;
				break;
			}
		}
		
		//return
		h1 = histPerson1;
		h2 = histPerson2;
		h3 = histPerson3;
		h4 = histPerson4;
		complVox.copyTo(complVoxR);
		centers = centerPoints;
		//---------------------------------------------------------------------//
	}
	void Scene3DRenderer::getHistoOff(vector<vector<vector<int>>> &h1, vector<vector<vector<int>>> &h2, vector<vector<vector<int>>> &h3, vector<vector<vector<int>>> &h4)
	{
		//===============================================================================================================
		//voxel stuff
		//get the visible voxels from frame 10 
		
			//voxel
			vector<Reconstructor::Voxel*> visVoxels(m_reconstructor.getVisibleVoxels());
			int clusterCount = 4;
			int sizeOfVisVoxels = visVoxels.size();
			Mat positions(sizeOfVisVoxels, 2, CV_32F);
			Mat center, bestlabels;

			//get points of voxels (x,y)
			//TODO: ignore height of the voxels, so we dont need all visvoxels. only add the ones from the upper body. 
			for (int r = 0; r < sizeOfVisVoxels; r++)
			{
				positions.at<float>(r, 0) = visVoxels[r]->x;
				positions.at<float>(r, 1) = visVoxels[r]->y;
			}
			//cluster. using KMEANS_PP_CENTERS gives uniformly distributed initial centers. this makes the chance of a local minimum very slim. 
			kmeans(positions, clusterCount, bestlabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.5), 3, KMEANS_PP_CENTERS, center);

			//project cluster center into image,
			vector<Point3d> centerPoints;
			for (int i = 0; i < center.rows; i++)
			{
				centerPoints.push_back(Point3d(center.at<float>(i, 0), center.at<float>(i, 1), 0));
			}
			vector<Point2d> projectedCenters;

			projectPoints(centerPoints, m_cameras[1]->getRvec(), m_cameras[1]->getTvec(), m_cameras[1]->getCamMatrix(), m_cameras[1]->getDistCoeff(), projectedCenters);


			//Mat test;
			//m_cameras[1]->getFrame().copyTo(test);
			//draw circles at the center points
			//circle(test, projectedCenters[0], 10, Scalar(255, 255, 255), CV_FILLED, 8, 0);
			//circle(test, projectedCenters[1], 10, Scalar(255, 255, 255), CV_FILLED, 8, 0);
			//circle(test, projectedCenters[2], 10, Scalar(255, 255, 255), CV_FILLED, 8, 0);
			//circle(test, projectedCenters[3], 10, Scalar(255, 255, 255), CV_FILLED, 8, 0);
			//imshow("test2", test);

			//project points with height
			//bestlabels corresponds to positions. so the first item in bestlabel is also first item in position. 
			Mat complVox(sizeOfVisVoxels, 4, CV_32F);
			for (int i = 0; i < bestlabels.rows; i++)
			{
				complVox.at<float>(i, 0) = visVoxels[i]->x;
				complVox.at<float>(i, 1) = visVoxels[i]->y;
				complVox.at<float>(i, 2) = visVoxels[i]->z;

				int label = bestlabels.at<int>(i, 0);
				complVox.at<float>(i, 3) = label;

				//set the voxel color, depending on the label
				/*switch (label)
				{
				case 0: visVoxels[i]->color = Scalar(1.0f, 0.0f, 0.0f, 1.0f);
					break;
				case 1: visVoxels[i]->color = Scalar(0.0f, 1.0f, 0.0f, 1.0f);
					break;
				case 2: visVoxels[i]->color = Scalar(0.0f, 0.0f, 1.0f, 1.0f);
					break;
				case 3: visVoxels[i]->color = Scalar(1.0f, 0.0f, 1.0f, 1.0f);
					break;
				default:visVoxels[i]->color = Scalar(0.5f, 0.5f, 0.5f, 1.0f);
					break;
				}*/
			}
			m_reconstructor.setVisibleVoxels(visVoxels);

			vector<Point3d> pointsWithHeight;
			for (int i = 0; i < complVox.rows; i++)
			{
				pointsWithHeight.push_back(Point3d(complVox.at<float>(i, 0), complVox.at<float>(i, 1), complVox.at<float>(i, 2)));
			}
			//projected points with height
			vector<Point2d> projectedPoints;
			projectPoints(pointsWithHeight, m_cameras[1]->getRvec(), m_cameras[1]->getTvec(), m_cameras[1]->getCamMatrix(), m_cameras[1]->getDistCoeff(), projectedPoints);

			//create matrix with all projected points , their labels and RGB colors,
			Mat allProjectedPointsRGB(projectedPoints.size(), 6, CV_32F);
			for (int i = 0; i < projectedPoints.size(); i++)
			{
				allProjectedPointsRGB.at<float>(i, 0) = projectedPoints[i].x;
				allProjectedPointsRGB.at<float>(i, 1) = projectedPoints[i].y;
				allProjectedPointsRGB.at<float>(i, 2) = bestlabels.at<int>(i, 0);
				allProjectedPointsRGB.at<float>(i, 3) = 0;
				allProjectedPointsRGB.at<float>(i, 4) = 0;
				allProjectedPointsRGB.at<float>(i, 5) = 0;
			}
			//cout <<"First "<< allProjectedPointsRGB << endl;
			//find values new try


			//find values try 1

			Mat test2;
			m_cameras[1]->getFrame().copyTo(test2);

			// fill allProjectedPointsRGB with the RGB values
			for (int r = 0; r < test2.rows; r++)
			{
				for (int c = 0; c < test2.cols; c++)
				{
					for (int i = 0; i < projectedPoints.size(); i++)
					{
						if (c == (int)(projectedPoints[i].x) && r == (int)(projectedPoints[i].y))
						{
							allProjectedPointsRGB.at<float>(i, 3) = test2.at<Vec3b>(r, c)[0];
							allProjectedPointsRGB.at<float>(i, 4) = test2.at<Vec3b>(r, c)[1];
							allProjectedPointsRGB.at<float>(i, 5) = test2.at<Vec3b>(r, c)[2];

						}
					}
				}
			}
			//cout << complVox2 << endl;
			//cout << "points size; " << projectedPoints.size()<< endl;
			//cout << "count; " << count << endl; 
			//cout << "loss in points(float to int); " << projectedPoints.size() - count << endl;


			//----------------------create color model with allProjectedPointsRGB--------------//
			//4 histograms, 1 per person. each histogram has 16 bins per color (0-15, 16-31, etc) and 1 value for how many there is of a color. 
			//vector<vector<vector<int>>> R, G and B go from 0 to 15, for the 16 bins, # is the amount of pixels with that combination of color values. 
			//http://stackoverflow.com/questions/29305621/problems-using-3-dimensional-vector for the vector.
			int binSize = 16;
			vector<vector<vector<int>>> histPerson1(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
			vector<vector<vector<int>>> histPerson2(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
			vector<vector<vector<int>>> histPerson3(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));
			vector<vector<vector<int>>> histPerson4(binSize, vector<vector<int>>(binSize, vector<int>(binSize, 0)));

			for (int i = 0; i < projectedPoints.size(); i++)
			{
				//the bin number for the color values
				int r = allProjectedPointsRGB.at<float>(i, 3) / 16;
				int g = allProjectedPointsRGB.at<float>(i, 4) / 16;
				int b = allProjectedPointsRGB.at<float>(i, 5) / 16;

				int label = allProjectedPointsRGB.at<float>(i, 2);

				switch (label)
				{
				case 0: histPerson1[r][g][b] += 1;
					//	cout << r << " " << g << " " << b << " " << histPerson1[r][g][b] << endl;
					break;
				case 1: histPerson2[r][g][b] += 1;
					//	cout << r << " " << g << " " << b << " " << histPerson2[r][g][b] << endl;
					break;
				case 2: histPerson3[r][g][b] += 1;
					//	cout << r << " " << g << " " << b << " " << histPerson3[r][g][b] << endl;
					break;
				case 3: histPerson4[r][g][b] += 1;
					//	cout << r << " " << g << " " << b << " " << histPerson4[r][g][b] << endl;
					break;
				}
			}

			h1 = histPerson1;
			h2 = histPerson2;
			h3 = histPerson3;
			h4 = histPerson4;
			
			//---------------------------------------------------------------------//



			//notes
			/*//find values try 2
			//create matrix with projected points for LUT
			Mat complVox3(projectedPointsWithHeight.size(), 2, CV_32F);
			for (int i = 0; i < projectedPointsWithHeight.size(); i++)
			{
			complVox3.at<int>(i, 0) = (int)projectedPointsWithHeight[i].x;
			complVox3.at<int>(i, 1) = (int)projectedPointsWithHeight[i].y;
			}

			Mat test2;
			m_cameras[1]->getFrame().copyTo(test2);
			Mat lutOutput;
			LUT(test2, complVox3, lutOutput);*/
			//create color model



			//part of splitting voxels in half is confusing leave it for later

			/*//draw on upper points and get number of upper points(count)
			int count = 0;
			for (int i= (int)(projectedPointsWithHeight.size() / 2); i<projectedPointsWithHeight.size();i++)
			{
			circle(test, projectedPointsWithHeight[i], 1, Scalar(255, 255, 255), CV_FILLED, 8, 0);
			//cout <<"projected points with height first"<< projectedPointsWithHeight[i] << endl;
			count++;
			}
			imshow("test3", test);
			cout << count;
			//create matrix with points  and labels,
			//int sizeX =
			Mat complVox2(count, 3, CV_32F);
			for (int i = 0; i<count; i++)
			{
			complVox2.at<float>(i, 0) = projectedPointsWithHeight[i].x;
			complVox2.at<float>(i, 1) = projectedPointsWithHeight[i].y;
			complVox2.at<float>(i, 2) = bestlabels.at<int>(i, 0);
			cout << projectedPointsWithHeight[i] << endl;
			}
			//cout << complVox2 << endl;*/

		

	}

} /* namespace nl_uu_science_gmt */
