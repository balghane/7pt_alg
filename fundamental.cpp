/*
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
void readme();
/*
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }
  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }


  //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
  int minHessian = 400;
  Ptr<SIFT> detector = SIFT::create();
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
  detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );


  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< std::vector< DMatch > > matches;
  matcher.knnMatch( descriptors_1, descriptors_2, matches, 2 );

  // get good matches
  std::vector< DMatch > good_matches;
  float ratio = 0.5f;
  for( int i = 0; i < descriptors_1.rows; i++ )
  { 
    if (matches[i][0].distance < ratio * matches[i][1].distance)
    { 
      good_matches.push_back( matches[i][0]); 
    }
  }

  //-- Step 3: Calculate fundamental matrix

  // RANSAC to find best fundamental matrix
  srand(time(NULL));
  unsigned int n_matches = good_matches.size();
  unsigned int its = 1;
  Mat pts1(7,2,CV_32F);
  Mat pts2(7,2,CV_32F);
  for (unsigned int it = 0 ; it < its ; it++)
  {
    // select matches and set up matrices
    for (unsigned int i = 0; i < 7; i++)
    {
      DMatch* this_match = &good_matches[rand() % n_matches];
      Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
      Point2f pt2 = keypoints_2[this_match->trainIdx].pt;
      pts1.at<float>(i,0) = pt1.x;
      pts1.at<float>(i,1) = pt1.y;
      pts2.at<float>(i,0) = pt2.x;
      pts2.at<float>(i,1) = pt2.y;
    }

    // compute F based on those matches
    Mat F(3,3,CV_32F);
    run7Point(pts1, pts2, F);

    // compute number of inliers

  }
  
}
/*
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }