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
  srand(time(NULL));
  unsigned int n_matches = good_matches.size();
  unsigned int its = 1;
  std::cout << std::rand() % n_matches << std::endl;

  // set up data structures to be used iteratively
  Mat A(7,9,CV_32F, cvScalar(0.));
  std::vector< Mat > rows;
  for (unsigned int i = 0; i < 7; i++)
  {
    rows.push_back(A.rowRange(i,i+1));
  }

  // RANSAC to find best fundamental matrix
  for (unsigned int it = 0 ; it < its ; it++)
  {
    // select matches and set up matrix
    for (unsigned int i = 0; i < 7; i++)
    {
      DMatch* this_match = &good_matches[rand() % n_matches];
      float data[9] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};
      Mat row = Mat(1,9,CV_32F, data);
      row.copyTo(rows[i]);
    }

    // get matrix based on these matches

    // compute number of inliers
  }
  
}
/*
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }