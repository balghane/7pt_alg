/*
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
void readme();


void normColumn(Mat& m, unsigned int col, double& mn, double& avg_dist)
{
  unsigned int n = 7;
  mn = 0;

  for (int i = 0; i < 7; i++)
  {
    mn = mn + m.at<double>(i,col);
  }

  mn /= n;


  for (int i = 0; i < 7; i++)
  {
    m.at<double>(i,col) = m.at<double>(i,col) - mn;
  }

  avg_dist = 0;

  for (int i = 0; i < 7; i++)
  {
    avg_dist += abs(m.at<double>(i,col) - mn);
  }

  avg_dist /= n;

  for (int i = 0; i < 7; i++)
  {
    m.at<double>(i,col)  /= avg_dist;
  }
}

void normPoints(Mat& m1, Mat& m2, Mat& T1, Mat& T2)
{
  double mn_x1, mn_y1, mn_x2, mn_y2;
  double avg_dist_x1, avg_dist_y1, avg_dist_x2, avg_dist_y2;

  normColumn(m1, 0, mn_x1, avg_dist_x1);
  normColumn(m1, 1, mn_y1, avg_dist_y1);
  normColumn(m2, 0, mn_x2, avg_dist_x2);
  normColumn(m2, 1, mn_y2, avg_dist_y2);

  double data11[9] = {1/avg_dist_x1, 0, 0, 0, 1/avg_dist_y1, 0, 0, 0, 1.0f};
  Mat scale1( 3, 3, CV_64F, data11 );
  double data12[9] = {1, 0, -mn_x1, 0, 1, -mn_y1, 0, 0, 1};
  Mat shift1( 3, 3, CV_64F, data12 );

  double data21[9] = {1/avg_dist_x2, 0, 0, 0, 1/avg_dist_y2, 0, 0, 0, 1.0f};
  Mat scale2( 3, 3, CV_64F, data21 );
  double data22[9] = {1, 0, -mn_x2, 0, 1, -mn_y2, 0, 0, 1};
  Mat shift2( 3, 3, CV_64F, data22 );

  T1 = scale1 * shift1;
  T2 = scale2 * shift2;
}

std::vector<Mat> run7Point( Mat _m1, Mat _m2)
{
  double a[7*9], w[7], u[9*9], v[9*9], c[4], r[3] = {0};
  double* f1, *f2;
  double t0, t1, t2;
  Mat A( 7, 9, CV_64F, a );
  Mat U( 7, 9, CV_64F, u );
  Mat Vt( 9, 9, CV_64F, v );
  Mat W( 7, 1, CV_64F, w );
  Mat coeffs( 1, 4, CV_64F, c );
  Mat roots( 1, 3, CV_64F, r );
  Mat T1 = Mat::eye(3, 3, CV_64F);
  Mat T2 = Mat::eye(3, 3, CV_64F);
  std::vector<Mat> ret = std::vector<Mat>();
  int i, k, n;

  normPoints(_m1, _m2, T1, T2);

  // form a linear system: i-th row of A(=a) represents
  // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
  for( i = 0; i < 7; i++ )
  {
      double x0 = _m1.at<double>(i, 0), y0 = _m1.at<double>(i, 1);
      double x1 = _m2.at<double>(i, 0), y1 = _m2.at<double>(i, 1);

      a[i*9+0] = x1*x0;
      a[i*9+1] = x1*y0;
      a[i*9+2] = x1;
      a[i*9+3] = y1*x0;
      a[i*9+4] = y1*y0;
      a[i*9+5] = y1;
      a[i*9+6] = x0;
      a[i*9+7] = y0;
      a[i*9+8] = 1;
  }

  // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
  // the solution is linear subspace of dimensionality 2.
  // => use the last two singular vectors as a basis of the space
  // (according to SVD properties)
  SVDecomp( A, W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV );
  f1 = v + 7*9;
  f2 = v + 8*9;

  // f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary fundamental matrix,
  // as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
  // so f ~ lambda*f1 + (1 - lambda)*f2.
  // use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
  // it will be a cubic equation.
  // find c - polynomial coefficients.
  for( i = 0; i < 9; i++ )
      f1[i] -= f2[i];

  t0 = f2[4]*f2[8] - f2[5]*f2[7];
  t1 = f2[3]*f2[8] - f2[5]*f2[6];
  t2 = f2[3]*f2[7] - f2[4]*f2[6];

  c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

  c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
  f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
  f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
  f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
  f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
  f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
  f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

  t0 = f1[4]*f1[8] - f1[5]*f1[7];
  t1 = f1[3]*f1[8] - f1[5]*f1[6];
  t2 = f1[3]*f1[7] - f1[4]*f1[6];

  c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
  f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
  f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
  f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
  f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
  f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
  f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

  c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

  // solve the cubic equation; there can be 1 to 3 roots ...
  n = solveCubic( coeffs, roots );

  if( n < 1 || n > 3 )
      return ret;

  for( k = 0; k < n; k++)
  {
      double data[9];

      // for each root form the fundamental matrix
      double lambda = r[k], mu = 1.;
      double s = f1[8]*r[k] + f2[8];

      // normalize each matrix, so that F(3,3) (~data[8]) == 1
      if( fabs(s) > DBL_EPSILON )
      {
          mu = 1./s;
          lambda *= mu;
          data[8] = 1.;
      }
      else
          data[8] = 0.;

      for( i = 0; i < 8; i++ )
          data[i] = f1[i]*lambda + f2[i]*mu;

      Mat tmp(3,3,CV_64F, data);
      Mat tmp2 = (T2.t() * tmp) * T1;
      ret.push_back(tmp2 / tmp2.at<double>(2,2));

      /* test of x'^T * F * x = 0
      for (int i = 0; i < 7; i++)
      {
        double p1d[3] = {_m1.at<double>(i, 0), _m1.at<double>(i, 1), 1};
        double p2d[3] = {_m2.at<double>(i, 0), _m2.at<double>(i, 1), 1};
        Mat p1(3,1,CV_64F,p1d), p2(3,1,CV_64F,p2d);
        std::cout << "test: " << p2.t() * tmp * p1 << std::endl;
      }
      */
  }

  return ret;
}

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
  unsigned int its = 3000;
  unsigned int n_inliers = 0;
  unsigned int best_inliers = 0;
  Mat best_F(3,3,CV_64F);
  double threshold = 0.0000001;
  for (unsigned int it = 0; it < its; it++)
  {
    Mat pts1(7,2,CV_64F);
    Mat pts2(7,2,CV_64F);
    std::vector<unsigned int> inds = std::vector<unsigned int>();
    // select matches and set up matrices
    for (unsigned int i = 0; i < 7; i++)
    {
      unsigned int ind = rand() % n_matches;
      inds.push_back(ind);
      DMatch* this_match = &good_matches[ind];
      Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
      Point2f pt2 = keypoints_2[this_match->trainIdx].pt;
      pts1.at<double>(i,0) = (double)pt1.x;
      pts1.at<double>(i,1) = (double)pt1.y;
      pts2.at<double>(i,0) = (double)pt2.x;
      pts2.at<double>(i,1) = (double)pt2.y;
    }

    // compute possible Fs based on those matches
    std::vector<Mat> sols = run7Point(pts1.clone(), pts2.clone());
    
    // for each solution, try it out
    for (int i = 0; i < sols.size(); i++)
    {
      // init
      n_inliers = 0;
      Mat F = sols[i];
      // std::cout << F << std::endl;

      // check each point to see if it's an inlier
      for (int j = 0; j < n_matches; j++)
      // for (int j = 0; j < n_matches; j++)
      {
        // get point info
        // unsigned int ind = inds[j];
        unsigned int ind = j;
        DMatch* this_match = &good_matches[ind];
        Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
        Point2f pt2 = keypoints_2[this_match->trainIdx].pt;

        double pt1_d[3] = {pt1.x, pt1.y, 1};
        Mat pt1_m(3,1,CV_64F, pt1_d);
        double pt2_d[3] = {pt2.x, pt2.y, 1};
        Mat pt2_m(3,1,CV_64F, pt2_d);

        /* test of x'^T * F * x = 0
        */
        // std::cout << "test: " << pt2_m.t() * F * pt1_m << std::endl;

        // get epipolar lines and normalize
        Mat l1(3,1,CV_64F);
        Mat l2(3,1,CV_64F);
        l1 = pt2_m.t() * F;
        l1 /= (l1.at<double>(0,0)*l1.at<double>(0,0) +
               l1.at<double>(0,1)*l1.at<double>(0,1) +
               l1.at<double>(0,2)*l1.at<double>(0,2));
        l2 = F * pt1_m;
        l2 /= (l2.at<double>(0,0)*l2.at<double>(0,0) +
               l2.at<double>(1,0)*l2.at<double>(1,0) +
               l2.at<double>(2,0)*l2.at<double>(2,0));

        Mat e1 = l1 * pt1_m;
        Mat e2 = pt2_m.t() * l2;
        double dist = (abs(e1.at<double>(0,0)) + 
                       abs(e2.at<double>(0,0))) / 2.0f;
        // std::cout << "pt1: " << pt1_m << std::endl;
        // std::cout << "test2: " << dist << std::endl;

        if (dist < threshold) n_inliers++;
      } // end for each point

      if (n_inliers > best_inliers)
      {
        best_inliers = n_inliers;
        best_F = F.clone();
      }
    } // end for each solution

  } // end for each iteration

  // show results
  std::cout << "best F: " << best_F << std::endl;
  std::cout << best_inliers << " out of " << n_matches << std::endl;

  std::vector<KeyPoint> inliers_1 = std::vector<KeyPoint>();
  std::vector<KeyPoint> inliers_2 = std::vector<KeyPoint>();
  std::vector<DMatch> inlier_matches = std::vector<DMatch>();
  // check each point to see if it's an inlier
  for (int j = 0; j < n_matches; j++)
  // for (int j = 0; j < n_matches; j++)
  {
    // get point info
    // unsigned int ind = inds[j];
    unsigned int ind = j;
    DMatch* this_match = &good_matches[ind];
    Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
    Point2f pt2 = keypoints_2[this_match->trainIdx].pt;

    double pt1_d[3] = {pt1.x, pt1.y, 1};
    Mat pt1_m(3,1,CV_64F, pt1_d);
    double pt2_d[3] = {pt2.x, pt2.y, 1};
    Mat pt2_m(3,1,CV_64F, pt2_d);

    /* test of x'^T * F * x = 0
    */
    // std::cout << "test: " << pt2_m.t() * F * pt1_m << std::endl;

    // get epipolar lines and normalize
    Mat l1(3,1,CV_64F);
    Mat l2(3,1,CV_64F);
    l1 = pt2_m.t() * best_F;
    l1 /= (l1.at<double>(0,0)*l1.at<double>(0,0) +
            l1.at<double>(0,1)*l1.at<double>(0,1) +
            l1.at<double>(0,2)*l1.at<double>(0,2));
    l2 = best_F * pt1_m;
    l2 /= (l2.at<double>(0,0)*l2.at<double>(0,0) +
            l2.at<double>(1,0)*l2.at<double>(1,0) +
            l2.at<double>(2,0)*l2.at<double>(2,0));

    Mat e1 = l1 * pt1_m;
    Mat e2 = pt2_m.t() * l2;
    double dist = (abs(e1.at<double>(0,0)) + 
                    abs(e2.at<double>(0,0))) / 2.0f;
    // std::cout << "pt1: " << pt1_m << std::endl;
    // std::cout << "test2: " << dist << std::endl;

    if (dist < threshold)
    {
      DMatch to_add = DMatch(inliers_1.size(), inliers_1.size(), dist);
      inliers_1.push_back(keypoints_1[this_match->queryIdx]);
      inliers_2.push_back(keypoints_2[this_match->trainIdx]);
      inlier_matches.push_back(to_add);
    }
  } // end for each point

  Mat img_matches;
  drawMatches( img_1, inliers_1, img_2, inliers_2,
               inlier_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imshow( "Good Matches", img_matches );
  waitKey(0);
  
}
/*
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }