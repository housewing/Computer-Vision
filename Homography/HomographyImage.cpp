#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv[])
{
	Mat src = imread("ntust.png");
	Mat dst = Mat(100, 400, CV_8UC3, Scalar::all(0));

	int x1 = 54, y1 = 45;
	int x2 = 58, y2 = 196;
	int x3 = 332, y3 = 172;
	int x4 = 329, y4 = 91;
	//circle(src, Point(x1, y1), 1, Scalar(0, 255, 0), 1, 8, 0);
	//circle(src, Point(x2, y2), 1, Scalar(0, 255, 0), 1, 8, 0);
	//circle(src, Point(x3, y3), 1, Scalar(0, 255, 0), 1, 8, 0);
	//circle(src, Point(x4, y4), 1, Scalar(0, 255, 0), 1, 8, 0);

	int p1x = 0, p1y = 0;
	int p2x = 0, p2y = 100;
	int p3x = 400, p3y = 100;
	int p4x = 400, p4y = 0;

	Mat mat8x8 = (Mat_<float>(8, 8) << x1, y1, 1, 0, 0, 0, -p1x*x1, -p1x*y1,
		0, 0, 0, x1, y1, 1, -p1y*x1, -p1y*y1,
		x2, y2, 1, 0, 0, 0, -p2x*x2, -p2x*y2,
		0, 0, 0, x2, y2, 1, -p2y*x2, -p2y*y2,
		x3, y3, 1, 0, 0, 0, -p3x*x3, -p3x*y3,
		0, 0, 0, x3, y3, 1, -p3y*x3, -p3y*y3,
		x4, y4, 1, 0, 0, 0, -p4x*x4, -p4x*y4,
		0, 0, 0, x4, y4, 1, -p4y*x4, -p4y*y4);
	Mat mat8x1 = (Mat_<float>(8, 1) << p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y);

	////DLT
	//Mat mat8x9 = (Mat_<float>(8, 9) <<
	//	0, 0, 0, -1 * x1, -1 * y1, -1, p1y*x1, p1y*y1, p1y,
	//	x1, y1, 1, 0, 0, 0, -1 * p1x*x1, -1 * p1x*y1, -1 * p1x,
	//	0, 0, 0, -1 * x2, -1 * y2, -1, p2y*x2, p2y*y2, p2y,
	//	x2, y2, 1, 0, 0, 0, -1 * p2x*x2, -1 * p2x*y2, -1 * p2x,
	//	0, 0, 0, -1 * x3, -1 * y3, -1, p3y*x3, p3y*y3, p3y,
	//	x3, y3, 1, 0, 0, 0, -1 * p3x*x3, -1 * p3x*y3, -1 * p3x,
	//	0, 0, 0, -1 * x4, -1 * y4, -1, p4y*x4, p4y*y4, p4y,
	//	x4, y4, 1, 0, 0, 0, -1 * p4x*x4, -1 * p4x*y4, -1 * p4x);
	////cout << mat8x9 << endl;

	//cv::SVD svd(mat8x9);
	//Mat v = svd.vt.t();
	////cout << v << endl;
	//Mat tmp1 = v.col(7) / v.at<float>(8, 7);
	////cout << tmp1 << endl;
	//Mat H1 = (Mat_<float>(3, 3) <<	tmp1.at<float>(0, 0), tmp1.at<float>(1, 0), tmp1.at<float>(2, 0),
	//								tmp1.at<float>(3, 0), tmp1.at<float>(4, 0), tmp1.at<float>(5, 0),
	//								tmp1.at<float>(6, 0), tmp1.at<float>(7, 0), tmp1.at<float>(8, 0));
	//cout << H1 << endl;
	//Mat mat3x1 = (Mat_<float>(3, 1) << x3, y3, 1);
	//Mat tmp2 = H1 * mat3x1;
	//cout << tmp2 << endl;
	//cout << tmp2.at<float>(0, 0) / tmp2.at<float>(2, 0) << " " << tmp2.at<float>(1, 0) / tmp2.at<float>(2, 0) << endl;

	Mat invMat = mat8x8.inv();
	Mat tmp = Mat(8, 1, CV_32F);
	tmp = invMat * mat8x1;
	Mat H = (Mat_<float>(3, 3) <<	tmp.at<float>(0, 0), tmp.at<float>(1, 0), tmp.at<float>(2, 0),
									tmp.at<float>(3, 0), tmp.at<float>(4, 0), tmp.at<float>(5, 0),
									tmp.at<float>(6, 0), tmp.at<float>(7, 0), 1);
	//cout << H << endl;

	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			Mat t1 = (Mat_<float>(3, 1) << x, y, 1);
			Mat t2 = H.inv() * t1;
			int tmpx = t2.at<float>(0, 0) / t2.at<float>(2, 0);
			int tmpy = t2.at<float>(1, 0) / t2.at<float>(2, 0);

			double b = src.at<Vec3b>(tmpy, tmpx)[0];
			double g = src.at<Vec3b>(tmpy, tmpx)[1];
			double r = src.at<Vec3b>(tmpy, tmpx)[2];

			Vec3b pixel(b, g, r);
			dst.at<Vec3b>(y, x) = pixel;
		}
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	return 0;
}