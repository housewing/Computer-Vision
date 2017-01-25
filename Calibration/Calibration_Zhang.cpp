#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

class Point3D{
public:
	double x;
	double y;
	double z;

	Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

Mat getHomography(vector<Point3D> _O, vector<Point> _A)
{
	float* pointSet = new float[_A.size() * 18]; //homography matrix
	for (int i = 0; i < _A.size(); i++)
	{
		pointSet[i * 18] = 0;
		pointSet[i * 18 + 1] = 0;
		pointSet[i * 18 + 2] = 0;
		pointSet[i * 18 + 3] = -_O[i].x;
		pointSet[i * 18 + 4] = -_O[i].y;
		pointSet[i * 18 + 5] = -1;
		pointSet[i * 18 + 6] = _A[i].y * _O[i].x;
		pointSet[i * 18 + 7] = _A[i].y * _O[i].y;
		pointSet[i * 18 + 8] = _A[i].y;
		pointSet[i * 18 + 9] = _O[i].x;
		pointSet[i * 18 + 10] = _O[i].y;
		pointSet[i * 18 + 11] = 1;
		pointSet[i * 18 + 12] = 0;
		pointSet[i * 18 + 13] = 0;
		pointSet[i * 18 + 14] = 0;
		pointSet[i * 18 + 15] = -_A[i].x * _O[i].x;
		pointSet[i * 18 + 16] = -_A[i].x * _O[i].y;
		pointSet[i * 18 + 17] = -_A[i].x;
	}

	Mat mat8x9 = Mat(_A.size() * 2, 9, CV_32F, pointSet);
	//cout << mat8x9 << endl;

	cv::SVD svd(mat8x9, SVD::FULL_UV); //solved by SVD
	Mat v = svd.vt.t();
	//cout << "v size " << v.size() << ", col " << v.cols << ", row " << v.rows << endl;
	Mat tmp = v.col(8) / v.at<float>(8, 8); // get the last column of v
	//cout << "lats column of v " << tmp << endl;

	Mat H = (Mat_<float>(3, 3) << tmp.at<float>(0, 0), tmp.at<float>(1, 0), tmp.at<float>(2, 0),
		tmp.at<float>(3, 0), tmp.at<float>(4, 0), tmp.at<float>(5, 0),
		tmp.at<float>(6, 0), tmp.at<float>(7, 0), tmp.at<float>(8, 0));
	//cout << "H matrix " << H << endl;

	return H; //return H matrix
}

Mat getRT(Mat _K, Mat _H)
{
	Mat rA = _K.inv() * _H;
	double rA_length = sqrt(pow(rA.at<float>(0, 0), 2) + pow(rA.at<float>(1, 0), 2) + pow(rA.at<float>(2, 0), 2)); //get unit length of r1
	//cout << rA_length << endl;

	Mat newrA = rA / rA_length; //make last value be 1
	Mat rA1 = newrA.col(0);
	Mat rA2 = newrA.col(1);
	//cout << "rA1 " << rA1 << endl;

	Mat rA3 = rA1.cross(rA2); //r3 = r1 cross r2
	double rA3_length = sqrt(pow(rA3.at<float>(0, 0), 2) + pow(rA3.at<float>(0, 1), 2) + pow(rA3.at<float>(0, 2), 2)); // get unit length of r3
	rA3 = rA3 / rA3_length; //make last value be 1
	//cout << "rA3 " << rA3 << endl;

	rA2 = rA3.cross(rA1); //r2 = r3 cross r1
	//cout << "rA2 " << rA2 << endl;

	Mat rAt = (_K.inv() * _H).col(2) / rA_length; //get Rt from K inv * H
	//cout << "rAt " << rAt << endl;

	Mat RT = (Mat_<float>(3, 4) << // get Rt matrix
		rA1.at<float>(0, 0), rA2.at<float>(0, 0), rA3.at<float>(0, 0), rAt.at<float>(0, 0),
		rA1.at<float>(1, 0), rA2.at<float>(1, 0), rA3.at<float>(1, 0), rAt.at<float>(1, 0),
		rA1.at<float>(2, 0), rA2.at<float>(2, 0), rA3.at<float>(2, 0), rAt.at<float>(2, 0));
	//cout << "RT " << RT << endl;

	return RT;
}

int main(int argc, char** argv)
{
	vector<Point> O;
	O.push_back(Point(0, 0));
	O.push_back(Point(0, 1));
	O.push_back(Point(1, 1));
	O.push_back(Point(1, 0));

	vector<Point3D> O1;
	O1.push_back(Point3D(40, 40, 0)); //40 40 0
	O1.push_back(Point3D(40, 10, 0)); //40 10 0
	O1.push_back(Point3D(10, 10, 0)); //10 10 0
	O1.push_back(Point3D(10, 40, 0)); //10 40 0

	vector<Point3D> O2;
	O2.push_back(Point3D(10, 40, 0)); //10 40 0
	O2.push_back(Point3D(40, 40, 0)); //40 40 0
	O2.push_back(Point3D(40, 10, 0)); //40 10 0
	O2.push_back(Point3D(10, 10, 0)); //10 10 0

	vector<Point3D> O3;
	O3.push_back(Point3D(40, 40, 0)); //40 40 0
	O3.push_back(Point3D(40, 10, 0)); //40 10 0
	O3.push_back(Point3D(10, 10, 0)); //10 10 0
	O3.push_back(Point3D(10, 40, 0)); //10 40 0

	vector<Point> A;
	A.push_back(Point(215, 235));
	A.push_back(Point(375, 291));
	A.push_back(Point(378, 484));
	A.push_back(Point(239, 404));

	vector<Point> B;
	B.push_back(Point(493, 285));
	B.push_back(Point(591, 219));
	B.push_back(Point(568, 380));
	B.push_back(Point(478, 475));

	vector<Point> C;
	C.push_back(Point(388, 120));
	C.push_back(Point(539, 143));
	C.push_back(Point(427, 188));
	C.push_back(Point(261, 152));

	Mat HA = getHomography(O1, A);
	Mat HB = getHomography(O2, B);
	Mat HC = getHomography(O3, C);
	//cout << HA << endl;
	//cout << HB << endl;
	//cout << HC << endl;

	double HA11 = HA.at<float>(0, 0); //H1
	double HA12 = HA.at<float>(1, 0);
	double HA13 = HA.at<float>(2, 0);
	double HA21 = HA.at<float>(0, 1); //H2
	double HA22 = HA.at<float>(1, 1);
	double HA23 = HA.at<float>(2, 1);
	//cout << "11 " << HA11 << ", 12 " << HA12 << ", 13 " << HA13 << endl;
	//cout << "21 " << HA21 << ", 22 " << HA22 << ", 23 " << HA23 << endl;

	double HB11 = HB.at<float>(0, 0); //H1
	double HB12 = HB.at<float>(1, 0);
	double HB13 = HB.at<float>(2, 0);
	double HB21 = HB.at<float>(0, 1); //H2
	double HB22 = HB.at<float>(1, 1);
	double HB23 = HB.at<float>(2, 1);
	//cout << "11 " << HB11 << ", 12 " << HB12 << ", 13 " << HB13 << endl;
	//cout << "21 " << HB21 << ", 22 " << HB22 << ", 23 " << HB23 << endl;

	double HC11 = HC.at<float>(0, 0); //H1
	double HC12 = HC.at<float>(1, 0);
	double HC13 = HC.at<float>(2, 0);
	double HC21 = HC.at<float>(0, 1); //H2
	double HC22 = HC.at<float>(1, 1);
	double HC23 = HC.at<float>(2, 1);
	//cout << "11 " << HC11 << ", 12 " << HC12 << ", 13 " << HC13 << endl;
	//cout << "21 " << HC21 << ", 22 " << HC22 << ", 23 " << HC23 << endl;

	Mat mat6x6 = (Mat_<float>(6, 6) << // solved by conic
		HA11*HA21, HA11*HA22 + HA12*HA21, HA11*HA23 + HA13*HA21, HA12*HA22, HA12*HA23 + HA13*HA22, HA13*HA23,
		HA11*HA11 - HA21*HA21, 2 * (HA11*HA12 - HA21*HA22), 2 * (HA11*HA13 - HA21*HA23), HA12*HA12 - HA22*HA22, 2 * (HA12*HA13 - HA22*HA23), HA13*HA13 - HA23*HA23,
		HB11*HB21, HB11*HB22 + HB12*HB21, HB11*HB23 + HB13*HB21, HB12*HB22, HB12*HB23 + HB13*HB22, HB13*HB23,
		HB11*HB11 - HB21*HB21, 2 * (HB11*HB12 - HB21*HB22), 2 * (HB11*HB13 - HB21*HB23), HB12*HB12 - HB22*HB22, 2 * (HB12*HB13 - HB22*HB23), HB13*HB13 - HB23*HB23,
		HC11*HC21, HC11*HC22 + HC12*HC21, HC11*HC23 + HC13*HC21, HC12*HC22, HC12*HC23 + HC13*HC22, HC13*HC23,
		HC11*HC11 - HC21*HC21, 2 * (HC11*HC12 - HC21*HC22), 2 * (HC11*HC13 - HC21*HC23), HC12*HC12 - HC22*HC22, 2 * (HC12*HC13 - HC22*HC23), HC13*HC13 - HC23*HC23);

	SVD svd(mat6x6);
	Mat v = svd.vt.t();
	//cout << v.col(5) / v.at<float>(5,5) << endl;

	Mat tmp = v.col(5) / v.at<float>(5, 5); // get last column of v
	Mat W = (Mat_<float>(3, 3) <<
		tmp.at<float>(0, 0), tmp.at<float>(1, 0), tmp.at<float>(2, 0),
		tmp.at<float>(1, 0), tmp.at<float>(3, 0), tmp.at<float>(4, 0),
		tmp.at<float>(2, 0), tmp.at<float>(4, 0), tmp.at<float>(5, 0));
	//cout << W << endl;

	W = W.inv();
	W = W / W.at<float>(2, 2);

	double c = W.at<float>(0, 2);
	double e = W.at<float>(1, 2);
	double d = sqrt(W.at<float>(1, 1) - e*e);
	double b = (W.at<float>(0, 1) - c*e) / d;
	double a = sqrt(W.at<float>(0, 0) - b*b - c*c);
	//cout << "A " << a << " B " << b << " C " << c << " D " << d << " E " << e << endl;

	Mat K = (Mat_<float>(3, 3) << a, b, c, 0, d, e, 0, 0, 1); //get K matrix
	cout << "K " << endl << K << endl << endl;

	Mat RtA = getRT(K, HA); // get Rt matrix
	Mat RtB = getRT(K, HB); // get Rt matrix
	Mat RtC = getRT(K, HC); // get Rt matrix
	cout << "RtA " << endl << RtA << endl << endl;
	cout << "RtB " << endl << RtB << endl << endl;
	cout << "RtC " << endl << RtC << endl << endl;

	//verify 3D to 2D
	Mat mat4x1 = (Mat_<float>(4, 1) << O3[3].x, O3[3].y, O3[3].z, 1);
	Mat output = K * RtC * mat4x1;
	cout << "O3[3] to C[3]  " << output.at<float>(0, 0) / output.at<float>(2, 0) << " " << output.at<float>(1, 0) / output.at<float>(2, 0) << endl;

	//verify camera
	Mat cam = (Mat_<float>(4, 4) << 
		RtA.at<float>(0, 0), RtA.at<float>(0, 1), RtA.at<float>(0, 2), RtA.at<float>(0, 3),
		RtA.at<float>(1, 0), RtA.at<float>(1, 1), RtA.at<float>(1, 2), RtA.at<float>(1, 3),
		RtA.at<float>(2, 0), RtA.at<float>(2, 1), RtA.at<float>(2, 2), RtA.at<float>(2, 3),
		0.0, 0.0, 0.0, 1.0);
	cout << "camera " << endl << cam.inv() << endl;

	system("pause");
	return 0;
}