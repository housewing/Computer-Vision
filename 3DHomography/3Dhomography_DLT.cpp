#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream> // read file

using namespace std;
using namespace cv;

class Point3D{
public:
	double x;
	double y;
	double z;
};

void readObj(String _objStr, vector<Point3D>& _points)
{
	Point3D *points;

	ifstream ifsDST(_objStr, ifstream::in);
	string str;
	while (getline(ifsDST, str)) // read line from obj
	{
		if (str[0] == 'v')
		{
			istringstream in(str);
			points = new Point3D();
			string tmp;
			in >> tmp >> points->x >> points->y >> points->z; //get x y z point
			_points.push_back(*points);
		}
	}
}

int main(int argc, char** argv)
{
	vector<Point3D> DST;
	vector<Point3D> SRC;

	readObj("DST.obj", DST); 
	readObj("SRC.obj", SRC);
	cout << "DST " << DST.size() << ", SRC " << SRC.size() << endl;

	//for (auto tmp : DST)
	//{
	//	cout << tmp.x << " " << tmp.y << " " << tmp.z << endl;
	//}

	float* pointSet = new float[3 * DST.size() * 16]; //store all points in array and use DLT algorithm
	for (int i = 0; i < DST.size(); i++)
	{
		pointSet[i * 48] = DST[i].x;
		pointSet[i * 48 + 1] = DST[i].y;
		pointSet[i * 48 + 2] = DST[i].z;
		pointSet[i * 48 + 3] = 1;
		pointSet[i * 48 + 4] = 0;
		pointSet[i * 48 + 5] = 0;
		pointSet[i * 48 + 6] = 0;
		pointSet[i * 48 + 7] = 0;
		pointSet[i * 48 + 8] = 0;
		pointSet[i * 48 + 9] = 0;
		pointSet[i * 48 + 10] = 0;
		pointSet[i * 48 + 11] = 0;
		pointSet[i * 48 + 12] = -SRC[i].x * DST[i].x;
		pointSet[i * 48 + 13] = -SRC[i].x * DST[i].y;
		pointSet[i * 48 + 14] = -SRC[i].x * DST[i].z;
		pointSet[i * 48 + 15] = -SRC[i].x;
		pointSet[i * 48 + 16] = 0;
		pointSet[i * 48 + 17] = 0;
		pointSet[i * 48 + 18] = 0;
		pointSet[i * 48 + 19] = 0;
		pointSet[i * 48 + 20] = DST[i].x;
		pointSet[i * 48 + 21] = DST[i].y;
		pointSet[i * 48 + 22] = DST[i].z;
		pointSet[i * 48 + 23] = 1;
		pointSet[i * 48 + 24] = 0;
		pointSet[i * 48 + 25] = 0;
		pointSet[i * 48 + 26] = 0;
		pointSet[i * 48 + 27] = 0;
		pointSet[i * 48 + 28] = -SRC[i].y * DST[i].x;
		pointSet[i * 48 + 29] = -SRC[i].y * DST[i].y;
		pointSet[i * 48 + 30] = -SRC[i].y * DST[i].z;
		pointSet[i * 48 + 31] = -SRC[i].y;
		pointSet[i * 48 + 32] = 0;
		pointSet[i * 48 + 33] = 0;
		pointSet[i * 48 + 34] = 0;
		pointSet[i * 48 + 35] = 0;
		pointSet[i * 48 + 36] = 0;
		pointSet[i * 48 + 37] = 0;
		pointSet[i * 48 + 38] = 0;
		pointSet[i * 48 + 39] = 0;
		pointSet[i * 48 + 40] = DST[i].x;
		pointSet[i * 48 + 41] = DST[i].y;
		pointSet[i * 48 + 42] = DST[i].z;
		pointSet[i * 48 + 43] = 1;
		pointSet[i * 48 + 44] = -SRC[i].z * DST[i].x;
		pointSet[i * 48 + 45] = -SRC[i].z * DST[i].y;
		pointSet[i * 48 + 46] = -SRC[i].z * DST[i].z;
		pointSet[i * 48 + 47] = -SRC[i].z;
	}

	Mat pointMat = Mat(3 * DST.size(), 16, CV_32F, pointSet);
	//cout << "pointMat " << pointMat << endl << endl;

	SVD svd(pointMat); // solve by svd
	Mat v = svd.vt.t();
	cout << "v size " << v.size() << ", col " << v.cols << ", row " << v.rows << endl;
	cout << v.col(15) / v.at<float>(15, 15) << endl;
	Mat tmp = v.col(15); // get last column of v

	Mat H = (Mat_<float>(4, 4) <<	// put v into h matrix
		tmp.at<float>(0, 0), tmp.at<float>(1, 0), tmp.at<float>(2, 0), tmp.at<float>(3, 0),
		tmp.at<float>(4, 0), tmp.at<float>(5, 0), tmp.at<float>(6, 0), tmp.at<float>(7, 0),
		tmp.at<float>(8, 0), tmp.at<float>(9, 0), tmp.at<float>(10, 0), tmp.at<float>(11, 0),
		tmp.at<float>(12, 0), tmp.at<float>(13, 0), tmp.at<float>(14, 0), tmp.at<float>(15, 0));
	cout << "H matrix " << H << endl;

	vector<Point3D> output; // store result point
	for (int i = 0; i < SRC.size(); i++)
	{
		Mat mat4x1 = (Mat_<float>(4, 1) << SRC[i].x, SRC[i].y, SRC[i].z, 1);
		Mat tmp = H.inv() * mat4x1;	//DST = H.inv * SRC

		Point3D tmp3D;
		tmp3D.x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);	//normalize x
		tmp3D.y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);	//normalize y
		tmp3D.z = tmp.at<float>(2, 0) / tmp.at<float>(3, 0);	//normalize z

		output.push_back(tmp3D);
	}

	ofstream fout("DSTtrans.xyz"); // Write File
	if (!fout)
	{
		cout << "無法寫入檔案\n";
		return 1;
	}
	for (unsigned int i = 0; i < output.size(); i++)
	{
		fout << output[i].x << " " << output[i].y << " " << output[i].z << endl;  //Write File
	}
	fout.close();  //Write File

	return 0;
}