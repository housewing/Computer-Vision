#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

class Point3D{
public:
	double x;
	double y;
	double z;
};

bool isInside(Mat&src, Mat& mat4x1, Mat& extrinsic, Mat& intrinsic)
{
	Mat mat3x1 = Mat(3, 1, CV_32F);
	mat3x1 = intrinsic * extrinsic * mat4x1; // K RT 

	// 取XYZ點 投影回2D平面
	double pointX = mat3x1.at<float>(0, 0);	
	double pointY = mat3x1.at<float>(1, 0);
	double pointZ = mat3x1.at<float>(2, 0);
	double pixelX = pointX / pointZ;
	double pixelY = pointY / pointZ;

	// 避免取出超過影像大小的像素
	int height = src.rows;
	int width = src.cols;
	if (pixelX < 0) { pixelX = 0; }
	else if (pixelX > width) { pixelX = width; }
	if (pixelY < 0) { pixelY = 0; }
	else if (pixelY > height) { pixelY = height; }

	// 取得項素值
	double intensity = src.at<uchar>(pixelY, pixelX);

	if (intensity != 0)
	{
		return true;
	}
}

int main(int argc, char** argv[])
{
	//從資料夾讀取image
	vector<Mat> ImgSet; 
	string path = "bunny/0";
	for (int pic = 1; pic <= 7; pic++)
	{
		stringstream picName;
		picName << path << pic << ".bmp";
		Mat src = imread(picName.str(), 0);
		threshold(src, src, 127, 255, 0);

		ImgSet.push_back(src);
	}
	cout << "ImgSet size " << ImgSet.size() << endl;
	imshow("01", ImgSet[0]);

	Mat Intrinsic = (Mat_<float>(3, 3) << 
		770.050769, 0.000000, 316.456402,
		0.000000, 770.204817, 244.266149,
		0.000000, 0.000000, 1.000000);
	Mat Extrinsic1 = (Mat_<float>(3, 4) << 
		0.999246, -0.038808, -0.001376, -90.395012,
		-0.038826, -0.997804, -0.053670, 53.949322,
		0.000710, 0.053683, -0.998558, 488.423401);
	Mat Extrinsic2 = (Mat_<float>(3, 4) << 
		0.999834, 0.017880, -0.003618, -75.462051,
		0.013915, -0.875766, -0.482535, 38.996746,
		-0.011796, 0.482404, -0.875869, 436.306244);
	Mat Extrinsic3 = (Mat_<float>(3, 4) << 
		0.500272, 0.865867, -0.001711, -100.514107,
		0.369591, -0.215325, -0.903901, 23.817722,
		-0.783027, 0.451564, -0.427738, 443.282562);
	Mat Extrinsic4 = (Mat_<float>(3, 4) << 
		-0.563910, 0.825830, -0.003181, 7.595384,
		0.662878, 0.450336, -0.598156, -105.127739,
		-0.492543, -0.339415, -0.801373, 578.296204);
	Mat Extrinsic5 = (Mat_<float>(3, 4) << 
		-0.999865, -0.016348, -0.001608, 88.027512,
		-0.005064, 0.399850, -0.916566, 17.813726,
		0.015627, -0.916435, -0.399879, 451.576294);
	Mat Extrinsic6 = (Mat_<float>(3, 4) << 
		-0.489544, -0.871971, -0.003669, 104.966927,
		-0.744480, 0.420150, -0.518867, 13.762801,
		0.453978, -0.251277, -0.854847, 459.715698);
	Mat Extrinsic7 = (Mat_<float>(3, 4) << 
		0.442191, -0.896919, -0.001636, 19.009016,
		-0.355062, -0.173374, -0.918625, 72.252525,
		0.823649, 0.406789, -0.395126, 330.565369);

	vector<Point3D> voxel;
	for (int i = 0; i < 200; i++)
	{
		for (int j = 0; j < 200; j++)
		{
			for (int k = 0; k < 200; k++)
			{
				float mat4[] = { i, j, k, 1 };
				Mat mat4x1 = Mat(4, 1, CV_32F, mat4);

				bool tmpE1 = isInside(ImgSet[0], mat4x1, Extrinsic1, Intrinsic); // 判斷是否在background
				bool tmpE2 = isInside(ImgSet[1], mat4x1, Extrinsic2, Intrinsic); // 判斷是否在background
				bool tmpE3 = isInside(ImgSet[2], mat4x1, Extrinsic3, Intrinsic); // 判斷是否在background
				bool tmpE4 = isInside(ImgSet[3], mat4x1, Extrinsic4, Intrinsic); // 判斷是否在background
				bool tmpE5 = isInside(ImgSet[4], mat4x1, Extrinsic5, Intrinsic); // 判斷是否在background
				bool tmpE6 = isInside(ImgSet[5], mat4x1, Extrinsic6, Intrinsic); // 判斷是否在background
				bool tmpE7 = isInside(ImgSet[6], mat4x1, Extrinsic7, Intrinsic); // 判斷是否在background
				if (tmpE1 == true && tmpE2 == true && tmpE3 == true && tmpE4 == true && tmpE5 == true && tmpE6 == true && tmpE7 == true) //將不是在background的voxel存起來
				{
					Point3D tmp;
					tmp.x = i;
					tmp.y = j;
					tmp.z = k;

					voxel.push_back(tmp);
				}
			}
		}
	}
	cout << "voxel size " << voxel.size() << endl;

	ofstream fout("bunny.xyz"); // Write File
	if (!fout)
	{
		cout << "無法寫入檔案\n";
		return 1;
	}
	for (unsigned int i = 0; i < voxel.size(); i++)
	{
		fout << voxel[i].x << " " << voxel[i].y << " " << voxel[i].z << endl;  //Write File
	}
	fout.close();  //Write File

	waitKey();
	return 0;
}