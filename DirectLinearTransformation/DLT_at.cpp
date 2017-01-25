#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv[])
{
	vector<Point> A;
	A.push_back(Point(42, 207));
	A.push_back(Point(548, 186));
	A.push_back(Point(548, 498));
	A.push_back(Point(42, 495));

	vector<Point> B;
	B.push_back(Point(578, 184));
	B.push_back(Point(1214, 104));
	B.push_back(Point(1226, 526));
	B.push_back(Point(578, 500));

	Mat mat8x9 = (Mat_<float>(8, 9) <<
		0, 0, 0, -1 * A[0].x, -1 * A[0].y, -1, B[0].y*A[0].x, B[0].y*A[0].y, B[0].y,
		A[0].x, A[0].y, 1, 0, 0, 0, -1 * B[0].x*A[0].x, -1 * B[0].x*A[0].y, -1 * B[0].x,
		0, 0, 0, -1 * A[1].x, -1 * A[1].y, -1, B[1].y*A[1].x, B[1].y*A[1].y, B[1].y,
		A[1].x, A[1].y, 1, 0, 0, 0, -1 * B[1].x*A[1].x, -1 * B[1].x*A[1].y, -1 * B[1].x,
		0, 0, 0, -1 * A[2].x, -1 * A[2].y, -1, B[2].y*A[2].x, B[2].y*A[2].y, B[2].y,
		A[2].x, A[2].y, 1, 0, 0, 0, -1 * B[2].x*A[2].x, -1 * B[2].x*A[2].y, -1 * B[2].x,
		0, 0, 0, -1 * A[3].x, -1 * A[3].y, -1, B[3].y*A[3].x, B[3].y*A[3].y, B[3].y,
		A[3].x, A[3].y, 1, 0, 0, 0, -1 * B[3].x*A[3].x, -1 * B[3].x*A[3].y, -1 * B[3].x);
	cout << mat8x9 << endl;
	cout << "-----------" << endl;

	cv::SVD svd(mat8x9, SVD::FULL_UV);
	Mat v = svd.vt.t();
	cout << "v size " << v.size() << ", col " << v.cols << ", row " << v.rows << endl;
	Mat tmp = v.col(8);
	cout << "lats column of v " << tmp << endl;

	Mat H = (Mat_<float>(3, 3) <<	tmp.at<float>(0, 0), tmp.at<float>(1, 0), tmp.at<float>(2, 0),
									tmp.at<float>(3, 0), tmp.at<float>(4, 0), tmp.at<float>(5, 0),
									tmp.at<float>(6, 0), tmp.at<float>(7, 0), tmp.at<float>(8, 0));
	cout << "H matrix " << H << endl;

	//Mat mat3x1 = (Mat_<float>(3, 1) << B[3].x, B[3].y, 1);
	//Mat tmp1 = H.inv() * mat3x1; // B = H.inv() * A
	//cout << tmp1.at<float>(0, 0) / tmp1.at<float>(2, 0) << " " << tmp1.at<float>(1, 0) / tmp1.at<float>(2, 0) << endl;


	//read Video
	VideoCapture cap("clip.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	vector<Mat> videoMat;
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		if (frame.empty())
			break;

		videoMat.push_back(frame);
	}
	cout << "videoMat size " << videoMat.size() << endl;

	Mat src = videoMat[0].clone();
	//imshow("clone", src);
	vector<Point> Apt;
	vector<Point> Bpt;
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (pointPolygonTest(Mat(B), Point(x, y), true) > 0)
			{
				Bpt.push_back(Point(x, y));
			}
			else if (pointPolygonTest(Mat(A), Point(x, y), true) > 0)
			{
				Apt.push_back(Point(x, y));
			}
		}
	}
	cout << "Bpt size " << Bpt.size() << ", Apt size " << Apt.size() << endl;


	vector<Mat> homographyMat;
	for (unsigned int j = 0; j < videoMat.size(); j++)
	{
		cout << j << " start" << endl;
		Mat copyMat = videoMat[j].clone();
		for (unsigned int i = 0; i < Bpt.size(); i++)
		{
			int tx = Bpt[i].x;
			int ty = Bpt[i].y;

			Mat mat3x1 = (Mat_<float>(3, 1) << tx, ty, 1);
			Mat tmp1 = H.inv() * mat3x1; // B = H.inv() * A

			int tmpX = tmp1.at<float>(0, 0) / tmp1.at<float>(2, 0);
			int tmpY = tmp1.at<float>(1, 0) / tmp1.at<float>(2, 0);

			copyMat.at<Vec3b>(ty, tx)[0] = videoMat[j].at<Vec3b>(tmpY, tmpX)[0];
			copyMat.at<Vec3b>(ty, tx)[1] = videoMat[j].at<Vec3b>(tmpY, tmpX)[1];
			copyMat.at<Vec3b>(ty, tx)[2] = videoMat[j].at<Vec3b>(tmpY, tmpX)[2];
		}

		for (unsigned int i = 0; i < Apt.size(); i++)
		{
			int tx = Apt[i].x;
			int ty = Apt[i].y;

			Mat mat3x1 = (Mat_<float>(3, 1) << tx, ty, 1);
			Mat tmp1 = H * mat3x1; // A = H * B

			int tmpX = tmp1.at<float>(0, 0) / tmp1.at<float>(2, 0);
			int tmpY = tmp1.at<float>(1, 0) / tmp1.at<float>(2, 0);

			copyMat.at<Vec3b>(ty, tx)[0] = videoMat[j].at<Vec3b>(tmpY, tmpX)[0];
			copyMat.at<Vec3b>(ty, tx)[1] = videoMat[j].at<Vec3b>(tmpY, tmpX)[1];
			copyMat.at<Vec3b>(ty, tx)[2] = videoMat[j].at<Vec3b>(tmpY, tmpX)[2];
		}
		cout << j << " end" << endl;
		homographyMat.push_back(copyMat);
	}
	cout << "homographyMat size " << homographyMat.size() << endl;


	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame Size = " << dWidth << "x" << dHeight << endl;
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter oVideoWriter("test.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, frameSize, true); //initialize the VideoWriter object 

	for (unsigned int i = 0; i < homographyMat.size(); i++)
	{
		if (homographyMat[i].empty())
			break;

		oVideoWriter.write(homographyMat[i]);
	}
	cout << "done" << endl;

	waitKey();
	return 0;
}