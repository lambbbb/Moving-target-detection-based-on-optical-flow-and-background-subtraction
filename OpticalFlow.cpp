#include<opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

//光流法变量定义
const int MAX_CORNERS = 100;
static int i = 0;//用于遍历视频帧
static char imgName[100];//用于图片文件名称
static char preimgName[100];//用于图片文件名称
CvPoint point0[100];
CvPoint point1[100];
double length[100];

//背景减除法使用的内存
CvCapture* pCapture = NULL;
IplImage *imgFR = NULL;//imgFR为前景，imgBK为背景，imgCR为当前帧
IplImage *imgBK = NULL;
IplImage *imgCR = NULL;
CvMat* matFR = NULL;
CvMat* matBK = NULL;
CvMat* matCR = NULL;

//用于保存结果
static char resultName[100];//用于图片文件名称
static char videoName[100];//用于图片文件名称


//光流法-背景减除法法处理图像
void opticalBS()
{
	//将光流法与背景减除法结果结合使用的内存
	CvMemStorage *storage = cvCreateMemStorage(0);
	vector<vector<Point>> contours;//轮廓存储二维数组
	vector<Vec4i> hierarchy;//轮廓结构信息
	double area = 0;
	double avg_ptx = 0;
	double avg_pty = 0;//计算轮廓中心点坐标
	double distance = 0;//存储轮廓点与运动角点距离
	double distancestandard = 760;//轮廓点与运动角点距离阈值


	//光流法使用的内存
	double xtotal = 0;//求所有交点横坐标之和
	double ytotal = 0;//求所有交点纵坐标之和

	double lengthstandard = 2;//光流长度阈值，大于该阈值，则认为该像素点在运动
	double lengthsum = 0;//记录所有超过阈值的光流向量的长度和；
	double num = 0;//记录角点的个数
	char imgAName[100], imgBName[100];//用于导入图像
	int win_size = 10;//窗口尺寸
	int corner_count = MAX_CORNERS;//角点个数
	CvPoint2D32f *cornersA = new CvPoint2D32f[MAX_CORNERS];//光流向量起点
	CvPoint2D32f *cornersB = new CvPoint2D32f[MAX_CORNERS];//光流向量终点
	char features_found[MAX_CORNERS];
	float features_errors[MAX_CORNERS];//存放光流法处理结果
	CvPoint move[MAX_CORNERS];//存储光流法检测得到的运动角点
	int movenum = 0;//记录运动角点个数

	i++;

	if (i == 1)//若为第一帧图像，对背景减除法的前景、背景进行初始化
	{
		imgFR = cvCreateImage(cvSize(imgCR->width, imgCR->height), IPL_DEPTH_8U, 1);//初始化前景图像
		imgBK = cvCreateImage(cvSize(imgCR->width, imgCR->height), IPL_DEPTH_8U, 1);;//初始化背景图像
		matFR = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		matBK = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		matCR = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		//仅当mat为IIplmage*类型，且其depth为IPL_DEPTH_8U（8bit无符号整形）时，有较好的显示效果
		//对于其他深度的IplImage*或者CvMat*类数据，可以使用CvConvert函数进行转换
		cvCvtColor(imgCR, imgBK, CV_BGR2GRAY);
		cvCvtColor(imgCR, imgFR, CV_BGR2GRAY);
		cvConvert(imgFR, matCR);
		cvConvert(imgFR, matFR);
		cvConvert(imgFR, matBK);
	}
	else
	{
		//背景减除法
		//颜色空间转换
		cvCvtColor(imgCR, imgFR, CV_BGR2GRAY);
		cvConvert(imgFR, matCR);
		//高斯滤波
		cvSmooth(matCR, matCR, CV_MEDIAN, 3, 0, 0);
		//当前帧与背景图做差分，求得运动部分，结果存入前景
		cvAbsDiff(matCR, matBK, matFR);
		//二值化前景,阈值设为30，阈值越小，检测越敏感
		cvThreshold(matFR, imgFR, 30, 255.0, CV_THRESH_BINARY);
		//将当前帧按照0.003的权重累加到背景中，从而更新背景
		cvRunningAvg(matCR, matBK, 0.003, 0);
		//将背景转化为图像格式，用以显示
		cvConvert(matBK, imgBK);



		//光流法
		//读入当前帧与上一帧图像
		sprintf(imgAName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i - 1, ".jpg");
		sprintf(imgBName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i, ".jpg");
		IplImage *imgA = cvLoadImage(imgAName, CV_LOAD_IMAGE_GRAYSCALE);//导入上一帧图像
		IplImage *imgB = cvLoadImage(imgBName, CV_LOAD_IMAGE_GRAYSCALE);//导入当前帧图像
		CvSize img_sz = cvGetSize(imgA);
		IplImage *eig_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
		IplImage *tmp_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);

		cvGoodFeaturesToTrack(
			imgA,
			eig_image,
			tmp_image,
			cornersA,
			&corner_count,//角点数
			0.3,//角点质量
			70.0,//角点间最小距离
			0,
			3,
			0,
			0.04
		);//寻找角点

		cvFindCornerSubPix(
			imgA,
			cornersA,
			corner_count,
			cvSize(win_size, win_size),
			cvSize(-1, -1),
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03)
		);//精确角点位置

		CvSize pyr_sz = cvSize(imgA->width + 8, imgB->height / 3);
		IplImage *pyrA = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
		IplImage *pyrB = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
		cvCalcOpticalFlowPyrLK(
			imgA,
			imgB,
			pyrA,
			pyrB,
			cornersA,
			cornersB,
			corner_count,
			cvSize(win_size, win_size),
			5,
			features_found,
			features_errors,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),
			0
		);//光流法处理

		for (int r = 0; r < corner_count; ++r)
		{
			if (features_found[r] == 0 || features_errors[r] > 550)
			{
				cout << "Error is " << features_errors[r];
				continue;
			}
			cout << "Got it!" << endl;

			point0[r] = cvPoint(cvRound(cornersA[r].x), cvRound(cornersA[r].y));
			point1[r] = cvPoint(cvRound(cornersB[r].x), cvRound(cornersB[r].y));
			length[r] = sqrt((cornersA[r].x - cornersB[r].x)*(cornersA[r].x - cornersB[r].x) + (cornersA[r].y - cornersB[r].y)*(cornersA[r].y - cornersB[r].y));//计算光流长度
			lengthsum += length[r];
			num++;
		}

		lengthstandard = 0.5;

		for (int index = 0; index < num; ++index)
		{
			if (length[index] > lengthstandard)
			{
				cvCircle(imgCR, point1[index], 6, (0, 0, 255), -1);
				cvLine(imgCR, point0[index], point1[index], CV_RGB(255, 0, 0), 4);//画出光流向量
				movenum++;
				move[movenum] = point1[index];
			}
		}

		num = 0;
		lengthsum = 0;


		//检测光流法得到的运动目标坐标是否在背景减除法前景中运动目标区域
		//找到前景中的轮廓
		IplImage *result = NULL;
		Mat FR = cvarrToMat(imgFR);
		findContours(FR, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
		for (int h = 0; h < contours.size(); h++)//遍历所有轮廓
		{
			if (contours[h].size() > 5)
			{
				for (int j = 0; j < (int)contours[h].size(); j++)//遍历一个轮廓中的所有点
				{
					CvPoint pt = cvPoint(contours[h][j].x, contours[h][j].y);//存储轮廓中的点
					avg_ptx += pt.x;
					avg_pty += pt.y;
				}
				avg_ptx = avg_ptx / (int)contours[h].size();
				avg_pty = avg_pty / (int)contours[h].size();
				for (int t = 0; t < movenum; t++)
				{
					distance = sqrt((avg_ptx - move[t].x)*(avg_ptx - move[t].x) + (avg_pty - move[t].y)*(avg_pty - move[t].y));//计算轮廓点与运动角点的距离
					if (distance > distancestandard)//距离过大，认为轮廓有问题，即背景减除法得到的运动区域有问题，将该区域涂黑
					{
						drawContours(FR, contours, h, CV_RGB(0, 253, 0), CV_FILLED);
						result = &IplImage(FR);
					}
				}
				avg_ptx = 0;
				avg_pty = 0;
			}
		}
		for (int o = 0; o < MAX_CORNERS; o++)
		{
			move[o] = NULL;
		}

		//反转图片并显示
		cvFlip(imgCR, imgCR, 0);
		cvShowImage("result", imgFR);
		Mat resultmat = cvarrToMat(imgFR);
		sprintf(resultName, "%s%d%s", "Y:\\GraduationProject\\result\\img", i, ".jpg");
		imwrite(resultName, resultmat);

		cvFlip(imgCR, imgCR, 0);
		cvShowImage("video", imgCR);

		Mat videomat = cvarrToMat(imgCR);
		sprintf(videoName, "%s%d%s", "Y:\\GraduationProject\\video\\img", i, ".jpg");
		imwrite(videoName, videomat);
		cvWaitKey(2);
	}
}


int main()
{
	//使用摄像头用此句
	//VideoCapture capture0(0);

	//使用视频用此句
	VideoCapture capture0("Y:\\GraduationProject\\Project\\PETS2006.avi");

	//创建窗口
	cvNamedWindow("video", 1);
	cvNamedWindow("result", 1);
	//使窗口有序排列
	cvMoveWindow("video", 10, 0);
	cvMoveWindow("result", 710, 0);
	cv::Mat frame;
	pCapture = cvCaptureFromAVI("Y:\\GraduationProject\\Project\\PETS2006.avi");

	while (true)
	{
		//读取当前帧
		capture0 >> frame;

		if (frame.empty()) break;

		//保存当前帧并处理g
		//保存当前帧
		sprintf(imgName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i + 1, ".jpg");
		imwrite(imgName, frame);
		imgCR = cvQueryFrame(pCapture);

		//对当前帧处理
		opticalBS();

		//删除上上一帧
		if (i > 2)
		{
			sprintf(preimgName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i - 2, ".jpg");
			remove(preimgName);
		}
	}

	capture0.release();
	destroyAllWindows();

}