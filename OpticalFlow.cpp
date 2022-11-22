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

//��������������
const int MAX_CORNERS = 100;
static int i = 0;//���ڱ�����Ƶ֡
static char imgName[100];//����ͼƬ�ļ�����
static char preimgName[100];//����ͼƬ�ļ�����
CvPoint point0[100];
CvPoint point1[100];
double length[100];

//����������ʹ�õ��ڴ�
CvCapture* pCapture = NULL;
IplImage *imgFR = NULL;//imgFRΪǰ����imgBKΪ������imgCRΪ��ǰ֡
IplImage *imgBK = NULL;
IplImage *imgCR = NULL;
CvMat* matFR = NULL;
CvMat* matBK = NULL;
CvMat* matCR = NULL;

//���ڱ�����
static char resultName[100];//����ͼƬ�ļ�����
static char videoName[100];//����ͼƬ�ļ�����


//������-����������������ͼ��
void opticalBS()
{
	//���������뱳��������������ʹ�õ��ڴ�
	CvMemStorage *storage = cvCreateMemStorage(0);
	vector<vector<Point>> contours;//�����洢��ά����
	vector<Vec4i> hierarchy;//�����ṹ��Ϣ
	double area = 0;
	double avg_ptx = 0;
	double avg_pty = 0;//�����������ĵ�����
	double distance = 0;//�洢���������˶��ǵ����
	double distancestandard = 760;//���������˶��ǵ������ֵ


	//������ʹ�õ��ڴ�
	double xtotal = 0;//�����н��������֮��
	double ytotal = 0;//�����н���������֮��

	double lengthstandard = 2;//����������ֵ�����ڸ���ֵ������Ϊ�����ص����˶�
	double lengthsum = 0;//��¼���г�����ֵ�Ĺ��������ĳ��Ⱥͣ�
	double num = 0;//��¼�ǵ�ĸ���
	char imgAName[100], imgBName[100];//���ڵ���ͼ��
	int win_size = 10;//���ڳߴ�
	int corner_count = MAX_CORNERS;//�ǵ����
	CvPoint2D32f *cornersA = new CvPoint2D32f[MAX_CORNERS];//�����������
	CvPoint2D32f *cornersB = new CvPoint2D32f[MAX_CORNERS];//���������յ�
	char features_found[MAX_CORNERS];
	float features_errors[MAX_CORNERS];//��Ź�����������
	CvPoint move[MAX_CORNERS];//�洢���������õ����˶��ǵ�
	int movenum = 0;//��¼�˶��ǵ����

	i++;

	if (i == 1)//��Ϊ��һ֡ͼ�񣬶Ա�����������ǰ�����������г�ʼ��
	{
		imgFR = cvCreateImage(cvSize(imgCR->width, imgCR->height), IPL_DEPTH_8U, 1);//��ʼ��ǰ��ͼ��
		imgBK = cvCreateImage(cvSize(imgCR->width, imgCR->height), IPL_DEPTH_8U, 1);;//��ʼ������ͼ��
		matFR = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		matBK = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		matCR = cvCreateMat(imgCR->height, imgCR->width, CV_32FC1);
		//����matΪIIplmage*���ͣ�����depthΪIPL_DEPTH_8U��8bit�޷������Σ�ʱ���нϺõ���ʾЧ��
		//����������ȵ�IplImage*����CvMat*�����ݣ�����ʹ��CvConvert��������ת��
		cvCvtColor(imgCR, imgBK, CV_BGR2GRAY);
		cvCvtColor(imgCR, imgFR, CV_BGR2GRAY);
		cvConvert(imgFR, matCR);
		cvConvert(imgFR, matFR);
		cvConvert(imgFR, matBK);
	}
	else
	{
		//����������
		//��ɫ�ռ�ת��
		cvCvtColor(imgCR, imgFR, CV_BGR2GRAY);
		cvConvert(imgFR, matCR);
		//��˹�˲�
		cvSmooth(matCR, matCR, CV_MEDIAN, 3, 0, 0);
		//��ǰ֡�뱳��ͼ����֣�����˶����֣��������ǰ��
		cvAbsDiff(matCR, matBK, matFR);
		//��ֵ��ǰ��,��ֵ��Ϊ30����ֵԽС�����Խ����
		cvThreshold(matFR, imgFR, 30, 255.0, CV_THRESH_BINARY);
		//����ǰ֡����0.003��Ȩ���ۼӵ������У��Ӷ����±���
		cvRunningAvg(matCR, matBK, 0.003, 0);
		//������ת��Ϊͼ���ʽ��������ʾ
		cvConvert(matBK, imgBK);



		//������
		//���뵱ǰ֡����һ֡ͼ��
		sprintf(imgAName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i - 1, ".jpg");
		sprintf(imgBName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i, ".jpg");
		IplImage *imgA = cvLoadImage(imgAName, CV_LOAD_IMAGE_GRAYSCALE);//������һ֡ͼ��
		IplImage *imgB = cvLoadImage(imgBName, CV_LOAD_IMAGE_GRAYSCALE);//���뵱ǰ֡ͼ��
		CvSize img_sz = cvGetSize(imgA);
		IplImage *eig_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
		IplImage *tmp_image = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);

		cvGoodFeaturesToTrack(
			imgA,
			eig_image,
			tmp_image,
			cornersA,
			&corner_count,//�ǵ���
			0.3,//�ǵ�����
			70.0,//�ǵ����С����
			0,
			3,
			0,
			0.04
		);//Ѱ�ҽǵ�

		cvFindCornerSubPix(
			imgA,
			cornersA,
			corner_count,
			cvSize(win_size, win_size),
			cvSize(-1, -1),
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03)
		);//��ȷ�ǵ�λ��

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
		);//����������

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
			length[r] = sqrt((cornersA[r].x - cornersB[r].x)*(cornersA[r].x - cornersB[r].x) + (cornersA[r].y - cornersB[r].y)*(cornersA[r].y - cornersB[r].y));//�����������
			lengthsum += length[r];
			num++;
		}

		lengthstandard = 0.5;

		for (int index = 0; index < num; ++index)
		{
			if (length[index] > lengthstandard)
			{
				cvCircle(imgCR, point1[index], 6, (0, 0, 255), -1);
				cvLine(imgCR, point0[index], point1[index], CV_RGB(255, 0, 0), 4);//������������
				movenum++;
				move[movenum] = point1[index];
			}
		}

		num = 0;
		lengthsum = 0;


		//���������õ����˶�Ŀ�������Ƿ��ڱ���������ǰ�����˶�Ŀ������
		//�ҵ�ǰ���е�����
		IplImage *result = NULL;
		Mat FR = cvarrToMat(imgFR);
		findContours(FR, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
		for (int h = 0; h < contours.size(); h++)//������������
		{
			if (contours[h].size() > 5)
			{
				for (int j = 0; j < (int)contours[h].size(); j++)//����һ�������е����е�
				{
					CvPoint pt = cvPoint(contours[h][j].x, contours[h][j].y);//�洢�����еĵ�
					avg_ptx += pt.x;
					avg_pty += pt.y;
				}
				avg_ptx = avg_ptx / (int)contours[h].size();
				avg_pty = avg_pty / (int)contours[h].size();
				for (int t = 0; t < movenum; t++)
				{
					distance = sqrt((avg_ptx - move[t].x)*(avg_ptx - move[t].x) + (avg_pty - move[t].y)*(avg_pty - move[t].y));//�������������˶��ǵ�ľ���
					if (distance > distancestandard)//���������Ϊ���������⣬�������������õ����˶����������⣬��������Ϳ��
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

		//��תͼƬ����ʾ
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
	//ʹ������ͷ�ô˾�
	//VideoCapture capture0(0);

	//ʹ����Ƶ�ô˾�
	VideoCapture capture0("Y:\\GraduationProject\\Project\\PETS2006.avi");

	//��������
	cvNamedWindow("video", 1);
	cvNamedWindow("result", 1);
	//ʹ������������
	cvMoveWindow("video", 10, 0);
	cvMoveWindow("result", 710, 0);
	cv::Mat frame;
	pCapture = cvCaptureFromAVI("Y:\\GraduationProject\\Project\\PETS2006.avi");

	while (true)
	{
		//��ȡ��ǰ֡
		capture0 >> frame;

		if (frame.empty()) break;

		//���浱ǰ֡������g
		//���浱ǰ֡
		sprintf(imgName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i + 1, ".jpg");
		imwrite(imgName, frame);
		imgCR = cvQueryFrame(pCapture);

		//�Ե�ǰ֡����
		opticalBS();

		//ɾ������һ֡
		if (i > 2)
		{
			sprintf(preimgName, "%s%d%s", "Y:\\GraduationProject\\images\\img", i - 2, ".jpg");
			remove(preimgName);
		}
	}

	capture0.release();
	destroyAllWindows();

}