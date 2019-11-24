#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <vector>
#include <sys/io.h>
#include <fstream>
#include <map>
#include<ctime>
using namespace std;
using namespace cv;
string fileName,resultName;
////////////////
////  model part

struct ImageParas
{
	std::vector<double>	    tabletAreas;		// 连通域的面积
	std::vector<cv::Point>  centerPoints;	// 中心点位置
	std::vector<cv::Rect>   boxRects;		// 每个药片的区域
	ImageParas()
	{
		tabletAreas.clear();
		centerPoints.clear();
		boxRects.clear();
	};
};

struct ModelInfo
{
	bool        status;         // 模板是否正确
	string		modelName;		// 模板名称
	int			tabletNum;		// 药片总数
	int			bigColNum;		// 行数（一行几板药）
	int			bigRowNum;		// 列数（一列几板药）
	int			smallColNum;	// 子行数
	int			smallRowNum;	// 子列数
	double		areaErrorRate;  // 药片区域的面积容错率
	bool        single;
	cv::Point	            startPoint;		// 截取开始位置
	cv::Point	            endPoint;		// 截取结束位置
	ImageParas  paras;          // 模板图片上的参数
	ModelInfo() :
		tabletNum(0),
		bigColNum(0),
		bigRowNum(0),
		smallColNum(0),
		smallRowNum(0),
		areaErrorRate(0.0),
		endPoint(0, 0),
		startPoint(0, 0),
		single(false)
	{
		paras = ImageParas();
	};
    /*
	static void copyParas(ModelInfo &paras, ModelInfo copyParas)
	{
		paras.areaErrorRate = copyParas.areaErrorRate;
		paras.bigColNum = copyParas.bigColNum;
		paras.bigRowNum = copyParas.bigRowNum;
		paras.modelName = copyParas.modelName;
		paras.single = copyParas.single;
		paras.smallColNum = copyParas.smallColNum;
		paras.smallRowNum = copyParas.smallRowNum;
		paras.tabletNum = copyParas.tabletNum;
		paras.startPoint = copyParas.startPoint;
		paras.endPoint = copyParas.endPoint;
		paras.paras.boxRects.clear();
		paras.paras.centerPoints.clear();
		paras.paras.tabletAreas.clear();
		for (auto value : copyParas.paras.boxRects)
		{
			paras.paras.boxRects.push_back(value);
		}
		for (auto value : copyParas.paras.centerPoints)
		{
			paras.paras.centerPoints.push_back(value);
		}
		for (auto value : copyParas.paras.tabletAreas)
		{
			paras.paras.tabletAreas.push_back(value);
		}
	}
    */
};

struct ModelResult
{
	int status;		// 错误码 0-模板药片数量不对，1-模板正确
	ModelInfo  paras;   // 模型参数
};

Mat preProcess(Mat input, Rect rect)
{
	//直接二值化效果有一点点缺陷
	Mat srcImage;
	if (rect == Rect(0, 0, 0, 0)) {
		rect.y = (int)floor(input.rows / 5);
		rect.x = (int)floor(input.cols * 1 / 12);
		rect.height = (int)floor(input.rows * 3 / 5);
		rect.width = (int)floor(input.cols * 10 / 12);
	}
	input(rect).copyTo(srcImage);
	
	Mat patch = Mat::zeros(input.rows, input.cols, CV_8UC1);
	double thresh = threshold(srcImage, patch, 0.0, 255.0, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);   // 168
	/*imshow("BWCC", patch);
	waitKey(); */
	return patch;
	


	// 二值化

	for (int i = 0; i < srcImage.rows; i++)
		for (int j = 0; j < srcImage.cols; j++)
			if (srcImage.at<uchar>(i, j) < (thresh + 0.2 * 255)) {
				if (Rect(0, 0, 0, 0) == rect) {
					patch.at<uchar>(i + (int)floor(input.rows / 5), j + (int)floor(input.cols / 12)) = 255;
				}
				else {
					patch.at<uchar>(i + rect.y, j + rect.x) = 255;
				}
			}
	return patch;
}

void regionPropsClq(Mat BW, vector<double>& area, vector<Rect>& bbox, vector<Point>& center)
{
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(BW, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<double> area0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i]);
		area0.push_back(a);
	}
	double maxArea = *max_element(area0.begin(), area0.end());

	for (size_t i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i]);
		if (a > maxArea * 0.2)
		{
			Rect r = boundingRect(contours[i]);
			area.push_back(a);
			bbox.push_back(r);

			Point cent = Point(0, 0);
			cent.x = r.x + int(r.width / 2);
			cent.y = r.y + int(r.height / 2);
			center.push_back(cent);
		}
	}
}

ModelResult SetupModel(cv::Mat image, cv::Rect rect, int tabletNum)
{
	ModelResult result;		// 检测结果
	Mat grayModel;

	cvtColor(image, grayModel, CV_BGR2GRAY);
	Mat modelpic = preProcess(grayModel, rect);


	vector<double> modelArea;
	vector<Rect> modelBox;
	vector<Point> modelCenter;

	regionPropsClq(modelpic, modelArea, modelBox, modelCenter);

	if (modelCenter.size() == tabletNum) {
		result.paras.status = true;
		result.paras.paras.boxRects = modelBox;
		result.paras.paras.centerPoints = modelCenter;
		result.paras.paras.tabletAreas = modelArea;
		result.paras.tabletNum = tabletNum;
	}
	else {
		result.paras.status = false;  // 模板药片数量不对
	}
	return result;
}

////////////////
////   test part

int calcuWrongIndex(int x, int c, int cols)
{
	int per_cols = cols / c;
	for (int i = 0; i < c; i++)
	{
		if (x > i * per_cols && x < (i + 1) * per_cols)
			return i;
	}
	return -1;
}

map<int, int> checkImage(ModelInfo info, Mat image, Rect rect)
{
	map<int, int> wrongIndex;

	if (!info.status || !image.data)
	{
        if (!info.status)
        {
            fprintf(stderr, "***6***");
        }
        if (!image.data)
        {
            fprintf(stderr, "***7***");
        }
        fprintf(stderr, "***22***");
		return wrongIndex;
	}

	Mat grayDect;
	cvtColor(image, grayDect, CV_BGR2GRAY);
    fprintf(stderr, "***23***");

	vector<Point> finalDetectCenter;

	Mat patch = preProcess(grayDect, rect);
    fprintf(stderr, "***24***");

	//【2】连通域操作
	vector<double> area, area2;
	vector<Rect> bbox, bbox2;
	vector<Point> center, center2;
	regionPropsClq(patch, area, bbox, center);
    fprintf(stderr, "***25***");

	//【3】对每一个连通域进行分析,检测缺损和多粒，然后画框
	double minArea = *min_element(info.paras.tabletAreas.begin(), info.paras.tabletAreas.end());
	double maxArea = *max_element(info.paras.tabletAreas.begin(), info.paras.tabletAreas.end());

	double s1 = minArea * 0.05;
	double s2 = minArea * (1 - 0.05);
	double s3 = maxArea * (1 + 0.05);
    fprintf(stderr, "***26***");

	int num = 0;
	int bbox_w = info.paras.boxRects[0].width;   //模板里第一颗药的外接矩宽和高
	int bbox_h = info.paras.boxRects[0].height;
	int side = 10;
    fprintf(stderr, "***8***");

	// 面积检测
	Rect r;
	for (size_t i = 0; i < area.size(); i++)
	{
		if ((area[i] > s1 && area[i] < s2) || area[i] > s3)
		{
			num++;
			r.x = (center[i].x + (int)floor(image.cols * 1 / 12)) - bbox_w / 2 - side;
			r.y = (center[i].y + (int)floor(image.rows / 5)) - bbox_h / 2 - side;
			r.width = bbox_w + 2 * side;
			r.height = bbox_h + 2 * side;
			finalDetectCenter.push_back(center[i]);
			rectangle(image, r, Scalar(0, 0, 255), 2);
			/*imwrite("BWCCA.jpg", image);
            fprintf(stderr, "***5***");
			waitKey();*/
			//			putText(result, to_string(finalDetectCenter.size()), Point(r.x, r.y), 32, 1, Scalar(0, 0, 255));
		}
		else    //存下可能正常的药片，用于计算下一步的双粒
		{
			r.x = (center[i].x + (int)floor(image.cols * 1 / 12)) - bbox_w / 2 - side;
			r.y = (center[i].y + (int)floor(image.rows / 5)) - bbox_h / 2 - side;
			r.width = bbox_w + 2 * side;
			r.height = bbox_h + 2 * side;
			finalDetectCenter.push_back(center[i]);
			rectangle(image, r, Scalar(0, 255, 255), 2);
			/*imwrite("BWCCA.jpg", image);
            fprintf(stderr, "***5***");
			waitKey();*/
			area2.push_back(area[i]);
			bbox2.push_back(bbox[i]);
			center2.push_back(center[i]);
		}
	}
    fprintf(stderr, "***9***");
	//【4】新增双粒检测——针对小药片，一个坑里面能放两粒，而且两粒没有黏在一起的情形
	//CapsuleTest::doubelDect(result, area2, center2, r, side);
	for (int i = 0; i < area2.size(); i++) {
		for (int j = i + 1; j < area2.size(); j++) {
			if ((area2[i] > s2 && area2[j] > s2))
			{
				double d2 = pow(center2[i].x - center2[j].x, 2) + pow(center2[i].y - center2[j].y, 2);   //距离的平方
				if (d2 < pow(bbox_h * 3 / 2, 2))
				{
					r.x = (center2[i].x + center2[j].x + (int)floor(image.cols * 1 / 12)) / 2 - 4 * side;
					r.y = (center2[i].y + center2[j].y + (int)floor(image.rows / 5)) / 2 - 4 * side;
					r.width = bbox_w + 2 * side;
					r.height = bbox_h + 2 * side;
					finalDetectCenter.push_back(center2[i]);
					rectangle(image, r, Scalar(0, 255, 0), 2);
					/*imwrite("BWCCB.jpg", image);
					waitKey();*/
				}
			}
		}
	}
    fprintf(stderr, "***10***");

	//【5】与目标进行匹配，检测缺粒
	vector< vector<double> > distance(info.paras.centerPoints.size(), vector<double>(center.size(), 0));  //创建二维动态数组
	for (size_t i = 0; i < info.paras.centerPoints.size(); i++)
	{
		for (size_t j = 0; j < center.size(); j++)
		{
			distance[i][j] = pow(info.paras.centerPoints[i].x - center[j].x, 2) + pow(info.paras.centerPoints[i].y - center[j].y, 2);   //距离的平方
		}
	}

	for (size_t i = 0; i < info.paras.centerPoints.size(); i++)
	{
		// auto minDist = Config::getInstance()->minDistance;
		double minValue = *min_element(distance[i].begin(), distance[i].end());   //每一行的最小值
		if (minValue > pow(bbox_h * 1.5, 2))//minDist)//))
		{
			num++;
			r.y = (info.paras.centerPoints[i].y + (int)floor(image.rows / 5)) - bbox_h / 2 - side;
			r.x = (info.paras.centerPoints[i].x + (int)floor(image.cols * 1 / 12)) - bbox_w / 2 - side;
			r.width = bbox_w + 2 * side;
			r.height = bbox_h + 2 * side;
			finalDetectCenter.push_back(info.paras.centerPoints[i]);
			rectangle(image, r, Scalar(255, 0, 0), 2);
			/*imwrite("BWCCC.jpg", image);
			waitKey();*/
		}
	}
    fprintf(stderr, "***11***");
	/* 
	for (int i = 0; i < info.bigColNum; ++i)
	{
		wrongIndex[i] = 0;
	}
	for (size_t i = 0; i < finalDetectCenter.size(); i++)
	{
		int index = calcuWrongIndex(finalDetectCenter[i].x, info.bigColNum, patch.cols);
		if (index != -1)
		{
			wrongIndex[index] = wrongIndex[index] + 1;
		}
	}
	for (int i = 0; i < info.bigColNum; ++i)
	{
		wrongIndex[i] = wrongIndex[i] > 0 ? 1 : 0;
	}

	*/
    imwrite(resultName, image);
	return wrongIndex;
}

std::map<int, int> tabletDect(Mat& image, ModelInfo info, Rect rect/* = Rect(0, 0, 0, 0)*/)
{
	Mat grayDect;
	cvtColor(image, grayDect, CV_BGR2GRAY);

	std::map<int, int> wrongIndex;

	vector<Point> finalDetectCenter;

	Mat patch = preProcess(grayDect, rect);
	//imwrite("test.bmp", patch);
	//【2】连通域操作
	vector<double> area, area2;
	vector<Rect> bbox, bbox2;
	vector<Point> center, center2;
	regionPropsClq(patch, area, bbox, center);

	//【3】对每一个连通域进行分析,检测缺损和多粒，然后画框
	double minArea = *min_element(info.paras.tabletAreas.begin(), info.paras.tabletAreas.end());
	double maxArea = *max_element(info.paras.tabletAreas.begin(), info.paras.tabletAreas.end());

	double s1 = minArea * 0.05;
	double s2 = minArea * (1 - 0.05);
	double s3 = maxArea * (1 + 0.05);

	int num = 0;
	int bbox_w = info.paras.boxRects[0].width;   //模板里第一颗药的外接矩宽和高
	int bbox_h = info.paras.boxRects[0].height;
	int side = 10;

	// 面积检测
	Rect r;
	for (size_t i = 0; i < area.size(); i++)
	{
		if ((area[i] > s1 && area[i] < s2) || area[i] > s3)
		{
			num++;
			r.x = center[i].x - bbox_w / 2 - side;
			r.y = center[i].y - bbox_h / 2 - side;
			r.width = bbox_w + 2 * side;
			r.height = bbox_h + 2 * side;
			finalDetectCenter.push_back(center[i]);
			rectangle(image, r, Scalar(0, 0, 255), 2);
			//			putText(result, to_string(finalDetectCenter.size()), Point(r.x, r.y), 32, 1, Scalar(0, 0, 255));
		}
		else    //存下可能正常的药片，用于计算下一步的双粒
		{
			area2.push_back(area[i]);
			bbox2.push_back(bbox[i]);
			center2.push_back(center[i]);
		}
	}
	//【4】新增双粒检测——针对小药片，一个坑里面能放两粒，而且两粒没有黏在一起的情形
	//CapsuleTest::doubelDect(result, area2, center2, r, side);
	for (int i = 0; i < area2.size(); i++) {
		for (int j = i + 1; j < area2.size(); j++) {
			if ((area2[i] > s2 && area2[j] > s2))
			{
				double d2 = pow(center2[i].x - center2[j].x, 2) + pow(center2[i].y - center2[j].y, 2);   //距离的平方
				if (d2 < pow(bbox_h * 3 / 2, 2))
				{
					r.x = (center2[i].x + center2[j].x) / 2 - 4 * side;
					r.y = (center2[i].y + center2[j].y) / 2 - 4 * side;
					r.width = bbox_w + 2 * side;
					r.height = bbox_h + 2 * side;
					finalDetectCenter.push_back(center2[i]);
					rectangle(image, r, Scalar(0, 255, 0), 2);
				}
			}
		}
	}

	//【5】与目标进行匹配，检测缺粒
	vector< vector<double> > distance(info.paras.centerPoints.size(), vector<double>(center.size(), 0));  //创建二维动态数组
	for (size_t i = 0; i < info.paras.centerPoints.size(); i++)
	{
		for (size_t j = 0; j < center.size(); j++)
		{
			distance[i][j] = pow(info.paras.centerPoints[i].x - center[j].x, 2) + pow(info.paras.centerPoints[i].y - center[j].y, 2);   //距离的平方
		}
	}

	for (size_t i = 0; i < info.paras.centerPoints.size(); i++)
	{
		// auto minDist = Config::getInstance()->minDistance;
		double minValue = *min_element(distance[i].begin(), distance[i].end());   //每一行的最小值
		if (minValue > pow(bbox_h * 1.5, 2))//minDist)//))
		{
			num++;
			r.y = info.paras.centerPoints[i].y - bbox_h / 2 - side;
			r.x = info.paras.centerPoints[i].x - bbox_w / 2 - side;
			r.width = bbox_w + 2 * side;
			r.height = bbox_h + 2 * side;
			finalDetectCenter.push_back(info.paras.centerPoints[i]);
			rectangle(image, r, Scalar(255, 0, 0), 2);
		}
	}
	for (int i = 0; i < info.bigColNum; ++i)
	{
		wrongIndex[i] = 0;
	}
	for (size_t i = 0; i < finalDetectCenter.size(); i++)
	{
		int index = calcuWrongIndex(finalDetectCenter[i].x, info.bigColNum, patch.cols);
		if (index != -1)
		{
			wrongIndex[index] = wrongIndex[index] + 1;
		}
	}
	for (int i = 0; i < info.bigColNum; ++i)
	{
		wrongIndex[i] = wrongIndex[i] > 0 ? 1 : 0;
	}
	return wrongIndex;
}


int main()
{
	cv::Mat image = cv::imread("../pic1/107.bmp");
	cv::Rect rect = Rect(0, 0, 0, 0);
	int tabletNum = 30;
	// cv::imshow("Input", image);
	ModelResult model = SetupModel(image, rect, tabletNum);
    
    clock_t start = clock();
    
    for (int i = 0; i <= 10; i++)
	{
        stringstream ss;
        ss<<i; 
        string s1 = ss.str();
        cout<<s1<<endl; // 30
        fileName = "test(" + s1 + ").bmp";
        resultName = "result(" + s1 + ").jpg";
        
        cv::Mat image2 = cv::imread("../pic2/"+fileName);
        fprintf(stderr, "***2***");

        map<int, int> checkResult;
        fprintf(stderr, "***21***");
        checkResult = checkImage(model.paras, image2, rect);
        fprintf(stderr, "***3***");
    }
    clock_t end = clock();
    cout << "take" << (double)(end - start)*1000 / CLOCKS_PER_SEC << "ms" << endl;
    
    
	return 0;
}