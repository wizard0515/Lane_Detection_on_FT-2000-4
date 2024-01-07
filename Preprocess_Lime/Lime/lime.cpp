#include <opencv2/core/core.hpp> // OpenCV 核心模块，包含了基本的数据结构和算法
#include <opencv2/highgui/highgui.hpp> // OpenCV GUI 模块，包含了图像和视频的 I/O 函数
#include <opencv2/imgproc/imgproc.hpp> // OpenCV 图像处理模块，包含了图像处理函数
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <iostream> // 输入输出流库
#include <cstring>
#include <math.h>
#include <complex.h>
#include <vector>
#include <cmath>

using namespace std;

namespace LIME 
{

    class lime 
    {
        public: 
        cv::Mat img_norm; //输入图像归一化
        cv::Mat R;
        cv::Mat out_lime; //增强后的图像
        cv::Mat dv;
        cv::Mat dh;
        cv::Mat T_hat;    //初始化的光照图
        cv::Mat W; 	      //权重矩阵
        cv::Mat veCDD;
        int channel; 	  //存储图像通道数
        int row; 		  //图像的行数
        int col; 		  //图像的列数
        float alpha=1;
        float rho =2;
        float gamma = 0.7;
        float epsilon;    //迭代系数
		float thd;  //迭代收敛时间阈值

    	public: 
        lime(cv::Mat src); 
        cv::Mat Mat2Vec(cv::Mat mat);                         //矩阵向量化
        cv::Mat reshape1D(cv::Mat mat);						  //按行压缩矩阵到一维
        cv::Mat getReal(cv::Mat mat);						  //获取矩阵实部
        cv::Mat derivative(cv::Mat matrix);					  //求解矩阵导数
        cv::Mat solveT(cv::Mat G, cv::Mat Z, float u);		  //求解子问题T
		cv::Mat solveG(cv::Mat T,cv::Mat Z,float u,cv::Mat W);//求解子问题G
        cv::Mat solveZ(cv::Mat T,cv::Mat G,cv::Mat Z,float u);//求解子问题Z
        cv::Mat Dev(int n, int k);							  //求解矩阵一阶导数
        cv::Mat getMax(const cv::Mat &bgr);					  //获取色彩通道最大值
        cv::Mat optIllumMap();								  //获取优化后光照图
		cv::Mat enhance(cv::Mat &src);						  //图像增强					

        static inline float comp(float& a, float& b, float& c) // 声明一个静态内联函数，用于比较三个浮点数的大小，并返回最大值
        {
            return fmax(a, fmax(b, c));
        }

        void weightStrategy();
        void _init_IllumMap(cv::Mat src);
        void Illum_filter(cv::Mat& img_in, cv::Mat& img_out); //滤波、GAMMA矫正
        void Illumination(cv::Mat& src, cv::Mat& out); 		  //求解每个像素的光照强度
        float solveU(float u);								  //求解子问题u
        float Frobenius(cv::Mat mat);
    };

	//构造函数
    lime::lime(cv::Mat src) 
    {
		//获取输入图像的通道数
        channel = src.channels(); 
    }

	//光照图初始化
    void lime::_init_IllumMap(cv::Mat src){ 
		// 将输入图像转换为 float 类型，并进行归一化
	    src.convertTo(img_norm, CV_32F, 1 / 255.0, 0); 
		// 获取归一化图像的大小
        cv::Size sz(img_norm.size()); 
        row = img_norm.rows;
        col = img_norm.cols;
		//构建初始照明图T
        T_hat = lime::getMax(img_norm);  
        //求T_hat的f范数 
        epsilon = Frobenius(T_hat)*0.001;
        dv = Dev(row, 1);
	    dh = Dev(col, -1);
		float u = dv.at<float>(0,0);
		float u2 = dh.at<float>(0,0);
		veCDD = cv::Mat(1,row*col, CV_32F, cv::Scalar::all(0.0));
	   	//定义一维矩阵并初始化为0
        veCDD.at<float>(0,0) = 4;
		veCDD.at<float>(0,1) = -1;
		veCDD.at<float>(0,row) = -1;
		veCDD.at<float>(0,row*col-1) = -1;
		veCDD.at<float>(0,row*col-row) = -1;   
	}
 
	//获取色彩通道最大值
	cv::Mat lime::getMax(const cv::Mat& bgr)
        {
            cv::Mat temp_mat(row, col, CV_32F, cv::Scalar::all(0.0));
            std::vector<cv::Mat> img_norm_rgb;
            cv::Mat img_norm_b, img_norm_g, img_norm_r;
            cv::split(bgr, img_norm_rgb);
            img_norm_g = img_norm_rgb.at(0);
            img_norm_b = img_norm_rgb.at(1);
            img_norm_r = img_norm_rgb.at(2);
			for(int i = 0; i < row; i++){
				for(int j = 0; j< col; j++){
					temp_mat.at<float>(i,j) = MAX(MAX(img_norm_g.at<float>(i,j),img_norm_b.at<float>(i,j)), img_norm_r.at<float>(i,j));
				}
			}
            return temp_mat;
        }

	//求解矩阵范数
	float lime::Frobenius(cv::Mat mat)
	{
		int row = mat.rows;
		int col = mat.cols;
		float total = 0.0;
		for(int i = 0; i < row ; i++){
			for(int j =0; j< col ; j++){
				total = total + pow(mat.at<float>(i,j), 2);
			}
		}
		total = sqrt(total);
		return total;
	}

	//求解矩阵导数
	cv::Mat lime::derivative(cv::Mat matrix){  
		cv::Mat v = dv * matrix;  
		cv::Mat h = matrix * dh;
		cv::Mat matrix_C ;
		//矩阵垂直拼接
		cv::vconcat(v,h,matrix_C); 
		return matrix_C;
	}

	//求解子问题T
	cv::Mat lime::solveT(cv::Mat G, cv::Mat Z, float u){  
		cv::Mat X = G - (Z / u);   //bug
		int row_temp = X.rows;
		cv::Mat Xv = X.rowRange(0, row);  
		cv::Mat Xh = X.rowRange(row,row_temp);//要取 -1
		cv::Mat temp = dv*Xv+ Xh*dh;
		cv::Mat numerator;
		cv::Mat denominator;
		cv::Mat mat_temp1;
		mat_temp1 = Mat2Vec(2*T_hat + u*temp);
		cv::dft(mat_temp1,numerator,cv::DFT_COMPLEX_OUTPUT);
		cv::Mat mat_temp2 = veCDD* u;
		cv::dft(mat_temp2, denominator,cv::DFT_COMPLEX_OUTPUT);
		denominator = denominator + 2;
		cv::Mat T_temp;
		temp = numerator / denominator;
		temp = getReal(temp);
		dft(temp,T_temp,cv::DFT_COMPLEX_OUTPUT);  
		T_temp = getReal(T_temp);
		T_temp = T_temp/(T_temp.cols);
		auto u5 = T_temp.at<float>(0,0);
		auto u6 = T_temp.at<float>(0,4);
		normalize(T_temp,T_temp,0.2,1,CV_MINMAX); 
		cv::Mat T = reshape1D(T_temp);
		T.convertTo(T, CV_32F);  
		return T;
	}
	cv::Mat lime::getReal(cv::Mat mat){ //获取矩阵的实部
		int col_temp = mat.cols;
		cv::Mat mat_return(1,col_temp, CV_32F, cv::Scalar::all(0.0));
		for(int i =0; i<col_temp; i++){
				mat_return.at<float>(0,i) = mat.at<float>(0,2*i);
			}
		return mat_return;

	}

	//将多维矩阵压缩成一维
	cv::Mat lime::Mat2Vec(cv::Mat mat){  
		mat = mat.t(); //矩阵转置
		int row = mat.rows;
		int col = mat.cols;
		cv::Mat mat_one(1,row * col, CV_32F);
		int num_elements = row * col;

		for(int i = 0; i < row ; i++){
			for(int j =0; j< col ; j++){
				mat_one.at<float>(0,i*col+j) = mat.at<float>(i,j);
			}
		}

		return mat_one;       
	}

	//将多维矩阵压缩成一维
	cv::Mat lime::reshape1D(cv::Mat mat){  
		cv::Mat mat_temp(row,col, CV_32F);
		
		for(int i = 0; i < col ; i++){
		for(int j =0; j< row ; j++){
			
			mat_temp.at<float>(j,i) = mat.at<float>(0,i*row + j);
			}
		}
		return mat_temp;       
	}

	//求解子问题G
	cv::Mat lime::solveG(cv::Mat T,cv::Mat Z,float u,cv::Mat W){
		//求出 T的一阶导数
		cv::Mat dT = derivative(T); 
		cv::Mat epsilon = alpha * W / u; 
		cv::Mat X = dT + Z / u;
		//获取一个图像矩阵的符号矩阵
		int row_temp = X.rows;
		int col_temp = X.cols;
		cv::Mat mat_temp(row_temp,col_temp,CV_32F);

	   for(int i = 0; i < row_temp ; i++){
		for(int j =0; j< col_temp ; j++){
			if (X.at<float>(i,j) >0){
				mat_temp.at<float>(i,j) = 1;
			}
			else if(X.at<float>(i,j)<0){
				mat_temp.at<float>(i,j) =-1;
			}
			else 
			mat_temp.at<float>(i,j) = 0;
		}
	  }

		cv::Mat S_ce =cv::max(cv::abs(X) - epsilon, 0);
		cv::Mat O = mat_temp.mul(S_ce);
		return O;
	}

	//求解子问题Z
	cv::Mat lime::solveZ(cv::Mat T,cv::Mat G,cv::Mat Z,float u){ 
		cv::Mat dT = derivative(T);
		return Z + u*(dT - G);
	}

	//求解子问题u
	float lime::solveU(float u){  
		return u* rho;
	}

	//求解权重矩阵
	void lime::weightStrategy(){ 
		cv::Mat dTv = dv * T_hat;
		cv::Mat dTh = T_hat* dh;
		cv::Mat Wv = 1/ (cv::abs(dTv) + 1);
		cv::Mat Wh = 1/ (cv::abs(dTh) + 1);
		cv::vconcat(Wv, Wh, W);
	}

	//求矩阵导数
	cv::Mat lime::Dev(int n, int k){   //求一阶导数的方法
		cv::Mat mat_temp = cv::Mat::eye(n,n,CV_32F);  
		mat_temp = mat_temp *-1;//让矩阵的对角元素为-1
		//让矩阵k对角的元素为1
		if(k > 0){
		for(int y = 0;y <n - k; y++){ 
		mat_temp.at<float>(y,y + k) = 1;
		}
	}
	else{
		for(int y = -k;y <n ; y++){ 
		mat_temp.at<float>(y,y + k) = 1;
		}
	}
	return mat_temp;
	}

	//获取光照图
	cv::Mat lime::optIllumMap(){
		//得到权重矩阵W
		weightStrategy();  
		cv::Mat T(row,col, CV_32F, cv::Scalar::all(0.0));
		cv::Mat G(row*2,col, CV_32F, cv::Scalar::all(0.0));
		cv::Mat Z(row*2,col, CV_32F, cv::Scalar::all(0.0));
		int t = 0;
		float u = 1;

		while (true){
			T = solveT(G,Z,u);
			G = solveG(T,Z,u,W);
			Z = solveZ(T,G,Z,u);
			u = solveU(u);

			//加速收敛过程
			if(t == 0){
				float temp = Frobenius(derivative(T) - G);
				thd = ceil(2* log(temp / epsilon));
			}
			t += 1;
			//达到收敛阈值就结束迭代
			if(t >=thd){ 
				break;
			}
		}
		auto r1 = T.at<float>(0,0);
		auto r2 = T.at<float>(1,0);
		auto r3 = T.at<float>(2,0);
		return T;	
	}

	//图像增强
	cv::Mat lime::enhance(cv::Mat &src){
		_init_IllumMap(src);
		cv::Size sz(img_norm.size());
		R = cv::Mat(sz, CV_32F, cv::Scalar::all(0.0));
		std::vector<cv::Mat> img_norm_rgb; // 定义一个存储三通道分量的向量
		cv::Mat img_norm_b, img_norm_g, img_norm_r; // 定义三个矩阵，分别用于存储三个通道的分量

		cv::split(img_norm, img_norm_rgb); // 将归一化图像分解为三个通道
		img_norm_g = img_norm_rgb.at(0); // 获取绿色通道
		img_norm_b = img_norm_rgb.at(1); // 获取蓝色通道
		img_norm_r = img_norm_rgb.at(2); // 获取红色通道
		cv::Mat T = optIllumMap();

		auto g = img_norm_g / T ;// 计算增强后的绿色通道
		auto b = img_norm_b / T; // 计算增强后的蓝色通道
		auto r = img_norm_r / T; // 计算增强后的红色通道

		cv::Mat g1, b1, r1;
		threshold(g, g1, 0.0, 0.0, 3);
		threshold(b, b1, 0.0, 0.0, 3);
		threshold(r, r1, 0.0, 0.0, 3);

		img_norm_rgb.clear(); 		// 清空 img_norm_rgb 向量
		img_norm_rgb.push_back(g1); // 将处理后的绿色通道添加到向量中
		img_norm_rgb.push_back(b1); // 将处理后的蓝色通道添加到向量中
		img_norm_rgb.push_back(r1); // 将处理后的红色通道添加到向量中

		cv::merge(img_norm_rgb, out_lime); // 将处理后的三个通道合并为一个图像
		out_lime.convertTo(out_lime, CV_8U, 255); // 将 float 类型的图像转换回 uchar 类型，并将像素值范围恢复到 [0, 255]
		return out_lime;
	}

	// 滤波和伽马校正
	void lime::Illum_filter(cv::Mat& img_in, cv::Mat& img_out) 
	{
		// 定义滤波器的尺寸
		int ksize = 5; 
		// 使用均值滤波器对输入图像进行滤波
		blur(img_in, img_out, cv::Size(ksize, ksize));
		// 对滤波后的图像进行伽马校正
		int row = img_out.rows;
		int col = img_out.cols;
		float tem;
		float gamma = 0.8;
		
		// 计算当前像素的伽马校正值
		for (int i = 0; i < row; i++)
			{
			for (int j = 0; j < col; j++)
				{
					// 如果校正值小于等于 0，则设置为 0.0001
					// 如果校正值大于 1，则设置为 1
					tem = pow(img_out.at<float>(i, j), gamma); 
					tem = tem <= 0 ? 0.0001 : tem; 
					tem = tem > 1 ? 1 : tem; 
					// 将校正后的像素值存储在 img_out 矩阵中
					img_out.at<float>(i, j) = tem;
				}
			}
	}

	//计算每个像素亮度
	void lime::Illumination(cv::Mat& src, cv::Mat& out) 
	{
		int row = src.rows, col = src.cols;

		
		for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					// 调用 compare 函数计算亮度，并将结果存储在 out 矩阵中
					out.at<float>(i, j) = lime::comp(src.at<cv::Vec3f>(i, j)[0],
					src.at<cv::Vec3f>(i, j)[1],
					src.at<cv::Vec3f>(i, j)[2]);
				}

			}
	}
}

int main()
{
	double t = cv::getTickCount();
	cv::Mat img_in = cv::imread("../data/test6.jpg");
    cv::Mat img_out;
	if(img_in.empty()) // 判断读入图片是否成功
    {
        std::cout<<"Error Input!"<<std::endl; // 输出错误信息
        return -1; // 退出程序
    }
	cv::imshow("raw_picture", img_in); // 显示原始图片
	LIME::lime* l; // 创建 LIME 算法对象
    l = new LIME::lime(img_in); // 初始化 LIME 算法对象
    img_out = l->enhance(img_in); // 对输入图片进行 LIME 增强处理，得到输出图片
    cv::imshow("img_lime", img_out); // 显示增强图片
	cv::imwrite("output.jpg", img_out);
	t = (cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "时间：" << t << "s" << endl;
	cv::waitKey(0); // 等待按键
    return 0;
}