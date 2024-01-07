#ifndef FEATURE_EXACTION_LIME_H // 防止头文件被重复包含
#define FEATURE_EXACTION_LIME_H // 定义宏，表示已包含此头文件

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
#include <omp.h>

using namespace std;

namespace feature // 定义一个名为 feature 的命名空间
{

    class lime // 定义一个名为 lime 的类
    {
    public: // 定义公共成员
    int channel; // 声明一个整型变量，用于存储图像通道数
    cv::Mat img_norm;
    cv::Mat R;
    cv::Mat out_lime; // 声明一个 OpenCV 矩阵对象，用于存储增强后的图像
    cv::Mat dv;
    cv::Mat dh;
    float alpha=1;
    float rho =2;
    float gamma = 0.7;
    cv::Mat T_hat; //初始化的光照图
    cv::Mat W; //权重矩阵
    int row; //图像矩阵的行数
    int col; //图像矩阵的列数
    cv::Mat veCDD;
    float epsilon;
    float expert_t;



    public: // 定义公共成员函数
    lime(cv::Mat src); // 构造函数，接收一个 OpenCV 矩阵对象作为输入
    cv::Mat vectorize(cv::Mat mat);
    cv::Mat reshape(cv::Mat mat);
        
    cv::Mat lime_enhance(cv::Mat& src); // 声明一个名为 lime_enhance 的成员函数，接收一个 OpenCV 矩阵对象的引用作为输入

        static inline float compare(float& a, float& b, float& c) // 声明一个静态内联函数，用于比较三个浮点数的大小，并返回最大值
        {
            return fmax(a, fmax(b, c));
        }
    void __weightingStrategy_2();
    cv::Mat get_real(cv::Mat mat);
    cv::Mat __derivative(cv::Mat matrix);
    cv::Mat __T_subproblem(cv::Mat G, cv::Mat Z, float u);
    cv::Mat __Z_subproblem(cv::Mat T,cv::Mat G,cv::Mat Z,float u);
    cv::Mat __G_subproblem(cv::Mat T,cv::Mat Z,float u,cv::Mat W);
    float __u_subproblem(float u);
    cv::Mat firstOrderDerivative(int n, int k);
    cv::Mat calcMax(const cv::Mat &bgr);
    void Illumination(cv::Mat& src, cv::Mat& out); // 声明一个名为 Illumination 的成员函数，接收两个 OpenCV 矩阵对象的引用作为输入
    cv::Mat optimizeIllumMap();
    float Frobenius(cv::Mat mat);
    void __initIllumMap(cv::Mat src);
    void Illumination_filter(cv::Mat& img_in, cv::Mat& img_out); // 声明一个名为 Illumination_filter 的成员函数，接收两个 OpenCV 矩阵对象的引用作为输入
    cv::Mat enhance(cv::Mat &src);
    cv::Mat long_mat(int k, int len);
    };

     lime::lime(cv::Mat src) // 构造函数 获取初始化的值 通道数
    {
        channel = src.channels(); // 获取输入图像的通道数
    }

    void lime::__initIllumMap(cv::Mat src){ //初始化，并得到各种参数
	    src.convertTo(img_norm, CV_32F, 1 / 255.0, 0); // 将输入图像转换为 float 类型，并进行归一化

        cv::Size sz(img_norm.size()); // 获取归一化图像的大小

        row = img_norm.rows;

        col = img_norm.cols;

        T_hat = lime::calcMax(img_norm);  //构建初始照明图T
			//float u = T_hat.at<float>(0,0);
            //求T_hat的f范数 
        epsilon = Frobenius(T_hat)*0.001;
        dv = firstOrderDerivative(row, 1);
	    dh = firstOrderDerivative(col, -1);
		float u = dv.at<float>(0,0);
		float u2 = dh.at<float>(0,0);
		veCDD = cv::Mat(1,row*col, CV_32F, cv::Scalar::all(0.0));
	   		//定义一维矩阵并初始化为0
        veCDD.at<float>(0,0) = 4;
		veCDD.at<float>(0,1) = -1;
		veCDD.at<float>(0,row) = -1;
		veCDD.at<float>(0,row*col-1) = -1;
		veCDD.at<float>(0,row*col-row) = -1;   //测试一下这个有没有问题
	}

	// cv::Mat lime::calcMax(const cv::Mat& bgr){//求RGB三个通道的最大值用于构建初始化的光照图
	// 	cv::Mat temp_mat(row, col, CV_32F, cv::Scalar::all(0.0));
	// 	std::vector<cv::Mat> img_norm_rgb; // 定义一个存储三通道分量的向量
	// 	cv::Mat img_norm_b, img_norm_g, img_norm_r; // 定义三个矩阵，分别用于存储三个通道的分量

	// 	cv::split(bgr, img_norm_rgb); // 将归一化图像分解为三个通道

	// 	img_norm_g = img_norm_rgb.at(0); // 获取绿色通道
	// 	img_norm_b = img_norm_rgb.at(1); // 获取蓝色通道
	// 	img_norm_r = img_norm_rgb.at(2); // 获取红色通道
		
	// 	for(int i = 0; i < row; i++){
	// 		for(int j = 0; j< col; j++){
				
	// 			temp_mat.at<float>(i,j) = MAX(MAX(img_norm_g.at<float>(i,j),img_norm_b.at<float>(i,j)), img_norm_r.at<float>(i,j));
	// 		}
	// 	}
	// 	return temp_mat;
	// }    

	cv::Mat lime::calcMax(const cv::Mat& bgr)
        {
            cv::Mat temp_mat(row, col, CV_32F, cv::Scalar::all(0.0));
            std::vector<cv::Mat> img_norm_rgb;
            cv::Mat img_norm_b, img_norm_g, img_norm_r;

            cv::split(bgr, img_norm_rgb);
            img_norm_g = img_norm_rgb.at(0);
            img_norm_b = img_norm_rgb.at(1);
            img_norm_r = img_norm_rgb.at(2);

            // 使用NEON加速计算最大值
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j += 4)  // 每次处理4个元素
                {
                    float32x4_t g = vld1q_f32(img_norm_g.ptr<float>(i) + j);
                    float32x4_t b = vld1q_f32(img_norm_b.ptr<float>(i) + j);
                    float32x4_t r = vld1q_f32(img_norm_r.ptr<float>(i) + j);

                    float32x4_t max_val = vmaxq_f32(g, vmaxq_f32(b, r));

                    vst1q_f32(temp_mat.ptr<float>(i) + j, max_val);
                }
            }
            return temp_mat;
        }




	// float lime::Frobenius(cv::Mat mat){   //求一个矩阵的F范数
	// 		int row_temp = mat.rows;
	// 		int col_temp = mat.cols;
	// 		float total = 0.0;
			
	// 		for(int i = 0; i < row_temp ; i++){
	// 		for(int j =0; j< col_temp ; j++){
				
	// 			total = total + pow(mat.at<float>(i,j), 2);
	// 		}
	// 	}
	// 	total = sqrt(total);
	// 	return total;

	// }

		float lime::Frobenius(cv::Mat mat)
	{
		int row_temp = mat.rows;
		int col_temp = mat.cols;

		float32x4_t total_sum = vdupq_n_f32(0.0f);

		for (int i = 0; i < row_temp; i++)
		{
			for (int j = 0; j < col_temp; j += 4)  // 每次处理4个元素
			{
				float32x4_t values = vld1q_f32(mat.ptr<float>(i) + j);
				float32x4_t squared_values = vmulq_f32(values, values);

				total_sum = vaddq_f32(total_sum, squared_values);
			}
		}

		// 将向量中的4个部分求和
		total_sum = vpaddq_f32(total_sum, total_sum);
		total_sum = vpaddq_f32(total_sum, total_sum);

		// 提取结果
		float32x2_t result = vget_low_f32(total_sum);
		float squared_sum = vget_lane_f32(result, 0);

		return squared_sum;
	}

	cv::Mat lime::__derivative(cv::Mat matrix){  //求一个矩阵的导数
		cv::Mat v = dv * matrix;  //矩阵的乘法
		cv::Mat h = matrix * dh;
		cv::Mat matrix_C ;
		cv::vconcat(v,h,matrix_C); //矩阵垂直拼接
		return matrix_C;
	}

	cv::Mat lime::__T_subproblem(cv::Mat G, cv::Mat Z, float u){  //解决T的子问题
		cv::Mat X = G - (Z / u);   //bug
		int row_temp = X.rows;
		cv::Mat Xv = X.rowRange(0, row);  
		cv::Mat Xh = X.rowRange(row,row_temp);//要取 -1
		cv::Mat temp = dv*Xv+ Xh*dh;
		cv::Mat numerator;
		cv::Mat denominator;
		cv::Mat mat_temp1;
		mat_temp1 = vectorize(2*T_hat + u*temp);
		cv::dft(mat_temp1,numerator,cv::DFT_COMPLEX_OUTPUT);
		cv::Mat mat_temp2 = veCDD* u;
		cv::dft(mat_temp2, denominator,cv::DFT_COMPLEX_OUTPUT);
		denominator = denominator + 2;
		cv::Mat T_temp;
		temp = numerator / denominator;
		temp = get_real(temp);
		dft(temp,T_temp,cv::DFT_COMPLEX_OUTPUT);  
		T_temp = get_real(T_temp);
		T_temp = T_temp/(T_temp.cols);
		auto u5 = T_temp.at<float>(0,0);
		auto u6 = T_temp.at<float>(0,4);
		normalize(T_temp,T_temp,0.2,1,CV_MINMAX); 
		cv::Mat T = reshape(T_temp);
		T.convertTo(T, CV_32F);  
		return T;
	}
	cv::Mat lime::get_real(cv::Mat mat){ //获取矩阵的实部
		int col_temp = mat.cols;
		cv::Mat mat_return(1,col_temp, CV_32F, cv::Scalar::all(0.0));
		for(int i =0; i<col_temp; i++){
				mat_return.at<float>(0,i) = mat.at<float>(0,2*i);
			}
		// for (int i = 0; i < col_temp; i += 8)  // 每次处理8个元素
		// {
		// 	float32x4x2_t values = vld2q_f32(mat.ptr<float>(0) + 2 * i);

		// 	// 拷贝前4个元素
		// 	vst1q_f32(mat_return.ptr<float>(0) + i, values.val[0]);

		// 	// 拷贝后4个元素
		// 	vst1q_f32(mat_return.ptr<float>(0) + i + 4, values.val[1]);
		// }
		return mat_return;

	}

	cv::Mat lime::long_mat(int k, int len){
		cv::Mat long_mat(1,len, CV_32F, cv::Scalar::all(0.0));
		float PI = 3.14159265;
		for(int i=0; i<len; i++){
			long_mat.at<float>(0,i) = cos(2*PI*i*k/len);
		}

		// const float two_pi_over_len = 2 * PI / len;

    	// float32x4_t vec_k = vdupq_n_f32(2 * PI * k);

		// for (int i = 0; i < len; i += 4)  // 每次处理4个元素
		// {
		// 	float32x4_t vec_i = vdupq_n_f32(i);

		// 	// 计算cos(2*PI*i*k/len)的近似值
		// 	float32x4_t vec_cos = vcvtq_f32_s32(vcvtq_s32_f32(vmulq_f32(vec_i, vec_k)));

		// 	// 将近似值存储到目标矩阵
		// 	vst1q_f32(long_mat.ptr<float>(0) + i, vec_cos);
		// }
		return long_mat;
	}
	
	cv::Mat lime::vectorize(cv::Mat mat){  //将多维矩阵压缩成一维
			mat = mat.t(); //现将矩阵转置
			int row_temp = mat.rows;
			int col_temp = mat.cols;
			cv::Mat mat_one(1,row_temp * col_temp, CV_32F);
			
		// 	for(int i = 0; i < row_temp ; i++){
		// 	for(int j =0; j< col_temp ; j++){
				
		// 		mat_one.at<float>(0,i*col_temp+j) = mat.at<float>(i,j);
		// 	}
		// }

		int num_elements = row_temp * col_temp;

		for (int i = 0; i < num_elements; i += 4)  // 每次处理4个元素
		{
			// 加载4个源矩阵中的元素
			float32x4_t vec_src = vld1q_f32(mat.ptr<float>(0) + i);

			// 存储到目标矩阵
			vst1q_f32(mat_one.ptr<float>(0) + i, vec_src);
		}
		return mat_one;       
	}

	cv::Mat lime::reshape(cv::Mat mat){  //将多维矩阵压缩成一维
		cv::Mat mat_temp(row,col, CV_32F);
		
		for(int i = 0; i < col ; i++){
		for(int j =0; j< row ; j++){
			
			mat_temp.at<float>(j,i) = mat.at<float>(0,i*row + j);
			}
		}
		//用不了，用了图不对
		// for (int i = 0; i < col; ++i)  // 列循环
		// {
		// 	for (int j = 0; j < row; j += 4)  // 行循环，每次处理4行
		// 	{
		// 		// 加载4个元素到NEON向量寄存器中
		// 		float32x4_t vec_src = vld1q_f32(mat.ptr<float>(0) + (i * row + j));

		// 		// 转置存储到目标矩阵
		// 		vst1q_f32(mat_temp.ptr<float>(j) + i, vec_src);
		// 	}
		// }

		return mat_temp;       
	}

	cv::Mat lime::__G_subproblem(cv::Mat T,cv::Mat Z,float u,cv::Mat W){//解决G的子问题
		cv::Mat dT = __derivative(T); //求出 T的一阶导数
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

	cv::Mat lime::__Z_subproblem(cv::Mat T,cv::Mat G,cv::Mat Z,float u){ //解决Z的子问题
		cv::Mat dT = __derivative(T);
		return Z + u*(dT - G);
	}

	float lime::__u_subproblem(float u){  //解决u的子问题
		return u* rho;
	}

	void lime::__weightingStrategy_2(){ //解决权重矩阵
		cv::Mat dTv = dv * T_hat;
		cv::Mat dTh = T_hat* dh;
		cv::Mat Wv = 1/ (cv::abs(dTv) + 1);
		cv::Mat Wh = 1/ (cv::abs(dTh) + 1);
		cv::vconcat(Wv, Wh, W);
	}

	cv::Mat lime::firstOrderDerivative(int n, int k){   //求一阶导数的方法
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

	cv::Mat lime::optimizeIllumMap(){
		__weightingStrategy_2();  //得到权重矩阵W

		cv::Mat T(row,col, CV_32F, cv::Scalar::all(0.0));
		cv::Mat G(row*2,col, CV_32F, cv::Scalar::all(0.0));
		cv::Mat Z(row*2,col, CV_32F, cv::Scalar::all(0.0));
		int t = 0;
		float u = 1;

		while (true){
			T = __T_subproblem(G,Z,u);
			G = __G_subproblem(T,Z,u,W);
			Z = __Z_subproblem(T,G,Z,u);
			u = __u_subproblem(u);

			//加速收敛过程
			if(t == 0){
				float temp = Frobenius(__derivative(T) - G);
				expert_t = ceil(2* log(temp / epsilon));
			}
			t += 1;
			if(t >=4){ 
				break;
			}
		}
		//T = pow(4.0, gamma);
		auto r1 = T.at<float>(0,0);
		auto r2 = T.at<float>(1,0);
		auto r3 = T.at<float>(2,0);
		return T;	
	}

	cv::Mat lime::enhance(cv::Mat &src){
		__initIllumMap(src);
		cv::Size sz(img_norm.size());
		R = cv::Mat(sz, CV_32F, cv::Scalar::all(0.0));
		std::vector<cv::Mat> img_norm_rgb; // 定义一个存储三通道分量的向量
		cv::Mat img_norm_b, img_norm_g, img_norm_r; // 定义三个矩阵，分别用于存储三个通道的分量

		cv::split(img_norm, img_norm_rgb); // 将归一化图像分解为三个通道
		cv::Mat T = optimizeIllumMap();
		cv::Mat g1, b1, r1;
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				img_norm_g = img_norm_rgb.at(0); // 获取绿色通道
				auto g = img_norm_g / T ;// 计算增强后的绿色通道
				threshold(g, g1, 0.0, 0.0, 3);
			}
			
			#pragma omp section
			{
				img_norm_b = img_norm_rgb.at(1); // 获取蓝色通道
				auto b = img_norm_b / T; // 计算增强后的蓝色通道
				threshold(b, b1, 0.0, 0.0, 3);
			}

			#pragma omp section
			{
				img_norm_r = img_norm_rgb.at(2); // 获取红色通道
				auto r = img_norm_r / T; // 计算增强后的红色通道
				threshold(r, r1, 0.0, 0.0, 3);
			}
		}

		img_norm_rgb.clear(); 		// 清空 img_norm_rgb 向量
		img_norm_rgb.push_back(g1); // 将处理后的绿色通道添加到向量中
		img_norm_rgb.push_back(b1); // 将处理后的蓝色通道添加到向量中
		img_norm_rgb.push_back(r1); // 将处理后的红色通道添加到向量中

		cv::merge(img_norm_rgb, out_lime); // 将处理后的三个通道合并为一个图像
		out_lime.convertTo(out_lime, CV_8U, 255); // 将 float 类型的图像转换回 uchar 类型，并将像素值范围恢复到 [0, 255]
		return out_lime;
	}

	void lime::Illumination_filter(cv::Mat& img_in, cv::Mat& img_out) // 定义一个用于滤波和伽马校正的 Illumination_filter 函数
	{
		int ksize = 5; // 定义滤波器的尺寸
		// 使用均值滤波器对输入图像进行滤波
		blur(img_in, img_out, cv::Size(ksize, ksize));
		// GaussianBlur(img_in,img_mean_filter,Size(ksize,ksize),0,0);

		// 对滤波后的图像进行伽马校正
		int row = img_out.rows;
		int col = img_out.cols;
		float tem;
		float gamma = 0.8;

		
		for (int i = 0; i < row; i++)
			{
			for (int j = 0; j < col; j++)
				{
					tem = pow(img_out.at<float>(i, j), gamma); // 计算当前像素的伽马校正值
					tem = tem <= 0 ? 0.0001 : tem; // 如果校正值小于等于 0，则设置为 0.0001
					tem = tem > 1 ? 1 : tem; // 如果校正值大于 1，则设置为 1

					img_out.at<float>(i, j) = tem; // 将校正后的像素值存储在 img_out 矩阵中
				}
			}
	}

	void lime::Illumination(cv::Mat& src, cv::Mat& out) // 定义一个用于计算每个像素亮度的 Illumination 函数
	{
		int row = src.rows, col = src.cols;

		
		for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					out.at<float>(i, j) = lime::compare(src.at<cv::Vec3f>(i, j)[0],
					src.at<cv::Vec3f>(i, j)[1],
					src.at<cv::Vec3f>(i, j)[2]);
	// 调用 compare 函数计算亮度，并将结果存储在 out 矩阵中
				}

			}
	}
}

#endif //FEATURE_EXACTION_LIME_H // 结束头文件保护