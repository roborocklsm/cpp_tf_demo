#include <fstream>
#include <utility>
#include <iostream>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
using namespace tensorflow;
 
// 定义一个函数讲OpenCV的Mat数据转化为tensor，python里面只要对cv2.read读进来的矩阵进行np.reshape之后，
// 数据类型就成了一个tensor，即tensor与矩阵一样，然后就可以输入到网络的入口了，但是C++版本，我们网络开放的入口
// 也需要将输入图片转化成一个tensor，所以如果用OpenCV读取图片的话，就是一个Mat，然后就要考虑怎么将Mat转化为
// Tensor了
void CVMat_to_Tensor(Mat img,Tensor* output_tensor,int input_rows,int input_cols)
{
    //imshow("input image",img);
    //图像进行resize处理
    resize(img,img,cv::Size(input_cols,input_rows));
    //imshow("resized image",img);
 
    //归一化
    img.convertTo(img,CV_32FC1);
    img=1-img/255;
 
    //创建一个指向tensor的内容的指针
    float *p = output_tensor->flat<float>().data();
 
    //创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
    cv::Mat tempMat(input_rows, input_cols, CV_32FC1, p);
    img.convertTo(tempMat,CV_32FC1);
 
//    waitKey(0);
 
}
 
int main(int argc, char** argv )
{
    /*--------------------------------配置关键信息------------------------------*/
    string model_path="../networks/sp_coco_tiny128_bow.pb";
    string image_path="../images/6.png";
    int input_height = 480;
    int input_width = 640;
    vector<string> input_tensor_name={"superpoint/image"};
    vector<string> output_tensor_name={ "superpoint/pts",
                                        "superpoint/desc",
                                        "superpoint/pred_tower0/tiny_descriptor/bn2_1/FusedBatchNorm",
                                    };
 
    /*--------------------------------创建session------------------------------*/
    Session* session;
    Status status = NewSession(SessionOptions(), &session);//创建新会话Session
 
    /*--------------------------------从pb文件中读取模型--------------------------------*/
    GraphDef graphdef; //Graph Definition for current model
 
    Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
    if (!status_load.ok()) {
        cout << "ERROR: Loading model failed..." << model_path << std::endl;
        cout << status_load.ToString() << "\n";
        return -1;
    }
    Status status_create = session->Create(graphdef); //将模型导入会话Session中;
    if (!status_create.ok()) {
        cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
        return -1;
    }
    cout << "<----Successfully created session and load graph.------->"<< endl;
 
    /*---------------------------------载入测试图片-------------------------------------*/
    cout<<endl<<"<------------loading test_image-------------->"<<endl;
    Mat img=imread(image_path,0);
    if(img.empty())
    {
        cout<<"can't open the image!!!!!!!"<<endl;
        return -1;
    }
 
    //创建一个tensor作为输入网络的接口
    Tensor resized_tensor(DT_FLOAT, TensorShape({1,input_height,input_width,1}));
 
    //将Opencv的Mat格式的图片存入tensor
    CVMat_to_Tensor(img,&resized_tensor,input_height,input_width);
 
    cout << resized_tensor.DebugString()<<endl;
 
    /*-----------------------------------用网络进行测试-----------------------------------------*/
    cout<<endl<<"<-------------Running the model with test_image--------------->"<<endl;
    //前向运行，输出结果一定是一个tensor的vector
    vector<tensorflow::Tensor> outputs;
    //string output_node = output_tensor_name;
    Status status_run = session->Run({{input_tensor_name[0], resized_tensor}}, output_tensor_name, {}, &outputs);
 
    if (!status_run.ok()) {
        cout << "ERROR: RUN failed..."  << std::endl;
        cout << status_run.ToString() << "\n";
        return -1;
    }
    //把输出值给提取出来
    cout << "Output tensor size:" << outputs.size() << std::endl;
    for (std::size_t i = 0; i < outputs.size(); i++) {
        cout << outputs[i].DebugString()<<endl;
    }
 
    Tensor keypoint = outputs[0];
    Tensor desc = outputs[1];
    Tensor desc_raw = outputs[2];           
    // 输出结果
    cout<<"keypoints: "<<endl;
    

    return 0;
}