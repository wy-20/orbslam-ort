#include "iostream"
#include "line_extractor.h"
#include "line_match.h"
#include "vector"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

vector<vector<vector<float>>> input_for_test(const string& filename, int c, int h, int w)
{
    cout << filename << endl;
    vector<vector<vector<float>>> featureMap(c, vector<vector<float>>(h, vector<float>(w)));
    ifstream inFile(filename, ios::in|ios::binary);
    for(int i = 0; i < c; i++)
    {
        for(int j = 0; j < h; j++)
        {
            for(int k = 0; k < w; k++)
            {
                float x;
                inFile.read((char*)&x, sizeof(float));
                featureMap[i][j][k] = x;
            }
        }
    }
    cout << "read featureMap feature map test: " << endl;
    cout << "featureMap[0][0][1] : " << featureMap[0][0][1] << endl;
    cout << "featureMap[0][1][0] : " << featureMap[0][1][0] << endl;
    cout << "featureMap[1][0][0] : " << featureMap[1][0][0] << endl; // ok
    inFile.close();
    return featureMap;
}

void show_line_detected(Mat mat, vector<pair<pair<int, int>, pair<int, int>>> line_seg)
{
    int n = line_seg.size();
    cout << "detect " << n << " lines" <<endl;
    for(auto l : line_seg)
    {
        Point s = Point(l.first.first, l.first.second);
        Point e = Point(l.second.first, l.second.second);
        line(mat, s, e, Scalar(rand()&255, rand()&255, rand()&255), 3, 4);
    }

}

void show_line_matched(Mat mat1, vector<pair<pair<int, int>, pair<int, int>>> line_seg1,
                       Mat mat2, vector<pair<pair<int, int>, pair<int, int>>> line_seg2,
                       vector<int> match1, vector<int>match2)
{
    for(int i = 0; i < line_seg1.size(); i++)
    {
        int j = match1[i];
        if(j > 0 && j < match2.size() && match2[j] == i)
        {
            int a = rand(), b = rand(), c = rand();
            Point s1 = Point(line_seg1[i].first.first, line_seg1[i].first.second);
            Point e1 = Point(line_seg1[i].second.first, line_seg1[i].second.second);
            line(mat1, s1, e1, Scalar(a & 255, b & 255, c & 255), 3, 4);
            Point s2 = Point(line_seg2[j].first.first, line_seg2[j].first.second);
            Point e2 = Point(line_seg2[j].second.first, line_seg2[j].second.second);
            line(mat2, s2, e2, Scalar(a & 255, b & 255, c & 255), 3, 4);
        }
    }
}

void input_from_model(cv::Mat& input_mat, vector<vector<vector<float>>>& junc, vector<vector<vector<float>>>& heat, vector<vector<vector<float>>>& desc)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* model_path = "../line/SOLD2.onnx";

    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names = {"torch_img"};
    std::vector<const char*> output_node_names = {"junctions","heatmap", "descriptors"};

    std::vector<int64_t> input_node_dims = {1, 1, 376, 1240};
    size_t input_tensor_size = 376 * 1240;
    std::vector<float> input_tensor_values(input_tensor_size);
    for(int i = 0, cnt = 0; i < 376; i++)
    {
        for(int j = 0; j < 1240; j++)
        {
            input_tensor_values[cnt++] = static_cast<float>(input_mat.at<cv::Vec3b>(i, j)[0]) / 255.0;
        }
    }

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

//    std::vector<int64_t> input_mask_node_dims = {1, 20, 4};
//    size_t input_mask_tensor_size = 1 * 20 * 4;
//    std::vector<float> input_mask_tensor_values(input_mask_tensor_size);
//    for (unsigned int i = 0; i < input_mask_tensor_size; i++)
//        input_mask_tensor_values[i] = (float)i / (input_mask_tensor_size + 1);
//    // create input tensor object from data values
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value input_mask_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, input_mask_tensor_values.data(), input_mask_tensor_size, input_mask_node_dims.data(), 3);
//    assert(input_mask_tensor.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
//    ort_inputs.push_back(std::move(input_mask_tensor));
    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 3);

    // Get pointer to output tensor float values
    float* junctions = output_tensors[0].GetTensorMutableData<float>();
    float* heatmap = output_tensors[1].GetTensorMutableData<float>();
    float* descriptors = output_tensors[2].GetTensorMutableData<float>();

    for(int i = 0, cnt = 0; i < 65; i++)
    {
        for(int j = 0; j < 47; j++)
        {
            for(int k = 0; k < 155; k++)
            {
                junc[i][j][k] = junctions[cnt++];
            }
        }
    }
    for(int i = 0, cnt = 0; i < 2; i++)
    {
        for(int j = 0; j < 376; j++)
        {
            for(int k = 0; k < 1240; k++)
            {
                heat[i][j][k] = heatmap[cnt++];
            }
        }
    }
    for(int i = 0, cnt = 0; i < 128; i++)
    {
        for(int j = 0; j < 94; j++)
        {
            for(int k = 0; k < 310; k++)
            {
                desc[i][j][k] = descriptors[cnt++];
            }
        }
    }

    printf("Done!\n");


}

int main() {
    cout << "Read bin file" << endl;
    // 需要修改文件路径
//    string img1_junction = "/home/wy/bytecamp/deep-orbslam/line/img1_junction",
//    img2_junction = "/home/wy/bytecamp/deep-orbslam/line/img2_junction",
//    img1_heatmap = "/home/wy/bytecamp/deep-orbslam/line/img1_heatmap",
//    img2_heatmap = "/home/wy/bytecamp/deep-orbslam/line/img2_heatmap",
//    img1_description = "/home/wy/bytecamp/deep-orbslam/line/img1_description",
//    img2_description = "/home/wy/bytecamp/deep-orbslam/line/img2_description";
//    auto img1_junctions = input_for_test(img1_junction, 65, 47, 155);
//    auto img2_junctions = input_for_test(img2_junction, 65, 47, 155);
//    auto img1_heatmaps = input_for_test(img1_heatmap, 2, 376, 1240);
//    auto img2_heatmaps = input_for_test(img2_heatmap, 2, 376, 1240);
//    auto img1_descriptions = input_for_test(img1_description, 128, 94, 310);
//    auto img2_descriptions = input_for_test(img2_description, 128, 94, 310);
    // tested

    cv::Mat mat1 = cv::imread("/home/wy/bytecamp/deep-orbslam/line/000000.png");
    cv::Mat mat2 = cv::imread("/home/wy/bytecamp/deep-orbslam/line/000001.png");
    vector<vector<vector<float>>>img1_junctions(65, vector<vector<float>>(47, vector<float>(155)));
    vector<vector<vector<float>>>img2_junctions(65, vector<vector<float>>(47, vector<float>(155)));
    vector<vector<vector<float>>>img1_heatmaps(2, vector<vector<float>>(376, vector<float>(1240)));
    vector<vector<vector<float>>>img2_heatmaps(2, vector<vector<float>>(376, vector<float>(1240)));
    vector<vector<vector<float>>>img1_descriptions(128, vector<vector<float>>(94, vector<float>(310)));
    vector<vector<vector<float>>>img2_descriptions(128, vector<vector<float>>(94, vector<float>(310)));


    input_from_model(mat1, img1_junctions, img1_heatmaps, img1_descriptions);
    input_from_model(mat2, img2_junctions, img2_heatmaps, img2_descriptions);
    // 设置线段匹配阈值 源代码设置为 0.25
    float detect_thresh = 0.12;
    int min_dist_pts = 3;
    int num_samples = 15;

    auto lext = line_extractor(detect_thresh, min_dist_pts, num_samples);
    auto line_seg1 = lext.line_detector(img1_junctions, img1_heatmaps);
    auto line_seg2 = lext.line_detector(img2_junctions, img2_heatmaps);

    // 采样点
    vector<vector<pair<float, float>>> line_points1(line_seg1.size(), vector<pair<float, float>>(num_samples, {0, 0}));
    vector<vector<pair<float, float>>> line_points2(line_seg2.size(), vector<pair<float, float>>(num_samples, {0, 0}));
    // 采样点标记
    vector<vector<bool>> valid_points1(line_seg1.size(), vector<bool>(num_samples, false));
    vector<vector<bool>> valid_points2(line_seg2.size(), vector<bool>(num_samples, false));
    // 得到线描述子
    auto line_des1 = lext.line_descriptor(line_seg1, img1_descriptions, line_points1, valid_points1);
    auto line_des2 = lext.line_descriptor(line_seg2, img2_descriptions, line_points2, valid_points2);

    // TODO 添加线段拼接

    // 可视化提取的线
//    show_line_detected(mat1, line_seg1);
//    show_line_detected(mat2, line_seg2);
//    cv::imshow("lines detected from img1",mat1);
//    cv::imshow("lines detected from img2",mat2);
//    cv::waitKey();

    // 线段匹配
    auto lmch = line_match();
    // NW 算法
//    auto matches12 = lmch.NW_match(line_points1, line_points2,
//                valid_points1, valid_points2, line_des1, line_des2);
//    auto matches21 = lmch.NW_match(line_points2, line_points1,
//                valid_points2, valid_points1, line_des2, line_des1);
    // Faiss 匹配
    auto matches12 = lmch.Faiss_match(line_points1, line_points2,
                                   valid_points1, valid_points2, line_des1, line_des2);
    auto matches21 = lmch.Faiss_match(line_points2, line_points1,
                                   valid_points2, valid_points1, line_des2, line_des1);

    show_line_matched(mat1, line_seg1, mat2, line_seg2, matches12, matches21);
    cv::imshow("lines matched img1 by Faiss",mat1);
    cv::imshow("lines matched img2 by Faiss",mat2);
    cv::waitKey();

    return 0;
}
