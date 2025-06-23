/**
 * @file main.cpp
 * @author letterso
 * @brief modified form OroChippw/LightGlue-OnnxRunner
 * @version 0.5
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "Configuration.h"
#include "BaseOnnxRunner.h"
#include "LightGlueOnnxRunner.h"
#include "LightGlueDecoupleOnnxRunner.h"
#include "viz2d.h"
#define THIS_COM "/root/onnx_runtime_cpp"


inline bool fileExists(const std::string &filename)
{
    std::ifstream file(filename.c_str());
    return file.good();
}

std::vector<cv::Mat> ReadImage(std::vector<cv::String> image_filelist, bool grayscale = false)
{
    /*
    Func:
        Read an image from path as RGB or grayscale

    */
    int mode = cv::IMREAD_COLOR;
    if (grayscale)
    {
        mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    }

    std::vector<cv::Mat> image_matlist;
    for (const auto &file : image_filelist)
    {
        std::cout << "[FILE INFO] : " << file << std::endl;
        cv::Mat image = cv::imread(file, mode);
        if (image.empty())
        {
            throw std::runtime_error("[ERROR] Could not read image at " + file);
        }
        if (!grayscale)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR -> RGB
        }
        image_matlist.emplace_back(image);
    }

    return image_matlist;
}

int main(int argc, char *argv[])
{
    /* ****** CONFIG START ****** */
    bool end2end = false;
    std::string extractor_path = THIS_COM + std::string("/superpoint.onnx");
    std::string lightglue_path = THIS_COM + std::string("/superpoint_lightglue.trt.onnx");

    // bool end2end = true;
    // std::string lightglue_path = THIS_COM + std::string("/weights/superpoint_lightglue_end2end_512.onnx");

    // Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.
    std::string extractor_type = "superpoint";
    // Sample image size for ONNX tracing , resize the longer side of the images to this value. Supported image size {512 , 1024 , 2048}
    unsigned int image_size = 512;
    bool grayscale = false;
    std::string device = "cpu"; // Now support "cpu" / "cuda"
    bool viz = true;
    float matchThresh = 0.5f;

    std::string image_path1 = THIS_COM + std::string("/scripts/lightglue/LightGlue/assets/DSC_0410.JPG");
    std::string image_path2 = THIS_COM + std::string("/scripts/lightglue/LightGlue/assets/DSC_0411.JPG");
    std::string save_path = THIS_COM + std::string(" /assets/");

    /* ****** CONFIG END ****** */

    /* ****** Load Cfg , Mode And Image Start****** */
    Configuration cfg;
    cfg.lightgluePath = lightglue_path;
    cfg.extractorPath = extractor_path;

    cfg.extractorType = extractor_type;
    cfg.isEndtoEnd = end2end;
    cfg.grayScale = grayscale;
    cfg.image_size = image_size;
    cfg.threshold = matchThresh;
    cfg.device = device;
    cfg.viz = viz;

    std::transform(cfg.extractorType.begin(), cfg.extractorType.end(),
                   cfg.extractorType.begin(), ::tolower);
    if (cfg.extractorType != "superpoint" && cfg.extractorType != "disk")
    {
        std::cerr << "[ERROR] Unsupported feature extractor type: " << extractor_type << std::endl;

        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "[INFO] Extractor Type : " << cfg.extractorType << std::endl;
    }

    if (fileExists(cfg.lightgluePath))
    {
        if (cfg.isEndtoEnd)
        {
            if (!fileExists(cfg.lightgluePath))
            {
                std::cerr << "[ERROR] The specified LightGlue mode at is not end-to-end. Please pass the extractor_path argument." << extractor_type << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    else
    {
        std::cerr << "[ERROR] LightGlue onnx model Path is not exist : " << cfg.lightgluePath << std::endl;
    }

    std::vector<cv::String> image_filelist1;
    std::vector<cv::String> image_filelist2;
    cv::glob(image_path1, image_filelist1);
    cv::glob(image_path2, image_filelist2);
    if (image_filelist1.size() != image_filelist2.size())
    {
        std::cout << "[INFO] Image Matlist1 size : " << image_filelist1.size() << std::endl;
        std::cout << "[INFO] Image Matlist2 size : " << image_filelist2.size() << std::endl;
        std::cerr << "[ERROR] The number of images in the source folder and \
                    the destination folder is inconsistent"
                  << std::endl;

        return EXIT_FAILURE;
    }

    std::cout << "[INFO] => Building Image Matlist1" << std::endl;
    std::vector<cv::Mat> image_matlist1 = ReadImage(image_filelist1, cfg.grayScale);
    std::cout << "[INFO] => Building Image Matlist2" << std::endl;
    std::vector<cv::Mat> image_matlist2 = ReadImage(image_filelist2, cfg.grayScale);
    /* ****** Load Cfg , Mode And Image End****** */

    /* ****** ONNX Infer Start****** */
    std::shared_ptr<BaseFeatureMatchOnnxRunner> FeatureMatcher;
    if (cfg.isEndtoEnd)
    {
        FeatureMatcher = std::make_shared<LightGlueOnnxRunner>();
        FeatureMatcher->InitOrtEnv(cfg);
        FeatureMatcher->SetMatchThresh(cfg.threshold);
    }
    else
    {
        FeatureMatcher = std::make_shared<LightGlueDecoupleOnnxRunner>();
        FeatureMatcher->InitOrtEnv(cfg);
        FeatureMatcher->SetMatchThresh(cfg.threshold);
    }

    auto iter1 = image_matlist1.begin();
    auto iter2 = image_matlist2.begin();
    std::string mode = cfg.isEndtoEnd ? "LightGlueOnnxRunner" : "LightGlueDecoupleOnnxRunner";

    for (; iter1 != image_matlist1.end() && iter2 != image_matlist2.end();
         ++iter1, ++iter2)
    {
        auto startTime = std::chrono::steady_clock::now();
        auto kpts_result = FeatureMatcher->InferenceImage(cfg, *iter1, *iter2);
        auto endTime = std::chrono::steady_clock::now();

        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "[INFO] " << mode << " single picture whole process takes time : "
                  << elapsedTime << " ms" << std::endl;
        // if (cfg.viz)
        // {
        //     std::vector<cv::Mat> imagesPair = {*iter1, *iter2};
        //     std::vector<std::string> titlePair = {"srcImage", "destImage"};
        //     cv::Mat figure = plotImages(imagesPair, kpts_result, titlePair);
        // }
        // auto kpts = FeatureMatcher->GetKeypointsResult();
        if (cfg.viz)
        {
            const auto& pts0 = kpts_result.first;
            const auto& pts1 = kpts_result.second;
            int   numKp0 = static_cast<int>(pts0.size());


        // 2) Point2f → KeyPoint 변환
            std::vector<cv::KeyPoint> kp0, kp1;
            kp0.reserve(numKp0);  kp1.reserve(numKp0);
            for (int i = 0; i < numKp0; ++i) {
                kp0.emplace_back(pts0[i], 1.0f);
                // pts1.x 는 상대 이미지이므로, 오른쪽으로 이동 없이 그대로 사용하나
                kp1.emplace_back(pts1[i], 1.0f);
            }

            // 3) DMatch 리스트 생성 (i↔i 대응)
            std::vector<cv::DMatch> matches;
            matches.reserve(numKp0);
            for (int i = 0; i < numKp0; ++i) {
                matches.emplace_back(i, i, 0.0f);
            }

            // 4) drawMatches 로 시각화
            cv::Mat matchesImg;
            cv::drawMatches(
                *iter1, kp0,
                *iter2, kp1,
                matches,
                matchesImg,
                cv::Scalar::all(-1),  // 매칭선 색
                cv::Scalar::all(-1),  // 키포인트 색
                std::vector<char>(),
                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
            );


            // 4) 파일로 저장
            cv::imwrite("light_glue_point_light_glue_good_matches.jpg", matchesImg);
            cv::imshow("light_glue_point_light_glue_good_matches", matchesImg);
            cv::waitKey();
        }

    }
    /* ****** ONNX Infer End****** */

    if (cfg.isEndtoEnd)
    {
        printf("[INFO] End2End model inference %zu images mean cost %.2f ms", image_filelist1.size(), (FeatureMatcher->GetTimer() / image_filelist1.size()));
    }
    else
    {
        printf("[INFO] Decouple model extractor inference %zu images mean cost %.2f ms , matcher mean cost %.2f", image_filelist1.size(),
               (FeatureMatcher->GetTimer("extractor") / image_filelist1.size()), (FeatureMatcher->GetTimer() / image_filelist1.size()));
    }

    return EXIT_SUCCESS;
}
