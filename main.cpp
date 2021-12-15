#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <filesystem>

at::Tensor inferOneImage(std::string imageFullPath)
{
  // Read images using opencv========================================================================
  cv::Mat cvimg = cv::imread(imageFullPath);
  cv::Mat resizedImg;

  if (cvimg.empty())
  {
    std::cout << "Could not open or find the image: " << imageFullPath << std::endl;
  }

  cv::cvtColor(cvimg, cvimg, CV_BGR2RGB);
  cv::resize(cvimg, resizedImg, cv::Size(100, 75), 0, 0, CV_INTER_AREA);
  // cv::cvtColor(resizedImg, resizedImg, CV_RGB2BGR);
  // cv::imwrite("/home/sm/Documents/work/data/hrpTestData/cppCVImg.png", resizedImg);

  resizedImg.convertTo(resizedImg, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(resizedImg.data, {resizedImg.rows, resizedImg.cols, 3});
  input_tensor = input_tensor.permute({2, 0, 1});
  input_tensor.unsqueeze_(0);

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::string filename = "/home/sm/Documents/work/dl/cpp-pytorch/libtorch_Modified_banded_w_grand_best_achieved_acc_old.pt";
  torch::jit::script::Module module = torch::jit::load(filename);
  module.eval();

  // // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor.to(at::kCPU));

  // // Execute the model and turn its output into a tensor.
  at::Tensor modelOut = module.forward(inputs).toTensor();
  at::Tensor finalOut = torch::softmax(modelOut, 1);

  return finalOut;
}

int main()
{
  // Get image paths================================================================================
  //  std::string dataPath = "/home/sm/Documents/work/data/inferData/CommonCarotidArtery/unwrapped/";
  std::string dataPath = "/home/sm/Documents/work/data/hrpTestData/";
  std::string dataOutPath = "/home/sm/Documents/work/dl/cpp-pytorch/";
  std::string ext = ".png";
  std::vector<std::string> unwrappedImages;

  for (auto &img : std::filesystem::recursive_directory_iterator(dataPath))
  {
    // std::string imgPath = img.path().filename().u8string();
    std::string imgPath = img.path().u8string();
    // std::string fullPath = dataPath + imgPath;

    if ((imgPath.find(ext) != std::string::npos) &&
        (imgPath.find("ct") == std::string::npos) &&
        (imgPath.find("Histology") == std::string::npos) &&
        (imgPath.find("donut") == std::string::npos))
    {
      // unwrappedImages.push_back(dataPath + imgPath);
      unwrappedImages.push_back(imgPath);
    }
  }
  std::sort(unwrappedImages.begin(), unwrappedImages.end());

  // for (auto i : unwrappedImages)
  //   std::cout << i << std::endl;
  // std::cout << unwrappedImages.at(0) << std::endl;
  // File to write out to
  std::ofstream csvFile;
  csvFile.open(dataOutPath + "outCppCV.csv");

  for (auto i : unwrappedImages)
  {
    // std::cout << "Inferring unwrapped image: " << i << std::endl;
    at::Tensor output = inferOneImage(i);
    cv::Mat outArr(3, 1, CV_32F, output.data_ptr<float>());
    std::string outVal0(std::to_string(outArr.at<float>(0)));
    std::string outVal1(std::to_string(outArr.at<float>(1)));
    std::string outVal2(std::to_string(outArr.at<float>(2)));

    csvFile << i + "," + outVal0 + "," + outVal1 + "," + outVal2 + "\n";
  }
  csvFile.close();
  return 0;
}