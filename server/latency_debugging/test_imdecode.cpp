#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>    // For wall-clock timing
#include <ctime>     // For CPU timing
#include <vector>

int main() {
    // Load the image from disk into a byte vector (simulates image loading from a compressed source)
    std::string imagePath = "qifeid01.jpg";
    std::vector<uchar> buffer;
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!\n";
        return -1;
    }

    // Encode the image into a buffer (simulate compressed data)
    cv::imencode(".jpg", image, buffer);

    // Perform imdecode() (decoding the compressed image buffer)
    int trial_count = 1000;
    double wallClockTimes[1000];
    double cpuTimes[1000];
    cv::Mat decoded = cv::Mat::zeros(1920, 1080, CV_8UC3);
    for (int i = 0; i < trial_count; i++) {
        // Wall-clock time measurement using chrono
        auto wallClockStart = std::chrono::high_resolution_clock::now();

        // CPU time measurement using clock()
        std::clock_t cpuStart = std::clock();

        cv::imdecode(buffer, cv::IMREAD_COLOR, &decoded);
        if (decoded.empty()) {
            std::cerr << "Decoding failed!\n";
        }

        // Wall-clock time measurement after imdecode()
        auto wallClockEnd = std::chrono::high_resolution_clock::now();

        // CPU time measurement after imdecode()
        std::clock_t cpuEnd = std::clock();

        // Calculate wall-clock time
        std::chrono::duration<double> wallClockDuration = wallClockEnd - wallClockStart;
        wallClockTimes[i] = wallClockDuration.count();

        // Calculate CPU time
        double cpuTimeDuration = double(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        cpuTimes[i] = cpuTimeDuration;
    }

    std::cout << "\nWall-clock time: \n";
    for (int i = 0; i < trial_count; i++) {
        std::cout << wallClockTimes[i] << ", ";
    }

    std::cout << "\nCPU time: \n";
    for (int i = 0; i < trial_count; i++) {
        std::cout << cpuTimes[i] << ", ";
    }

    return 0;
}
