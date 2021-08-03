//
// Created by wy on 2021/7/29.
//

#ifndef LINE_LINE_EXTRACTOR_H
#define LINE_LINE_EXTRACTOR_H
#include "vector"
#include "array"
#include "utility"
#include "iostream"
#include "cmath"
#include "algorithm"

using namespace std;

class line_extractor {
    vector<pair<int, int>> junctions_post(vector<vector<vector<float>>>&junctions);
    vector<vector<float>> heatmap_post(vector<vector<vector<float>>>&heatmap);
public:
    vector<pair<pair<int, int>, pair<int, int>>> line_detector(vector<vector<vector<float>>>&junc, vector<vector<vector<float>>>&heat);
    explicit line_extractor(float detect_thresh, int min, int sample);
    vector<vector<array<float, 128>>> line_descriptor(vector<pair<pair<int, int>,
            pair<int, int>>> line_segment, vector<vector<vector<float>>>desc,
            vector<vector<pair<float, float>>>& line_points, vector<vector<bool>>& valid_points);
};



#endif //LINE_LINE_EXTRACTOR_H
