//
// Created by wy on 2021/7/29.
//

#ifndef LINE_LINE_MATCH_H
#define LINE_LINE_MATCH_H

#include "iostream"
#include "vector"
#include "array"
#include "algorithm"
using namespace std;

class line_match {

public:
    vector<int> NW_match(vector<vector<pair<float, float>>> line_points1, vector<vector<pair<float, float>>> line_points2,
                         vector<vector<bool>> valid_points1, vector<vector<bool>> valid_points2,
                         vector<vector<array<float, 128>>> line_des1, vector<vector<array<float, 128>>> line_des2);
    vector<int> Faiss_match(vector<vector<pair<float, float>>> line_points1, vector<vector<pair<float, float>>> line_points2,
                         vector<vector<bool>> valid_points1, vector<vector<bool>> valid_points2,
                         vector<vector<array<float, 128>>> line_des1, vector<vector<array<float, 128>>> line_des2);
};


#endif //LINE_LINE_MATCH_H
