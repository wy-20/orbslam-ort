//
// Created by wy on 2021/7/29.
//


#include "line_match.h"
#include <faiss/IndexFlat.h>

vector<int>
line_match::NW_match(vector<vector<pair<float, float>>> line_points1, vector<vector<pair<float, float>>> line_points2,
                     vector<vector<bool>> valid_points1, vector<vector<bool>> valid_points2,
                     vector<vector<array<float, 128>>> line_des1, vector<vector<array<float, 128>>> line_des2) {
    int num_lines1 = line_points1.size(), num_lines2 = line_points2.size(), num_samples = line_points1[0].size();
    vector<vector<vector<vector<float>>>> scores(num_lines1,
              vector<vector<vector<float>>>(num_lines2,
              vector<vector<float>>(num_samples, vector<float>(num_samples, -1))));

    for(int n1 = 0; n1 < num_lines1; n1++)
    {
        for(int n2 = 0; n2 < num_lines2; n2++)
        {
            for(int s1 = 0; s1 < num_samples; s1++)
            {
                if(!valid_points1[n1][s1])
                    continue;
                for(int s2 = 0; s2 < num_samples; s2++)
                {
                    if(!valid_points2[n2][s2])
                        continue;

                    float sum = 0.0f;
                    for(int f = 0; f < 128; f++)
                    {
                        sum += line_des1[n1][s1][f] * line_des2[n2][s2][f];
                    }
                    scores[n1][n2][s1][s2] = sum;
                }
            }
        }
    }
    vector<vector<pair<float, int>>> line_scores(num_lines1);
    for(int n1 = 0; n1 < num_lines1; n1++)
    {
        for(int n2 = 0; n2 < num_lines2; n2++)
        {
            vector<float> line_score1(num_samples, -100);
            vector<float> line_score2(num_samples, -100);
            for(int s1 = 0; s1 < num_samples; s1++)
            {
                for(int s2 = 0; s2 < num_samples; s2++)
                {
                    line_score1[s1] = max(line_score1[s1], scores[n1][n2][s1][s2]);
                    line_score2[s2] = max(line_score2[s2], scores[n1][n2][s1][s2]);
                }
            }
            int cnt1 = 0, cnt2 = 0;
            float sum1 = 0, sum2 = 0;
            for(auto s : line_score1)
            {
                if(s > 0)
                {
                    cnt1++;
                    sum1 += s;
                }
            }
            for(auto s : line_score2)
            {
                if(s > 0)
                {
                    cnt2++;
                    sum2 += s;
                }
            }
            if(cnt1 == 0 || cnt2 == 0)
                continue;
            float line_score = sum1 / (float)cnt1 / 2 + sum2 / (float)cnt2 / 2;
            line_scores[n1].push_back({line_score, n2});
        }
    }
    for(auto &l : line_scores)
    {
        sort(l.begin(), l.end(), greater<pair<float, int>>());
    }
    int topk = 10;
    vector<vector<int>> topk_lines(num_lines1, vector<int>(topk));
    for(int i = 0; i < num_lines1; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            topk_lines[i][j] = line_scores[i][j].second;
        }
    }
    vector<vector<vector<vector<float>>>> top_scores(num_lines1,
                   vector<vector<vector<float>>>(2 * topk,
                   vector<vector<float>>(num_samples,vector<float>(num_samples))));
    for(int i = 0; i < num_lines1; i++)
    {
        for(int j = 0; j < topk; j++)
        {
            int l2 = topk_lines[i][j];
            top_scores[i][j] = scores[i][l2];
        }
        for(int j = topk; j < topk * 2; j++)
        {
            for(int f1 = 0; f1 < num_samples; f1++)
            {
                for(int f2 = 0; f2 < num_samples; f2++)
                {
                    top_scores[i][j][f1][f2] =
                            top_scores[i][j - topk][num_samples - f1 - 1][num_samples - f2 - 1];
                }
            }
        }
    }
    // Step 4. Needleman-Wunsch algorithm
    float gap = 0.1;
    vector<vector<vector<vector<float>>>> nw_grid(num_lines1,
                vector<vector<vector<float>>>(2 * topk,
                vector<vector<float>>(num_samples + 1,
                vector<float>(num_samples + 1))));
    for(int l = 0; l < num_lines1; l ++)
    {
        for(int t = 0; t < 2 * topk; t++)
        {
            for(int i = 0; i < num_samples; i++)
            {
                for(int j = 0; j < num_samples; j++)
                {
                    nw_grid[l][t][i + 1][j + 1] = max(nw_grid[l][t][i][j + 1], max(
                            nw_grid[l][t][i + 1][j], nw_grid[l][t][i][j] + top_scores[l][t][i][j] - gap));
                }
            }
        }
    }
    vector<vector<float>> nw_scores(num_lines1, vector<float>(2 * topk));
    vector<int> matches(num_lines1);
    // 表示 第 i 条直线 匹配第 j 条直线的 第 m 个点（起点）， 第 n 个点（终点）
    vector<array<int, 3>> matches_points(num_lines1);
    for(int i = 0; i < num_lines1; i++)
    {
        float mmax = 0.0f;
        int l2 = 0;
        for(int j = 0; j < 2 * topk; j++)
        {
            nw_scores[i][j] = nw_grid[i][j][num_samples][num_samples];
            if(nw_scores[i][j] > mmax)
            {
                mmax = nw_scores[i][j];
                l2 = j;
            }
        }
        matches[i] = l2;
    }
    vector<int> match_lines(num_lines1);
    for(int i = 0; i < num_lines1; i++)
    {
        match_lines[i] = topk_lines[i][matches[i] % topk];
    }
    return match_lines;
}

vector<int> line_match::Faiss_match(vector<vector<pair<float, float>>> line_points1,
                                    vector<vector<pair<float, float>>> line_points2, vector<vector<bool>> valid_points1,
                                    vector<vector<bool>> valid_points2, vector<vector<array<float, 128>>> line_des1,
                                    vector<vector<array<float, 128>>> line_des2) {
    int num_samples = 15;
    int d = 128 * num_samples;
    int nb = (int)line_points1.size();
    int nq = (int)line_points2.size();

    vector<float> xb(d * nb);
    vector<float> xq(d * nq);
    int cnt1 = 0;
    for(int i = 0; i < line_points1.size(); i++)
    {
        for(int j = 0; j < num_samples; j++)
        {
            for(int k = 0; k < 128; k++)
            {
                xb[cnt1++] = line_des1[i][j][k];
            }
        }
    }
    cnt1 = 0;
    for(int i = 0; i < line_points2.size(); i++)
    {
        for(int j = 0; j < num_samples; j++)
        {
            for(int k = 0; k < 128; k++)
            {
                xq[cnt1++] = line_des2[i][j][k];
            }
        }
    }
    faiss::IndexFlat index(d);
    index.add(nq, xq.data());
    auto *I = new faiss::Index::idx_t[nb];
    auto *D = new float[nb];
    index.search(nb, xb.data(), 1, D, I);
    vector<int> matches(line_points1.size());
    for(int i = 0; i < line_points1.size(); i++)
    {
        matches[i] = (int)I[i];
    }
    return matches;
}
