//
// Created by wy on 2021/7/29.
//

#include "line_extractor.h"

using namespace std;

int H, W;
float detect_thresh = 0.4; // 原始为 0.25
int min_dist_pts = 8;
int num_samples = 5;
vector<pair<int, int>> line_extractor::junctions_post(vector<vector<vector<float>>>&junctions){
    size_t c = junctions.size(), h = junctions[0].size(), w = junctions[0][0].size();
    cout << "c: " << c << " w: " << w << " h: " << h << endl;
    H = (int)h  * 8, W = (int)w * 8;
    // step1. 对 65 个通道做 softmax
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < c; k++)
            {
                junctions[k][i][j] = exp(junctions[k][i][j]);
                sum += junctions[k][i][j];
            }
            for(int k = 0; k < c; k++)
            {
                junctions[k][i][j] /= sum;
            }
        }
    }
//    cout << "junctions feature map after softmax test: " << endl;
//    cout << "junctions[0][0][1] : " << junctions[0][0][1] << endl;
//    cout << "junctions[0][1][0] : " << junctions[0][1][0] << endl;
//    cout << "junctions[1][0][0] : " << junctions[1][0][0] << endl; // ok

    // step2. pixel_shuffle 还原成 (w * 8 , h * 8)
    vector<pair<int, int>> idx; // 辅助数组
    for(int i = 0; i < 8; i++)
    {
        for(int j = 0; j < 8; j++)
        {
            idx.emplace_back(i, j);
        }
    }
    vector<vector<float>> junc_pred_np(h * 8, vector<float>(w * 8));
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            int x = i * 8, y = j * 8;
            for(int id = 0; id < 64; id++)
            {
                int dx = idx[id].first, dy = idx[id].second;
                junc_pred_np[x + dx][y + dy] = junctions[id][i][j];
            }
        }
    }
//    cout << "junc_pred_np after pixel_shuffle test: " << endl;
//    cout << "junc_pred_np[0][8] : " << junc_pred_np[0][8] << endl;
//    cout << "junc_pred_np[0][9] : " << junc_pred_np[1][9] << endl; // ok

    // Step3. NMS
    // Step3.1 找到大于阈值的点
    float prob_thresh = 0.01;
    vector<pair<int, int>> points;
    // 这里就先用宽度方向为 x 高度方向为 y 和图片坐标系是不一样的
    for(int i = 0; i < W; i++)
    {
        for(int j = 0; j < H; j++)
        {
            if(junc_pred_np[j][i] >= prob_thresh)
            {
                points.emplace_back(i, j);
            }
        }
    }// 源代码是 HW format 是反过来的

    // Step3.2 得到对应的置信度
    vector<pair<float, pair<int, int>>> prob_score(points.size());
    for(int i = 0; i < points.size(); i++)
    {
        int x = points[i].first, y = points[i].second;
        prob_score[i].first = junc_pred_np[y][x];
        prob_score[i].second = points[i];
    }

    // Step3.3 执行 NMS super
    // 在图像四周各增加 8 行 8 列 0
    int pad = 8;
    vector<vector<int>> grid(W + 2 * pad, vector<int>(H + 2 * pad, 0));
    vector<vector<float>> inds(W + 2 * pad, vector<float>(H + 2 * pad, 0));
    // 根据置信度从大到小排序
    sort(prob_score.begin(), prob_score.end(), greater<pair<float, pair<int, int>>>());
    // 向网格里填数
    for(auto p : prob_score)
    {
        int x = p.second.first, y = p.second.second;
        grid[x + pad][y + pad] = 1;
        inds[x + pad][y + pad] = p.first;
    }
    // NMS
    int cnt = 0;
    for(auto p : prob_score)
    {
        int x = p.second.first + pad, y = p.second.second + pad;
        if(grid[x][y] == 1)
        {
            for(int m = x - pad; m < x + pad + 1; m++)
            {
                for(int n = y - pad; n < y + pad + 1; n++)
                {
                    grid[m][n] = 0;
                }
            }
            grid[x][y] = -1;
            cnt++;
        }
    }
    // 保存结果
    vector<pair<float , pair<int, int>>> keep_points(cnt);
    vector<pair<int, int>> junc_points;
    for(int i = 0; i < W + 2 * pad; i++)
    {
        for(int j = 0; j < H + 2 * pad; j++)
        {
            if(grid[i][j] == -1)
            {
                int x = i, y = j;
                float p = inds[x][y];
                keep_points.push_back({p, {x - pad, y - pad}});
            }
        }
    }
    // 只保留置信度最高的 300 个
    sort(keep_points.begin(), keep_points.end(), greater<pair<float , pair<int, int>>>());
    int np = keep_points.size();
    int topK = min(300, np);

    vector<vector<float>> junc_pred_np_nms(W, vector<float>(H, 0));
    for(int i = 0; i < topK; i++){
        int x = keep_points[i].second.first, y = keep_points[i].second.second;
        junc_pred_np_nms[x][y] = keep_points[i].first;
    }
    for(int i = 0; i < W; i++)
    {
        for(int j = 0; j < H; j++)
        {
            if(junc_pred_np_nms[i][j] > 0)
            {
                junc_points.push_back({i, j});
            }
        }
    }

    return junc_points; // WH  格式的 节点坐标集合
    // checked
}

// 对 heatmap 做后处理
vector<vector<float>> line_extractor::heatmap_post(vector<vector<vector<float>>>&heatmap)
{
    cout << heatmap.size() << heatmap[0].size() << heatmap[0][0].size();
    // step1. 对 2 个通道做 softmax
    vector<vector<float>> heatmap_2d(H, vector<float>(W));

    for(int i = 0; i < H; i ++)
    {
        for(int j = 0; j < W; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < 2; k++)
            {
                heatmap[k][i][j] = exp(heatmap[k][i][j]);
                sum += heatmap[k][i][j];
            }

            heatmap_2d[i][j] =  heatmap[1][i][j] / sum;
        }
    }
    return heatmap_2d;
    // checked
}

vector<pair<pair<int, int>, pair<int, int>>> line_extractor::line_detector(vector<vector<vector<float>>>&junc, vector<vector<vector<float>>>&heatmap)
{
    vector<pair<int, int>> junc_points = line_extractor::junctions_post(junc);
    auto heat = heatmap_post(heatmap);

    // Step 1.初始化
    // 源代码都是调用的 pytorch tensor 在 GPU 上跑的
    // 统计节点个数
    int num_junctions = (int)junc_points.size();
    // line_map_pred 300 x 300 全 0
    vector<vector<int>> line_map_pred(num_junctions, vector<int>(num_junctions, 0));
    // candidate_map 只有上三角为 1 为了 candidate_index_map 做准备的，没啥用， torch 可以直接生成
    // [0 1 1 .... 1]
    // [0 0 1 .... 1]
    // [...........0]
    vector<vector<int>> candidate_map(num_junctions, vector<int>(num_junctions, 0));
    for(int i = 0; i < num_junctions; i++)
    {
        for(int j = i + 1; j < num_junctions; j++)
        {
            candidate_map[i][j] = 1;
        }
    }
    // 生成对应的可能匹配 id （44850 x 2）
    vector<pair<int, int>> candidate_index_map;
    for(int i = 0; i < num_junctions; i++)
    {
        for(int j = i + 1; j < num_junctions; j++)
        {
            candidate_index_map.emplace_back(i, j);
        }
    }
    // 44858 对 junction 的起始终点坐标
    vector<pair<float, float>> candidate_junc_start(candidate_index_map.size());
    vector<pair<float, float>> candidate_junc_end(candidate_index_map.size());
    for(int i = 0; i < candidate_index_map.size(); i++)
    {
        // c++ 17
        auto x = candidate_index_map[i].first;
        auto y = candidate_index_map[i].second;
//        auto [x, y] = candidate_index_map[i];
        candidate_junc_start[i] = junc_points[x];
        candidate_junc_end[i] = junc_points[y];
    }
    // 采样 N x 64 从 1/64 2/64 ... 1
    vector<float> sampler(64, 0);
    float base = (float)1 / 64;
    for(int i = 1; i < 64; i++)
    {
        sampler[i] = sampler[i - 1] + base;
    }
    // 分别是 44850 x 64 的矩阵
    vector<vector<float>> cand_samples_h(candidate_index_map.size(), vector<float>(64));
    vector<vector<float>> cand_samples_w(candidate_index_map.size(), vector<float>(64));
    for(int i = 0; i < candidate_index_map.size(); i++)
    {
        auto siw = candidate_junc_start[i].first;
        auto sih = candidate_junc_start[i].second;
        auto eiw = candidate_junc_end[i].first;
        auto eih = candidate_junc_end[i].second;
//        auto [siw, sih] = candidate_junc_start[i];
//        auto [eiw, eih] = candidate_junc_end[i];
        for(int j = 0; j < 64; j++)
        {
            cand_samples_w[i][j] = siw * sampler[j] + eiw * (1 - sampler[j]);
            cand_samples_h[i][j] = sih * sampler[j] + eih * (1 - sampler[j]);
        }
    }
    // 这里源代码判断了一下 cand_samples_w、h 会不会越界的问题， 我觉得应该不会吧

    // 计算线段长度 并归一化处理
    vector<float> segments_length(candidate_index_map.size());
    for(int i = 0; i < candidate_index_map.size(); i++)
    {
        float dx = candidate_junc_start[i].first - candidate_junc_end[i].first;
        float dy = candidate_junc_start[i].second - candidate_junc_end[i].second;
        segments_length[i] = sqrt(dx * dx + dy * dy);
    }
    float base_HW = sqrt((float)H * H + (float)W * W);
    vector<float> normalized_seg_length(candidate_index_map.size());
    for(int i = 0; i < candidate_index_map.size(); i++)
    {
        normalized_seg_length[i] = segments_length[i] / base_HW;
    }

    // Step 2. 判断是不是直线
    // 一次操作不超过 10000 对点
    // 总的点对数 44850
    int num_cand = (int)cand_samples_h.size();
    int group_size = 10000;
    // 这里 就不分成一万一万的了， 可能需要更多的内存！！
//    if(num_cand > group_size)
//    {
//        int num_iter = ceil(num_cand / group_size);
//        for(int i = 0; i < num_iter; i++)
//        {
            vector<float> normalized_seg_length_;
            vector<vector<float>> cand_h_, cand_w_;
//            if(i != num_iter - 1)
//            {
//                auto begin = cand_samples_h.begin() + num_iter * group_size;
//                auto end = begin + group_size;
//                cand_h_ = vector<vector<float>> (begin, end);
//                begin = cand_samples_w.begin() + num_iter * group_size;
//                end = begin + group_size;
//                cand_w_ = vector<vector<float>> (begin, end);
//                auto begin2 = normalized_seg_length.begin() + num_iter * group_size;
//                auto end2 = begin2 + group_size;
//                normalized_seg_length_ = vector<float>(begin2, end2);
//            }
//            else
//            {
//                auto begin = cand_samples_h.begin() + num_iter * group_size;
                auto begin = cand_samples_h.begin();
                auto end = cand_samples_h.end();
                cand_h_ = vector<vector<float>> (begin, end);
//                begin = cand_samples_w.begin() + num_iter * group_size;
                begin = cand_samples_w.begin();
                end = cand_samples_w.end();
                cand_w_ = vector<vector<float>> (begin, end);
//                auto begin2 = normalized_seg_length.begin() + num_iter * group_size;
                auto begin2 = normalized_seg_length.begin();
                auto end2 = normalized_seg_length.end();
                normalized_seg_length_ = vector<float>(begin2, end2);
//            }
            // 上面操作就是为了每一次得到不超过 10000 条 直线 cand_h_ cand_w_
            // Detection by local maximum search 部分

            int lambda_radius = 2;
            vector<vector<float>> dist_thresh(normalized_seg_length_.size(), vector<float>(64));
            for(int j = 0; j < normalized_seg_length_.size(); j++)
            {
                dist_thresh[j] = vector<float>(64, (float)0.5 * sqrt((float)2) + normalized_seg_length_[j]);
            }
            // N x 64 x 2 的 坐标
            vector<vector<pair<int, int>>> cand_points_round(cand_h_.size(), vector<pair<int, int>>(64));
            vector<vector<pair<float, float>>> cand_points(cand_h_.size(), vector<pair<float, float>>(64));
            for(int j = 0; j < cand_h_.size(); j++)
            {
                for(int k = 0; k < 64; k++)
                {
                    float x = cand_h_[j][k];
                    float y = cand_w_[j][k];
                    cand_points[j][k] = {x, y};
                    cand_points_round[j][k] = {(int)x, (int)y};
                }
            }
            int local_patch_radius = 3;
            // 源代码注释写的 9 x 9 实际是 7 x 7 在以 （3, 3）为圆心半径为 3 的圆内 的整数格点
            vector<vector<int>> patch_mask(2 * local_patch_radius + 1, vector<int>(2 * local_patch_radius + 1, 0));
            vector<pair<int, int>> patch_points;
            for(int x = 0; x < 2 * local_patch_radius + 1; x++)
            {
                for(int y = 0; y < 2 * local_patch_radius + 1; y++)
                {
                    if((x - local_patch_radius) * (x - local_patch_radius)
                     + (y - local_patch_radius) * (y - local_patch_radius) <= local_patch_radius)
                        patch_points.emplace_back(x - 3, y - 3); // 平移到原点为中心
                    // 应该有 29 个点 满足要求
                }
            }
            // 构造局部特征掩码 这里会生成一个四维的张量
            // 10000 对点 x 均匀采样的 64 个采样点 x 每个采样点周围的 29 个点 x xy 坐标（2）
            // N x 64 x 29 x 2
            // 这个感觉不是很适合用 vector
            vector<vector<vector<pair<int, int>>>> patch_points_shifted(num_cand,
                        vector<vector<pair<int, int>>>(64, vector<pair<int, int>>(29)));
            vector<vector<vector<bool>>> patch_dist_mask(num_cand,
                                vector<vector<bool>>(64, vector<bool>(29, false)));
            vector<vector<vector<int>>> points_H(num_cand,vector<vector<int>>(64, vector<int>(29)));
            vector<vector<vector<int>>> points_W(num_cand,vector<vector<int>>(64, vector<int>(29)));
            for(int a = 0; a < num_cand; a++)
            {
                for(int b = 0; b < 64; b++)
                {
                    int x = cand_points_round[a][b].first;
                    int y = cand_points_round[a][b].second;
                    float xc = cand_points[a][b].first;
                    float yc = cand_points[a][b].second;
                    for(int c = 0; c < 29; c++)
                    {
                        auto dx = patch_points[c].first;
                        auto dy = patch_points[c].second;
//                        auto [dx, dy] = patch_points[c];
                        if(x + dx < 0 || y + dy < 0 || x + dx >= H || y + dy > W)
                            continue;
//                        points_H[a][b][c] = min(0, max(H, x + dx));
//                        points_W[a][b][c] = min(0, max(W, y + dy));
                        points_H[a][b][c] = x + dx;
                        points_W[a][b][c] = y + dy;
                        // 这里源代码又分了两个变量，我就合一起了，等同于 points
                        patch_points_shifted[a][b][c] = {points_H[a][b][c], points_W[a][b][c]};
                        float  dist = (xc - x - dx ) * (xc - x - dx ) + (yc - y - dy ) * (yc - y - dy);
                        if(dist < dist_thresh[a][b] * dist_thresh[a][b])
                            patch_dist_mask[a][b][c] = true;
                        // 应该很多是 false dist_thresh 很多都比 1 小
                    }
                }
            }
            // 得到对应点的 heatmap 值 mask 为 fasle 的地方置 0
            // 这个可以和上面的循环合并
            vector<vector<vector<float>>> sampled_feat(num_cand,vector<vector<float>>(64, vector<float>(29)));
            // 29 个点里 heat 最大值
            vector<vector<float>> sampled_feat_lmax(num_cand, vector<float>(64));
            for(int a = 0; a < num_cand; a++)
            {
                for(int b = 0; b < 64; b++)
                {
                    float maxx = 0;
                    for(int c = 0; c < 29; c++)
                    {
                        if(patch_dist_mask[a][b][c])
                        {
                            int x = patch_points_shifted[a][b][c].first;
                            int y = patch_points_shifted[a][b][c].second;
                            sampled_feat[a][b][c] = heat[x][y];
                        } else
                            sampled_feat[a][b][c] = 0;
                        maxx = max(maxx, sampled_feat[a][b][c]);
                    }
                    sampled_feat_lmax[a][b] = maxx;
                }
            }
            // 最终得到 sampled_feat_lmax
//        }
//    }

    vector<bool> detection_results(num_cand, false);
    int cnt_line = 0;
    for(int i = 0; i < num_cand; i++)
    {
        float sum = 0;
        for(int j = 0; j < 64; j++)
        {
            sum += sampled_feat_lmax[i][j];
        }
        if(sum > detect_thresh * 64)
        {
            detection_results[i] = true;
            cnt_line++;
        }
    }
    vector<pair<int, int>> detected_junc_indexes;
    for(int i = 0; i < num_cand; i++)
    {
        if(detection_results[i])
        {
            detected_junc_indexes.push_back(candidate_index_map[i]);
            auto x = candidate_index_map[i].first;
            auto y = candidate_index_map[i].second;
//            auto [x, y] = candidate_index_map[i];
            line_map_pred[x][y] = 1;
            line_map_pred[y][x] = 1;
        }
    }

    vector<pair<pair<int, int>, pair<int, int>>> line_segment(cnt_line);
    for(int i = 0; i < cnt_line; i++)
    {
        auto l1 = detected_junc_indexes[i].first;
        auto l2 = detected_junc_indexes[i].second;
        auto x1 = junc_points[l1].first;
        auto y1 = junc_points[l1].second;
        auto x2 = junc_points[l2].first;
        auto y2 = junc_points[l2].second;
//        auto [l1, l2] = detected_junc_indexes[i];
//        auto [x1, y1] = junc_points[l1];
//        auto [x2, y2] = junc_points[l2];
        line_segment[i] = {{x1, y1}, {x2, y2}};
    }

    return line_segment;
}

vector<vector<array<float, 128>>> line_extractor::line_descriptor(
        vector<pair<pair<int, int>, pair<int, int>>> line_segment, vector<vector<vector<float>>>desc,
        vector<vector<pair<float, float>>>& line_points, vector<vector<bool>>& valid_points)
{
    int grid_size = 8;
    int h = desc[0].size(), w = desc[0][0].size();
    pair<float, float> img_size = {w * grid_size, h * grid_size};

    // 两个图片分别的线段数目
    int num_lines = (int)line_segment.size();
    // 计算每条线段长度
    vector<float> line_lengths(num_lines);

    for(int i = 0; i < num_lines; i++)
    {
        float dx = (float)line_segment[i].first.first - (float)line_segment[i].second.first;
        float dy = (float)line_segment[i].first.second - (float)line_segment[i].second.second;
        line_lengths[i] = sqrt(dx * dx + dy * dy);
    }

    // 可以作为传入参数 最小的间隔距离 和 采样数

    // 上限 num_samples 下限 2
    vector<int> num_samples_lst(num_lines);
    for(int i = 0; i < num_lines; i++)
    {
        num_samples_lst[i] = max(2, min(num_samples, (int)line_lengths[i] / min_dist_pts));
    }

    // 采样后的点
//    vector<vector<pair<float, float>>> line_points(num_lines, vector<pair<float, float>>(num_samples, {0, 0}));

    // 标记是否是用来填充的 即不足 num_samples 个点 的地方为 False 对应 linepoints 置 0
//    vector<vector<bool>> valid_points(num_lines, vector<bool>(num_samples, false));

    for(int i = 2; i <= num_samples; i++)
    {
        for(int j = 0; j < num_lines; j++)
        {
            bool cur_mask = num_samples_lst[j] == i;
            if(cur_mask)
            {
                float dx = (float)line_segment[j].second.first - (float)line_segment[j].first.first;
                float dy = (float)line_segment[j].second.second - (float)line_segment[j].first.second;
                auto x = line_segment[j].first.first;
                auto y = line_segment[j].first.second;
//                auto [x, y] = line_segment[j].first;
                for(int k = 0; k < i; k++)
                {
                    line_points[j][k] = {dx * k / i + x, dy * k / i + y};
                    valid_points[j][k] = true;
                }
            }
        }
    }
    // sample_line_points 结束 得到了 line_points valid_points
    for(auto &p : line_points)
    {
        for (auto &q : p)
        {
//            q.first = q.first * 2 / img_size.first;
//            q.second = q.second * 2 / img_size.second;
            q.first /= 4; // 不做归一化了直接找到 desc 上对应位置
            q.second /= 4;
        }
    }

    // keypoints_to_grid 结束 line_points 归一化到 [-1, 1]
    // 之后源代码又调用了 pyTorch 双线性插值的 api grid_sample
    // 手写一个效率不太高的吧 毕竟 128 维的向量还是做矩阵运算更快
    vector<vector<array<float, 128>>> des(num_lines, vector<array<float,128>>(num_samples));
    int dW1 = w, dH1 = h;
    for(int i = 0; i < num_lines; i++)
    {
        for(int j = 0; j < num_samples; j++)
        {
            auto x = line_points[i][j].first;
            auto y = line_points[i][j].second;
//            auto [x, y] = line_points[i][j];
            // 首先找到 采样点在 description 中的 真实位置
//            x = x * (float)dW1, y = y * (float)dH1;
            // 找到它四周的坐标
            int x1 = floor(x), y1 = floor(y);
            int x2 = x1 + 1, y2 = y1 + 1; // 这里就不考虑越界的情况了
            // 计算 四个 特征向量的权重
            // [x1, y2(w3)         x2, y2(w4)]
            // [            x,y              ]
            // [x1, y1(w1)         x2, y1(w2)]
            // x2 > x > x1    y2 > y > y1
            float w1 = ((float)x2 - x) * ((float)y2 - y);
            float w2 = (x - (float)x1) * ((float)y2 - y);
            float w3 = ((float)x2 - x) * (y - (float)y1);
            float w4 = (x - (float)x1) * (y - (float)y1);
            // w1 + w2 + w3 + w4 = (x2 - x1) * (y2 - y1) = 1

            for(int f = 0; f < 128; f++)
            {
                x1 = max(0, min(H / 4 - 1, x1));
                x2 = max(0, min(H / 4 - 1, x2));
                y1 = max(0, min(W / 4 - 1, y1));
                y2 = max(0, min(W / 4 - 1, y2));
                des[i][j][f] = w1 * desc[f][x1][y1] + w2 * desc[f][x2][y1]
                        + w3 * desc[f][x1][y2] + w4 * desc[f][x2][y2];
            }


        }
    }
    // 之后源代码又调用了 pyTorch 的 Normalize 函数
    // 对每个128维特征值取 L2 范数规范化
    // xi -> xi / sqrt(x1^2 + x2^2 + ... + x128^2)
    for(int i = 0; i < num_lines; i++)
    {
        for(int j = 0; j < num_samples; j++)
        {
            float denom = 0;
            for(auto d : des[i][j])
            {
                denom += d * d;
            }
            denom = sqrt(denom);
            for(auto &d : des[i][j])
            {
                d /= denom;
            }
        }
    }
    return des;
}

//vector<vector<array<float, 128>>> line_extractor::line_descriptor(
//        vector<pair<pair<int, int>, pair<int, int>>> line_segment, vector<vector<vector<float>>>desc,
//        vector<vector<pair<float, float>>>& line_points, vector<vector<bool>>& valid_points)
//{
//    // 线段数目
//    int num_lines = (int)line_segment.size();
//    // 计算每条线段长度
//    vector<float> line_lengths(num_lines);
//
//    for(int i = 0; i < num_lines; i++)
//    {
//        float dx = (float)line_segment[i].first.first - (float)line_segment[i].second.first;
//        float dy = (float)line_segment[i].first.second - (float)line_segment[i].second.second;
//        line_lengths[i] = sqrt(dx * dx + dy * dy);
//    }
//
//    // 可以作为传入参数 最小的间隔距离 和 采样数
//
//    // 上限 num_samples 下限 2
//    vector<int> num_samples_lst(num_lines);
//    for(int i = 0; i < num_lines; i++)
//    {
//        num_samples_lst[i] = max(2, min(num_samples, (int)line_lengths[i] / min_dist_pts));
//    }
//
//    for(int i = 2; i <= num_samples; i++)
//    {
//        for(int j = 0; j < num_lines; j++)
//        {
//            bool cur_mask = num_samples_lst[j] == i;
//            if(cur_mask)
//            {
//                float dx = (float)line_segment[j].second.first - (float)line_segment[j].first.first;
//                float dy = (float)line_segment[j].second.second - (float)line_segment[j].first.second;
////                auto [x, y] = line_segment[j].first;
//                auto x = line_segment[j].first.first;
//                auto y = line_segment[j].first.second;
//                for(int k = 0; k < i; k++)
//                {
//                    line_points[j][k] = {dx * (float)k + (float)x, dy * (float)k + (float)y};
//                    valid_points[j][k] = true;
//                }
//            }
//        }
//    }
//    // sample_line_points 结束 得到了 line_points valid_points
//}

line_extractor::line_extractor(float thresh, int min, int sample) {
    detect_thresh = thresh;
    min_dist_pts = min;
    num_samples = sample;
}