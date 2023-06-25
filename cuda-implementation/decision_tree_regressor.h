#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct SplitInfo {
    int column;
    float threshold;

    SplitInfo(int temp_column, float temp_threshold) {
        column = temp_column;
        threshold = temp_threshold;
    }
    SplitInfo() {
        column = 0;
        threshold = 0;
    }
};



__global__ void cuda_fill_array(float* features, float* output, SplitInfo* unique_val_arr, float* info_gain_arr, int rows, int columns, int uni_size, float parent_loss);

struct node {
    node() : value(0.0f), threshold(0.0f), index(0), left(nullptr), right(nullptr) {}
    float value;
    float threshold;
    int index;
    node* left;
    node* right;
};

class DecisionTreeRegressor {
private:
    node* root;
    int m_max_depth;
    int m_min_samples_split;

    static void calculate_metrics(float* features, float* output, int rows, int columns, int index, float threshold, int* left_size_ptr, int* right_size_ptr, float* left_loss_ptr, float* right_loss_ptr);

public:
    DecisionTreeRegressor(int max_depth, int min_samples_split, const char* loss_func = "mse");
    ~DecisionTreeRegressor();


    static float* flatten_matrix(float** matrix, int rows, int columns);
    static void delete_node(node* curr_node);
    void cuda_fit(float* features, float* output, int rows, int columns);

    node* get_root_pointer();

    static void cuda_build_tree(node* curr_node, float* features, float* output, float parent_loss, int rows, int columns, int curr_depth, bool root, int max_depth, int min_samples_split);
    static void cuda_get_split(float* features, float* output, int index, float threshold, int rows, int columns, float** left_features, float** left_output, float** right_features, float** right_output, int left_size, int right_size);

    static float recurisve_predict(float* data, node* curr_node);
    float predict(float* data);
    void predict(float** data, float* predicted_arr, int rows);
    static float get_mean_sqaured_error(float* dataset, int rows);
    static float calculate_mean(float* data, int rows);

};
