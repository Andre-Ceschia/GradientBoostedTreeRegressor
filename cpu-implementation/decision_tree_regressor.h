#pragma once

#include <thread>
#include <math.h>
#include <unordered_map>

struct node{
    node(): value(0.0f), threshold(0.0f), index(0), left(nullptr), right(nullptr){}
    float value;
    float threshold;
    int index;
    node* left;
    node* right;
};

class DecisionTreeRegressor{
private:
    node* root;
    int m_max_depth;
    int m_min_samples_split;
    float (*loss_func)(float*, int);

    static void delete_node(node* curr_node);
    float recurisve_predict(float* data, node* curr_node);

    static void build_tree(node* curr_node, float** features, float* output, float parent_loss, int rows, int columns, int curr_depth, bool root, int max_depth, int min_samples_split, int threads);

    static void validate_index(int* size_array, int &row_index, int &column_index, int last_row, int last_col);

    static void get_split(float** features, float* output, int index, float threshold, int rows, float*** left_features, float** left_output, float*** right_features, float** right_output, int left_size, int right_size);
    static void get_mean_sqaured_error_split(float** features, float* output, int index, float threshold, int rows, float* left_loss, int* left_size, float* right_loss, int* right_size);

    static void thread_fill_array(float** features, float* output, float** unique_val_arr, int* size_arr, int row_start_index, int column_start_index, int row_end_index, int column_end_index, int rows, float parent_loss, int* left_size_ptr, int* right_size_ptr, float* left_loss_ptr, float* right_loss_ptr, float* info_gain_ptr, float* threshold_ptr, int* index_ptr);

public:
    double time;
    DecisionTreeRegressor(int max_depth, int min_samples_split, const char* loss_func="mse");
    ~DecisionTreeRegressor();

    void fit(float** features, float* output, int rows, int columns);

    float predict(float* data);
    void predict(float** data, float* predicted_arr, int rows);
    static float get_mean_sqaured_error(float* dataset, int rows);
    static float calcualte_mean(float* data, int rows);

};