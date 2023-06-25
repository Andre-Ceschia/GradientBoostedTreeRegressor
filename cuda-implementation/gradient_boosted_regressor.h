#pragma once

#include "decision_tree_regressor.h"
#include <fstream>
#include <math.h>
#include <string>


__global__ void fill_loss_arr(float* features, float* output, int rows, int columns, float* loss_grad_arr, float lr, float initial_value, node** node_arr, int val_trees);
__device__ float cuda_node_predict(float* features, int row, int columns, node* curr_node);

class GradientBoostedRegressor {
private:
    node** node_arr;
    int trees;
    int val_trees;
    float inital_value;
    float lr;
    int m_min_samples_split;
    int m_max_depth;

    static float calculate_mean(float* values, int rows);
    static void get_columns_rows(const char* filename, int* columns, int* rows);

    static void populate_array(const char* filename, float** features, float* output, int columns);
    


public:
    GradientBoostedRegressor(int iterators, int max_depth, int min_samples_split, float learning_rate = 0.1);
    ~GradientBoostedRegressor();

    static float inner_predict(float* data, float initial_value, float lr, node** node_arr, int val_trees);

    static void load_csv(const char* filename, float*** features_ptr, float** output_ptr, int* rows_ptr, int* column_ptr);
    void fit(float** features, float* output, int rows, int columns);
    void fit(const char* filename);
    float predict(float* data);
};
