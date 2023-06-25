#include <iostream>
#include <unordered_map>
#include <stdio.h>

#include "decision_tree_regressor.h"

float* DecisionTreeRegressor::flatten_matrix(float** matrix, int rows, int columns) {
    float* vector = new float[rows * columns];
    int vector_index = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            vector[vector_index] = matrix[i][j];
            vector_index++;
        }
    }

    return vector;
}

void DecisionTreeRegressor::cuda_fit(float* features, float* output, int rows, int columns) {
    float inital_loss = get_mean_sqaured_error(output, rows);

    cuda_build_tree(root, features, output, inital_loss, rows, columns, 1, true, m_max_depth, m_min_samples_split);

}

node* DecisionTreeRegressor::get_root_pointer() {
    return root;
}

void DecisionTreeRegressor::calculate_metrics(float* features, float* output, int rows, int columns, int index, float threshold, int* left_size_ptr, int* right_size_ptr, float* left_loss_ptr, float* right_loss_ptr) {

    int left_size = 0;
    int right_size = 0;

    float left_sum = 0;
    float right_sum = 0;

    float left_sqaured_sum = 0;
    float right_sqaured_sum = 0;

    float left_loss = 0;
    float right_loss = 0;

    for (int i = 0; i < rows; i++) {
        float data_point = features[i * columns + index];
        float output_point = output[i];

        if (data_point <= threshold) {
            left_size++;
            left_sum += output_point;
            left_sqaured_sum += output_point * output_point;
        }
        else {
            right_size++;
            right_sum += output_point;
            right_sqaured_sum += output_point * output_point;
        }

    }
    
    if (left_size != 0) {
        left_loss = (-1 / float(left_size * left_size)) * (left_sum * left_sum) + (1 / float(left_size)) * left_sqaured_sum;
    }

    if (right_size != 0) {
		right_loss = (-1 / float(right_size * right_size)) * (right_sum * right_sum) + (1 / float(right_size)) * right_sqaured_sum;

    }

    *left_size_ptr = left_size;
    *right_size_ptr = right_size;

    *left_loss_ptr = left_loss;
    *right_loss_ptr = right_loss;



}

__global__ void cuda_fill_array(float* features, float* output, SplitInfo* unique_val_arr, float* info_gain_arr, int rows, int columns, int uni_size, float parent_loss) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= uni_size) {
        return;
    }

    float threshold = unique_val_arr[index].threshold;
    int column = unique_val_arr[index].column;

    int left_size = 0;
    int right_size = 0;

    float left_sum = 0;
    float right_sum = 0;

    float left_sqaured_sum = 0;
    float right_sqaured_sum = 0;

    float left_loss = 0;
    float right_loss = 0;

    float weighted_left_loss = 0;
    float weighted_right_loss = 0;

    for (int i = 0; i < rows; i++) {
        float data_point = features[i * columns + column];
        float output_point = output[i];

        if (data_point <= threshold) {
            left_size++;
            left_sum += output_point;
            left_sqaured_sum += output_point * output_point;
        }
        else {
            right_size++;
            right_sum += output_point;
            right_sqaured_sum += output_point * output_point;
        }

    }

    if (left_size != 0) {
        left_loss = (-1 / float(left_size * left_size)) * (left_sum * left_sum) + (1 / float(left_size)) * left_sqaured_sum;
        weighted_left_loss = left_loss * (float(left_size) / float(rows));
    }

    if (right_size != 0) {
        right_loss = (-1 / float(right_size * right_size)) * (right_sum * right_sum) + (1 / float(right_size)) * right_sqaured_sum;

        weighted_right_loss = right_loss * (float(right_size) / rows);
    }

    float info_gain = parent_loss - weighted_left_loss - weighted_right_loss;

    info_gain_arr[index] = info_gain;


}

void DecisionTreeRegressor::cuda_build_tree(node* curr_node, float* features, float* output, float parent_loss, int rows, int columns, int curr_depth, bool root, int max_depth, int min_samples_split) {
    int total_size = 0;

    SplitInfo* split_info_arr = new SplitInfo[rows * columns];


    std::unordered_map<float, int> map;

    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            float value = features[j * columns + i];

            if (map.find(value) == map.end()) {
				map[value] = 0;
                SplitInfo temp(i, value);
                split_info_arr[total_size] = temp;

                total_size++;

            }
        }
        map.clear();
    }

    float* info_gain_arr = new float[total_size];

    float* cuda_features, * cuda_output, * cuda_info_gain;



    SplitInfo* cuda_split_info;

    // note features has ot be a 1d array indexed like arr[row * num_rows + column] 

    cudaMalloc((void**)&cuda_features, sizeof(float) * rows * columns);
    cudaMalloc((void**)&cuda_output, sizeof(float) * rows);
    cudaMalloc((void**)&cuda_info_gain, sizeof(float) * total_size);
    cudaMalloc((void**)&cuda_split_info, sizeof(SplitInfo) * rows * columns);

    cudaMemcpy(cuda_features, features, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_output, output, sizeof(float) * rows, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_split_info, split_info_arr, sizeof(SplitInfo) * rows * columns, cudaMemcpyHostToDevice);

    int cuda_threads, cuda_blocks;

    if (total_size <= 1024) {
        cuda_threads = total_size;
        cuda_blocks = 1;
    }
    else {
        cuda_threads = 1024;
        cuda_blocks = ceil((float)total_size / (float)1024);
    }

    cuda_fill_array <<<cuda_blocks, cuda_threads >>> (cuda_features, cuda_output, cuda_split_info, cuda_info_gain, rows, columns, total_size, parent_loss);


    cudaDeviceSynchronize();

    cudaMemcpy(info_gain_arr, cuda_info_gain, sizeof(float) * total_size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_features);
    cudaFree(cuda_output);
    cudaFree(cuda_info_gain);
    cudaFree(cuda_split_info);

    float max_info_gain = std::numeric_limits<float>::infinity() * -1;
    int best_index;

    for (int i = 0; i < total_size; i++) {
        if (info_gain_arr[i] > max_info_gain) {
            max_info_gain = info_gain_arr[i];
            best_index = i;
        }
    }

    float threshold = split_info_arr[best_index].threshold;
    float column_index = split_info_arr[best_index].column;

    delete[] split_info_arr;
    delete[] info_gain_arr;

    int left_size, right_size;
    float left_loss, right_loss;

    calculate_metrics(features, output, rows, columns, column_index, threshold, &left_size, &right_size, &left_loss, &right_loss);


    float* left_features, * left_output, * right_features, * right_output;

    cuda_get_split(features, output, column_index, threshold, rows, columns, &left_features, &left_output, &right_features, &right_output, left_size, right_size);

    curr_node->index = column_index;
    curr_node->threshold = threshold;

    node* left_node;
    node* right_node;

    cudaMallocManaged(&left_node, sizeof(node));
    cudaMallocManaged(&right_node, sizeof(node));

    curr_node->left = left_node;
    curr_node->right = right_node;

    if (left_size <= min_samples_split || curr_depth+1 == max_depth) {
        left_node->value = calculate_mean(left_output, left_size);
        delete[] left_features;
        delete[] left_output;

    }
    else {
        cuda_build_tree(left_node, left_features, left_output, left_loss, left_size, columns, curr_depth + 1, false, max_depth, min_samples_split);
    }

    if (right_size <= min_samples_split || curr_depth+1 == max_depth) {
        right_node->value = calculate_mean(right_output, right_size);
        delete[] right_features;
        delete[] right_output;
    }
    else {
        cuda_build_tree(right_node, right_features, right_output, right_loss, right_size, columns, curr_depth + 1, false, max_depth, min_samples_split);

    }

    if (!root) {
        delete[] features;
        delete[] output;
    }

}

void DecisionTreeRegressor::cuda_get_split(float* features, float* output, int index, float threshold, int rows, int columns, float** left_features, float** left_output, float** right_features, float** right_output, int left_size, int right_size) {

    float* temp_left_features = new float[left_size * columns];
    float* temp_left_output = new float[left_size];

    float* temp_right_features = new float[right_size * columns];
    float* temp_right_output = new float[right_size];

    int left_index = 0;
    int right_index = 0;

    for (int i = 0; i < rows; i++) {
        float value = features[i * columns + index];

        if (value <= threshold) {

            for (int j = 0; j < columns; j++) {
                temp_left_features[left_index * columns + j] = features[i * columns + j];
            }
            temp_left_output[left_index] = output[i];
            left_index++;

        }
        else {
            for (int j = 0; j < columns; j++) {
                temp_right_features[right_index * columns + j] = features[i * columns + j];
            }
            temp_right_output[right_index] = output[i];
            right_index++;

        }
    }

    *left_features = temp_left_features;
    *left_output = temp_left_output;

    *right_features = temp_right_features;
    *right_output = temp_right_output;

}


DecisionTreeRegressor::DecisionTreeRegressor(int max_depth, int min_samples_split, const char* loss_function) {
    cudaMallocManaged(&root, sizeof(node));

    m_max_depth = max_depth;
    m_min_samples_split = min_samples_split;

}
// this is so that the nodes arent deleted, has to be empty and not deleting the nodes, 
// no memory leak though, I am deleting them in the destructor of GradientBoostedRegressor
DecisionTreeRegressor::~DecisionTreeRegressor() {}

void DecisionTreeRegressor::delete_node(node* curr_node) {
    if (curr_node->left != nullptr) {
        delete_node(curr_node->left);
    }

    if (curr_node->right != nullptr) {
        delete_node(curr_node->right);
    }

    cudaFree(curr_node);

}

float DecisionTreeRegressor::recurisve_predict(float* data, node* curr_node) {
    if (curr_node->right == nullptr && curr_node->left == nullptr) {
        return curr_node->value;
    }

    if (data[curr_node->index] <= curr_node->threshold) {
        return recurisve_predict(data, curr_node->left);
    }
    else {
        return recurisve_predict(data, curr_node->right);
    }

}

float DecisionTreeRegressor::predict(float* data) {
    return recurisve_predict(data, root);
}

void DecisionTreeRegressor::predict(float** data, float* predicted_arr, int rows) {

    for (int i = 0; i < rows; i++) {
        predicted_arr[i] = recurisve_predict(data[i], root);
    }

}

float DecisionTreeRegressor::get_mean_sqaured_error(float* dataset, int rows) {

    float data_sum = 0;
    float data_squared_sum = 0;

    for (int i = 0; i < rows; i++) {
        data_sum += dataset[i];
        data_squared_sum += dataset[i] * dataset[i];
    }

    float loss = (-1 / float(rows * rows)) * (data_sum * data_sum) + (1 / float(rows)) * data_squared_sum;

    return loss;

}
float DecisionTreeRegressor::calculate_mean(float* data, int rows) {
    if (rows == 0) {
        return 0;
    }

    float total = 0;

    for (int i = 0; i < rows; i++) {
        total += data[i];
    }

    return total / (float)rows;
}
