#include <iostream>
#include "decision_tree_regressor.h"

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth, int min_samples_split, const char* loss_function){
    root = new node;
    m_max_depth = max_depth;
    m_min_samples_split = min_samples_split;
}

DecisionTreeRegressor::~DecisionTreeRegressor(){
    delete_node(root);
}

void DecisionTreeRegressor::delete_node(node* curr_node){
    if (curr_node->left != nullptr){
        delete_node(curr_node->left);
    }

    if (curr_node->right != nullptr){
        delete_node(curr_node->right);
    }

    delete curr_node;

}

void DecisionTreeRegressor::validate_index(int* size_array, int &row_index, int &column_index, int last_row, int last_col){
    if (row_index >= last_row && column_index == last_col){
        row_index = last_row;
        return;
    }

    while (row_index >= size_array[column_index]){
        if (row_index >= last_row && column_index == last_col){
            row_index = last_row;
            return;
        }
        row_index -= size_array[column_index];
        column_index++;
    }
}

void DecisionTreeRegressor::build_tree(node* curr_node, float** features, float* output, float parent_loss, int rows, int columns, int curr_depth, bool root, int max_depth, int min_samples_split, int threads){
    float** unique_vals = new float*[columns];
    int* size_arr = new int[columns];
    int total_size = 0;


    std::unordered_map<float, int> map;

    for (int i = 0; i < columns; i++){
        unique_vals[i] = new float[rows];
        size_arr[i] = 0;

        for (int j = 0; j < rows; j++){
            float value = features[j][i];
            if (map.find(value) == map.end()){
                map[value] = 0;

                unique_vals[i][size_arr[i]] = value;
                size_arr[i]++;
                total_size++;
            }
        }
        map.clear();
    }

    int split = ceil(double(total_size) / double(threads));

    int row_start_index = 0;
    int column_start_index = 0;

    int row_end_index = split+1;
    int column_end_index = 0;

    validate_index(size_arr, row_end_index, column_end_index, size_arr[columns-1], columns-1);

    float left_loss_arr[threads], right_loss_arr[threads];
    int left_size_arr[threads], right_size_arr[threads];
    

    float info_gain_arr[threads];

    float threshold_arr[threads];
    int index_arr[threads];

    std::thread thread_arr[threads];

    for (int i = 0; i < threads; i++){
        thread_arr[i] = std::thread(thread_fill_array, features, output, unique_vals, size_arr, row_start_index, column_start_index, row_end_index, column_end_index, rows, parent_loss, &left_size_arr[i], &right_size_arr[i], &left_loss_arr[i], &right_loss_arr[i], &info_gain_arr[i], &threshold_arr[i], &index_arr[i]);
        row_start_index += split;
        validate_index(size_arr, row_start_index, column_start_index, size_arr[columns-1], columns-1);
        row_end_index += split;
        validate_index(size_arr, row_end_index, column_end_index, size_arr[columns-1], columns-1);
    }

    for (int i = 0; i < threads; i++){
        thread_arr[i].join();
    }

    for (int i = 0; i < columns; i++){
        delete[] unique_vals[i];
    }

    delete[] unique_vals;
    delete[] size_arr;



    float max_info_gain = std::numeric_limits<float>::infinity() * -1;

    float best_left_loss, best_right_loss;
    int best_left_size, best_right_size;
    int best_index;
    float best_threshold;

    for (int i = 0; i < threads; i++){
        if (info_gain_arr[i] > max_info_gain){
            max_info_gain = info_gain_arr[i];

            best_left_loss = left_loss_arr[i];
            best_right_loss = right_loss_arr[i];

            best_left_size = left_size_arr[i];
            best_right_size = right_size_arr[i];

            best_threshold = threshold_arr[i];
            // THIS INDEX IS HOLDING THREAD NUMBER NOT COLUMN NUMBER 
            best_index = index_arr[i];
        }

    }

    float** node_left_features;
    float** node_right_features;

    float* node_left_output;
    float* node_right_output;

    bool left_thread_exist = false;
    bool right_thread_exist = false;

    node* right_node = new node;
    node* left_node = new node;

    curr_node->index = best_index;
    curr_node->threshold = best_threshold;
    std::thread left_thread;
    std::thread right_thread;


    get_split(features, output, best_index, best_threshold, rows, &node_left_features, &node_left_output, &node_right_features, &node_right_output, best_left_size, best_right_size);

    if (best_left_size <= min_samples_split || curr_depth+1 == max_depth){
        left_node->value = calcualte_mean(node_left_output, best_left_size);
    } else {
        left_thread = std::thread(build_tree, left_node, node_left_features, node_left_output, best_left_loss, best_left_size, columns, curr_depth+1, false, max_depth, min_samples_split, threads); 

        left_thread_exist = true;
    }

    if (best_right_size <= min_samples_split || curr_depth+1 == max_depth){
        right_node->value = calcualte_mean(node_right_output, best_right_size);
    } else {
        right_thread = std::thread(build_tree, right_node, node_right_features, node_right_output, best_right_loss, best_right_size, columns, curr_depth+1, false, max_depth, min_samples_split, threads);

        right_thread_exist = true;
    }

    if (left_thread_exist){

        left_thread.join();
    } else {
        delete[] node_left_features;
        delete[] node_left_output;
    }

    if (right_thread_exist){

        right_thread.join();
    } else {
        delete[] node_right_features;
        delete[] node_right_output;
    }

    curr_node->left = left_node;
    curr_node->right = right_node;

    if (!root){
        delete[] features;    
        delete[] output;
    }

   
}

void DecisionTreeRegressor::fit(float** features, float* output, int rows, int columns){
    float inital_loss = get_mean_sqaured_error(output, rows);
    int threads = std::thread::hardware_concurrency();

    build_tree(root, features, output, inital_loss, rows, columns, 1, true, m_max_depth, m_min_samples_split, threads);

}

float DecisionTreeRegressor::recurisve_predict(float* data, node* curr_node){
    if (curr_node->right == nullptr && curr_node->left == nullptr){
        return curr_node->value;
    }

    if (data[curr_node->index] <= curr_node->threshold){
        return recurisve_predict(data, curr_node->left);
    } else{
        return recurisve_predict(data, curr_node->right);
    }

}

void DecisionTreeRegressor::thread_fill_array(float** features, float* output, float** unique_val_arr, int* size_arr, int row_start_index, int column_start_index, int row_end_index, int column_end_index, int rows, float parent_loss, int* left_size_ptr, int* right_size_ptr, float* left_loss_ptr, float* right_loss_ptr, float* info_gain_ptr, float* threshold_ptr, int* index_ptr){
    float max_info_gain = std::numeric_limits<float>::infinity() * -1;

    float best_left_loss, best_right_loss;
    int best_left_size, best_right_size;
    float best_threshold;
    int best_index;

    float left_loss, right_loss;
    int left_size, right_size;
    float info_gain;

    float weighted_left_loss, weighted_right_loss;

    while (row_start_index < row_end_index || column_start_index < column_end_index){
        if (column_start_index > column_end_index){
            break;
        }
        float threshold = unique_val_arr[column_start_index][row_start_index];

        get_mean_sqaured_error_split(features, output, column_start_index, threshold, rows, &left_loss, &left_size, &right_loss, &right_size);

        if (left_size != 0){
            weighted_left_loss = (float(left_size) / float(rows)) * left_loss;
        } else {
            weighted_left_loss = 0;
        }

        if (right_size != 0){
            weighted_right_loss = (float(right_size) / float(rows)) * right_loss;
        } else {
            weighted_right_loss = 0;
        }

        info_gain = parent_loss - weighted_left_loss - weighted_right_loss;

        if (info_gain > max_info_gain){
            max_info_gain = info_gain;
            best_left_loss = left_loss;
            best_right_loss = right_loss;

            best_left_size = left_size;
            best_right_size = right_size;
            best_threshold = threshold;
            best_index = column_start_index;
        }

        row_start_index++;

        if (row_start_index == size_arr[column_start_index]){
            row_start_index = 0;
            column_start_index++;
        }

    }

    *info_gain_ptr = max_info_gain;

    *left_loss_ptr = best_left_loss;
    *right_loss_ptr = best_right_loss;

    *left_size_ptr = best_left_size;
    *right_size_ptr = best_right_size;

    *threshold_ptr = best_threshold;
    *index_ptr = best_index;

}
float DecisionTreeRegressor::predict(float* data){
    return recurisve_predict(data, root);
}

void DecisionTreeRegressor::predict(float** data, float* predicted_arr, int rows){

    for (int i = 0; i < rows; i++){
        predicted_arr[i] = recurisve_predict(data[i], root);
    }

}

float DecisionTreeRegressor::get_mean_sqaured_error(float* dataset, int rows){

    float data_sum = 0;
    float data_squared_sum = 0;

    for (int i = 0; i < rows; i++){
        data_sum += dataset[i];
        data_squared_sum += dataset[i] * dataset[i];
    }

    float loss = (-1 / float(rows * rows)) * (data_sum * data_sum) + (1 / float(rows)) * data_squared_sum;

    return loss;

}

void DecisionTreeRegressor::get_mean_sqaured_error_split(float** features, float* output, int index, float threshold, int rows, float* left_loss, int* left_size, float* right_loss, int* right_size){
    int temp_left_size = 0;
    int temp_right_size = 0;

    float left_sum = 0;
    float right_sum = 0;

    float left_sqaured_sum = 0;
    float right_sqaured_sum = 0;

    float temp_left_loss = 0;
    float temp_right_loss = 0;

    for (int i = 0; i < rows; i++){
        float data_point = features[i][index];
        float output_point = output[i];

        if (data_point <= threshold){
            temp_left_size++;
            left_sum += output_point;
            left_sqaured_sum += output_point * output_point;
        } else {
            temp_right_size++;
            right_sum += output_point;
            right_sqaured_sum += output_point * output_point;
        }

    }
    if (left_size != 0 ){
        temp_left_loss = (-1 / float(temp_left_size * temp_left_size)) * (left_sum * left_sum) + (1 / float(temp_left_size)) * left_sqaured_sum;
    }

    if (right_size != 0){
        temp_right_loss = (-1 / float(temp_right_size * temp_right_size)) * (right_sum * right_sum) + (1 / float(temp_right_size)) * right_sqaured_sum;
    }

    *left_loss = temp_left_loss;
    *right_loss = temp_right_loss;

    *left_size = temp_left_size;
    *right_size = temp_right_size;
}


float DecisionTreeRegressor::calcualte_mean(float* data, int rows){
    if (rows == 0){
        return 0;
    }

    float total = 0;

    for (int i = 0; i < rows; i++){
        total += data[i];
    }

    return total / (float)rows;
}

// THESE ARRAYS MUST BE DESTROYED BY THE USER, THEY ARE ON THE HEAP
void DecisionTreeRegressor::get_split(float** features, float* output, int index, float threshold, int rows, float*** left_features, float** left_output, float*** right_features, float** right_output, int left_size, int right_size){

    // dont have to initliaiton an array in each row bc it is already intilaiztin in features and I am just movinv pointers
    float** temp_left_features = new float*[left_size];
    float* temp_left_output = new float[left_size];

    float** temp_right_features = new float*[right_size];
    float* temp_right_output = new float[right_size];

    int left_index = 0;
    int right_index = 0;

    for (int i = 0; i < rows; i++){
        float value = features[i][index];
        
        if (value <= threshold){
            temp_left_features[left_index] = features[i];
            temp_left_output[left_index] = output[i];
            left_index++;

        } else {
            temp_right_features[right_index] = features[i];
            temp_right_output[right_index] = output[i];
            right_index++;
        }
    }

    

    *left_features = temp_left_features;
    *left_output = temp_left_output;
    
    *right_features = temp_right_features;
    *right_output = temp_right_output;

}

