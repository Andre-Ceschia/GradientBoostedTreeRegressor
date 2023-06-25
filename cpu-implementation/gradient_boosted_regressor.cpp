#include <iostream>

#include "gradient_boosted_regressor.h"

GradientBoostedRegressor::GradientBoostedRegressor(int iterators, int max_depth, int min_samples_split, float learning_rate, const char* function){
    tree_arr = new DecisionTreeRegressor*[iterators];

    for (int i = 0; i < iterators; i++) {
        tree_arr[i] = new DecisionTreeRegressor(max_depth, min_samples_split);

    }

    trees = iterators;
    lr = learning_rate;



}

GradientBoostedRegressor::~GradientBoostedRegressor(){
    for (int i = 0; i < trees; i++){
        delete tree_arr[i];

    }

    delete[] tree_arr;

}

void GradientBoostedRegressor::load_csv(const char* filename, float*** features_ptr, float** output_ptr, int* rows_ptr, int* column_ptr) {
    int rows, columns;
    get_columns_rows(filename, &columns, &rows);
    columns--;

    float** features = new float* [rows];
    float* output = new float[rows];

    for (int i = 0; i < rows; i++) {
        features[i] = new float[columns];
    }

    populate_array(filename, features, output, columns);

    *features_ptr = features;
    *output_ptr = output;
    *rows_ptr = rows;
    *column_ptr = columns;
}



float GradientBoostedRegressor::predict(float* data){
    float result = 0;
    result += inital_value;

    for (int i = 0; i < val_trees; i++){
        result -= (lr * tree_arr[i]->predict(data));
    }

    return result;

}

void GradientBoostedRegressor::get_columns_rows(const char* filename, int* columns, int* rows) {
    std::ifstream file(filename);
    std::string text;

    int temp_columns = 1;
    int temp_rows = 0;
    bool set = false;

    while (getline(file, text)) {

        if (!set) {
            for (int i = 0; i < text.length(); i++) {
                if (text[i] == ',') {
                    temp_columns++;
                }
            }
            set = true;
            continue;
        }
        temp_rows++;
    }

    *rows = temp_rows;
    *columns = temp_columns;

}

void GradientBoostedRegressor::populate_array(const char* filename, float** features, float* output, int columns) {
    std::ifstream file(filename);
    std::string buffer;

    bool set = false;

    int row_index = 0;
    while (getline(file, buffer)) {

        if (!set) {
            set = true;
            continue;
        }

        int char_index = 0;
        for (int i = 0; i < columns; i++) {
            std::string col_buf;

            while (buffer[char_index] != ',') {
                col_buf += buffer[char_index];
                char_index++;
            }

            features[row_index][i] = std::stof(col_buf);
            char_index++;
        }

        std::string output_buf;
        while (char_index < buffer.length()) {
            output_buf += buffer[char_index];
            char_index++;
        }
        output[row_index] = std::stof(output_buf);
        row_index++;

    }

    file.close();
}

void GradientBoostedRegressor::fit(const char* filename) {
    float** features;
    float* output;
    int rows, columns;

    load_csv(filename, &features, &output, &rows, &columns);

    fit(features, output, rows, columns);
    
}



void GradientBoostedRegressor::fit(float** features, float* output, int rows, int columns){
    val_trees = 0;
    inital_value = DecisionTreeRegressor::calcualte_mean(output, rows);
    int threads = std::thread::hardware_concurrency();

    float* loss_grad_arr = new float[rows];

    if (rows < threads){
        threads = rows;
    }

    int split = ceil(double(rows) / double(threads));

    
    std::thread thread_arr[threads];
    for (int i = 0; i < trees; i++){

        int curr_start = 0;
        int curr_end = split+1;

        for (int j = 0; j < threads; j++){

            if (curr_end > rows){
                curr_end = rows;
            }

            thread_arr[j] = std::thread(populate_loss_arr, features, output, this, loss_grad_arr, curr_start, curr_end, rows);
            curr_start += split;
            curr_end += split;
        }

        for (int j = 0; j < threads; j++){
            thread_arr[j].join();
        }

        tree_arr[val_trees]->fit(features, loss_grad_arr, rows, columns);

        val_trees++;

    }

    delete[] loss_grad_arr;
    

   
}

// not including end index
void GradientBoostedRegressor::populate_loss_arr(float** features, float* output, GradientBoostedRegressor* model, float* loss_grad_arr, int start_index, int end_index, int rows){

    while (start_index < end_index){

        loss_grad_arr[start_index] = -2 * (output[start_index] - model->predict(features[start_index]));
        start_index++;

    }

}
    
