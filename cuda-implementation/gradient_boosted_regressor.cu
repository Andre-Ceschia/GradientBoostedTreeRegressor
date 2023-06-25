#include "gradient_boosted_regressor.h"
#include "decision_tree_regressor.h"

GradientBoostedRegressor::GradientBoostedRegressor(int iterators, int max_depth, int min_samples_split, float learning_rate) {
    cudaMallocManaged(&node_arr, sizeof(node*) * iterators);

    m_max_depth = max_depth;
    m_min_samples_split = min_samples_split;
    trees = iterators;
    lr = learning_rate;

}

GradientBoostedRegressor::~GradientBoostedRegressor() {
    for (int i = 0; i < trees; i++) {
        DecisionTreeRegressor::delete_node(node_arr[i]);

    }

    cudaFree(node_arr);

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


float GradientBoostedRegressor::predict(float* data) {
    return inner_predict(data, inital_value, lr, node_arr, val_trees);
}

float node_predict(float* data, node* curr_node) {
    if (curr_node->left == nullptr && curr_node->right == nullptr) {
        return curr_node->value;
    }

    if (data[curr_node->index] <= curr_node->threshold) {
        return node_predict(data, curr_node->left);
    }
    else {
        return node_predict(data, curr_node->right);
    }
}

float GradientBoostedRegressor::inner_predict(float* data, float initial_value, float lr, node** node_arr, int val_trees) {
    float result = 0;
    result += initial_value;

    for (int i = 0; i < val_trees; i++) {
        float prediction = node_predict(data, node_arr[i]);

        result -= lr * prediction;
    }

    return result;

}

float GradientBoostedRegressor::calculate_mean(float* values, int rows) {
    float sum = 0;

    for (int i = 0; i < rows; i++) {
        sum += values[i];
    }

    if (sum != 0) {
        sum /= float(rows);
    }

    return sum;
}

__device__ float cuda_node_predict(float* features, int row, int columns, node* curr_node) {
    if (curr_node->left == nullptr && curr_node->right == nullptr) {
        return curr_node->value;
    }

    float value = features[columns * row + curr_node->index];

    if (value <= curr_node->threshold) {
        return cuda_node_predict(features, row, columns, curr_node->left);
    }
    else {
        return cuda_node_predict(features, row, columns, curr_node->right);
    }



}

__global__ void fill_loss_arr(float* features, float* output, int rows, int columns, float* loss_grad_arr, float lr, float initial_value, node** node_arr, int val_trees) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= rows) {
        return;
    }

    float prediction = 0;
    prediction += initial_value;

    for (int i = 0; i < val_trees; i++) {
        prediction -= lr * cuda_node_predict(features, index, columns, node_arr[i]);
    }


    loss_grad_arr[index] = -2 * (output[index] - prediction);

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

void GradientBoostedRegressor::fit(const char* filename) {
    float** features;
    float* output;
    int rows, columns;

    load_csv(filename, &features, &output, &rows, &columns);

    fit(features, output, rows, columns);
    
}

void GradientBoostedRegressor::fit(float** features, float* output, int rows, int columns) {
    float* flattened_features = DecisionTreeRegressor::flatten_matrix(features, rows, columns);

    val_trees = 0;
    inital_value = calculate_mean(output, rows);


    for (int i = 0; i < trees; i++) {

        float* cuda_loss_arr;
        float* cuda_features;
        float* cuda_output;

        cudaMalloc(&cuda_loss_arr, sizeof(float) * rows);
        cudaMalloc(&cuda_features, sizeof(float) * rows * columns);
        cudaMalloc(&cuda_output, sizeof(float) * rows);

        cudaMemcpy(cuda_features, flattened_features, sizeof(float) * rows * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_output, output, sizeof(float) * rows, cudaMemcpyHostToDevice);

        int cuda_threads, cuda_blocks;

        if (rows <= 1024) {
            cuda_threads = rows;
            cuda_blocks = 1;
        }
        else {
            cuda_threads = 1024;
            cuda_blocks = ceil(float(rows) / float(1024));
        }

        fill_loss_arr << <cuda_blocks, cuda_threads >> > (cuda_features, cuda_output, rows, columns, cuda_loss_arr, lr, inital_value, node_arr, val_trees);
        cudaDeviceSynchronize();

        float* loss_grad_arr = new float[rows];

        cudaMemcpy(loss_grad_arr, cuda_loss_arr, sizeof(float) * rows, cudaMemcpyDeviceToHost);

        cudaFree(cuda_features);
        cudaFree(cuda_output);
        cudaFree(cuda_loss_arr);

        DecisionTreeRegressor temp_tree(m_max_depth, m_min_samples_split);

        temp_tree.cuda_fit(flattened_features, loss_grad_arr, rows, columns);

        delete[] loss_grad_arr;

        node* root_node = temp_tree.get_root_pointer();

        node_arr[val_trees] = root_node;
        val_trees++;

    }

    delete[] flattened_features;



}
