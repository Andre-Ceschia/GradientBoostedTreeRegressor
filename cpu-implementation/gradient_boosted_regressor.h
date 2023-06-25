#pragma once

#include "decision_tree_regressor.h"
#include <fstream>
#include <math.h>
#include <string>



class GradientBoostedRegressor{
private:
   DecisionTreeRegressor** tree_arr; 
   int trees;
   int val_trees;
   float inital_value;
   float lr;

   static void populate_loss_arr(float** features, float* output, GradientBoostedRegressor* model, float* loss_grad_arr, int start_index, int end_index, int rows);

   static void get_columns_rows(const char* filename, int* columns, int* rows);

    static void populate_array(const char* filename, float** features, float* output, int columns);
  

public:

    static void load_csv(const char* filename, float*** features_ptr, float** output_ptr, int* rows_ptr, int* column_ptr);
    void fit(const char* filename);


    GradientBoostedRegressor(int iterators, int max_depth, int min_samples_split, float learning_rate=0.1, const char* loss_function="mse");
    ~GradientBoostedRegressor();

    void fit(float** features, float* output, int rows, int columns);
    float predict(float* data);
};