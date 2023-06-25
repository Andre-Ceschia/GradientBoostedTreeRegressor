# GPU Based Implementation
I have developed an implementation of the Gradient Boosted Tree Regressor algorithm in C++ that utilizes the concurrency offered by the user's Graphical Processing Unit (GPU) to decrease the time it takes to train models with said algorithim. I utilized Nvidia's Compute Unified Device Architre (CUDA) API to achieve this. 

## CPU Based Implementation  

In addition to my GPU-based algorithm, I have also developed an implementation of said algorithm that utilizes the user's CPU for those who do not have access to computers equipped with a GPU. To expedite the training of models with the CPU implementation of the Gradient Boosted Tree Regressor, I utilized the C++ threading API to perform mathematical operations concurrently.

## Loss Function 

The loss function that I decided to utilize is **Mean Squared Error**. This function was utilized for the individual decision trees to calculate the maximum information gain of each split in the dataset, and its gradient was utilized for the boosting portion of the algorithm. 

## Improved Implementation Of Info Gain Algorithm
One of the most crucial and time-consuming steps in fitting a Gradient Boosted Regressor model is calculating the maximum information gain that each possible split in each decision tree can result in. I was able to mathematically derive an improved version of this algorithm that resulted in a nearly 45% decrease in training time for the CPU-based model by reducing the number of iterations through the dataset needed to calculate this value. The derivation of this formula can be found in this repository [here](derivation_of_improved_info_gain.pdf). Note: this change had a minuscule effect on the GPU-based algorithm due to the amount of parallelism used for this step. 

## Usage
```c++
// This program fits a model with the data provided and calculates the mean absolute error associated with the validation set

#include <iostream>
#include <math.h>

#include "gradient_boosted_regressor.h"


int main(){
    // This line initilizes the model with 100 iterators, a max depth of 9 for each decision tree, and the minimum samples for a leaf to be formed to be 20.
    GradientBoostedRegressor model(100, 9, 20);

    // This line trains the model with the data in training.csv
    model.fit("training.csv");

    // This chunk of code loads in the data in validaition.csv assuming the last column is the output

    // Note: these arrays are on the heap so they sould be freed by the user after use
    float** features;
    float* output;

    int rows, columns;

    GradientBoostedRegressor::load_csv("validation.csv", &features, &output, &rows, &columns);


    // This chunk of code is calculating and printing the mean absoulte error of the model on the validation dataset
    float error = 0;

    for (int i = 0; i < rows; i++){
        error += abs(model.predict(features[i]));
    }

    error /= float(rows);

    std::cout << "The Mean Absolute Error of the model on the validation dataset is " << error << std::endl;

    // This chunk of code is responsibile for the de-allocation of the memory produced by the GradientBoostedRegressor::load_csv function
    delete[] output;

    for (int i = 0; i < rows; i++){
        delete[] features[i];
    }

    delete[] features;
}

```