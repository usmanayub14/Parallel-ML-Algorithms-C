#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

#define MAX_ROWS 200  // Max dataset rows to read
#define NUM_FEATURES 3 // Number of input features

// Struct for dataset
struct Dataset {
    double** X;
    double* Y;
    int num_examples; // Number of training examples
};

// Struct for thread data
struct ThreadData {
    int start;
    int end;
    Dataset dataset;
    double b;
    double w1;
    double w2;
    double w3;
};

// Global variables
double total_cost = 0.0;
mutex cost_mutex;

// Function to compute cost for a portion of the dataset
void compute_cost(ThreadData data) {
    double sub_cost = 0.0;
    for (int i = data.start; i < data.end; i++) {
        double h_x = data.b + data.w1 * data.dataset.X[i][0] + data.w2 * data.dataset.X[i][1] + data.w3 * data.dataset.X[i][2];
        sub_cost += (h_x - data.dataset.Y[i]) * (h_x - data.dataset.Y[i]);
    }
    lock_guard<mutex> lock(cost_mutex); // Acquire lock for thread-safe update
    total_cost += sub_cost;
}

int main() {
    ifstream file("lab6_dataset_original.csv");
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return -1;
    }

    // Allocate memory for dataset
    Dataset dataset;
    dataset.X = new double*[MAX_ROWS];
    dataset.Y = new double[MAX_ROWS];

    for (int i = 0; i < MAX_ROWS; i++) {
        dataset.X[i] = new double[NUM_FEATURES];
    }

    string line;
    int row = 0;

    // Read CSV data
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0, feature_idx = 0;

        if (row > 0) { // Skip header row
            while (getline(ss, cell, ',')) {
                if (col == 3) dataset.X[row - 1][feature_idx++] = stod(cell); // Engine Size
                if (col == 4) dataset.X[row - 1][feature_idx++] = stod(cell); // Cylinders
                if (col == 9) dataset.X[row - 1][feature_idx++] = stod(cell); // Comb Fuel
                if (col == 11) dataset.Y[row - 1] = stod(cell); // CO2 Emissions
                col++;
            }
        }
        row++;
    }
    file.close();

    dataset.num_examples = row - 1; // Actual number of training examples

    int num_threads; // Number of threads to use (changeable)
    cout << "Enter the number of threads: ";
    cin >> num_threads;

    // Initialize weights randomly
    double b = 1.0, w1 = 0.5, w2 = -0.3, w3 = 0.8;

    vector<thread> threads;

    // Calculate batch sizes and create thread data
    int base_batch_size = dataset.num_examples / num_threads;
    int remainder = dataset.num_examples % num_threads; // Handle uneven division

    // Create threads
    int start = 0;
    for (int i = 0; i < num_threads; i++) {
        int batch_size = base_batch_size + (i < remainder ? 1 : 0); // Add 1 to batch size for the first 'remainder' threads
        int end = start + batch_size;

        ThreadData data = {start, end, dataset, b, w1, w2, w3};
        threads.emplace_back(compute_cost, data);

        start = end;
    }

    // Wait for threads to finish
    for (auto& t : threads)
        t.join();

    // Compute final cost
    total_cost = total_cost / (2 * dataset.num_examples);
    cout << "Total cost with " << num_threads << " threads: " << total_cost << endl;

    // Free allocated memory
    for (int i = 0; i < MAX_ROWS; i++)
        delete[] dataset.X[i];
    delete[] dataset.X;
    delete[] dataset.Y;

    return 0;
}