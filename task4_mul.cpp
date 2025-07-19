#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono> // For timing

using namespace std;

#define MAX_ROWS 200
#define NUM_FEATURES 6

struct Dataset { double** X; double* Y; int num_examples; };
struct ThreadData { int start, end; Dataset dataset; double b, w[NUM_FEATURES], *dw[NUM_FEATURES], *db; };

mutex dw_mutexes[NUM_FEATURES];
mutex db_mutex;

void compute_derivatives(ThreadData data) {
    double local_dw[NUM_FEATURES] = {0.0}, local_db = 0.0;
    for (int i = data.start; i < data.end; i++) {
        double h_x = data.b;
        for (int j = 0; j < NUM_FEATURES; j++) h_x += data.w[j] * data.dataset.X[i][j];
        double error = h_x - data.dataset.Y[i];
        for (int j = 0; j < NUM_FEATURES; j++) local_dw[j] += error * data.dataset.X[i][j];
        local_db += error;
    }
    for (int j = 0; j < NUM_FEATURES; j++) { lock_guard<mutex> lock(dw_mutexes[j]); *data.dw[j] += local_dw[j]; }
    { lock_guard<mutex> lock(db_mutex); *data.db += local_db; }
}

double compute_cost(Dataset dataset, double b, double w[]) {
    double total_cost = 0.0;
    for (int i = 0; i < dataset.num_examples; i++) {
        double h_x = b;
        for (int j = 0; j < NUM_FEATURES; j++) h_x += w[j] * dataset.X[i][j];
        total_cost += (h_x - dataset.Y[i]) * (h_x - dataset.Y[i]);
    }
    return total_cost / (2 * dataset.num_examples);
}

int main() {
    ifstream file("lab6_dataset_original.csv");
    if (!file.is_open()) { cerr << "Error opening file!" << endl; return -1; }

    Dataset dataset;
    dataset.X = new double*[MAX_ROWS];
    dataset.Y = new double[MAX_ROWS];
    for (int i = 0; i < MAX_ROWS; i++) dataset.X[i] = new double[NUM_FEATURES];

    string line;
    int row = 0;
    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0;
        if (row > 0) {
           while (getline(ss, cell, ',')) {
                if (col == 3) { try { dataset.X[row - 1][0] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 4) { try { dataset.X[row - 1][1] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 7) { try { dataset.X[row - 1][2] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 8) { try { dataset.X[row - 1][3] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 9) { try { dataset.X[row - 1][4] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 10) { try { dataset.X[row - 1][5] = stod(cell); } catch (...) { cerr << "err"; return -1; } }
                else if (col == 11) { try { dataset.Y[row - 1] = stod(cell); } catch (...) { cerr << "err"; return -1; } break; }
                col++;
           }
        }
        row++;
    }
    file.close();
    dataset.num_examples = row - 1;

    double b = 1.0, w[NUM_FEATURES] = {0.5};
    double alpha = 0.001;
    int num_epochs = 100; // Hyperparameter: Set the number of epochs
    vector<double> training_losses; // Store training loss for each epoch

        cout << "Initial Weights: b=" << b << ", w="; for (int i = 0; i < NUM_FEATURES; i++) cout << w[i] << " "; cout << endl;

    // Single-threaded training loop
    auto start_time = chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double dw[NUM_FEATURES] = {0.0}, db = 0.0;

        //Compute cost and save to training_losses
        double cost = compute_cost(dataset, b, w);
        training_losses.push_back(cost);

                //Single threaded derivative calcluation to find appropriate training
                ThreadData data = {0, dataset.num_examples, dataset, b, {w[0],w[1],w[2],w[3],w[4],w[5]}, {&dw[0],&dw[1],&dw[2],&dw[3],&dw[4],&dw[5]}, &db}; // Initialize w and dw inside the struct
                compute_derivatives(data);


        // Update weights (Gradient Descent)
        for (int i = 0; i < NUM_FEATURES; i++) w[i] -= alpha * (dw[i] / dataset.num_examples);
        b -= alpha * (db / dataset.num_examples);

        cout << "Epoch " << epoch + 1 << ": Cost=" << cost << endl;
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        cout << "Training completed in "<< duration.count() << " milliseconds" << endl;


    cout << "Final Weights: b=" << b << ", w="; for (int i = 0; i < NUM_FEATURES; i++) cout << w[i] << " "; cout << endl;

       //Outputting training loss to training_loss.txt
    ofstream loss_file("training_loss.txt");
    if (loss_file.is_open()) {
        for (size_t i = 0; i < training_losses.size(); i++) {
            loss_file << i + 1 << " " << training_losses[i] << endl;
        }
        loss_file.close();
        cout << "Training loss data written to training_loss.txt" << endl;
    } else {
        cerr << "Unable to open training_loss.txt for writing." << endl;
    }

    for (int i = 0; i < MAX_ROWS; i++) delete[] dataset.X[i];
    delete[] dataset.X;
    delete[] dataset.Y;

    return 0;
}