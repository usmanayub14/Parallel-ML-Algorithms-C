#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#define MAX_ROWS 200  // Maximum rows to read
#define NUM_FEATURES 6 // Number of input features

int main() {
    ifstream file("lab6_dataset_original.csv");
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return -1;
    }

    double* X[MAX_ROWS]; // Pointer array for input features
    double* Y = new double[MAX_ROWS]; // Pointer for target variable (CO2 Emissions)

    for (int i = 0; i < MAX_ROWS; i++)
        X[i] = new double[NUM_FEATURES];

    string line;
    int row = 0;

    while (getline(file, line) && row < MAX_ROWS) {
        stringstream ss(line);
        string cell;
        int col = 0, feature_idx = 0;

        if (row > 0) {  // Skip header row
            while (getline(ss, cell, ',')) {
                if (col == 3) X[row - 1][feature_idx++] = stod(cell); // Engine Size
                if (col == 4) X[row - 1][feature_idx++] = stod(cell); // Cylinders
                if (col == 7) X[row - 1][feature_idx++] = stod(cell); // City Fuel
                if (col == 8) X[row - 1][feature_idx++] = stod(cell); // Hwy Fuel
                if (col == 9) X[row - 1][feature_idx++] = stod(cell); // Comb Fuel
                if (col == 10) X[row - 1][feature_idx++] = stod(cell); // Comb MPG
                if (col == 11) Y[row - 1] = stod(cell); // CO2 Emissions
                col++;
            }
        }
        row++;
    }
    file.close();

    // Print first 5 rows for verification
    cout << "First 5 rows of data:\n";
    for (int i = 0; i < min(row - 1, 5); i++) {
        for (int j = 0; j < NUM_FEATURES; j++)
            cout << X[i][j] << " ";
        cout << "| CO2: " << Y[i] << endl;
    }

    // Free allocated memory
    for (int i = 0; i < MAX_ROWS; i++)
        delete[] X[i];
    delete[] Y;

    return 0;
}
