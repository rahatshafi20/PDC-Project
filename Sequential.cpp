#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Simplified graph structure using adjacency lists
unordered_map<int, vector<int>> U_to_V;
unordered_map<int, vector<int>> V_to_U;

void readEdgeList(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    int u, v;

    cout << "Reading graph..." << endl;

    while (getline(infile, line)) {
        istringstream iss(line);
        if (!(iss >> u >> v)) continue;

        U_to_V[u].push_back(v);
        V_to_U[v].push_back(u);
    }

    infile.close();

    // Remove duplicate edges
    for (auto& pair : U_to_V) {
        sort(pair.second.begin(), pair.second.end());
        pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());
    }

    for (auto& pair : V_to_U) {
        sort(pair.second.begin(), pair.second.end());
        pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());
    }

    cout << "Graph loaded: " << U_to_V.size() << " U-nodes, "
        << V_to_U.size() << " V-nodes" << endl;
}

long long countButterflies(long long& total_wedges) {
    cout << "Counting butterflies..." << endl;

    long long total_butterflies = 0;
    total_wedges = 0;
    int progress = 0;
    int v_node_count = V_to_U.size();

    for (const auto& v_entry : V_to_U) {
        const auto& u_neighbors = v_entry.second;
        long long degree = u_neighbors.size();

        // Only process V-nodes with at least 2 neighbors
        if (degree >= 2) {
            total_wedges += degree * (degree - 1) / 2;  // Count wedges for this V-node

            // Count butterflies directly from wedges
            for (size_t i = 0; i < u_neighbors.size(); ++i) {
                for (size_t j = i + 1; j < u_neighbors.size(); ++j) {
                    int u1 = u_neighbors[i];
                    int u2 = u_neighbors[j];

                    // Check if this edge forms a wedge
                   if (find(U_to_V[u1].begin(), U_to_V[u1].end(), v_entry.first) != U_to_V[u1].end() &&
    find(U_to_V[u2].begin(), U_to_V[u2].end(), v_entry.first) != U_to_V[u2].end() &&
    find(U_to_V[u1].begin(), U_to_V[u1].end(), u2) != U_to_V[u1].end()) {
    total_butterflies++;
}
                }
            }
        }

        if (++progress % 1000 == 0) {
            cout << "Processed " << progress << "/" << v_node_count << " V-nodes" << endl;
        }
    }

    return total_butterflies;
}

int main() {
    string inputFile = "large_graph2.txt";  // Replace with your dataset

    auto start_total = high_resolution_clock::now();

    readEdgeList(inputFile);
    long long total_wedges = 0;
    auto start_count = high_resolution_clock::now();
    long long butterflies = countButterflies(total_wedges);
    auto end_count = high_resolution_clock::now();

    auto end_total = high_resolution_clock::now();
    duration<double> count_time = end_count - start_count;
    duration<double> total_time = end_total - start_total;

    cout << "Total butterflies in bipartite graph: " << butterflies << endl;
    cout << "Total wedges in bipartite graph: " << total_wedges << endl;
    cout << "Counting time: " << count_time.count() << " seconds" << endl;
    cout << "Total execution time (including file loading): " << total_time.count() << " seconds" << endl;

    return 0;
}

