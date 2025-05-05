#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace std::chrono;

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

    total_wedges = 0;
    long long total_butterflies = 0;

    // Convert V_to_U to vector for indexed access
    vector<pair<int, vector<int>>> v_entries(V_to_U.begin(), V_to_U.end());
    int v_node_count = v_entries.size();

    #pragma omp parallel
    {
        long long thread_wedges = 0;
        long long thread_butterflies = 0;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < v_node_count; ++i) {
            const auto& v_entry = v_entries[i];
            const int v = v_entry.first;
            const auto& u_neighbors = v_entry.second;
            size_t degree = u_neighbors.size();

            if (degree >= 2) {
                thread_wedges += degree * (degree - 1) / 2;

                for (size_t j = 0; j < u_neighbors.size(); ++j) {
                    int u1 = u_neighbors[j];
                    const auto& u1_neighbors = U_to_V[u1]; // Avoid repeated lookup

                    for (size_t k = j + 1; k < u_neighbors.size(); ++k) {
                        int u2 = u_neighbors[k];

                        // Check if u1 and u2 share another V-node (i.e., form a butterfly)
                        const auto& u2_neighbors = U_to_V[u2];

                        vector<int> intersection;
                        set_intersection(u1_neighbors.begin(), u1_neighbors.end(),
                                         u2_neighbors.begin(), u2_neighbors.end(),
                                         back_inserter(intersection));

                        for (int shared_v : intersection) {
                            if (shared_v != v) {
                                thread_butterflies++;
                            }
                        }
                    }
                }
            }

            if (i % 1000 == 0 && omp_get_thread_num() == 0) {
                cout << "Processed " << i << "/" << v_node_count << " V-nodes" << endl;
            }
        }

        #pragma omp atomic
        total_wedges += thread_wedges;

        #pragma omp atomic
        total_butterflies += thread_butterflies;
    }

    return total_butterflies / 2;  // Each butterfly counted twice
}

int main() {
    string inputFile = "large_graph2.txt";  // Your dataset file

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

