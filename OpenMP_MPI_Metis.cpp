#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

unordered_map<int, vector<int>> U_to_V;
unordered_map<int, vector<int>> V_to_U;
vector<int> v_node_ids; // Stores V-node IDs in order

void readEdgeList(const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    int u, v;
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
        v_node_ids.push_back(pair.first);
    }
}

vector<int> readMetisPartition(const string& filename) {
    vector<int> partition;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open METIS partition file " << filename << endl;
        exit(1);
    }
    int label;
    while (infile >> label) {
        partition.push_back(label);
    }
    return partition;
}

pair<long long, long long> countButterflies(const vector<int>& assigned_v_nodes) {
    long long local_wedges = 0;
    long long local_butterflies = 0;

    #pragma omp parallel
    {
        long long thread_wedges = 0;
        long long thread_butterflies = 0;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < assigned_v_nodes.size(); ++i) {
            int v = assigned_v_nodes[i];
            const auto& u_neighbors = V_to_U[v];
            size_t degree = u_neighbors.size();

            if (degree >= 2) {
                thread_wedges += degree * (degree - 1) / 2;

                for (size_t j = 0; j < u_neighbors.size(); ++j) {
                    int u1 = u_neighbors[j];
                    const auto& u1_neighbors = U_to_V[u1];

                    for (size_t k = j + 1; k < u_neighbors.size(); ++k) {
                        int u2 = u_neighbors[k];
                        const auto& u2_neighbors = U_to_V[u2];

                        vector<int> intersection;
                        set_intersection(u1_neighbors.begin(), u1_neighbors.end(),
                                         u2_neighbors.begin(), u2_neighbors.end(),
                                         back_inserter(intersection));
                        for (int shared_v : intersection) {
                            if (shared_v != v) thread_butterflies++;
                        }
                    }
                }
            }
        }

        #pragma omp atomic
        local_wedges += thread_wedges;
        #pragma omp atomic
        local_butterflies += thread_butterflies;
    }

    return {local_wedges, local_butterflies / 2}; // Each butterfly counted twice
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0)
            cerr << "Usage: mpirun -n <num_procs> ./a.out <graph.txt> <graph.metis.N>" << endl;
        MPI_Finalize();
        return 1;
    }

    string inputGraph = argv[1];
    string metisFile = argv[2];

    auto start_total = high_resolution_clock::now();

    // All processes read graph and partition file (simpler than broadcasting large maps)
    readEdgeList(inputGraph);
    vector<int> metis_partition = readMetisPartition(metisFile);

    // Assign V-nodes to this MPI rank
    vector<int> assigned_v_nodes;
    for (size_t i = 0; i < v_node_ids.size(); ++i) {
        if (metis_partition[i] == world_rank)
            assigned_v_nodes.push_back(v_node_ids[i]);
    }

    auto start_count = high_resolution_clock::now();
    auto [local_wedges, local_butterflies] = countButterflies(assigned_v_nodes);
    auto end_count = high_resolution_clock::now();

    long long global_wedges = 0;
    long long global_butterflies = 0;

    MPI_Reduce(&local_wedges, &global_wedges, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_butterflies, &global_butterflies, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end_total = high_resolution_clock::now();

    if (world_rank == 0) {
        duration<double> count_time = end_count - start_count;
        duration<double> total_time = end_total - start_total;
        cout << "Total butterflies: " << global_butterflies << endl;
        cout << "Total wedges: " << global_wedges << endl;
        cout << "Counting time: " << count_time.count() << " seconds" << endl;
        cout << "Total execution time: " << total_time.count() << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

