import networkx as nx
from torch_geometric.utils import dense_to_sparse
import torch 
import numpy as np
import random
import csv
import matplotlib as plt
import time

def create_small_network(n,k,p, show=False):
    G = nx.watts_strogatz_graph(n, k, p)
    clustering = nx.average_clustering(G)
    path_length = nx.average_shortest_path_length(G)

    print(f"Average clustering coefficient: {clustering:.3f}")
    print(f"Average shortest path length: {path_length:.3f}")
    if show:
        plt.figure(figsize=(6, 6))
        nx.draw(G, node_size=50, with_labels=False)
        plt.title(f"Watts-Strogatz Small-World Network (p={p})")
        plt.show()
    return G

def to_csv(graph, path):
    A = nx.to_numpy_array(graph).astype(int)
    A = torch.tensor(A, dtype=torch.int)
    edge_index, _ = dense_to_sparse(A)
    edge_index = edge_index.tolist()
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(edge_index)
        writer.writerow(['_'])
    
def csv_to_graph(path):
    A = np.loadtxt(path, delimiter=",")
    network = nx.from_numpy_array(A)
    return network

def get_neighbours(network, node):
    return np.where(network[node]==1)[0]

def simulate_outbreak_fast(network, infection_rate, recovery_rate, iterations):
    A = nx.to_numpy_array(network)
    n_nodes = len(A)
    susceptible = np.ones(n_nodes, dtype=bool)
    infected = np.zeros(n_nodes, dtype=bool)
    recovered = np.zeros(n_nodes, dtype=bool)

    source_node = random.randint(0, n_nodes - 1)
    susceptible[source_node] = False
    infected[source_node] = True
    current_state = np.full(n_nodes, 1, dtype=np.int8) 
    current_state[source_node] = 2
    all_snapshots = [current_state.copy()]

    for i in range(iterations):
        infected_indices = np.where(infected)[0]
        potential_infections = A[infected_indices, :].sum(axis=0)
        candidates = susceptible & (potential_infections > 0)
        candidate_indices = np.where(candidates)[0]
        K = potential_infections[candidate_indices]
        p_infected_per_node = 1 - (1 - infection_rate)**K
        newly_infected_mask = np.random.rand(len(candidate_indices)) < p_infected_per_node
        newly_infected_indices = candidate_indices[newly_infected_mask]
        susceptible[newly_infected_indices] = False
        infected[newly_infected_indices] = True
        recovery_mask = np.random.rand(len(infected_indices)) <= recovery_rate
        newly_recovered_indices = infected_indices[recovery_mask]
        infected[newly_recovered_indices] = False
        recovered[newly_recovered_indices] = True
        current_state[newly_infected_indices] = 2 # Infected
        current_state[newly_recovered_indices] = 3 # Recovered
        all_snapshots.append(current_state.copy())

    all_snapshots = np.array(all_snapshots, dtype=np.int8)
    return all_snapshots

def simulate_outbreak(network, infection_rate, recovery_rate, iterations, show=False):
    #setup
    A = nx.to_numpy_array(network)
    pos = nx.spring_layout(network, seed=42) 

    n_nodes = len(A[0])
    susceptible = np.ones(n_nodes)
    infected = np.zeros(n_nodes)
    recovered = np.zeros(n_nodes)


    source_node = random.randint(0, len(A[0])-1)
    susceptible[source_node] = 0
    infected[source_node] = 1

    all_snapshots = [np.array([susceptible, infected, recovered])]

    for i in range(iterations):
        if show:
            color_list = np.full(len(susceptible), 'skyblue', dtype=object)
            color_list[infected==1]='red'
            color_list[recovered==1]='green'

            plt.figure(figsize=(6, 6))
            nx.draw(network,pos,node_color=color_list, node_size=100, with_labels=False)
            plt.title(f"Iteration {i}")
            plt.show()

        ###### Infection Process ######
        old_infected = np.where(infected==1)[0]
        newly_infected_total = []
        for j in old_infected:
            neighbours = get_neighbours(A,j)
            susceptible_neighbours = neighbours[susceptible[neighbours]==1]
            gets_infected = np.random.rand(len(susceptible_neighbours))<=infection_rate
            newly_infected = susceptible_neighbours[gets_infected]
            newly_infected_total.extend(newly_infected)  
        newly_infected_total = np.array(newly_infected_total, dtype=int)
        infected[newly_infected_total]=1
        susceptible[newly_infected_total]=0
        ###### Infection Process End ######

        ###### Recovery Process ######
        gets_recovered = np.random.rand(len(old_infected))<=recovery_rate
        newly_recovered = old_infected[gets_recovered]
        recovered[newly_recovered]=1
        infected[newly_recovered]=0
        ###### Recovery Process END ######

        ###### Saving Data ######
        snapshot = np.array([susceptible, infected, recovered])
        all_snapshots.append(snapshot)
    all_snapshots = np.array(all_snapshots, dtype=int)
    return all_snapshots
    
def save_snapshots(data, path):
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        for snapshot in data:
            susceptible_index = np.where(snapshot[0]==1)[0]
            infected_index = np.where(snapshot[1]==1)[0]
            recovered_index = np.where(snapshot[2]==1)[0]
            new_snapshot = np.zeros_like(snapshot[0])
            new_snapshot[susceptible_index]=1
            new_snapshot[infected_index]=2
            new_snapshot[recovered_index]=3
            writer.writerow(new_snapshot)
        writer.writerow(['_'])

def save_snapshots_fast(data_snapshots, path):
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_snapshots)
        writer.writerow(['_'])

def training_data_generation(num_samples,num_nodes,k_mean, infection_rate, recovery_rate,num_iterations, path_network, path_snapshots, p_prob=None):
    for i in range(num_samples):
        if not p_prob:
            p_prob = random.uniform(0.001, 0.1)
        network = create_small_network(num_nodes,k_mean, p_prob)
        to_csv(network, path_network)
        data_snapshots = simulate_outbreak_fast(network, infection_rate,recovery_rate,num_iterations-1)
        save_snapshots_fast(data_snapshots, path_snapshots)

def inference_data_generation(num_samples, path_network, path_snapshots):
    for i in range(num_samples):
        random_p = random.uniform(0.0, 0.5)
        network = create_small_network(10,4, random_p)
        to_csv(network, path_network)
        random_infectionrate = random.random()
        random_recoveryrate = random.random()
        data_snapshots = simulate_outbreak(network, random_infectionrate,random_recoveryrate,10)
        save_snapshots(data_snapshots, path_snapshots)

    
