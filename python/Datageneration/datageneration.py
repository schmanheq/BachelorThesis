import networkx as nx
import igraph as ig
from torch_geometric.utils import dense_to_sparse
import torch 
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

def create_small_network(n,k,p, show=False):
    G_ig = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k//2, p=p)
    print("Graph created.")
    return G_ig

def calc_infection_rate(G, recovery_rate):
    degree_list = [d for d in G.degree()]
    first_moment = np.mean(degree_list)
    second_moment = np.mean([d**2 for d in degree_list]) 
    upperbound = 10.0
    lowerbound = 9.5
    beta_lower = (lowerbound*recovery_rate)/(-lowerbound+lowerbound*recovery_rate+((second_moment-first_moment)/first_moment))
    beta_upper = (upperbound*recovery_rate)/(-upperbound+upperbound*recovery_rate+((second_moment-first_moment)/first_moment))
    test_upper = beta_upper/(beta_upper+recovery_rate-beta_upper*recovery_rate) * ((second_moment-first_moment)/first_moment)
    test_lower = beta_lower/(beta_lower+recovery_rate-beta_lower*recovery_rate) * ((second_moment-first_moment)/first_moment)
    print(f"infection rates are {beta_lower, beta_upper}")
    print(f"Reproduction rates: {test_lower, test_upper}")
    return beta_lower, beta_upper

def to_csv(graph, path):
    edge_index = np.array(graph.get_edgelist()).T
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

def simulate_outbreak_fast(network, infection_rate, recovery_rate, iterations,path_logfile, show=False):
    sparse_matrix = network.get_adjacency_sparse()

    A = sparse_matrix.toarray()
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

    infections_caused_by = np.zeros(n_nodes, dtype=int)
    log_data = []
    early_death = None

    log_data.append(f"____________ Simulation start ____________ \n")
    for i in range(iterations):
        if infected.sum()==0:
            early_death=i
            break
        # See who is infected
        infected_indices = np.where(infected)[0] 

        # check who of the infected's neighbours are susceptible
        potential_infections = A[infected_indices, :].sum(axis=0)
        candidates = susceptible & (potential_infections > 0)
        candidate_indices = np.where(candidates)[0]
        K = potential_infections[candidate_indices]

        # determine if they get infected
        p_infected_per_node = infection_rate**K
        newly_infected_mask = np.random.rand(len(candidate_indices)) <= p_infected_per_node
        newly_infected_indices = candidate_indices[newly_infected_mask]


        # this is for debugging purposes
        # if len(newly_infected_indices) > 0:
        #     for target in newly_infected_indices:
        #         potential_sources = np.intersect1d(np.where(A[:, target] > 0)[0], infected_indices)
        #         actual_source = np.random.choice(potential_sources)
        #         infections_caused_by[actual_source] += 1

        # updating our susceptible and infected list
        susceptible[newly_infected_indices] = False
        infected[newly_infected_indices] = True

        # determine who of the infected gets to recovered
        recovery_mask = np.random.rand(len(infected_indices)) <= recovery_rate
        newly_recovered_indices = infected_indices[recovery_mask]

        # this is for debugging purposes
        # if recovery_mask.sum()>0:
        #     total = 0
        #     for r_node in newly_recovered_indices:
        #         count = infections_caused_by[r_node]
        #         total+=count
        #     print(f"average basic reproduction number per node in timestamp {i}: {total/recovery_mask.sum()}")

        # updating infected and recovered and saving the timestamp
        infected[newly_recovered_indices] = False
        recovered[newly_recovered_indices] = True
        current_state[newly_infected_indices] = 2 
        current_state[newly_recovered_indices] = 3 
        all_snapshots.append(current_state.copy())
       
        ### Visual presentation of infection spreading ###
        if show:
            pos = nx.spring_layout(network, seed=42) 
            color_list = np.full(len(susceptible), 'skyblue', dtype=object)
            color_list[infected==1]='red'
            color_list[recovered==1]='green'

            plt.figure(figsize=(6, 6))
            nx.draw(network,pos,node_color=color_list, node_size=100, with_labels=False)
            plt.title(f"Iteration {i}")
            plt.show()
    print(i)
    log_data.append(f'early death in timestamp {early_death} \n')
    log_data.append(f'susceptible:{susceptible.sum()}, infected:{infected.sum()}, recovered:{recovered.sum()} \n')
    all_snapshots = np.array(all_snapshots, dtype=np.int8)
    with open(path_logfile, 'a') as logfile:
        logfile.writelines(log_data)
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
        print(len(newly_infected_total))
        infected[newly_infected_total]=1
        susceptible[newly_infected_total]=0
        ###### Infection Process End ######
        print("__________________")

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
    print(recovered.sum)
    return all_snapshots

def save_snapshots_fast(data_snapshots, path):
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_snapshots)
        writer.writerow(['_'])

def training_data_generation(num_samples,num_nodes,k_mean, recovery_rate,num_iterations, path_network, path_snapshots,path_logfile, p_prob=None, show=False):
    for i in range(num_samples):
        if not p_prob:
            p_prob = random.uniform(0.001, 0.1)
        network = create_small_network(num_nodes,k_mean, p_prob)
        to_csv(network, path_network)
        beta_lower, beta_upper = calc_infection_rate(network, recovery_rate)
        #beta_upper = 0.33
        data_snapshots = simulate_outbreak_fast(network, beta_upper,recovery_rate,num_iterations-1,path_logfile, show )
        save_snapshots_fast(data_snapshots, path_snapshots)

def get_gamma(num_nodes, k_mean, recovery_rate):
    network = create_small_network(num_nodes, k_mean, p=0.1)
    gamma = calc_infection_rate(network, k_mean, recovery_rate)
    return gamma

    
