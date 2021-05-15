#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random


# In[2]:


data = np.loadtxt("att48_d.txt")


# In[3]:


# defining a cost function to find the distance between pth city and the qth city

def city_cost(p,q):
    cost = -data[p][q]
    
    return cost


# In[4]:


# defining a Stochastic Search Algorithm to find the city nearest to the pth city
b = 4 
rho = 2 
T = 12 # Number of Iterations



def search_city(p, search_space):
    
    if (len(search_space) == 0):
        print("All cities visited. None left")
        return -1
    
    print("Length of Search Space: ", len(search_space))
    if (len(search_space) == 1):
        return search_space[0]
    
    
    pop_t = np.array([], dtype=int) # create a list of generated city samples from which to choose the best b samples
    
    initial_pop = np.empty((b,), dtype=int)
    for k in range(0,b):
        alpha = np.random.uniform(0, 1)
        
        initial_pop[k] = search_space[(int)(np.random.uniform(0, len(search_space)))]
    
    parent_pop = initial_pop
    
    
    for t in range(0,T):
        
        print("Interation ", t+1)
        
        pop_t = np.append(pop_t, parent_pop)
        
        print("Initial Population")
        print(parent_pop)
        
        # generating children by random walk
        random_walk_pop = np.empty((b,))
        print("Empty random_walk_pop")
        print(random_walk_pop)
        
        k = 0
        while k < b:
            random_exploit = np.random.choice([-1, 1])
 
    
            random_walk = pop_t[k] + random_exploit * np.random.randint(0, rho)
            

            if (random_walk not in search_space):
                print("Random Walk", random_walk)
                continue
            
            random_walk_pop[k] = random_walk

            k = k + 1
            
        pop_t = np.append(pop_t, random_walk_pop)
        
        print("Random Walk")
        print(pop_t)
            
        # generating children by random linear combination
        random_lin_pop = np.empty((b,), dtype=int)
        k = 0
        while k<b:
            alpha = np.random.uniform(0, 1)
            
            # random_lin_pop[k] = np.rint((1 - alpha) * np.random.choice(pop_t) + alpha * np.random.choice(pop_t))
            idx1 = pop_t[(int)(np.random.uniform(0,pop_t.shape[0]))]
            print("idx1 = ", idx1)
            
            idx2 = pop_t[(int)(np.random.uniform(0,pop_t.shape[0]))]
            print("idx2 = ", idx2)
            
            # idx = (int)(np.rint((1 - alpha) * pop_t[idx1] + alpha * pop_t[idx2]))
            idx = (int)(np.rint((1 - alpha) * idx1 + alpha * idx2))
            print("idx = ", idx)
            print("pop_t shape = ", pop_t.shape[0])
            
            # if (idx == p):
            if (idx not in search_space):
                print("Lin Combi")
                continue
                
            random_lin_pop[k] = idx
            
            k = k + 1
            
        pop_t = np.append(pop_t, random_lin_pop)
        
        print("Random Linear Combination")
        print(pop_t)
        
        # generating samples by random re-initialization
        # random_pop = np.random.randint(x_min, x_max, b, dtype=int)
        random_pop = np.empty((b,), dtype=int)
        for q in range(0,b):
            random_pop[q] = search_space[(int)(np.random.uniform(0,len(search_space)))]
        
        pop_t = np.append(pop_t, random_pop)    
        
        print('Random Sampling')
        print(pop_t)
        
        # Parent Selection: Tournaments
        
        
        for k in range(0,b):
            temp_parent = np.empty((10,))
            temp_index = np.empty((10,), dtype=int)
            for q in range(0, 10):
                temp_index[q] = (int)(np.random.choice(pop_t))
                temp_parent[q] = city_cost(p, temp_index[q])
            
            parent_pop[k] = temp_index[np.argmax(temp_parent)]
        
        print("Parent_pop")
        print(parent_pop)
        
        pop_t = np.array([])
        print("\n")
    
    print(parent_pop)
    parent_cost = np.empty((b,))
    for q in range(0,b):
        parent_cost[q] = city_cost(p, q)
    
    return int(parent_pop[np.argmax(parent_cost)])
        
    


# In[14]:


search_space = list(range(0, data.shape[0]))



dist = 0

# Choose a random city as starting point
initial_pt = int(np.random.randint(0,data.shape[0]))
solution = [initial_pt]


pt = initial_pt
stop = 0
while stop < data.shape[0]:
    search_space.remove(pt)
    min_city = search_city(pt, search_space)
    
    
    if min_city == -1:
        solution.append(initial_pt)
        break
        
    solution.append(min_city)
    
    print("Nearest City: ", min_city)
    print("Current Solution:", solution)
    print("search space", search_space)
    
    dist = dist + data[pt][min_city]
    
    pt = min_city
    stop = stop + 1

dist += data[pt][initial_pt]

print("\n")

print("Final Solution: ", solution)
print("Total min Distance: ", dist)
    


# In[ ]:


### End of CODE