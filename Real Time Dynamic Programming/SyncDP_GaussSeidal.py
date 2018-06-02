"""
Created on Wed Feb  1 19:27:41 2017

@author: Monica Patel
"""
#--------------------------- Imports ----------------------------
import cv2
import itertools
import numpy as np
import random
import math
#---------------------- Global Parameters ----------------------------------
world = cv2.imread('pic4.png',0)

height, width = world.shape
state_space = world

print "-----World Shape-------" + str(world.shape) + '---------World Shape-----------'

X_min = 0
Y_min = 0
X_max = width - 1 
Y_max = height - 1
states = dict()

start = set()
goal = set()

move = {1,-1}

admisible_actions = set(itertools.product(move,move))

p_correct = 0.9

ci_nonGoal = 1
ci_Goal = 0
ci_wall = 1000

delta = 0.1

#----------------------------- Helper FUnctions -------------------------------


def generate_startGoal():
    """
    Generates a set of Start points and Goal points from total State Space
    """
    global goal, start, world
    
    for i in range(31,50):
        goal.add((i,0))
        goal.add((i,1))
        
    for j in range(0,20):
        start.add((j,0))
        start.add((j,1))
        
   
def stateNeighbours(state):
    """
    Takes in node, (Image point) and returns all the neighboring image points
    @param node <-- tuple
    @return list(tuple)
    """
    neighbours = list()
    
    x = state[0]
    y = state[1]
    
    if not ((x+1) > X_max or (y+1) > Y_max ):
        neighbours.append((x+1,y+1))
    if not ((x+1) > X_max ):
        neighbours.append((x+1,y))
    if not ((y+1) > Y_max ):
        neighbours.append((x,y+1))
    if not ((x-1) < X_min or (y-1) < Y_min ):
        neighbours.append((x-1,y-1))
    if not ((x-1) < X_min ):
        neighbours.append((x-1,y))
    if not ((y-1) < Y_min ):
        neighbours.append((x,y-1))
    if not ((x+1) > X_max or (y-1) < Y_min ):
        neighbours.append((x+1,y-1))
    if not ((x-1) < X_min or (y+1) > Y_max ):
        neighbours.append((x-1,y+1))
        
    return neighbours   

def search_heuristic(state,goal):
    """
    Hueristic for estimating cost of future nodes.
    @param state - tuple, goal - tuple
    @return float - estimated cost of goal from the state. In this case Euclidean Dist
    """
    dist = math.sqrt((goal[0] - state[0])**2 + (goal[1] - state[1])**2)
    return dist
   
def convergence_condition(state_val, epoch_val, point):
    """
    Check if the cost are converged to the optimal values
    @param: cost of states from one epoch back and current epoch
    @return Bool - True if difference between values are less than fixed delta.
    """
    diffrence = (state_val[point[0],point[1]] - epoch_val[point[0],point[1]]) 
    
    if (diffrence < delta):
        return True
    else:
        return False
       
def syncDP(start,goal):
    """
    Implementation of the Synchronous version of DP for shortest path problem
    @param: start, goal
    @return: Optimal Value function
    """
    global state_space
    
    epochs = 0

    state_val = np.full((width+1,height+1),999.0)
    epoch_val = np.full((width+1,height+1),999.0)

    condition = False
    #Repeat until Convergence
    while (not condition):
        
        for i in range(X_max):
            for j in range(Y_max):
                # Update costs of the states
                cost = dict()
                curr_state = (i,j)
                neighbours = stateNeighbours((curr_state))
    
                for n in neighbours:
                    cost_n = ci_nonGoal + search_heuristic(n,goal)
                    cost[cost_n] = n
                
                if not len(neighbours) == 0:
                    min_cost = min(cost.keys())
                

                if epoch_val[curr_state[0],curr_state[1]] > min_cost:
                    epoch_val[curr_state[0],curr_state[1]] = min_cost
        
        epochs += 1
        print '-----------', epochs, '\t  epochs Complete ----------------'
        x = random.randint(X_min,X_max)
        y = random.randint(Y_min,Y_max)
        condition = convergence_condition(state_val,epoch_val,(x,y))
        
        #Back up the costs of all states at once
        state_val = epoch_val
        
        #state_space, ret = cv2.threshold(state_space,127,1,cv2.THRESH_BINARY)
        #cv2.imshow('window',state_space)
        #cv2.waitKey(1)
        
    return state_val
   

def gauss_seidal(start,goal):
    """
    Implementation of Gauss_seidal algorithm for solving MDP
    @param start goal
    @Return: Optimal Value function
    """
    epochs = 0

    state_val = np.full((width+1,height+1),999.0)
    val_kMinus1 = np.full((width+1,height+1),999.0)

    condition = False
    while (not condition):
        
        for i in range(X_max):
            for j in range(Y_max):
                cost = dict()
                curr_state = (i,j)
                
                neighbours = stateNeighbours((curr_state))
    
                for n in neighbours:
                    cost_n = ci_nonGoal + search_heuristic(n,goal)
                    cost[cost_n] = n
                
                if not len(neighbours) == 0:
                    min_cost = min(cost.keys())
                
                #Keep backing up the costs of states as when the sweep is done
                if state_val[curr_state[0],curr_state[1]] > min_cost:
                    state_val[curr_state[0],curr_state[1]] = min_cost
                
                #cv2.circle(state_space,curr_state, 2, 1, -1)    
                #cv2.imshow('window',state_space)
                #cv2.waitKey(1)
        
        epochs += 1
        print '-----------', epochs, '\t  epochs Complete ----------------'
        x = random.randint(X_min,X_max)
        y = random.randint(Y_min,Y_max)
        condition = convergence_condition(state_val,val_kMinus1,(x,y))
        val_kMinus1 = state_val
        
    return state_val
                    
def greedy_policy(start_point,goal_point,state_val):
    """
    Generates a greedy policy based on optimal value function
    @param start, goal, Optimal value function
    @return Path or optimal policy
    """
    path = list()
    
    curr_state = start_point
    while not(curr_state == goal_point):
        cost = dict()
        neighbours = stateNeighbours(curr_state)
        
        for n in neighbours:
            cost_n = state_val[n]
            cost[cost_n] = n
        
        
        if not len(neighbours) == 0:
            min_cost = min(cost.keys())
            next_state = cost[min_cost]
            path.append(next_state)
            
            cv2.line(world, (next_state[0],next_state[1]), (curr_state[0],curr_state[1]), 1, 1)
            curr_state = next_state
            
        cv2.imshow('window',world)
        if cv2.waitKey(50) == 27:
            break
    
    cv2.destroyAllWindows()
    return path
    
if __name__ == '__main__':    
    generate_startGoal()
    
    start_point = (50, 60)
    goal_point =  (305,180)

    cv2.circle(world, start_point, 5, color=1, thickness=-1, lineType=8, shift=0)
    cv2.circle(world, goal_point, 5, color=1, thickness=-1, lineType=8, shift=0)

############ PLEASE UNCOMMENT THE ONE TO RUN AND COMMENT ONE NOT NEEDED ##########
    state_val = syncDP(start_point,goal_point)
    #state_val = gauss_seidal(start_point,goal_point)
#----------------------------------------------------------------------------    
    print '---------------------Update complete-------------------'

    path = greedy_policy(start_point,goal_point,state_val)
    
