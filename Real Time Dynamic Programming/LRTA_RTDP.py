"""
Created on Wed Feb  1 19:27:41 2017

@author: monica
"""
#--------------------------- Imports ----------------------------
import cv2
import itertools
import numpy as np
import random
import math

#-------------------------Class --------------------------------------
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

#---------------------- Global Parameters ----------------------------------
#img1 = cv2.imread('pic3.jpg',0)
img1 = cv2.imread('pic4.png',0)

ret,world = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
height, width = img1.shape

print "------------" + str(img1.shape)

X_min = 0
Y_min = 0
X_max = width - 1
Y_max = height - 1
states = dict()

start = set()
goal = set()

state_val = np.full((width,height),999.0)

move = {1,-1}

admisible_actions = set(itertools.product(move,move))

p_correct = 0.9
ci_nonGoal = 1
ci_Goal = 0


#----------------------------- Helper FUnctions -------------------------------

def generate_startGoal():
    global goal, start, world
    
    for i in range(0,70):
        goal.add((X_max,i))
        goal.add((X_max-1,i))
        
    for j in range(Y_max-70,Y_max):
        start.add((X_max,j))
        start.add((X_max-1,j))
        
   

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
          
def lrta_star(start,goal):
    """
    LRTA* implementation for shortest path problem in grid world
    @param start, goal
    """
    global world
    path = list()
    cost = dict()
    current_state = start
    
    while(not(current_state in goal)):
        
        #Get all the nodes reachable by admisible action
        neighbours = stateNeighbours(current_state)
        
        #For nodes find the cost
        for n in neighbours:
            cost_n = ci_nonGoal + search_heuristic(n,goal)
            cost[cost_n] = n
        
        min_cost = min(cost.keys())
        
        #Update current node cost to min cost of neighbour node, move to neighbour node
        if state_val[current_state[0],current_state[1]] > min_cost:
            state_val[current_state[0],current_state[1]] = min_cost
            new_state = cost[min_cost]
            path.append(current_state)
            cv2.line(world, (current_state[0],current_state[1]), (new_state[0],new_state[1]), 1, 1)
            current_state = new_state
            
        cv2.imshow('window',world)
        
        if cv2.waitKey(50)==27:
            break
    cv2.destroyAllWindows()
    
def populate_Bt(frontNode,depth):
    """
    Forward search function which gives set for asynchrous DP update
    @param current_node, depth upto which the search is to be done
    @return subset of state set
    """
    count = 0
    Bt = set()
    s = Stack()
    s.push(frontNode)
    while( not (s.size() == 0)):
        count += 1
        node = s.pop()
        Bt.add(node)
        neighbours = stateNeighbours(node)
        if count <= depth:
            for n in neighbours:
                s.push(n)  
    return Bt
  
def rtdp(start,goal):
    global world
    path = list()
    cost = dict()
    costf= dict()
    sweep_val = np.full((width,height),999.0)
    
    frontNode = start
    
    #While goal is not reached
    while(not(frontNode in goal)):
        Bt = populate_Bt(frontNode,1)
        
        #perform asynchrous updates for all nodes in Bt
        for item in list(Bt):
            neighbours = stateNeighbours(item)
        
            for n in neighbours:
                cost_n = ci_nonGoal + search_heuristic(n,goal)
                cost[cost_n] = n

            min_cost = min(cost.keys())
        
            if sweep_val[item[0],item[1]] > min_cost:
                sweep_val[item[0],item[1]] = min_cost
        
        #Choose a greedy control action for current node   
        state_val = sweep_val
        neighbours_front = stateNeighbours(frontNode)
        
        for n in neighbours_front:
            cost_nf = state_val[n[0],n[1]]
            costf[cost_nf] = n
            
        min_costf = min(costf.keys())
        
        new_state = cost[min_costf]
        path.append(frontNode)
        cv2.line(world, (frontNode[0],frontNode[1]), (new_state[0],new_state[1]), 1, 1)
        frontNode = new_state
            
        cv2.imshow('window',world)
        
        if cv2.waitKey(50)==27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':      
    generate_startGoal()
    start_point = list(start)[random.randint(0,len(start) - 1)]
    goal_point = (100,256) 
    cv2.circle(world, start_point, 2, color=1, thickness=-1, lineType=8, shift=0)
    cv2.circle(world, goal_point, 2, color=1, thickness=-1, lineType=8, shift=0)
    
    ############ PLEASE UNCOMMENT THE ONE TO RUN AND COMMENT ONE NOT NEEDED ##########
    lrta_star(start_point,goal_point)
    #rtdp(start_point,goal_point)