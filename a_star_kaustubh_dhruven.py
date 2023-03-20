import numpy as np
from math import dist
import matplotlib.pyplot as plt
import time
import heapq


#Defning a class
#Storing nodes as objects
class Node:

    def __init__(self, x, y, theta, cost, parent_id, cost2go=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = cost
        self.parent_id = parent_id
        self.cost2go = cost2go

    def __lt__(self, other):
        return self.cost + self.cost2go < other.cost + other.cost2go


#Defining actions and calculating the cost to go for each action that is performed
#Extreme Left Action
def extreme_left(x, y, theta, step, cost):
    theta = theta + 60
    x = x + (step * np.cos(np.radians(theta)))
    y = y + (step * np.sin(np.radians(theta)))

    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x, y, theta, cost

#Slight Left Action
def slight_left(x, y, theta, step, cost):
    theta = theta + 30
    x = x + (step * np.cos(np.radians(theta)))
    y = y + (step * np.sin(np.radians(theta)))

    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x, y, theta, cost

#Staright Action
def straight(x, y, theta, step, cost):
    theta = theta + 0
    x = x + (step * np.cos(np.radians(theta)))
    y = y + (step * np.sin(np.radians(theta)))

    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x, y, theta, cost

#Slight Right Action
def slight_right(x, y, theta, step, cost):
    theta = theta - 30
    x = x + (step * np.cos(np.radians(theta)))
    y = y + (step * np.sin(np.radians(theta)))

    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x, y, theta, cost

#Extreme Right Action
def extreme_right(x, y, theta, step, cost):
    theta = theta - 60
    x = x + (step * np.cos(np.radians(theta)))
    y = y + (step * np.sin(np.radians(theta)))

    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x, y, theta, cost


#A function defined to perform actions
def Action_set(move, x, y, theta, step, cost):
    if move == 0:
        return extreme_left(x, y, theta, step, cost)
    elif move == 1:
        return slight_left(x, y, theta, step, cost)
    elif move == 2:
        return straight(x, y, theta, step, cost)
    elif move == 3:
        return slight_right(x, y, theta, step, cost)
    elif move == 4:
        return extreme_right(x, y, theta, step, cost)
    else:
        return None


#A function to define the workspace
def ws(width, height, robot_clearance, robot_radius):
    #Generating an empty workspace
    ws = np.full((height, width), 0)

    for y in range(0, height):
        for x in range(0, width):

            #Using half plane equations to plot the buffer space
            #Adding buffer space to avoid collision between the robot and the walls

            #Upper Rectangular Obstacle
            r11_buffer = (x + (robot_clearance + robot_radius)) - 100
            r12_buffer = (y - (robot_clearance + robot_radius)) - 100
            r13_buffer = (x - (robot_clearance + robot_radius)) - 150

            #Lower Rectangular Obstacle
            r21_buffer = (x + (robot_clearance + robot_radius)) - 100
            r23_buffer = (x - (robot_clearance + robot_radius)) - 150
            r24_buffer = (y + (robot_clearance + robot_radius)) - 150

            #Hexagonal Obstacle
            h6_buffer = (y + (robot_clearance + robot_radius)) + 0.58 * (
                        x + (robot_clearance + robot_radius)) - 223.18
            h5_buffer = (y + (robot_clearance + robot_radius)) - 0.58 * (
                        x - (robot_clearance + robot_radius)) + 123.21
            h4_buffer = (x - (robot_clearance + robot_radius)) - 364.95
            h3_buffer = (y - (robot_clearance + robot_radius)) + 0.58 * (
                        x - (robot_clearance + robot_radius)) - 373.21
            h2_buffer = (y - (robot_clearance + robot_radius)) - 0.58 * (
                        x + (robot_clearance + robot_radius)) - 26.82
            h1_buffer = (x + (robot_clearance + robot_radius)) - 235.040

            #Triangular Obstacle
            t1_buffer = (x + (robot_clearance + robot_radius)) - 460
            t2_buffer = (y - (robot_clearance + robot_radius)) + 2 * (x - (robot_clearance + robot_radius)) - 1145
            t3_buffer = (y + (robot_clearance + robot_radius)) - 2 * (x - (robot_clearance + robot_radius)) + 895

            #Setting of line constrain to avoid collision between the robot and the obstacles
            if ((
                    h6_buffer > 0 and h5_buffer > 0 and h4_buffer < 0 and h3_buffer < 0 and h2_buffer < 0 and h1_buffer > 0) or (
                    r11_buffer > 0 and r12_buffer < 0 and r13_buffer < 0) or (
                    r21_buffer > 0 and r23_buffer < 0 and r24_buffer > 0) or (
                    t1_buffer > 0 and t2_buffer < 0 and t3_buffer > 0)):
                ws[y, x] = 1

            #Using half plane equation to plot the actual obstacle map

            #Upper Rectangular Obstacle
            r11 = (x) - 100
            r12 = (y) - 100
            r13 = (x) - 150
            # r14 = y - 0

            #Lower Rectangular Obstacle
            r21 = (x) - 100
            # r22 = (y) - 250
            r23 = (x) - 150
            r24 = (y) - 150

            #Hexagonal Obstacle
            h6 = (y) + 0.58 * (x) - 223.18
            h5 = (y) - 0.58 * (x) + 123.21
            h4 = (x) - 364.95
            h3 = (y) + 0.58 * (x) - 373.21
            h2 = (y) - 0.58 * (x) - 26.82
            h1 = (x) - 235.04

            #Triangular Obstacle
            t1 = (x) - 460
            t2 = (y) + 2 * (x) - 1145
            t3 = (y) - 2 * (x) + 895

            #Setting of line constraint with the buffer
            if ((h6 > 0 and h5 > 0 and h4 < 0 and h3 < 0 and h2 < 0 and h1 > 0) or (
                    r11 > 0 and r12 < 0 and r13 < 0) or (r21 > 0 and r23 < 0 and r24 > 0) or (
                    t1 > 0 and t2 < 0 and t3 > 0)):
                ws[y, x] = 2

    return ws


# A function to check the validity of the move
def Move_Validity(x_v, y_v, ws):
    Move_Validity = ws.shape

    if (x_v > Move_Validity[1] or x_v < 0 or y_v > Move_Validity[0] or y_v < 0):
        return False

    else:
        try:
            if (ws[y_v][x_v] == 1 or ws[y_v][x_v] == 2):
                return False
        except:
            pass
    return True


#A function that verifies wether the current node is goal node or not 
def Goal_checker(current, goal):
    distance = dist((current.x, current.y), (goal.x, goal.y))
    threshold = 1.5

    if distance < threshold:
        return True
    else:
        return False


#A function that generates an unique ID for every node
def UID(node):
    UID = 222 * node.x + 111 * node.y
    return UID


#A function that defines the A start algorithm and returns all the required nodes
def a_star(start, goal, ws, step):
    if Goal_checker(start, goal):
        return None
    goal_node = goal
    start_node = start

    moves = [0, 1, 2, 3, 4] # All possible moves
    ol = {}  #Open list
    cl = {}  #Closed list
    priority = []  #A priority queue based on the cost of each node
    all_nodes = []  #A list that stores all the nodes that were visited

    start_id = UID(start_node)  #Assigning an unique ID to the start node
    ol[(start_id)] = start_node #Adding the start node to the open list

    heapq.heappush(priority, [start_node.cost, start_node])  #Prioritizing the next node that is to be explored based on their total cost

    while (len(priority) != 0): #Loop until the priority queue is empty

        present_node = (heapq.heappop(priority))[1] #Get the node with the least cost from the priority queue
        all_nodes.append([present_node.x, present_node.y, present_node.theta]) #Adding the node to the list of all nodes visited
        present_id = UID(present_node) #Get the unique ID of the present node
        if Goal_checker(present_node, goal_node): #Check if the present node is the goal node
            goal_node.parent_id = present_node.parent_id
            goal_node.cost = present_node.cost
            print("Goal Node found")
            return all_nodes, 1

        if present_id in cl: #Check if the present node is in the closed list
            continue
        else:  #Add the present node to the closed list
            cl[present_id] = present_node

        del ol[present_id] #Remove the present node from the open list

        for move in moves: #Looping through all possible moves
            x, y, theta, cost = Action_set(move, present_node.x, present_node.y, present_node.theta, step,
                                           present_node.cost)

            cost2go = dist((x, y), (goal.x, goal.y))  #Calculate the cost to go

            new_node = Node(x, y, theta, cost, present_node, cost2go)

            new_node_id = UID(new_node)

            if not Move_Validity(new_node.x, new_node.y, ws): #Checking if the move is valid (i.e., if the new state is inside the workspace and does not collide with obstacles)
                continue
            elif new_node_id in cl: #Checking if the new node is already in the closed list
                continue

            if new_node_id in ol: #Checking if the new node is already in the open list
                if new_node.cost < ol[new_node_id].cost:
                    ol[new_node_id].cost = new_node.cost
                    ol[new_node_id].parent_id = new_node.parent_id
            else: #If the new node is not in the open list, add it to the open list
                ol[new_node_id] = new_node

            heapq.heappush(priority, [(new_node.cost + new_node.cost2go), new_node])

    return all_nodes, 0


#A function that traces the shortest path
def path_tracer(goal):
    x_path = []
    y_path = []
    x_path.append(goal.x)
    y_path.append(goal.y)

    parent = goal.parent_id
    while parent != -1:
        x_path.append(parent.x)
        y_path.append(parent.y)
        parent = parent.parent_id

    x_path.reverse()
    y_path.reverse()

    x = np.asarray(x_path)
    y = np.asanyarray(y_path)

    return x, y


#Plotting the workspace and the shortest path
def plot(start_node, goal_node, x_path, y_path, all_nodes, ws):
    plt.figure()
    #Plotting the start node and the goal node
    plt.plot(start_node.x, start_node.y, "Dw")
    plt.plot(goal_node.x, goal_node.y, "Dr")

    #Plotting the workspace
    plt.imshow(ws, "YlOrBr")
    ax = plt.gca()
    ax.invert_yaxis()  # y-axis inversion

    #Plotting all the explored nodes
    for i in range(len(all_nodes)):
        plt.plot(all_nodes[i][0], all_nodes[i][1], "2g-")
        # plt.pause(0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001)

    #plotting the most optimal path or the shortest path found
    plt.plot(x_path, y_path, ':y')
    plt.show()
    plt.pause(3)
    plt.close('all')


#Main function
if __name__ == '__main__':
    robot_clearance = 5  #Robot's clearance
    robot_radius = 5 #Robot's radius
    robot_step = 3  #Robot's step size

    width = 600 #Width of the workspace
    height = 250 #Height of the workspace
    ws = ws(width, height, robot_clearance, robot_radius) #Generating the workspace with the advised buffer
    cost2go = 0 #Initial cost to go

    #Asking start co-ordinates and orientation of the robot from the user
    start_x = int(input("Enter the X Co-ordinate of the starting point: "))
    start_y = int(input("Enter the Y Co-ordinate of the starting point: "))
    start_theta = int(input("Enter the Start Orientation of the Robot: "))

    #Rounding off the start orientation value to the nearest multiple of 30
    number = int(start_theta)
    remainder = number % 30
    if remainder < 15:
        start_theta = number - remainder
    else:
        start_theta = number + (30 - remainder)

    #Checking the validity of the given starting point in the workspace
    if not Move_Validity(start_x, start_y, ws):
        print("Start node is either out of bounds or in the obstacle")
        exit(-1)

    #Asking goal co-ordinates and orientation of the robot from the user
    goal_x = int(input("Enter the X Co-ordinate of the goal node: "))
    goal_y = int(input("Enter the Y Co-ordinate of the goal node: "))
    goal_theta = int(input("Enter the Goal Orientation of the Robot: "))
    
    #Rounding off the goal orientation value to the nearest multiple of 30
    number = int(goal_theta)
    remainder = number % 30
    if remainder < 15:
        goal_theta = number - remainder
    else:
        goal_theta = number + (30 - remainder)

    #Checking the validity of the given goal point in the workspace
    if not Move_Validity(goal_x, goal_y, ws):
        print("Goal node is either out of bounds or in the obstacle")
        exit(-1)

    timer_start = time.time() #Initialising timer to calculate the total computational time

    #Forming the start node and the goal node objects
    start_node = Node(start_x, start_y, start_theta, 0.0, -1, cost2go)
    goal_node = Node(goal_x, goal_y, goal_theta, 0.0, -1, cost2go)
    all_nodes, flag = a_star(start_node, goal_node, ws, robot_step)

    #Plotting the most optimal path after verifying that the goal node has been reached
    if (flag) == 1:
        x_path, y_path = path_tracer(goal_node)
        cost = goal_node.cost #Total cost to reach the goal node from the start node
        print("Total cost:", cost)
        plot(start_node, goal_node, x_path, y_path, all_nodes, ws)
        timer_stop = time.time() #Stopping the timer and displaying the time taken to reach the goal node from the start node while exploring all the nodes
        total_time = timer_stop - timer_start
        print("Total Time taken:  ", total_time)

    else:
        print("Path couldn't be found")
