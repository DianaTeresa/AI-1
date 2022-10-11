from queue import PriorityQueue
import math
import UCS_GBFS as GBFS

row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
weight = 1


def reconstruct_path(trace, start, end):
    current = (end[0],end[1])
    path = [current]
    while trace:
        current = trace.pop()
        path.append(current)
    path.reverse()
    return path

def Is_Valid_Position(graph, node): 
    if node[0] >= 0 and node[0] < len(graph) and node[1] >=0 and node[1] <len(graph):
        return graph[node[0]][node[1]] == 0
    return False

def Heuristic_1(current_node, goal):
    "Manhattan distance"
    current_row, current_col = current_node
    end_row, end_col = goal
    return abs(current_row - end_row) + abs(current_col - end_col)

def Heuristic_2(current_node, goal):
    "Diagonal distance"
    current_row, current_col = current_node
    end_row, end_col = goal
    dx = abs(current_row - end_row)
    dy = abs(current_col - end_col)
    return dx + dy + (math.sqrt(2)-2) * min(dx, dy)

def a_star(graph, start, end):
    distances = {(start[0], start[1]): 0}
    trace = [[None for i in range(len(graph[0]))] for j in range(len(graph))]
    visited = set()

    pq = PriorityQueue()
    pq.put((0, (start[0], start[1])))

    while not pq.empty():
        _, node = pq.get()
        cur_row, cur_col = node
        if node == (end[0], end[1]):
            break
        
        for i in range(0,4):
            new_node = (cur_row + row[i], cur_col + col[i])
            
            if Is_Valid_Position(graph, new_node) and new_node not in visited:
                old_distance = distances.get(new_node, float('inf'))
                new_distance = distances[node] + weight
                
                if new_distance < old_distance:
                    distances[new_node] = new_distance
                    priority = new_distance + Heuristic_1(new_node, end)
                    pq.put((priority, new_node))
                    trace[new_node[0]][new_node[1]] = node
        
        visited.add(node)

    return trace #reconstruct_path(trace, start, end)
  
 
graph, start, end = GBFS.readInput("1.txt")
trace = a_star(graph, start, end)
GBFS.printMaze(GBFS.createPath(graph,trace,start,end), start, "output1.txt")