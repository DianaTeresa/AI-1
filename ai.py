import os
from re import S
import matplotlib.pyplot as plt
import copy
from heapq import heappop
from heapq import heappush
from queue import PriorityQueue
import math
import pygame
import teleport_waypoint_version as tlp
row = [-1, 0, 1, 0]
col = [0, 1, 0, -1]
weight = 1 
oo = 100000000
def visualize_maze(matrix, bonus, portal, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. portal: The array of portals,
      4. start, end: The starting and ending points,
      5. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    if bonus:
        plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='silver')

    if portal:
        for i in range(len(portal)):
            plt.text(portal[i][1], -portal[i][0], s=chr(i + 49), color='green', fontsize = 10, weight='bold')
            plt.text(portal[i][3], -portal[i][2], s=chr(i + 49), color='green', fontsize = 10, weight='bold')

    plt.text(end[1],-end[0],'EXIT',color='red',
        horizontalalignment='center',
        verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    if bonus:
        for _, point in enumerate(bonus):
            print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')

# def read_file(file_name: str = 'maze.txt'):
#     f=open(file_name,'r')
#     n_bonus_points = int(next(f)[:-1])
#     bonus_points = []
#     for i in range(n_bonus_points):
#         x, y, reward = map(int, next(f)[:-1].split(' '))
#         bonus_points.append((x, y, reward))

#     text=f.read()
#     matrix=[list(i) for i in text.splitlines()]
#     f.close()

#     return bonus_points, matrix

def read_file(file_name: str):
    f = open(file_name,'r')
    n = int(next(f)[:-1])
    portals = bonus_points = None
    if n > 0:
        temp = [i for i in map(int, next(f)[:-1].split(' '))]
        if len(temp) == 4:
            portals = [(temp[0], temp[1], temp[2], temp[3])]
        else:
            bonus_points = [(temp[0], temp[1], temp[2])]
        if portals is not None:
            for i in range(n - 1):
                x, y, x1, y1 = map(int, next(f)[:-1].split(' '))
                portals.append((x, y, x1, y1))
        else: 
            for i in range(n - 1):
                x, y, z = map(int, next(f)[:-1].split(' '))
                bonus_points.append((x, y, z))
            pass
    text = f.read()
    matrix = [list(i) for i in text.splitlines()]
    f.close()
    if n > 0: # Insert portals
        if portals is not None:
            for p in portals:
                x, y, x1, y1 = p
                matrix[x][y] = (x1, y1)
                matrix[x1][y1] = (x, y)
        else: # Insert bonus points
            for b in bonus_points:
                x, y, z = b
                matrix[x][y] = z
    return bonus_points, portals, matrix


# Tìm kiếm mù
def dfs(graph, start, end):

    trace={start:(0,0)} #mảng truy vết
    stack=[start]
    Free=copy.deepcopy(graph)
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            Free[i][j]=True
    Free[start[0]][start[1]]=False

    while len(stack)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=stack.pop()    #Lấy ra vị trí nằm trên đầu của stack
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if Free[neighbor[0]][neighbor[1]]==True and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    stack.append(neighbor)
                    Free[neighbor[0]][neighbor[1]] =False      
    
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path

def bfs(graph, start, end):

    trace={start:(0,0)} #mảng truy vết
    queue=[start]
    
    while len(queue)>0:   #Duyệt cho đến khi không còn đỉnh còn đường đi 
        top=queue.pop(0)    #Lấy ra vị trí nằm trên đầu của queue
        if top==end:    
            break
        for neighbor in [(top[0],top[1]+1),(top[0],top[1]-1),(top[0]+1,top[1]),(top[0]-1,top[1])]:     # Xét 4 vị trí liền kề vị trí đang xét top
            if neighbor not in trace and graph[neighbor[0]][neighbor[1]] == 0: 
                    trace[neighbor]=top  #Truy vết đường đi trước đó của neighbor là top
                    queue.append(neighbor)      
    
    
    path=[]
    f=end
    while trace[f]!=(0,0):
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path
def reconstruct_path(trace, start, end):
    current = end
    path = [current]
    while trace:
        current = trace.pop()
        path.append(current)
    path.reverse()
    return path

def Is_Valid_Position(graph, node): 
    if node[0] >= 0:
        if  node[0] < len(graph):
            if node[1] >=0:
                if node[1] <len(graph[0]):
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
    trace = {start:None}
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
                    trace[new_node] = node
        
        visited.add(node)

    path=[]
    f=end
    while trace[f]!=None:
        path.append(trace[f])
        f=trace[f]
        
    path.reverse()
    path.append(end)
    return path

def createPath(trace, start, dest):
    l = [dest]
    while dest != start:
        dest = trace[dest[0]][dest[1]]
        l.append(dest)
    return l[0:][slice(None, None, -1)]
# First heuristic: Euclidean distance
def GBFS_Heur1(a, start, dest):
    """ Heuristic function: h(x, y) = d((x, y), dest) 
        with d(A, B) is the Euclidean distance of A and B
    """
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    PQ.put((0, start))
    op = []
    while not PQ.empty():
        _, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            trace[v[0]][v[1]] = u
            op.append(v)
            PQ.put((math.sqrt((v[0] - dest[0]) ** 2 + (v[1] - dest[1]) ** 2), v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(trace, start, dest), op
# Second heuristic: Mahattan distance
def GBFS_Heur2(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    PQ = PriorityQueue(r_size * c_size)
    PQ.put((0, start))
    op = []
    while not PQ.empty():
        _, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            trace[v[0]][v[1]] = u
            op.append(v)
            PQ.put((abs(v[0] - dest[0]) + abs(v[1] - dest[1]), v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(trace, start, dest), op

def UCS(a, start, dest):
    r_size, c_size = len(a), len(a[0])
    d = [[oo for i in range(c_size)] for j in range(r_size)]
    trace = [[None for i in range(c_size)] for j in range(r_size)]
    d[start[0]][start[1]] = 0
    PQ = PriorityQueue(len(a) * len(a[0]))
    PQ.put((0, start))
    op = []
    while not PQ.empty():
        p, u = PQ.get()
        if (u == dest):
            break
        for k in range(4):
            v = (u[0] + row[k], u[1] + col[k])
            if v[0] < 0 or v[0] > r_size or v[1] < 0 or v[1] > c_size:
                continue
            if a[v[0]][v[1]] == 1 or trace[v[0]][v[1]] != None: 
                continue
            w = 1
            if d[v[0]][v[1]] > p + w:
                trace[v[0]][v[1]] = u
                d[v[0]][v[1]] = p + w
                op.append(v)
                PQ.put((d[v[0]][v[1]], v))
    if trace[dest[0]][dest[1]] == None: return None
    return createPath(trace, start, dest), op

def PGAME():
    weight =500
    height = 500
    WIN =pygame.display.set_mode((weight,height)) 
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run =False
    pygame.quit()
def main():
    
    for mapId in range(3):
        # bonus_points, portals, matrix = read_file(f'{mapId + 1}.txt')
        bonus_points, portals, matrix = read_file(f'bonus_map{mapId + 1}.txt')
        print(f'The height of the matrix: {len(matrix)}')
        print(f'The width of the matrix: {len(matrix[0])}')

        # Xác định 2 điểm đầu cuối
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)

                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                        
                else:
                    pass

        
        row=len(matrix)
        col=len(matrix[0])
        graph=[]

        for i in range(len(matrix)):
            adj=[]
            for j in range(len(matrix[0])):
                if matrix[i][j]!='x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        # Test map with portals
        # wayout = tlp.BFS(matrix, start, end)
        # visualize_maze(matrix, bonus_points, portals, start, end, wayout)
        # print(f'Cost = {len(wayout) - 1}\n')

        # Test map with bonus points
        wayout = tlp.A_star(graph, start, end)
        visualize_maze(matrix, bonus_points, portals, start, end, wayout)

        # wayoutDFS=dfs(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutDFS)
        # print(f'DFS: Cost = {len(wayoutDFS)-1}\n')

        # wayoutBFS=bfs(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutBFS)
        # print(f'BFS: Cost = {len(wayoutBFS)-1}\n')
        
        # wayoutUCS=UCS(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutUCS)
        # print(f'UCS: Cost = {len(wayoutUCS)-1}\n')

        # wayoutGBFS1=GBFS_Heur1(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutGBFS1)
        # print(f'GBFS1: Cost = {len(wayoutGBFS1)-1}\n')

        # wayoutGBFS2=GBFS_Heur2(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutGBFS2)
        # print(f'GBFS2: Cost = {len(wayoutGBFS2)-1}\n')

        # wayoutASTAR=a_star(graph, start, end)
        # visualize_maze(matrix,bonus_points,start,end,wayoutASTAR)
        # print(f'A_STAR: Cost = {len(wayoutASTAR)-1}\n')
        PGAME()

if __name__=="__main__":
    main()
