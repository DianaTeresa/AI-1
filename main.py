import os
import ai_pygame as ap

def getInput(path):
    dir = os.scandir(path)
    maps, nMaps = [], 0
    for entry in dir:
        if entry.is_file():
            nMaps += 1
    for mapID in range(nMaps):
        input_path = path + f'\\input{mapID + 1}.txt'
        maps.append(ap.read_file(input_path))
    return maps

def level_1(maps):
    algos = {'dfs', 'bfs', 'ucs', 'gbfs_heuristic_1', 'gbfs_heuristic_2', 'astar_heuristic_1', 'astar_heuristic_2'}
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    new_abs_path = os.path.join(dir_path, f'output\\level_1')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        new_next_abs_path = os.path.join(new_abs_path, f'input{i + 1}')
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
        for algo in algos:
            alg_abs_path = os.path.join(new_next_abs_path, f'{algo}')
            if not os.path.exists(alg_abs_path):
                os.mkdir(alg_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue
        graph = []
        for i in range(len(matrix)):
            adj = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 'x':
                    adj.append(0)
                else:
                    adj.append(1)
            graph.append(adj)
        
        wayoutDFS, openDFS = ap.dfs(graph, start, goal)
        #ap.PGAME(graph, start, goal, wayoutDFS, openDFS)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutDFS, f'output\\level_1\\input{mapID}\\dfs\\dfs')

        wayoutBFS, openBFS = ap.bfs(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutBFS, f'output\\level_1\\input{mapID}\\bfs\\bfs')

        wayoutUCS, openUCS = ap.UCS(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutUCS, f'output\\level_1\\input{mapID}\\ucs\\ucs')

        wayoutGBFS_1, openGBFS_1 = ap.GBFS_Heur1(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutGBFS_1, f'output\\level_1\\input{mapID}\\gbfs_heuristic_1\\gbfs_heuristic_1')

        wayoutGBFS_2, openGBFS_2 = ap.GBFS_Heur2(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutGBFS_2, f'output\\level_1\\input{mapID}\\gbfs_heuristic_2\\gbfs_heuristic_2')

        wayoutAstar_1, openAstar_1 = ap.a_star1(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutAstar_1, f'output\\level_1\\input{mapID}\\astar_heuristic_1\\astar_heuristic_1')

        wayoutAstar_2, openAstar_2 = ap.a_star2(graph, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayoutAstar_2, f'output\\level_1\\input{mapID}\\astar_heuristic_2\\astar_heuristic_2')

        mapID += 1

def level_2(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    new_abs_path = os.path.join(dir_path, f'output\\level_2')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        new_next_abs_path = os.path.join(new_abs_path, f'input{i + 1}')
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue 
        wayout, open = ap.bonus_astar(matrix, bonus_points, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayout, f'output\\level_2\\input{mapID}\\output{mapID}')

        mapID += 1
    pass

def advance(maps):
    script_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(script_path)
    new_abs_path = os.path.join(dir_path, f'output\\advance')
    if not os.path.exists(new_abs_path):
        os.mkdir(new_abs_path)
    for i in range(len(maps)):
        new_next_abs_path = os.path.join(new_abs_path, f'input{i + 1}')
        if not os.path.exists(new_next_abs_path):
            os.mkdir(new_next_abs_path)
    mapID = 1
    for obj in maps:
        bonus_points, portals, matrix = obj
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 'S':
                    start = (i,j)
                elif matrix[i][j] == ' ':
                    if (i == 0) or (i == len(matrix) - 1) or (j == 0) or (j == len(matrix[0]) - 1):
                        goal = (i,j)
                else:
                    continue     
        wayout, open = ap.BFS_teleport(matrix, start, goal)
        ap.visualize_maze(matrix, bonus_points, portals, start, goal, wayout, f'output\\advance\\input{mapID}\\output{mapID}')

        mapID += 1
    pass

#level_1(getInput('input\\level_1'))
level_2(getInput('input\\level_2'))
#advance(getInput('input\\advance'))



