def neighbor(n):
    print(n)

def astar(graph, start, end):
    open = set(start)
    close = set()
    g = {}
    parents = {}
    g[start] = 0
    parents[start] = start

    while open:
        n= None
        for u in open:
            if n == None or g[u] + h[u] < g[n] + h[n]:
                n = u
        if n == end or graph[n] == None:
            pass
        else:
            for (m, w) in neighbor(n):
                if m not in open and m not in close:
                    open.add(m)
                    parents[m] = n
                    g[m] = g[n] + w