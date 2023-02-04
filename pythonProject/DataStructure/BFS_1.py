graph = {
    'S':['B','D','A'],
    'A':['C'],
    'B':['D'],
    'C':['G','D'],
    'D':['G']

}

def bfs(graph,start,goal):
    visited = []
    queue = [[start]]
    while queue:
        path = queue.pop()
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adjcent_node = graph.get(node,[])
            for node2 in adjcent_node:
                new_path = path.copy()
                new_path.append(node2)
                queue.append(new_path)

solution = bfs(graph,'S','G')

print(solution)