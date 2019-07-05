import eculid
import pdb

@profile
def traverse(graph, start, end, action):
    path = []
    net = eculid.init()
    net.forward()
    visited = [start]
    while visited:
        
        current = visited.pop(0)
        if current not in path:
            path.append(current)
            if current == end:
                return (True, path)
            # 两个顶点不相连，则跳过
            if current not in graph:
                continue
        visited = action(visited, graph[current])
    return (False, path)


def extend_bfs_path(visited, current):
    return visited + current


def extend_dfs_path(visited, current):
    return current + visited


class Pairs(object):

    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    def addNTimes(self, n):
        if n > 0:
            self.first += 1
            self.second += 1
            self.addNTimes(n-1)

    def getfirst(self):
        return self.first
    
    def getsecond(self):
        return self.second
    



@profile
def main():
    graph = {
        'Frankfurt': ['Mannheim', 'Wurzburg', 'Kassel'],
        'Mannheim': ['Karlsruhe'],
        'Karlsruhe': ['Augsburg'],
        'Augsburg': ['Munchen'],
        'Wurzburg': ['Erfurt', 'Nurnberg'],
        'Nurnberg': ['Stuttgart', 'Munchen'],
        'Kassel': ['Munchen'],
        'Erfurt': [],
        'Stuttgart': [],
        'Munchen': []
    }
    pair = Pairs(4, 8)
    pair.addNTimes(4)
    m = pair.getfirst()
    n = pair.getsecond()

    i = eculid.gcd(m, n)
    net = eculid.init()
    net.forward()
    
    
    

    

if __name__ == "__main__":
    graph = {
        'Frankfurt': ['Mannheim', 'Wurzburg', 'Kassel'],
        'Mannheim': ['Karlsruhe'],
        'Karlsruhe': ['Augsburg'],
        'Augsburg': ['Munchen'],
        'Wurzburg': ['Erfurt', 'Nurnberg'],
        'Nurnberg': ['Stuttgart', 'Munchen'],
        'Kassel': ['Munchen'],
        'Erfurt': [],
        'Stuttgart': [],
        'Munchen': []
    }

    bfs_path = traverse(graph, 'Frankfurt', 'Nurnberg', extend_bfs_path)
    dfs_path = traverse(graph, 'Frankfurt', 'Nurnberg', extend_dfs_path)
    print('bfs Frankfurt-Nurnberg: {}'.format(bfs_path[1] if bfs_path[0] else 'Not found'))
    print('dfs Frankfurt-Nurnberg: {}'.format(dfs_path[1] if dfs_path[0] else 'Not found'))

    bfs_nopath = traverse(graph, 'Wurzburg', 'Kassel', extend_bfs_path)
    print('bfs Wurzburg-Kassel: {}'.format(bfs_nopath[1] if bfs_nopath[0] else 'Not found'))
    dfs_nopath = traverse(graph, 'Wurzburg', 'Kassel', extend_dfs_path)
    print('dfs Wurzburg-Kassel: {}'.format(dfs_nopath[1] if dfs_nopath[0] else 'Not found'))
    main()
