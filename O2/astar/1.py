import math
import Map
import heapq
import numpy as np

# The following function solves both task 3 and 4 by using a min heap 
# ordered on a optimistic manhattan heuristic, allowing it to always find the shortest path, even with varying tile costs
def astar(map_obj: Map.Map_Obj, save_path: str):
    dirs = [(0,-1),(0,1),(-1,0),(1,0)]
    end = map_obj.get_end_goal_pos()
    start = map_obj.get_start_pos()
    int_map, str_map = map_obj.get_maps()
    # returns optimistic manhattan distance to end
    dist_to_end = lambda pos: abs(pos[0]-end[0])+abs(pos[1]-end[1])
    is_end = lambda pos: pos==end
    is_start = lambda pos: pos==start
    # min heap with tiles to explore as tuples (d(pos)+h(pos), h(pos), d(pos), pos), 
    # where pos is the coords, h is the manhattan heuristic and d is the traveled distance 
    explore = [( dist_to_end(start), dist_to_end(start), 0, start )]
    # map that tracks the minimum distance found to each tile
    dists =np.zeros(map_obj.get_maps()[1].shape, dtype=int)
    dists +=np.sum(dists.shape)
    dists[start[0],start[1]] = 0

    while True:
        # top priority tile 
        ( _, _, steps, curr ) = heapq.heappop(explore)
        # do not overwrite start and end
        if not(is_end(curr) or is_start(curr)):
            map_obj.set_cell_value(curr, ' O ')
        # check end
        if is_end(curr):
            break
        # loop though neighbors
        for d in dirs:
            new_pos = pos_add(curr, d)
            # outside the map
            if ( not in_map(*dists.shape, new_pos) ):
                continue
            # travel cost of the new tile {1,2,3,4}
            cost = int_map[new_pos[0], new_pos[1]]
            # found end
            if is_end(new_pos):
                heapq.heappush(explore, ( steps+cost+dist_to_end( new_pos ), dist_to_end(new_pos),steps+cost, new_pos ))
                break
            # check if moveable
            if not is_legal(new_pos, map_obj):
                continue
            # a better path to the tile has been found
            if steps+cost>=dists[new_pos[0], new_pos[1]]:
                continue
            # mark as explored and update the distance
            map_obj.set_cell_value(new_pos, ' E ')
            dists[new_pos[0], new_pos[1]] = steps + cost
            # add to queue
            heapq.heappush(explore, ( steps+cost+dist_to_end( new_pos ), dist_to_end(new_pos), steps+cost, new_pos ))
    # extract the path from the dists map and mark path in string map
    path = get_and_set_path(end, dists, map_obj, is_start)
    # display the map
    map_obj.show_map(save_path)
    return path, steps

# returns the shortest path found by the algorithm and inserts it into the string map 
def get_and_set_path(end, dists: np.ndarray, map_obj: Map.Map_Obj, is_start):
    dirs = [(0,-1),(0,1),(-1,0),(1,0)]
    path = [end]
    curr = end
    # start at the end and find the neighbor woth the lowest distance untill start is found
    while True:
        best = math.inf
        best_dir = dirs[0]
        for d in dirs:
            new = pos_add(curr, d)
            if dists[new[0],new[1]] < best:
                best =dists[new[0],new[1]] 
                best_dir = new
        path.append(best_dir)
        if is_start(best_dir):
            break
        map_obj.set_cell_value(best_dir, ' P ')
        curr = best_dir
    return path[::-1]


def pos_add(pos, direc):
    return [pos[0]+direc[0],pos[1]+direc[1]]

def in_map(h, w, pos):
    if pos[0] < 0 or pos[0] >= h:
        return False
    if pos[1] < 0 or pos[1] >= w:
        return False
    return True

def is_legal(pos, map_obj: Map.Map_Obj):
    str_map = map_obj.get_maps()[1]
    return str_map[pos[0]][pos[1]]!=' # '

map_obj1 = Map.Map_Obj(1)
map_obj2 = Map.Map_Obj(2)
map_obj3 = Map.Map_Obj(3)
map_obj4 = Map.Map_Obj(4)
astar(map_obj1,'3_1.png')
astar(map_obj2,'3_2.png')
astar(map_obj3,'4_1.png')
astar(map_obj4,'4_2.png')