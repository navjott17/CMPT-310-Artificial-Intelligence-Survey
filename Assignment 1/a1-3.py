# a1.py
import random
import time
import sys

from asyncio import PriorityQueue

from cytoolz import memoize
from search import *

# ...

'''----------------------------------------------------ANSWER 1----------------------------------------------------'''

'''makes a random eight puzzle problem that is solvable'''


def make_rand_8puzzle():
    while True:
        initial_list = []
        for i in range(9):
            temp = random.randint(0, 8)
            while temp in initial_list:
                temp = random.randint(0, 8)
            initial_list.append(temp)                       #appends the list
        eight_puzzle = EightPuzzle(initial_list)
        solvable = eight_puzzle.check_solvability(initial_list)

        if solvable:                                        #checks solvability
            return eight_puzzle
        else:
            continue


'''displays the eight puzzle problem'''


def display(state):
    list1 = state.initial
    for i in range(3):                                    #prints first three tiles
        if list1[i] == 0:
            print("*\t", end="")
        else:
            print(f'{list1[i]} \t', end="")
    print("\n", end="")
    i = 3
    while i < 6:                                          #prints next three tiles
        if list1[i] == 0:
            print("*\t", end="")
        else:
            print(f'{list1[i]} \t', end="")
        i = i + 1
    print("\n", end="")
    i = 6
    while i < 9:                                          #prints last three tiles
        if list1[i] == 0:
            print("*\t", end="")
        else:
            print(f'{list1[i]} \t', end="")
        i = i + 1
    print("\n", end="")


'''----------------------------------------------------ANSWER 2----------------------------------------------------'''

'''algorithm to solve eight puzzle problem using misplaced-tile hueristic'''


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            expanded_path = len(explored) + 1                 #number of expanded paths
            return node, expanded_path                        #return expanded path along with node
        explored.add(tuple(node.state))
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
        expanded_path = len(explored) + 1
    return None, expanded_path


'''A* algorithm to solve the eight puzzle problem using heuristics'''


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


'''----------------------------------------------------ANSWER 3----------------------------------------------------'''


'''returns the manhattan distance to solve the eight puzzle problem using that distance'''


def manhattan(n):
    manhattan_distance = 0
    state = n.state
    index_goal = {0: [2, 2], 1: [0, 0], 2: [0, 1], 3: [0, 2], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [2, 0], 8: [2, 1]}
    index_state = {}
    index = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    for i in range(len(state)):
        index_state[state[i]] = index[i]

    for i in range(1, 9):                       #range is from (1,9) not (0,8)
        for j in range(2):
            manhattan_distance = abs(index_goal[i][j] - index_state[i][j]) + manhattan_distance

    return manhattan_distance


'''----------------------------------------------------ANSWER 4----------------------------------------------------'''


'''returns gaschnig distance to solve the eight puzzle problem using that distance'''


def gaschnig(n):
    gaschnig_distance = 0                              #distance counter
    goal_list = [1, 2, 3, 4, 5, 6, 7, 8, 0]            #goal list
    initial_list = list(n.state)                       #initial list
    while True:                                        #loop continues till initial list == goal list
        if initial_list == goal_list:                  #check the terminating condition
            return gaschnig_distance
        else:
            star_index = initial_list.index(0)         #index of 0 in initial list
            temp = goal_list[star_index]
            if temp != 0:                              #if goal and initial list at star_index are not 0 then swaping occurs
                temp_index = initial_list.index(temp)
                initial_list[star_index] = temp
                initial_list[temp_index] = 0
            else:
                for i in range(len(n.state)):          #swaps the initial list with last element in order to sort them to reach goal state
                    if initial_list[i] == goal_list[i]:
                        continue
                    else:
                        initial_list[star_index] = initial_list[i]
                        initial_list[i] = 0
                        break
            gaschnig_distance = gaschnig_distance + 1


'''----------------------------------------------------ANSWER 5----------------------------------------------------'''


'''displays 10 random puzzles and calculates their time, number of tiles moved and expanded nodes'''


def random_puzzle():
    for i in range(10):
        print(f'\n\nPuzzle {i+1}:')
        eight_puzzle = make_rand_8puzzle()                   #make random puzzle
        display(eight_puzzle)                                #display puzzle
        print("Q2 MISPLACED-TILES HEURISTIC")
        start_time = time.time()                             #time for misplaced-tiles heuristic
        node, expanded_path = astar_search(eight_puzzle, eight_puzzle.h)
        elapsed_time = time.time() - start_time
        print(f'Total running time (in seconds): {elapsed_time}s')
        print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
        print(f'The total number of nodes that were expanded: {expanded_path}')

        print("Q3 MANHATTAN HEURISTIC")
        start_time = time.time()                             #time for manhattan heuristic
        node, expanded_path = astar_search(eight_puzzle, manhattan)
        elapsed_time = time.time() - start_time
        print(f'Total running time (in seconds): {elapsed_time}s')
        print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
        print(f'The total number of nodes that were expanded: {expanded_path}')

        print("Q4 GASCHNIG HEURISTIC")
        start_time = time.time()                             #time for gaschnig heuristic
        node, expanded_path = astar_search(eight_puzzle, gaschnig)
        elapsed_time = time.time() - start_time
        print(f'Total running time (in seconds): {elapsed_time}s')
        print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
        print(f'The total number of nodes that were expanded: {expanded_path}')


'''----------------------------------------------------MAIN----------------------------------------------------'''


if len(sys.argv) == 10:                #checks if user inputs the argument
    initial_list = []
    for j in range(1, len(sys.argv)):
        temp = int(sys.argv[j])
        initial_list.append(temp)
    eight_puzzle = EightPuzzle(initial_list)
    solvable = eight_puzzle.check_solvability(initial_list)
    if not solvable:
        eight_puzzle = make_rand_8puzzle()
else:                                    #else make random puzzle
    eight_puzzle = make_rand_8puzzle()
display(eight_puzzle)

print("Q2 MISPLACED-TILES HEURISTIC")
start_time = time.time()                  #time for misplaced-tiles heuristic
node, expanded_path = astar_search(eight_puzzle, eight_puzzle.h)
elapsed_time = time.time() - start_time
print(f'Total running time (in seconds): {elapsed_time}s')
print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
print(f'The total number of nodes that were expanded: {expanded_path}')

print("Q3 MANHATTAN HEURISTIC")
start_time = time.time()                  #time for manhattan heuristic
node, expanded_path = astar_search(eight_puzzle, manhattan)
elapsed_time = time.time() - start_time
print(f'Total running time (in seconds): {elapsed_time}s')
print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
print(f'The total number of nodes that were expanded: {expanded_path}')

print("Q4 GASCHNIG HEURISTIC")
start_time = time.time()                  #time for gaschnig heuristic
node, expanded_path = astar_search(eight_puzzle, gaschnig)
elapsed_time = time.time() - start_time
print(f'Total running time (in seconds): {elapsed_time}s')
print(f'The length (number of tiles moved) of the solution: {len(node.solution())}')
print(f'The total number of nodes that were expanded: {expanded_path}')

random_puzzle()                           # 10 random puzzles