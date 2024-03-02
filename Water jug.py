from collections import deque

def get_transitions(state, capacities):
    transitions = []
    a, b = state
    ca, cb = capacities

    transitions.append((ca, b))        
    transitions.append((a, cb))        
    transitions.append((0, b))         
    transitions.append((a, 0))         
    transitions.append((min(a + b, ca), max(0, a + b - ca)))  
    transitions.append((max(0, a + b - cb), min(a + b, cb))) 

    return transitions
def solve_water_jug(start, end, capacities):
    visited = set()
    queue = deque([(start, [])])

    while queue:
        current_state, path = queue.popleft()
        if current_state == end:
            return path

        visited.add(current_state)

        transitions = get_transitions(current_state, capacities)
        for next_state in transitions:
            if next_state not in visited:
                queue.append((next_state, path + [next_state]))

    return None

# Example usage:
start_state = (0, 0)
end_state = (2, 0)  
jug_capacities = (4, 3)  

path = solve_water_jug(start_state, end_state, jug_capacities)
if path:
    print("Solution Path:", path)
else:
    print("No solution found!")
