import copy
import math
import numpy as np
import operator
end_state_matrix = []
list_matrices = []
cars = 0
obstacles = 0
network_size = 0
start = []
network_coordinates_dict = {}
cars_directions = []
alignments = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def extract_values_from_file():
    global end_state_matrix
    global list_matrices
    global cars
    global obstacles
    global network_size

    file = open("input.txt", "r")
    input_list = file.read().split('\n')
    network_size = int(input_list[0])
    cars = int(input_list[1])
    obstacles = int(input_list[2])

    network = [[-1 for j in range(network_size)] for i in range(network_size)]
    end = 3+obstacles
    for i in range(3, end):
        coordinate = input_list[i].split(',')
        network[int(coordinate[1])][int(coordinate[0])] = -101

    for i in range(end, end + cars):
        coordinate = input_list[i].split(',')
        start.append((int(coordinate[1]), int(coordinate[0])))

    list_matrices = []
    for i in range(end + cars, end + cars + cars):
        coordinate = input_list[i].split(',')
        end_state_matrix.append((int(coordinate[1]), int(coordinate[0])))
        network1 = copy.deepcopy(network)
        network1[int(coordinate[1])][int(coordinate[0])] = +99
        list_matrices.append(network1)


def output_into_file():
    with open('output.txt', 'w') as file:
        for i in range(0, len(output)):
            file.write("%s\n" % output[i])


def update_object(y, **actions):
    if isinstance(y, dict):
        y.update(actions)
    else:
        y.__dict__.update(actions)
    return y


def turn_right(alignment):
    return alignments[(alignments.index(alignment) + 3) % len(alignments)]


def turn_back(alignment):
    return alignments[(alignments.index(alignment) + 2) % len(alignments)]


def turn_left(alignment):
    return alignments[(alignments.index(alignment) + 1) % len(alignments)]


def state_exists(mm, tt, pp):
    if mm:
        if callable(tt): return tt()
        return tt
    else:
        if callable(pp): return pp()
        return pp


def addition(a, b):
    return tuple(map(operator.add, a, b))


class MarkovModel:
    def __init__(self, init, action_list, end_state, gamma=.9):
        update_object(self, init=init, action_list=action_list, end_state=end_state, gamma=gamma, states=set(), reward={})

    def get_reward(self, state):
        return self.reward[state]

    def take_turn(self, state, action):
        """Abstract Method"""

    def actions(self, state):
        if state == self.end_state:
            return [None]
        else:
            return self.action_list


class CarQuestion(MarkovModel):
    def __init__(self, network, end_state, init=(0, 0), gamma=.9):
        MarkovModel.__init__(self, init, action_list=alignments, end_state=end_state, gamma=gamma)
        update_object(self, network=network, rows=len(network), columns=len(network))
        for x in range(self.rows):
            for y in range(self.columns):
                self.reward[x, y] = network[x][y]
                if network[x][y] is not None:
                    self.states.add((x, y))

    def take_turn(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.7, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_back(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        state1 = addition(state, direction)
        return state_exists(state1 in self.states, state1, state)


def iteration(markov_object, eps=0.1):
    global network_coordinates_dict
    network_coordinates_dict = dict([(s, 0) for s in markov_object.states])
    utility_new = dict([(s, 0) for s in markov_object.states])
    gamma = markov_object.gamma
    while True:
        utility = utility_new.copy()
        delta = 0
        for s in markov_object.states:
            utility_new[s] = markov_object.get_reward(s) + gamma * max([sum([p * utility[s1] for (p, s1) in markov_object.take_turn(s, a)])
                                        for a in markov_object.actions(s)])
            delta = max(delta, abs(utility_new[s] - utility[s]))
        if delta < eps * (1 - gamma) / gamma:
            return utility


def get_optimal_directions(network, end_state):
    markov_object = CarQuestion(network, end_state=end_state)
    utility = iteration(markov_object)
    actions = {"North": (-1, 0), "South": (1, 0), "East": (0, 1), "West": (0, -1)}

    # For debug purposes
    coordinate_network = [[(0, 0) for c in range(network_size)] for d in range(network_size)]

    for item in utility:
        if item == end_state:
            coordinate = (0, 0)
        else:
            north = (item[0] + actions["North"][0], item[1] + actions["North"][1])
            south = (item[0] + actions["South"][0], item[1] + actions["South"][1])
            east = (item[0] + actions["East"][0], item[1] + actions["East"][1])
            west = (item[0] + actions["West"][0], item[1] + actions["West"][1])

            max_value = -100000000
            coordinate = (0, 0)

            if north in utility:
                if utility[north] > max_value:
                    max_value = utility[north]
                    coordinate = actions["North"]

            if south in utility:
                if utility[south] > max_value:
                    max_value = utility[south]
                    coordinate = actions["South"]

            if east in utility:
                if utility[east] > max_value:
                    max_value = utility[east]
                    coordinate = actions["East"]

            if west in utility:
                if utility[west] > max_value:
                    coordinate = actions["West"]

        coordinate_network[item[0]][item[1]] = coordinate

    return coordinate_network


def simulation():
    average_value_list = []
    for i in range(0, cars):
        # score = 0
        score_list = []
        for j in range(10):
            score = 0
            pos = start[i]
            np.random.seed(j)
            swerve = np.random.random_sample(1000000)
            k = 0
            current_alignment = cars_directions[i][pos[0]][pos[1]]
            while pos != end_state_matrix[i]:
                if swerve[k] > 0.7:
                    if swerve[k] > 0.8:
                        if swerve[k] > 0.9:
                            move = turn_back(current_alignment)
                            temp_position = (pos[0] + move[0], pos[1] + move[1])
                            if temp_position in network_coordinates_dict:
                                pos = (pos[0] + move[0], pos[1] + move[1])
                                current_alignment = cars_directions[i][pos[0]][pos[1]]
                            score += list_matrices[i][pos[0]][pos[1]]
                        else:
                            move = turn_right(current_alignment)
                            temp_position = (pos[0] + move[0], pos[1] + move[1])
                            if temp_position in network_coordinates_dict:
                                pos = (pos[0] + move[0], pos[1] + move[1])
                                current_alignment = cars_directions[i][pos[0]][pos[1]]
                            score += list_matrices[i][pos[0]][pos[1]]
                    else:
                        move = turn_left(current_alignment)
                        temp_position = (pos[0] + move[0], pos[1] + move[1])
                        if temp_position in network_coordinates_dict:
                            pos = (pos[0] + move[0], pos[1] + move[1])
                            current_alignment = cars_directions[i][pos[0]][pos[1]]
                        score += list_matrices[i][pos[0]][pos[1]]
                else:
                    move = current_alignment
                    temp_position = (pos[0] + move[0], pos[1] + move[1])
                    if temp_position in network_coordinates_dict:
                        pos = (pos[0] + move[0], pos[1] + move[1])
                        current_alignment = cars_directions[i][pos[0]][pos[1]]
                    score += list_matrices[i][pos[0]][pos[1]]

                k += 1
            score_list.append(score)
        average_value_list.append(int(math.floor(sum(score_list)/10)))
    return average_value_list


extract_values_from_file()

for i in range(0, cars):
    cars_directions.append(get_optimal_directions(list_matrices[i], end_state_matrix[i]))

output = simulation()
print(output)

output_into_file()
