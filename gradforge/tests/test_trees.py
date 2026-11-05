def gini(array: list) -> float:
    element_counter: dict = {}
    for element in array:
        if element not in element_counter:
            element_counter[element] = 1
        else:
            element_counter[element] += 1

    raw_probabilities = list(element_counter.values())
    gini_probabilities = [(p / len(array)) ** 2 for p in raw_probabilities]
    return 1 - sum(gini_probabilities)

def weighted_gini(y_left: list, y_right: list) -> float:
    left_gini = len(y_left)/(len(y_left) + len(y_right)) * gini(y_left)
    right_gini = len(y_right)/(len(y_left) + len(y_right)) * gini(y_right)

    return left_gini + right_gini
    
def generate_tresholds(array: list) -> list:
    tresholds: list = []
    e_last = None
    for e in array:
        if e_last is None:
            e_last = e
            continue
        tresholds.append((e_last+e)/2)
        e_last = e
    return tresholds

def split_by_treshold(array:list, treshold: float) -> tuple[list]:
    return ([e for e in array if e < treshold], [e for e in array if e > treshold])

def find_best_split(array_x: list, array_y: list) -> float:
    best_treshold = None
    min_gini = None
    tresholds = generate_tresholds(array_x)
    for treshold in tresholds:
        left, right = split_by_treshold(array_x, treshold)
        left_y = array_y[:len(left)]
        right_y = array_y[len(left):]
        gini = weighted_gini(left_y, right_y)
        if best_treshold is None:
            best_treshold = treshold
            min_gini = gini
            continue
        if gini < min_gini:
            min_gini = gini
            best_treshold = treshold

    return best_treshold

def sort_input(X: list, y: list) -> tuple[list]:
    X, y = zip(*sorted(zip(X, y)))
    return (list(X), list(y))

def test_gini():
    a = [0, 1, 0, 1]
    b = [1, 1, 1, 1]
    data = [2, 6, 4, 8]
    gini_a = gini(a)
    gini_b = gini(b)
    weighted = weighted_gini(a, b)

    data, a = sort_input(data, a)
    treshold = find_best_split(data, a)
    left, right = split_by_treshold(data, treshold=treshold)

    assert gini_a == 0.5
    assert gini_b == 0.0
    assert weighted == 0.25
    assert left == [2, 4]
    assert right == [6, 8]

if __name__=="__main__":
    test_gini()