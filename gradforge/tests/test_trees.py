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
    

def test_gini():
    a = [1, 0, 0, 1]
    b = [1, 1, 1, 1]

    gini_a = gini(a)
    gini_b = gini(b)

    weighted = weighted_gini(a, b)

    assert gini_a == 0.5
    assert gini_b == 0.0
    assert weighted == 0.25

if __name__=="__main__":
    test_gini()


    
