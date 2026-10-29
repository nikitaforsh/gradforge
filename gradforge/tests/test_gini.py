def gini(array: list[int]):
    element_counter: dict = {}
    for element in array:
        if element not in element_counter:
            element_counter[element] = 1
        else:
            element_counter[element] += 1

    raw_probabilities = list(element_counter.values())
    gini_probabilities = [(p / len(array)) ** 2 for p in raw_probabilities]
    return sum(gini_probabilities)
    

def test_gini():
    a = [1, 0, 0, 1]
    b = [1, 1, 1, 1]

    gini_a = gini(a)
    gini_b = gini(b)

    assert gini_a == 0.5
    assert gini_b == 1.0

test_gini()


    
