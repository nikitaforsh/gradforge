from ..split_criteria import SplitCriteria


class GiniCriteria(SplitCriteria):
    def __init__(self):
        super().__init__()

    def gini(self, array) -> float:
        element_counter: dict = {}
        for element in array:
            if element not in element_counter:
                element_counter[element] = 1
            else:
                element_counter[element] += 1

        raw_probabilities = list(element_counter.values())
        gini_probabilities = [(p / len(array)) ** 2 for p in raw_probabilities]
        return 1 - sum(gini_probabilities)

    def weighted_gini(self, y_left, y_right) -> float:
        left_gini = len(y_left)/(len(y_left) + len(y_right)) * self.gini(y_left)
        right_gini = len(y_right)/(len(y_left) + len(y_right)) * self.gini(y_right)

        return left_gini + right_gini