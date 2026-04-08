from common import LLMService, AssignmentService, get_logger
from assignments.assignment import Assignment


class Lesson5_1(Assignment):
    def __init__(self):
        super().__init__("Lesson 5_1", "Lesson 5_1")

    def solve(self):
        return "Lesson 5_1"


if __name__ == "__main__":
    lesson5_1 = Lesson5_1()
    lesson5_1.log.info("Starting Lesson 5_1")
    lesson5_1.solve()

