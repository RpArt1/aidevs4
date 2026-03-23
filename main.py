from common import setup_logging, get_logger
from assignments.lesson2.l2 import Lesson2

setup_logging()
log = get_logger(__name__)



def main():
    log.info("Starting Lesson 2")
    lesson2 = Lesson2()
    results = lesson2.solve()
    print(results)

if __name__ == "__main__":
    main()
    