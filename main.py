from common import setup_logging, get_logger
from assignments.lesson1.l1 import Lesson1

setup_logging()
log = get_logger(__name__)





def main():
    log.info("Hello, world!")
    lesson1 = Lesson1()
    results = lesson1.solve()
    print(results)

if __name__ == "__main__":
    main()
    