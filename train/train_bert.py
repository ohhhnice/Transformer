import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.set_seed import set_all_seeds


def main():
    SEED = 5
    set_all_seeds(SEED)
    pass



if __name__ == "__main__":
    main()