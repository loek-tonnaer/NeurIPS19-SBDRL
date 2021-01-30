import argparse
import os
import numpy as np

import data_loader


def setup_random_sequence(n_steps=9999, grid_x=64, grid_y=64):
    # pick random starting point in grid, initialise sequence of grid points
    start = np.array([np.random.randint(grid_x), np.random.randint(grid_y)])
    sequence = np.empty((n_steps, 2), dtype=np.int64)
    sequence[0] = start

    # create random sequence of steps of +-1 in either x or y direction (but not both! XOR)
    actions = np.random.randint(4, size=n_steps, dtype=np.int64)
    random_steps = np.empty((n_steps, 2), dtype=np.int64)
    random_steps[actions == 0] = np.array([0, 5])  # down
    random_steps[actions == 1] = np.array([-5, 0])  # left
    random_steps[actions == 2] = np.array([0, -5])  # up
    random_steps[actions == 3] = np.array([5, 0])  # right

    # populate sequence
    for i in range(n_steps - 1):
        sequence[i + 1] = sequence[i] + random_steps[i]

    # wrap values to they fall within grid
    sequence[:, 0] = np.mod(sequence[:, 0], grid_x)
    sequence[:, 1] = np.mod(sequence[:, 1], grid_y)

    return sequence, actions


def main():
    # parse arguments
    desc = "Generate random observation sequence for given dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("data", choices=["arrow", "pixel", "modelnet"])
    args = parser.parse_args()

    if args.data == "arrow":
        data_params = {
            "data": "arrow",
            "arrow_size": 64,
            "n_hues": 64,
            "n_rotations": 64,
        }
    elif args.data == "pixel":
        data_params = {
            "data": "pixel",
            "height": 64,
            "width": 64,
            "step_size_vert": 1,
            "step_size_hor": 1,
            "square_size": 4,
        }
    elif args.data == "modelnet":
        data_params = {
            "dataset_filename": "modelnet_color_single_64_64.h5",
            "data": "modelnet_colors"
        }
    else:
        raise Exception()

    # load the correct data
    data = data_loader.load_factor_data(root_path=os.getcwd(), **data_params)

    # setup and save inputs & actions
    sequence, actions = setup_random_sequence()
    inputs = data.images[sequence[:, 0], sequence[:, 1]]
    inputs = inputs.astype("float32")  # convert to float32 (since that's flat Flatland also uses)

    np.save('inputs', inputs)
    np.save('actions', actions)

    print(f"inputs.npy and actions.npy successfully created for {args.data} dataset")


if __name__ == "__main__":
    main()
