import numpy as np
import os
from PIL import Image
import h5py

from factor_dataset import FactorImageDataset
from simple_worlds.transform_image import TransformImage


def load_factor_data(data, root_path=None, **kwargs):
    if data == "sim_toy":
        return get_sim_toy(root_path)
    elif data == "dsprites":
        return get_dsprites(root_path)
    elif data == "cars":
        return get_cars(root_path)
    elif data == "modelnet_colors":
        return get_modelnet_colors(root_path, **kwargs)
    elif data == "pixel":
        return get_transformed_pixel(**kwargs)
    elif data == "arrow":
        return get_transformed_arrow(root_path, **kwargs)
    elif data == "transformed_image":
        return get_transformed_image(**kwargs)
    else:
        raise Exception(f"ERROR: unknown data {data}")


def get_sim_toy(root_path):
    assert root_path is not None, "project root path is not supplied"
    filepath = os.path.join(root_path, "data", "sim_toy_ordered", "sim_toy_np_ordered.npz")
    with np.load(filepath, mmap_mode="r") as f:
        images = f["images"]
    images = images.astype('float32') / 255.
    images = images.reshape((4, 4, 2, 3, 3, 40, 40, 64, 64, 3))
    factor_names = ["object_color", "object_shape", "object_size", "camera_height", "background_color",
                    "horizontal_axis", "vertical_axis"]
    return FactorImageDataset(images, factor_names=factor_names)


def get_cars(root_path):
    from modules.utils.disentanglement_load_folder import cars3d
    assert root_path is not None, "project root path is not supplied"
    cars_path = os.path.join(root_path, "data", "cars")
    cars_class = cars3d.Cars3D(cars_path)
    images = cars_class.images  # np array of shape (24 * 4 * 183, 64, 64, 3) containing 0's and 1's
    images = images.reshape((4, 24, 183, 64, 64, 3))
    max_factor_values = [4, 2 * np.pi, 183]
    factor_names = ["inclination", "rotation", "car_type"]
    return FactorImageDataset(images, max_factor_values=max_factor_values, factor_names=factor_names)


def get_dsprites(root_path):
    from modules.utils.disentanglement_load_folder import dsprites
    assert root_path is not None, "project root path is not supplied"
    dsprites_path = os.path.join(root_path, "data", "dsprites", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    dsprites_class = dsprites.DSprites(dsprites_path)
    images = dsprites_class.images  # np array of shape (737280, 64, 64) containing 0's and 1's
    images = images.reshape(images.shape + (1,))
    images = images.reshape((3, 6, 40, 32, 32, 64, 64, 1))
    max_factor_values = [3, 6, 2 * np.pi, 32, 32]
    factor_names = ["shape", "scale", "orientation", "x_pos", "y_pos"]
    return FactorImageDataset(images, max_factor_values=max_factor_values, factor_names=factor_names)


def get_transformed_pixel(height=32, width=32, square_size=1, **kwargs_transform):
    pixel_img = np.zeros((height, width, 3))  # TODO: changed to RBG (depth=3 instead of 1)
    pixel_img[0:square_size, 0:square_size, :] = np.array([1, 1, 1])
    return TransformImage(pixel_img, **kwargs_transform)


def get_transformed_image(image, **kwargs_transform):
    return TransformImage(image, **kwargs_transform)


def get_transformed_arrow(root_path, arrow_size=32, **kwargs_transform):
    assert arrow_size in [32, 64, 128], "arrow size not supported"
    image_rgba = Image.open(os.path.join(root_path, "data", "single_images", f"arrow_{arrow_size}.png"))
    image_rgb = image_rgba.convert("RGB")
    arrow_img = np.asarray(image_rgb)
    arrow_img = arrow_img.astype('float32') / 255.
    return TransformImage(arrow_img, **kwargs_transform)


def get_modelnet_colors(root_path, dataset_filename, object_type=None, normalize=True):
    """
    Returns a TransformImage object created from ModelNet40 dataset of objects with periodic colors and rotated
    Args:
        root_path: path to the root of the project
        dataset_filename: filename of the .h5 data to be loaded
        object_type: type of object saved in the data file
        normalize: whether data should be in the range [0,1] (True) or [0, 255] (False).

    Returns:
        FactorImageDataset object
    """
    dataset_filepath = os.path.join(root_path, "data", "modelnet40", dataset_filename)
    # Read the images
    images = read_data_h5(dataset_filepath, object_type, "images")
    if normalize:
        images = images.astype('float32') / np.amax(images)

    # Read the factors
    colors = read_data_h5(dataset_filepath, object_type, "colors")
    views = read_data_h5(dataset_filepath, object_type, "views")

    # Convert integer range to angular range
    unique_angle_colors = np.unique(colors)
    unique_views = np.unique(views)
    # Create FactorImageDataset lists
    factor_values = [unique_angle_colors, unique_views]
    max_factor_values = [np.amax(factor) for factor in factor_values]

    return FactorImageDataset(images=images,
                              factor_values_list=factor_values,
                              max_factor_values=max_factor_values,
                              factor_names=["colors_angle", "rotation_angle"])


# noinspection PyBroadException
def read_data_h5(data_filepath, object_type, data_type):
    """
    Args:
        data_filepath: path to the .h5 file from which data is loaded
        object_type: if None return all object types.
        data_type: data types available are images, colors and views
    Returns:
    """
    with h5py.File(data_filepath, "r") as file:
        # Get the data
        if object_type is None:
            for object_ in file.keys():
                object_data = file.get(object_)
                for identity in object_data.keys():
                    ids_data = object_data.get(identity)
                    data = np.array(ids_data.get(data_type))
        else:
            try:
                object_data = file.get(object_type)
                for identity in object_data.keys():
                    ids_data = object_data.get(identity)
                    data = np.array(ids_data.get(data_type))
            except:
                print(
                    f"Data with object type: {object_type} and data type {data_type} is not available in {data_filepath}")
                data = None
    return data
