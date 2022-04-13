import os
import unittest

from models.unet import get_unet_model
from train import load_data, load_dataset, shuffling


class TestClassifier(unittest.TestCase):

    def test_model(self):
        """
        Check the input layer anf the output layer of the model.
        Args:
            None
        Returns:
            None
        """
        model = get_unet_model((512, 512, 3))
        # Check the image height
        self.assertEqual(
            model.get_layer("input_1").input_shape[0][0], tuple(model.get_layer("conv2d_18").output.shape)[0]
        )
        # Check the image width
        self.assertEqual(
            model.get_layer("input_1").input_shape[0][1], tuple(model.get_layer("conv2d_18").output.shape)[1]
        )

    def test_dataloader(self):
        """
        Check the number of images and mask of the training anf validation pipeline.
        Args:
            None
        Returns:
            None
        """
        dataset_path = os.path.join("new_data")
        train_path = os.path.join(dataset_path, "train")
        valid_path = os.path.join(dataset_path, "valid")

        train_x, train_y = load_data(train_path)
        train_x, train_y = shuffling(train_x, train_y)
        valid_x, valid_y = load_data(valid_path)

        train_dataset = load_dataset(train_x, train_y, batch_size=16)
        valid_dataset = load_dataset(valid_x, valid_y, batch_size=16)

        train_image, train_mask = next(iter(train_dataset))
        valid_image, valid_mask = next(iter(valid_dataset))

        self.assertEqual(
            len(train_image), len(train_mask)
        )
        self.assertEqual(
            len(valid_image), len(valid_mask)
        )


if __name__ == "__main__":
    unittest.main()
