from typing import Dict
import torch
import numpy as np
import torchvision.transforms.functional as TF

torch.manual_seed(0)
np.random.seed(0)


class RandomRotation(object):
    def __init__(self) -> None:
        """
        Initializes the RandomRotation transform.
        """
        pass

    def __call__(self, sample: Dict) -> Dict:
        """
        Applies a random rotation (0, 90, 180, or 270 degrees) to both data and label in the sample.

        Args:
            sample (Dict): Dictionary with 'data' and 'lbl' keys.

        Returns:
            Dict: Rotated sample.
        """
        data, lbl = sample["data"], sample["lbl"]
        rnd = np.random.randint(0, 3)
        data = TF.rotate(img=data, angle=rnd * 90)
        lbl = TF.rotate(img=lbl, angle=rnd * 90)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p) -> None:
        """
        Initializes the RandomVerticalFlip transform.

        Args:
            p (float): Probability of applying the vertical flip.
        """
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        """
        Randomly applies a vertical flip to both data and label in the sample with probability p.

        Args:
            sample (Dict): Dictionary with 'data' and 'lbl' keys.

        Returns:
            Dict: Vertically flipped sample (or unchanged).
        """
        data, lbl = sample["data"], sample["lbl"]
        if torch.rand(1) < self.p:
            data = TF.vflip(data)
            lbl = TF.vflip(lbl)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p) -> None:
        """
        Initializes the RandomHorizontalFlip transform.

        Args:
            p (float): Probability of applying the horizontal flip.
        """
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        """
        Randomly applies a horizontal flip to both data and label in the sample with probability p.

        Args:
            sample (Dict): Dictionary with 'data' and 'lbl' keys.

        Returns:
            Dict: Horizontally flipped sample (or unchanged).
        """
        data, lbl = sample["data"], sample["lbl"]
        if torch.rand(1) < self.p:
            data = TF.hflip(data)
            lbl = TF.hflip(lbl)
        sample["data"] = data
        sample["lbl"] = lbl
        return sample
