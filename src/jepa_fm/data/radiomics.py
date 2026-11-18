import logging

import numpy as np
import SimpleITK as sitk
import torch


class RadiomicsExtractor:
    """pyradiomics features of one view, as a fixed-length vector.

    shortcut: the ROI is the whole crop (all-ones mask), so shape features are
    constant and are left disabled; load the real mask and augment it jointly with
    the image if shape features are wanted.
    """

    def __init__(
        self,
        bin_width: float = 0.01,
        feature_classes: tuple[str, ...] = ("firstorder", "glcm", "glrlm"),
        label: int = 1,
    ):
        from radiomics import featureextractor, setVerbosity # lazy: optional dependency

        setVerbosity(logging.ERROR)
        self.label = label
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            binWidth=bin_width, label=label
        )
        self.extractor.disableAllFeatures()
        for name in feature_classes:
            self.extractor.enableFeatureClassByName(name)

    def __call__(self, volume: torch.Tensor) -> np.ndarray:
        """(D, H, W) float view -> (F,) float32 features, in sorted-name order."""
        array = volume.detach().cpu().numpy()
        mask = np.ones(array.shape, dtype=np.uint8)
        result = self.extractor.execute(
            sitk.GetImageFromArray(array), sitk.GetImageFromArray(mask), label=self.label
        )
        return np.array(
            [float(v) for k, v in sorted(result.items()) if not k.startswith("diagnostics")],
            dtype=np.float32,
        )
