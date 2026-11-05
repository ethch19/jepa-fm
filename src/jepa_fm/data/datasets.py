import pandas as pd
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

_EXTENSIONS = ["csv", "parquet"]


class PretrainDataset(Dataset):
    def __init__(self, meta_dir: str, views: int):
        self.meta_dir = Path(meta_dir)
        self.views = views
        if views > 1:
            self.transform = self._transform()
        else:
            self.transform = self._inf_transform()

        data_dfs = []
        for file in self.meta_dir.rglob("*"):
            if file.is_file and any(file.name.endswith(ext) for ext in _EXTENSIONS):
                match file.suffix:
                    case "csv":
                        tabular = pd.read_csv(file)
                    case "parquet":
                        tabular = pd.read_parquet(
                            file, columns=["subject_id", "img_path", "seg_path"]
                        )
                    case _:
                        raise RuntimeError(f"Did not match any suffix for {file.name}")

                nii_only = (
                    tabular["img_path"].astype(str).str.endswith(".nii.gz", na=False)
                    & tabular["seg_path"].astype(str).str.endswith(".nii.gz", na=False)
                ).all()

                invalid = tabular[~nii_only]
                print(
                    f"Invalid files found in dataframe: {invalid.to_markdown(index=False)}"
                )

                data_dfs.append(nii_only)

        self.samples = (
            pd.concat(data_dfs, ignore_index=True) if data_dfs else pd.DataFrame()
        )

        self.samples = self.samples.sort_values(
            by="subject_id", ascending=True
        ).reset_index(drop=True)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No files found under {self.img_dir} with {_EXTENSIONS}"
            )

        self.subjects = self.samples["subject_id"].tolist()
        self.img_paths = self.samples["img_path"].tolist()
        self.seg_paths = self.samples["seg_path"].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = self._load_file(self.img_paths[idx])
        seg = self._load_file(self.seg_paths[idx])

        # TODO: 3D volume patch containing tumour
        # TODO: two views from different tumours
        # - a separate function/class for tumour patch retrieval
        # - can be adapted for same scan -> multiple tumours

        return (
            torch.stack([self.transform(img) for _ in range(self.V)]),
            self.subjects[idx],
        )

    def _load_file(file_path: str):
        img_path = Path(file_path)
        if img_path.is_file() and img_path.name.endswith("nii.gz"):
            img = nib.load(img_path)
            return img.get_fdata()
        else:
            raise RuntimeError(f"Non NIFTI-file found calling _load_file: {img_path}")

    # TODO: replace with grayscale image transformations
    def _transform():
        return v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _inf_transform():
        return v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
