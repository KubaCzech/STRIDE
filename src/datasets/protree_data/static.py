from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, get_args, TypeAlias

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

from src.datasets.protree_data.utils import BinaryScaler
from src.recurrence.protree.meta import RANDOM_SEED
from src.recurrence.protree.transformations import MultilabelHotEncoder

DEFAULT_DATA_DIR = os.environ.get("DEFAULT_DATA_DIR", "./data")

TStationaryDataset: TypeAlias = Literal["breast_cancer", "caltech", "compass", "diabetes", "mnist", "rhc"]
categorical_columns: dict[TStationaryDataset, list[str]] = {
    "breast_cancer": [],
    "caltech": [],
    "compass": ["age_cat", "priors_count", "c_charge_degree"],
    "diabetes": [],
    "mnist": [],
    "rhc": [
        "ca",
        "death",
        "sex",
        "dth30",
        "swang1",
        "dnr1",
        "ninsclas",
        "resp",
        "card",
        "neuro",
        "gastr",
        "renal",
        "meta",
        "hema",
        "seps",
        "trauma",
        "ortho",
        "race",
        "income",
    ],
}


class StationaryDataset:
    def __init__(
        self,
        name: TStationaryDataset,
        directory: str = DEFAULT_DATA_DIR,
        encode_categorical_variables: bool = True,
        normalise: bool = False,
        lazy: bool = True,
    ) -> None:
        """Dataset is a class designed to handle datasets for machine learning tasks. It provides functionalities for loading,
         transforming, and accessing training, validation, and test datasets.

        :param name: The name of the dataset.
        :param directory: The directory where the dataset files are located. Defaults to the DEFAULT_DATA_DIR.
        :param encode_categorical_variables: Whether to encode categorical variables. Defaults to True.
        :param lazy: Whether to lazy load the datasets. If True, the datasets are loaded only when accessed. Defaults to True.
        """
        self.name = name
        self.directory = Path(directory)
        self.encode_categorical_variables = encode_categorical_variables
        self.normalise = normalise
        self.lazy = lazy

        self._x_encoder = None
        self._x_binary_encoder = None
        self._y_encoder = None

        self._x_scaler = None

        sub_datasets = ["train", "valid", "test"]
        for ds in sub_datasets:
            setattr(self, f"_{ds}", None)

        if not lazy:
            for ds in sub_datasets:
                getattr(self, ds)

    def _x_transform(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        if self.encode_categorical_variables and categorical_columns[self.name]:
            if not self._x_encoder:
                self._x_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1, dtype=int
                ).fit(x)
            x = pd.DataFrame(data=self._x_encoder.transform(x), columns=columns)
        if self.normalise:
            if not self._x_scaler:
                self._x_scaler = MinMaxScaler().fit(x)
            if not self._x_binary_encoder:
                self._x_binary_encoder = BinaryScaler().fit(x, y)
            x = pd.DataFrame(data=self._x_scaler.transform(x), columns=columns)
            x = pd.DataFrame(data=self._x_binary_encoder.transform(x), columns=columns)
        return x

    def _y_transform(self, y: pd.DataFrame) -> pd.DataFrame:
        if y.shape[1] > 1:
            if not self._y_encoder:
                self._y_encoder = MultilabelHotEncoder().fit(y)
            y = self._y_encoder.transform(y)
        return y

    def _read_file(self, dataset: str) -> pd.DataFrame:
        return pd.read_csv(self.directory / f"{self.name}_{dataset}.csv", index_col=[0])

    @staticmethod
    def x_y_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = df[[col for col in df.columns if "target" not in col]]
        y = df[[col for col in df.columns if "target" in col]]
        if isinstance(y, pd.Series):
            y = pd.DataFrame({"target": y})
        return x, y

    def _get_x_y(self, dataset: Literal["train", "valid", "test"]) -> tuple[pd.DataFrame, pd.DataFrame]:
        x, y = StationaryDataset.x_y_split(self._read_file(dataset))
        x = self._x_transform(x, y)
        y = self._y_transform(y)
        return x, y

    @property
    def train(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Training dataset."""
        if not self._train:
            self._train = self._get_x_y("train")
        return self._train

    @property
    def valid(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validation dataset."""
        if not self._valid:
            self._valid = self._get_x_y("valid")
        return self._valid

    @property
    def test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Testing dataset."""
        if not self._test:
            self._test = self._get_x_y("test")
        return self._test

    @property
    def x_cols(self) -> list[str]:
        """List of feature columns"""
        return self.train[0].columns.tolist()

    @property
    def y_cols(self) -> list[str]:
        """List of label columns"""
        return self.train[1].columns.tolist()


def _save_dataframe(df: pd.DataFrame, dataset_name: str, directory: str = DEFAULT_DATA_DIR) -> None:
    path = Path(directory) / f"{dataset_name}.csv"
    df.reset_index(drop=True).to_csv(path)


def _split_dataframe_and_save(
    df: pd.DataFrame,
    dataset_name: str,
    directory: str = DEFAULT_DATA_DIR,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
) -> None:
    if train_size + valid_size + test_size != 1.0:
        raise ValueError("Sum of train, valid, and test sizes have to sum up to 1.0")

    from sklearn.model_selection import train_test_split

    y_column = [col for col in df.columns if "target" in col][0]

    train, valid_test = train_test_split(df, stratify=df[[y_column]], train_size=train_size, random_state=RANDOM_SEED)
    valid, test = train_test_split(
        valid_test, stratify=valid_test[[y_column]], train_size=valid_size / (valid_size + test_size), random_state=RANDOM_SEED
    )

    _save_dataframe(df=train, dataset_name=f"{dataset_name}_train", directory=directory)
    _save_dataframe(df=valid, dataset_name=f"{dataset_name}_valid", directory=directory)
    _save_dataframe(df=test, dataset_name=f"{dataset_name}_test", directory=directory)


def _download_diabetes(directory: str = DEFAULT_DATA_DIR) -> None:
    from io import StringIO
    from requests import get

    response = get(
        url="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", params={"downloadformat": "csv"}
    )

    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    df = df.rename(columns={"Outcome": "target"})
    _split_dataframe_and_save(df=df, dataset_name="diabetes", directory=directory)


def _download_breast_cancer(directory: str = DEFAULT_DATA_DIR) -> None:
    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer(as_frame=True)
    _split_dataframe_and_save(df=breast_cancer.frame, dataset_name="breast_cancer", directory=directory)


def _download_rhc(directory: str = DEFAULT_DATA_DIR) -> None:
    from io import StringIO
    from requests import get

    response = get(url="https://hbiostat.org/data/repo/rhc.csv", params={"downloadformat": "csv"})

    df = pd.read_csv(StringIO(response.content.decode("utf-8")), index_col=[0])
    df.insert(df.shape[1] - 1, "target", df.pop("cat1"))
    _split_dataframe_and_save(df=df, dataset_name="rhc", directory=directory)


def _download_compass(directory: str = DEFAULT_DATA_DIR) -> None:
    from io import StringIO
    from requests import get

    response = get(
        url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        params={"downloadformat": "csv"},
    )
    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    df = df[
        [
            "age",
            "c_charge_degree",
            "race",
            "age_cat",
            "score_text",
            "sex",
            "priors_count",
            "days_b_screening_arrest",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_jail_in",
            "c_jail_out",
        ]
    ]

    ix = df["days_b_screening_arrest"] <= 30 & (df["days_b_screening_arrest"] >= -30) & (df["is_recid"] != -1) & (
        df["c_charge_degree"] != "O"
    ) & (df["score_text"] != "N/A")
    df = df.loc[ix]
    df["length_of_stay"] = (pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])).apply(lambda x: x.days)

    df_cut = df.loc[~df["race"].isin(["Native American", "Hispanic", "Asian", "Other"]), :]
    df_cut_q = df_cut[
        ["sex", "race", "age_cat", "c_charge_degree", "score_text", "priors_count", "is_recid", "two_year_recid"]
    ].copy()
    df_cut_q["priors_count"] = df_cut_q["priors_count"].apply(
        lambda x: "0" if x <= 0 else ("1 to 3" if 1 <= x <= 3 else "More than 3")
    )
    df_cut_q["score_text"] = df_cut_q["score_text"].apply(lambda x: "MediumHigh" if (x == "High") | (x == "Medium") else x)
    df_cut_q["age_cat"] = df_cut_q["age_cat"].replace({"25 - 45": "25 to 45"})
    df_cut_q["sex"] = df_cut_q["sex"].replace({"Female": 1.0, "Male": 0.0})
    df_cut_q["race"] = df_cut_q["race"].apply(lambda x: 1.0 if x == "Caucasian" else 0.0)

    df = df_cut_q[["two_year_recid", "sex", "race", "age_cat", "priors_count", "c_charge_degree"]].rename(
        columns={"two_year_recid": "target"}
    )
    _split_dataframe_and_save(df=df, dataset_name="compass", directory=directory)


def _download_mnist(directory: str = DEFAULT_DATA_DIR) -> None:
    import gzip
    from tempfile import TemporaryDirectory

    from mnist import MNIST
    from requests import get
    from sklearn.model_selection import train_test_split

    train = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

    with TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        for f in train + test:
            response = get(
                url=f"http://yann.lecun.com/exdb/mnist/{f}",
            )
            with (tmp_dir / f).open("wb") as fp:
                fp.write(response.content)
            with gzip.open((tmp_dir / f)) as gzip_fp:
                with (tmp_dir / f).with_suffix("").open("wb") as bin_fp:
                    bin_fp.write(gzip_fp.read())

        mnist = MNIST(_tmp_dir)

        test = mnist.load_testing()
        test_df = pd.DataFrame(
            data=[pixels + [label] for pixels, label in zip(test[0], test[1]) if label in (4, 9)],
            columns=[f"x{i}" for i in range(784)] + ["target"],
        )
        _save_dataframe(df=test_df, dataset_name="mnist_test", directory=directory)

        train_valid = mnist.load_training()
        train_valid_df = pd.DataFrame(
            data=[pixels + [label] for pixels, label in zip(train_valid[0], train_valid[1]) if label in (4, 9)],
            columns=[f"x{i}" for i in range(784)] + ["target"],
        )

        train, valid = train_test_split(
            train_valid_df, stratify=train_valid_df[["target"]], test_size=test_df.shape[0], random_state=RANDOM_SEED
        )
        _save_dataframe(df=train, dataset_name="mnist_train", directory=directory)
        _save_dataframe(df=valid, dataset_name="mnist_valid", directory=directory)


def _download_caltech(directory: str = DEFAULT_DATA_DIR) -> None:
    from tempfile import TemporaryDirectory

    import torch
    from torchvision import transforms
    from torchvision.datasets import Caltech256
    from torchvision.models.resnet import resnet50, ResNet50_Weights

    def collate_fn(batch: list[tuple[torch.Tensor, int]]):
        images = []
        labels = []
        for idx, (image, label) in enumerate(batch):
            images.append(image.repeat(3, 1, 1) if image.shape[0] == 1 else image)
            labels.append(label)
        return torch.stack(images, dim=0), labels

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    resnet = resnet50(ResNet50_Weights.IMAGENET1K_V2).to(device)

    with TemporaryDirectory() as _tmp_dir:
        caltech = Caltech256(root=_tmp_dir, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(caltech, batch_size=32, shuffle=False, collate_fn=collate_fn)

        x = []
        y = []

        for i, (images, labels) in enumerate(dataloader):
            vector = resnet(images.to(device))
            x.extend(vector.cpu().tolist())
            y.extend(labels)

        df = pd.DataFrame(data=x, columns=[f"x{i + 1}" for i in range(len(x[0]))])
        df["target"] = y
    df = df[df["target"].isin((62, 135))]
    _split_dataframe_and_save(df=df, dataset_name="caltech", directory=directory)


def download_all(
    directory: str = DEFAULT_DATA_DIR, dataset_names: list[Literal["all"] | TStationaryDataset] = "all", verbose: bool = True
) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    allowable_datasets = get_args(TStationaryDataset)

    if "all" in dataset_names:
        dataset_names = allowable_datasets

    for dataset_name in dataset_names:
        if dataset_name not in allowable_datasets:
            raise NameError(f"Unknown dataset: {dataset_name}")
        if verbose:
            print(f"Processing {dataset_name}...")
        globals()[f"_download_{dataset_name}"](directory=directory)
