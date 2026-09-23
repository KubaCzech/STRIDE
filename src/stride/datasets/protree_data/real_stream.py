from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias, Literal

import pandas as pd
from scipy.io import arff

TRealStream: TypeAlias = Literal["airline", "pakdd", "poker", "sensor"]


class RealStreamGeneratorFactory:
    @staticmethod
    def create(name: TRealStream) -> IRealStream:
        name = "".join([n.capitalize() for n in name.split("_")])
        return globals()[f"{name}"]()


class IRealStream(ABC):
    def __init__(self, block_size: int = 1000) -> None:
        self.block_size = block_size
        self.df = self._read_data()

        self._iter_counter = 0

    @abstractmethod
    def _read_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.df)

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            if self._iter_counter >= len(self.df):
                break
            row = self.df.iloc[self._iter_counter]
            x = row.drop("target").to_dict()
            y = int(row["target"])
            result.append((x, y))
            self._iter_counter += 1
        return result

    def __iter__(self) -> IRealStream:
        self._iter_counter = 0
        return self

    def __next__(self) -> tuple[dict[str, float], int]:
        if self._iter_counter >= len(self.df):
            raise StopIteration
        row = self.df.iloc[self._iter_counter]
        x = row.to_dict()
        y = int(row["target"])
        self._iter_counter += 1
        return x, y

    def reset(self) -> None:
        self._iter_counter = 0


class Airline(IRealStream):
    @staticmethod
    def _group_airline(airline: str) -> int:
        if airline in {"UA", "AA", "DL", "WN", "US", "CO", "B6", "AS"}:
            return 2
        elif airline in {"MQ", "OH", "EV", "9E", "YV", "OO", "XE"}:
            return 1
        else:
            return 0

    @staticmethod
    def _group_airport(airport: str) -> int:
        major_hubs = {"ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "CLT", "LAS", "PHX"}
        medium_airports = {"SEA", "MIA", "IAH", "MSP", "DTW", "BOS", "PHL", "LGA", "EWR", "BWI"}
        if airport in major_hubs:
            return 2
        elif airport in medium_airports:
            return 1
        else:
            return 0

    def _read_data(self) -> pd.DataFrame:
        df = arff.loadarff("data/airlines.arff")
        df = pd.DataFrame(df[0])
        df["DayOfWeek"] = df["DayOfWeek"].astype(int)
        df["Delay"] = df["Delay"].astype(int)
        df["Airline"] = df["Airline"].apply(Airline._group_airline)
        df["AirportFrom"] = df["AirportFrom"].apply(Airline._group_airport)
        df["AirportTo"] = df["AirportTo"].apply(Airline._group_airport)
        df = df.rename(columns={"Delay": "target"})
        df = df.drop(columns=["AirportFrom", "AirportTo"])
        return df


class Poker(IRealStream):
    def _read_data(self) -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        poker_hand = fetch_ucirepo(id=158)
        df = poker_hand.data.features
        df["target"] = poker_hand.data.targets

        return df


class Sensor(IRealStream):
    def _read_data(self) -> pd.DataFrame:
        df = arff.loadarff("data/sensor.arff")
        df = pd.DataFrame(df[0])
        df["class"] = df["class"].astype(int)
        df = df.rename(columns={"class": "target"})
        df = df.drop(columns=["rcdminutes"])
        return df


class Pakdd(IRealStream):
    def _read_data(self) -> pd.DataFrame:
        df = arff.loadarff("data/pakdd.arff")
        df = pd.DataFrame(df[0])
        df["class"] = df["class"].str.decode("utf-8").astype(float).astype(int)
        df = df.rename(columns={"class": "target"})
        return df
