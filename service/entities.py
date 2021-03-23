import typing as t
import typing_extensions as te

from pydantic import BaseModel, Field, ConstrainedInt, PositiveInt, PositiveFloat


# class ModelInput(BaseModel):
#     YrSold: int
#     YearBuilt: int
#     YearRemodAdd: int
#     GarageYrBlt: int
#     LotArea: int
#     Neighborhood: str
#     HouseStyle: str


# NeighborhoodLiteral = te.Literal[
#     "Blmgtn",
#     "Blueste",
#     "BrDale",
#     "BrkSide",
#     "ClearCr",
#     "CollgCr",
#     "Crawfor",
#     "Edwards",
#     "Gilbert",
#     "IDOTRR",
#     "Meadow",
#     "Mitchel",
#     "Names",
#     "NoRidge",
#     "NPkVill",
#     "NridgHt",
#     "NWAmes",
#     "OldTwon",
#     "SWISU",
#     "Sawyer",
#     "SawyerW",
#     "Somerst",
#     "StoneBr",
#     "Timber",
#     "Veenker",
# ]
# HouseStyleLiteral = te.Literal[
#     "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"
# ]


# class ModelInput(BaseModel):
#     YrSold: PositiveInt
#     YearBuilt: PositiveInt
#     YearRemodAdd: PositiveInt
#     GarageYrBlt: PositiveInt
#     LotArea: PositiveFloat
#     Neighborhood: NeighborhoodLiteral
#     HouseStyle: HouseStyleLiteral


class boolInteger(ConstrainedInt):
    ge = 0
    le = 1

class mnthInteger(ConstrainedInt):
    ge = 0
    le = 12

class hrInteger(ConstrainedInt):
    ge = 0
    le = 24

class weekInteger(ConstrainedInt):
    ge = 0
    le = 12




class ModelInput(BaseModel):
    yr: boolInteger
    mnth: mnthInteger
    hr: hrInteger
    holiday: boolInteger
    weekday: weekInteger
    workingday: boolInteger
    weathersit: mnthInteger
    temp: PositiveFloat
    atemp: PositiveFloat
    hum: PositiveFloat
    windspeed: PositiveFloat


