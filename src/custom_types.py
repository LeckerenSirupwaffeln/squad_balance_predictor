# Python imports
from enum import StrEnum
from dataclasses import dataclass, fields
import io
from typing import Callable

# 3rd party imports
import pandas as pd


NUMERICAL_PREFIX = "numerical_"
CATEGORICAL_PREFIX = "categorical_"
FEATURES_PREFIX = "features_"
TARGET_PREFIX = "target_"

NUMERICAL_FEATURES_PREFIX = f"{NUMERICAL_PREFIX}{FEATURES_PREFIX}"
CATEGORICAL_FEATURES_PREFIX = f"{CATEGORICAL_PREFIX}{FEATURES_PREFIX}"
NUMERICAL_TARGET_PREFIX = f"{NUMERICAL_PREFIX}{TARGET_PREFIX}"
WEIGHT_COLUMN_NAME = "weight"
K_FOLD_INDEX_COLUMN_NAME = "k_fold_idx"



class MatchKey(StrEnum):
    SERVER = "server"
    SERVER_ID = "serverId"
    DLC = "dlc"
    MOD = "mod"
    GAME_MODE = "gamemode"
    MAP_CLASS = "mapClassname"
    LAYER = "layerName"
    START_TIME = "startTime"
    END_TIME = "endTime"
    DURATION = "duration"
    WINNING_TEAM = "winningTeam"
    WINNING_TEAM_ID = "winningTeamID"
    WINNING_SUBFACTION = "winningSubfaction"
    WINNING_TICKETS = "winningTickets"
    WINNING_KILLS = "winningKills"
    LOSING_TEAM = "losingTeam"
    LOSING_TEAM_ID = "losingTeamID"
    LOSING_SUBFACTION = "losingSubfaction"
    LOSING_TICKETS = "losingTickets"
    LOSING_KILLS = "losingKills"


class GameMap(StrEnum):
    AL_BASRAH = "AlBasrah"
    ANVIL = "Anvil"
    BELAYA = "Belaya"
    BLACK_COAST = "BlackCoast"
    CHORA = "Chora"
    FALLUJAH = "Fallujah"
    FOOLS_ROAD = "FoolsRoad"
    GOOSE_BAY = "GooseBay"
    GORODOK = "Gorodok"
    HARJU = "Harju"
    KAMDESH = "Kamdesh"
    KOHAT = "Kohat"
    KOKAN = "Kokan"
    LASHKAR = "Lashkar"
    LOGAR = "Logar"
    MANICOUAGAN = "Manicouagan"
    MESTIA = "Mestia"
    MUTAHA = "Mutaha"
    NARVA = "Narva"
    PACIFIC_PROVING_GROUNDS = "PacificProvingGrounds"
    SANXIAN = "Sanxian"
    SKORPO = "Skorpo"
    SUMARI = "Sumari"
    TALLIL = "Tallil"
    YEHORIVKA = "Yehorivka"


class GameMode(StrEnum):
    AAS = "AAS"
    RAAS = "RAAS"
    INVASION = "Invasion"


class MapVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class MapGroup(StrEnum):
    LARGE_WITH_HELI = "LARGE_WITH_HELI"
    LARGE_NO_HELI = "LARGE_NO_HELI"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"


@dataclass(frozen=True, eq=True)
class Layer:
    game_map: GameMap
    game_mode: GameMode
    map_version: MapVersion

    def __repr__(self) -> str:
        return f"{self.game_map},{self.game_mode},{self.map_version}"

    def game_repr(self) -> str:
        return f"{self.game_map}_{self.game_mode}_{self.map_version}"

    @staticmethod
    def column_repr() -> str:
        cls_fields = fields(Layer)
        prefixed_fields = [
            f"{CATEGORICAL_FEATURES_PREFIX}{field.name}" for field in cls_fields
        ]
        separator = ","
        columns = separator.join(prefixed_fields)
        return columns


class Team(StrEnum):
    ADF = "ADF"
    BAF = "BAF"
    CAF = "CAF"
    USA = "USA"
    USMC = "USMC"
    PLA = "PLA"
    PLAAGF = "PLAAGF"
    PLANMC = "PLANMC"
    VDV = "VDV"
    RGF = "RGF"
    MEA = "MEA"
    TLF = "TLF"
    INS = "INS"
    IMF = "IMF"
    WPMC = "WPMC"


class UnitType(StrEnum):
    AIR_ASSAULT = "AirAssault"
    ARMORED = "Armored"
    COMBINED_ARMS = "CombinedArms"
    LIGHT_INFANTRY = "LightInfantry"
    MECHANIZED = "Mechanized"
    MOTORIZED = "Motorized"
    SUPPORT = "Support"
    AMPHIBIOUS_ASSAULT = "AMPHIBIOUS_ASSAULT"


@dataclass(frozen=True, eq=True)
class Faction:
    team: Team
    unit_type: UnitType

    def __repr__(self) -> str:
        return f"{self.team},{self.unit_type}"

    @staticmethod
    def column_repr(suffix: str) -> str:
        cls_fields = fields(Faction)
        prefixed_fields = [
            f"{CATEGORICAL_FEATURES_PREFIX}{field.name}" for field in cls_fields
        ]
        suffixed_fields = [f"{field}{suffix}" for field in prefixed_fields]
        separator = ","
        columns = separator.join(suffixed_fields)
        return columns


class Alliance(StrEnum):
    BLUEFOR = "BLUEFOR"
    PAC = "PAC"
    REDFOR = "REDFOR"
    MEA = "MEA"
    TLF = "TLF"
    INS = "INS"
    IMF = "IMF"
    WPMC = "WPMC"


@dataclass(frozen=True, eq=True)
class Matchup:
    layer: Layer
    team1_faction: Faction
    team2_faction: Faction

    def __repr__(self) -> str:
        return (
            f"{self.layer},"
            f"{self.team1_faction},"
            f"{self.team2_faction}"
        )


@dataclass
class MatchupResult:
    tickets: list[int]
    ages_in_days: list[int]

    def __repr__(self) -> str:
        return (
            f"{self.tickets},"
            f"{self.ages_in_days}"
        )

    def append(self, tickets: int, age_in_days: int) -> None:
        self.tickets.append(tickets)
        self.ages_in_days.append(age_in_days)

    def get_average_tickets_and_weight(
            self,
            age_in_days_to_weight: Callable
        ) -> tuple[float, float]:
        sum_tickets = 0
        sum_weights = 0.0
        for ticket, age_in_days in zip(self.tickets, self.ages_in_days):
            weight = age_in_days_to_weight(age_in_days)
            sum_tickets += ticket * weight
            sum_weights += weight

        return (sum_tickets / sum_weights), sum_weights


@dataclass
class Match:
    layer: Layer
    team1_faction: Faction
    team2_faction: Faction
    tickets: int
    age_in_days: int

    def __repr__(self) -> str:
        return (
            f"{self.layer},"
            f"{self.team1_faction},"
            f"{self.team2_faction},"
            f"{self.tickets},"
            f"{self.age_in_days}"
        )


@dataclass(frozen=True, eq=True)
class TeamCombination:
    team1: Team
    team2: Team


@dataclass
class SubunitModel:
    hat_num: int
    tandem_rounds_per_hat: int
    bikes_num: int
    light_vehicle_logi_num: int
    wheeled_logi_num: int
    tracked_logi_num: int
    open_top_light_vic_num: int
    rws_50cal_light_vic_num: int
    weakap_armored_lav_num: int
    wheeled_ifv_num: int
    wheeled_ifv_average_dps: int
    wheeled_ifv_average_health: int
    tracked_ifv_num: int
    tracked_ifv_average_dps: int
    tracked_ifv_average_health: int
    tank_killing_ifvs: int
    mbt_num: int
    mbt_delay: int
    mgs_num: int
    mgs_delay: int
    transport_heli_num: int
    transport_heli_delay: int
    has_cas: int
    has_vehicle_mortars: int
    has_extra_hab: int
    tow_vehicle_num: int
    aa_vehicle_num: int

    def __repr__(self) -> str:
        return (
            f"{self.hat_num},"
            f"{self.tandem_rounds_per_hat},"
            f"{self.bikes_num},"
            f"{self.light_vehicle_logi_num},"
            f"{self.wheeled_logi_num},"
            f"{self.tracked_logi_num},"
            f"{self.open_top_light_vic_num},"
            f"{self.rws_50cal_light_vic_num},"
            f"{self.weakap_armored_lav_num},"
            f"{self.wheeled_ifv_num},"
            f"{self.wheeled_ifv_average_dps},"
            f"{self.wheeled_ifv_average_health},"
            f"{self.tracked_ifv_num},"
            f"{self.tracked_ifv_average_dps},"
            f"{self.tracked_ifv_average_health},"
            f"{self.tank_killing_ifvs},"
            f"{self.mbt_num},"
            f"{self.mbt_delay},"
            f"{self.mgs_num},"
            f"{self.mgs_delay},"
            f"{self.transport_heli_num},"
            f"{self.transport_heli_delay},"
            f"{self.has_cas},"
            f"{self.has_vehicle_mortars},"
            f"{self.has_extra_hab},"
            f"{self.tow_vehicle_num},"
            f"{self.aa_vehicle_num}"
        )

    @staticmethod
    def column_repr(suffix: str) -> str:
        cls_fields = fields(SubunitModel)
        prefixed_fields = [
            f"{NUMERICAL_FEATURES_PREFIX}{field.name}" for field in cls_fields
        ]
        suffixed_fields = [f"{field}{suffix}" for field in prefixed_fields]
        separator = ","
        columns = separator.join(suffixed_fields)
        return columns

    def remove_helis(self) -> None:
        self.transport_heli_num = 0
        self.transport_heli_delay = 0
        self.has_cas = 0


@dataclass
class ModelData:
    # categorical data
    layer: Layer
    team1_faction: Faction
    team2_faction: Faction

    # numerical and/or ordinal data
    team1_subunit_model: SubunitModel
    team2_subunit_model: SubunitModel

    tickets: float
    weight: float
    k_fold_index: int

    def __repr__(self) -> str:
        return (
            f"{self.layer},"
            f"{self.team1_faction},"
            f"{self.team2_faction},"
            f"{self.team1_subunit_model},"
            f"{self.team2_subunit_model},"
            f"{self.tickets},"
            f"{self.weight},"
            f"{self.k_fold_index}"
        )

    def to_prediction_input(self) -> pd.DataFrame:
        buffer = io.StringIO(
            f"{ModelData.prediction_input_column_repr()}\n"
            f"{self.layer},"
            f"{self.team1_faction},"
            f"{self.team2_faction},"
            f"{self.team1_subunit_model},"
            f"{self.team2_subunit_model}"
        )
        data_frame = pd.read_csv(
            buffer,
            header=0
        )
        for column_name in data_frame:
            if CATEGORICAL_FEATURES_PREFIX in column_name:
                data_frame[column_name] = data_frame[column_name].astype("category")

        return data_frame

    @staticmethod
    def column_repr() -> str:
        team1_suffix = "_1"
        team2_suffix = "_2"
        return (
            f"{Layer.column_repr()},"
            f"{Faction.column_repr(team1_suffix)},"
            f"{Faction.column_repr(team2_suffix)},"
            f"{SubunitModel.column_repr(team1_suffix)},"
            f"{SubunitModel.column_repr(team2_suffix)},"
            f"{NUMERICAL_TARGET_PREFIX}tickets,"
            f"{WEIGHT_COLUMN_NAME},"
            f"{K_FOLD_INDEX_COLUMN_NAME}"
        )

    @staticmethod
    def prediction_input_column_repr() -> str:
        team1_suffix = "_1"
        team2_suffix = "_2"
        return (
            f"{Layer.column_repr()},"
            f"{Faction.column_repr(team1_suffix)},"
            f"{Faction.column_repr(team2_suffix)},"
            f"{SubunitModel.column_repr(team1_suffix)},"
            f"{SubunitModel.column_repr(team2_suffix)}"
        )
