# Module to train a tree-based model on Squad match-ups
# A match-up consists of layer, team1 faction, and team 2 faction
# Learns to predict how balanced a Squad match-up is

# Python imports
import itertools
from pathlib import Path
import os

# 3rd party imports
import lightgbm as lgb
from loguru import logger
import pandas as pd

# Relative imports
from .custom_types import (
    Faction,
    Match
)
from .constants import (
    MATCH_FILTERS,
    MAP_GROUP_BY_LAYER,
    TEAMS_BY_GAME_MAP,
    ALLOWED_UNIT_TYPES_BY_MAP_GROUP,
    ALLIANCES_BY_TEAM
)
from .process_matches import get_model_data_from_match


def _generate_matches_by_layer() -> dict:
    UNIT_TYPE_FILTER_BY_TEAM = MATCH_FILTERS.UNIT_TYPE_FILTER_BY_TEAM
    matches_by_layer = {}
    for layer in MAP_GROUP_BY_LAYER:
        game_map = layer.game_map
        map_group = MAP_GROUP_BY_LAYER[layer]
        allowed_unit_types = ALLOWED_UNIT_TYPES_BY_MAP_GROUP[map_group]
        if layer not in matches_by_layer:
            matches_by_layer[layer] = []

        matches = matches_by_layer[layer]
        possible_teams = TEAMS_BY_GAME_MAP[game_map]
        possible_team_matches = itertools.permutations(possible_teams, 2)
        team_matches = [
            (team1, team2, ) for (team1, team2, ) in possible_team_matches
            if ALLIANCES_BY_TEAM[team1] != ALLIANCES_BY_TEAM[team2]
        ]
        for team1, team2 in team_matches:
            all_team1_unit_types = UNIT_TYPE_FILTER_BY_TEAM[team1].values()
            team1_unit_types = {
                unit_type for unit_type in iter(all_team1_unit_types)
                if unit_type in allowed_unit_types
            }

            all_team2_unit_types = UNIT_TYPE_FILTER_BY_TEAM[team2].values()
            team2_unit_types = {
                unit_type for unit_type in iter(all_team2_unit_types)
                if unit_type in allowed_unit_types
            }

            unit_type_combos = itertools.product(team1_unit_types, team2_unit_types)
            for team1_unit_type, team2_unit_type in unit_type_combos:
                match = Match(
                    layer,
                    Faction(team1, team1_unit_type),
                    Faction(team2, team2_unit_type),
                    -1,
                    -1
                )
                matches.append(match)

    return matches_by_layer


def _analyze_match(booster: lgb.Booster, match: Match) -> float:
    model_data = get_model_data_from_match(
        match=match,
        k_fold_idx=-1
    )
    prediction_input = model_data.to_prediction_input()
    y_pred = booster.predict(prediction_input, validate_features=True)
    return y_pred[0]
# end _analyze_match() definition


def _generate_result_files(
        booster_balance: lgb.Booster,
        booster_volatility: lgb.Booster,
        results_folder: Path,
        matches_by_layer: dict
    ) -> None:
    for layer in matches_by_layer:
        logger.info(f"Analyzing layer: {layer}")
        matches = matches_by_layer[layer]
        game_layer = layer.game_repr()
        result_file = results_folder / game_layer
        analyzed_matches = []
        for match in matches:
            y_balance = _analyze_match(booster_balance, match)
            y_volatility = _analyze_match(booster_volatility, match)
            analyzed_matches.append((match, y_balance, y_volatility, ))

        analyzed_matches = sorted(
            analyzed_matches,
            key=lambda x : abs(x[1])
        )
        with open(result_file.with_suffix(".csv"), "w") as file:
            logger.info(f"Writing results to file: {result_file}\n")
            for match, pred_balance, pred_vol in analyzed_matches:
                file.write(
                    f"{match.team1_faction},"
                    f"{match.team2_faction},"
                    f"{pred_balance},"
                    f"{pred_vol}\n"
                )

    return
# end _generate_result_files() definition


@logger.catch
def predict_matches(
        model_balance_file: str,
        model_volatility_file: str,
        results_folder: str
    ) -> None:
    ABORT_MESSAGE = "Aborting predict_matches()"
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True)
    if len(os.listdir(results_folder)) != 0:
        logger.info(
            f"results folder: {results_folder} not empty\n"
            f"{ABORT_MESSAGE}"
        )
        return

    model_balance_file = Path(model_balance_file)
    if not model_balance_file.is_file():
        logger.error(
            f"model file: {model_balance_file} doesn't exist\n"
            f"{ABORT_MESSAGE}"
        )
        return

    model_volatility_file = Path(model_volatility_file)
    if not model_volatility_file.is_file():
        logger.error(
            f"model file: {model_volatility_file} doesn't exist\n"
            f"{ABORT_MESSAGE}"
        )
        return

    booster_balance = lgb.Booster(model_file=model_balance_file)
    booster_volatility = lgb.Booster(model_file=model_volatility_file)
    matches_by_layer = _generate_matches_by_layer()
    _generate_result_files(
        booster_balance,
        booster_volatility,
        results_folder,
        matches_by_layer
    )
    return
# end predict_match_balance() definition
