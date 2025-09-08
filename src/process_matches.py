# Module to process matches downloaded from MySquadStats API into model files
# CSV files will be used to train a ML model
# 1) Filter the matches to remove incorrect data
# 2) Convert the filtered matches to features and their targets
# 3) Stream results into file


# Python imports
import json
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone
from itertools import batched
from random import shuffle

# 3rd party imports
from loguru import logger

# Relative imports
from .constants import (
    K_FOLD_SPLITS,
    MATCH_FILTERS,
    MAP_GROUP_BY_LAYER
)
from .constants_subunit_model import (
    DEFAULT_SUBUNIT_MODEL_BY_FACTION,
    MEDIUM_SUBUNIT_MODEL_BY_FACTION,
    SMALL_SUBUNIT_MODEL_BY_FACTION
)
from .custom_types import (
    MatchKey,
    MapGroup,
    Layer,
    UnitType,
    Faction,
    Matchup,
    MatchupResult,
    Match,
    TeamCombination,
    SubunitModel,
    ModelData
)


def _filter_match(match: dict) -> Optional[tuple[Layer, Faction, int, Faction, int]]:
    """
    Filters a match and returns processed data if compliant

    Args:
        match (dict): Match data in a dictionary
        See enum "MatchKey" to see what keys are used in match

    Returns:
        None: Only if match did not pass filter, otherwise:

        Layer: The match's layer, i.e: Layer(
            GameMap.AL_BASRAH, GameMode.RAAS, MapVersion.V1
        )
        Faction: Team1 faction, i.e: Faction(ADF, AirAssault)
        int:     Team1's tickets, i.e: 150
        Faction: Team2 faction, i.e: Faction(INS, CombinedArms)
        int:     Team2 tickets, i.e: 0

    Raises:
        ValueError: If "winning_team_id" is not in (1, 2)
        KeyError: If any accessed keys do not exist
    """
    # start _filter_match() definition
    _server = match[MatchKey.SERVER]
    _server_id = match[MatchKey.SERVER_ID]
    dlc = match[MatchKey.DLC]
    mod = match[MatchKey.MOD]
    game_mode = match[MatchKey.GAME_MODE]
    _map_class = match[MatchKey.MAP_CLASS]
    layer = match[MatchKey.LAYER]
    _start_time = match[MatchKey.START_TIME]
    end_time = match[MatchKey.END_TIME]
    age_in_days = (datetime.now(timezone.utc) - datetime.fromisoformat(end_time)).days
    duration = match[MatchKey.DURATION]
    winning_team_id = match[MatchKey.WINNING_TEAM_ID]
    has_passed_filter = (
        dlc in MATCH_FILTERS.DLC_FILTER
        and mod in MATCH_FILTERS.MOD_FILTER
        and game_mode in MATCH_FILTERS.GAMEMODE_FILTER
        and duration > MATCH_FILTERS.MIN_DURATION_IN_SECONDS
        and duration < MATCH_FILTERS.MAX_DURATION_IN_SECONDS
        and winning_team_id is not None
        and age_in_days <= MATCH_FILTERS.MAX_AGE_IN_DAYS
        and age_in_days >= MATCH_FILTERS.MIN_AGE_IN_DAYS
    )
    if not has_passed_filter:
        return None

    match winning_team_id:
        case 1:
            team1 = match[MatchKey.WINNING_TEAM]
            team1_subfaction = match[MatchKey.WINNING_SUBFACTION]
            team1_tickets = match[MatchKey.WINNING_TICKETS]
            team1_kills = match[MatchKey.WINNING_KILLS]
            team2 = match[MatchKey.LOSING_TEAM]
            team2_subfaction = match[MatchKey.LOSING_SUBFACTION]
            team2_tickets = match[MatchKey.LOSING_TICKETS]
            team2_kills = match[MatchKey.LOSING_KILLS]
        case 2:
            team2 = match[MatchKey.WINNING_TEAM]
            team2_subfaction = match[MatchKey.WINNING_SUBFACTION]
            team2_tickets = match[MatchKey.WINNING_TICKETS]
            team2_kills = match[MatchKey.WINNING_KILLS]
            team1 = match[MatchKey.LOSING_TEAM]
            team1_subfaction = match[MatchKey.LOSING_SUBFACTION]
            team1_tickets = match[MatchKey.LOSING_TICKETS]
            team1_kills = match[MatchKey.LOSING_KILLS]
        case _:
            raise ValueError(
                f"winning_team_id: {winning_team_id}"
                ", but should be only 1 or 2"
                f"\nDumping match dict: {match}"
            )
    # end match winning_team_id

    is_missing_data = None in (
        team1,
        team1_subfaction,
        team1_tickets,
        team1_kills,
        team2,
        team2_subfaction,
        team2_tickets,
        team2_kills,
    )
    if is_missing_data:
        return None

    has_passed_filter = (
        team1_kills + team2_kills > MATCH_FILTERS.MIN_KILLS_BOTH_SIDES
        and team1 in MATCH_FILTERS.TEAM_FILTER
        and team2 in MATCH_FILTERS.TEAM_FILTER
    )
    if not has_passed_filter:
        return None

    layer: Layer = MATCH_FILTERS.LAYER_FILTER[layer]
    team1 = MATCH_FILTERS.TEAM_FILTER[team1]
    team1_faction: Faction = Faction(
        team1, MATCH_FILTERS.UNIT_TYPE_FILTER_BY_TEAM[team1][team1_subfaction]
    )
    team2 = MATCH_FILTERS.TEAM_FILTER[team2]
    team2_faction: Faction = Faction(
        team2, MATCH_FILTERS.UNIT_TYPE_FILTER_BY_TEAM[team2][team2_subfaction]
    )

    is_missing_data = None in (
        layer,
        team1_faction.unit_type,
        team2_faction.unit_type,
    )
    if is_missing_data:
        return None

    has_passed_filter = UnitType.AMPHIBIOUS_ASSAULT not in (
        team1_faction.unit_type,
        team2_faction.unit_type,
    )
    if not has_passed_filter:
        return None

    return (
        layer,
        team1_faction,
        team1_tickets,
        team2_faction,
        team2_tickets,
        age_in_days,
    )
# end _filter_match() definition


def _process_input_file(matches_file: Path) -> tuple[dict, dict]:
    matches_by_team_combo = {}
    matchups_by_team_combo = {}
    with open(matches_file, "r") as in_file:
        for line in in_file.readlines():
            loaded_data = json.loads(line)
            for match in loaded_data["data"]:
                results = _filter_match(match)
                if results is None:
                    continue

                (
                    layer,
                    team1_faction,
                    team1_tickets,
                    team2_faction,
                    team2_tickets,
                    age_in_days
                ) = results
                team_combo = TeamCombination(team1_faction.team, team2_faction.team)
                if team_combo not in matches_by_team_combo:
                    matches_by_team_combo[team_combo] = []

                matches = matches_by_team_combo[team_combo]
                match = Match(
                    layer,
                    team1_faction,
                    team2_faction,
                    team1_tickets - team2_tickets,
                    age_in_days
                )
                matches.append(match)

                if team_combo not in matchups_by_team_combo:
                    matchups_by_team_combo[team_combo] = {}

                matchups = matchups_by_team_combo[team_combo]
                matchup = Matchup(
                    layer,
                    team1_faction,
                    team2_faction
                )
                if matchup not in matchups:
                    matchups[matchup] = MatchupResult([0], [0])

                matchups[matchup].append(team1_tickets - team2_tickets, age_in_days)

    return matches_by_team_combo, matchups_by_team_combo
# end _process_input_file() definition


def _get_subunit_models(
        team1_faction: Faction,
        team2_faction: Faction,
        map_group: MapGroup
    ) -> Optional[Tuple[SubunitModel, SubunitModel]]:
    team1_subunit_model = None
    team2_subunit_model = None
    match map_group:
        case MapGroup.LARGE_WITH_HELI:
            team1_subunit_model=DEFAULT_SUBUNIT_MODEL_BY_FACTION.get(team1_faction)
            team2_subunit_model=DEFAULT_SUBUNIT_MODEL_BY_FACTION.get(team2_faction)
            if None in (team1_subunit_model, team2_subunit_model, ):
                logger.warning(
                    f"Could not retrieve default subunit models\n"
                    f"Map group: {map_group}, team1 faction: {team1_faction}, "
                    f"team2_faction: {team2_faction}"
                )
                return None
        case MapGroup.LARGE_NO_HELI:
            team1_subunit_model=DEFAULT_SUBUNIT_MODEL_BY_FACTION.get(team1_faction)
            team2_subunit_model=DEFAULT_SUBUNIT_MODEL_BY_FACTION.get(team2_faction)
            if None in (team1_subunit_model, team2_subunit_model, ):
                logger.warning(
                    f"Could not retrieve default subunit models\n"
                    f"Map group: {map_group}, team1 faction: {team1_faction}, "
                    f"team2_faction: {team2_faction}"
                )
                return None

            team1_subunit_model.remove_helis()
            team2_subunit_model.remove_helis()
        case MapGroup.MEDIUM:
            team1_subunit_model=MEDIUM_SUBUNIT_MODEL_BY_FACTION.get(team1_faction)
            team2_subunit_model=MEDIUM_SUBUNIT_MODEL_BY_FACTION.get(team2_faction)
            if None in (team1_subunit_model, team2_subunit_model, ):
                logger.warning(
                    f"Could not retrieve medium subunit models\n"
                    f"Map group: {map_group}, team1 faction: {team1_faction}, "
                    f"team2_faction: {team2_faction}"
                )
                return None
        case MapGroup.SMALL:
            team1_subunit_model=SMALL_SUBUNIT_MODEL_BY_FACTION.get(team1_faction)
            team2_subunit_model=SMALL_SUBUNIT_MODEL_BY_FACTION.get(team2_faction)
            if None in (team1_subunit_model, team2_subunit_model, ):
                logger.warning(
                    f"Could not retrieve small subunit models\n"
                    f"Map group: {map_group}, team1 faction: {team1_faction}, "
                    f"team2_faction: {team2_faction}"
                )
                return None
        case _:
            logger.warning(
                f"Encountered unknown map group: {map_group}"
            )
            return None
    # end match statement
    return team1_subunit_model, team2_subunit_model
# end _get_model_features() definition


def _get_weight_from_age_in_days(age_in_days: int) -> float:
    return max(0.1, age_in_days / MATCH_FILTERS.MAX_AGE_IN_DAYS ** 2)
# end _get_weight_from_age_in_days() definition


def get_model_data_from_match(
        match: Match,
        k_fold_idx: int
    ) -> Optional[ModelData]:
    team1_faction = match.team1_faction
    team2_faction = match.team2_faction
    map_group = MAP_GROUP_BY_LAYER[match.layer]
    subunit_models = _get_subunit_models(team1_faction, team2_faction, map_group)
    if subunit_models is None:
        logger.warning(f"Failed to retrieve subunit models for match: {match}")
        return None

    team1_subunit_model, team2_subunit_model = subunit_models
    return ModelData(
        layer=match.layer,
        team1_faction=match.team1_faction,
        team2_faction=match.team2_faction,
        team1_subunit_model=team1_subunit_model,
        team2_subunit_model=team2_subunit_model,
        tickets=match.tickets,
        weight=_get_weight_from_age_in_days(match.age_in_days),
        k_fold_index=k_fold_idx
    )
# end get_model_data() definition


def get_model_data_from_matchup(
        matchup: Matchup,
        matchup_result: MatchupResult,
        k_fold_idx: int
    ) -> Optional[ModelData]:
    team1_faction = matchup.team1_faction
    team2_faction = matchup.team2_faction
    map_group = MAP_GROUP_BY_LAYER[matchup.layer]
    subunit_models = _get_subunit_models(team1_faction, team2_faction, map_group)
    if subunit_models is None:
        logger.warning(f"Failed to retrieve subunit models for matchup: {matchup}")
        return None

    team1_subunit_model, team2_subunit_model = subunit_models
    tickets, weight = matchup_result.get_average_tickets_and_weight(
        _get_weight_from_age_in_days
    )
    return ModelData(
        layer=matchup.layer,
        team1_faction=matchup.team1_faction,
        team2_faction=matchup.team2_faction,
        team1_subunit_model=team1_subunit_model,
        team2_subunit_model=team2_subunit_model,
        tickets=tickets,
        weight=weight,
        k_fold_index=k_fold_idx
    )
# end get_model_data() definition


def _get_fold_idx_by_team_combo(any_by_team_combo: dict) -> dict:
    # Get our unique team vs team combinations and shuffle them
    all_team_combos = list(any_by_team_combo.keys())
    shuffle(all_team_combos)

    team_combos_sets = list(batched(
        all_team_combos,
        len(all_team_combos) // K_FOLD_SPLITS + 1
    ))

    # Assign fold indices to team combinations
    fold_idx_by_team_combo = {}
    for idx, team_combos in enumerate(team_combos_sets):
        for team_combo in team_combos:
            fold_idx_by_team_combo[team_combo] = idx
    return fold_idx_by_team_combo
# end _get_fold_idx_by_team_combo() definition


def _generate_balance_training_file(
        training_balance_file: Path,
        matchups_by_team_combo: dict
    ) -> None:
    if len(matchups_by_team_combo) == 0:
        logger.error("Cannot generate training file, zero matchups given")
        return

    with open(training_balance_file, "w") as file:
        file.write(f"{ModelData.column_repr()}\n")
        fold_idx_by_team_combo = _get_fold_idx_by_team_combo(matchups_by_team_combo)
        for team_combo in matchups_by_team_combo:
            matchups = matchups_by_team_combo[team_combo]
            k_fold_idx = fold_idx_by_team_combo[team_combo]
            for matchup, matchup_result in matchups.items():
                model_data = get_model_data_from_matchup(
                    matchup,
                    matchup_result,
                    k_fold_idx
                )
                if model_data is None:
                    logger.warning(
                        f"Model data not found for matchup: {matchup}\n"
                        "Skipping matchup"
                    )
                    continue

                file.write(f"{model_data}\n")
# end _generate_balance_training_file() definition


def _generate_volatility_training_file(
        training_volatility_file: Path,
        matches_by_team_combo: dict
    ) -> None:
    if len(matches_by_team_combo) == 0:
        logger.error("Cannot generate training file, zero matches given")
        return

    with open(training_volatility_file, "w") as file:
        file.write(f"{ModelData.column_repr()}\n")
        fold_idx_by_team_combo = _get_fold_idx_by_team_combo(matches_by_team_combo)
        for team_combo in matches_by_team_combo:
            matches = matches_by_team_combo[team_combo]
            k_fold_idx = fold_idx_by_team_combo[team_combo]
            for match in matches:
                model_data = get_model_data_from_match(match, k_fold_idx)
                if model_data is None:
                    logger.warning(
                        f"Model data not found for match: {match}\n"
                        "Skipping match"
                    )
                    continue

                file.write(f"{model_data}\n")
# end _generate_balance_training_file() definition


@logger.catch
def process_matches(
        matches_file: str,
        training_balance_file: str,
        training_volatility_file: str
    ) -> None:
    ABORT_MESSAGE = "Aborting process_matches()"
    matches_file = Path(matches_file)
    training_balance_file = Path(training_balance_file)
    training_volatility_file = Path(training_volatility_file)

    if training_balance_file.is_file():
        logger.info(
            f"Training file: {training_balance_file} already exists\n"
            f"{ABORT_MESSAGE}"
        )
        return

    if training_volatility_file.is_file():
        logger.info(
            f"Training file: {training_volatility_file} already exists\n"
            f"{ABORT_MESSAGE}"
        )
        return

    if not matches_file.is_file():
        logger.error(
            f"Matches_file: {matches_file} doesn't exist\n"
            f"{ABORT_MESSAGE}"
        )
        return

    matches_by_team_combo, matchups_by_team_combo = _process_input_file(matches_file)
    _generate_balance_training_file(
        training_balance_file,
        matchups_by_team_combo
    )
    _generate_volatility_training_file(
        training_volatility_file,
        matches_by_team_combo
    )
    return
# end process_matches() definition
