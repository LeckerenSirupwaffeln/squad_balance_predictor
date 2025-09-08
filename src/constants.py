# Relative imports
from .custom_types import (
    GameMap,
    GameMode,
    MapVersion,
    MapGroup,
    Layer,
    Team,
    UnitType,
    Alliance,
)

MATCHES_FILE = "./data/mysquadstats_matches.json"
TRAINING_BALANCE_FILE = "./data/training_balance.csv"
TRAINING_VOLATILITY_FILE = "./data/training_volatility.csv"
RESULTS_FOLDER = "./data/results"
MODEL_BALANCE_FILE = "./data/model_balance.booster"
MODEL_VOLATILITY_FILE = "./data/model_volatility.booster"
K_FOLD_SPLITS = 10


class MATCH_FILTERS:
    DLC_FILTER = ["Game", "SanxianIslands", "BlackCoast", "Harju"]
    MOD_FILTER = ["Vanilla"]
    GAMEMODE_FILTER = ["AAS", "RAAS"]

    MAX_DURATION_IN_SECONDS = 7200  # 7200 seconds = 120 minutes
    MIN_DURATION_IN_SECONDS = 1200  # 1200 seconds = 20 minutes
    MIN_KILLS_BOTH_SIDES = 100
    MAX_AGE_IN_DAYS = 270
    MIN_AGE_IN_DAYS = 0

    LAYER_FILTER = {
        "AlBasrah AAS v1": Layer(GameMap.AL_BASRAH, GameMode.AAS, MapVersion.V1),
        "AlBasrah Insurgency v1": None,
        "AlBasrah Invasion v1": Layer(
            GameMap.AL_BASRAH, GameMode.INVASION, MapVersion.V1
        ),
        "AlBasrah Invasion v2": Layer(
            GameMap.AL_BASRAH, GameMode.INVASION, MapVersion.V2
        ),
        "AlBasrah RAAS v1": Layer(GameMap.AL_BASRAH, GameMode.RAAS, MapVersion.V1),
        "Anvil AAS v1": Layer(GameMap.ANVIL, GameMode.AAS, MapVersion.V1),
        "Anvil Invasion v1": Layer(GameMap.ANVIL, GameMode.INVASION, MapVersion.V1),
        "Anvil RAAS v1": Layer(GameMap.ANVIL, GameMode.RAAS, MapVersion.V1),
        "Anvil RAAS v2": Layer(GameMap.ANVIL, GameMode.RAAS, MapVersion.V2),
        "Belaya AAS v1": Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V1),
        "Belaya AAS v2": Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V2),
        "Belaya AAS v3": Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V3),
        "Belaya Invasion v1": Layer(GameMap.BELAYA, GameMode.INVASION, MapVersion.V1),
        "Belaya Invasion v2": Layer(GameMap.BELAYA, GameMode.INVASION, MapVersion.V2),
        "Belaya RAAS v1": Layer(GameMap.BELAYA, GameMode.RAAS, MapVersion.V1),
        "BlackCoast AAS v1": Layer(GameMap.BLACK_COAST, GameMode.AAS, MapVersion.V1),
        "BlackCoast AAS v2": Layer(GameMap.BLACK_COAST, GameMode.AAS, MapVersion.V2),
        "BlackCoast RAAS v1": Layer(GameMap.BLACK_COAST, GameMode.RAAS, MapVersion.V1),
        "BlackCoast RAAS v2": Layer(GameMap.BLACK_COAST, GameMode.RAAS, MapVersion.V2),
        "BlackCoast Invasion v1": Layer(
            GameMap.BLACK_COAST, GameMode.INVASION, MapVersion.V1
        ),
        "BlackCoast Invasion v2": Layer(
            GameMap.BLACK_COAST, GameMode.INVASION, MapVersion.V2
        ),
        "Chora AAS v1": Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V1),
        "Chora AAS v2": Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V2),
        "Chora AAS v3": Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V3),
        "Chora Invasion v1": Layer(GameMap.CHORA, GameMode.INVASION, MapVersion.V1),
        "Chora Invasion v2": Layer(GameMap.CHORA, GameMode.INVASION, MapVersion.V2),
        "Chora RAAS v1": Layer(GameMap.CHORA, GameMode.RAAS, MapVersion.V1),
        "Fallujah AAS v1": Layer(GameMap.FALLUJAH, GameMode.AAS, MapVersion.V1),
        "Fallujah Invasion v1": Layer(
            GameMap.FALLUJAH, GameMode.INVASION, MapVersion.V1
        ),
        "Fallujah Invasion v2": Layer(
            GameMap.FALLUJAH, GameMode.INVASION, MapVersion.V2
        ),
        "Fallujah RAAS v1": Layer(GameMap.FALLUJAH, GameMode.RAAS, MapVersion.V1),
        "Fallujah RAAS v2": Layer(GameMap.FALLUJAH, GameMode.RAAS, MapVersion.V2),
        "FoolsRoad AAS v1": Layer(GameMap.FOOLS_ROAD, GameMode.AAS, MapVersion.V1),
        "FoolsRoad AAS v2": Layer(GameMap.FOOLS_ROAD, GameMode.AAS, MapVersion.V2),
        "FoolsRoad Invasion v1": Layer(
            GameMap.FOOLS_ROAD, GameMode.INVASION, MapVersion.V1
        ),
        "FoolsRoad RAAS v1": Layer(GameMap.FOOLS_ROAD, GameMode.RAAS, MapVersion.V1),
        "FoolsRoad RAAS v2": None,
        "FoolsRoad RAAS v3": None,
        "GooseBay AAS v1": Layer(GameMap.GOOSE_BAY, GameMode.AAS, MapVersion.V1),
        "GooseBay Invasion v1": Layer(
            GameMap.GOOSE_BAY, GameMode.INVASION, MapVersion.V1
        ),
        "GooseBay RAAS v1": Layer(GameMap.GOOSE_BAY, GameMode.RAAS, MapVersion.V1),
        "GooseBay RAAS v2": Layer(GameMap.GOOSE_BAY, GameMode.RAAS, MapVersion.V2),
        "Gorodok AAS v1": Layer(GameMap.GORODOK, GameMode.AAS, MapVersion.V1),
        "Gorodok Invasion v1": Layer(GameMap.GORODOK, GameMode.INVASION, MapVersion.V1),
        "Gorodok Invasion v2": Layer(GameMap.GORODOK, GameMode.INVASION, MapVersion.V2),
        "Gorodok RAAS v1": Layer(GameMap.GORODOK, GameMode.RAAS, MapVersion.V1),
        "Gorodok RAAS v2": Layer(GameMap.GORODOK, GameMode.RAAS, MapVersion.V2),
        "Harju AAS v1": Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V1),
        "Harju AAS v2": Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V2),
        "Harju AAS v3": Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V3),
        "Harju Invasion v1": Layer(GameMap.HARJU, GameMode.INVASION, MapVersion.V1),
        "Harju Invasion v2": Layer(GameMap.HARJU, GameMode.INVASION, MapVersion.V2),
        "Harju Invasion v3": Layer(GameMap.HARJU, GameMode.INVASION, MapVersion.V3),
        "Harju RAAS v1": Layer(GameMap.HARJU, GameMode.RAAS, MapVersion.V1),
        "Harju RAAS v2": Layer(GameMap.HARJU, GameMode.RAAS, MapVersion.V2),
        "Harju RAAS v3": None,
        "Kamdesh AAS v1": Layer(GameMap.KAMDESH, GameMode.AAS, MapVersion.V1),
        "Kamdesh Invasion v1": Layer(GameMap.KAMDESH, GameMode.INVASION, MapVersion.V1),
        "Kamdesh RAAS v1": Layer(GameMap.KAMDESH, GameMode.RAAS, MapVersion.V1),
        "Kohat AAS v1": Layer(GameMap.KOHAT, GameMode.AAS, MapVersion.V1),
        "Kohat Invasion v1": Layer(GameMap.KOHAT, GameMode.INVASION, MapVersion.V1),
        "Kohat Invasion v2": Layer(GameMap.KOHAT, GameMode.INVASION, MapVersion.V2),
        "Kohat RAAS v1": Layer(GameMap.KOHAT, GameMode.RAAS, MapVersion.V1),
        "Kohat RAAS v2": Layer(GameMap.KOHAT, GameMode.RAAS, MapVersion.V2),
        "Kokan AAS v1": Layer(GameMap.KOKAN, GameMode.AAS, MapVersion.V1),
        "Kokan AAS v2": Layer(GameMap.KOKAN, GameMode.AAS, MapVersion.V2),
        "Kokan Invasion v1": Layer(GameMap.KOKAN, GameMode.INVASION, MapVersion.V1),
        "Kokan RAAS v1": Layer(GameMap.KOKAN, GameMode.RAAS, MapVersion.V1),
        "Kokan RAAS v2": None,
        "Lashkar AAS v1": Layer(GameMap.LASHKAR, GameMode.AAS, MapVersion.V1),
        "Lashkar AAS v2": Layer(GameMap.LASHKAR, GameMode.AAS, MapVersion.V2),
        "Lashkar Invasion v1": Layer(GameMap.LASHKAR, GameMode.INVASION, MapVersion.V1),
        "Lashkar RAAS v1": Layer(GameMap.LASHKAR, GameMode.RAAS, MapVersion.V1),
        "Logar AAS v1": Layer(GameMap.LOGAR, GameMode.AAS, MapVersion.V1),
        "Logar RAAS v1": Layer(GameMap.LOGAR, GameMode.RAAS, MapVersion.V1),
        "Manicouagan AAS v1": Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V1),
        "Manicouagan AAS v2": Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V2),
        "Manicouagan AAS v3": Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V3),
        "Manicouagan Invasion v1": Layer(
            GameMap.MANICOUAGAN, GameMode.INVASION, MapVersion.V1
        ),
        "Manicouagan Invasion v2": Layer(
            GameMap.MANICOUAGAN, GameMode.INVASION, MapVersion.V2
        ),
        "Manicouagan RAAS v1": Layer(GameMap.MANICOUAGAN, GameMode.RAAS, MapVersion.V1),
        "Manicouagan RAAS v2": Layer(GameMap.MANICOUAGAN, GameMode.RAAS, MapVersion.V2),
        "Mestia AAS v1": Layer(GameMap.MESTIA, GameMode.AAS, MapVersion.V1),
        "Mestia AAS v2": Layer(GameMap.MESTIA, GameMode.AAS, MapVersion.V2),
        "Mestia Invasion v1": Layer(GameMap.MESTIA, GameMode.INVASION, MapVersion.V1),
        "Mestia RAAS v1": Layer(GameMap.MESTIA, GameMode.RAAS, MapVersion.V1),
        "Mutaha AAS v1": Layer(GameMap.MUTAHA, GameMode.AAS, MapVersion.V1),
        "Mutaha AAS v2": Layer(GameMap.MUTAHA, GameMode.AAS, MapVersion.V2),
        "Mutaha Invasion v1": Layer(GameMap.MUTAHA, GameMode.INVASION, MapVersion.V1),
        "Mutaha RAAS v1": Layer(GameMap.MUTAHA, GameMode.RAAS, MapVersion.V1),
        "Mutaha RAAS v2": Layer(GameMap.MUTAHA, GameMode.RAAS, MapVersion.V2),
        "Narva AAS v1": Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V1),
        "Narva AAS v2": Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V2),
        "Narva AAS v3": Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V3),
        "Narva Invasion v1": Layer(GameMap.NARVA, GameMode.INVASION, MapVersion.V1),
        "Narva Invasion v2": Layer(GameMap.NARVA, GameMode.INVASION, MapVersion.V2),
        "Narva RAAS v1": Layer(GameMap.NARVA, GameMode.RAAS, MapVersion.V1),
        "PacificProvingGrounds AAS v1": Layer(
            GameMap.PACIFIC_PROVING_GROUNDS, GameMode.AAS, MapVersion.V1
        ),
        "Sanxian AAS v1": Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V1),
        "Sanxian AAS v2": Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V2),
        "Sanxian AAS v3": Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V3),
        "Sanxian Invasion v1": Layer(GameMap.SANXIAN, GameMode.INVASION, MapVersion.V1),
        "Sanxian Invasion v2": Layer(GameMap.SANXIAN, GameMode.INVASION, MapVersion.V2),
        "Sanxian RAAS v1": Layer(GameMap.SANXIAN, GameMode.RAAS, MapVersion.V1),
        "Sanxian RAAS v2": Layer(GameMap.SANXIAN, GameMode.RAAS, MapVersion.V2),
        "Skorpo Invasion v1": Layer(GameMap.SKORPO, GameMode.INVASION, MapVersion.V1),
        "Skorpo Invasion v2": Layer(GameMap.SKORPO, GameMode.INVASION, MapVersion.V2),
        "Skorpo RAAS v1": Layer(GameMap.SKORPO, GameMode.RAAS, MapVersion.V1),
        "Sumari AAS v1": Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V1),
        "Sumari AAS v2": Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V2),
        "Sumari AAS v3": Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V3),
        "Sumari Invasion v1": Layer(GameMap.SUMARI, GameMode.INVASION, MapVersion.V1),
        "Sumari RAAS v1": Layer(GameMap.SUMARI, GameMode.RAAS, MapVersion.V1),
        "Tallil AAS v1": Layer(GameMap.TALLIL, GameMode.AAS, MapVersion.V1),
        "Tallil Invasion v1": Layer(GameMap.TALLIL, GameMode.INVASION, MapVersion.V1),
        "Tallil RAAS v1": Layer(GameMap.TALLIL, GameMode.RAAS, MapVersion.V1),
        "Tallil RAAS v2": Layer(GameMap.TALLIL, GameMode.RAAS, MapVersion.V2),
        "Yehorivka AAS v1": Layer(GameMap.YEHORIVKA, GameMode.AAS, MapVersion.V1),
        "Yehorivka AAS v2": Layer(GameMap.YEHORIVKA, GameMode.AAS, MapVersion.V2),
        "Yehorivka Invasion v1": Layer(
            GameMap.YEHORIVKA, GameMode.INVASION, MapVersion.V1
        ),
        "Yehorivka Invasion v2": Layer(
            GameMap.YEHORIVKA, GameMode.INVASION, MapVersion.V2
        ),
        "Yehorivka RAAS v1": Layer(GameMap.YEHORIVKA, GameMode.RAAS, MapVersion.V1),
        "Yehorivka RAAS v2": Layer(GameMap.YEHORIVKA, GameMode.RAAS, MapVersion.V2),
    }

    TEAM_FILTER = {
        "Australian Defence Force": Team.ADF,
        "British Armed Forces": Team.BAF,
        "Canadian Armed Forces": Team.CAF,
        "United States Army": Team.USA,
        "United States Marine Corps": Team.USMC,
        "People's Liberation Army": Team.PLA,
        "PLA Amphibious Ground Forces": Team.PLAAGF,
        "PLA Navy Marine Corps": Team.PLANMC,
        "Russian Airborne Forces": Team.VDV,
        "Russian Ground Forces": Team.RGF,
        "Middle Eastern Alliance": Team.MEA,
        "Turkish Land Forces": Team.TLF,
        "Middle Eastern Insurgents": Team.INS,
        "Irregular Militia Forces": Team.IMF,
        "Western Private Military Contractors": Team.WPMC,
    }

    UNIT_TYPE_FILTER_BY_TEAM = {
        Team.ADF: {
            "1st Battalion, Royal Australian Regiment": UnitType.MECHANIZED,
            "3rd Brigade Battle Group": UnitType.COMBINED_ARMS,
            "3rd Battalion, Royal Australian Regiment": UnitType.AIR_ASSAULT,
        },
        Team.BAF: {
            "3rd Division Battle Group": UnitType.COMBINED_ARMS,
            "Queen's Royal Hussars Battle Group": UnitType.ARMORED,
            "1 Yorks Battle Group": UnitType.MECHANIZED,
            "Royal Logistics Corps Battle Group": UnitType.SUPPORT,
            "2nd Battalion, Parachute Regiment": UnitType.AIR_ASSAULT,
            "British Armed Forces": None,
        },
        Team.CAF: {
            "1 Canadian Mechanized Brigade Group": UnitType.COMBINED_ARMS,
            "12e Régiment Blindé du Canada": UnitType.MOTORIZED,
            "Lord Strathcona's Horse Regiment": UnitType.ARMORED,
            "1st Battalion, Royal 22e Régiment": UnitType.MECHANIZED,
            "6 Canadian Combat Support Brigade": UnitType.SUPPORT,
            "3rd Battalion, Royal Canadian Regiment": UnitType.AIR_ASSAULT,
            "Canadian Armed Forces": None,
        },
        Team.USA: {
            "2nd Cavalry Stryker Brigade Combat Team": UnitType.MOTORIZED,
            "37th Armored Regiment, 1st Armored Division": UnitType.ARMORED,
            "1st Brigade Combat Team, 82nd Airborne Division": UnitType.AIR_ASSAULT,
            "1st Cavalry Regiment": UnitType.MECHANIZED,
            "1st Brigade Combat Team, 10th Mountain Division": UnitType.LIGHT_INFANTRY,
            "3rd Brigade Combat Team, 1st Infantry Division": UnitType.COMBINED_ARMS,
            "497th Combat Sustainment Support Battalion": UnitType.SUPPORT,
            "United States Army": None,
            "1st Infantry Division": None,
        },
        Team.USMC: {
            "31st Marine Expeditionary Unit": UnitType.COMBINED_ARMS,
            "3rd Light Armored Recon Battalion": UnitType.MOTORIZED,
            "4th Marines Amphibious Ready Group": UnitType.AMPHIBIOUS_ASSAULT,
            "1st Marines Regimental Combat Team": UnitType.LIGHT_INFANTRY,
            "2nd Marine Logistics Group": UnitType.SUPPORT,
            "1st Tank Battalion, 1st Marines": UnitType.ARMORED,
            "United States Marine Corps": None,
        },
        Team.PLA: {
            "195th Heavy Combined Arms Brigade": UnitType.ARMORED,
            "112th Medium Combined Arms Brigade": UnitType.MOTORIZED,
            "118th Combined Arms Brigade": UnitType.COMBINED_ARMS,
            "80th Support Brigade": UnitType.SUPPORT,
            "149th Mountain Infantry Brigade": UnitType.LIGHT_INFANTRY,
            "161st Air Assault Brigade": UnitType.AIR_ASSAULT,
            "80th Combined Arms Brigade": UnitType.SUPPORT,
        },
        Team.PLAAGF: {
            "14th Amphibious Combined Arms Brigade": UnitType.COMBINED_ARMS,
            "9th Heavy Combined Arms Battalion": UnitType.ARMORED,
            "4th Medium Combined Arms Battalion": UnitType.MECHANIZED,
        },
        Team.PLANMC: {
            "7th Marine Medium Battalion": UnitType.MOTORIZED,
            "17th Marine Support Battalion": UnitType.SUPPORT,
            "3rd Marine Heavy Battalion": UnitType.ARMORED,
            "4th Marine Special Combat Battalion": UnitType.AIR_ASSAULT,
            "5th Marine Brigade": UnitType.COMBINED_ARMS,
        },
        Team.VDV: {
            "7th Guards Mountain Air Assault Division": UnitType.COMBINED_ARMS,
            "104th Tank Battalion": UnitType.ARMORED,
            "108th Guards Air Assault Regiment": UnitType.MECHANIZED,
            "150th Support Battalion": UnitType.SUPPORT,
            "217th Guards Airborne Regiment": UnitType.AIR_ASSAULT,
        },
        Team.RGF: {
            "49th Combined Arms Army": UnitType.COMBINED_ARMS,
            "205th Separate Motor Rifle Brigade": UnitType.MECHANIZED,
            "3rd Motor Rifle Brigade": UnitType.MOTORIZED,
            "1398th Separate Reconnaissance Battalion": UnitType.LIGHT_INFANTRY,
            "78th Detached Logistics Brigade": UnitType.SUPPORT,
            "6th Separate Tank Brigade": UnitType.ARMORED,
            "336th Guards Naval Infantry Brigade": UnitType.AMPHIBIOUS_ASSAULT,
            "Russian Ground Forces": None,
        },
        Team.MEA: {
            "1st Battalion, Legion of Babylon": UnitType.COMBINED_ARMS,
            "60th Prince Assur Armored Brigade": UnitType.ARMORED,
            "Vizir Hussein 2nd Support Battalion": UnitType.SUPPORT,
            "3rd King Qadesh Mechanized Infantry Brigade": UnitType.MECHANIZED,
            "4th Border Guards Group": UnitType.LIGHT_INFANTRY,
            "91st Air Assault Battalion": UnitType.AIR_ASSAULT,
        },
        Team.TLF: {
            "66th Mechanized Infantry Brigade Battle Group": UnitType.MECHANIZED,
            "51st Motorized Infantry Brigade Battle Group": UnitType.MOTORIZED,
            "1st Army Battle Group": UnitType.COMBINED_ARMS,
            "4th Armored Brigade Battle Group": UnitType.ARMORED,
            "1st Commando Brigade Battle Group": UnitType.AIR_ASSAULT,
            "Land Forces Logistics Command Battle Group": UnitType.SUPPORT,
            "Turkish Land Forces": None,
        },
        Team.INS: {
            "Irregular Light Infantry": UnitType.LIGHT_INFANTRY,
            "Irregular Armored Squadron": UnitType.ARMORED,
            "Irregular Mechanized Platoon": UnitType.MECHANIZED,
            "Irregular Battle Group": UnitType.COMBINED_ARMS,
            "Irregular Fire Support Group": UnitType.SUPPORT,
            "Irregular Motorized Platoon": UnitType.MOTORIZED,
            "Insurgents": None,
        },
        Team.IMF: {
            "Irregular Light Infantry": UnitType.LIGHT_INFANTRY,
            "Irregular Mechanized Platoon": UnitType.MECHANIZED,
            "Irregular Battle Group": UnitType.COMBINED_ARMS,
            "Irregular Fire Support Group": UnitType.SUPPORT,
            "Irregular Armored Squadron": UnitType.ARMORED,
            "Irregular Motorized Platoon": UnitType.MOTORIZED,
            "Irregular Militia Forces": None,
        },
        Team.WPMC: {
            "Overwatch 6 Patrol Group": UnitType.LIGHT_INFANTRY,
            "Manticore Security Task Force": UnitType.COMBINED_ARMS,
            "Murk Water Air Wing": UnitType.AIR_ASSAULT,
        },
    }
# end class MATCH_FILTERS definition

MAP_GROUP_BY_LAYER = {
    Layer(GameMap.AL_BASRAH, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.AL_BASRAH, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.ANVIL, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.ANVIL, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.ANVIL, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.BELAYA, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BELAYA, GameMode.AAS, MapVersion.V3): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.BLACK_COAST, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BLACK_COAST, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BLACK_COAST, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.BLACK_COAST, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.CHORA, GameMode.RAAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V2): MapGroup.MEDIUM,
    Layer(GameMap.CHORA, GameMode.AAS, MapVersion.V3): MapGroup.MEDIUM,

    Layer(GameMap.FALLUJAH, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_NO_HELI,
    Layer(GameMap.FALLUJAH, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_NO_HELI,
    Layer(GameMap.FALLUJAH, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_NO_HELI,

    Layer(GameMap.FOOLS_ROAD, GameMode.RAAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.FOOLS_ROAD, GameMode.AAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.FOOLS_ROAD, GameMode.AAS, MapVersion.V2): MapGroup.MEDIUM,

    Layer(GameMap.GOOSE_BAY, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.GOOSE_BAY, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.GOOSE_BAY, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.GORODOK, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.GORODOK, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.GORODOK, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.HARJU, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.HARJU, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V2): MapGroup.MEDIUM,
    Layer(GameMap.HARJU, GameMode.AAS, MapVersion.V3): MapGroup.MEDIUM,

    Layer(GameMap.KAMDESH, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.KAMDESH, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.KOHAT, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.KOHAT, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.KOHAT, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.KOKAN, GameMode.RAAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.KOKAN, GameMode.AAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.KOKAN, GameMode.AAS, MapVersion.V2): MapGroup.MEDIUM,

    Layer(GameMap.LASHKAR, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.LASHKAR, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.LASHKAR, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.LOGAR, GameMode.RAAS, MapVersion.V1): MapGroup.SMALL,
    Layer(GameMap.LOGAR, GameMode.AAS, MapVersion.V1): MapGroup.SMALL,

    Layer(GameMap.MANICOUAGAN, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MANICOUAGAN, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MANICOUAGAN, GameMode.AAS, MapVersion.V3): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.MESTIA, GameMode.RAAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.MESTIA, GameMode.AAS, MapVersion.V1): MapGroup.MEDIUM,
    Layer(GameMap.MESTIA, GameMode.AAS, MapVersion.V2): MapGroup.MEDIUM,

    Layer(GameMap.MUTAHA, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MUTAHA, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MUTAHA, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.MUTAHA, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.NARVA, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.NARVA, GameMode.AAS, MapVersion.V3): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.PACIFIC_PROVING_GROUNDS, GameMode.AAS, MapVersion.V1): MapGroup.SMALL,

    Layer(GameMap.SANXIAN, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.SANXIAN, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.SANXIAN, GameMode.AAS, MapVersion.V3): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.SKORPO, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.SUMARI, GameMode.RAAS, MapVersion.V1): MapGroup.SMALL,
    Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V1): MapGroup.SMALL,
    Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V2): MapGroup.SMALL,
    Layer(GameMap.SUMARI, GameMode.AAS, MapVersion.V3): MapGroup.SMALL,

    Layer(GameMap.TALLIL, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.TALLIL, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.TALLIL, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,

    Layer(GameMap.YEHORIVKA, GameMode.RAAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.YEHORIVKA, GameMode.RAAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.YEHORIVKA, GameMode.AAS, MapVersion.V1): MapGroup.LARGE_WITH_HELI,
    Layer(GameMap.YEHORIVKA, GameMode.AAS, MapVersion.V2): MapGroup.LARGE_WITH_HELI,
}
# end MAP_GROUP_BY_LAYER definition

TEAMS_BY_GAME_MAP = {
    GameMap.AL_BASRAH: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.PLANMC,
        Team.PLAAGF,
        Team.TLF,
        Team.WPMC
    },
    GameMap.ANVIL: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.BELAYA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.TLF,
        Team.WPMC
    },
    GameMap.BLACK_COAST: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.TLF,
        Team.PLANMC,
        Team.PLAAGF,
        Team.WPMC
    },
    GameMap.CHORA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.FALLUJAH: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.FOOLS_ROAD: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.IMF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.GOOSE_BAY: {
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.PLANMC,
        Team.PLAAGF,
        Team.WPMC
    },
    GameMap.GORODOK: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.TLF,
        Team.WPMC
    },
    GameMap.HARJU: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.WPMC
    },
    GameMap.KAMDESH: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.KOHAT: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.KOKAN: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.LASHKAR: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.LOGAR: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.MANICOUAGAN: {
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.PLANMC,
        Team.PLAAGF,
        Team.WPMC
    },
    GameMap.MESTIA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.IMF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.MUTAHA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.NARVA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.WPMC
    },
    GameMap.PACIFIC_PROVING_GROUNDS: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.PLANMC,
        Team.PLAAGF,
        Team.USA,
        Team.USMC,
        Team.WPMC
    },
    GameMap.SANXIAN: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.PLANMC,
        Team.PLAAGF,
        Team.USA,
        Team.USMC,
        Team.WPMC
    },
    GameMap.SKORPO: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.WPMC
    },
    GameMap.SUMARI: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.TALLIL: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.INS,
        Team.MEA,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.TLF,
        Team.WPMC
    },
    GameMap.YEHORIVKA: {
        Team.ADF,
        Team.BAF,
        Team.CAF,
        Team.PLA,
        Team.RGF,
        Team.USA,
        Team.USMC,
        Team.VDV,
        Team.IMF,
        Team.TLF,
        Team.WPMC
    },
}
# end TEAMS_BY_GAME_MAP definition

ALLOWED_UNIT_TYPES_BY_MAP_GROUP = {
    MapGroup.LARGE_WITH_HELI: {
        UnitType.AIR_ASSAULT,
        UnitType.ARMORED,
        UnitType.COMBINED_ARMS,
        UnitType.LIGHT_INFANTRY,
        UnitType.MECHANIZED,
        UnitType.MOTORIZED,
        UnitType.SUPPORT
    },
    MapGroup.LARGE_NO_HELI: {
        UnitType.ARMORED,
        UnitType.COMBINED_ARMS,
        UnitType.LIGHT_INFANTRY,
        UnitType.MECHANIZED,
        UnitType.MOTORIZED,
        UnitType.SUPPORT
    },
    MapGroup.MEDIUM: {
        UnitType.COMBINED_ARMS
    },
    MapGroup.SMALL: {
        UnitType.COMBINED_ARMS
    }
}
# end ALLOWED_UNIT_TYPES_BY_MAP_GROUP

ALLIANCES_BY_TEAM = {
    Team.ADF: Alliance.BLUEFOR,
    Team.BAF: Alliance.BLUEFOR,
    Team.CAF: Alliance.BLUEFOR,
    Team.USA: Alliance.BLUEFOR,
    Team.USMC: Alliance.BLUEFOR,

    Team.PLA: Alliance.PAC,
    Team.PLAAGF: Alliance.PAC,
    Team.PLANMC: Alliance.PAC,

    Team.VDV: Alliance.REDFOR,
    Team.RGF: Alliance.REDFOR,

    Team.MEA: Alliance.MEA,
    Team.TLF: Alliance.TLF,
    Team.INS: Alliance.INS,
    Team.IMF: Alliance.IMF,
    Team.WPMC: Alliance.WPMC
}
# end ALLIANCES_BY_TEAM
