# Module to download MySquadStats matches to a .json file

# Python imports
import concurrent.futures
import requests
import json
from typing import Optional
from pathlib import Path
import time

# 3rd party imports
from loguru import logger


def _get_page(url: str) -> Optional[dict]:
    request_headers = {}
    request_headers["referer"] = "https://mysquadstats.com/"
    logger.debug(f"Fetching URL: {url}")
    result = requests.get(url, headers=request_headers)

    if result.status_code != 200:
        logger.warning(
            "Result's status_code is not 200(OK)\n"
            f"Ignoring and dumping result: {result}"
        )
        return None

    result_data = None
    try:
        result_data = result.json()
    except Exception:
        logger.warning(
            "Failed to get result's JSON data\n"
            f"Ignoring and dumping result: {result}"
        )
        return None

    if result_data["status"] != "Success":
        logger.warning(
            "Result_data[\"status\"] is not \"Success\"\n"
            f"Ignoring result, dumping result_data: {result_data}"
        )
        return None

    return result_data
# end _get_page() definition


def _get_url_for_page(page: int, page_size) -> str:
    return f"https://api.mysquadstats.com/matchData?draw=2&page={page}&pageSize={page_size}&sortColumn=startTime&sortDirection=DESC&search="


def _get_results(get_urls: list[str]) -> list:
    MAX_WORKERS = 4
    results = []
    def inner_add_result_if_exists(result) -> None:
        if result is None:
            return
        results.append(result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        data_by_futures = {executor.submit(_get_page, url) for url in get_urls}
        data_by_retry_futures = {}
        for future in concurrent.futures.as_completed(data_by_futures):
            if future.exception():
                time.sleep(60)
                last_data = data_by_futures[future]
                new_future = executor.submit(_get_page, last_data)
                data_by_retry_futures[new_future] = last_data
            else:
                inner_add_result_if_exists(future.result())

        # Retry one more time if needed
        retries_left = len(data_by_retry_futures)
        if retries_left > 0:
            logger.info(
                f"\nRetries left: {retries_left}"
            )
            for future in concurrent.futures.as_completed(data_by_retry_futures):
                if future.exception():
                    data = data_by_retry_futures[future]
                    logger.warning(f"Failure on retry: {data}, not trying again")
                else:
                    inner_add_result_if_exists(future.result())

    return results
# end _get_results() definition


def download_matches(matches_file: str, pages: int, page_size: int) -> None:
    ABORT_MESSAGE = "Aborting download_matches()"
    matches_file = Path(matches_file)
    if matches_file.is_file():
        logger.info(
            f"Matches file: {matches_file} already exists\n"
            f"{ABORT_MESSAGE}"
        )
        return

    # Populate URLs
    get_urls = []
    for page in range(1, pages + 1, 1):
        get_urls.append(_get_url_for_page(page, page_size))

    # Get pages from URLs
    results = _get_results(get_urls)

    # Check if results have enough data
    if len(results) < 2:
        logger.error(
            f"Found less than 2 results after running jobs\n"
            f"Dumping results: {results}\n"
            f"{ABORT_MESSAGE}"
        )
        return

    # Write each result on a new line in form of a JSON dump
    with open(matches_file, "w") as file:
        for result in results[:-1]:
            file.write(json.dumps(result))
            file.write("\n")

        result = results[-1]
        file.write(json.dumps(result))
# end download_matches() definition
