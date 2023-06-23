"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""
import asyncio
import pytest
import logging
from typing import Dict

import websockets
import responses
from pandas import DataFrame

from iktos.spaya import (
    BearerToken,
    RetrosynthesisResult,
    SpayaClientAsync,
    StatusCode,
    SettingsAsync,
)


LOGGER = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_nominal(websocket_nominal, url_websocket, token):
    async for smiles_to_send, check in websocket_nominal:
        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            async for smiles, score in client.consume():
                LOGGER.info(f"{smiles}: {score}")
                result[smiles] = score

        check(result)


@pytest.mark.asyncio
async def test_too_many_smiles(websocket_nominal, url_websocket, token):
    async for smiles_to_send, check in websocket_nominal:
        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket,
            authorization=BearerToken(token=token),
            settings=SettingsAsync(max_smiles_per_request=2),
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            async for smiles, score in client.consume():
                LOGGER.info(f"{smiles}: {score}")
                result[smiles] = score

        check(result)


@pytest.mark.asyncio
async def test_wait_result(websocket_nominal, url_websocket, token):
    async for smiles_to_send, check in websocket_nominal:

        async def progression(progress: float):
            LOGGER.info(f"progression:{progress}")

        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            await client.wait_result(callback_progression=progression)
            for s in smiles_to_send:
                LOGGER.info(f"s :{s}")
                if isinstance(s, list):
                    for s2 in s:
                        result[s2] = client[s2]
                else:
                    result[s] = client[s]
        check(result)


@pytest.mark.asyncio
async def test_nominal_already_done(
    websocket_nominal_already_done, url_websocket, token
):
    async for smiles_to_send, check in websocket_nominal_already_done:
        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            await client.wait_result()
            async for smiles, score in client.consume():
                LOGGER.info(f"{smiles}: {score}")
                result[smiles] = score

        check(result)


@pytest.mark.asyncio
async def test_queue_full(websocket_nominal_queue_full, url_websocket, token):
    async for smiles_to_send, check in websocket_nominal_queue_full:
        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            await client.wait_result()
            async for smiles, score in client.consume():
                LOGGER.info(f"{smiles}: {score}")
                result[smiles] = score

        check(result)


@pytest.mark.asyncio
async def test_reconnect(websocket_stop_and_continue, url_websocket, token):
    async for smiles_to_send, check in websocket_stop_and_continue:
        result = dict()  # type: Dict[str, RetrosynthesisResult]
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            async for smiles, score in client.consume():
                LOGGER.info(f"{smiles}: {score}")
                result[smiles] = score

        check(result)


def test_uri(url, url_websocket, url_websocket_path):
    test_expected = [
        (url, f"ws://localhost:8000{url_websocket_path}"),
        ("http://localhost", f"ws://localhost{url_websocket_path}"),
        ("https://localhost/", f"wss://localhost{url_websocket_path}"),
        ("https://localhost:8000", f"wss://localhost:8000{url_websocket_path}"),
        ("https://localhost:8000", f"wss://localhost:8000{url_websocket_path}"),
        (
            "http://localhost:8000/extra",
            f"ws://localhost:8000/extra{url_websocket_path}",
        ),
        (
            "https://localhost:8000/extra/",
            f"wss://localhost:8000/extra{url_websocket_path}",
        ),
        ("localhost", ValueError()),
        ("localhost:8000", ValueError()),
        ("localhost/extra", ValueError()),
        ("localhost:8000/extra", ValueError()),
        ("ws://localhost:8000/", f"ws://localhost:8000{url_websocket_path}"),
        (
            f"wss://localhost:8000{url_websocket_path}",
            f"wss://localhost:8000{url_websocket_path}",
        ),
        (url_websocket, url_websocket),
    ]
    for url_test, expected in test_expected:
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError):
                SpayaClientAsync(url=url_test, authorization=BearerToken(token="token"))
        else:
            client = SpayaClientAsync(
                url=url_test, authorization=BearerToken(token="token")
            )
            assert client._url_websocket == expected


def test_error_url(token):
    with pytest.raises(ValueError):
        SpayaClientAsync(url="me@localhost", authorization=BearerToken(token=token))


@pytest.mark.asyncio
async def test_error_websocket_at_start(error_websocket_at_start, url_websocket, token):
    async for smiles_to_send in error_websocket_at_start:
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            with pytest.raises(websockets.WebSocketException):
                for smiles_list in smiles_to_send:
                    await client.start_retrosynthesis(smiles_list)


@pytest.mark.asyncio
async def test_error_websocket_consume(
    websocket_error_after_start, url_websocket, token
):
    async for smiles_to_send in websocket_error_after_start:
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            with pytest.raises(websockets.WebSocketException):
                async for smiles, score in client.consume():
                    LOGGER.info(f"{smiles}: {score}")


@pytest.mark.asyncio
async def test_error_websocket_wait_result(
    websocket_error_after_start, url_websocket, token
):
    async for smiles_to_send in websocket_error_after_start:
        async with SpayaClientAsync(
            url=url_websocket, authorization=BearerToken(token=token)
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            with pytest.raises(websockets.WebSocketException):
                await client.wait_result()


# Split function to stay compatible with responses.activate
async def status_async(url, token):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        status = await client.get_status()
        LOGGER.debug(f"status:{repr(status)}")
        assert status.queue_size == 3

        with pytest.raises(ValueError):
            await client.get_status()

        with pytest.raises(ConnectionError):
            await client.get_status()


@responses.activate
def test_status(url, token, url_status):
    responses.add(responses.GET, url_status, json={"queue_size": 3}, status=200)
    responses.add(responses.GET, url_status, body="NOPE", status=200)
    responses.add(responses.GET, url_status, body="NOPE", status=404)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(status_async(url, token))


# Split function to stay compatible with responses.activate
async def commercial_compounds_providers_async(url, token, providers):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        result_cc_providers = await client.get_commercial_compounds_providers()
        assert result_cc_providers == providers


@responses.activate
def test_commercial_compounds_providers(url, token, url_cc_providers):
    providers = ["molport", "dealer"]
    responses.add(
        responses.GET,
        url_cc_providers,
        json={"providers": ["molport", "dealer"]},
        status=200,
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(commercial_compounds_providers_async(url, token, providers))


# Split function to stay compatible with responses.activate
async def commercial_compounds_async(url, token, smiles):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        results = await client.get_commercial_compounds(smiles=smiles)
        assert len(results) == 1


@responses.activate
def test_commercial_compounds(url, token, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = "A"
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(commercial_compounds_async(url, token, smiles))


# Split function to stay compatible with responses.activate
async def name_reactions_async(url, token, name_reactions, filter_name_reactions):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        result_name_reactions = await client.get_name_reactions(
            filter_name_reactions=filter_name_reactions
        )
        assert result_name_reactions == name_reactions


@responses.activate
def test_name_reactions_async(url, token, url_name_reaction):
    name_reactions = ["name1", "name2"]
    filter_name_reactions = "name"
    responses.add(
        responses.GET,
        url_name_reaction,
        json={"name_reactions": name_reactions},
        status=200,
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        name_reactions_async(url, token, name_reactions, filter_name_reactions)
    )


async def route_nominal_dataframe_async(url, token):
    smiles_column = "smiles_input"
    df = DataFrame(
        {smiles_column: ["CCCC", "HHHHHH"], "other_info": ["CCCC", "HHHHHH"]}
    )

    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        result_routes = await client.routes(
            smiles=df, dataframe_smiles_column=smiles_column
        )

        LOGGER.info(f"result_routes: {result_routes}")

        assert len(result_routes) == 2
        route_cccc = result_routes["CCCC"]
        assert len(route_cccc) == 2
        assert route_cccc[0].rscore == 0.3
        assert route_cccc[0].nb_steps == 3
        assert route_cccc[0].tree is not None
        assert route_cccc[1].rscore == 0.4
        assert route_cccc[1].nb_steps == 1
        assert route_cccc[1].tree is not None
        route_hhhhhh = result_routes["HHHHHH"]
        assert len(route_hhhhhh) == 1
        assert route_hhhhhh[0].rscore is None
        assert route_hhhhhh[0].nb_steps is None
        assert route_hhhhhh[0].tree is None


@responses.activate
def test_route_nominal_dataframe(url, url_routes, token):
    responses.add(
        responses.POST,
        url_routes,
        json={
            "routes": [
                {
                    "root_smiles": "CCCC",
                    "rscore": 0.3,
                    "nb_steps": 3,
                    "tree": {"CCCC": {"OOO": {}}},
                },
                {
                    "root_smiles": "CCCC",
                    "rscore": 0.4,
                    "nb_steps": 1,
                    "tree": {"CCCC": {"HH": {}}},
                },
                {
                    "root_smiles": "HHHHHH",
                    "rscore": None,
                    "nb_steps": None,
                    "tree": None,
                },
            ]
        },
        status=200,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(route_nominal_dataframe_async(url, token))


async def cluster_nominal_no_extra_async(url, token, smiles_list, check):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        result = await client.clustering(
            smiles=smiles_list,
            min_relative_size=0.1,
            max_cluster=42,
            max_coverage=0.9,
            alpha=0.5,
            min_route_rscore=0.6,
        )
        assert result.status == StatusCode.DONE
        assert len(result.clusters) == 2
        assert result.clusters[0].key == "C"
        assert result.clusters[0].smiles == smiles_list[0:3]
        assert result.clusters[0].mean_depths == 0.1
        assert result.clusters[0].mean_max_score == 0.8
        assert result.clusters[1].key == "O"
        assert result.clusters[1].smiles == smiles_list[3:]
        assert result.clusters[1].mean_depths == 0.2
        assert result.clusters[1].mean_max_score == 0.9

        check()


@responses.activate
def test_cluster_nominal_no_extra(cluster_nominal_scenario_no_extra, url, token):
    smiles_list, check = cluster_nominal_scenario_no_extra

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        cluster_nominal_no_extra_async(url, token, smiles_list, check)
    )


# Split function to stay compatible with responses.activate
async def quota_async(url, token, quota_result):
    async with SpayaClientAsync(
        url=url, authorization=BearerToken(token=token)
    ) as client:
        result_quota = await client.get_retrosynthesis_quota()
        assert result_quota == quota_result


@responses.activate
def test_quota(url, url_retrosynthesis_quota, token):
    responses.add(
        responses.GET,
        url_retrosynthesis_quota,
        json={"retrosynthesis_left": 2},
        status=200,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(quota_async(url, token, 2))
