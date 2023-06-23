"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import pytest
import asyncio
import logging
from typing import Dict

import websockets
from threading import Event

from iktos.spaya import (
    BearerToken,
    RetrosynthesisResult,
    SpayaClientCallback,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_callback(websocket_nominal, url_websocket, token):
    async for smiles_to_send, check in websocket_nominal:
        nb_smiles = 0
        for s in smiles_to_send:
            if isinstance(s, list):
                nb_smiles += len(s)
            else:
                nb_smiles += 1

        result = dict()  # type: Dict[str, RetrosynthesisResult]

        async def callback(smiles: str, score: RetrosynthesisResult):
            LOGGER.info(f"{smiles}: {score}")
            result[smiles] = score

        async with SpayaClientCallback(
            url=url_websocket, authorization=BearerToken(token=token), callback=callback
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            while nb_smiles > len(result):
                await asyncio.sleep(0.1)

        check(result)


@pytest.mark.asyncio
async def test_reconnect(websocket_stop_and_continue, url_websocket, token):
    async for smiles_to_send, check in websocket_stop_and_continue:
        nb_smiles = 0
        for s in smiles_to_send:
            if isinstance(s, list):
                nb_smiles += len(s)
            else:
                nb_smiles += 1

        result = dict()  # type: Dict[str, RetrosynthesisResult]

        async def callback(smiles: str, score: RetrosynthesisResult):
            result[smiles] = score

        async with SpayaClientCallback(
            url=url_websocket, authorization=BearerToken(token=token), callback=callback
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            while nb_smiles > len(result):
                await asyncio.sleep(0.1)

        check(result)


@pytest.mark.asyncio
async def test_error_websocket(error_websocket_at_start, url_websocket, token):
    async for smiles_to_send in error_websocket_at_start:

        async def callback(smiles: str, score: RetrosynthesisResult):
            LOGGER.info(f"{smiles}: {score}")
            assert False  # never called

        async with SpayaClientCallback(
            url=url_websocket, authorization=BearerToken(token=token), callback=callback
        ) as client:
            with pytest.raises(websockets.WebSocketException):
                while True:
                    for smiles_list in smiles_to_send:
                        await client.start_retrosynthesis(smiles_list)
                    await asyncio.sleep(0.3)


@pytest.mark.asyncio
async def test_error_websocket_error_callback(
    websocket_error_after_start, url_websocket, token
):
    async for smiles_to_send in websocket_error_after_start:

        async def callback(smiles: str, score: RetrosynthesisResult):
            LOGGER.info(f"{smiles}: {score}")
            await asyncio.sleep(0.3)
            assert False  # never called

        async def callback_error(s_client: SpayaClientCallback, e: BaseException):
            pass

        async def progression(progress: float):
            LOGGER.info(f"progression:{progress}")

        async with SpayaClientCallback(
            url=url_websocket,
            authorization=BearerToken(token=token),
            callback=callback,
            error_callback=callback_error,
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            await client.wait_result(callback_progression=progression)


@pytest.mark.asyncio
async def test_error_websocket_error_callback_2(
    error_websocket_at_start, url_websocket, token
):
    async for smiles_to_send in error_websocket_at_start:

        async def callback(smiles: str, score: RetrosynthesisResult):
            LOGGER.info(f"{smiles}: {score}")
            await asyncio.sleep(0.3)
            assert False  # never callede

        async def callback_error(s_client: SpayaClientCallback, e: BaseException):
            # Check if user can close in callback
            await s_client.close()
            await asyncio.sleep(0.2)
            raise e

        exception_found = False
        async with SpayaClientCallback(
            url=url_websocket,
            authorization=BearerToken(token=token),
            callback=callback,
            error_callback=callback_error,
        ) as client:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
            try:
                await client.wait_result()
            except websockets.WebSocketException:
                exception_found = True
        assert exception_found


@pytest.mark.asyncio
async def test_error_noserver(url_websocket, token):
    smiles_to_send = ["C"]

    async def callback(smiles: str, score: RetrosynthesisResult):
        LOGGER.info(f"{smiles}: {score}")
        assert False  # never called

    async with SpayaClientCallback(
        url=url_websocket, authorization=BearerToken(token=token), callback=callback
    ) as client:
        with pytest.raises(OSError):
            while True:
                for smiles_list in smiles_to_send:
                    await client.start_retrosynthesis(smiles_list)
                await asyncio.sleep(0.3)


@pytest.mark.asyncio
async def test_error_nocallback(url_websocket, token):
    smiles_to_send = ["C"]

    async def callback(smiles: str, score: RetrosynthesisResult):
        LOGGER.info(f"{smiles}: {score}")
        assert False  # never called

    client = SpayaClientCallback(
        url=url_websocket, authorization=BearerToken(token=token), callback=callback
    )
    with pytest.raises(Exception):
        while True:
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)


@pytest.mark.asyncio
async def test_error_no_double_callback(url_websocket, token):
    async def callback(smiles: str, score: RetrosynthesisResult):
        LOGGER.info(f"{smiles}: {score}")
        assert False  # never called

    client = SpayaClientCallback(
        url=url_websocket, authorization=BearerToken(token=token), callback=callback
    )
    await client.start_callback()
    await client.start_callback()
    await client.close()


@pytest.mark.asyncio
async def test_error_callback_on_start(
    websocket_nominal_already_done, url_websocket, token
):
    async for smiles_to_send, check in websocket_nominal_already_done:
        wait_for_kill = Event()

        async def callback(smiles: str, score: RetrosynthesisResult):
            LOGGER.info(f"{smiles}: {score}")
            wait_for_kill.set()
            raise AttributeError("TEST")

        client = SpayaClientCallback(
            url=url_websocket, authorization=BearerToken(token=token), callback=callback
        )
        await client.start_callback()

        dont_wait_too_long = 100
        with pytest.raises(AttributeError):
            for smiles_list in smiles_to_send:
                await client.start_retrosynthesis(smiles_list)
                while not wait_for_kill.is_set():
                    await asyncio.sleep(0.2)
                    dont_wait_too_long -= 1
                    if dont_wait_too_long < 0:
                        raise Exception("TAke too long")
                await client.start_retrosynthesis(smiles_list)
