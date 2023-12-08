#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.

import json
import asyncio
import logging
from typing import Union, List, Optional, Callable, Awaitable
from threading import Event

import websockets
from pandas import DataFrame

from iktos.spaya.authorization import Authorization
from iktos.spaya.spaya_client_async import SpayaClientAsync
from iktos.spaya.model import (
    RetrosynthesisParameters,
    RetrosynthesisResult,
    SettingsCallback,
)

LOGGER = logging.getLogger(__name__)


class SpayaClientCallback(SpayaClientAsync):
    """
    This async client send SMILES to the Spaya websocket and
    gives the results through a callback

    Examples:
        >>> import asyncio
        ... from iktos.spaya import (
        ...     BearerToken, SpayaClientCallback, RetrosynthesisResult
        ... )
        ...
        ... async def generator(client: SpayaClientCallback):
        ...     # Generate and start scoring SMILES
        ...     for smiles in ["O=C1CCCCO1", "O=C1CCCNN1",]:
        ...         await client.start_retrosynthesis(smiles)
        ...
        ... async def callback(smiles: str, result: RetrosynthesisResult):
        ...     # Handle the results
        ...     print(f"{smiles}: {result.rscore} / {result.nb_steps}")
        ...
        ... async def generate_and_score(url: str):
        ...     async with SpayaClientCallback(url=url,
        ...                                    authorization=BearerToken("myT0ken"),
        ...                                    callback=callback) as client:
        ...         # Generate SMILES
        ...         await generator(client)
        ...
        ...         # Block until the ends
        ...         await client.wait_result()
        ...
        ... asyncio.run(generate_and_score(url="https://spaya.ai"))
    """

    __slots__ = ["_callback", "_future", "_error_callback", "_is_closing"]

    def __init__(
        self,
        callback: Callable[[str, RetrosynthesisResult], Awaitable],
        url: str,
        authorization: Authorization,
        parameters: Optional[RetrosynthesisParameters] = None,
        error_callback: Optional[
            Callable[["SpayaClientCallback", BaseException], Awaitable]
        ] = None,
        settings: Optional[SettingsCallback] = None,
    ):
        """
        Args:
            callback: An async callable(smiles, result)
            url: URL to the Spaya API
            authorization: Login authorization
            parameters: Retrosynthesis algorithm parameters
            error_callback: Callback called in case of error or disconnection
            settings: Client settings
        """
        if settings is None:
            settings = SettingsCallback()
        super().__init__(
            url=url,
            authorization=authorization,
            parameters=parameters,
            settings=settings,
        )
        self._callback = callback
        self._future: Optional[asyncio.Task] = None
        self._error_callback = error_callback
        self._is_closing = Event()

    async def __aenter__(self):
        await super().__aenter__()
        await self.start_callback()
        return self

    async def start_retrosynthesis(
        self,
        smiles: Union[str, List[str], DataFrame],
        dataframe_smiles_column: str = "smiles",
    ) -> None:
        """
        Add SMILES to score

        Args:
            smiles: SMILES to score
            dataframe_smiles_column: smiles column name in dataframe
        """
        if self._future is not None and not self._future.done():
            smiles_list = self._extract_list_smiles(
                smiles=smiles, dataframe_smiles_column=dataframe_smiles_column
            )
            websocket = await self._get_websocket()
            # We have to save the smiles else wait result cannot work until
            # we receive the first result
            for s in smiles_list:
                if s not in self._smiles_left:
                    self._smiles_left[s] = RetrosynthesisResult()
            await self._send(websocket=websocket, smiles=smiles_list)
        elif self._future is None:
            raise Exception("Start callback first")
        elif self._future.done():
            await self._no_future()

    async def wait_result(
        self, callback_progression: Optional[Callable[[float], Awaitable]] = None
    ) -> None:
        """
        Block until all the SMILES are scored

        Args:
            callback_progression: a callable with progression as parameter
        """
        while (
            self._smiles_left
            and not self._stop.is_set()
            and self._future is not None
            and not self._future.done()
        ):
            if callback_progression is not None:
                await callback_progression(self.progression)
            await asyncio.sleep(0.5)
        await self._no_future()

    async def _callback_task(self) -> None:
        """
        Asyncio.Task loop consuming the result and calling the callback
        """
        try:
            while not self._stop.is_set():
                try:
                    websocket = await self._get_websocket()
                    while not self._stop.is_set():
                        response = await websocket.recv()
                        response_json = json.loads(response)
                        self._update_result(response_json=response_json)
                        while self._smiles_done:
                            smiles, result = self._smiles_done.popitem()
                            await self._callback(smiles, result)
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                await asyncio.sleep(0.5)

        except Exception as e:
            if self._error_callback is not None:
                await self._error_callback(self, e)
            else:
                raise e

    async def start_callback(self) -> None:
        """
        Start the task consuming the result and calling the callback function
        """
        if self._future is not None:
            await self._no_future()
        if self._future is None:
            self._future = asyncio.create_task(self._callback_task())

    async def close(self) -> None:
        """
        Close the websocket and stop the callback
        """
        try:
            # Avoid recursion loop if user close in error callback
            if not self._is_closing.is_set():
                self._is_closing.set()
                await super().close()
                await self._no_future()
        finally:
            self._is_closing.clear()

    async def _no_future(self) -> None:
        """
        Make sure the _future did not return any exception
        Call _error_callback
        Returns:

        """
        if self._future is not None:
            if self._future != asyncio.current_task():
                if not self._future.done() and self._stop.is_set():
                    self._future.cancel()
                elif self._future.done():
                    e = self._future.exception()
                    self._future = None
                    if e is not None:
                        if self._error_callback is not None:
                            await self._error_callback(self, e)
                        else:
                            raise e
