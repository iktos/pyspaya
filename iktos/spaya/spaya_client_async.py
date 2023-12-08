#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.

import os
import json
import urllib.parse
import asyncio
import logging
from typing import (
    Union,
    List,
    Optional,
    Tuple,
    AsyncGenerator,
    Dict,
    Callable,
    Awaitable,
)
from threading import Event

import websockets
from pandas import DataFrame, Series

from iktos.spaya.authorization import Authorization
from iktos.spaya.spaya_client import SpayaClient
from iktos.spaya.model import (
    Catalog,
    ClusteringResult,
    CommercialCompound,
    RetrosynthesisParameters,
    RetrosynthesisResult,
    Status,
    Route,
    SettingsAsync,
)


LOGGER = logging.getLogger(__name__)


class SpayaClientAsync(SpayaClient):
    """
    This async client send SMILES to the Spaya websocket

    Examples:
        >>> import asyncio
        ... from iktos.spaya import BearerToken, SpayaClientAsync
        ...
        ... async def score():
        ...     async with SpayaClientAsync(url="https://spaya.ai",
        ...                                 authorization=BearerToken("myT0ken")
        ...     ) as client:
        ...         # Start scoring SMILES
        ...         await client.start_retrosynthesis(["O=C1CCCCO1", "O=C1CCCNN1",])
        ...
        ...         # Wait and print scores as soon as received
        ...         async for smiles, result in client.consume():
        ...             print(f"{smiles}: {result.rscore} / {result.nb_steps}")
        ...
        ... asyncio.run(score())
    """

    __slots__ = [
        "_url_websocket",
        "_websocket",
        "_stop",
        "_lock_websocket_creation",
        "_smiles_to_resend",
    ]

    def __init__(
        self,
        url: str,
        authorization: Authorization,
        parameters: Optional[RetrosynthesisParameters] = None,
        settings: Optional[SettingsAsync] = None,
    ):
        """
        Args:
            url: URL to the Spaya API
            authorization: Login authorization
            parameters: Retrosynthesis algorithm parameters
            settings: Client settings
        """
        if settings is None:
            settings = SettingsAsync()
        super().__init__(
            url=url,
            authorization=authorization,
            parameters=parameters,
            settings=settings,
        )
        self._url_websocket = self._parse_url(url)
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._stop = Event()
        self._lock_websocket_creation = asyncio.Lock()

    def _parse_url(self, url):
        """
        Transform any url to a ws://host:port/path/ws/<version>/batch_smiles
        Args:
            url: the url provided

        Returns:
            A transformed url
        """
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme.startswith("http") and not parsed.scheme.startswith("ws"):
            raise ValueError(f"Bad url: {url}")

        endpoint = f"{self.ROOT}/ws{self.VERSION}/batch-smiles"
        if parsed.path.endswith(endpoint):
            path = parsed.path
        else:
            if endpoint.startswith("/"):
                # To make join works
                endpoint = endpoint[1:]
            path = os.path.join(parsed.path, endpoint)

        if not path.startswith("/"):
            path = "/" + path

        if parsed.scheme.startswith("https") or parsed.scheme.startswith("wss"):
            scheme = "wss"
        else:
            scheme = "ws"
        path = f"{scheme}://{parsed.netloc}{path}"
        return path

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """
        Close the websocket and stop any blocking method
        """
        self._stop.set()
        if self._websocket is not None:
            await self._websocket.wait_closed()

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
        smiles_list = self._extract_list_smiles(
            smiles=smiles, dataframe_smiles_column=dataframe_smiles_column
        )
        websocket = await self._get_websocket()
        await self._send(websocket=websocket, smiles=smiles_list)
        # Wait until all smiles are at least SUBMITED
        while smiles_list and not self._stop.is_set():
            try:
                websocket = await self._get_websocket()
                while smiles_list and not self._stop.is_set():
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    smiles, result = self._update_result(response_json=response_json)
                    if result.status.need_retry:
                        LOGGER.debug("Queue full, resending")
                        await asyncio.sleep(0.1)
                        await self._send(websocket=websocket, smiles=[smiles])
                    else:
                        try:
                            smiles_list.remove(smiles)
                        except ValueError:
                            pass
            except websockets.exceptions.ConnectionClosedOK:
                pass
            await asyncio.sleep(0.2)

    async def wait_result(
        self, callback_progression: Optional[Callable[[float], Awaitable]] = None
    ) -> None:
        """
        Block until all the SMILES are scored

        Args:
            callback_progression: a callable with progression as parameter
        """
        while self._smiles_left and not self._stop.is_set():
            try:
                websocket = await self._get_websocket()
                while self._smiles_left and not self._stop.is_set():
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    if callback_progression is not None:
                        await callback_progression(self.progression)
                    self._update_result(response_json=response_json)
            except websockets.exceptions.ConnectionClosedOK:
                pass
            await asyncio.sleep(0.2)

    async def consume(self) -> AsyncGenerator[Tuple[str, RetrosynthesisResult], None]:
        """
        Update, return all finished SMILES results and remove them

        Returns:
            A generator of SMILES result
        """
        while self._smiles_left and not self._stop.is_set():
            try:
                websocket = await self._get_websocket()
                while self._smiles_left and not self._stop.is_set():
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    self._update_result(response_json=response_json)
                    while self._smiles_done:
                        yield self._smiles_done.popitem()
            except websockets.exceptions.ConnectionClosedOK:
                pass
            await asyncio.sleep(0.5)
        while self._smiles_done:
            yield self._smiles_done.popitem()

    async def _get_websocket(self) -> websockets.WebSocketClientProtocol:
        """
        Returns: a connected websocket
        """
        async with self._lock_websocket_creation:
            if self._websocket is None or self._websocket.closed:
                self._websocket = await websockets.connect(
                    self._url_websocket, extra_headers=self._authorization.headers()
                )
                if self._smiles_left:
                    await self._send(
                        websocket=self._websocket, smiles=list(self._smiles_left.keys())
                    )
        return self._websocket

    async def _send(
        self, websocket: websockets.WebSocketClientProtocol, smiles: List[str]
    ) -> None:
        """
        Send SMILES to be scored
        Args:
            websocket: a connected websocket
            smiles: a list of SMILES to send
        """
        max_smiles_per_request = self._settings.max_smiles_per_request
        for index in range(0, len(smiles), max_smiles_per_request):
            entry = self._create_entry(
                smiles=smiles[index : index + max_smiles_per_request]
            )
            await websocket.send(json.dumps(entry))

    async def get_status(self) -> Status:
        """
        Returns:
            Current spaya api status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_status)

    async def get_commercial_compounds_providers(self) -> List[str]:
        """
        Returns:
            List of available commercial compounds providers.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_commercial_compounds_providers
        )

    async def get_commercial_compounds(
        self,
        smiles: Union[str, List[str], Route, Series],
        provider: Optional[List[str]] = None,
        catalog: Optional[List[Catalog]] = None,
        packaging_g_min: Optional[float] = None,
        packaging_g_max: Optional[float] = None,
        price_per_g_min: Optional[float] = None,
        price_per_g_max: Optional[float] = None,
        delivery_date_max_day: Optional[int] = None,
        delivery_included: Optional[bool] = None,
    ) -> Dict[str, List[CommercialCompound]]:
        """
        Access to commercial compounds

        Args:
            smiles: Molecules represented as either, a SMILES, a list of SMILES,
             a Route, or a Series.
            provider: List of desired commercial compounds providers.
             None or an empty list select them all
            catalog: Select the type of compounds
             (building block / screening / virtual).
             A null value or an empty list select them all
            packaging_g_min: Minimum packaging in gramme for a commercial compounds
            packaging_g_max: Maximum packaging in gramme for a commercial compounds
            price_per_g_min: Minimum price per gramme for a commercial compounds
            price_per_g_max: Maximum price per gramme for a commercial compounds
            delivery_date_max_day: Maximum delivery time in day.
             A null value select them all
            delivery_included: If True the comparaison will done with
             the maximum delivery date. If False the comparaison will done with
             the minimum delivery date",

        Returns:
            A dictionnary SMILES -> commercial compounds found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._get_commercial_compounds,
            smiles,
            provider,
            catalog,
            packaging_g_min,
            packaging_g_max,
            price_per_g_min,
            price_per_g_max,
            delivery_date_max_day,
            delivery_included,
        )

    async def get_name_reactions(
        self, filter_name_reactions: Optional[str] = None
    ) -> List[str]:
        """
        Get the possible value for name_reactions

        Args:
            filter_name_reactions: optional case unsensitive regex

        Returns:
            List of name reactions found.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_name_reactions, filter_name_reactions
        )

    async def routes(
        self,
        smiles: Union[str, List[str], DataFrame],
        top_k_routes: Optional[int] = None,
        dataframe_smiles_column: str = "smiles",
    ) -> Dict[str, List[Route]]:
        """
        Get routes for batches of SMILES, these SMILES should have
        been retrosynthesised first.

        Args:
            smiles: a list of smiles
            top_k_routes: Number of routes to fetch per molecule
            dataframe_smiles_column: smiles column name in dataframe

        Returns:
            A dictionnary of smiles and their routes

        Raises:
            ValueError: The SMILES are not DONE, and still need to be retrosynthesised

        Examples:
            >>> # 1. Using a list of smiles
            ... import asyncio
            ... from iktos.spaya import BearerToken, SpayaClientAsync
            ...
            ... async def get_route():
            ...     # 1.1- Create client with authorization
            ...     async with SpayaClientAsync(
            ...         url="https://spaya.ai",
            ...         authorization=BearerToken("myT0ken")
            ...     ) as client:
            ...         # 1.2- Start a retrosynthesis and wait for the results
            ...         smiles_list = ["O=C1CCCCO1", "O=C1CCCNN1"]
            ...         await client.start_retrosynthesis(smiles_list)
            ...         await client.wait_result()
            ...
            ...         # 1.3- Get best 2 routes for each smiles
            ...         best_routes = await client.routes(
            ...             smiles=smiles_list,
            ...             top_k_routes=2,
            ...         )
            ...
            ...         # 1.4- Show results
            ...         for smiles_str, routes_list in best_routes.items():
            ...             for route in routes_list:
            ...                 print(f"{smiles_str}: {route.rscore} / {route.nb_steps}"
            ...                       f" -> {route.tree}")
            ...
            ... asyncio.run(get_route())

            >>> # 2. Using a DataFrame
            ... import asyncio
            ... from iktos.spaya import BearerToken, SpayaClientAsync
            ...
            ... async def get_route():
            ...     # 2.1- Create client with authorization
            ...     async with SpayaClientAsync(
            ...         url="https://spaya.ai",
            ...         authorization=BearerToken("myT0ken")
            ...     ) as client:
            ...         # 2.2- Start a retrosynthesis and wait for the results
            ...         df = DataFrame({"input_smiles": ["O=C1CCCCO1", "O=C1CCCNN1",]})
            ...         await client.start_retrosynthesis(
            ...             smiles=df,
            ...             dataframe_smiles_column="input_smiles"
            ...         )
            ...         await client.wait_result()
            ...
            ...         # 1.3- Get best 2 routes for each smiles
            ...         best_routes = await client.routes(
            ...             smiles=df,
            ...             top_k_routes=2,
            ...             dataframe_smiles_column="input_smiles",
            ...         )
            ...
            ...         # 1.4- Show results
            ...         for smiles_str, routes_list in best_routes.items():
            ...             for route in routes_list:
            ...                 print(f"{smiles_str}: {route.rscore} / {route.nb_steps}"
            ...                       f" -> {route.tree}")
            ...
            ... asyncio.run(get_route())
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._routes, smiles, top_k_routes, dataframe_smiles_column
        )

    async def clustering(
        self,
        smiles: Union[str, List[str], DataFrame],
        dataframe_smiles_column: str = "smiles",
        min_relative_size: Optional[float] = None,
        max_cluster: Optional[int] = None,
        max_coverage: Optional[float] = None,
        alpha: Optional[float] = None,
        min_route_rscore: Optional[float] = None,
        extra_smiles: Optional[List[Tuple[RetrosynthesisParameters, List[str]]]] = None,
    ) -> ClusteringResult:
        """
        Get clusters for batches of SMILES, these SMILES should have been
        retrosynthesised first.

        Args:
            smiles: a list of smiles or a dataframe
            dataframe_smiles_column: smiles column name in dataframe
            min_relative_size: minimum relative number of atoms
                (compared to average size) for intermediates, values in [0 ; 1]
            max_cluster: Maximum number of clusters to create
            max_coverage: stopping criteria on the % of initial smiles in a cluster.
                values in [0 ; 1]
            alpha: relative importance of the size of the intermediate in
                the clusters' scoring, values in [0 ; 1]
            min_route_rscore: Minimum rscore to filter routes, values in [0 ; 1]
            extra_smiles: a list of different parameters and SMILES to add to this
                clustering. Only max_nb_iterations and early_stopping_timeout can be
                different

        Returns:
            List of clusters. One cluster contains routes that are grouped by
            the smiles they lead to

        Examples:
            >>> import asyncio
            ... from iktos.spaya import BearerToken, SpayaClientAsync
            ...
            ... async def get_clusters():
            ...     # 1- Create client with authorization
            ...     async with SpayaClientAsync(
            ...         url="https://spaya.ai",
            ...         authorization=BearerToken("myT0ken"),
            ...         parameters=RetrosynthesisParameters(
            ...             early_stopping_timeout=1
            ...         )
            ...     ) as client:
            ...         # 2- Start a retrosynthesis and wait for the results
            ...         smiles_list = ["O=C1CCCCO1", "O=C1CCCNN1"]
            ...         await client.start_retrosynthesis(smiles_list)
            ...         await client.wait_result()
            ...
            ...         # 3- Get clusters
            ...         best_routes = await client.clustering(
            ...             smiles=smiles_list,
            ...             min_relative_size=0.1,
            ...             max_cluster=10,
            ...             max_coverage=0.95,
            ...             alpha=0.,
            ...             extra_smiles=[
            ...                 (
            ...                     RetrosynthesisParameters(early_stopping_timeout=3),
            ...                     ["Cc1occc(=O)c1O", "CN1C=CCSC1=N"]
            ...                 )
            ...             ]
            ...         )
            ...
            ...         # 4- Show results
            ...         for cluster in best_routes.clusters:
            ...             print(f"{cluster.key}:"
            ...             print(f"  mean_depths:{cluster.mean_depths}")
            ...             print(f"  mean_max_score:{cluster.mean_max_score}")
            ...             print(f"  smiles:{cluster.smiles}")
            ...
            ... asyncio.run(get_clusters())
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._clustering,
            smiles,
            1,
            dataframe_smiles_column,
            min_relative_size,
            max_cluster,
            max_coverage,
            alpha,
            min_route_rscore,
            extra_smiles,
        )

    async def get_retrosynthesis_quota(self) -> Optional[int]:
        """
        Returns:
            The number of retrosynthesis left or None if illimited
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_retrosynthesis_quota)
