#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.

import time
import logging
from datetime import datetime, timedelta
from typing import Union, List, Optional, Tuple, Generator, Dict, Callable

import requests
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
    SettingsREST,
)


LOGGER = logging.getLogger(__name__)


class SpayaClientREST(SpayaClient):
    """
    This client sends SMILES to the Spaya API to get an RScore.

    Examples:
        >>> from iktos.spaya import BearerToken, SpayaClientREST
        ...
        ... # Create client with authorization
        ... client = SpayaClientREST(url="https://spaya.ai",
        ...                          authorization=BearerToken("myT0ken"))
        ...
        ... # Start a retrosynthesis and wait for the results
        ... scored_smiles = client.score_smiles(smiles=["O=C1CCCCO1", "O=C1CCCNN1",])
        ...
        ... # Show the results
        ... for smiles, result in scored_smiles.items():
        ...     print(f"{smiles}: {result.rscore} / {result.nb_steps}")

        >>> from iktos.spaya import BearerToken, SpayaClientREST
        ... from pandas import DataFrame
        ...
        ... # Create client with authorization
        ... client = SpayaClientREST(url="https://spaya.ai",
        ...                          authorization=BearerToken("myT0ken"))
        ...
        ... # Start a retrosynthesis and wait for the results with a dataframe
        ... df = DataFrame({"input_smiles": ["O=C1CCCCO1", "O=C1CCCNN1",]})
        ... df = client.score_smiles(
        ...     smiles=df,
        ...     dataframe_smiles_column="input_smiles",
        ...     callback_progression=lambda p: print(f"progression:{p}")
        ... )
        ...
        ... # Show the results
        ... print(f"result: {df}")

        >>> from iktos.spaya import BearerToken, SpayaClientREST
        ...
        ... # Create client with authorization
        ... client = SpayaClientREST(url="https://spaya.ai",
        ...                          authorization=BearerToken("myT0ken"))
        ...
        ... # Start a retrosynthesis
        ... client.start_retrosynthesis(smiles=["O=C1CCCCO1", "O=C1CCCNN1",])
        ...
        ... # Consume the results as soon as possible
        ... while not client.is_empty:
        ...     for smiles, result in client.consume():
        ...         print(f"{smiles} {result}")
    """

    __slots__ = ["_last_update", "__minimum_update_period", "_settings_rest"]

    def __init__(
        self,
        url: str,
        authorization: Authorization,
        parameters: Optional[RetrosynthesisParameters] = None,
        settings: Optional[SettingsREST] = None,
    ):
        """

        Args:
            url: URL to the Spaya API
            authorization: Login authorization
            parameters: Retrosynthesis algorithm parameters
            settings: Client settings
        """
        if settings is None:
            settings = SettingsREST()
        if settings.minimum_update_period < 1:
            raise ValueError("minimum_update_period must be >= 1")
        super().__init__(
            url=url,
            authorization=authorization,
            parameters=parameters,
            settings=settings,
        )
        self._last_update = datetime.min

        self._settings_rest = settings
        self.__minimum_update_period = timedelta(
            seconds=self._settings_rest.minimum_update_period
        )

    @property
    def progression(self) -> float:
        self._update()
        return super().progression

    @property
    def is_retro_finished(self) -> bool:
        self._update()
        return not self._smiles_left

    def get_status(self) -> Status:
        """
        Returns:
            Current spaya api status
        """
        return self._get_status()

    def get_commercial_compounds_providers(self) -> List[str]:
        """
        Returns:
            List of available commercial compounds providers.
        """
        return self._get_commercial_compounds_providers()

    def get_commercial_compounds(
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
        return self._get_commercial_compounds(
            smiles=smiles,
            provider=provider,
            catalog=catalog,
            packaging_g_min=packaging_g_min,
            packaging_g_max=packaging_g_max,
            price_per_g_min=price_per_g_min,
            price_per_g_max=price_per_g_max,
            delivery_date_max_day=delivery_date_max_day,
            delivery_included=delivery_included,
        )

    def get_name_reactions(
        self, filter_name_reactions: Optional[str] = None
    ) -> List[str]:
        """
        Get the possible value for name_reactions

        Args:
            filter_name_reactions: optional case unsensitive regex

        Returns:
            List of name reactions found.
        """
        return self._get_name_reactions(filter_name_reactions=filter_name_reactions)

    def score_smiles(
        self,
        smiles: Union[str, List[str], DataFrame],
        dataframe_smiles_column: str = "smiles",
        dataframe_rscore_column: str = "rscore",
        dataframe_nb_steps_column: str = "nb_steps",
        callback_progression: Optional[Callable[[float], None]] = None,
    ) -> Union[Dict[str, RetrosynthesisResult], DataFrame]:
        """
        Start and wait for the retrosynthesis score for one or many SMILES

        Args:
            smiles: one or a list of SMILES to be scored
            dataframe_smiles_column: smiles column name in dataframe
            dataframe_rscore_column: rscore column name to be created in dataframe
            dataframe_nb_steps_column: number of steps column name to be created in
              dataframe
            callback_progression: a callable with progression as parameter
        Returns:
            If smiles is str or a List: A dictionary of SMILES -> retrosynthesis result
            If smiles is a dataframe: a dataframe completed with rscore and nb_steps
        """
        self.start_retrosynthesis(
            smiles, dataframe_smiles_column=dataframe_smiles_column
        )
        self.wait_result(callback_progression=callback_progression)
        if isinstance(smiles, DataFrame):
            return self.pop_finished_to_dataframe(
                df=smiles,
                smiles_column=dataframe_smiles_column,
                rscore_column=dataframe_rscore_column,
                nb_steps_column=dataframe_nb_steps_column,
            )
        else:
            result = self._smiles_done
            self._smiles_done = dict()
            return result

    def start_retrosynthesis(
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
        max_smiles_per_request = self._settings.max_smiles_per_request
        for index in range(0, len(smiles_list), max_smiles_per_request):
            self._send_entry(smiles_list[index : index + max_smiles_per_request])

    def wait_result(
        self,
        callback_progression: Optional[Callable[[float], None]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Wait for the retrosynthesis of all SMILES

        Args:
            callback_progression: a callable with progression as parameter
            timeout: stop waiting after a value in second
        """
        if timeout is None:
            end = datetime.max
        else:
            end = datetime.now() + timedelta(seconds=timeout)
        while True:
            self._update()
            if self.is_retro_finished or end < datetime.now():
                break
            if callback_progression is not None:
                callback_progression(self.progression)
            time.sleep(
                max(
                    0.0,
                    min(
                        self.__minimum_update_period, end - datetime.now()
                    ).total_seconds(),
                )
            )

    def consume(self) -> Generator[Tuple[str, RetrosynthesisResult], None, None]:
        """
        Update and return all finished SMILES and remove them

        Returns:
            All finished SMILES
        """
        self._update()
        for smiles, result in self._smiles_done.items():
            yield smiles, result
        self._smiles_done = {}

    def _update(self) -> None:
        """
        Send smiles to the API to get score and progress update
        This function can only be called every <minimum_update_period> to avoid
        too much calls as most of the function call it.
        """
        if datetime.now() < self._last_update + self.__minimum_update_period:
            # Avoid too much calls
            return

        self._send_entry(
            smiles=list(self._smiles_left.keys())[
                0 : self._settings.max_smiles_per_request
            ]
        )

    def _send_entry(self, smiles: List[str]):
        """
        Send smiles to the API to get score and progress update
        The number of smiles must be lower than <max_smiles_per_request>
        """
        if not smiles:
            return

        entry = self._create_entry(smiles=smiles)
        response_json = self._send_and_retry(
            func=requests.post, endpoint="/batch-smiles", json=entry
        )
        self._last_update = datetime.now()
        self._update_result_batch(response_json=response_json)

    def routes(
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
            ... from iktos.spaya import BearerToken, SpayaClientREST
            ...
            ... # 1.1- Create client with authorization
            ... client = SpayaClientREST(url="https://spaya.ai",
            ...                          authorization=BearerToken("myT0ken"))
            ...
            ... # 1.2- Start a retrosynthesis and wait for the results
            ... smiles_list = ["O=C1CCCCO1", "O=C1CCCNN1"]
            ... client.score_smiles(smiles=smiles_list)
            ...
            ... # 1.3- Get best 2 routes for each smiles
            ... best_routes = client.routes(smiles=smiles_list, top_k_routes=2)
            ...
            ... # 1.4- Show results
            ... for smiles_str, routes_list in best_routes.items():
            ...     for route in routes_list:
            ...         print(f"{smiles_str}: {route.rscore} / {route.nb_steps}"
            ...               f" -> {route.tree}")

            >>> # 2. Using a DataFrame
            ... from iktos.spaya import BearerToken, SpayaClientREST
            ...
            ... # 2.1- Create client with authorization
            ... client = SpayaClientREST(url="https://spaya.ai",
            ...                          authorization=BearerToken("myT0ken"))
            ...
            ... # 2.2- Start a retrosynthesis and wait for the results
            ... df = DataFrame({"input_smiles": ["O=C1CCCCO1", "O=C1CCCNN1",]})
            ... client.score_smiles(smiles=df, dataframe_smiles_column="input_smiles")
            ...
            ... # 2.3- Get best 2 routes for each smiles
            ... best_routes = client.routes(
            ...     smiles=df,
            ...     top_k_routes=2,
            ...     dataframe_smiles_column="input_smiles",
            ... )
            ...
            ... # 2.4- Show results
            ... for smiles_str, routes_list in best_routes.items():
            ...     for route in routes_list:
            ...         print(f"{smiles_str}: {route.rscore} / {route.nb_steps}"
            ...               f" -> {route.tree}")
        """
        return self._routes(
            smiles=smiles,
            top_k_routes=top_k_routes,
            dataframe_smiles_column=dataframe_smiles_column,
        )

    def clustering(
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
            >>> from iktos.spaya import BearerToken, SpayaClientREST
            ...
            ... # 1- Create client with authorization
            ... client = SpayaClientREST(
            ...     url="https://spaya.ai",
            ...     authorization=BearerToken("myT0ken"),
            ...     parameters=RetrosynthesisParameters(
            ...         early_stopping_timeout=1
            ...     )
            ... )
            ...
            ... # 2- Start a retrosynthesis and wait for the results
            ... smiles_list = ["O=C1CCCCO1", "O=C1CCCNN1"]
            ... client.score_smiles(smiles=smiles_list)
            ...
            ... # 3- Get clusters
            ... best_routes = client.clustering(
            ...     smiles=smiles_list,
            ...     min_relative_size=0.1,
            ...     max_cluster=10,
            ...     max_coverage=0.95,
            ...     alpha=0.,
            ...     extra_smiles=[
            ...         (
            ...             RetrosynthesisParameters(early_stopping_timeout=3),
            ...             ["Cc1occc(=O)c1O", "CN1C=CCSC1=N"]
            ...         )
            ...     ]
            ... )
            ...
            ... # 4- Show results
            ... for cluster in best_routes.clusters:
            ...     print(f"{cluster.key}:"
            ...     print(f"  mean_depths:{cluster.mean_depths}")
            ...     print(f"  mean_max_score:{cluster.mean_max_score}")
            ...     print(f"  smiles:{cluster.smiles}")
        """
        return self._clustering(
            smiles=smiles,
            sleep_timeout=self.__minimum_update_period.total_seconds(),
            dataframe_smiles_column=dataframe_smiles_column,
            min_relative_size=min_relative_size,
            max_cluster=max_cluster,
            max_coverage=max_coverage,
            alpha=alpha,
            min_route_rscore=min_route_rscore,
            extra_smiles=extra_smiles,
        )

    def get_retrosynthesis_quota(self) -> Optional[int]:
        """
        Returns:
            The number of retrosynthesis left or None if illimited
        """
        return self._get_retrosynthesis_quota()
