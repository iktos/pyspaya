#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.
import copy

import time
import urllib.parse
import logging
from time import sleep
from typing import Dict, Optional, List, Union, Any, Tuple

import requests
from pandas import DataFrame, Series

from iktos.spaya.authorization import Authorization
from iktos.spaya.model import (
    Catalog,
    ClusteringResult,
    CommercialCompound,
    RetrosynthesisParameters,
    RetrosynthesisResult,
    Route,
    Status,
    StatusCode,
    Settings,
)


LOGGER = logging.getLogger(__name__)


class SpayaClient:
    """
    Abstract class handling storage and helpers
    """

    __slots__ = [
        "_url",
        "_authorization",
        "_parameters",
        "_settings",
        "_smiles_left",
        "_smiles_done",
    ]
    ROOT = "/retrosynthesis-api"
    VERSION = "/v1"

    def __init__(
        self,
        url: str,
        authorization: Authorization,
        parameters: Optional[RetrosynthesisParameters] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Args:
            url: url to the spaya api
            authorization: Login authorization
            parameters: Retrosynthesis algorithms parameters
            settings: Client settings
        """
        self._url = url
        self._authorization = authorization
        if parameters is None:
            self._parameters = RetrosynthesisParameters()
        else:
            self._parameters = parameters
        if settings is None:
            settings = Settings()
        if settings.max_retry < 0:
            raise ValueError("max_retry must be >= 0")
        self._settings = settings
        self._smiles_left: Dict[str, RetrosynthesisResult] = dict()
        self._smiles_done: Dict[str, RetrosynthesisResult] = dict()

    @property
    def url(self) -> str:
        """
        Spaya URL
        """
        return self._url

    @property
    def parameters(self) -> RetrosynthesisParameters:
        """
        Retrosynthesis algorithm parameters
        """
        return self._parameters

    @property
    def progression(self) -> float:
        """
        Progression (as percentage) of SMILES not consumed
        """
        if self._smiles_left or self._smiles_done:
            progression = sum([run.progress for run in self._smiles_left.values()])
            left = len(self._smiles_left)
            done = len(self._smiles_done)
            return ((100.0 * done) + progression) / (left + done)
        else:
            return 0.0

    @property
    def is_retro_finished(self) -> bool:
        """
        True if all SMILES have been processed
        """
        return not self._smiles_left

    @property
    def is_empty(self) -> bool:
        """
        True if all SMILES have been consumed
        """
        return self.is_retro_finished and not self._smiles_done

    @property
    def unfinished_smiles(self) -> Dict[str, RetrosynthesisResult]:
        """
        All smiles submited and not finished
        """
        return self._smiles_left

    @property
    def _root_version_path(self) -> str:
        return f"{self.ROOT}{self.VERSION}"

    def remove(self, smiles: str) -> None:
        """
        Remove a SMILES from the client without removing the SMILES
        from the retrosynthesis queue

        Args:
            smiles: SMILES to remove

        Raises:
            KeyError: if the SMILES is not in the client
        """
        del self[smiles]

    def pop_finished(self, smiles: str) -> Optional[RetrosynthesisResult]:
        """
        Remove and return a Spaya RScore of a finished SMILES

        Args:
            smiles: SMILES to access

        Returns:
            A retrosynthesis result or None if the SMILES is not finished

        Raises:
            KeyError: if the SMILES is not in the client
        """
        result = self[smiles]
        if result.is_finished:
            del self[smiles]
            return result
        else:
            return None

    def _complete_dataframe_finished_smiles(self, elem: Series) -> Series:
        """
        Complete a dataframe if a SMILES is finished

        Args:
            elem: series of [SMILES, rscore, nb_steps]

        Returns:
            A series of updated rscore, nb_steps
        """
        try:
            smiles: str = elem[0]
            result = self[smiles]
            if result.is_finished:
                return Series([result.rscore, result.nb_steps])
        except KeyError:
            pass
        # Don't override value if nothing to update
        rscore: Optional[float] = elem[1]
        nb_steps: Optional[int] = elem[2]
        return Series([rscore, nb_steps])

    def _remove_dataframe_finished_smiles(self, smiles: str) -> None:
        """
        Removed returned SMILES from the client

        Args:
            smiles: a SMILES to check
        """
        try:
            result = self[smiles]
            if result.is_finished:
                del self[smiles]
        except KeyError:
            pass

    def pop_finished_to_dataframe(
        self,
        df: DataFrame,
        smiles_column: str = "smiles",
        rscore_column: str = "rscore",
        nb_steps_column: str = "nb_steps",
    ) -> DataFrame:
        """
        Remove and complete a dataframe with a Spaya RScore and number of steps for
        each finished smiles

        Args:
            df: dataframe to complete with RScore and number of steps
            smiles_column: smiles column name
            rscore_column: rscore column name
            nb_steps_column: number of steps column name

        Returns:
            A completed dataframe
        """
        # Create column once for all to have access to previous data during apply
        if rscore_column not in df.columns:
            df[rscore_column] = None
        if nb_steps_column not in df.columns:
            df[nb_steps_column] = None
        df[[rscore_column, nb_steps_column]] = df[
            [smiles_column, rscore_column, nb_steps_column]
        ].apply(self._complete_dataframe_finished_smiles, axis=1)
        df[smiles_column].apply(self._remove_dataframe_finished_smiles)
        return df

    @staticmethod
    def _extract_list_smiles(
        smiles: Union[str, List[str], DataFrame],
        dataframe_smiles_column: str = "smiles",
    ) -> List[str]:
        """
        Get a list of smiles from multiple type
        Args:
            smiles:
            dataframe_smiles_column: SMILES column in the DataFrame

        Returns:
            A list of smiles
        """
        if isinstance(smiles, DataFrame):
            return smiles[dataframe_smiles_column].tolist()
        elif isinstance(smiles, str):
            return [smiles]
        else:
            return copy.copy(smiles)

    def _update_result_batch(self, response_json: Dict) -> None:
        """
        Update all smiles result with an api response

        Args:
            response_json: Request response in json format
        """
        for response_smiles in response_json.get("smiles", []):
            self._update_result(response_json=response_smiles)

    def _update_result(self, response_json: Dict) -> Tuple[str, RetrosynthesisResult]:
        """
        Update a single smiles result with an api response

        Args:
            response_json: Request response in json format for one smiles
        """
        name = response_json["smiles"]
        if name in self._smiles_left:
            info = self._smiles_left[name]
            info.from_dict(response_json)
            if info.status.is_finished:
                self._smiles_done[name] = self._smiles_left.pop(name)
        elif name in self._smiles_done:
            info = self._smiles_done[name]
            if info.status.can_be_retried:
                info.from_dict(response_json)
                if not info.status.is_finished:
                    self._smiles_left[name] = self._smiles_done.pop(name)
        else:
            info = RetrosynthesisResult()
            info.from_dict(response_json)
            if info.status.is_finished:
                self._smiles_done[name] = info
            else:
                self._smiles_left[name] = info
        return name, info

    def _create_entry(self, smiles: List[str]) -> Dict:
        """
        Create a api request body

        Args:
            smiles: list of smiles to send

        Returns:
            A structure to be serialized
        """
        entry = self._parameters.to_dict()
        entry["batch_smiles"] = smiles
        return entry

    def __delitem__(self, smiles: str):
        """
        Remove a smiles from the client
        !! It doesn't remove the smiles from the retrosynthesis queue !!
        Args:
            smiles: smiles to removed

        Raises:
            KeyError: if the smiles is not in the client
        """
        try:
            del self._smiles_done[smiles]
        except KeyError:
            del self._smiles_left[smiles]

    def __contains__(self, smiles: str) -> bool:
        """

        Args:
            smiles: smiles to check

        Returns:
            True if the smiles in this client
        """
        return smiles in self._smiles_done or smiles in self._smiles_left

    def __getitem__(self, smiles: str):
        """

        Args:
            smiles: smiles to access

        Returns:
            Smiles current result

        Raises:
            KeyError: if the smiles is not in the client
        """
        try:
            return self._smiles_done[smiles]
        except KeyError:
            return self._smiles_left[smiles]

    def _send_and_retry(
        self, func, endpoint: str, json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        url_endpoint = urllib.parse.urljoin(
            self._url, f"{self._root_version_path}{endpoint}"
        )
        response = None
        for retry_index in range(0, self._settings.max_retry + 1):
            response = func(
                url=url_endpoint,
                headers=self._authorization.headers(),
                verify=self._settings.verify_tls,
                json=json,
            )
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as ve:
                    LOGGER.error(f"Received: {response.text}")
                    raise ve
            if retry_index < self._settings.max_retry:
                time.sleep(self._settings.retry_sleep)
        if response is not None:
            raise ConnectionError(f"Bad request: {response} {response.text}")
        else:
            raise AttributeError("max_retry < 0")

    def _get_status(self) -> Status:
        """
        Returns:
            Current spaya api status
        """
        response_json = self._send_and_retry(func=requests.get, endpoint="/status")
        return Status().from_dict(data=response_json)

    def _get_commercial_compounds_providers(self) -> List[str]:
        """
        Returns:
            List of available commercial compounds providers.
        """
        response_json = self._send_and_retry(
            func=requests.get, endpoint="/commercial-compounds-providers"
        )
        return response_json.get("providers", [])

    @staticmethod
    def _convert_smile_type(smiles: Union[str, List[str], Route, Series]) -> List[str]:
        """
        Convert any type of collection to a list of strings

        Args:
            smiles: Molecules represented as either, a SMILES, a list of SMILES,
             a Route, or a Series.
        """
        if isinstance(smiles, str):
            smiles_list: List[str] = [smiles]
        elif isinstance(smiles, Series):
            smiles_list = smiles.to_list()  # type: ignore[assignment]
        elif isinstance(smiles, Route):
            smiles_list = smiles.tree_leaf()
            root_smiles = smiles.root_smiles()
            if smiles_list and root_smiles is not None:
                smiles_list.append(root_smiles)
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            raise TypeError("unknown type for smiles")
        return smiles_list

    def _get_commercial_compounds(
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
        smiles_list = self._convert_smile_type(smiles)

        result: Dict[str, List[CommercialCompound]] = dict()
        entry = {}
        schema: List[Tuple[str, Optional[Any], Optional[Any]]] = [
            ("provider", self._parameters.cc_providers, provider),
            ("catalog", self._parameters.cc_catalog, catalog),
            ("packaging_g_min", None, packaging_g_min),
            ("packaging_g_max", None, packaging_g_max),
            ("price_per_g_min", None, price_per_g_min),
            ("price_per_g_max", self._parameters.cc_max_price_per_g, price_per_g_max),
            (
                "delivery_date_max_day",
                self._parameters.cc_max_delivery_days,
                delivery_date_max_day,
            ),
            ("delivery_included", None, delivery_included),
        ]
        for parameters_name, default_value, overwrite_value in schema:
            if overwrite_value is not None:
                entry[parameters_name] = overwrite_value
            elif default_value is not None:
                entry[parameters_name] = default_value

        for smiles in smiles_list:
            entry["smiles"] = smiles
            response_json = self._send_and_retry(
                func=requests.get, endpoint="/commercial-compounds", json=entry
            )
            results_cc: List[CommercialCompound] = list()
            for response_cc in response_json["commercial_compounds"]:
                results_cc.append(CommercialCompound().from_dict(response_cc))
            result[smiles] = results_cc
        return result

    def _get_name_reactions(
        self, filter_name_reactions: Optional[str] = None
    ) -> List[str]:
        """
        Get the possible value for name_reactions

        Args:
            filter_name_reactions: optional case unsensitive regex

        Returns:
            List of name reactions found.
        """
        response_json = self._send_and_retry(
            func=requests.get,
            endpoint="/name-reactions",
            json={"filter_name_reactions": filter_name_reactions},
        )
        return response_json.get("name_reactions", [])

    def _routes(
        self,
        smiles: Union[str, List[str], DataFrame],
        top_k_routes: Optional[int] = None,
        dataframe_smiles_column: str = "smiles",
    ) -> Dict[str, List[Route]]:
        """
        Get routes for batches of SMILES, these SMILES should have
        been retrosynthesised first.

        Args:
            smiles: a list of smiles or a dataframe
            top_k_routes: Number of routes to fetch per molecule
            dataframe_smiles_column: smiles column name in dataframe

        Returns:
            A dictionnary of smiles and their routes

        Raises:
            ValueError: The SMILES are not DONE, and still need to be retrosynthesised
        """
        smiles_list = self._extract_list_smiles(
            smiles=smiles, dataframe_smiles_column=dataframe_smiles_column
        )
        for smiles in smiles_list:
            if smiles in self._smiles_left:
                raise ValueError(f"The SMILES {smiles} is not DONE")

        route_entry: Dict[str, Union[str, int, dict]] = {
            "batch": self._create_entry(smiles=smiles_list)
        }
        if top_k_routes is not None:
            route_entry["top_k_routes"] = top_k_routes

        response_json = self._send_and_retry(
            func=requests.post, endpoint="/routes", json=route_entry
        )

        result: Dict[str, List[Route]] = dict()
        for route in response_json["routes"]:
            root_smiles = route["root_smiles"]
            if root_smiles not in result:
                result[root_smiles] = [Route().from_dict(route)]
            else:
                result[root_smiles].append(Route().from_dict(route))
        return result

    @staticmethod
    def _clustering_entry(
        batches: List[Tuple[RetrosynthesisParameters, List[str]]],
        min_relative_size: Optional[float],
        max_cluster: Optional[int],
        max_coverage: Optional[float],
        alpha: Optional[float],
        min_route_rscore: Optional[float],
    ) -> Dict[str, Union[List[Dict], int, float]]:
        batches_list = list()
        for parameters, batch_smiles in batches:
            entry = parameters.to_dict()
            entry["batch_smiles"] = batch_smiles
            batches_list.append(entry)

        clustering_entry: Dict[str, Union[List[Dict], int, float]] = {
            "batches": batches_list
        }
        if min_relative_size is not None:
            clustering_entry["min_relative_size"] = min_relative_size
        if max_cluster is not None:
            clustering_entry["max_cluster"] = max_cluster
        if max_coverage is not None:
            clustering_entry["max_coverage"] = max_coverage
        if alpha is not None:
            clustering_entry["alpha"] = alpha
        if min_route_rscore is not None:
            clustering_entry["min_route_rscore"] = min_route_rscore
        return clustering_entry

    def _clustering(
        self,
        smiles: Union[str, List[str], DataFrame],
        sleep_timeout: float,
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
        retrosynthesised first with the same RetrosynthesisParameters.

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
        """
        # Create all batches with nb smiles < max_smiles_per_request
        smiles_list = self._extract_list_smiles(
            smiles=smiles, dataframe_smiles_column=dataframe_smiles_column
        )
        batches_to_convert = [(self.parameters, smiles_list)]
        if extra_smiles is not None:
            batches_to_convert.extend(extra_smiles)
        max_smiles_per_request = self._settings.max_smiles_per_request
        for parameters, batch_smiles in batches_to_convert[:]:
            if not self.parameters.equal_no_timing(parameters):
                raise ValueError(
                    "Only max_nb_iterations and early_stopping_timeout can be different"
                )
            if len(batch_smiles) > self._settings.max_smiles_per_request:
                batches_to_convert.remove((parameters, batch_smiles))
                for i in range(0, len(batch_smiles), max_smiles_per_request):
                    batches_to_convert.append(
                        (parameters, batch_smiles[i : i + max_smiles_per_request])
                    )
            elif not batch_smiles:
                batches_to_convert.remove((parameters, batch_smiles))

        # Create structure
        clustering_entry = self._clustering_entry(
            batches=batches_to_convert,
            min_relative_size=min_relative_size,
            max_cluster=max_cluster,
            max_coverage=max_coverage,
            alpha=alpha,
            min_route_rscore=min_route_rscore,
        )

        while True:
            response_json = self._send_and_retry(
                func=requests.post, endpoint="/clustering", json=clustering_entry
            )
            status = StatusCode(
                response_json.get("status", StatusCode.INVALID_SMILES.value)
            )
            if status.is_finished:
                return ClusteringResult().from_dict(response_json)
            else:
                LOGGER.debug(f"status: {status}")
            sleep(sleep_timeout)

    def _get_retrosynthesis_quota(self) -> Optional[int]:
        """
        Returns:
            The number of retrosynthesis left or None if illimited
        """
        response_json = self._send_and_retry(func=requests.get, endpoint="/quota")
        return response_json.get("retrosynthesis_left")
