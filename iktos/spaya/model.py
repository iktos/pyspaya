#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.

import logging
from typing import Optional, Dict, Union, List, Any, Set
from enum import Enum


LOGGER = logging.getLogger(__name__)


class Settings:
    """
    Generic client settings
    """

    __slots__: List[str] = [
        "max_smiles_per_request",
        "verify_tls",
        "max_retry",
        "retry_sleep",
    ]

    def __init__(
        self,
        max_smiles_per_request: int = 1000,
        verify_tls: bool = True,
        max_retry: int = 2,
        retry_sleep: float = 10.0,
    ):
        """
        Args:
            max_smiles_per_request: maximum number of smiles per request
            verify_tls: verify the server's TLS certificate
            max_retry: maximum number of request retry before return an error
            retry_sleep: time to wait between two retry
        """
        self.max_smiles_per_request = max_smiles_per_request
        self.verify_tls = verify_tls
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep


class SettingsREST(Settings):
    """
    REST Client settings
    """

    __slots__: List[str] = ["minimum_update_period"]

    def __init__(
        self,
        max_smiles_per_request: int = 1000,
        verify_tls: bool = True,
        max_retry: int = 2,
        retry_sleep: float = 10.0,
        minimum_update_period: float = 2,
    ):
        """
        Args:
            max_smiles_per_request: maximum number of smiles per request
            verify_tls: verify the server's TLS certificate
            max_retry: maximum number of request retry before return an error
            retry_sleep: time to wait between two retry
            minimum_update_period: Minimum time in seconds between 2 requests
             to the API; must be >= 1.
        """
        super(SettingsREST, self).__init__(
            max_smiles_per_request=max_smiles_per_request,
            verify_tls=verify_tls,
            max_retry=max_retry,
            retry_sleep=retry_sleep,
        )
        self.minimum_update_period = minimum_update_period


class SettingsAsync(Settings):
    """
    Async Client settings
    """

    __slots__: List[str] = []

    def __init__(
        self,
        max_smiles_per_request: int = 1000,
        verify_tls: bool = True,
        max_retry: int = 2,
        retry_sleep: float = 10.0,
    ):
        """
        Args:
            max_smiles_per_request: maximum number of smiles per request
            verify_tls: verify the server's TLS certificate
            max_retry: maximum number of request retry before return an error
            retry_sleep: time to wait between two retry
        """
        super(SettingsAsync, self).__init__(
            max_smiles_per_request=max_smiles_per_request,
            verify_tls=verify_tls,
            max_retry=max_retry,
            retry_sleep=retry_sleep,
        )


class SettingsCallback(SettingsAsync):
    """
    Callback Client settings
    """

    __slots__: List[str] = []

    def __init__(
        self,
        max_smiles_per_request: int = 1000,
        verify_tls: bool = True,
        max_retry: int = 2,
        retry_sleep: float = 10.0,
    ):
        """
        Args:
            max_smiles_per_request: maximum number of smiles per request
            verify_tls: verify the server's TLS certificate
            max_retry: maximum number of request retry before return an error
            retry_sleep: time to wait between two retry
        """
        super(SettingsCallback, self).__init__(
            max_smiles_per_request=max_smiles_per_request,
            verify_tls=verify_tls,
            max_retry=max_retry,
            retry_sleep=retry_sleep,
        )


class StatusCode(str, Enum):
    """SMILES retrosynthesis status"""

    NOT_SENT = "NOT_SENT"
    """The SMILES is still not sent to spaya"""

    SUBMITTED = "SUBMITTED"
    """The SMILES is in queue"""

    RUNNING = "RUNNING"
    """The SMILES is being processed"""

    DONE = "DONE"
    """The SMILES is finished"""

    INVALID_SMILES = "INVALID SMILES"
    """The SMILES is not valid"""

    ERROR = "ERROR"
    """Something went wrong"""

    KILLED = "KILLED"
    """The retrosynthesis was stopped for maintenance"""

    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    """Your retrosynthesis quota has been exceeded"""

    QUEUE_FULL = "QUEUE_FULL"
    """Too many SMILES are waiting in queue"""

    @property
    def is_finished(self) -> bool:
        """
        Returns:
            True if a SMILES is no longer being processed by the algorithm
        """
        return self in [
            StatusCode.DONE,
            StatusCode.ERROR,
            StatusCode.INVALID_SMILES,
            StatusCode.QUOTA_EXCEEDED,
        ]

    @property
    def can_be_retried(self) -> bool:
        """
        Returns:
            True if a SMILES is finished but can be submited again
        """
        return self in [StatusCode.ERROR, StatusCode.QUOTA_EXCEEDED]

    @property
    def need_retry(self) -> bool:
        """
        Returns:
            True if a SMILES is not started and must be submited again
        """
        return self in [StatusCode.QUEUE_FULL]


class Catalog(str, Enum):
    """Type of compounds"""

    BUILDING_BLOCK = "building block"

    SCREENING = "screening"


class RetrosynthesisParameters:
    """
    Parameters for a retrosynthesis
    """

    __slots__: List[str] = [
        "model",
        "max_depth",
        "max_nb_iterations",
        "early_stopping_score",
        "early_stopping_timeout",
        "intermediate_smiles",
        "imposed_structures",
        "forbidden_structures",
        "first_disconnections",
        "cc_providers",
        "cc_max_price_per_g",
        "cc_max_delivery_days",
        "cc_catalog",
        "cc_extra_compounds_smiles",
        "remove_chirality",
        "name_reactions_only",
        "name_reactions_exclude",
        "name_reactions_at_least",
        "filter_regio_issues",
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_nb_iterations: Optional[int] = None,
        early_stopping_score: Optional[float] = None,
        early_stopping_timeout: Optional[float] = None,
        intermediate_smiles: Optional[List[str]] = None,
        imposed_structures: Optional[List[str]] = None,
        forbidden_structures: Optional[List[str]] = None,
        first_disconnections: Optional[List[int]] = None,
        cc_providers: Optional[List[str]] = None,
        cc_max_price_per_g: Optional[float] = None,
        cc_max_delivery_days: Optional[int] = None,
        cc_catalog: Optional[List[Catalog]] = None,
        cc_extra_compounds_smiles: Optional[List[str]] = None,
        remove_chirality: Optional[bool] = None,
        name_reactions_only: Optional[List[str]] = None,
        name_reactions_exclude: Optional[List[str]] = None,
        name_reactions_at_least: Optional[List[str]] = None,
        filter_regio_issues: Optional[bool] = None,
    ):
        """
        Args:
            model: Spaya's retrosynthesis engine (model) version to use
            max_depth: Maximum route depth
            max_nb_iterations: Maximum number of steps
            early_stopping_score: Score threshold to stop the retrosynthesis of a SMILES
            early_stopping_timeout: Timeout to stop the retrosynthesis of a SMILES
             in minutes
            intermediate_smiles: Desired intermediate products (as a list of SMILES).
             This will force our models to only find routes which have at least one of
             these molecules as an intermediate product. This helps guide the process
             through reactions or molecules you already own or have expertise with for
             example.
            imposed_structures: Desired imposed substructures (as a list of SMARTS).
            forbidden_structures: Desired forbidden substructures (as a list of SMARTS).
            first_disconnections: Desired atoms indices to use as 1st disconnections (as a list of integer).
            cc_providers: List of desired commercial compounds providers.
             An empty list select them all
            cc_max_price_per_g: Maximum price per gramme for a commercial compound
            cc_max_delivery_days: Maximum delivery time in day.
             A null value select them all
            cc_catalog: Select the type of compounds (building block or screening).
             A null value select them all
            cc_extra_compounds_smiles: A list of smiles to add as commercial compounds
            remove_chirality : When True, remove the chirality from all inputs
            name_reactions_only: List of allowed name reactions
            name_reactions_exclude: List of excluded name reactions
            name_reactions_at_least: List of mandatory name reactions
            filter_regio_issues: When True, disables the regioselectivity
        """
        self.model = model
        self.max_depth = max_depth
        self.max_nb_iterations = max_nb_iterations
        self.early_stopping_score = early_stopping_score
        self.early_stopping_timeout = early_stopping_timeout
        self.intermediate_smiles = intermediate_smiles
        self.forbidden_structures = forbidden_structures
        self.imposed_structures = imposed_structures
        self.first_disconnections = first_disconnections
        self.cc_providers = cc_providers
        self.cc_max_price_per_g = cc_max_price_per_g
        self.cc_max_delivery_days = cc_max_delivery_days
        self.cc_catalog = cc_catalog
        self.cc_extra_compounds_smiles = cc_extra_compounds_smiles
        self.remove_chirality = remove_chirality
        self.name_reactions_only = name_reactions_only
        self.name_reactions_exclude = name_reactions_exclude
        self.name_reactions_at_least = name_reactions_at_least
        self.filter_regio_issues = filter_regio_issues

    def to_dict(self) -> Dict:
        """
        Returns:
            A dictionary for serialization
        """
        result: Dict[str, Union[int, float, str, List[str], List[int]]] = dict()
        if self.model is not None:
            result["model"] = self.model
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.max_nb_iterations is not None:
            result["max_nb_iterations"] = self.max_nb_iterations
        if self.early_stopping_score is not None:
            result["early_stopping_score"] = self.early_stopping_score
        if self.early_stopping_timeout is not None:
            result["early_stopping_timeout"] = self.early_stopping_timeout
        if self.intermediate_smiles is not None:
            result["intermediate_smiles"] = self.intermediate_smiles
        if self.imposed_structures is not None:
            result["imposed_structures"] = self.imposed_structures
        if self.forbidden_structures is not None:
            result["forbidden_structures"] = self.forbidden_structures
        if self.first_disconnections is not None:
            result["first_disconnections"] = self.first_disconnections
        if self.remove_chirality is not None:
            result["remove_chirality"] = self.remove_chirality
        if self.filter_regio_issues is not None:
            result["filter_regio_issues"] = self.filter_regio_issues
        self._to_dict_add_name_reaction(result)
        self._to_dict_add_cc(result)
        return result

    def _to_dict_add_name_reaction(
        self, data: Dict[str, Union[int, float, str, List[str], List[int]]]
    ):
        if self.name_reactions_only is not None:
            data["name_reactions_only"] = self.name_reactions_only
        if self.name_reactions_exclude is not None:
            data["name_reactions_exclude"] = self.name_reactions_exclude
        if self.name_reactions_at_least is not None:
            data["name_reactions_at_least"] = self.name_reactions_at_least

    def _to_dict_add_cc(
        self, data: Dict[str, Union[int, float, str, List[str], List[int]]]
    ):
        if self.cc_providers is not None:
            data["cc_providers"] = self.cc_providers
        if self.cc_max_price_per_g is not None:
            data["cc_max_price_per_g"] = self.cc_max_price_per_g
        if self.cc_max_delivery_days is not None:
            data["cc_max_delivery_days"] = self.cc_max_delivery_days
        if self.cc_catalog is not None:
            data["cc_catalog"] = [e.value for e in self.cc_catalog]
        if self.cc_extra_compounds_smiles is not None:
            data["cc_extra_compounds_smiles"] = self.cc_extra_compounds_smiles

    def equal_no_timing(self, other: "RetrosynthesisParameters") -> bool:
        """
        Args:
            other: other parameters to compare with

        Returns:
            True if all parameters others than timing restriction are equal
        """
        return (
            self.model == other.model
            and self.max_depth == other.max_depth
            and self.early_stopping_score == other.early_stopping_score
            and self.intermediate_smiles == other.intermediate_smiles
            and self.imposed_structures == other.imposed_structures
            and self.forbidden_structures == other.forbidden_structures
            and self.first_disconnections == other.first_disconnections
            and self.cc_providers == other.cc_providers
            and self.cc_max_price_per_g == other.cc_max_price_per_g
            and self.cc_max_delivery_days == other.cc_max_delivery_days
            and self.cc_catalog == other.cc_catalog
            and self.cc_extra_compounds_smiles == other.cc_extra_compounds_smiles
            and self.remove_chirality == other.remove_chirality
            and self.name_reactions_only == other.name_reactions_only
            and self.name_reactions_exclude == other.name_reactions_exclude
            and self.name_reactions_at_least == other.name_reactions_at_least
            and self.filter_regio_issues == other.filter_regio_issues
        )


class RetrosynthesisResult:
    """
    Retrosynthesis output
    """

    __slots__: List[str] = ["rscore", "nb_steps", "status", "progress"]

    def __init__(self) -> None:
        self.rscore: Optional[float] = None
        """The RScore is a metric provided by the Spaya algorithms,
        related to the probability of a disconnection and consequently
        to the confidence the algorithm has on this disconnection/route.
        The RScore is returned only when a retrosynthesis is DONE
        with one or more routes"""

        self.nb_steps: Optional[int] = None
        """The number of steps counts the depth of the tree which corresponds
        to the longest linear sequence of the considered route.
        The number of steps is returned only when a retrosynthesis is DONE
        with one or more routes"""

        self.status: StatusCode = StatusCode.NOT_SENT
        """SMILES retrosynthesis status"""

        self.progress: int = 0
        """SMILES retrosynthesis progress (as percentage)"""

    def from_dict(self, obj: Dict[str, Any]) -> "RetrosynthesisResult":
        """
        Update this object with new information

        Args:
            obj: New information
        """
        self.rscore = obj.get("rscore", None)
        self.nb_steps = obj.get("nb_steps", None)
        self.status = StatusCode(obj.get("status", StatusCode.INVALID_SMILES.value))
        self.progress = obj.get("progress", 0)
        return self

    @property
    def is_finished(self) -> bool:
        """
        Check the retrosynthesis status

        Returns:
            True if this result is final
        """
        return self.status.is_finished

    def __str__(self) -> str:
        return (
            f"score: {self.rscore}, nb_steps:{self.nb_steps}, "
            f"status:{self.status}, progress:{self.progress}"
        )

    def __repr__(self) -> str:
        return f"{type(self)}({id(self)}) <{str(self)}>"


class Status:
    """
    Current Spaya API status
    """

    __slots__: List[str] = ["queue_size"]

    def __init__(self, queue_size: Optional[int] = None) -> None:
        """

        Args:
            queue_size: number of job waiting to be processed
        """
        self.queue_size = queue_size

    def from_dict(self, data: Dict[str, Any]) -> "Status":
        """
        Complete this status with new value

        Args:
            data: a dictionnary to be parsed

        Returns:
            The current object
        """
        self.queue_size = data.get("queue_size", None)
        return self

    def __str__(self) -> str:
        return f"queue_size: {self.queue_size}"

    def __repr__(self) -> str:
        return f"{type(self)}({id(self)}) <{str(self)}>"


class Route:
    """
    A route computed by Spaya
    """

    __slots__: List[str] = ["rscore", "nb_steps", "tree"]

    def __init__(self) -> None:
        self.rscore: Optional[float] = None
        """The RScore is a metric provided by the Spaya algorithms,
        related to the probability of a disconnection and consequently
        to the confidence the algorithm has on this disconnection/route."""

        self.nb_steps: Optional[int] = None
        """The number of steps counts the depth of the tree which corresponds
        to the longest linear sequence of the considered route."""

        self.tree: Optional[Dict[str, dict]] = None
        """The route for a SMILES"""

    def from_dict(self, data: Dict[str, Any]) -> "Route":
        """
        Complete this status with new value

        Args:
            data: a dictionnary to be parsed

        Returns:
            The current object
        """
        self.rscore = data.get("rscore", None)
        self.nb_steps = data.get("nb_steps", None)
        self.tree = data.get("tree", None)
        return self

    def _get_leaf(self, data: Dict[str, dict]) -> Set[str]:
        """
        Recurse a dictionnary until the leaf are found
        Args:
            data: dictionnary to explore

        Returns:
            The leafs of data
        """
        result: Set[str] = set()
        for key, values in data.items():
            if values:
                result |= self._get_leaf(values)
            else:
                result.add(key)
        return result

    def tree_leaf(self) -> List[str]:
        """
        Returns:
            All the leaf of this Route
        """
        if self.tree:
            return list(self._get_leaf(self.tree))
        else:
            return list()

    def root_smiles(self) -> Optional[str]:
        """
        Returns:
            The root of the tree, the first SMILES in the tree
        """
        if self.tree:
            return list(self.tree.keys())[0]
        else:
            return None

    def __str__(self) -> str:
        return f"rscore: {self.rscore}, nb_steps:{self.nb_steps}, tree:{self.tree}"

    def __repr__(self) -> str:
        return f"{type(self)}({id(self)}) <{str(self)}>"


class CommercialCompound:
    """
    A route computed by Spaya
    """

    __slots__: List[str] = [
        "smiles",
        "provider",
        "url",
        "reference",
        "cas",
        "catalog",
        "packaging_g",
        "price_per_g",
        "delivery_date_min_day",
        "delivery_date_max_day",
        "purity",
        "chemical_name",
        "description",
    ]

    def __init__(self) -> None:
        self.smiles: str = ""
        """The molecule represented as a SMILES."""

        self.provider: str = ""
        """Name of the provider"""

        self.url: Optional[str] = None
        """Link to get the compounds"""

        self.reference: Optional[str] = None
        """Literature reference"""

        self.cas: Optional[str] = None
        """CAS number for this compound"""

        self.catalog: Optional[str] = None
        """The type of compounds (building block / screening / virtual)."""

        self.packaging_g: Optional[float] = None
        """Size of the packaging in gramme"""

        self.price_per_g: Optional[float] = None
        """Price per gramme"""

        self.delivery_date_min_day: Optional[int] = None
        """Minimum delivery time in day"""

        self.delivery_date_max_day: Optional[int] = None
        """Maximum delivery time in day"""

        self.purity: Optional[int] = None
        """Purity of the compounds"""

        self.chemical_name: Optional[str] = None
        """Name of the chemical"""

        self.description: Optional[str] = None
        """Extra informations from the provider"""

    def from_dict(self, data: Dict[str, Any]) -> "CommercialCompound":
        """
        Complete this status with new value

        Args:
            data: a dictionnary to be parsed

        Returns:
            The current object
        """
        self.smiles = data.get("smiles", "")
        self.provider = data.get("provider", "")
        self.url = data.get("url", None)
        self.reference = data.get("reference", None)
        self.cas = data.get("cas", None)
        self.catalog = data.get("catalog", None)
        self.packaging_g = data.get("packaging_g", None)
        self.price_per_g = data.get("price_per_g", None)
        self.delivery_date_min_day = data.get("delivery_date_min_day")
        self.delivery_date_max_day = data.get("delivery_date_max_day")
        self.purity = data.get("purity", None)
        self.chemical_name = data.get("chemical_name", None)
        self.description = data.get("description", None)
        return self

    def __str__(self) -> str:
        return (
            f"smiles: {self.smiles}, provider:{self.provider}, url:{self.url}, "
            f"reference: {self.reference}, cas:{self.cas}, catalog:{self.catalog}, "
            f"packaging_g: {self.packaging_g}, price_per_g:{self.price_per_g}, "
            f"delivery_date_min_day: {self.delivery_date_min_day}, "
            f"delivery_date_max_day: {self.delivery_date_max_day}, "
            f"purity: {self.purity}, chemical_name:{self.chemical_name}, "
            f"description: {self.description}"
        )

    def __repr__(self) -> str:
        return f"{type(self)}({id(self)}) <{str(self)}>"


class Cluster:
    """
    A cluster for a batch of smiles
    """

    __slots__: List[str] = ["key", "smiles", "mean_depths", "mean_max_score"]

    def __init__(self) -> None:
        self.key: str = ""
        """ The key of the clustering, a smiles common in all the routes"""

        self.smiles: List[str] = list()
        """Smiles in this cluster"""

        self.mean_depths: float = 0.0
        """The average depth of the common intermediate for the routes in the cluster"""

        self.mean_max_score: float = 0.0
        """The average of the max scores of the routes to create the SMILES in the
         cluster"""

    def from_dict(self, data: Dict[str, Any]) -> "Cluster":
        """
        Complete this status with new value

        Args:
            data: a dictionnary to be parsed

        Returns:
            The current object
        """
        self.key = data["key"]
        self.smiles = data["smiles"]
        self.mean_depths = data["mean_depths"]
        self.mean_max_score = data["mean_max_score"]
        return self

    def to_dict(self) -> Dict:
        """
        Returns:
            A dictionary for serialization
        """
        return {
            "key": self.key,
            "smiles": self.smiles,
            "mean_depths": self.mean_depths,
            "mean_max_score": self.mean_max_score,
        }

    def __str__(self) -> str:
        return (
            f"key: {self.key}, mean_depths:{self.mean_depths},"
            f" mean_max_score:{self.mean_max_score}"
            f" smiles:{self.smiles}"
        )

    def __repr__(self) -> str:
        return f"{type(self)}({id(self)}) <{str(self)}>"


class ClusteringResult:
    __slots__: List[str] = ["clusters", "status"]

    def __init__(self) -> None:
        self.clusters: List[Cluster] = list()
        """List of clusters. One cluster contains routes that are grouped by the
        smiles they lead to"""

        self.status: StatusCode = StatusCode.NOT_SENT
        """Clustering status"""

    def from_dict(self, data: Dict[str, Any]) -> "ClusteringResult":
        """
        Complete this status with new value

        Args:
            data: a dictionnary to be parsed

        Returns:
            The current object
        """
        self.clusters = [Cluster().from_dict(c) for c in data["clusters"]]
        self.status = StatusCode(data.get("status", StatusCode.INVALID_SMILES.value))
        return self
