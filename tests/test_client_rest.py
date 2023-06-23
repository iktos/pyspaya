"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import pytest
import responses
import logging

import numpy as np
import time
from pandas import DataFrame, Series

from iktos.spaya import (
    BearerToken,
    SpayaClientREST,
    StatusCode,
    RetrosynthesisParameters,
    SettingsREST,
    Route,
    Catalog,
)


LOGGER = logging.getLogger(__name__)
MIN_UPDATE_PERIOD = 1


@responses.activate
def test_simple(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )
    smiles_list, check = nominal_scenario

    result = client.score_smiles(
        smiles=smiles_list,
        callback_progression=lambda x: LOGGER.info(f"progression:{x}"),
    )
    assert len(result) == 3

    check()


@responses.activate
def test_simple_dataframe(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )
    smiles_list, check = nominal_scenario
    smiles_column = "smiles_input"
    rscore_column = "spaya_score"
    nb_steps_column = "steps"
    df = DataFrame({smiles_column: smiles_list, "other_info": smiles_list})
    ref = DataFrame(
        {
            smiles_column: smiles_list,
            "other_info": smiles_list,
            rscore_column: [np.nan, 0.7, 0.8],
            nb_steps_column: [np.nan, float(3.0), float(42.0)],
        }
    )

    result = client.score_smiles(
        smiles=df,
        dataframe_smiles_column=smiles_column,
        dataframe_rscore_column=rscore_column,
        dataframe_nb_steps_column=nb_steps_column,
    )
    assert smiles_column in result.columns
    assert rscore_column in result.columns
    assert nb_steps_column in result.columns
    assert result.compare(ref).empty
    check()


@responses.activate
def test_wait_result(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )

    smiles_list, check = nominal_scenario

    client.start_retrosynthesis(smiles=smiles_list)
    client.wait_result()
    assert client.progression == 100

    # Check wait for empty result
    time.sleep(MIN_UPDATE_PERIOD)
    client.wait_result()

    check()


@responses.activate
def test_wait_result_timeout(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )

    smiles_list, check = nominal_scenario

    client.start_retrosynthesis(smiles=smiles_list)
    nb_timeout = 0
    while not client.is_retro_finished:
        client.wait_result(timeout=MIN_UPDATE_PERIOD + 0.00000001)
        nb_timeout += 1
    assert nb_timeout == 2
    assert client.progression == 100

    check()


@responses.activate
def test_check_progression(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )

    smiles_list, check = nominal_scenario

    client.start_retrosynthesis(smiles=smiles_list)
    while client.progression < 100.0:
        pass

    for smiles in smiles_list:
        assert client[smiles].is_finished

    check()


@responses.activate
def test_too_many_smiles(nominal_scenario_too_many_smiles, url, token):
    smiles_list, check, max_smiles_per_request = nominal_scenario_too_many_smiles
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(
            minimum_update_period=MIN_UPDATE_PERIOD,
            max_smiles_per_request=max_smiles_per_request,
        ),
    )

    result = client.score_smiles(
        smiles=smiles_list,
        callback_progression=lambda x: LOGGER.info(f"progression:{x}"),
    )
    assert len(result) == 5

    check()


@responses.activate
def test_consume(nominal_scenario, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )

    smiles_list, check = nominal_scenario

    client.start_retrosynthesis(smiles=smiles_list)
    while not client.is_empty:
        for smiles, result in client.consume():
            print(f"{smiles} {result}")

    check()


@responses.activate
def test_restart_error(error_scenario_fixed, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )
    smiles_list, check = error_scenario_fixed

    for percent_expected in [50, 3, 15, 100]:
        client.start_retrosynthesis(smiles=smiles_list)
        client.wait_result(
            callback_progression=lambda x: LOGGER.info(f"progression:{x}")
        )
        assert client.is_retro_finished
        assert client[smiles_list[0]].progress == percent_expected

    check()


def test_error_no_smiles(url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )
    client.start_retrosynthesis(smiles=[])


@responses.activate
def test_error_bad_parameters(url):
    with pytest.raises(ValueError):
        SpayaClientREST(
            url=url,
            authorization=BearerToken(token="token"),
            settings=SettingsREST(minimum_update_period=0.1),
        )


@responses.activate
def test_error_bad_response(url, url_batch):
    responses.add(responses.POST, url_batch, body="NOPE", status=200)

    with pytest.raises(ValueError):
        client_not_json = SpayaClientREST(
            url=url,
            authorization=BearerToken(token="token"),
            settings=SettingsREST(max_retry=0),
        )
        client_not_json.start_retrosynthesis("C")


@responses.activate
def test_error_404(url, url_batch):
    responses.add(responses.POST, url_batch, body="NOPE", status=404)
    with pytest.raises(ConnectionError):
        client_not_json = SpayaClientREST(
            url=url,
            authorization=BearerToken(token="token"),
            settings=SettingsREST(max_retry=0),
        )
        client_not_json.start_retrosynthesis("C")


@responses.activate
def test_status(url, url_status):
    responses.add(responses.GET, url_status, json={"queue_size": 3}, status=200)
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    status = client.get_status()
    LOGGER.debug(f"status:{repr(status)}")
    assert status.queue_size == 3


@responses.activate
def test_commercial_compounds_providers(url, url_cc_providers):
    providers = ["molport", "dealer"]
    responses.add(
        responses.GET,
        url_cc_providers,
        json={"providers": ["molport", "dealer"]},
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds_providers()
    assert result_cc_providers == providers


@responses.activate
def test_commercial_compounds_list(url, url_commercial_compounds):
    # Full test with a list as an input
    smiles_1 = "CC"
    provider = "molport"
    url_cc = "https://molport.com/somewhere"
    reference = "US43876538465"
    cas = "123123-45"
    catalog = "screening"
    packaging_g = 400
    price_per_g = 4.5
    delivery_date_min_day = 3
    delivery_date_max_day = 5
    purity = 99
    chemical_name = "Roger"
    description = "A nice guy"
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={
            "commercial_compounds": [
                {
                    "smiles": smiles_1,
                    "provider": provider,
                    "url": url_cc,
                    "reference": reference,
                    "cas": cas,
                    "catalog": catalog,
                    "packaging_g": packaging_g,
                    "price_per_g": price_per_g,
                    "delivery_date_min_day": delivery_date_min_day,
                    "delivery_date_max_day": delivery_date_max_day,
                    "purity": purity,
                    "chemical_name": chemical_name,
                    "description": description,
                },
                {
                    "smiles": "CCO",
                    "provider": "molport",
                    "url": "https://molport.com/somewhere",
                },
            ]
        },
        status=200,
    )

    smiles_2 = "CCCC"
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={
            "commercial_compounds": [
                {
                    "smiles": smiles_2,
                    "provider": "mcule",
                    "url": "https://mcule.com/somewhere",
                    "reference": "US43876538465",
                }
            ]
        },
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds([smiles_1, smiles_2])
    assert len(result_cc_providers) == 2
    first_results = result_cc_providers[smiles_1]
    assert len(first_results) == 2
    first_cc = first_results[0]
    LOGGER.debug(f"{first_cc}")
    assert first_cc.smiles == smiles_1
    assert first_cc.provider == provider
    assert first_cc.url == url_cc
    assert first_cc.reference == reference
    assert first_cc.cas == cas
    assert first_cc.catalog == catalog
    assert first_cc.packaging_g == packaging_g
    assert first_cc.price_per_g == price_per_g
    assert first_cc.delivery_date_min_day == delivery_date_min_day
    assert first_cc.delivery_date_max_day == delivery_date_max_day
    assert first_cc.purity == purity
    assert first_cc.chemical_name == chemical_name
    assert first_cc.description == description
    req = responses.calls[0].request
    assert req.body.decode("utf-8") == '{"smiles": "' + smiles_1 + '"}'

    assert len(result_cc_providers[smiles_2]) == 1
    LOGGER.debug(f"{repr(result_cc_providers[smiles_2][0])}")
    req = responses.calls[1].request
    assert req.body.decode("utf-8") == '{"smiles": "' + smiles_2 + '"}'


@responses.activate
def test_commercial_compounds_route(url, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = Route()
    smiles.tree = {
        "ABCDEF": {
            "ABCD": {"ABC": {"AB": {"A": {}, "B": {}}, "C": {}}, "D": {}},
            "EF": {"E": {}, "F": {}, "A": {}},  # Duplicate
        }
    }
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds(smiles)
    assert len(result_cc_providers) == 7
    for values in result_cc_providers.values():
        assert len(values) == 0
    assert len(responses.calls) == 7
    bodies = [responses.calls[i].request.body.decode("utf-8") for i in range(0, 7)]
    bodies.sort()
    for index, value in enumerate(["A", "ABCDEF", "B", "C", "D", "E", "F"]):
        assert bodies[index] == '{"smiles": "' + value + '"}'


@responses.activate
def test_commercial_compounds_route_no_tree(url):
    # Check if a route is sent and check no result
    smiles = Route()
    smiles.tree = None
    assert smiles.root_smiles() is None
    assert len(smiles.tree_leaf()) == 0
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds(smiles)
    assert len(result_cc_providers) == 0


@responses.activate
def test_commercial_compounds_route_just_root(url, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = Route()
    smiles.tree = {"A": {}}
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds(smiles)
    assert len(result_cc_providers) == 1
    assert responses.calls[0].request.body.decode("utf-8") == '{"smiles": "A"}'


@responses.activate
def test_commercial_compounds_series(url, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = Series(["A", "B"])
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_cc_providers = client.get_commercial_compounds(smiles)
    assert len(result_cc_providers) == 2
    assert responses.calls[0].request.body.decode("utf-8") == '{"smiles": "A"}'
    assert responses.calls[1].request.body.decode("utf-8") == '{"smiles": "B"}'


@responses.activate
def test_commercial_compounds_parameters_retro(url, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = "A"
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token="token"),
        parameters=RetrosynthesisParameters(
            cc_providers=["molport", "mcules"],
            cc_max_price_per_g=2,
            cc_max_delivery_days=3,
            cc_catalog=[Catalog.BUILDING_BLOCK],
        ),
    )
    result_cc_providers = client.get_commercial_compounds(smiles)
    assert len(result_cc_providers) == 1
    assert (
        responses.calls[0].request.body.decode("utf-8")
        == '{"provider": ["molport", "mcules"],'
        ' "catalog": ["building block"],'
        ' "price_per_g_max": 2,'
        ' "delivery_date_max_day": 3,'
        ' "smiles": "A"}'
    )


@responses.activate
def test_commercial_compounds_parameters_func(url, url_commercial_compounds):
    # Check if a route is sent and check no result
    smiles = "A"
    responses.add(
        responses.GET,
        url_commercial_compounds,
        json={"commercial_compounds": []},
        status=200,
    )
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token="token"),
        parameters=RetrosynthesisParameters(
            cc_providers=["other", "emolecules"],
            cc_max_price_per_g=45,
            cc_max_delivery_days=38,
            cc_catalog=[Catalog.BUILDING_BLOCK],
        ),
    )
    result_cc_providers = client.get_commercial_compounds(
        smiles=smiles,
        provider=["molport", "mcules"],
        catalog=[Catalog.SCREENING],
        packaging_g_min=2,
        packaging_g_max=3,
        price_per_g_min=4,
        price_per_g_max=5,
        delivery_date_max_day=6,
        delivery_included=True,
    )
    assert len(result_cc_providers) == 1
    assert (
        responses.calls[0].request.body.decode("utf-8")
        == '{"provider": ["molport", "mcules"],'
        ' "catalog": ["screening"],'
        ' "packaging_g_min": 2,'
        ' "packaging_g_max": 3,'
        ' "price_per_g_min": 4,'
        ' "price_per_g_max": 5,'
        ' "delivery_date_max_day": 6,'
        ' "delivery_included": true,'
        ' "smiles": "A"}'
    )


@responses.activate
def test_name_reactions(url, url_name_reaction):
    name_reactions = ["name1", "name2"]
    responses.add(
        responses.GET,
        url_name_reaction,
        json={"name_reactions": name_reactions},
        status=200,
    )
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_name_reactions = client.get_name_reactions(filter_name_reactions="name")
    assert result_name_reactions == name_reactions


@responses.activate
def test_route_nominal(url, url_routes):
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
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    result_routes = client.routes(smiles=["CCCC", "HHHHHH"], top_k_routes=2)

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
def test_route_nominal_dataframe(url, url_routes):
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
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    smiles_column = "smiles_input"
    df = DataFrame(
        {smiles_column: ["CCCC", "HHHHHH"], "other_info": ["CCCC", "HHHHHH"]}
    )
    result_routes = client.routes(
        smiles=df, top_k_routes=2, dataframe_smiles_column=smiles_column
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
def test_route_smiles_not_done(url, retro_response_add):
    smiles = "CCC"

    retro_response_add(
        smiles_result=[{"status": StatusCode.SUBMITTED.value, "smiles": smiles}]
    )

    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))
    client.start_retrosynthesis(smiles=smiles)
    with pytest.raises(ValueError):
        client.routes(smiles=smiles)


@responses.activate
def test_cluster_nominal_no_extra(cluster_nominal_scenario_no_extra, url, token):
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(minimum_update_period=MIN_UPDATE_PERIOD),
    )
    smiles_list, check = cluster_nominal_scenario_no_extra

    result = client.clustering(
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
def test_cluster_nominal_w_extra(cluster_nominal_scenario_w_extra, url, token):
    smiles_list, max_smiles_per_request, check = cluster_nominal_scenario_w_extra
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(
            minimum_update_period=MIN_UPDATE_PERIOD,
            max_smiles_per_request=max_smiles_per_request,
        ),
    )

    result = client.clustering(
        smiles=smiles_list[0],
        min_relative_size=0.1,
        max_cluster=42,
        max_coverage=0.9,
        alpha=0.5,
        extra_smiles=[
            (RetrosynthesisParameters(max_nb_iterations=1), smiles_list[i])
            for i in range(1, len(smiles_list))
        ],
    )
    assert result.status == StatusCode.DONE
    assert len(result.clusters) == 2
    LOGGER.debug(f"{result.clusters[0]}")
    LOGGER.debug(f"{repr(result.clusters[0])}")
    LOGGER.debug(f"{result.clusters[0].to_dict()}")
    assert result.clusters[0].key == "C"
    assert result.clusters[0].smiles == smiles_list[0]
    assert result.clusters[0].mean_depths == 0.1
    assert result.clusters[0].mean_max_score == 0.8
    LOGGER.debug(f"{result.clusters[1]}")
    LOGGER.debug(f"{repr(result.clusters[1])}")
    assert result.clusters[1].key == "O"
    assert result.clusters[1].smiles == smiles_list[1]
    assert result.clusters[1].mean_depths == 0.2
    assert result.clusters[1].mean_max_score == 0.9

    check()


@responses.activate
def test_cluster_error_parameters(cluster_nominal_scenario_w_extra, url, token):
    smiles_list, max_smiles_per_request, check = cluster_nominal_scenario_w_extra
    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(
            minimum_update_period=MIN_UPDATE_PERIOD,
            max_smiles_per_request=max_smiles_per_request,
        ),
    )

    with pytest.raises(ValueError):
        client.clustering(
            smiles=smiles_list[0],
            min_relative_size=0.1,
            max_cluster=42,
            max_coverage=0.9,
            alpha=0.5,
            extra_smiles=[
                (RetrosynthesisParameters(early_stopping_score=0.1), smiles_list[i])
                for i in range(1, len(smiles_list))
            ],
        )


@responses.activate
def test_quota(url, url_retrosynthesis_quota):
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))

    responses.add(
        responses.GET,
        url_retrosynthesis_quota,
        json={"retrosynthesis_left": 2},
        status=200,
    )
    quota = client.get_retrosynthesis_quota()
    assert quota == 2


@responses.activate
def test_quota_none(url, url_retrosynthesis_quota):
    client = SpayaClientREST(url=url, authorization=BearerToken(token="token"))

    responses.add(
        responses.GET,
        url_retrosynthesis_quota,
        json={"retrosynthesis_left": None},
        status=200,
    )
    quota_none = client.get_retrosynthesis_quota()
    assert quota_none is None


@responses.activate
def test_retry_ok(url, token, url_status):
    responses.add(responses.GET, url_status, status=404)
    responses.add(responses.GET, url_status, status=501)
    responses.add(responses.GET, url_status, json={"queue_size": 42}, status=200)

    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(max_retry=2, retry_sleep=0.2),
    )

    result = client.get_status()
    assert result.queue_size == 42


@responses.activate
def test_retry_nok(url, token, url_status):
    error_text = "NOPE"
    error_code = 504
    responses.add(responses.GET, url_status, status=404)
    responses.add(responses.GET, url_status, status=501)
    responses.add(responses.GET, url_status, status=error_code, body=error_text)

    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(max_retry=2, retry_sleep=0.2),
    )

    try:
        client.get_status()
        assert False
    except ConnectionError as ce:
        assert str(ce) == f"Bad request: <Response [{error_code}]> {error_text}"


def test_retry_bad_settings(url, token):
    # Max retry >= 0
    with pytest.raises(ValueError):
        SpayaClientREST(
            url=url,
            authorization=BearerToken(token=token),
            settings=SettingsREST(max_retry=-1, retry_sleep=0.2),
        )

    client = SpayaClientREST(
        url=url,
        authorization=BearerToken(token=token),
        settings=SettingsREST(max_retry=3, retry_sleep=0.2),
    )
    client._settings.max_retry = -1

    with pytest.raises(AttributeError):
        client.get_status()
