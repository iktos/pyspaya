"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import pytest
import logging

import numpy as np
from pandas import DataFrame, concat

from iktos.spaya import (
    BearerToken,
    RetrosynthesisParameters,
    StatusCode,
    Catalog,
)
from iktos.spaya.spaya_client import SpayaClient

from tests.utils import (
    check_smiles_client,
    create_retro_response_result,
    create_retro_result,
)

LOGGER = logging.getLogger(__name__)


def test_create_entry():
    url = "https://localhost:8000"
    parameters = RetrosynthesisParameters(
        max_depth=4,
        max_nb_iterations=20,
        early_stopping_score=0.2,
        early_stopping_timeout=40.2,
        intermediate_smiles=["Middleman"],
        cc_providers=["molport"],
        cc_max_price_per_g=1.2,
        cc_max_delivery_days=14,
        cc_catalog=[Catalog.BUILDING_BLOCK],
        cc_extra_compounds_smiles=["CO", "CCO"],
        remove_chirality=False,
        name_reactions_only=["reaction1", "reaction2"],
        name_reactions_exclude=["reaction3", "reaction4"],
        name_reactions_at_least=["reaction5", "reaction6"],
    )

    client = SpayaClient(
        url=url, authorization=BearerToken(token="token"), parameters=parameters
    )

    assert client.url == url
    assert client.parameters == parameters

    smiles_list = ["C", "CC", "CCC", "CCCC", "CCCCC"]
    entry = client._create_entry(smiles=smiles_list)
    assert entry == {
        "max_depth": 4,
        "max_nb_iterations": 20,
        "early_stopping_score": 0.2,
        "early_stopping_timeout": 40.2,
        "intermediate_smiles": ["Middleman"],
        "cc_providers": ["molport"],
        "cc_max_price_per_g": 1.2,
        "cc_max_delivery_days": 14,
        "cc_catalog": ["building block"],
        "cc_extra_compounds_smiles": ["CO", "CCO"],
        "batch_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "remove_chirality": False,
        "name_reactions_only": ["reaction1", "reaction2"],
        "name_reactions_exclude": ["reaction3", "reaction4"],
        "name_reactions_at_least": ["reaction5", "reaction6"],
    }


def test_progression():
    client = SpayaClient(url="", authorization=BearerToken(token="token"))
    assert client.progression == 0.0

    smiles_list = ["XD", "C", "CC", "CCC", "CCCC", "CCCCC"]

    assert client.progression == 0

    #############################
    # Step 1
    smiles_invalid = create_retro_result(
        smiles=smiles_list[0], progression=100, status=StatusCode.INVALID_SMILES
    )
    smiles_error = create_retro_result(
        smiles=smiles_list[1], progression=100, status=StatusCode.ERROR
    )
    smiles_submited = create_retro_result(
        smiles=smiles_list[2], progression=0, status=StatusCode.SUBMITTED
    )
    smiles_running = create_retro_result(
        smiles=smiles_list[3], progression=50, status=StatusCode.RUNNING
    )
    smiles_queue_full = create_retro_result(
        smiles=smiles_list[5], progression=0, status=StatusCode.QUEUE_FULL
    )
    response_json_1 = create_retro_response_result(
        status=StatusCode.INVALID_SMILES,
        smiles=[
            smiles_invalid,
            smiles_error,
            smiles_submited,
            smiles_running,
            smiles_queue_full,
        ],
    )
    client._update_result_batch(response_json=response_json_1)

    assert len(client.unfinished_smiles) == 3
    assert smiles_list[0] in client
    assert smiles_list[1] in client
    assert smiles_list[2] in client
    assert smiles_list[3] in client
    assert smiles_list[5] in client
    assert client.is_retro_finished is False
    assert client.progression == 50.0

    check_smiles_client(
        client=client,
        smiles=smiles_list[0],
        progression=100,
        status=StatusCode.INVALID_SMILES,
        is_finished=True,
        rscore=None,
        nb_steps=None,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[1],
        progression=100,
        status=StatusCode.ERROR,
        is_finished=True,
        rscore=None,
        nb_steps=None,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[2],
        progression=0,
        status=StatusCode.SUBMITTED,
        is_finished=False,
        rscore=None,
        nb_steps=None,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[3],
        progression=50,
        status=StatusCode.RUNNING,
        is_finished=False,
        rscore=None,
        nb_steps=None,
    )

    client.remove(smiles=smiles_list[0])
    assert smiles_list[0] not in client
    with pytest.raises(KeyError):
        client.remove(smiles=smiles_list[0])

    result_1 = client.pop_finished(smiles=smiles_list[1])
    assert result_1.status == StatusCode.ERROR
    with pytest.raises(KeyError):
        client.pop_finished(smiles=smiles_list[1])

    assert client.pop_finished(smiles=smiles_list[3]) is None

    #############################
    # Step 2
    smiles_running_2 = create_retro_result(
        smiles=smiles_list[2], progression=50, status=StatusCode.RUNNING
    )
    smiles_done_2 = create_retro_result(
        smiles=smiles_list[3],
        progression=100,
        status=StatusCode.DONE,
        nb_steps=3,
        rscore=0.8,
    )
    smiles_submited_2 = create_retro_result(
        smiles=smiles_list[4], progression=0, status=StatusCode.SUBMITTED
    )
    smiles_full_2 = create_retro_result(
        smiles=smiles_list[5], progression=0, status=StatusCode.SUBMITTED
    )
    response_json_2 = create_retro_response_result(
        status=StatusCode.SUBMITTED,
        smiles=[smiles_running_2, smiles_done_2, smiles_submited_2, smiles_full_2],
    )
    client._update_result_batch(response_json=response_json_2)

    assert client.is_retro_finished is False
    assert client.progression == 37.5

    check_smiles_client(
        client=client,
        smiles=smiles_list[2],
        progression=50,
        status=StatusCode.RUNNING,
        is_finished=False,
        rscore=None,
        nb_steps=None,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[3],
        progression=100,
        status=StatusCode.DONE,
        is_finished=True,
        rscore=0.8,
        nb_steps=3,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[4],
        progression=0,
        status=StatusCode.SUBMITTED,
        is_finished=False,
        rscore=None,
        nb_steps=None,
    )

    check_smiles_client(
        client=client,
        smiles=smiles_list[5],
        progression=0,
        status=StatusCode.SUBMITTED,
        is_finished=False,
        rscore=None,
        nb_steps=None,
    )

    client.remove(smiles=smiles_list[2])
    assert smiles_list[2] not in client
    with pytest.raises(KeyError):
        client.remove(smiles=smiles_list[2])

    #############################
    # Step 3
    all_done = list()
    for smiles in smiles_list[3:]:
        all_done.append(
            create_retro_result(
                smiles=smiles,
                progression=100,
                status=StatusCode.DONE,
                nb_steps=3,
                rscore=0.8,
            )
        )
    response_json_3 = create_retro_response_result(
        status=StatusCode.DONE, smiles=all_done
    )
    client._update_result_batch(response_json=response_json_3)
    assert client.is_retro_finished is True
    assert client.progression == 100.0
    for smiles in smiles_list[3:]:
        check_smiles_client(
            client=client,
            smiles=smiles,
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=0.8,
            nb_steps=3,
        )

    result_1 = client.pop_finished(smiles=smiles_list[3])
    assert result_1.status == StatusCode.DONE
    with pytest.raises(KeyError):
        client.pop_finished(smiles=smiles_list[3])

    LOGGER.error(f"{client._smiles_done}")

    client.pop_finished(smiles=smiles_list[4])
    client.pop_finished(smiles=smiles_list[5])
    assert client.is_empty


def test_dataframe_multi_pop():
    client = SpayaClient(url="", authorization=BearerToken(token="token"))

    duplicated_smiles = "CC"
    smiles_list = ["C", duplicated_smiles, "CCC"]
    extra_smiles = "CCCC"
    smiles_column = "smiles_input"
    rscore_column = "smiles_rscore"
    nb_steps_column = "smiles_nb_steps"
    df = DataFrame({smiles_column: smiles_list, "other_info": smiles_list})

    smiles_done_0 = create_retro_result(
        smiles=smiles_list[0],
        progression=100,
        status=StatusCode.DONE,
        rscore=0.1,
        nb_steps=2,
    )
    smiles_running_1 = create_retro_result(
        smiles=smiles_list[1], progression=50, status=StatusCode.RUNNING
    )
    smiles_submited_2 = create_retro_result(
        smiles=smiles_list[2], progression=0, status=StatusCode.SUBMITTED
    )
    response_json = create_retro_response_result(
        status=StatusCode.SUBMITTED,
        smiles=[smiles_done_0, smiles_running_1, smiles_submited_2],
    )
    client._update_result_batch(response_json=response_json)

    df_duplicated = DataFrame(
        {smiles_column: [duplicated_smiles], "other_info": [duplicated_smiles]}
    )
    df = concat([df, df_duplicated], ignore_index=True)

    df_extra_smiles = DataFrame(
        {smiles_column: [extra_smiles], "other_info": [extra_smiles]}
    )
    df = concat([df, df_extra_smiles], ignore_index=True)

    df = client.pop_finished_to_dataframe(
        df=df,
        smiles_column=smiles_column,
        rscore_column=rscore_column,
        nb_steps_column=nb_steps_column,
    )

    smiles_done_1 = create_retro_result(
        smiles=smiles_list[1],
        progression=100,
        status=StatusCode.DONE,
        rscore=0.2,
        nb_steps=3,
    )
    smiles_done_2 = create_retro_result(
        smiles=smiles_list[2], progression=100, status=StatusCode.DONE
    )
    response_json_2 = create_retro_response_result(
        status=StatusCode.DONE, smiles=[smiles_done_1, smiles_done_2]
    )
    client._update_result_batch(response_json=response_json_2)

    df = client.pop_finished_to_dataframe(
        df=df,
        smiles_column=smiles_column,
        rscore_column=rscore_column,
        nb_steps_column=nb_steps_column,
    )

    ref = DataFrame(
        {
            smiles_column: smiles_list + [duplicated_smiles, extra_smiles],
            "other_info": smiles_list + [duplicated_smiles, extra_smiles],
            rscore_column: [0.1, 0.2, np.nan, 0.2, np.nan],
            nb_steps_column: [float(2.0), float(3.0), np.nan, float(3.0), np.nan],
        }
    )

    assert smiles_column in df.columns
    assert rscore_column in df.columns
    assert nb_steps_column in df.columns
    assert df.compare(ref).empty

    with pytest.raises(KeyError):
        assert client[duplicated_smiles].is_finished
