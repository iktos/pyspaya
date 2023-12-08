"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""
from typing import Optional, List, Dict, Callable, Tuple
import logging

import pytest
import responses
import websockets

from iktos.spaya.spaya_client import SpayaClient
from iktos.spaya import (
    RetrosynthesisParameters,
    RetrosynthesisResult,
    StatusCode,
)

from tests.utils import (
    WebsocketServerTest,
    WebsocketServerRunner,
    create_retro_result,
    create_retro_request_str,
    check_retro_result,
    create_retro_response_result,
    create_cluster_response_result,
    create_cluster_request_str,
)


LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def port() -> int:
    return 8000


@pytest.fixture(scope="function")
def url(port) -> str:
    return f"http://localhost:{port}"


@pytest.fixture(scope="function")
def root() -> str:
    return SpayaClient.ROOT


@pytest.fixture(scope="function")
def version() -> str:
    return SpayaClient.VERSION


@pytest.fixture(scope="function")
def url_root(url, root) -> str:
    return f"{url}{root}"


@pytest.fixture(scope="function")
def url_root_version(url_root, version) -> str:
    return f"{url_root}{version}"


@pytest.fixture(scope="function")
def url_batch(url_root_version) -> str:
    return f"{url_root_version}/batch-smiles"


@pytest.fixture(scope="function")
def url_status(url_root_version) -> str:
    return f"{url_root_version}/status"


@pytest.fixture(scope="function")
def url_retrosynthesis_quota(url_root_version) -> str:
    return f"{url_root_version}/quota"


@pytest.fixture(scope="function")
def url_cc_providers(url_root_version) -> str:
    return f"{url_root_version}/commercial-compounds-providers"


@pytest.fixture(scope="function")
def url_commercial_compounds(url_root_version) -> str:
    return f"{url_root_version}/commercial-compounds"


@pytest.fixture(scope="function")
def url_name_reaction(url_root_version) -> str:
    return f"{url_root_version}/name-reactions"


@pytest.fixture(scope="function")
def url_routes(url_root_version) -> str:
    return f"{url_root_version}/routes"


@pytest.fixture(scope="function")
def url_clustering(url_root_version) -> str:
    return f"{url_root_version}/clustering"


@pytest.fixture(scope="function")
def url_websocket_path(root, version) -> str:
    return f"{root}/ws{version}/batch-smiles"


@pytest.fixture(scope="function")
def url_websocket(url_websocket_path, port) -> str:
    path = f"ws://localhost:{port}{url_websocket_path}"
    return path


@pytest.fixture(scope="function")
def token() -> str:
    return "aliudsh78falskj345dfnasdf34bxj938764"


@pytest.fixture(scope="function")
def retro_response_add(url_batch):
    def retro_response_add_impl(
        smiles_result: List[Dict], status: Optional[StatusCode] = None
    ):
        if status is None:
            # We don't use it any way
            status = StatusCode.RUNNING
        responses.add(
            responses.POST,
            url_batch,
            json=create_retro_response_result(status=status, smiles=smiles_result),
            status=200,
        )

    return retro_response_add_impl


@pytest.fixture(scope="function")
def retro_request_check(url_batch, token):
    def retro_request_check_impl(
        smiles_list: List[str],
        parameters: Optional[RetrosynthesisParameters] = None,
        num_call: Optional[int] = None,
    ):
        text_ref = create_retro_request_str(
            smiles_list=smiles_list, parameters=parameters
        )

        if num_call is None:
            requ = responses.calls[-1].request
        else:
            requ = responses.calls[num_call].request
        assert requ.url == url_batch
        assert requ.headers["Authorization"] == f"Bearer {token}"
        assert requ.body.decode("utf-8") == text_ref

    return retro_request_check_impl


@pytest.fixture(scope="function")
def nominal_scenario(
    retro_request_check, retro_response_add
) -> Tuple[List[str], Callable[[], None]]:
    smiles_list = ["C", "CC", "CCC"]

    smiles_invalid_1 = create_retro_result(
        smiles=smiles_list[0], progression=100, status=StatusCode.INVALID_SMILES
    )
    smiles_submited_1 = create_retro_result(
        smiles=smiles_list[1], progression=0, status=StatusCode.SUBMITTED
    )
    smiles_running_1 = create_retro_result(
        smiles=smiles_list[2], progression=50, status=StatusCode.RUNNING
    )
    retro_response_add(
        smiles_result=[smiles_invalid_1, smiles_submited_1, smiles_running_1]
    )

    smiles_running_2 = create_retro_result(
        smiles=smiles_list[1], progression=50, status=StatusCode.RUNNING
    )
    smiles_done_2 = create_retro_result(
        smiles=smiles_list[2],
        progression=100,
        status=StatusCode.DONE,
        rscore=0.8,
        nb_steps=42,
    )
    retro_response_add(smiles_result=[smiles_running_2, smiles_done_2])

    smiles_done_3 = create_retro_result(
        smiles=smiles_list[1],
        progression=100,
        status=StatusCode.DONE,
        rscore=0.7,
        nb_steps=3,
    )
    retro_response_add(smiles_result=[smiles_done_3])

    def check_nominal_scenario():
        retro_request_check(smiles_list=smiles_list, num_call=0)
        retro_request_check(smiles_list=smiles_list[1:], num_call=1)
        retro_request_check(smiles_list=smiles_list[1:2], num_call=2)

    return smiles_list, check_nominal_scenario


@pytest.fixture(scope="function")
def nominal_scenario_too_many_smiles(
    retro_request_check, retro_response_add
) -> Tuple[List[str], Callable[[], None], int]:
    max_smiles_per_request = 2
    smiles_list = ["C", "CC", "CCC", "CCCC", "CCCCC"]

    smiles_0_invalid_1 = create_retro_result(
        smiles=smiles_list[0], progression=100, status=StatusCode.INVALID_SMILES
    )
    smiles_1_submited_1 = create_retro_result(
        smiles=smiles_list[1], progression=0, status=StatusCode.SUBMITTED
    )
    smiles_2_running_1 = create_retro_result(
        smiles=smiles_list[2], progression=50, status=StatusCode.RUNNING
    )
    smiles_3_submited_1 = create_retro_result(
        smiles=smiles_list[3], progression=0, status=StatusCode.SUBMITTED
    )
    smiles_4_submited_1 = create_retro_result(
        smiles=smiles_list[4], progression=0, status=StatusCode.SUBMITTED
    )
    retro_response_add(smiles_result=[smiles_0_invalid_1, smiles_1_submited_1])
    retro_response_add(smiles_result=[smiles_2_running_1, smiles_3_submited_1])
    retro_response_add(smiles_result=[smiles_4_submited_1])

    smiles_1_submited_2 = create_retro_result(
        smiles=smiles_list[1], progression=100, status=StatusCode.DONE
    )
    smiles_2_running_2 = create_retro_result(
        smiles=smiles_list[2], progression=75, status=StatusCode.RUNNING
    )
    retro_response_add(smiles_result=[smiles_1_submited_2, smiles_2_running_2])

    smiles_2_running_3 = create_retro_result(
        smiles=smiles_list[2], progression=100, status=StatusCode.DONE
    )
    smiles_3_submited_3 = create_retro_result(
        smiles=smiles_list[3], progression=100, status=StatusCode.DONE
    )
    retro_response_add(smiles_result=[smiles_2_running_3, smiles_3_submited_3])

    smiles_4_submited_4 = create_retro_result(
        smiles=smiles_list[4], progression=100, status=StatusCode.DONE
    )
    retro_response_add(smiles_result=[smiles_4_submited_4])

    def check_nominal_scenario():
        retro_request_check(smiles_list=smiles_list[0:2], num_call=0)
        retro_request_check(smiles_list=smiles_list[2:4], num_call=1)
        retro_request_check(smiles_list=smiles_list[4:], num_call=2)

        retro_request_check(smiles_list=smiles_list[1:3], num_call=3)
        retro_request_check(smiles_list=smiles_list[2:4], num_call=4)
        retro_request_check(smiles_list=smiles_list[4:], num_call=5)

    return smiles_list, check_nominal_scenario, max_smiles_per_request


@pytest.fixture(scope="function")
def nominal_scenario_one_smile(
    retro_request_check, retro_response_add
) -> Tuple[List[str], Callable[[], None]]:
    smiles = "C"

    smiles_submited_1 = create_retro_result(
        smiles=smiles, progression=0, status=StatusCode.SUBMITTED
    )
    retro_response_add(smiles_result=[smiles_submited_1])

    smiles_running_2 = create_retro_result(
        smiles=smiles, progression=40, status=StatusCode.RUNNING
    )
    retro_response_add(smiles_result=[smiles_running_2])

    smiles_done_3 = create_retro_result(
        smiles=smiles, progression=100, status=StatusCode.DONE
    )
    retro_response_add(smiles_result=[smiles_done_3])

    def check_nominal_scenario_one_smile():
        for idx in range(0, 3):
            retro_request_check(smiles_list=[smiles], num_call=idx)

    return [smiles], check_nominal_scenario_one_smile


@pytest.fixture(scope="function")
def error_scenario_fixed(
    retro_request_check, retro_response_add
) -> Tuple[List[str], Callable[[], None]]:
    smiles_list = ["C"]

    smiles_submited_1 = create_retro_result(
        smiles=smiles_list[0], progression=0, status=StatusCode.SUBMITTED
    )
    retro_response_add(smiles_result=[smiles_submited_1])

    smiles_error_1 = create_retro_result(
        smiles=smiles_list[0], progression=50, status=StatusCode.ERROR
    )
    retro_response_add(smiles_result=[smiles_error_1])

    smiles_submited_2 = create_retro_result(
        smiles=smiles_list[0], progression=0, status=StatusCode.SUBMITTED
    )
    retro_response_add(smiles_result=[smiles_submited_2])

    smiles_error_2 = create_retro_result(
        smiles=smiles_list[0], progression=3, status=StatusCode.ERROR
    )
    retro_response_add(smiles_result=[smiles_error_2])

    smiles_running_3 = create_retro_result(
        smiles=smiles_list[0], progression=15, status=StatusCode.RUNNING
    )
    retro_response_add(smiles_result=[smiles_running_3])

    smiles_error_3 = create_retro_result(
        smiles=smiles_list[0], progression=15, status=StatusCode.ERROR
    )
    retro_response_add(smiles_result=[smiles_error_3])

    smiles_done_4 = create_retro_result(
        smiles=smiles_list[0], progression=100, status=StatusCode.DONE
    )
    retro_response_add(smiles_result=[smiles_done_4])

    def check_nominal_scenario():
        for i in range(0, 7):
            retro_request_check(smiles_list=smiles_list, num_call=i)

    return smiles_list, check_nominal_scenario


@pytest.fixture(scope="function")
async def websocket_nominal(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["XD", "C", "CC"]

    smiles_0_1_received = create_retro_request_str(smiles_list=smiles_list[0:2])
    smiles_0_1 = [
        (
            0,
            create_retro_result(
                smiles=smiles_list[0], progression=100, status=StatusCode.INVALID_SMILES
            ),
        ),
        (
            0,
            create_retro_result(
                smiles=smiles_list[1], progression=0, status=StatusCode.SUBMITTED
            ),
        ),
        (
            5,
            create_retro_result(
                smiles=smiles_list[1], progression=100, status=StatusCode.ERROR
            ),
        ),
    ]
    for sleep_time, elem in smiles_0_1:
        ws.add_to_yield_on_received(
            received=smiles_0_1_received, elem=elem, sleep_time=sleep_time
        )

    smiles_2_received = create_retro_request_str(smiles_list=smiles_list[2:3])
    smiles_2 = [
        (
            0.1,
            create_retro_result(
                smiles=smiles_list[2], progression=0, status=StatusCode.SUBMITTED
            ),
        ),
        (
            2,
            create_retro_result(
                smiles=smiles_list[2], progression=30, status=StatusCode.RUNNING
            ),
        ),
        (
            2,
            create_retro_result(
                smiles=smiles_list[2], progression=60, status=StatusCode.RUNNING
            ),
        ),
        (
            1,
            create_retro_result(
                smiles=smiles_list[2],
                progression=100,
                status=StatusCode.DONE,
                rscore=0.4,
                nb_steps=42,
            ),
        ),
    ]
    for sleep_time, elem in smiles_2:
        ws.add_to_yield_on_received(
            received=smiles_2_received, elem=elem, sleep_time=sleep_time
        )

    def check_nominal_scenario(to_check: Dict[str, RetrosynthesisResult]):
        check_retro_result(
            result=to_check[smiles_list[0]],
            progression=100,
            status=StatusCode.INVALID_SMILES,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )
        check_retro_result(
            result=to_check[smiles_list[1]],
            progression=100,
            status=StatusCode.ERROR,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )
        check_retro_result(
            result=to_check[smiles_list[2]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=0.4,
            nb_steps=42,
        )

    async with WebsocketServerRunner(server=ws):
        yield [smiles_list[0:2], smiles_list[2]], check_nominal_scenario


@pytest.fixture(scope="function")
async def websocket_stop_and_continue(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["C"]

    smiles_0_received = create_retro_request_str(smiles_list=smiles_list)
    smiles_0 = [
        (
            0.1,
            create_retro_result(
                smiles=smiles_list[0], progression=0, status=StatusCode.SUBMITTED
            ),
        ),
        (
            0.3,
            websockets.ConnectionClosedOK(
                websockets.frames.Close(code=1001, reason="goinn away"), None
            ),
        ),
        (
            0.2,
            create_retro_result(
                smiles=smiles_list[0],
                progression=100,
                status=StatusCode.DONE,
                rscore=0.4,
                nb_steps=42,
            ),
        ),
    ]
    for sleep_time, elem in smiles_0:
        ws.add_to_yield_on_received(
            received=smiles_0_received, elem=elem, sleep_time=sleep_time
        )

    def check_nominal_scenario(to_check: Dict[str, RetrosynthesisResult]):
        check_retro_result(
            result=to_check[smiles_list[0]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=0.4,
            nb_steps=42,
        )

    async with WebsocketServerRunner(server=ws):
        yield smiles_list, check_nominal_scenario


@pytest.fixture(scope="function")
async def websocket_nominal_already_done(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["C", "CC"]
    smiles_list_previous_sent = ["CCC"]

    smiles_0_1_received = create_retro_request_str(smiles_list=smiles_list)
    smiles_0_1 = [
        (
            0,
            create_retro_result(
                smiles=smiles_list[0], progression=100, status=StatusCode.DONE
            ),
        ),
        (
            0,
            # Smiles outside list to simulate a reception from a previous request
            create_retro_result(
                smiles=smiles_list_previous_sent[0],
                progression=100,
                status=StatusCode.DONE,
            ),
        ),
        (
            0,
            create_retro_result(
                smiles=smiles_list[1], progression=100, status=StatusCode.DONE
            ),
        ),
    ]
    for sleep_time, elem in smiles_0_1:
        ws.add_to_yield_on_received(
            received=smiles_0_1_received, elem=elem, sleep_time=sleep_time
        )

    def check_nominal_scenario(to_check: Dict[str, RetrosynthesisResult]):
        check_retro_result(
            result=to_check[smiles_list[0]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )
        check_retro_result(
            result=to_check[smiles_list[1]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )
        check_retro_result(
            result=to_check[smiles_list_previous_sent[0]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )

    async with WebsocketServerRunner(server=ws):
        yield [smiles_list], check_nominal_scenario


@pytest.fixture(scope="function")
async def websocket_nominal_queue_full(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["C", "CC"]

    smiles_0_1_received = create_retro_request_str(smiles_list=smiles_list)
    smiles_0_1 = [
        (
            0,
            create_retro_result(
                smiles=smiles_list[0], progression=100, status=StatusCode.DONE
            ),
        ),
        (
            0,
            create_retro_result(
                smiles=smiles_list[1], progression=0, status=StatusCode.QUEUE_FULL
            ),
        ),
    ]
    for sleep_time, elem in smiles_0_1:
        ws.add_to_yield_on_received(
            received=smiles_0_1_received, elem=elem, sleep_time=sleep_time
        )

    smiles_1_received = create_retro_request_str(smiles_list=[smiles_list[1]])
    elem_1 = create_retro_result(
        smiles=smiles_list[1], progression=100, status=StatusCode.DONE
    )
    ws.add_to_yield_on_received(received=smiles_1_received, elem=elem_1, sleep_time=0.1)

    def check_nominal_scenario(to_check: Dict[str, RetrosynthesisResult]):
        check_retro_result(
            result=to_check[smiles_list[0]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )
        check_retro_result(
            result=to_check[smiles_list[1]],
            progression=100,
            status=StatusCode.DONE,
            is_finished=True,
            rscore=None,
            nb_steps=None,
        )

    async with WebsocketServerRunner(server=ws):
        yield [smiles_list], check_nominal_scenario


@pytest.fixture(scope="function")
async def error_websocket_at_start(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["C"]

    smiles_0_received = create_retro_request_str(smiles_list=smiles_list)
    ws.add_to_yield_on_received(
        received=smiles_0_received,
        elem=websockets.ConnectionClosedError(
            websockets.frames.Close(code=1002, reason="protocol error"), None
        ),
        sleep_time=0.1,
    )

    async with WebsocketServerRunner(server=ws):
        yield smiles_list


@pytest.fixture(scope="function")
async def websocket_error_after_start(url_websocket_path, port):
    ws = WebsocketServerTest(websocket_path=url_websocket_path, port=port)
    smiles_list = ["C"]

    smiles_0_received = create_retro_request_str(smiles_list=smiles_list)
    ws.add_to_yield_on_received(
        received=smiles_0_received,
        elem=create_retro_result(
            smiles=smiles_list[0],
            progression=0,
            status=StatusCode.SUBMITTED,
            rscore=None,
            nb_steps=None,
        ),
        sleep_time=0,
    )
    ws.add_to_yield_on_received(
        received=smiles_0_received,
        elem=websockets.ConnectionClosedError(
            websockets.frames.Close(code=1002, reason="protocol error"), None
        ),
        sleep_time=3,
    )
    async with WebsocketServerRunner(server=ws):
        yield smiles_list


@pytest.fixture(scope="function")
def cluster_response_add(url_clustering):
    def cluster_response_add_impl(
        status: StatusCode,
        keys: Optional[List[str]] = None,
        smiles: Optional[List[List[str]]] = None,
        mean_depths: Optional[List[float]] = None,
        mean_max_score: Optional[List[float]] = None,
    ):
        if status is None:
            # We don't use it any way
            status = StatusCode.RUNNING
        responses.add(
            responses.POST,
            url_clustering,
            json=create_cluster_response_result(
                status=status,
                keys=keys,
                smiles=smiles,
                mean_depths=mean_depths,
                mean_max_score=mean_max_score,
            ),
            status=200,
        )

    return cluster_response_add_impl


@pytest.fixture(scope="function")
def cluster_request_check(url_clustering, token):
    def cluster_request_check_impl(
        batches: List[Tuple[RetrosynthesisParameters, List[str]]],
        min_relative_size: Optional[float] = None,
        max_cluster: Optional[int] = None,
        max_coverage: Optional[float] = None,
        alpha: Optional[float] = None,
        min_route_rscore: Optional[float] = None,
        num_call: Optional[int] = None,
    ):
        text_ref = create_cluster_request_str(
            batches=batches,
            min_relative_size=min_relative_size,
            max_cluster=max_cluster,
            max_coverage=max_coverage,
            alpha=alpha,
            min_route_rscore=min_route_rscore,
        )

        if num_call is None:
            requ = responses.calls[-1].request
        else:
            requ = responses.calls[num_call].request
        assert requ.url == url_clustering
        assert requ.headers["Authorization"] == f"Bearer {token}"
        assert requ.body.decode("utf-8") == text_ref

    return cluster_request_check_impl


@pytest.fixture(scope="function")
def cluster_nominal_scenario_no_extra(
    cluster_request_check, cluster_response_add
) -> Tuple[List[str], Callable[[], None]]:
    smiles_list = ["C", "CC", "CCC", "O", "OO", "OOO"]

    cluster_response_add(status=StatusCode.SUBMITTED)
    cluster_response_add(status=StatusCode.RUNNING)
    cluster_response_add(
        status=StatusCode.DONE,
        keys=["C", "O"],
        smiles=[smiles_list[0:3], smiles_list[3:]],
        mean_depths=[0.1, 0.2],
        mean_max_score=[0.8, 0.9],
    )

    def check_nominal_scenario():
        for i in range(0, 3):
            cluster_request_check(
                batches=[(RetrosynthesisParameters(), smiles_list)],
                min_relative_size=0.1,
                max_cluster=42,
                max_coverage=0.9,
                alpha=0.5,
                min_route_rscore=0.6,
                num_call=i,
            )

    return smiles_list, check_nominal_scenario


@pytest.fixture(scope="function")
def cluster_nominal_scenario_w_extra(
    cluster_request_check, cluster_response_add
) -> Tuple[List[List[str]], int, Callable[[], None]]:
    max_smiles_per_request = 3
    smiles_list: List[List[str]] = [["C", "CC", "CCC"], ["O", "OO", "OOO", "OOOO"], []]

    cluster_response_add(status=StatusCode.SUBMITTED)
    cluster_response_add(status=StatusCode.RUNNING)
    cluster_response_add(
        status=StatusCode.DONE,
        keys=["C", "O"],
        smiles=[smiles_list[0], smiles_list[1]],
        mean_depths=[0.1, 0.2],
        mean_max_score=[0.8, 0.9],
    )

    def check_nominal_scenario() -> None:
        for i in range(0, 3):
            cluster_request_check(
                batches=[
                    (RetrosynthesisParameters(), smiles_list[0]),
                    (
                        RetrosynthesisParameters(max_nb_iterations=1),
                        smiles_list[1][0:3],
                    ),
                    (RetrosynthesisParameters(max_nb_iterations=1), smiles_list[1][3:]),
                ],
                min_relative_size=0.1,
                max_cluster=42,
                max_coverage=0.9,
                alpha=0.5,
                num_call=i,
            )

    return smiles_list, max_smiles_per_request, check_nominal_scenario
