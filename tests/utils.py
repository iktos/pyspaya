"""
Copyright (C) Iktos - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, List, Callable, Dict, Coroutine, Any, TypeVar, Union, Tuple
from multiprocessing import Process, Event

import websockets

from iktos.spaya.spaya_client import SpayaClient
from iktos.spaya import (
    StatusCode,
    RetrosynthesisParameters,
    RetrosynthesisResult,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

_WEBSOCKET = TypeVar("_WEBSOCKET", bound=websockets.WebSocketCommonProtocol)


class WebsocketServerTest:
    __slots__ = [
        "_websocket_path",
        "_port",
        "_scenario",
        "_to_yield",
        "_server",
        "_task_list",
        "_timeout_receiving",
    ]

    def __init__(
        self,
        websocket_path: str,
        port: int,
        timeout_receiving: int = 1,
        scenario: Optional[
            Callable[[_WEBSOCKET, str], Coroutine[Any, Any, None]]
        ] = None,
    ):
        self._websocket_path = websocket_path
        self._port = port
        self._timeout_receiving = timeout_receiving
        if scenario is None:
            self._scenario: Callable[
                [_WEBSOCKET, str], Coroutine[Any, Any, None]
            ] = self._scenario_yield
        else:
            self._scenario = scenario
        self._to_yield: Dict[str, List] = dict()
        self._server: Optional[websockets.WebSocketServer] = None
        self._task_list: Dict[str, asyncio.Task] = dict()

    def add_to_yield_on_received(
        self,
        received: str,
        elem: Union[Dict, websockets.WebSocketException],
        sleep_time: float = 0,
    ) -> None:
        if received not in self._to_yield:
            self._to_yield[received] = list()
        self._to_yield[received].append((sleep_time, elem))

    async def _scenario_yield(self, websocket: _WEBSOCKET, received: str) -> None:
        scenario_for_received = self._to_yield.get(received, [])
        try:
            if not scenario_for_received:
                raise websockets.ConnectionClosedError(code=1007, reason="invalid data")
            while len(scenario_for_received) > 0:
                sleep_time, elem = scenario_for_received[0]
                await asyncio.sleep(sleep_time)
                scenario_for_received.pop(0)
                if isinstance(elem, websockets.WebSocketException):
                    raise elem
                else:
                    response = json.dumps(elem)
                    await websocket.send(response)
        except websockets.WebSocketException as wse:
            if isinstance(wse, websockets.ConnectionClosed):
                await websocket.close(code=wse.code, reason=wse.reason)
            else:
                await websocket.close()

    async def _received_task(self, websocket: _WEBSOCKET):
        try:
            while True:
                received = await websocket.recv()
                if received in self._task_list and self._task_list[received].done():
                    del self._task_list[received]
                if received not in self._task_list:
                    self._task_list[received] = asyncio.create_task(
                        self._scenario(websocket, received)
                    )
                await asyncio.sleep(0)
        except websockets.WebSocketException:
            pass

    async def _execute_scenario(self, websocket: _WEBSOCKET, path):
        if path != self._websocket_path:
            raise ValueError(f"Wrong path {path} != {self._websocket_path}")
        task_received = asyncio.create_task(self._received_task(websocket))
        last_info = datetime.now()
        timeout = timedelta(seconds=self._timeout_receiving)
        while True:
            still_task = False
            for task in self._task_list.values():
                still_task |= not task.done()
            if still_task:
                last_info = datetime.now()
            elif datetime.now() - last_info > timeout:
                break
            await asyncio.sleep(0.2)
        while not task_received.done():
            await asyncio.sleep(0.1)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        self._server = await websockets.serve(self._execute_scenario, port=self._port)

    async def wait_all_scenario_done(self):
        has_message = True
        while has_message:
            await asyncio.sleep(0.2)
            has_message = True in [bool(msg) for msg in self._to_yield.values()]

    async def close(self):
        self._server.close()
        await self._server.wait_closed()

    def check(self) -> bool:
        missing_message = False
        for received, msg_list in self._to_yield.items():
            if msg_list:
                missing_message = True
                LOGGER.error(f"{received} as missing message: {msg_list}")
        return not missing_message


class WebsocketServerRunner:
    __slots__ = ["_server", "_process", "_started_flag"]

    def __init__(self, server: WebsocketServerTest):
        self._server = server
        self._process: Optional[Process] = None
        self._started_flag = Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _start_server(server: WebsocketServerTest, started_flag):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(server.start())
        started_flag.set()
        LOGGER.info(f"Waiting: {server}")
        loop.run_until_complete(server.wait_all_scenario_done())
        LOGGER.info(f"All scenarios done: {server}")
        loop.run_until_complete(server.close())
        server.check()
        LOGGER.info("PROCESS END")

    def start(self):
        self._process = Process(
            target=self._start_server,
            kwargs={"server": self._server, "started_flag": self._started_flag},
        )
        LOGGER.info("Server starting")
        self._process.start()
        end = datetime.now() + timedelta(seconds=10)
        while not self._started_flag.is_set() and end > datetime.now():
            time.sleep(0.1)
        LOGGER.info("Server started")

    def close(self):
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                raise RuntimeError("The process was still alive")
            LOGGER.info(f"terminated: {self._process.exitcode}")


def check_retro_result(
    result: RetrosynthesisResult,
    progression: int,
    status: Optional[StatusCode],
    is_finished: bool,
    rscore: Optional[float],
    nb_steps: Optional[int],
) -> None:
    assert result.status == status
    assert result.progress == progression
    assert result.is_finished is is_finished
    assert result.rscore == rscore
    assert result.nb_steps == nb_steps


def check_smiles_client(
    client: SpayaClient,
    smiles: str,
    progression: int,
    status: StatusCode,
    is_finished: bool,
    rscore: Optional[float],
    nb_steps: Optional[int],
) -> None:
    check_retro_result(
        result=client[smiles],
        progression=progression,
        status=status,
        is_finished=is_finished,
        rscore=rscore,
        nb_steps=nb_steps,
    )


def create_retro_result(
    smiles: str,
    progression: int,
    status: StatusCode,
    rscore: Optional[float] = None,
    nb_steps: Optional[int] = None,
) -> Dict:
    return {
        "smiles": smiles,
        "status": status.value,
        "rscore": rscore,
        "nb_steps": nb_steps,
        "progress": progression,
    }


def create_retro_response_result(status: StatusCode, smiles: List[Dict]) -> Dict:
    return {"status": status.value, "smiles": smiles}


def create_retro_request_dict(
    smiles_list: List[str], parameters: Optional[RetrosynthesisParameters] = None
) -> Dict:
    if parameters is None:
        parameters = RetrosynthesisParameters()

    dict_ref = parameters.to_dict()
    dict_ref["batch_smiles"] = smiles_list
    return dict_ref


def create_retro_request_str(
    smiles_list: List[str], parameters: Optional[RetrosynthesisParameters] = None
) -> str:
    dict_ref = create_retro_request_dict(smiles_list=smiles_list, parameters=parameters)
    return json.dumps(dict_ref)


def create_cluster_response_result(
    status: StatusCode,
    keys: Optional[List[str]] = None,
    smiles: Optional[List[List[str]]] = None,
    mean_depths: Optional[List[float]] = None,
    mean_max_score: Optional[List[float]] = None,
) -> Dict:
    if keys is None:
        keys = []
    if smiles is None:
        smiles = [[]]
    if mean_depths is None:
        mean_depths = [0.1]
    if mean_max_score is None:
        mean_max_score = [0.9]
    return {
        "status": status.value,
        "clusters": [
            {
                "key": key,
                "smiles": smiles[index],
                "mean_depths": mean_depths[index],
                "mean_max_score": mean_max_score[index],
            }
            for index, key in enumerate(keys)
        ],
    }


def create_cluster_request_dict(
    batches: List[Tuple[RetrosynthesisParameters, List[str]]],
    min_relative_size: Optional[float] = None,
    max_cluster: Optional[int] = None,
    max_coverage: Optional[float] = None,
    alpha: Optional[float] = None,
    min_route_rscore: Optional[float] = None,
) -> Dict:
    result: Dict = {"batches": []}
    for parameters, smiles_list in batches:
        dict_ref = parameters.to_dict()
        dict_ref["batch_smiles"] = smiles_list
        result["batches"].append(dict_ref)
    if min_relative_size is not None:
        result["min_relative_size"] = min_relative_size
    if max_cluster is not None:
        result["max_cluster"] = max_cluster
    if max_coverage is not None:
        result["max_coverage"] = max_coverage
    if alpha is not None:
        result["alpha"] = alpha
    if min_route_rscore is not None:
        result["min_route_rscore"] = min_route_rscore
    return result


def create_cluster_request_str(
    batches: List[Tuple[RetrosynthesisParameters, List[str]]],
    min_relative_size: Optional[float] = None,
    max_cluster: Optional[int] = None,
    max_coverage: Optional[float] = None,
    alpha: Optional[float] = None,
    min_route_rscore: Optional[float] = None,
) -> str:
    dict_ref = create_cluster_request_dict(
        batches=batches,
        min_relative_size=min_relative_size,
        max_cluster=max_cluster,
        max_coverage=max_coverage,
        alpha=alpha,
        min_route_rscore=min_route_rscore,
    )
    return json.dumps(dict_ref)
