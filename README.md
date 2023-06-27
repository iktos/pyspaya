# pyspaya

[![Documenation](https://img.shields.io/badge/Documentation-8A2BE2)](https://iktos.github.io/pyspaya/)
![PyPI](https://img.shields.io/pypi/v/pyspaya)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyspaya)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/iktos/pyspaya/release.yml)
![GitHub](https://img.shields.io/github/license/iktos/pyspaya)

[Description](#description)
- [Spaya API](#spaya-api)
- [Retrosynthesis parameters](#retrosynthesis-parameters)
- [Score](#score)

[Clients](#clients)
- [REST](#rest)
- [Async](#async)
- [Callback](#callback)

[Url and Authorization](#url-and-authorization)

## Description

### Spaya API

This package provides an easy way to score SMILES with the Spaya API.

This restrosynthesis score (RScore) is a metric provided by the Spaya algorithms,
related to the probability of a disconnection and consequently
to the confidence the algorithm has on this disconnection/route.

Spaya API employs a data-driven AI approach to discover retrosynthetic routes.
An iterative exploration of all possible routes is performed until
commercially available starting materials are identified.

Useful links:
- [Spaya Public](https://spaya.ai)
- [Spaya API](https://iktos.ai/spaya-api/)
- [Spaya REST documentation](https://spaya.ai/retrosynthesis-api/redoc)
- [Spaya Websocket documentation](https://spaya.ai/retrosynthesis-api/static/asyncapi.html)

### Score

Each scored SMILES return a RetrosynthesisResult:

```python
from typing import Dict
from iktos.spaya import BearerToken, SpayaClientREST, RetrosynthesisResult

# Create a client
client = SpayaClientREST(url="https://spaya.ai", authorization=BearerToken(token="YourToken"))

# Start a retrosynthesis and wait for the results
result: Dict[str, RetrosynthesisResult] = client.score_smiles(smiles=["c1ccn2nccc2c1"])

# Show results
retro_result: RetrosynthesisResult = result["c1ccn2nccc2c1"]
print(f"Progress:{retro_result.progress} Status:{retro_result.status}")
print(f"Score:{retro_result.rscore} Number of steps:{retro_result.nb_steps}")
```

## Clients

To match every case, this API provides 3 different clients:
- ```SpayaClientREST```: A synchronous client
- ```SpayaClientAsync```: An asynchronous client with reconnection and helpers for the websocket
- ```SpayaClientCallback```: An asynchronous client that allows you to get results using a Callback



### REST
```SpayaClientREST``` sends SMILES synchronously to be scored

Example using a list of SMILES:
```python
from iktos.spaya import BearerToken, SpayaClientREST

# Create client with authorization
client = SpayaClientREST(url="https://spaya.ai",
                         authorization=BearerToken("myT0ken"))

# Start a retrosynthesis and wait for the results
scored_smiles = client.score_smiles(smiles=["O=C1CCCCO1", "O=C1CCCNN1",])

# Show the results
for smiles, result in scored_smiles.items():
    print(f"{smiles}: {result.rscore} / {result.nb_steps}")
```

Example using a DataFrame:
```python
from iktos.spaya import BearerToken, SpayaClientREST
from pandas import DataFrame

# Create client with authorization
client = SpayaClientREST(url="https://spaya.ai/",
                         authorization=BearerToken("myT0ken"))

# Start a retrosynthesis and wait for the results with a dataframe
df = DataFrame({"input_smiles": ["O=C1CCCCO1", "O=C1CCCNN1",]})
df = client.score_smiles(
    smiles=df,
    dataframe_smiles_column="input_smiles",
    callback_progression=lambda p: print(f"progression:{p}")
)

# Show the results
print(f"result: {df}")
```

Example using a list of SMILES and consume function:
```python
from iktos.spaya import BearerToken, SpayaClientREST

# Create client with authorization
client = SpayaClientREST(url="https://spaya.ai/",
                         authorization=BearerToken("myT0ken"))

# Start a retrosynthesis
client.start_retrosynthesis(smiles=["O=C1CCCCO1", "O=C1CCCNN1",])

# Consume the results as soon as possible
while not client.is_empty:
    for smiles, result in client.consume():
        print(f"{smiles} {result}")
```


### Async
```SpayaClientAsync``` sends SMILES asynchronously to be scored

```python
import asyncio
from iktos.spaya import BearerToken, SpayaClientAsync

async def score():
    async with SpayaClientAsync(url="https://spaya.ai",
                                authorization=BearerToken("myT0ken")
    ) as client:
        # Start scoring SMILES
        await client.start_retrosynthesis(["O=C1CCCCO1", "O=C1CCCNN1",])

        # Wait and print scores as soon as received
        async for smiles, result in client.consume():
            print(f"{smiles}: {result.rscore} / {result.nb_steps}")

asyncio.run(score())
```


### Callback
```SpayaClientCallback``` calls a function as soon as a SMILES is scored

```python
import asyncio
from iktos.spaya import (
    BearerToken, SpayaClientCallback, RetrosynthesisResult
)

async def generator(client: SpayaClientCallback):
    # Generate and start scoring SMILES
    for smiles in ["O=C1CCCCO1", "O=C1CCCNN1"]:
        await client.start_retrosynthesis(smiles)

async def callback(smiles: str, result: RetrosynthesisResult):
    # Handle the results
    print(f"{smiles}: {result.rscore} / {result.nb_steps}")

async def generate_and_score():
    async with SpayaClientCallback(url="https://spaya.ai",
                                   authorization=BearerToken("myT0ken"),
                                   callback=callback) as client:
        # Generate SMILES
        await generator(client)

        # Block until the ends
        await client.wait_result()

asyncio.run(generate_and_score())
```

## Url and Authorization

To access the Spaya API, you need to generate a token from the `Spaya API` page in your
Spaya account

Unless specified, use the BearerToken class


#### Bearer token
The server is protected by a token
```python
from iktos.spaya import BearerToken, SpayaClientREST
SpayaClientREST(url="https://spaya.ai", authorization=BearerToken(token="itb3cy0s..."))
```

#### Custom Bearer token
The server is protected by a token and it uses a custom header
```python
from iktos.spaya import CustomBearerToken, SpayaClientREST
SpayaClientREST(
    url="https://spaya.ai",
    authorization=CustomBearerToken(token="itb3cy0s...", header_key="X-Iktos-Authorization")
)
```
