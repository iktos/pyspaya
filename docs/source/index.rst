Spaya API Client Package
========================

.. automodule:: iktos
   :members:
   :undoc-members:
   :show-inheritance:

-----------------

.. toctree::
   :maxdepth: 4
   :caption: Content:

   iktos.spaya
   iktos.spaya.model
   iktos.spaya.authorization


This package provides an easy way to score SMILES with the Spaya API.

To match every case, this API provides 3 different clients:

* **SpayaClientREST**: A synchronous client
* **SpayaClientAsync**: An asynchronous client with reconnection and helpers for the websocket
* **SpayaClientCallback**: An asynchronous client that allows you to get results using a Callback

You can also write your own client using the following documentation:

* `Spaya-API openapi <https://spaya.ai/retrosynthesis-api/redoc>`_
* `Spaya-API asyncapi <https://spaya.ai/retrosynthesis-api/static/asyncapi.html>`_


What is Spaya API ?
-------------------

Spaya API employs a data-driven AI approach to discover retrosynthetic routes.
An iterative exploration of all possible routes is performed until
commercially available starting materials are identified.

If you want to know more, please visit `Spaya API <https://spaya.ai/#api>`_
or `Spaya <https://spaya.ai/>`_
