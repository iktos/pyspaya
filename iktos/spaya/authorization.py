#  Copyright (C) Iktos 2020 - All Rights Reserved.
#  Unauthorized copying of this file, via any medium is strictly prohibited.
#  Proprietary and confidential.

import abc
from typing import Dict


class Authorization(metaclass=abc.ABCMeta):
    """
    Abstract class for login
    """

    __slots__ = ()

    def __init__(self):
        pass

    @abc.abstractmethod
    def headers(self) -> Dict:
        """
        Returns:
            Request header with authorization information
        """


class CustomBearerToken(Authorization):
    """
    Custom Bearer Token Authorization
    """

    __slots__ = ["_token", "_header_key"]

    def __init__(self, token: str, header_key: str = "X-Iktos-Authorization"):
        super().__init__()
        self._token = token
        self._header_key = header_key

    def headers(self) -> Dict:
        return {self._header_key: f"Bearer {self._token}"}


class BearerToken(CustomBearerToken):
    """
    Simple Bearer Token Authorization
    """

    def __init__(self, token: str):
        super().__init__(token=token, header_key="Authorization")
