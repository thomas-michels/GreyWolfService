"""
    Module for Validate event
"""
from .event_schema import EventSchema
from pydantic_core import ValidationError
import json
from app.core.configs import get_logger

_logger = get_logger(name=__name__)


def payload_conversor(raw_payload) -> EventSchema:
    """
    Function to convert raw payload in EventSchema

    :param payload: dict
    :return: EventSchema
    """
    try:
        if type(raw_payload) != dict:
            raw_payload = json.loads(raw_payload)

        return EventSchema(**raw_payload)

    except ValidationError as err:
        _logger.error(
            f"Error on validate payload in {', '.join([i['loc'][0] for i in err.errors()])} field(s)"
        )
