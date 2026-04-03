import pytest
from hello_world.agent_bedrock import action_group_handler


def make_event(api_path, parameters=None):
    return {
        "actionGroup": "WeatherTools",
        "apiPath": api_path,
        "httpMethod": "GET",
        "parameters": [{"name": k, "value": v} for k, v in (parameters or {}).items()],
    }


def test_get_weather_returns_200():
    event = make_event("/get-weather", {"location": "London"})
    result = action_group_handler(event, None)
    assert result["response"]["httpStatusCode"] == 200
    assert "London" in result["response"]["responseBody"]["application/json"]["body"]


def test_calculator_returns_correct_result():
    event = make_event("/calculator", {"expression": "6 * 7"})
    result = action_group_handler(event, None)
    assert result["response"]["httpStatusCode"] == 200
    assert "42" in result["response"]["responseBody"]["application/json"]["body"]


def test_get_current_time_returns_200():
    event = make_event("/get-current-time")
    result = action_group_handler(event, None)
    assert result["response"]["httpStatusCode"] == 200


def test_unknown_api_path_returns_404():
    event = make_event("/unknown-tool")
    result = action_group_handler(event, None)
    assert result["response"]["httpStatusCode"] == 404


def test_response_includes_message_version():
    event = make_event("/get-current-time")
    result = action_group_handler(event, None)
    assert result["messageVersion"] == "1.0"


def test_response_echoes_action_group_and_path():
    event = make_event("/get-weather", {"location": "Paris"})
    result = action_group_handler(event, None)
    assert result["response"]["actionGroup"] == "WeatherTools"
    assert result["response"]["apiPath"] == "/get-weather"
