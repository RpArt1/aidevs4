"""
Tool definitions (JSON Schema) and handler implementations for the findhim agent.

TOOLS: list of tool schemas the LLM sees.
HANDLERS: maps tool names to Python callables that do the real work.
"""
from __future__ import annotations

import math
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AIDEVS_API_KEY", "")
VERIFY_URL = os.getenv("AIDEVS_VERIFY_URL", "AIDEVS_VERIFY_URL_ENV")
HUB_BASE = "AIDEVS_HUB_BASE_ENV"

# ── Tool schemas (Chat Completions format) ───────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "find_suspect_nearest_power_plant",
            "description": (
                "Given a list of suspects, fetch all power plants and each suspect's "
                "sightings, then compute haversine distances to find the suspect whose "
                "sighting is closest to any power plant. Returns the winning suspect "
                "with the matched power plant code, name, and distance in km."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "suspects": {
                        "type": "array",
                        "description": "List of suspect objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "First name of the suspect",
                                },
                                "surname": {
                                    "type": "string",
                                    "description": "Surname of the suspect",
                                },
                                "birthYear": {
                                    "type": "integer",
                                    "description": "Year of birth, e.g. 1987",
                                },
                            },
                            "required": ["name", "surname", "birthYear"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["suspects"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_access_level",
            "description": (
                "Get the access level of a person. Requires their birth year "
                "as an integer (e.g. 1987)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "First name of the person",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Surname / last name of the person",
                    },
                    "birthYear": {
                        "type": "integer",
                        "description": "Year of birth as integer, e.g. 1987",
                    },
                },
                "required": ["name", "surname", "birthYear"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the final answer to the verification endpoint. "
                "Call this once you have identified the suspect, their access level, "
                "and the power plant code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "First name of the identified suspect",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Surname of the identified suspect",
                    },
                    "accessLevel": {
                        "type": "integer",
                        "description": "Access level returned by get_access_level",
                    },
                    "powerPlant": {
                        "type": "string",
                        "description": "Power plant code, e.g. PWR1234PL",
                    },
                },
                "required": ["name", "surname", "accessLevel", "powerPlant"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ── Handler implementations ──────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def _geocode(city: str) -> tuple[float, float]:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": f"{city}, Poland", "format": "json", "limit": 1},
        headers={"User-Agent": "findhim-agent/1.0"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Geocoding failed for city: {city}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def find_suspect_nearest_power_plant(suspects: list[dict]) -> dict:
    plants_url = f"{HUB_BASE}/data/{API_KEY}/findhim_locations.json"
    plants_resp = requests.get(plants_url, timeout=30)
    plants_resp.raise_for_status()
    raw_plants = plants_resp.json().get("power_plants", {})

    plants: list[dict] = []
    for city, info in raw_plants.items():
        lat, lon = _geocode(city)
        plants.append({
            "city": city,
            "lat": lat,
            "lon": lon,
            "code": info["code"],
        })

    best: dict | None = None
    best_dist = math.inf

    for suspect in suspects:
        payload = {
            "apikey": API_KEY,
            "name": suspect["name"],
            "surname": suspect["surname"],
        }
        loc_resp = requests.post(
            f"{HUB_BASE}/api/location", json=payload, timeout=30
        )
        loc_resp.raise_for_status()
        sightings = loc_resp.json()

        for sighting in sightings:
            s_lat = float(sighting["latitude"])
            s_lon = float(sighting["longitude"])
            for plant in plants:
                dist = _haversine(s_lat, s_lon, plant["lat"], plant["lon"])
                if dist < best_dist:
                    best_dist = dist
                    best = {
                        "name": suspect["name"],
                        "surname": suspect["surname"],
                        "birthYear": int(suspect["birthYear"]),
                        "powerPlantCode": plant["code"],
                        "powerPlantName": plant["city"],
                        "distance_km": round(dist, 3),
                    }

    if best is None:
        return {"error": "No sightings found for any suspect"}
    return best


def get_access_level(name: str, surname: str, birthYear: int) -> dict:
    birthYear = int(birthYear)
    payload = {
        "apikey": API_KEY,
        "name": name,
        "surname": surname,
        "birthYear": birthYear,
    }
    resp = requests.post(f"{HUB_BASE}/api/accesslevel", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def submit_answer(
    name: str, surname: str, accessLevel: int, powerPlant: str
) -> dict:
    payload = {
        "apikey": API_KEY,
        "task": "findhim",
        "answer": {
            "name": name,
            "surname": surname,
            "accessLevel": int(accessLevel),
            "powerPlant": powerPlant,
        },
    }
    resp = requests.post(VERIFY_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


HANDLERS: dict[str, Any] = {
    "find_suspect_nearest_power_plant": find_suspect_nearest_power_plant,
    "get_access_level": get_access_level,
    "submit_answer": submit_answer,
}
