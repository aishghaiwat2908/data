"""Credential-free tests for the existing organization aggregator's distance flow."""

import json
import math
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).parents[1] / "src" / "saayam-org-aggregator"
sys.path.insert(0, str(SOURCE))
import distance
import helpers
import lambda_function


class DistanceTests(unittest.TestCase):
    def test_coordinate_forms_and_validation(self):
        self.assertEqual(distance.parse_coordinates("longitude:-121.9780,latitude:37.7799"), (37.7799, -121.978))
        self.assertEqual(distance.parse_coordinates("SRID=4326;POINT(-122.0841 37.4220)"), (37.422, -122.0841))
        self.assertEqual(distance.parse_coordinates({"coordinates": [-122.0, 37.0]}), (37.0, -122.0))
        self.assertEqual(distance.parse_coordinates({"latitude": 0, "longitude": 0}), (0.0, 0.0))
        for value in ({"latitude": 91, "longitude": 0}, {"latitude": float("nan"), "longitude": 0}, "POINT(foo bar)"):
            with self.subTest(value=value):
                self.assertIsNone(distance.parse_coordinates(value))

    def test_haversine_is_straight_line_and_zero_is_valid(self):
        origin = (37.7799, -121.9780)
        self.assertEqual(distance.haversine_miles(origin, origin), 0)
        self.assertTrue(math.isclose(distance.haversine_miles(origin, (37.4220, -122.0841)), 25.3, abs_tol=0.5))

    def test_request_location_has_priority_and_verifies_beneficiary(self):
        calls = []
        def rows(connection, query, params=()):
            calls.append((query, params))
            return [{"req_id": 9, "beneficiary_id": 4, "req_loc": "longitude:0,latitude:0"}]
        with patch.object(helpers, "fetch_rows", side_effect=rows):
            self.assertEqual(helpers.resolve_beneficiary_location(None, 9, 4), (0.0, 0.0))
            self.assertIsNone(helpers.resolve_beneficiary_location(None, 9, 5))
        self.assertEqual(len(calls), 2)  # No viewer/current/profile lookup.

    def test_current_location_then_profile_address_fallback(self):
        def rows(connection, query, params=()):
            if ".requests" in query:
                return [{"req_id": 9, "beneficiary_id": 4, "req_loc": "invalid"}]
            if ".user_locations" in query:
                return [{"curr_loc": "POINT(-122 37)"}]
            raise AssertionError("Profile must not be queried when current location exists")
        with patch.object(helpers, "fetch_rows", side_effect=rows):
            self.assertEqual(helpers.resolve_beneficiary_location(None, 9), (37.0, -122.0))

        addresses = []
        class Geocoder:
            def lookup(self, address):
                addresses.append(address)
                return (36.0, -121.0), "ok"
        def profile_rows(connection, query, params=()):
            if ".requests" in query:
                return [{"req_id": 9, "beneficiary_id": 4, "req_loc": None}]
            if ".user_locations" in query:
                return [{"curr_loc": None}]
            return [{"addr_ln1": "10 Main St", "city_name": "San Jose", "state_name": "California", "zip_code": "95101", "country_name": "United States"}]
        with patch.object(helpers, "fetch_rows", side_effect=profile_rows):
            self.assertEqual(helpers.resolve_beneficiary_location(None, 9, geocoder=Geocoder()), (36.0, -121.0))
        self.assertEqual(addresses, ["10 Main St, San Jose, California, 95101, United States"])

    def test_beneficiary_only_uses_own_latest_request_not_viewer(self):
        def rows(connection, query, params=()):
            self.assertEqual(params, (4,))
            self.assertIn("WHERE beneficiary_id = %s", query)
            return [{"beneficiary_id": 4, "req_loc": "POINT(-121 36)"}]
        with patch.object(helpers, "fetch_rows", side_effect=rows):
            self.assertEqual(helpers.resolve_beneficiary_location(None, beneficiary_id=4), (36.0, -121.0))

    def test_database_search_uses_bind_parameters_and_real_address_names(self):
        with patch.object(helpers, "fetch_rows", return_value=[{
            "org_name": "Example", "city_name": "San Jose", "street": "10 Main St",
            "state_name": "California", "country_name": "United States",
            "org_rating": 5, "is_collaborator": True,
        }]) as fetch:
            rows = helpers.get_orgs_from_db(None, "San Jose'", "Education")
        self.assertEqual(fetch.call_args.args[2], ("Education", "San Jose'"))
        self.assertIn("LEFT JOIN virginia_dev_saayam_rdbms.states", fetch.call_args.args[1])
        self.assertEqual(rows[0]["rating"], 5)
        self.assertTrue(rows[0]["Collaborator"])
        self.assertIn("10 Main St", distance.organization_address(rows[0]))

    def test_cached_geocode_and_provider_failures(self):
        calls = []
        service = distance.GeocodeService(
            provider=lambda address: calls.append(address) or {"latitude": 37, "longitude": -122},
            cache={},
        )
        self.assertEqual(service.lookup("10 Main St"), ((37.0, -122.0), "ok"))
        self.assertEqual(service.lookup("10 MAIN ST"), ((37.0, -122.0), "ok"))
        self.assertEqual(len(calls), 1)
        not_found = distance.GeocodeService(provider=lambda address: None, cache={})
        self.assertEqual(not_found.lookup("unknown"), (None, "not_found"))
        def fails(address):
            raise TimeoutError("geocoder timeout")
        self.assertEqual(distance.GeocodeService(provider=fails, cache={}).lookup("unknown"), (None, "error"))
        with patch.dict("os.environ", {"GEOCODER_LAMBDA_NAME": ""}):
            self.assertEqual(distance.GeocodeService(cache={}).lookup("unknown"), (None, "deferred"))

    def test_both_sources_unknown_online_and_zero(self):
        rows = [
            {"name": "Same", "db_or_ai": "db", "latitude": 0, "longitude": 0, "rating": 5, "Collaborator": True},
            {"name": "Online", "db_or_ai": "ai", "location": "online only"},
            {"name": "Missing", "db_or_ai": "ai"},
            {"name": "Address", "db_or_ai": "ai", "location": "San Jose, CA"},
        ]
        service = distance.GeocodeService(provider=lambda address: (0, 1), cache={})
        output = distance.add_distances(rows, (0, 0), service)
        self.assertEqual(output[0]["distance"], 0)
        self.assertEqual(output[0]["distance_status"], "ok")
        self.assertEqual(output[0]["rating"], 5)
        self.assertEqual(output[1]["distance_status"], "online")
        self.assertEqual(output[2]["distance_status"], "unknown_location")
        self.assertEqual(output[3]["distance_status"], "ok")
        self.assertGreater(output[3]["distance"], 0)
        self.assertIsNone(distance.add_distances(rows[:1], None, service)[0]["distance"])

    def test_handler_survives_genai_failure_and_sorts_null_last(self):
        class Connection:
            closed = False
            def close(self):
                self.closed = True
        connection = Connection()
        db_rows = [
            {"name": "Far", "db_or_ai": "db", "latitude": 0, "longitude": 1},
            {"name": "Missing", "db_or_ai": "ai"},
            {"name": "Near", "db_or_ai": "db", "latitude": 0, "longitude": 0},
        ]
        with patch.object(lambda_function, "connect_database", return_value=connection), patch.object(
            lambda_function, "resolve_beneficiary_location", return_value=(0, 0),
        ), patch.object(lambda_function, "get_orgs_from_db", return_value=db_rows), patch.object(
            lambda_function, "get_ai_orgs", side_effect=RuntimeError("GenAI unavailable"),
        ):
            response = lambda_function.lambda_handler({"body": json.dumps({
                "location": "San Jose", "category": "Education", "request_id": 9,
                "sort_by": "distance",
            })}, None)
        self.assertEqual(response["statusCode"], 200)
        output = json.loads(response["body"])
        self.assertEqual([row["name"] for row in output], ["Near", "Far", "Missing"])
        self.assertEqual(output[0]["distance"], 0)
        self.assertIsNone(output[-1]["distance"])
        self.assertTrue(connection.closed)

    def test_handler_keeps_genai_when_database_source_fails(self):
        class Connection:
            def close(self):
                pass
        with patch.object(lambda_function, "connect_database", return_value=Connection()), patch.object(
            lambda_function, "resolve_beneficiary_location", return_value=(0, 0),
        ), patch.object(lambda_function, "get_orgs_from_db", side_effect=RuntimeError("database unavailable")), patch.object(
            lambda_function, "get_ai_orgs", return_value=[{
                "name": "AI Organization", "db_or_ai": "ai", "latitude": 0, "longitude": 0,
            }],
        ):
            response = lambda_function.lambda_handler({"location": "San Jose", "category": "Education"}, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])[0]["distance"], 0)

    def test_handler_input_errors(self):
        self.assertEqual(lambda_function.lambda_handler({"body": "bad-json"}, None)["statusCode"], 400)
        self.assertEqual(lambda_function.lambda_handler({}, None)["statusCode"], 400)

    def test_json_serialization_keeps_records_with_missing_numeric_values(self):
        response = lambda_function._response(200, [{
            "name": "Example", "rating": float("nan"),
            "other": Decimal("NaN"), "distance": 0.0,
        }])
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"]), [{
            "name": "Example", "rating": None, "other": None, "distance": 0.0,
        }])


if __name__ == "__main__":
    unittest.main()
