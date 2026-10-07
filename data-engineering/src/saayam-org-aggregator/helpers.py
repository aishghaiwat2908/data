"""Database and GenAI sources for the existing organization aggregator.

Connections are opened per invocation, not while importing the module. This
also keeps the distance calculations testable without live AWS credentials.
"""

import json


SCHEMA = "virginia_dev_saayam_rdbms"
GEN_AI_LAMBDA = "More_Org_GenAI_Py_v3126"


def connect_database():
    import pg8000
    from aws_lambda_powertools.utilities import parameters

    credentials = json.loads(parameters.get_parameter(
        "/dev/saayam/db/Virginia/Analytics/user", decrypt=True, max_age=3600,
    ))
    return pg8000.connect(
        host=credentials["HOST"], user=credentials["USERNAME"],
        password=credentials["PASSWORD"], database=credentials["DATABASE NAME"],
        port=credentials["PORT"], ssl_context=True,
    )


def fetch_rows(connection, query, parameters=()):
    cursor = connection.cursor()
    try:
        cursor.execute(query, parameters)
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def get_orgs_from_db(connection, location, category):
    """Keep the existing mission/city search and retain all organization fields."""
    rows = fetch_rows(connection, f"""
        SELECT o.*, s.state_name, c.country_name, c.country_code
        FROM {SCHEMA}.organizations AS o
        LEFT JOIN {SCHEMA}.states AS s ON s.state_id = o.state_id
        LEFT JOIN {SCHEMA}.countries AS c ON c.country_id = s.country_id
        WHERE o.mission = %s AND o.city_name = %s
    """, (category, location))
    for row in rows:
        row["name"] = row.get("org_name")
        row["contact"] = row.get("phone")
        row["location"] = row.get("city_name")
        row["size"] = row.get("org_size")
        row["rating"] = row.get("org_rating")
        row["Collaborator"] = row.get("is_collaborator")
        row["Org-type"] = row.get("org_type")
        row["db_or_ai"] = "db"
    return rows


def get_ai_orgs(subject, description, location):
    import boto3

    response = boto3.client("lambda").invoke(
        FunctionName=GEN_AI_LAMBDA, InvocationType="RequestResponse",
        Payload=json.dumps({
            "subject": subject, "description": description, "location": location,
        }).encode(),
    )
    payload = json.loads(response["Payload"].read())
    if payload.get("statusCode") != 200:
        raise RuntimeError("GenAI organization source failed")
    body = payload.get("body", {})
    if isinstance(body, str):
        body = json.loads(body)
    rows = body.get("organizations", [])
    if not isinstance(rows, list):
        raise ValueError("GenAI organization response is not a list")
    result = []
    for original in rows:
        if not isinstance(original, dict):
            continue
        row = original.copy()
        row["name"] = row.get("name") or row.get("organization_name")
        row["db_or_ai"] = "ai"
        result.append(row)
    return result


def merge_organizations(db_organizations, genai_organizations):
    """Do not discard rating/collaborator or source-specific address fields."""
    return list(db_organizations) + list(genai_organizations)


def resolve_beneficiary_location(connection, request_id=None, beneficiary_id=None, geocoder=None):
    """Request coordinates, then beneficiary current location, then profile address.

    Never read viewer coordinates. If request_id is supplied, its beneficiary_id
    is authoritative; a mismatched caller-supplied ID is rejected.
    """
    from distance import parse_coordinates

    if request_id is not None:
        requests = fetch_rows(connection, f"""
            SELECT req_id, beneficiary_id, req_loc
            FROM {SCHEMA}.requests WHERE req_id = %s
        """, (request_id,))
        if not requests:
            return None
        request = requests[0]
        actual_id = request.get("beneficiary_id")
        if beneficiary_id is not None and str(beneficiary_id) != str(actual_id):
            return None
        beneficiary_id = actual_id
        coordinates = parse_coordinates(request.get("req_loc"))
        if coordinates is not None:
            return coordinates
    elif beneficiary_id is not None:
        # A request-details caller should send req_id. For beneficiary-only
        # callers, use that beneficiary's latest request, never the viewer's.
        requests = fetch_rows(connection, f"""
            SELECT req_id, beneficiary_id, req_loc
            FROM {SCHEMA}.requests WHERE beneficiary_id = %s
            ORDER BY submission_date DESC LIMIT 1
        """, (beneficiary_id,))
        if requests:
            coordinates = parse_coordinates(requests[0].get("req_loc"))
            if coordinates is not None:
                return coordinates

    if beneficiary_id is None:
        return None

    current = fetch_rows(connection, f"""
        SELECT curr_loc FROM {SCHEMA}.user_locations
        WHERE user_id = %s ORDER BY last_updated_at DESC LIMIT 1
    """, (beneficiary_id,))
    if current:
        coordinates = parse_coordinates(current[0].get("curr_loc"))
        if coordinates is not None:
            return coordinates

    profile = fetch_rows(connection, f"""
        SELECT u.addr_ln1, u.addr_ln2, u.addr_ln3, u.city_name, u.zip_code,
               s.state_name, c.country_name
        FROM {SCHEMA}.users AS u
        LEFT JOIN {SCHEMA}.states AS s ON s.state_id = u.state_id
        LEFT JOIN {SCHEMA}.countries AS c ON c.country_id = u.country_id
        WHERE u.user_id = %s
    """, (beneficiary_id,))
    if not profile or geocoder is None:
        return None
    from distance import address_from_parts
    address = address_from_parts(profile[0], (
        "addr_ln1", "addr_ln2", "addr_ln3", "city_name", "state_name",
        "zip_code", "country_name",
    ))
    return geocoder.lookup(address)[0] if address else None
