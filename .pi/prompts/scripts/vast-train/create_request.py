#!/usr/bin/env python3
"""Single-shot Vast create transport with durable, credential-free diagnostics.

The upstream API uses PUT /api/v0/asks/<offer>/ with Bearer authentication.
Unlike the CLI, this adapter retains HTTP status and never retries a mutation.
Unknown results require exact-label reconciliation, not another create request.
"""

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from workflow_common import IMAGE, now, write_json

MAX_RESPONSE_BYTES = 1024 * 1024
SAFE_ERROR_CODES = {"offer_unavailable", "offer_not_found", "insufficient_credit",
                    "invalid_args", "invalid_api_key", "too_many_requests"}


OWNED_LABEL = re.compile(r"toy-act-train-actv2-[a-zA-Z0-9_-]{1,128}")
FIXED_PAYLOAD = {
    "client_id": "me", "image": IMAGE, "disk": 100.0,
    "env": {"-p 22:22": "1"}, "runtype": "ssh_direc ssh_proxy",
    "cancel_unavail": True, "price": None, "extra": None, "onstart": None,
    "image_login": None, "jupyter_dir": None, "template_hash_id": None,
    "user": None, "python_utf8": False, "lang_utf8": False,
    "use_jupyter_lab": False, "force": False,
}
SAFE_PAYLOAD_FIELDS = frozenset(FIXED_PAYLOAD) | {"label", "id"}
MAX_DIAGNOSTIC_FIELDS = 8
SAFE_REASONS = {
    "invalid_json", "invalid_response_shape", "http_error", "confirmed_creation",
    "explicit_rejection", "unrecognized_response", "missing_credentials",
    "transport_error", "response_too_large", "response_read_error",
    "local_validation_failed", "unavailable_offer_rejection",
}


def build_payload(*, label: str) -> dict:
    """Keep the upstream CLI envelope; float disk is parity, not a proven fix."""
    return {**FIXED_PAYLOAD, "env": dict(FIXED_PAYLOAD["env"]), "label": label}


class RequestValidationError(ValueError):
    """Carry bounded categories/field names, never invalid caller values."""

    def __init__(self, *, category: str, fields: list[str]) -> None:
        super().__init__("Invalid create request")
        self.diagnostics = {"validation_category": category, "validation_fields": fields}


def request_snapshot(*, offer_id: int, label: str, payload: dict) -> dict:
    """Validate before transport and copy only known, fixed values into diagnostics."""
    if type(offer_id) is not int or offer_id <= 0:
        raise RequestValidationError(category="identity", fields=["offer_id"])
    if type(label) is not str or OWNED_LABEL.fullmatch(label) is None:
        raise RequestValidationError(category="identity", fields=["label"])
    if type(payload) is not dict or set(payload) != set(FIXED_PAYLOAD) | {"label"}:
        raise RequestValidationError(category="envelope_shape", fields=[])
    for field, expected in {**FIXED_PAYLOAD, "label": label}.items():
        value = payload[field]
        if type(value) is not type(expected) or value != expected:
            raise RequestValidationError(category="envelope_value", fields=[field])
        if field == "env" and any(type(key) is not str or type(item) is not str
                                  for key, item in value.items()):
            raise RequestValidationError(category="envelope_value", fields=["env"])
    # Do not copy caller-supplied arbitrary values, even after validation.
    return {"offer_id": offer_id,
            "payload": {**FIXED_PAYLOAD, "env": {"-p 22:22": "1"}, "label": label}}


def safe_diagnostics(result: dict) -> dict:
    """Revalidate persisted diagnostics before including them in user-visible logs."""
    safe = {}
    if type(result) is not dict:
        return safe
    if type(result.get("reason")) is str and result["reason"] in SAFE_REASONS:
        safe["reason"] = result["reason"]
    status = result.get("http_status")
    if type(status) is int and 100 <= status <= 599:
        safe["http_status"] = status
    code = result.get("provider_error_code")
    if type(code) is str and code in SAFE_ERROR_CODES:
        safe["provider_error_code"] = code
    http_code = result.get("provider_error_http_code")
    if type(http_code) is int and 100 <= http_code <= 599:
        safe["provider_error_http_code"] = http_code
    size = result.get("response_bytes")
    if type(size) is int and 0 <= size <= MAX_RESPONSE_BYTES:
        safe["response_bytes"] = size
    digest = result.get("response_sha256")
    if type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest):
        safe["response_sha256"] = digest
    category = result.get("validation_category")
    if type(category) is str and category in {"identity", "envelope_shape", "envelope_value"}:
        safe["validation_category"] = category
        fields = result.get("validation_fields")
        if type(fields) is list:
            safe["validation_fields"] = sorted({field for field in fields
                if type(field) is str and field in SAFE_PAYLOAD_FIELDS | {"offer_id"}})[:MAX_DIAGNOSTIC_FIELDS]
    identity = result.get("provider_error_id")
    if type(identity) is int and 0 <= identity <= 99999999:
        safe["provider_error_id"] = identity
    fields = result.get("provider_error_fields")
    if type(fields) is list:
        safe_fields = sorted({field for field in fields
                              if type(field) is str and field in SAFE_PAYLOAD_FIELDS})[:MAX_DIAGNOSTIC_FIELDS]
        if safe_fields:
            safe["provider_error_fields"] = safe_fields
    return safe


def diagnostic_hint(*, result: dict) -> str | None:
    """Explain only the known unavailable-offer signature, without inferring outcome."""
    diagnostics = safe_diagnostics(result)
    # Vast documents "error 404/3603: no_such_ask Instance type by id ... is not available."
    # https://github.com/vast-ai/docs/blob/main/api-reference/openapi/yaml/create_instance.yaml
    # https://docs.vast.ai/api-reference/instances/create-instance
    if (diagnostics.get("http_status") == 400
            and diagnostics.get("provider_error_code") == "invalid_args"
            and diagnostics.get("provider_error_http_code") == 404
            and diagnostics.get("provider_error_id") == 3603):
        return ("Vast's documented no_such_ask diagnostic indicates an unavailable offer; "
                "this does not establish that no rental was created")
    return None


def message_diagnostics(message: object) -> dict:
    """Recognize only the documented numeric prefix and quoted envelope names."""
    if type(message) is not str:
        return {}
    match = re.match(r"\Aerror ([1-5][0-9]{2})/([0-9]{1,8}):(?:\s|$)", message)
    if match is None:
        return {}
    fields = re.findall(r"(?<![A-Za-z0-9_])['\"]([a-z_]{1,32})['\"](?![A-Za-z0-9_])", message)
    return safe_diagnostics({"provider_error_http_code": int(match[1]),
                             "provider_error_id": int(match[2]),
                             "provider_error_fields": fields})


class NoRedirects(HTTPRedirectHandler):
    """Do not resend a mutation or forward credentials to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl, /) -> None:
        return None


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("Duplicate response fields")
        fields[key] = value
    return fields


def unavailable_offer_rejection(*, status: int, response: dict, offer_id: int | None) -> bool:
    """Accept only Vast's documented no_such_ask envelope for this requested ask.

    The official create-instance spec describes 404/3603 as "Offer not found or
    not available", with success=false and error=invalid_args. The observed
    HTTP 400 wrapper is accepted only with that exact nested rejection, not as
    a generic HTTP-error retry rule. Also accept the observed two-space separator
    after no_such_ask: offline reconstruction matched the 142-byte response hash
    in iteration 20261013T125627Z-d77828791c1b. No other whitespace is normalized.
    https://github.com/vast-ai/docs/blob/main/api-reference/openapi/yaml/create_instance.yaml
    """
    if (type(status) is not int or status != 400
            or type(offer_id) is not int or offer_id <= 0
            or not {"success", "error", "msg"} <= response.keys()
            or not response.keys() <= {"success", "error", "msg", "ask_id"}
            or response["success"] is not False
            or response["error"] != "invalid_args"
            or type(response["msg"]) is not str):
        return False
    # Exact text, not just numeric diagnostics; no contract/instance fields,
    # trailing text, alternate shapes or mismatched ask identities are accepted.
    documented_message = f"error 404/3603: no_such_ask Instance type by id {offer_id} is not available."
    observed_message = f"error 404/3603: no_such_ask  Instance type by id {offer_id} is not available."
    if response["msg"] not in (documented_message, observed_message):
        return False
    return ("ask_id" not in response
            or type(response["ask_id"]) is int and response["ask_id"] == offer_id)


def classify_response(*, status: int, body: bytes, offer_id: int | None = None) -> dict:
    result = {"http_status": status, "response_bytes": len(body),
              "response_sha256": hashlib.sha256(body).hexdigest(),
              "outcome": "unknown", "reason": "invalid_json"}
    try:
        response = json.loads(body, object_pairs_hook=unique_fields)
    except (ValueError, UnicodeError):
        return result
    if not isinstance(response, dict):
        result["reason"] = "invalid_response_shape"
        return result
    # Persist only allowlisted fields. Never persist free-form API messages,
    # headers, URLs or raw bodies: they can contain credentials/instance records.
    code = response.get("error")
    if isinstance(code, str) and code in SAFE_ERROR_CODES:
        result["provider_error_code"] = code
    result.update(message_diagnostics(response.get("msg")))
    identity = response.get("new_contract")
    if unavailable_offer_rejection(status=status, response=response, offer_id=offer_id):
        result.update(outcome="rejected", reason="unavailable_offer_rejection")
    elif not 200 <= status < 300:
        result["reason"] = "http_error"
    elif response.get("success") is True and type(identity) is int and identity > 0:
        result.update(outcome="created", reason="confirmed_creation", instance_id=identity)
    elif (response.get("success") is False
            and (identity is None or type(identity) is int and identity == 0)
            # Alternate identity fields can contradict rejection even when the
            # canonical new_contract field is absent or empty. Fail closed.
            and not {"instance_id", "instance", "id", "contract_id", "contract"} & response.keys()):
        result.update(outcome="rejected", reason="explicit_rejection")
    else:
        result["reason"] = "unrecognized_response"
    return result


def send_create(*, offer_id: int, label: str) -> dict:
    try:
        payload = build_payload(label=label)
        snapshot = request_snapshot(offer_id=offer_id, label=label, payload=payload)
    except RequestValidationError as error:
        return {"outcome": "unknown", "reason": "local_validation_failed", **error.diagnostics}
    except (TypeError, ValueError):
        return {"outcome": "unknown", "reason": "local_validation_failed"}
    diagnostics = {"request_snapshot": snapshot,
                   "request_snapshot_sha256": hashlib.sha256(
                       json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    key = os.environ.get("VAST_API_KEY", "")
    if not key or any(character.isspace() for character in key):
        return {**diagnostics, "outcome": "unknown", "reason": "missing_credentials"}
    request = Request(f"https://console.vast.ai/api/v0/asks/{offer_id}/",
                      data=json.dumps(payload).encode(), method="PUT",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        response = build_opener(NoRedirects()).open(request, timeout=90)
    except HTTPError as error:
        response = error
    except (URLError, OSError, ValueError, HTTPException):
        # Never serialize exceptions: they may contain request URLs or headers.
        return {**diagnostics, "outcome": "unknown", "reason": "transport_error"}
    try:
        with response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                return {**diagnostics, "outcome": "unknown", "reason": "response_too_large", "http_status": response.code}
            return {**diagnostics, **classify_response(status=response.code, body=body, offer_id=offer_id)}
    except (OSError, ValueError, HTTPException):
        return {**diagnostics, "outcome": "unknown", "reason": "response_read_error"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offer-id", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.offer_id <= 0 or not OWNED_LABEL.fullmatch(args.label):
        parser.error("Invalid offer ID or iteration-owned label")
    if args.result.exists():
        parser.error("Result already exists; do not resend a create request")
    result = {"version": 1, "offer_id": args.offer_id, "label": args.label,
              "recorded_at": now(), **send_create(offer_id=args.offer_id, label=args.label)}
    # Persist before stdout/parent state updates, so resume can recover the result.
    write_json(path=args.result, value=result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
