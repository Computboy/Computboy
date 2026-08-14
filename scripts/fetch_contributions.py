#!/usr/bin/env python3
"""Fetch the authenticated user's GitHub contribution calendar as date/count JSON.

The token is read from PROFILE_TOKEN by default. For private/internal contribution
counts, use a GitHub personal access token (classic) with the `read:user` scope.

The output intentionally contains only dates and contribution counts. Repository
names and other private repository details are never requested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.github.com/graphql"

QUERY = r"""
query($login: String!) {
  viewer {
    login
  }
  user(login: $login) {
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
            weekday
          }
        }
      }
    }
  }
}
"""


def fetch(username: str, token: str) -> tuple[str, list[dict[str, int | str]], int]:
    payload = json.dumps(
        {"query": QUERY, "variables": {"login": username}}
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Computboy-Contribution-Pulse",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub GraphQL HTTP {exc.code}: {details}") from exc

    if body.get("errors"):
        raise RuntimeError(f"GitHub GraphQL error: {body['errors']}")

    viewer = body["data"]["viewer"]["login"]
    user = body["data"].get("user")
    if user is None:
        raise RuntimeError(f"GitHub user not found: {username}")

    calendar = user["contributionsCollection"]["contributionCalendar"]
    records: list[dict[str, int | str]] = []
    for week in calendar["weeks"]:
        for day in week["contributionDays"]:
            records.append(
                {
                    "date": day["date"],
                    "count": int(day["contributionCount"]),
                }
            )

    return viewer, records, int(calendar["totalContributions"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--username",
        default=os.getenv("GITHUB_REPOSITORY_OWNER", "Computboy"),
        help="GitHub login whose contribution calendar should be rendered",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contributions.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--token-env",
        default="PROFILE_TOKEN",
        help="Environment variable containing the GitHub token",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = os.getenv(args.token_env, "").strip()
    if not token:
        print(
            f"Missing token: set the {args.token_env} environment variable.",
            file=sys.stderr,
        )
        return 2

    viewer, records, total = fetch(args.username, token)

    if viewer.lower() != args.username.lower():
        print(
            f"WARNING: token belongs to @{viewer}, but data was requested for "
            f"@{args.username}. Private contributions for @{args.username} may not "
            "be available. Use a token owned by the profile owner.",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    nonzero_days = sum(1 for item in records if int(item["count"]) > 0)
    print(
        f"Fetched {len(records)} calendar days for @{args.username}: "
        f"{total} contributions across {nonzero_days} active days."
    )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
