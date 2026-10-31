#!/bin/zsh
# Start from this script's own folder without changing the user's directory.
exec uv run --directory "${0:A:h}" local-redact
