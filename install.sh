#!/usr/bin/env bash
# Symlink skills/rlm into ~/.claude/skills/rlm so Claude Code discovers it.
set -eu

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO_DIR/skills/rlm"
DST_DIR="$HOME/.claude/skills"
DST="$DST_DIR/rlm"

if [ ! -d "$SRC" ]; then
  echo "error: source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DST_DIR"

if [ -L "$DST" ]; then
  current="$(readlink "$DST")"
  if [ "$current" = "$SRC" ]; then
    echo "already installed: $DST -> $SRC"
    exit 0
  fi
  echo "removing stale symlink: $DST -> $current"
  rm "$DST"
elif [ -e "$DST" ]; then
  echo "error: $DST already exists and is not a symlink." >&2
  echo "       move or remove it first, then rerun install.sh." >&2
  exit 1
fi

ln -s "$SRC" "$DST"
echo "installed: $DST -> $SRC"
echo
echo "next: export ANTHROPIC_API_KEY=... and use the skill from any Claude Code session."
