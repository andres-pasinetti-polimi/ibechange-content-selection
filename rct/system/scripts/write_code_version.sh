#!/bin/sh
# Writes system/CODE_VERSION, the commit the service is deployed from, before the copy to the server
# (rsync copies no .git). The commit gets "-dirty" when system/ has changes not committed, which
# scripts/server_check.py reports as a failure. Run from system/.
cd "$(dirname "$0")/.." || exit 1
version=$(git rev-parse HEAD) || exit 1
if [ -n "$(git status --porcelain -- .)" ]; then
    version="$version-dirty"
fi
echo "$version" > CODE_VERSION
echo "CODE_VERSION: $version"
