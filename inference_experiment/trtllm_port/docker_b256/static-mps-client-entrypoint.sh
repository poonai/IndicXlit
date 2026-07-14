#!/usr/bin/env bash
set -euo pipefail

source /opt/indicxlit/static-mps-client-env.sh
exec /opt/indicxlit/bin/indicxlit-rust-executor-server
