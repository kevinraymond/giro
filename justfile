# giro tasks. In the repo: `just <recipe>`. From anywhere: `just -g giro <recipe>`
# (~/.config/just/justfile mounts this file as the module `giro`).

set shell := ["bash", "-uc"]

port := "8470"
https_port := "8443"  # the Quest's WebXR page needs HTTPS over the LAN
log := "data/run/serve.log"

# list the recipes
default:
    @just --justfile {{justfile()}} --list

# who serves giro on the port, since when, and which jobs it is running
serve-status:
    #!/usr/bin/env bash
    pid=$(ss -ltnpH "sport = :{{port}}" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    if [ -z "$pid" ]; then echo "giro serve: not running on {{port}}"; exit 0; fi
    ps -o pid,lstart,etime,args -p "$pid"
    curl -s localhost:{{port}}/api/jobs | python3 -c "import json,sys; r=[j['id'] for j in json.load(sys.stdin) if j['status']=='running']; print('running jobs:', ', '.join(r) or 'none')"

# start giro serve detached from this terminal, logging to data/run/serve.log
serve-start:
    #!/usr/bin/env bash
    set -euo pipefail
    if ss -ltnH "sport = :{{port}}" | grep -q .; then echo "port {{port}} is taken (just serve-status)"; exit 1; fi
    mkdir -p "$(dirname {{log}})"
    setsid .venv/bin/giro serve --port {{port}} --https-port {{https_port}} > {{log}} 2>&1 < /dev/null &
    for _ in $(seq 60); do curl -sf -o /dev/null localhost:{{port}}/api/jobs && break; sleep 0.5; done
    cat {{log}}

# stop giro serve; refuses while jobs run unless given `yes` (they resume after a restart)
serve-stop force="no":
    #!/usr/bin/env bash
    set -euo pipefail
    pid=$(ss -ltnpH "sport = :{{port}}" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
    if [ -z "$pid" ]; then echo "giro serve: not running on {{port}}"; exit 0; fi
    running=$(curl -s localhost:{{port}}/api/jobs | python3 -c "import json,sys; print(' '.join(j['id'] for j in json.load(sys.stdin) if j['status']=='running'))")
    if [ -n "$running" ] && [ "{{force}}" != "yes" ]; then
        echo "jobs running: $running"; echo "stop anyway with: just serve-stop yes (or serve-restart yes)"; exit 1
    fi
    kill "$pid"
    while ss -ltnH "sport = :{{port}}" | grep -q .; do sleep 0.5; done
    echo "stopped giro serve (pid $pid)"

# restart giro serve, e.g. after a code change (it runs the code it started with)
serve-restart force="no": (serve-stop force) serve-start

# follow the server log
serve-log:
    tail -f {{log}}

# lint (pyflakes rules, catches undefined names after refactors) and run the tests
check:
    uvx ruff check --select F src tests
    uv run pytest -q

# build the UI into ui/dist, which giro serve serves
ui:
    cd ui && npm run build

image_name := "ghcr.io/kevinraymond/giro"

# build the container image (the first build compiles COLMAP and Brush: about an hour)
image tag="latest":
    docker build -t {{image_name}}:{{tag}} .

# run the image on every GPU, with the models store and ./data mounted
image-run tag="latest" models="~/models":
    docker run --rm -it --gpus all -p {{port}}:8470 -p {{https_port}}:8443 \
        -v {{models}}:/models -v "$PWD/data":/app/data {{image_name}}:{{tag}}

# push a built image to GitHub Container Registry (docker login ghcr.io first)
image-push tag="latest":
    docker push {{image_name}}:{{tag}}
