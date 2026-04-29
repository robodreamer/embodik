# Personal Note: EmbodiK Installer Gist

Gist URL:

- https://gist.github.com/robodreamer/adc0b4452d474586c5890877b629005b

Gist ID:

- `adc0b4452d474586c5890877b629005b`

## Quick download commands

Linux:

```bash
curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/6cd82af5dd0e1b8c58bbfb71e99d96fa446c3f57/install_embodik_linux.sh
bash install_embodik_linux.sh
```

macOS:

```bash
curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/6cd82af5dd0e1b8c58bbfb71e99d96fa446c3f57/install_embodik_macos.sh
bash install_embodik_macos.sh --python python3.12
```

Current pinned revision:

- `6cd82af5dd0e1b8c58bbfb71e99d96fa446c3f57`

## Update gist from local repo scripts

Run from the `embodik` repo root:

```bash
jq -n \
  --rawfile mac scripts/install_embodik_macos.sh \
  --rawfile linux scripts/install_embodik_linux.sh \
  '{files:{"install_embodik_macos.sh":{content:$mac},"install_embodik_linux.sh":{content:$linux}}}' \
  > /tmp/embodik_gist_payload.json

gh api --method PATCH /gists/adc0b4452d474586c5890877b629005b \
  --input /tmp/embodik_gist_payload.json
```

## Optional: get latest revision SHA (for pinned raw URLs)

```bash
gh api gists/adc0b4452d474586c5890877b629005b --jq '.history[0].version'
```
