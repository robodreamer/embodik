# Personal Note: EmbodiK Installer Gist

Gist URL:

- https://gist.github.com/robodreamer/adc0b4452d474586c5890877b629005b

Gist ID:

- `adc0b4452d474586c5890877b629005b`

## Quick download commands

Linux:

```bash
curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/ee3213c6f25fda3488913ba065fcccdc8f6e98fb/install_embodik_linux.sh
bash install_embodik_linux.sh
```

macOS:

```bash
curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/ee3213c6f25fda3488913ba065fcccdc8f6e98fb/install_embodik_macos.sh
bash install_embodik_macos.sh --python python3.12
```

Current pinned revision:

- `ee3213c6f25fda3488913ba065fcccdc8f6e98fb`

## Update gist from local repo scripts

Run from the `embodik` repo root:

```bash
gh gist edit adc0b4452d474586c5890877b629005b \
  scripts/install_embodik_linux.sh \
  scripts/install_embodik_macos.sh
```

## Optional: get latest revision SHA (for pinned raw URLs)

```bash
gh api gists/adc0b4452d474586c5890877b629005b --jq '.history[0].version'
```
