# GCP Pilot Operations

This note documents the Google Cloud pilot deployment for the Born in Bradford
assistant.

The pilot now runs on a GPU Compute Engine VM. Caddy serves HTTPS publicly and
reverse-proxies to the Flask app on localhost. The app requires Basic Auth.

## Current Deployment

```text
GCP project: bib-assistant
Primary VM: bib-assistant-gpu-primary
Zone: us-central1-a
Machine type: g2-standard-4
GPU: 1 x NVIDIA L4
Static IP name: bib-assistant-gpu-primary-static-ip
Static IP: 34.134.9.82
HTTPS URL: https://34-134-9-82.sslip.io/assistant
Service name: bib-assistant
Caddy service: caddy
Public firewall rule: allow-bib-assistant-http-https
```

The Flask app binds to `127.0.0.1:5050` on the VM. Port `5050` is not exposed
publicly.

The previous CPU VM (`bib-assistant-vm` in `europe-west2-a`) has been deleted,
and its old static IP (`35.242.187.34`) has been released. Do not use the old
`35-242-187-34.sslip.io` URL.

The previous GPU VM (`bib-assistant-gpu-test` in `europe-west4-c`) is stopped.
It was affected by repeated L4 capacity stockouts when restarting from the
office-hours schedule. The active pilot now uses `bib-assistant-gpu-primary` in
`us-central1-a`, and the schedule has been removed so the GPU remains allocated
during the pilot.

## Model Runtime

The hosted pilot runtime uses the bundled local GGUF model:

```text
Backend: llama_cpp
Model: models/bib-llama-3.1-8b.Q4_K_M.gguf
Context window: 8192
RAG retrieval: 5 results per collection, capped at 7000 context characters
GPU offload: --llama-n-gpu-layers -1
```

This avoids Hugging Face API credits, provider costs, and rate limits. The model
is quantized to make the pilot deployable, which reduces storage and memory
requirements but can reduce answer quality compared with larger hosted models.
The GPU VM is much faster than the CPU VM for the same quantized model. After
moving to the L4 GPU, the runtime was relaxed from the earlier CPU-safe settings
of `4096` context / `3` retrieval results / `3500` context characters to
`8192` context / `5` retrieval results / `7000` context characters. This gives
paper and questionnaire questions more evidence while staying below the point
where large prompts noticeably dilute answers or increase latency too much.

Check that the model is using the GPU:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='nvidia-smi'
```

Check the active model configuration:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo journalctl -u bib-assistant --no-pager --lines=80 | grep -E "LLM backend|GGUF model|Loading local GGUF|RAG results|RAG context"'
```

## Retrieval Quality Notes

Several retrieval changes were made after pilot testing:

- Broad publication questions now use a de-duplicated publication shortlist, so
  questions like `What has been published on childhood obesity?` retrieve
  distinct papers rather than repeated chunks from the same PDF.
- Ordinary topic words such as `childhood` and `obesity` are no longer treated
  as acronym/tool matches. This avoids irrelevant "Exact Acronym" context.
- Lowercase real acronyms such as `ckat` still get exact acronym/tool matching.
- Short follow-up questions such as `what other ones have been used` now inherit
  the previous user topic, so retrieval does not drift to unrelated uses of
  words like `other`.
- Covariate/adjustment questions now use a targeted evidence path for terms such
  as `covariates`, `confounders`, `adjusted for`, `controlled for`, `ethnicity`,
  `deprivation`, `smoking`, `alcohol`, `BMI`, and `income`.
- High-signal retrieval blocks skip the generic anchor guide so the useful
  evidence is less likely to be truncated by the 3500-character context cap.

When answer quality looks poor, first inspect what context retrieval produced.
Most recent failures were caused by noisy or thin retrieval context rather than
GPU execution itself.

## Availability And Scheduling

The active GPU VM intentionally has no office-hours schedule attached:

```text
Primary VM: bib-assistant-gpu-primary
Schedule: none
Reason: stopped GPU VMs do not reserve L4 capacity, so scheduled restarts can fail
```

Check that no schedule is attached:

```bash
gcloud compute instances describe bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --format='value(status,networkInterfaces[0].accessConfigs[0].natIP,resourcePolicies)'
```

Leaving the GPU running is more expensive than office-hours scheduling, but it
keeps the L4 allocated and avoids the `ZONE_RESOURCE_POOL_EXHAUSTED` restart
failure seen during pilot testing.

## Start And Stop The VM

Run these commands locally from a machine with `gcloud` installed and
authenticated.

Start the VM:

```bash
gcloud compute instances start bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant
```

Stop the VM:

```bash
gcloud compute instances stop bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant
```

Check VM status and external IP:

```bash
gcloud compute instances describe bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)'
```

The VM uses a reserved static external IP. The URL remains stable, but stopping
the VM can still make restart unreliable if the region runs out of L4 capacity.

## Check Or Restart Services

Check assistant service status:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo systemctl status bib-assistant --no-pager --lines=40'
```

Follow assistant logs:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo journalctl -u bib-assistant -f'
```

Restart the assistant:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo systemctl restart bib-assistant'
```

Check Caddy HTTPS proxy status:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo systemctl status caddy --no-pager --lines=40'
```

Restart Caddy:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo systemctl restart caddy'
```

## Updating The App

### Code-Only Updates

Use this for normal app changes, such as:

- `llm_poc/server.py`
- `llm_poc/bib_research_assistant.py`
- `llm_poc/bib_utils.py`
- files in `llm_poc/static/`
- prompt/reference text in `llm_poc/production_data_dictionary_reference.md`
- runtime environment or runtime requirements changes

Run locally from the full development repo:

```bash
packaging/deploy_gcp_code_update.sh
```

The script now defaults to:

```text
VM_NAME=bib-assistant-gpu-primary
ZONE=us-central1-a
REMOTE_ARCHIVE=/tmp/bib-assistant-code-update.tar.gz
USE_SUDO=1
REMOTE_OWNER=dawud_izza_york_ac_uk:dawud_izza_york_ac_uk
```

This uploads only the small runtime code/static package, extracts it over the
existing VM runtime, and restarts `bib-assistant.service`.

If dependencies changed and you want the VM to reinstall
`llm_poc/requirements_runtime.txt`, run:

```bash
INSTALL_REQUIREMENTS=1 packaging/deploy_gcp_code_update.sh
```

If `docs/` changed and you want to sync the data dictionary docs too, run:

```bash
INCLUDE_DOCS=1 packaging/deploy_gcp_code_update.sh
```

### Full Runtime Updates

Use the full runtime path only when one of these changes:

- quantized GGUF model in `models/`
- prebuilt Chroma DB in `llm_poc/.chroma_db/`
- bundled source PDFs in `papers/`
- a major runtime layout change

Rebuild locally:

```bash
packaging/make_runtime_bundle.sh
tar -czf dist/bib-assistant-runtime.tar.gz -C dist bib-assistant-runtime
```

Then upload and unpack on the VM. The archive is several GB, so prefer code-only
updates whenever possible.

## Basic Auth

The Flask app supports lightweight Basic Auth with environment variables:

```text
BIB_BASIC_AUTH_USER
BIB_BASIC_AUTH_PASSWORD
```

Current pilot username:

```text
bibpilot
```

The password is set on the VM in the systemd drop-in and should be shared only
with pilot users.

Check whether Basic Auth is set on the VM service without printing the password:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo systemctl cat bib-assistant | grep BIB_BASIC_AUTH_USER || true'
```

Set or rotate the Basic Auth credentials:

```bash
gcloud compute ssh bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant \
  --command='sudo mkdir -p /etc/systemd/system/bib-assistant.service.d && sudo tee /etc/systemd/system/bib-assistant.service.d/auth.conf >/dev/null <<EOF
[Service]
Environment=BIB_BASIC_AUTH_USER=bibpilot
Environment=BIB_BASIC_AUTH_PASSWORD=REPLACE_WITH_STRONG_PASSWORD
EOF
sudo systemctl daemon-reload
sudo systemctl restart bib-assistant'
```

Important: Basic Auth is shared for this pilot. Rotate the password if it is
shared too widely or when the pilot ends.

## User Access

Users no longer need to provide their public IP address.

Share:

```text
URL: https://34-134-9-82.sslip.io/assistant
Username: bibpilot
Password: <current pilot password>
```

The public firewall allows ports `80` and `443` only. Caddy redirects HTTP to
HTTPS and proxies HTTPS traffic to the local Flask app.

Check public firewall rules:

```bash
gcloud compute firewall-rules list \
  --project=bib-assistant \
  --filter='targetTags:bib-assistant' \
  --format='table(name,allowed,sourceRanges,disabled)'
```

## Quick Health Check

Ask the user to open:

```text
https://34-134-9-82.sslip.io/assistant
```

If it does not load:

- Confirm the VM is running.
- Confirm Caddy is active.
- Confirm the assistant service is active.
- Confirm the VM still has static IP `34.134.9.82` attached.
- Confirm the GPU is visible with `nvidia-smi`.

## Static IP

The pilot uses a reserved static IP:

```text
34.134.9.82
```

Check the static IP:

```bash
gcloud compute addresses describe bib-assistant-gpu-primary-static-ip \
  --region=us-central1 \
  --project=bib-assistant \
  --format='value(address,status,users)'
```

Release the static IP when the pilot ends:

```bash
gcloud compute addresses delete bib-assistant-gpu-primary-static-ip \
  --region=us-central1 \
  --project=bib-assistant
```

Only release it after the VM no longer needs the stable public URL.

## Cost Hygiene

Stop the GPU VM only when cost control is more important than reliable
availability:

```bash
gcloud compute instances stop bib-assistant-gpu-primary \
  --zone=us-central1-a \
  --project=bib-assistant
```

Persistent disk storage may still incur a smaller cost while the VM is stopped.
There is currently no weekday office-hours policy on the active GPU VM because
stopped L4 GPU VMs can fail to restart when Google Cloud capacity is exhausted.

## Retired CPU Deployment

The CPU VM was retired after the GPU deployment proved faster and produced
better answers with the same local GGUF model.

```text
Retired VM: bib-assistant-vm
Retired zone: europe-west2-a
Retired static IP: 35.242.187.34
Retired URL: https://35-242-187-34.sslip.io/assistant
```

Do not deploy to the retired CPU VM unless a new CPU instance is intentionally
created.
