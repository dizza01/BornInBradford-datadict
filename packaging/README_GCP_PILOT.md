# GCP Pilot Operations

This note documents the temporary Google Cloud pilot deployment for the Born in
Bradford assistant.

The current setup runs the runtime bundle on a single Compute Engine VM. Caddy
serves HTTPS publicly and reverse-proxies to the Flask app on localhost. The app
requires Basic Auth.

## Current Deployment

```text
GCP project: bib-assistant
VM name: bib-assistant-vm
Zone: europe-west2-a
Machine type: e2-standard-8
Static IP name: bib-assistant-static-ip
HTTPS URL: https://35-242-187-34.sslip.io/assistant
Service name: bib-assistant
Caddy service: caddy
Public firewall rule: allow-bib-assistant-http-https
```

The Flask app binds to `127.0.0.1:5050` on the VM. Port `5050` is not exposed
publicly.

## Basic Auth

The Flask app supports lightweight Basic Auth with environment variables:

```text
BIB_BASIC_AUTH_USER
BIB_BASIC_AUTH_PASSWORD
```

If both are set, every route requires a browser username/password prompt. If
either value is missing, Basic Auth is disabled.

Current pilot username:

```text
bibpilot
```

The password is set on the VM in the systemd drop-in and should be shared only
with pilot users.

Check whether Basic Auth is set on the VM service:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo systemctl cat bib-assistant | grep BIB_BASIC_AUTH || true'
```

Set or rotate the Basic Auth credentials:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
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

## Start And Stop The VM

Run these commands locally from a machine with `gcloud` installed and
authenticated.

Start the VM:

```bash
gcloud compute instances start bib-assistant-vm --zone=europe-west2-a --project=bib-assistant
```

Stop the VM:

```bash
gcloud compute instances stop bib-assistant-vm --zone=europe-west2-a --project=bib-assistant
```

Check VM status and external IP:

```bash
gcloud compute instances describe bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)'
```

The VM uses a reserved static external IP, so the URL should remain stable when
the VM is stopped and restarted.

## Check Or Restart The Assistant Service

Check service status:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo systemctl status bib-assistant --no-pager --lines=40'
```

Follow service logs:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo journalctl -u bib-assistant -f'
```

Restart the assistant:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo systemctl restart bib-assistant'
```

Check Caddy HTTPS proxy status:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo systemctl status caddy --no-pager --lines=40'
```

Follow Caddy logs:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo journalctl -u caddy -f'
```

Restart Caddy:

```bash
gcloud compute ssh bib-assistant-vm \
  --zone=europe-west2-a \
  --project=bib-assistant \
  --command='sudo systemctl restart caddy'
```

## Updating The App

There are two update paths.

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

You can combine both:

```bash
INSTALL_REQUIREMENTS=1 INCLUDE_DOCS=1 packaging/deploy_gcp_code_update.sh
```

After deployment, check:

```text
https://35-242-187-34.sslip.io/assistant
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

Then upload and unpack on the VM. The first deployment used `gcloud compute scp`
for this, but it is slow because the archive is around 4.6GB. Prefer code-only
updates whenever possible.

## User Access

Users no longer need to provide their public IP address.

Share:

```text
URL: https://35-242-187-34.sslip.io/assistant
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
https://35-242-187-34.sslip.io/assistant
```

If it does not load:

- Confirm the VM is running.
- Confirm Caddy is active.
- Confirm the assistant service is active.
- Confirm the VM still has static IP `35.242.187.34` attached.

## Static IP

The pilot uses a reserved static IP:

```text
35.242.187.34
```

Check the static IP:

```bash
gcloud compute addresses describe bib-assistant-static-ip \
  --region=europe-west2 \
  --project=bib-assistant \
  --format='value(address,status)'
```

Release the static IP when the pilot ends:

```bash
gcloud compute addresses delete bib-assistant-static-ip \
  --region=europe-west2 \
  --project=bib-assistant
```

Only release it after the VM no longer needs the stable public URL.

## Cost Hygiene

Stop the VM when the pilot is not being used:

```bash
gcloud compute instances stop bib-assistant-vm --zone=europe-west2-a --project=bib-assistant
```

Persistent disk storage may still incur a smaller cost while the VM is stopped.
