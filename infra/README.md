# Training infrastructure

Everything the training jobs need, in one Nebius project. These steps were run by hand on 2026-10-28 with Nebius CLI 0.12.297; `setup_infra.sh` will script them later.

## Resources

| Resource | Purpose |
|---|---|
| Object Storage bucket | Dataset, run outputs, the selected LoRA. Mounted into each job at `/data`. |
| Service account | Owns the S3 access key used by the bucket mount. |
| Group in the project, with role `storage.editor` on the project | Gives the service account bucket access and nothing else. |
| Access key for the service account | S3 credentials. Created with `--secret-delivery-mode mystery_box`, so the secret is never shown. |
| MysteryBox secret with keys `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` | What `nebius ai job create --volume s3://…:PROFILE@SECRET` reads. The secret created by the access key holds only a `secret` key, so the values are copied into a second secret with these names. |
| `deploy.env` (gitignored) | Project, subnet, bucket and secret IDs for the scripts. Identifiers only. |

## Steps

```bash
nebius storage bucket create --parent-id $PROJECT_ID --name <bucket>
nebius iam service-account create --parent-id $PROJECT_ID --name lora-training
nebius iam group create --parent-id $PROJECT_ID --name lora-training-storage
nebius iam access-permit create --parent-id $GROUP_ID --resource-id $PROJECT_ID --role storage.editor
nebius iam group-membership create --parent-id $GROUP_ID --member-id $SA_ID
nebius iam v2 access-key create --parent-id $PROJECT_ID --account-service-account-id $SA_ID \
  --name lora-training-s3 --secret-delivery-mode mystery_box
# then create a MysteryBox secret with S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY from that access key
```

## Local AWS profile

The CLI resolves `s3://` volumes on your machine before it submits the job, and needs a named AWS profile that gives the region and endpoint. It holds no credentials; those come from the MysteryBox secret. Add to `~/.aws/config`:

```ini
[profile nebius-eu-north1]
region = eu-north1
endpoint_url = https://storage.eu-north1.nebius.cloud
```

and set `S3_PROFILE=nebius-eu-north1` in `deploy.env`. Without it, job creation fails with "failed to get shared config profile".

## Bucket mount behaviour

The mount is mountpoint-s3. New files can be written sequentially, but existing files cannot be renamed or overwritten. That is why ai-toolkit writes to local disk and `training/run_job.sh` copies each finished file to the bucket once.

## Copying files

```bash
uv run infra/bucket.py put ./dataset dataset/v1
uv run infra/bucket.py ls runs/
uv run infra/bucket.py get runs/<run-name> ./results/<run-name>
```

`bucket.py` reads the S3 credentials from MysteryBox at run time through the Nebius CLI and does not store them.
