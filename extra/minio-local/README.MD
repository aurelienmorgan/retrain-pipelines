serves for s3-related unit-testing.
for, e.g., RP_ASSETS_CACHE and/or RP_ARTIFACTS_STORE and/or RP_WEB_SERVER_LOGS

Exposes:
  - S3 API at http://localhost:9000
  - Console UI at http://localhost:9001

Note:
  - to start the docker service, no .env is per-se "required" for authentication.
  Tests are operated with the MinIO default root profile credentials : minioadmin/minioadmin.
  You can choose to manage different profiles in MinIO
  and set S3_ACCESS_KEY and S3_SECRET_KEY accordingly.
  - MinIO is an archived project and is now replaced by MinIO AIStor
  which allows all usages including commercial use,
  as long as it doesn't involve multi-node distributed S3 stores
  (so, a single-node for unit-tests would be fine).
  In case it later proves any useful, we may consider switching
  to MinIO AIStor or use an alternative such as https://garagehq.deuxfleurs.fr/ .


Using it like S3 (env-driven) via python boto3 =>
For a typical client app, you’d set env vars like:
```bash
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_ENDPOINT_URL="http://localhost:9000"
export AWS_REGION="us-east-1"
```
