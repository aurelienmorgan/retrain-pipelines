
script that handles two services :
  - MinIO
    - starts the service (if not already running)
    - runs the pytests suites
    - shuts down the service (if it wasn't originally on)
  - PostgreSQL
    - starts the service (if not already running)
    - runs the pytests suites
    - shuts down the service (if it wasn't originally on)

-------------------------------------------------------------------------

If you are updating the test-services script on a Windows machine,
then the script get Windows (CRLF) line endings (most likely under WSL).


If this is the case for you, run the below after the update
so it can successfully be called as a pre-commit hook :

```bash
sed -i 's/\r$//' tests/unit/pytest_integration__minio__postgres.sh
```
