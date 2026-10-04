# Deploying DRE on a server — step by step

Order: server → module → check over the network → embedding into the program.
**Do not move to the next step until the previous one has worked.**

Notation (substitute your own values):
- `SERVER_IP` — the address of the new server (Xeon, 128 GB);
- `VM_IP` — the address of the virtual machine where the main program runs;
- `YOUR_COMPUTER_IP` — the address of the laptop you test from;
- `LOGIN` — the GitHub login where the `dre` repository lives.

---

## 1. Connect and inspect the server

From the laptop (PowerShell or Ubuntu):
```bash
ssh root@SERVER_IP
```

On the server:
```bash
cat /etc/os-release | head -3   # Ubuntu version
nproc                           # number of cores
free -h                         # memory
curl -sI https://github.com | head -1   # is there internet: should be HTTP/2 200
```
If there is no internet — you cannot go further; resolve it first with whoever provided the server.

## 2. Update the system and install Docker

```bash
apt update && apt upgrade -y
apt install -y git curl ufw
curl -fsSL https://get.docker.com | sh
docker run --rm hello-world
```
**Hello from Docker!** should appear.

## 3. Firewall

```bash
ufw allow 22/tcp
ufw allow from VM_IP to any port 8080 proto tcp
ufw allow from YOUR_COMPUTER_IP to any port 8080 proto tcp
ufw enable
ufw status
```
**Do not skip the line with `22`** — otherwise you will lose SSH access.

## 4. Fetch the project

```bash
git clone https://github.com/LOGIN/dre.git ~/dre
cd ~/dre
git checkout v0.2.1
```

## 5. Build and check

```bash
docker build -t dre:0.2.1 .
docker run --rm --cpus=1 dre:0.2.1 selfcheck
```
The first build takes 5–15 minutes. The self-check **must finish with `OK`**. If it does not — do not continue, save the output.

## 6. Access key and starting the service

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.dre_key
chmod 600 ~/.dre_key
cat ~/.dre_key          # copy the key for yourself — the program and the test script need it

docker run -d --name dre --restart=unless-stopped --cpus=1 \
  -p 8080:8080 -e DRE_API_KEY="$(cat ~/.dre_key)" dre:0.2.1

docker ps               # the dre container should be Up
docker logs --tail 30 dre
```

`--restart=unless-stopped` — the service comes back up by itself after a server reboot.

## 7. Check on the server itself

Put any PDF on the server (from the laptop: `scp test.pdf root@SERVER_IP:~/`), then:
```bash
curl -s -o test.docx -w "%{http_code}\n" \
  -H "Authorization: Bearer $(cat ~/.dre_key)" \
  --data-binary @test.pdf localhost:8080/convert
```
- `200` and a `test.docx` appeared — it works.
- A different code — the document was refused or there was an error: `cat test.docx` shows the reason, `docker logs --tail 50 dre` the details.

## 8. Check over the network from the laptop

```bash
curl -s -o test.docx -w "%{http_code}\n" \
  -H "Authorization: Bearer KEY" \
  --data-binary @test.pdf http://SERVER_IP:8080/convert
```
If it hangs — the firewall (step 3) or the network between the laptop and the server.

Then run a folder of documents with the `docs/test_remote.py` script (instructions inside the file) and open the results in Word next to the originals.

## 9. Embedding into the main program

1. Copy `docs/dre_client.py` into the main program's project.
2. Install the library: `pip install requests`.
3. Set the environment variables on the VM:
   - `DRE_URL=http://SERVER_IP:8080/convert`
   - `DRE_API_KEY=KEY`
4. Find the place where the program currently converts PDF to DOCX (in VS Code: Ctrl+Shift+F, search for `docx`, `convert`, `pdf`).
5. Replace the call to the old converter with:
   ```python
   from dre_client import convert_document

   docx_bytes = convert_document(path_to_pdf)
   if docx_bytes is None:
       # the document was not converted: do the same thing the program currently does with failed files
       ...
   else:
       with open(path_to_docx, "wb") as f:
           f.write(docx_bytes)
   ```
6. Test on 3–5 documents, then turn it on for the live flow.

## 10. Maintenance

```bash
docker logs --tail 100 dre        # journal
docker restart dre                # restart
docker stats --no-stream dre      # load
```

Updating to a new version:
```bash
cd ~/dre && git fetch --tags && git checkout NEW_TAG
docker build -t dre:NEW_VERSION .
docker run --rm --cpus=1 dre:NEW_VERSION selfcheck
docker rm -f dre
docker run -d --name dre --restart=unless-stopped --cpus=1 \
  -p 8080:8080 -e DRE_API_KEY="$(cat ~/.dre_key)" dre:NEW_VERSION
```

## Three rules

1. Do not move to the next step until the previous one has worked.
2. If the embedding does not work out — leave the old converter in place. A day without the module is better than a stopped flow.
3. For the first few days, selectively open the issued DOCX files by eye before sending them to clients (especially dates in faint handwriting and lines beneath signatures).

Details on the service parameters are in `docs/DEPLOY.md`.
