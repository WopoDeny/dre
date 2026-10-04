# Pilot: a folder of real documents on a new Ubuntu server

Goal: run real documents through, count the issued ones and the refusals by reason, and manually review every issued DOCX.
Server: Ubuntu 22.04 or newer, 1 core or more, 4 GB of RAM or more, internet not required.

## 1. Prepare the image (on the laptop, once)

```sh
cd ~/dre && docker build -t dre:1 .
docker save dre:1 | gzip > dre-1.tar.gz
scp dre-1.tar.gz user@server:~/
```

## 2. On the server (once)

```sh
sudo apt-get install -y docker.io                # if Docker is not installed yet
sudo usermod -aG docker $USER && newgrp docker
docker load < ~/dre-1.tar.gz
docker run --rm --cpus=1 dre:1 selfcheck     # the last line must be "OK"; note the speed.factor multiplier
```

If the self-check does not say "OK", send its full output.

## 3. Running the folder

```sh
mkdir -p ~/pilot/in ~/pilot/out
# copy the documents into ~/pilot/in (subfolders are fine): PDF, JPEG, PNG, WEBP, HEIC
docker run --rm --cpus=1 --user $(id -u):$(id -g) \
    -v ~/pilot:/data dre:1 batch /data/in /data/out
```

`--cpus=1` — same as on the target server, so the timing is honest. Long scans on a slow core may
get an `estimated_time_over_budget` refusal (not finishing within 45 s) — this is visible in the summary.
The run can be interrupted and started again: finished documents are skipped.

## 4. What you get

```
~/pilot/out/<name>.docx            issued documents
~/pilot/out/_refused/<name>.pdf    refusals: a copy of the source
~/pilot/out/_refused/<name>.json   refusal reason, time
~/pilot/out/journal.jsonl          one line per document
```

## 5. Quick count

```sh
docker run --rm -v ~/pilot:/data dre:1 summary /data/out
```

```
Documents: 120
Issued:    53 (44%)
Refused:   67 (56%)
    31  low_resolution
    22  unreliable_text
     9  unrecognized_text
     …
Time, s: median 3.8, max 21.4

To review (issued DOCX):
  order_12.docx    <- order_12.pdf
  …
```

The refusal reasons are explained in `DEPLOY.md`, section 7. The most common ones:
- `low_resolution` — letters smaller than 8 px, a scan below 300 dpi or a photo from too far away;
- `unreliable_text` — there are words read unreliably (in the refusal JSON, the `details.words` field).

## 6. Manual review of the issued DOCX

The main question of the pilot: is there even a single issued document with an error.
Open each DOCX in Word next to its source and mark it in the `~/pilot/review.csv` table:

```
file;result;what is wrong
order_12.docx;ok;
letter_3.docx;error;"2024" instead of "2021" in the date
letter_7.docx;layout;text is correct, the table shifted
```

- `ok` — text and layout are fine;
- `error` — a wrong word, digit, or a missing line (this is a "bad issue", the main thing we are looking for);
- `layout` — the text is correct, but the formatting is noticeably worse than the source.

A starter table with all the issued files:

```sh
(echo "file;result;what is wrong"; ls ~/pilot/out/*.docx | xargs -n1 basename | sed 's/$/;;/') > ~/pilot/review.csv
```

## 7. What to send back

1. `~/pilot/out/journal.jsonl` and the `summary` output;
2. `~/pilot/review.csv`;
3. the sources of all documents marked `error` and 5–10 refusals that, in your view, should have been issued
   (if the documents cannot be taken off-site — at least the JSON from `_refused/` and a description).
