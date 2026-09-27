# Развёртывание DRE на сервере — по шагам

Порядок: сервер → модуль → проверка по сети → встраивание в программу.
**Не переходи к следующему шагу, пока предыдущий не сработал.**

Условные обозначения (подставь свои значения):
- `IP_СЕРВЕРА` — адрес нового сервера (Xeon, 128 ГБ);
- `IP_ВИРТУАЛКИ` — адрес виртуальной машины, где работает основная программа;
- `IP_ТВОЕГО_КОМПЬЮТЕРА` — адрес ноутбука, с которого тестируешь;
- `ЛОГИН` — логин GitHub, где лежит репозиторий `dre`.

---

## 1. Подключиться и проверить сервер

С ноутбука (PowerShell или Ubuntu):
```bash
ssh root@IP_СЕРВЕРА
```

На сервере:
```bash
cat /etc/os-release | head -3   # версия Ubuntu
nproc                           # число ядер
free -h                         # память
curl -sI https://github.com | head -1   # есть ли интернет: должно быть HTTP/2 200
```
Если интернета нет — дальше не пойдёт, сначала решить это с тем, кто выдал сервер.

## 2. Обновить систему и поставить Docker

```bash
apt update && apt upgrade -y
apt install -y git curl ufw
curl -fsSL https://get.docker.com | sh
docker run --rm hello-world
```
Должно появиться **Hello from Docker!**

## 3. Файрвол

```bash
ufw allow 22/tcp
ufw allow from IP_ВИРТУАЛКИ to any port 8080 proto tcp
ufw allow from IP_ТВОЕГО_КОМПЬЮТЕРА to any port 8080 proto tcp
ufw enable
ufw status
```
**Строку с `22` не пропускай** — иначе потеряешь доступ по SSH.

## 4. Забрать проект

```bash
git clone https://github.com/ЛОГИН/dre.git ~/dre
cd ~/dre
git checkout v0.2.1
```

## 5. Собрать и проверить

```bash
docker build -t dre:0.2.1 .
docker run --rm --cpus=1 dre:0.2.1 selfcheck
```
Сборка в первый раз — 5–15 минут. Самопроверка **должна закончиться `OK`**. Если нет — дальше не идти, сохранить вывод.

## 6. Ключ доступа и запуск сервиса

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.dre_key
chmod 600 ~/.dre_key
cat ~/.dre_key          # скопируй ключ себе — он нужен программе и тестовому скрипту

docker run -d --name dre --restart=unless-stopped --cpus=1 \
  -p 8080:8080 -e DRE_API_KEY="$(cat ~/.dre_key)" dre:0.2.1

docker ps               # контейнер dre должен быть Up
docker logs --tail 30 dre
```

`--restart=unless-stopped` — сервис сам поднимется после перезагрузки сервера.

## 7. Проверка на самом сервере

Положи на сервер любой PDF (с ноутбука: `scp test.pdf root@IP_СЕРВЕРА:~/`), затем:
```bash
curl -s -o test.docx -w "%{http_code}\n" \
  -H "Authorization: Bearer $(cat ~/.dre_key)" \
  --data-binary @test.pdf localhost:8080/convert
```
- `200` и появился `test.docx` — работает.
- Другой код — документ отклонён или ошибка: `cat test.docx` покажет причину, `docker logs --tail 50 dre` — подробности.

## 8. Проверка по сети с ноутбука

```bash
curl -s -o test.docx -w "%{http_code}\n" \
  -H "Authorization: Bearer КЛЮЧ" \
  --data-binary @test.pdf http://IP_СЕРВЕРА:8080/convert
```
Если зависает — файрвол (шаг 3) или сеть между ноутбуком и сервером.

Потом прогони папку документов скриптом `docs/test_remote.py` (инструкция внутри файла) и открой результаты в Word рядом с оригиналами.

## 9. Встраивание в основную программу

1. Скопируй `docs/dre_client.py` в проект основной программы.
2. Поставь библиотеку: `pip install requests`.
3. Задай переменные окружения на виртуалке:
   - `DRE_URL=http://IP_СЕРВЕРА:8080/convert`
   - `DRE_API_KEY=КЛЮЧ`
4. Найди место, где программа сейчас конвертирует PDF в DOCX (в VS Code: Ctrl+Shift+F, ищи `docx`, `convert`, `pdf`).
5. Замени вызов старого конвертера на:
   ```python
   from dre_client import convert_document

   docx_bytes = convert_document(path_to_pdf)
   if docx_bytes is None:
       # документ не сконвертирован: сделай то же, что программа делает сейчас с неудачными файлами
       ...
   else:
       with open(path_to_docx, "wb") as f:
           f.write(docx_bytes)
   ```
6. Проверь на 3–5 документах, потом включай на поток.

## 10. Обслуживание

```bash
docker logs --tail 100 dre        # журнал
docker restart dre                # перезапуск
docker stats --no-stream dre      # нагрузка
```

Обновление на новую версию:
```bash
cd ~/dre && git fetch --tags && git checkout НОВЫЙ_ТЕГ
docker build -t dre:НОВАЯ_ВЕРСИЯ .
docker run --rm --cpus=1 dre:НОВАЯ_ВЕРСИЯ selfcheck
docker rm -f dre
docker run -d --name dre --restart=unless-stopped --cpus=1 \
  -p 8080:8080 -e DRE_API_KEY="$(cat ~/.dre_key)" dre:НОВАЯ_ВЕРСИЯ
```

## Три правила

1. Не переходи к следующему шагу, пока предыдущий не сработал.
2. Если встраивание не получается — оставь старый конвертер. Лучше день без модуля, чем остановленный поток.
3. Первые дни выборочно открывай выданные DOCX глазами перед отправкой заказчикам (особенно даты бледной ручкой и строки под подписями).

Подробности по параметрам сервиса — в `docs/DEPLOY.md`.
