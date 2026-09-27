#!/bin/sh
# Container commands: serve (default) | selfcheck | batch IN OUT | summary OUT | convert FILE --output OUT.docx
case "$1" in
  selfcheck) shift; exec python -m document_reconstruction selfcheck "$@" ;;
  batch) shift; exec python -m document_reconstruction batch "$@" ;;
  summary) shift; exec python -m document_reconstruction summary "$@" ;;
  convert) shift; exec python -m document_reconstruction "$@" ;;
  serve|"") exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 60 --graceful-timeout 55 --keep-alive 2 \
       document_reconstruction.service.wsgi:application ;;
  *) exec "$@" ;;
esac
