#!/bin/bash

gcloud beta functions deploy handler --runtime python37 --trigger-http --memory 2048 --region asia-northeast1 \
                                     --project geultto