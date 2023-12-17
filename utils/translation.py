import urllib

import keys
import json

naver_client_id = keys.naver_client_id
naver_client_secret = keys.naver_client_secret


def translate_text(target_lang, text):
    # detect language
    encQuery = urllib.parse.quote(text)
    data = "query=" + encQuery
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", naver_client_id)
    request.add_header("X-Naver-Client-Secret", naver_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if rescode == 200:
        response_body = json.loads(response.read().decode("utf-8"))
        print("source language " + (response_body["langCode"]))
        source_lang = response_body["langCode"]
    else:
        print("Error detecting language:" + rescode)
        return text
    # translate to target language
    if target_lang == source_lang:
        return text
    encText = urllib.parse.quote(text)
    data = f"source={source_lang}&target={target_lang}&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", naver_client_id)
    request.add_header("X-Naver-Client-Secret", naver_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if rescode == 200:
        response_body = json.loads(response.read().decode("utf-8"))
        print("translated to " + response_body["message"]["result"]["translatedText"])
        return response_body["message"]["result"]["translatedText"]
    else:
        print("Error Translating:" + rescode)
        return text
