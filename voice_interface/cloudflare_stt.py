"""Cloudflare Workers AI Whisper; credentials are supplied by the caller."""
import base64
from http.client import HTTPException
import json
import math
import re
import urllib.error
import urllib.request

from .transcription import QuotaExceeded, SerializedTranscriber, SttError, read_pcm


class CloudflareTranscriber(SerializedTranscriber):
    def __init__(self, account_id: str, token: str, timeout: float = 45):
        super().__init__()
        if (not re.fullmatch(r'[0-9a-fA-F]{32}', account_id) or not token
                or '\r' in token or '\n' in token):
            raise ValueError('Set a Cloudflare account ID and API token')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('STT timeout must be positive and finite')
        self._url = (f'https://api.cloudflare.com/client/v4/accounts/{account_id}'
                     '/ai/run/@cf/openai/whisper-large-v3-turbo')
        self._token = token
        self._timeout = timeout

    def _transcribe(self, wav: bytes) -> str:
        read_pcm(wav)
        body = json.dumps(dict(audio=base64.b64encode(wav).decode('ascii'),
                               language='ru', task='transcribe', beam_size=5,
                               vad_filter=False, condition_on_previous_text=True)).encode()
        request = urllib.request.Request(self._url, data=body, headers={
            'Authorization': f'Bearer {self._token}', 'Content-Type': 'application/json'},
            method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            codes = []
            try:
                failure = json.load(error)
                errors = failure.get('errors', []) if isinstance(failure, dict) else []
                codes = [item.get('code') for item in errors if isinstance(item, dict)]
            except (ValueError, TypeError, OSError, HTTPException):
                pass
            finally:
                error.close()
            if error.code == 429 and 3036 in codes:
                raise QuotaExceeded('Cloudflare daily STT quota exceeded') from None
            raise SttError(f'Cloudflare HTTP {error.code}') from None
        except (urllib.error.URLError, OSError, HTTPException):
            raise SttError('Cloudflare network/TLS/timeout failure') from None
        except ValueError:
            raise SttError('Invalid Cloudflare JSON response') from None
        if not isinstance(payload, dict) or payload.get('success') is not True:
            raise SttError('Cloudflare did not confirm success')
        result = payload.get('result')
        if not isinstance(result, dict):
            raise SttError('Invalid Cloudflare result')
        text = result.get('text')
        if not isinstance(text, str):
            info = result.get('transcription_info')
            text = info.get('text') if isinstance(info, dict) else None
        if not isinstance(text, str):
            raise SttError('Cloudflare returned no transcription text')
        return text.strip()
