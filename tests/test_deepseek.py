"""Real pinned SDK request serialization against a local HTTP transport."""
import asyncio
from contextlib import redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from dsbot.chat.deepseek import DeepseekConfig, LlmError


def response(content='answer'):
    return {'id': 'test', 'object': 'chat.completion', 'created': 0,
            'model': 'deepseek-flash', 'choices': [
                {'index': 0, 'finish_reason': 'stop', 'message':
                    {'role': 'assistant', 'content': content}}]}


class DeepseekTests(unittest.IsolatedAsyncioTestCase):
    async def config(self, handler):
        config = DeepseekConfig()
        client = AsyncOpenAI(api_key='fake', base_url='https://api.deepseek.com',
                             max_retries=0, http_client=httpx.AsyncClient(
                                 transport=httpx.MockTransport(handler)))
        config.openai_client = client
        self.addAsyncCleanup(config.close)
        return config

    async def test_wire_roles_flash_and_disabled_thinking(self):
        received = []

        def handle(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json=response())

        config = await self.config(handle)
        messages = [config.system_message('guide'),
                    {'role': 'user', 'content': '[id=10, name=alice] question'},
                    {'role': 'assistant', 'content': 'prior answer'},
                    {'role': 'user', 'content': '[id=10, name=alice] again'}]
        original = [item.copy() for item in messages]
        self.assertEqual(await config.send_message(messages), 'answer')
        self.assertEqual(messages, original)
        self.assertEqual(received[0]['messages'], original)
        self.assertEqual(received[0]['model'], 'deepseek-flash')
        self.assertEqual(received[0]['thinking'], {'type': 'disabled'})
        self.assertEqual(received[0]['max_tokens'], 4096)
        self.assertFalse(received[0]['stream'])

    async def test_errors_are_sanitized_not_retried_and_client_closes(self):
        for status in (401, 429, 500):
            calls = []

            def fail(request):
                calls.append(request)
                return httpx.Response(status, json={'error': {'message': 'private'}})

            config = await self.config(fail)
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(LlmError) as error:
                await config.send_message([{'role': 'user', 'content': 'hi'}])
            self.assertEqual(len(calls), 1)
            self.assertNotIn('private', str(error.exception) + output.getvalue())
            await config.close()
            self.assertTrue(config.openai_client.is_closed())

    async def test_timeout_empty_response_and_cancellation(self):
        def timeout(request):
            raise httpx.ReadTimeout('private', request=request)

        config = await self.config(timeout)
        with self.assertRaises(LlmError):
            await config.send_message([])
        for content in (None, '', '  '):
            config = await self.config(lambda request: httpx.Response(200, json=response(content)))
            with self.assertRaises(LlmError):
                await config.send_message([])

        entered = asyncio.Event()

        async def wait(request):
            entered.set()
            await asyncio.Future()

        config = await self.config(wait)
        pending = asyncio.create_task(config.send_message([]))
        await asyncio.wait_for(entered.wait(), 2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_factory_sets_timeout_and_disables_retry_without_network(self):
        config = DeepseekConfig()
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}):
            client = config.create_openai_client()
        try:
            self.assertIsInstance(client, AsyncOpenAI)
            self.assertEqual(client.timeout, 60)
            self.assertEqual(client.max_retries, 0)
        finally:
            await config.close()
