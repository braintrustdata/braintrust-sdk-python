import asyncio
import unittest

from braintrust.devserver.cors import check_origin, create_cors_middleware


class TestCorsOriginValidation(unittest.TestCase):
    def test_check_origin_allows_legitimate_preview_origin(self):
        self.assertTrue(check_origin("https://legit.preview.braintrust.dev"))

    def test_check_origin_rejects_preview_suffix_bypass(self):
        self.assertFalse(check_origin("https://evil.preview.braintrust.dev.attacker.com"))

    def test_options_response_does_not_reflect_disallowed_origin(self):
        async def app(scope, receive, send):
            raise AssertionError("OPTIONS requests should be handled by the CORS middleware")

        middleware = create_cors_middleware()(app)
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "method": "OPTIONS",
            "headers": [
                (b"origin", b"https://evil.preview.braintrust.dev.attacker.com"),
                (b"access-control-request-method", b"POST"),
            ],
        }

        asyncio.run(middleware(scope, receive, send))

        response_start = next(message for message in messages if message["type"] == "http.response.start")
        headers = dict(response_start["headers"])
        self.assertNotIn(b"access-control-allow-origin", headers)
        self.assertNotIn(b"access-control-allow-credentials", headers)


if __name__ == "__main__":
    unittest.main()
