import unittest
from unittest.mock import MagicMock, patch

import pilot


class MemoryReasoningTests(unittest.TestCase):
    def make_client(self):
        fake_client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "- Keep solutions minimal."
        response.system_fingerprint = "fp_test"
        fake_client.chat.completions.create.return_value = response
        return fake_client

    def test_groq_memory_request_disables_reasoning(self):
        fake_client = self.make_client()

        with (
            patch.object(
                pilot,
                "BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
            patch.object(pilot, "MODEL", "test-model"),
            patch.object(pilot, "client", return_value=fake_client),
        ):
            pilot.chat(
                [
                    {
                        "role": "system",
                        "content": pilot.MEMORY_SYSTEM,
                    }
                ],
                17,
            )

        kwargs = (
            fake_client.chat.completions.create.call_args.kwargs
        )
        self.assertEqual(
            kwargs.get("reasoning_effort"),
            "none",
        )

    def test_solver_request_keeps_default_reasoning(self):
        fake_client = self.make_client()

        with (
            patch.object(
                pilot,
                "BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
            patch.object(pilot, "MODEL", "test-model"),
            patch.object(pilot, "client", return_value=fake_client),
        ):
            pilot.chat(
                [
                    {
                        "role": "system",
                        "content": pilot.SOLVER_SYSTEM,
                    }
                ],
                17,
            )

        kwargs = (
            fake_client.chat.completions.create.call_args.kwargs
        )
        self.assertNotIn("reasoning_effort", kwargs)


if __name__ == "__main__":
    unittest.main()
