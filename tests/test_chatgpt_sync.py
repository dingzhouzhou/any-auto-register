import unittest
from unittest import mock

from core.db import AccountModel
from services.chatgpt_sync import backfill_chatgpt_account_to_cpa


class ChatGPTBackfillTests(unittest.TestCase):
    def _make_account(self) -> AccountModel:
        account = AccountModel(
            platform="chatgpt",
            email="demo@example.com",
            password="secret",
            token="access-token",
            status="registered",
        )
        account.set_extra(
            {
                "access_token": "access-token",
                "session_token": "session-token",
                "mail_provider": "cfworker",
            }
        )
        return account

    def test_backfill_skips_when_remote_auth_exists(self):
        account = self._make_account()

        with mock.patch(
            "services.cliproxyapi_sync.sync_chatgpt_cliproxyapi_status",
            return_value={
                "uploaded": True,
                "remote_state": "usable",
                "message": "",
            },
        ) as sync_mock:
            with mock.patch("platforms.chatgpt.status_probe.probe_local_chatgpt_status") as probe_mock:
                result = backfill_chatgpt_account_to_cpa(account, commit=False)

        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertFalse(result["uploaded"])
        self.assertIn("远端已存在", result["message"])
        self.assertEqual(sync_mock.call_count, 1)
        probe_mock.assert_not_called()

    def test_backfill_fails_when_local_probe_invalid(self):
        account = self._make_account()

        with mock.patch(
            "services.cliproxyapi_sync.sync_chatgpt_cliproxyapi_status",
            return_value={
                "uploaded": False,
                "remote_state": "not_found",
                "message": "未发现",
            },
        ):
            with mock.patch(
                "platforms.chatgpt.status_probe.probe_local_chatgpt_status",
                return_value={
                    "auth": {
                        "state": "access_token_invalidated",
                        "http_status": 401,
                        "error_code": "token_invalidated",
                        "message": "invalidated",
                    },
                    "codex": {"state": "skipped_auth_invalid"},
                },
            ):
                with mock.patch("services.chatgpt_sync.upload_account_model_to_cpa") as upload_mock:
                    result = backfill_chatgpt_account_to_cpa(account, commit=False)

        self.assertFalse(result["ok"])
        self.assertFalse(result["uploaded"])
        self.assertFalse(result["skipped"])
        self.assertEqual(account.status, "invalid")
        upload_mock.assert_not_called()

    def test_backfill_uploads_and_resyncs_when_remote_missing_and_local_valid(self):
        account = self._make_account()
        sync_results = [
            {
                "uploaded": False,
                "remote_state": "not_found",
                "message": "未发现",
            },
            {
                "uploaded": True,
                "remote_state": "usable",
                "message": "远端可用",
            },
        ]

        with mock.patch(
            "services.cliproxyapi_sync.sync_chatgpt_cliproxyapi_status",
            side_effect=sync_results,
        ) as sync_mock:
            with mock.patch(
                "platforms.chatgpt.status_probe.probe_local_chatgpt_status",
                return_value={
                    "auth": {
                        "state": "access_token_valid",
                        "http_status": 200,
                        "error_code": "",
                        "message": "ok",
                    },
                    "subscription": {"plan": "free"},
                    "codex": {"state": "usable"},
                },
            ):
                with mock.patch(
                    "services.chatgpt_sync.upload_account_model_to_cpa",
                    return_value=(True, "上传成功"),
                ) as upload_mock:
                    result = backfill_chatgpt_account_to_cpa(account, commit=False)

        self.assertTrue(result["ok"])
        self.assertTrue(result["uploaded"])
        self.assertFalse(result["skipped"])
        self.assertIn("补传完成", result["message"])
        self.assertEqual(sync_mock.call_count, 2)
        upload_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
