import unittest
from unittest import mock

from platforms.chatgpt.chatgpt_registration_mode_adapter import (
    AccessTokenOnlyChatGPTRegistrationAdapter,
    BaseChatGPTRegistrationModeAdapter,
    CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
    CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN,
    ChatGPTRegistrationContext,
    RefreshTokenChatGPTRegistrationAdapter,
    build_chatgpt_registration_mode_adapter,
    resolve_chatgpt_registration_mode,
)


class ChatGPTRegistrationModeAdapterTests(unittest.TestCase):
    def test_resolve_defaults_to_refresh_token_mode(self):
        self.assertEqual(
            resolve_chatgpt_registration_mode({}),
            CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN,
        )

    def test_resolve_supports_boolean_no_rt_flag(self):
        self.assertEqual(
            resolve_chatgpt_registration_mode(
                {"chatgpt_has_refresh_token_solution": False}
            ),
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )

    def test_build_account_marks_selected_mode(self):
        adapter = build_chatgpt_registration_mode_adapter(
            {"chatgpt_registration_mode": "access_token_only"}
        )
        result = type(
            "Result",
            (),
            {
                "email": "demo@example.com",
                "password": "pw",
                "account_id": "acct-demo",
                "access_token": "at-demo",
                "refresh_token": "",
                "id_token": "id-demo",
                "session_token": "session-demo",
                "workspace_id": "ws-demo",
                "source": "register",
            },
        )()

        account = adapter.build_account(result, fallback_password="fallback")

        self.assertEqual(account.email, "demo@example.com")
        self.assertEqual(account.password, "pw")
        self.assertEqual(
            account.extra["chatgpt_registration_mode"],
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )
        self.assertFalse(account.extra["chatgpt_has_refresh_token_solution"])

    def test_access_token_only_adapter_passes_runtime_context_to_engine(self):
        created = {}

        class FakeEngine:
            def __init__(self, **kwargs):
                created["kwargs"] = kwargs
                self.email = None
                self.password = None

            def run(self):
                created["email"] = self.email
                created["password"] = self.password
                return type("Result", (), {"success": True})()

        adapter = build_chatgpt_registration_mode_adapter(
            {"chatgpt_registration_mode": "access_token_only"}
        )
        context = ChatGPTRegistrationContext(
            email_service=object(),
            proxy_url="http://127.0.0.1:7890",
            callback_logger=lambda _msg: None,
            email="demo@example.com",
            password="pw-demo",
            browser_mode="headed",
            max_retries=5,
            extra_config={"register_max_retries": 5},
        )

        with mock.patch(
            "platforms.chatgpt.chatgpt_registration_mode_adapter.BaseChatGPTRegistrationModeAdapter._probe_connectivity",
            return_value=(True, "HTTP 200"),
        ), mock.patch(
            "platforms.chatgpt.access_token_only_registration_engine.AccessTokenOnlyRegistrationEngine",
            FakeEngine,
        ):
            adapter.run(context)

        self.assertEqual(created["email"], "demo@example.com")
        self.assertEqual(created["password"], "pw-demo")
        self.assertEqual(created["kwargs"]["browser_mode"], "headed")
        self.assertEqual(created["kwargs"]["max_retries"], 5)

    def test_refresh_token_adapter_auto_falls_back_to_access_token_only_on_oauth_403(self):
        created = {"refresh_runs": 0, "access_runs": 0}
        logs = []

        class FakeRefreshEngine:
            def __init__(self, **kwargs):
                created["refresh_kwargs"] = kwargs
                self.email = None
                self.password = None

            def run(self):
                created["refresh_runs"] += 1
                return type(
                    "Result",
                    (),
                    {
                        "success": False,
                        "error_message": "获取 Device ID 失败",
                        "logs": [
                            "[17:10:20] 获取 Device ID 失败: 建立 OAuth 会话返回 HTTP 403 (第 1/3 次)"
                        ],
                        "metadata": {},
                        "source": "register",
                    },
                )()

        class FakeAccessEngine:
            def __init__(self, **kwargs):
                created["access_kwargs"] = kwargs
                self.email = None
                self.password = None

            def run(self):
                created["access_runs"] += 1
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "email": "demo@example.com",
                        "password": "pw-demo",
                        "account_id": "acct-demo",
                        "access_token": "at-demo",
                        "refresh_token": "",
                        "id_token": "",
                        "session_token": "sess-demo",
                        "workspace_id": "ws-demo",
                        "error_message": "",
                        "logs": [],
                        "metadata": {},
                        "source": "register",
                    },
                )()

        adapter = build_chatgpt_registration_mode_adapter(
            {"chatgpt_registration_mode": "refresh_token"}
        )
        context = ChatGPTRegistrationContext(
            email_service=object(),
            proxy_url="http://127.0.0.1:7890",
            callback_logger=logs.append,
            email="demo@example.com",
            password="pw-demo",
            browser_mode="protocol",
            max_retries=3,
            extra_config={},
        )

        with mock.patch(
            "platforms.chatgpt.chatgpt_registration_mode_adapter.BaseChatGPTRegistrationModeAdapter._probe_connectivity",
            return_value=(True, "HTTP 200"),
        ), mock.patch(
            "platforms.chatgpt.refresh_token_registration_engine.RefreshTokenRegistrationEngine",
            FakeRefreshEngine,
        ), mock.patch(
            "platforms.chatgpt.access_token_only_registration_engine.AccessTokenOnlyRegistrationEngine",
            FakeAccessEngine,
        ):
            result = adapter.run(context)

        self.assertTrue(result.success)
        self.assertEqual(created["refresh_runs"], 1)
        self.assertEqual(created["access_runs"], 1)
        self.assertEqual(
            getattr(result, "_chatgpt_registration_mode"),
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )
        self.assertEqual(
            result.metadata["chatgpt_registration_fallback_from"],
            CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN,
        )
        self.assertTrue(any("AutoFallback" in line for line in logs))

        account = adapter.build_account(result, fallback_password="fallback")
        self.assertEqual(
            account.extra["chatgpt_registration_mode"],
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )
        self.assertFalse(account.extra["chatgpt_has_refresh_token_solution"])

    def test_refresh_token_precheck_skips_directly_to_access_token_only_when_auth_unreachable(self):
        created = {"refresh_runs": 0, "access_runs": 0}
        logs = []

        class FakeRefreshEngine:
            def __init__(self, **kwargs):
                self.email = None
                self.password = None

            def run(self):
                created["refresh_runs"] += 1
                return type("Result", (), {"success": True})()

        class FakeAccessEngine:
            def __init__(self, **kwargs):
                self.email = None
                self.password = None

            def run(self):
                created["access_runs"] += 1
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "email": "demo@example.com",
                        "password": "pw-demo",
                        "account_id": "acct-demo",
                        "access_token": "at-demo",
                        "refresh_token": "",
                        "id_token": "",
                        "session_token": "sess-demo",
                        "workspace_id": "ws-demo",
                        "error_message": "",
                        "logs": [],
                        "metadata": {},
                        "source": "register",
                    },
                )()

        probe_results = iter(
            [
                (False, "HTTP 403"),
                (True, "HTTP 200"),
                (True, "HTTP 200"),
            ]
        )
        adapter = build_chatgpt_registration_mode_adapter(
            {"chatgpt_registration_mode": "refresh_token"}
        )
        context = ChatGPTRegistrationContext(
            email_service=object(),
            proxy_url="http://127.0.0.1:7890",
            callback_logger=logs.append,
            email="demo@example.com",
            password="pw-demo",
            browser_mode="protocol",
            max_retries=3,
            extra_config={},
        )

        with mock.patch.object(
            BaseChatGPTRegistrationModeAdapter,
            "_probe_connectivity",
            side_effect=lambda *_args, **_kwargs: next(probe_results),
        ), mock.patch(
            "platforms.chatgpt.refresh_token_registration_engine.RefreshTokenRegistrationEngine",
            FakeRefreshEngine,
        ), mock.patch(
            "platforms.chatgpt.access_token_only_registration_engine.AccessTokenOnlyRegistrationEngine",
            FakeAccessEngine,
        ):
            result = adapter.run(context)

        self.assertTrue(result.success)
        self.assertEqual(created["refresh_runs"], 0)
        self.assertEqual(created["access_runs"], 1)
        self.assertEqual(
            getattr(result, "_chatgpt_registration_mode"),
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )
        self.assertTrue(any("跳过 Refresh Token 模式" in line for line in logs))

    def test_access_token_only_precheck_blocks_before_engine_when_chatgpt_unreachable(self):
        logs = []
        adapter = build_chatgpt_registration_mode_adapter(
            {"chatgpt_registration_mode": "access_token_only"}
        )
        context = ChatGPTRegistrationContext(
            email_service=object(),
            proxy_url="http://127.0.0.1:7890",
            callback_logger=logs.append,
            email="demo@example.com",
            password="pw-demo",
            browser_mode="protocol",
            max_retries=3,
            extra_config={},
        )

        with mock.patch.object(
            BaseChatGPTRegistrationModeAdapter,
            "_probe_connectivity",
            return_value=(False, "curl: (28) timeout"),
        ), mock.patch(
            "platforms.chatgpt.access_token_only_registration_engine.AccessTokenOnlyRegistrationEngine",
        ) as engine_cls:
            result = adapter.run(context)

        self.assertFalse(result.success)
        self.assertIn("chatgpt.com 不可用", result.error_message)
        self.assertEqual(
            getattr(result, "_chatgpt_registration_mode"),
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        )
        engine_cls.assert_not_called()
        self.assertTrue(any("chatgpt.com: 失败" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
