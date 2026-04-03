"""ChatGPT 注册模式适配器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from curl_cffi import requests as cffi_requests

from core.base_platform import Account, AccountStatus
from core.proxy_utils import build_requests_proxy_config

CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN = "refresh_token"
CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY = "access_token_only"
DEFAULT_CHATGPT_REGISTRATION_MODE = CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN


def normalize_chatgpt_registration_mode(value) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {
        CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
        "access_token",
        "at_only",
        "without_rt",
        "without_refresh_token",
        "no_rt",
        "0",
        "false",
    }:
        return CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY
    if normalized in {
        CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN,
        "rt",
        "with_rt",
        "has_rt",
        "1",
        "true",
    }:
        return CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN
    return DEFAULT_CHATGPT_REGISTRATION_MODE


def resolve_chatgpt_registration_mode(extra: Optional[dict]) -> str:
    extra = extra or {}
    if "chatgpt_registration_mode" in extra:
        return normalize_chatgpt_registration_mode(extra.get("chatgpt_registration_mode"))
    if "chatgpt_has_refresh_token_solution" in extra:
        return (
            CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN
            if bool(extra.get("chatgpt_has_refresh_token_solution"))
            else CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY
        )
    return DEFAULT_CHATGPT_REGISTRATION_MODE


@dataclass(frozen=True)
class ChatGPTRegistrationContext:
    email_service: object
    proxy_url: Optional[str]
    callback_logger: Callable[[str], None]
    email: Optional[str]
    password: Optional[str]
    browser_mode: str
    max_retries: int
    extra_config: dict


@dataclass
class ChatGPTRegistrationAdapterResult:
    success: bool = False
    email: str = ""
    password: str = ""
    account_id: str = ""
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    session_token: str = ""
    workspace_id: str = ""
    error_message: str = ""
    logs: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source: str = "register"


class BaseChatGPTRegistrationModeAdapter(ABC):
    mode: str

    @abstractmethod
    def _create_engine(self, context: ChatGPTRegistrationContext):
        """按模式构造底层注册引擎。"""

    def _emit_log(self, context: ChatGPTRegistrationContext, logs: list, message: str):
        logs.append(message)
        if context.callback_logger:
            context.callback_logger(message)

    def _probe_connectivity(self, url: str, proxy_url: Optional[str]) -> tuple[bool, str]:
        try:
            response = cffi_requests.get(
                url,
                proxies=build_requests_proxy_config(proxy_url),
                impersonate="chrome",
                allow_redirects=True,
                timeout=15,
            )
            detail = f"HTTP {response.status_code}"
            return 200 <= response.status_code < 400, detail
        except Exception as e:
            return False, str(e)

    def _precheck_target(
        self,
        context: ChatGPTRegistrationContext,
        logs: list,
        *,
        label: str,
        url: str,
    ) -> tuple[bool, str]:
        ok, detail = self._probe_connectivity(url, context.proxy_url)
        self._emit_log(
            context,
            logs,
            f"[Precheck] {label}: {'通过' if ok else '失败'} ({detail})",
        )
        return ok, detail

    def _prepend_logs(self, result, extra_logs: list):
        if result is None or not extra_logs:
            return result
        existing_logs = list(getattr(result, "logs", None) or [])
        setattr(result, "logs", list(extra_logs) + existing_logs)
        return result

    def _failure_result(
        self,
        *,
        error_message: str,
        logs: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> ChatGPTRegistrationAdapterResult:
        return ChatGPTRegistrationAdapterResult(
            success=False,
            error_message=error_message,
            logs=list(logs or []),
            metadata=dict(metadata or {}),
        )

    def _run_engine(self, context: ChatGPTRegistrationContext):
        engine = self._create_engine(context)
        if context.email is not None:
            engine.email = context.email
        if context.password is not None:
            engine.password = context.password
        return engine.run()

    def _attach_result_mode(self, result, mode: str, fallback_from: Optional[str] = None):
        if result is None:
            return None
        setattr(result, "_chatgpt_registration_mode", mode)
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            metadata = {}
            setattr(result, "metadata", metadata)
        if isinstance(metadata, dict):
            metadata["chatgpt_registration_mode"] = mode
            if fallback_from:
                metadata["chatgpt_registration_fallback_from"] = fallback_from
        return result

    def run(self, context: ChatGPTRegistrationContext):
        return self._attach_result_mode(self._run_engine(context), self.mode)

    def build_account(self, result, fallback_password: str) -> Account:
        actual_mode = getattr(result, "_chatgpt_registration_mode", self.mode)
        return Account(
            platform="chatgpt",
            email=getattr(result, "email", ""),
            password=getattr(result, "password", "") or fallback_password,
            user_id=getattr(result, "account_id", ""),
            token=getattr(result, "access_token", ""),
            status=AccountStatus.REGISTERED,
            extra=self._build_account_extra(result, actual_mode),
        )

    def _build_account_extra(self, result, actual_mode: str) -> dict:
        return {
            "access_token": getattr(result, "access_token", ""),
            "refresh_token": getattr(result, "refresh_token", ""),
            "id_token": getattr(result, "id_token", ""),
            "session_token": getattr(result, "session_token", ""),
            "workspace_id": getattr(result, "workspace_id", ""),
            "chatgpt_registration_mode": actual_mode,
            "chatgpt_has_refresh_token_solution": actual_mode == CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN,
            "chatgpt_token_source": getattr(result, "source", "register"),
        }


class RefreshTokenChatGPTRegistrationAdapter(BaseChatGPTRegistrationModeAdapter):
    mode = CHATGPT_REGISTRATION_MODE_REFRESH_TOKEN

    def _create_engine(self, context: ChatGPTRegistrationContext):
        from platforms.chatgpt.refresh_token_registration_engine import RefreshTokenRegistrationEngine

        return RefreshTokenRegistrationEngine(
            email_service=context.email_service,
            proxy_url=context.proxy_url,
            callback_logger=context.callback_logger,
            browser_mode=context.browser_mode,
        )

    def _should_fallback_to_access_token_only(self, result) -> bool:
        if not result or getattr(result, "success", False):
            return False

        parts = [str(getattr(result, "error_message", "") or "")]
        logs = getattr(result, "logs", None) or []
        parts.extend(str(item or "") for item in logs)
        text = "\n".join(parts).lower()

        return any(
            marker in text
            for marker in (
                "获取 device id 失败",
                "建立 oauth 会话返回 http 403",
                "oauth 授权页: http 403",
                "oauth 会话兜底接口: http 403",
                "预授权被拦截",
            )
        )

    def run(self, context: ChatGPTRegistrationContext):
        precheck_logs = []
        self._emit_log(context, precheck_logs, "[Precheck] 开始注册前连通性预检查...")
        auth_ok, auth_detail = self._precheck_target(
            context,
            precheck_logs,
            label="auth.openai.com",
            url="https://auth.openai.com/",
        )
        chatgpt_ok, chatgpt_detail = self._precheck_target(
            context,
            precheck_logs,
            label="chatgpt.com",
            url="https://chatgpt.com/",
        )

        if not auth_ok and not chatgpt_ok:
            return self._attach_result_mode(
                self._failure_result(
                    error_message=(
                        "注册前连通性预检查失败: "
                        f"auth.openai.com 不可用 ({auth_detail}); "
                        f"chatgpt.com 不可用 ({chatgpt_detail})"
                    ),
                    logs=precheck_logs,
                    metadata={
                        "precheck": {
                            "auth.openai.com": auth_detail,
                            "chatgpt.com": chatgpt_detail,
                        }
                    },
                ),
                self.mode,
            )

        if not auth_ok:
            self._emit_log(
                context,
                precheck_logs,
                "[Precheck] auth.openai.com 当前不可用，跳过 Refresh Token 模式，直接进入 Access Token Only",
            )
            fallback_adapter = AccessTokenOnlyChatGPTRegistrationAdapter()
            fallback_result = fallback_adapter.run(context)
            fallback_result = self._prepend_logs(fallback_result, precheck_logs)
            return self._attach_result_mode(
                fallback_result,
                CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
                fallback_from=self.mode,
            )

        if not chatgpt_ok:
            self._emit_log(
                context,
                precheck_logs,
                "[Precheck] chatgpt.com 当前不可用，Access Token Only 降级路径可能失败；继续尝试 Refresh Token 模式",
            )

        primary_result = self._attach_result_mode(
            self._run_engine(context),
            self.mode,
        )
        primary_result = self._prepend_logs(primary_result, precheck_logs)
        if not self._should_fallback_to_access_token_only(primary_result):
            return primary_result

        if context.callback_logger:
            context.callback_logger(
                "[AutoFallback] Refresh Token 模式获取 Device ID / OAuth 会话失败，自动降级到 Access Token Only 重试"
            )

        fallback_adapter = AccessTokenOnlyChatGPTRegistrationAdapter()
        fallback_result = fallback_adapter.run(context)
        fallback_result = self._prepend_logs(fallback_result, precheck_logs)
        fallback_result = self._attach_result_mode(
            fallback_result,
            CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY,
            fallback_from=self.mode,
        )

        if fallback_result and getattr(fallback_result, "success", False):
            return fallback_result

        primary_error = str(getattr(primary_result, "error_message", "") or "").strip()
        fallback_error = str(getattr(fallback_result, "error_message", "") or "").strip()
        combined_error = (
            f"Refresh Token 模式失败: {primary_error or '未知错误'}; "
            f"自动降级 Access Token Only 后仍失败: {fallback_error or '未知错误'}"
        )
        if fallback_result is not None:
            setattr(fallback_result, "error_message", combined_error)
            primary_logs = list(getattr(primary_result, "logs", None) or [])
            fallback_logs = list(getattr(fallback_result, "logs", None) or [])
            setattr(fallback_result, "logs", primary_logs + fallback_logs)
            return fallback_result
        return primary_result


class AccessTokenOnlyChatGPTRegistrationAdapter(BaseChatGPTRegistrationModeAdapter):
    mode = CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY

    def _create_engine(self, context: ChatGPTRegistrationContext):
        from platforms.chatgpt.access_token_only_registration_engine import AccessTokenOnlyRegistrationEngine

        return AccessTokenOnlyRegistrationEngine(
            email_service=context.email_service,
            proxy_url=context.proxy_url,
            browser_mode=context.browser_mode,
            callback_logger=context.callback_logger,
            max_retries=context.max_retries,
            extra_config=context.extra_config,
        )

    def run(self, context: ChatGPTRegistrationContext):
        precheck_logs = []
        self._emit_log(context, precheck_logs, "[Precheck] 开始注册前连通性预检查...")
        chatgpt_ok, chatgpt_detail = self._precheck_target(
            context,
            precheck_logs,
            label="chatgpt.com",
            url="https://chatgpt.com/",
        )
        if not chatgpt_ok:
            return self._attach_result_mode(
                self._failure_result(
                    error_message=(
                        "注册前连通性预检查失败: "
                        f"chatgpt.com 不可用 ({chatgpt_detail})"
                    ),
                    logs=precheck_logs,
                    metadata={"precheck": {"chatgpt.com": chatgpt_detail}},
                ),
                self.mode,
            )

        result = self._attach_result_mode(self._run_engine(context), self.mode)
        return self._prepend_logs(result, precheck_logs)


def build_chatgpt_registration_mode_adapter(
    extra: Optional[dict],
) -> BaseChatGPTRegistrationModeAdapter:
    mode = resolve_chatgpt_registration_mode(extra)
    if mode == CHATGPT_REGISTRATION_MODE_ACCESS_TOKEN_ONLY:
        return AccessTokenOnlyChatGPTRegistrationAdapter()
    return RefreshTokenChatGPTRegistrationAdapter()
