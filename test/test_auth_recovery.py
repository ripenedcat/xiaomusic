import asyncio
import logging
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from xiaomusic.auth import AuthManager
from xiaomusic.device_manager import DeviceManager
from xiaomusic.xiaomusic import XiaoMusic


class AuthRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = SimpleNamespace(
            conf_path=self.directory.name,
            account="test-account",
            password="test-password",
            cookie="",
            mi_did="",
            devices={},
            enable_pull_ask=False,
            enable_file_watch=False,
        )
        self.devices = DeviceManager(self.config, logging.getLogger(__name__))
        self.devices._update_devices = Mock()
        self.auth = AuthManager(self.config, logging.getLogger(__name__), self.devices)
        self.addAsyncCleanup(self.close_session)
        self.auth.get_cookie = Mock(return_value={"serviceToken": "test-token"})
        self.service = SimpleNamespace(device_list=AsyncMock(return_value=[]))

    async def close_session(self):
        await self.auth.mi_session.close()

    async def successful_login(self):
        self.auth.mina_service = self.service
        self.auth.login_acount = self.config.account
        self.auth.login_password = self.config.password
        return True

    async def test_initial_success_does_not_sleep(self):
        self.auth.login_miboy = AsyncMock(side_effect=self.successful_login)
        with patch("xiaomusic.auth.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await self.auth.init_until_ready()
        sleep.assert_not_awaited()
        self.auth.login_miboy.assert_awaited_once()
        self.assertIs(self.auth.cookie_jar, self.auth.mi_session.cookie_jar)

    async def test_repeated_login_failures_recover_with_capped_backoff(self):
        attempts = 0

        async def login():
            nonlocal attempts
            attempts += 1
            if attempts <= 7:
                return False
            return await self.successful_login()

        self.auth.login_miboy = AsyncMock(side_effect=login)
        with patch("xiaomusic.auth.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await self.auth.init_until_ready()
        self.assertEqual(
            [call.args[0] for call in sleep.await_args_list],
            [5, 10, 20, 40, 60, 60, 60],
        )
        self.assertEqual(attempts, 8)
        self.devices._update_devices.assert_called_once()
        self.assertIsNotNone(self.auth.cookie_jar)

    async def test_dns_login_exception_recovers_without_fresh_credentials(self):
        account = SimpleNamespace(
            token={},
            mi_request=AsyncMock(),
            login=AsyncMock(side_effect=[socket.gaierror(-3, "Try again"), True]),
        )
        with (
            patch("xiaomusic.auth.MiAccount", return_value=account),
            patch("xiaomusic.auth.MiNAService", return_value=self.service),
            patch("xiaomusic.auth.MiIOService"),
            patch("xiaomusic.auth.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            await self.auth.init_until_ready()
        self.assertEqual(account.login.await_count, 2)
        sleep.assert_awaited_once_with(5)
        self.assertIsNotNone(self.auth.cookie_jar)

    async def test_device_network_failure_is_not_an_empty_success(self):
        self.auth.login_miboy = AsyncMock(side_effect=self.successful_login)
        self.service.device_list.side_effect = socket.gaierror(-3, "Try again")
        self.assertFalse(await self.auth.init_all_data())
        self.devices._update_devices.assert_not_called()
        self.assertIsNone(self.auth.cookie_jar)
        self.service.device_list.side_effect = None
        self.assertTrue(await self.auth.init_all_data())
        self.devices._update_devices.assert_called_once()
        self.auth.login_miboy.assert_awaited_once()

    async def test_cookie_failure_is_not_ready(self):
        self.auth.login_miboy = AsyncMock(side_effect=self.successful_login)
        self.auth.get_cookie.return_value = None
        self.assertFalse(await self.auth.init_all_data())
        self.auth.get_cookie.return_value = {"serviceToken": "test-token"}
        self.assertTrue(await self.auth.init_all_data())

    async def test_missing_credentials_do_not_refresh_devices(self):
        self.auth.can_login = AsyncMock(return_value=False)
        self.auth.login_miboy = AsyncMock()
        self.assertFalse(await self.auth.init_all_data())
        self.auth.login_miboy.assert_not_awaited()
        self.devices._update_devices.assert_not_called()

    async def test_timeout_returns_failure_and_releases_lock(self):
        entered = asyncio.Event()

        async def blocked_login():
            entered.set()
            await asyncio.Event().wait()

        self.auth.login_miboy = AsyncMock(side_effect=blocked_login)
        with patch("xiaomusic.auth.INIT_LOCK_TIMEOUT_SEC", 0.02):
            self.assertFalse(await self.auth.init_all_data())
        self.assertTrue(entered.is_set())
        self.assertFalse(self.auth._init_lock.locked())
        self.auth.login_miboy.side_effect = self.successful_login
        self.assertTrue(await self.auth.init_all_data())

    async def test_concurrent_initialization_does_not_duplicate_login(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def login():
            entered.set()
            await release.wait()
            return await self.successful_login()

        self.auth.login_miboy = AsyncMock(side_effect=login)
        startup = asyncio.create_task(self.auth.init_until_ready())
        await entered.wait()
        manual = asyncio.create_task(self.auth.init_all_data())
        release.set()
        try:
            self.assertEqual(await asyncio.gather(startup, manual), [None, True])
        finally:
            for task in (startup, manual):
                task.cancel()
            await asyncio.gather(startup, manual, return_exceptions=True)
        self.auth.login_miboy.assert_awaited_once()

    async def test_cancellation_during_login_releases_lock(self):
        entered = asyncio.Event()

        async def login():
            entered.set()
            await asyncio.Event().wait()

        self.auth.login_miboy = AsyncMock(side_effect=login)
        task = asyncio.create_task(self.auth.init_until_ready())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.auth._init_lock.locked())
        self.assertIsNone(self.auth.cookie_jar)

    async def test_cancellation_during_backoff_stops_retries(self):
        sleeping = asyncio.Event()

        async def sleep(_delay):
            sleeping.set()
            await asyncio.Event().wait()

        self.auth.login_miboy = AsyncMock(return_value=False)
        with patch("xiaomusic.auth.asyncio.sleep", side_effect=sleep):
            task = asyncio.create_task(self.auth.init_until_ready())
            await sleeping.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.auth.login_miboy.assert_awaited_once()

    async def test_fresh_session_is_closed_on_error_and_cancellation(self):
        for error in (socket.gaierror(-3, "Try again"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                session = SimpleNamespace(close=AsyncMock())
                account = SimpleNamespace(token={}, login=AsyncMock(side_effect=error))
                with (
                    patch("xiaomusic.auth.ClientSession", return_value=session),
                    patch("xiaomusic.auth.MiAccount", return_value=account),
                    patch.object(self.auth, "set_token"),
                ):
                    if isinstance(error, asyncio.CancelledError):
                        with self.assertRaises(asyncio.CancelledError):
                            await self.auth._try_fresh_session_and_relogin(account)
                    else:
                        self.assertFalse(
                            await self.auth._try_fresh_session_and_relogin(account)
                        )
                session.close.assert_awaited_once()

    async def test_startup_recovers_with_polling_disabled_and_closes_session(self):
        app = XiaoMusic.__new__(XiaoMusic)
        app.config = self.config
        app.log = logging.getLogger(__name__)
        app.auth_manager = self.auth
        app.music_library = SimpleNamespace(try_gen_all_music_tag=Mock())
        app.crontab = SimpleNamespace(start=Mock())
        app.analytics = SimpleNamespace(send_startup_event=AsyncMock())
        app.stop_file_watch = Mock()
        app.analytics_task_daily = AsyncMock()
        ready = asyncio.Event()
        sleeping = asyncio.Event()
        retry = asyncio.Event()

        async def sleep(_delay):
            sleeping.set()
            await retry.wait()

        async def conversation(*_callbacks):
            self.assertIsNotNone(self.auth.cookie_jar)
            self.assertFalse(self.config.enable_pull_ask)
            ready.set()
            await asyncio.Event().wait()

        app.conversation_poller = SimpleNamespace(
            run_conversation_loop=AsyncMock(side_effect=conversation)
        )
        self.auth.login_miboy = AsyncMock(return_value=False)
        with patch("xiaomusic.auth.asyncio.sleep", side_effect=sleep):
            task = asyncio.create_task(app.run_forever())
            try:
                await asyncio.wait_for(sleeping.wait(), 2)
                app.conversation_poller.run_conversation_loop.assert_not_awaited()
                self.auth.login_miboy.side_effect = self.successful_login
                retry.set()
                await asyncio.wait_for(ready.wait(), 2)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertTrue(self.auth.mi_session.closed)
        app.stop_file_watch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
