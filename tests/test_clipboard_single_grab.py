"""Pasting an image grabs the clipboard once, off the event loop.

Upstream asks the native clipboard for its types (cheap) before reading the image. The Python
stand-in answered ``has_image`` by grabbing the whole image synchronously on the event loop,
then grabbed it a second time to read it.
"""
import asyncio
import threading

from misaka.utils import clipboard_image


class _Clipboard:
    def __init__(self, data):
        self.data = data
        self.grabs = []

    def has_image(self):
        raise AssertionError("has_image must not be consulted: it grabs the image on the event loop")

    async def get_image_binary(self):
        def grab():
            self.grabs.append(threading.current_thread() is threading.main_thread())
            return self.data
        return await asyncio.to_thread(grab)


async def test_an_image_is_grabbed_once_off_the_loop(monkeypatch):
    fake = _Clipboard(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(clipboard_image, "_get_native_clipboard", lambda: fake)
    image = await clipboard_image.read_clipboard_image_via_native_clipboard()
    assert image is not None and image.bytes.startswith(b"\x89PNG")
    assert fake.grabs == [False]


async def test_an_empty_clipboard_reads_as_no_image(monkeypatch):
    fake = _Clipboard(None)
    monkeypatch.setattr(clipboard_image, "_get_native_clipboard", lambda: fake)
    assert not await clipboard_image.read_clipboard_image_via_native_clipboard()
    assert fake.grabs == [False]
