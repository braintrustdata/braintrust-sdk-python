"""Prepare encoded recordings for the standard attachment exporter."""

import asyncio

from braintrust.logger import Attachment

from .worker import await_background


def prepare_recording(name, function, *args):
    encoded = function(*args)
    if encoded is not None:
        encoded["attachment"] = Attachment(
            data=encoded["data"],
            filename=f"{name}.{encoded['extension']}",
            content_type=encoded["mime_type"],
        )
    return encoded


class UploadFailed(RuntimeError):
    """The attachment did not reach confirmed upload completion."""


async def upload_recording(encoded):
    def upload():
        try:
            status = encoded["attachment"].upload()
        except Exception as error:
            raise UploadFailed("upload_failed") from error
        if status.get("upload_status") != "done":
            raise UploadFailed("upload_failed")

    await await_background(asyncio.to_thread(upload))
