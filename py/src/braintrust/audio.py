import struct


_CRC_TABLE = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) if _r & 0x80000000 else (_r << 1)
    _CRC_TABLE.append(_r & 0xFFFFFFFF)


def _crc(page: bytes) -> int:
    crc = 0
    for byte in page:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) ^ byte) & 0xFF]
    return crc


def _pages(stream: bytes) -> list[bytearray]:
    pages = []
    pos = 0
    while pos < len(stream):
        if stream[pos : pos + 4] != b"OggS":
            raise ValueError(f"Malformed Ogg page at byte {pos}")
        body = pos + 27 + (stream[pos + 26] if pos + 26 < len(stream) else 0)
        end = body + sum(stream[pos + 27 : body])
        if body > len(stream) or end > len(stream):
            raise ValueError(f"Truncated Ogg page at byte {pos}")
        pages.append(bytearray(stream[pos:end]))
        pos = end
    return pages


def _page_body(page: "bytearray | None") -> bytes:
    return bytes(page[27 + page[26] :]) if page else b""


def join_ogg_streams(streams: list[bytes]) -> bytes:
    """Joins the SDK's Ogg Opus recording segments into one logical stream, like
    `ffmpeg -f concat -c copy`. Appending the files gives a chained Ogg instead, whose
    probed duration is only the last segment's."""
    pages = []
    offset = 0
    for index, stream in enumerate(streams):
        stream_pages = _pages(stream)
        head, tags = (stream_pages + [None, None])[:2]
        if _page_body(head)[:8] != b"OpusHead" or _page_body(tags)[:8] != b"OpusTags":
            raise ValueError(f"Ogg stream {index} doesn't start with Opus headers")
        if tags[26 + tags[26]] == 255:
            raise ValueError(f"Ogg stream {index} has OpusTags over more than one page")
        first_head = _page_body(pages[0]) if pages else None
        if first_head and (_page_body(head)[9] != first_head[9] or _page_body(head)[18] != first_head[18]):
            raise ValueError(f"Ogg stream {index} has a different channel layout")
        if index == 0:
            pages += [head, tags]
        last_granule = 0
        for page in stream_pages[2:]:
            granule = struct.unpack_from("<Q", page, 6)[0]
            if granule != 2**64 - 1:  # -1 means no packet ends on this page
                last_granule = granule
                struct.pack_into("<Q", page, 6, offset + last_granule)
            pages.append(page)
        # Every 960-sample packet is kept, so the next stream starts after this one's padding.
        offset += -(-last_granule // 960) * 960
        if index < len(streams) - 1:
            struct.pack_into("<Q", pages[-1], 6, offset)

    serial = struct.unpack_from("<I", pages[0], 14)[0]
    for sequence, page in enumerate(pages):
        page[5] = (page[5] & 0x01) | (0x02 if sequence == 0 else 0) | (0x04 if sequence == len(pages) - 1 else 0)
        struct.pack_into("<III", page, 14, serial, sequence, 0)
        struct.pack_into("<I", page, 22, _crc(page))
    return b"".join(pages)
