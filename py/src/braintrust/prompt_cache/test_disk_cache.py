import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

from braintrust import prompt
from braintrust.prompt_cache import disk_cache


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache = disk_cache.DiskCache[dict](
            cache_dir=self.cache_dir,
            max_size=3,
            log_warnings=False,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        except Exception:
            pass

    def test_failed_json_encoding_preserves_previous_entry(self):
        self.cache.set("key", {"version": 1})

        self.cache.set("key", {"unsupported": {1, 2}})

        self.assertEqual(self.cache.get("key"), {"version": 1})
        self.assertEqual(os.listdir(self.cache_dir), [os.path.basename(self.cache._get_entry_path("key"))])

    def test_failed_serializer_preserves_previous_entry(self):
        def serialize(value):
            if value["version"] == 2:
                raise ValueError("Cannot serialize this version")
            return value

        cache = disk_cache.DiskCache[dict](self.cache_dir, serializer=serialize, log_warnings=False)
        cache.set("key", {"version": 1})

        cache.set("key", {"version": 2})

        self.assertEqual(cache.get("key"), {"version": 1})
        self.assertEqual(os.listdir(self.cache_dir), [os.path.basename(cache._get_entry_path("key"))])

    def test_read_during_replacement_returns_complete_previous_entry(self):
        started = threading.Event()
        release = threading.Event()

        def serialize(value):
            if value == "new":
                started.set()
                assert release.wait(timeout=5)
            return value

        cache = disk_cache.DiskCache[str](self.cache_dir, serializer=serialize, log_warnings=False)
        cache.set("key", "old")
        with ThreadPoolExecutor(max_workers=1) as executor:
            write = executor.submit(cache.set, "key", "new")
            try:
                self.assertTrue(started.wait(timeout=5))
                self.assertEqual(cache.get("key"), "old")
            finally:
                release.set()
                write.result(timeout=5)

        self.assertEqual(cache.get("key"), "new")

    def test_eviction_does_not_remove_an_in_progress_write(self):
        started = threading.Event()
        release = threading.Event()

        def serialize(value):
            if value == "pending":
                started.set()
                assert release.wait(timeout=5)
            return value

        cache = disk_cache.DiskCache[str](self.cache_dir, max_size=1, serializer=serialize, log_warnings=False)
        cache.set("existing", "old")
        with ThreadPoolExecutor(max_workers=1) as executor:
            write = executor.submit(cache.set, "pending", "pending")
            try:
                self.assertTrue(started.wait(timeout=5))
                cache.set("existing", "updated")
                self.assertEqual(cache.get("existing"), "updated")
                # Avoid equal-mtime ties on filesystems with a coarse clock.
                old_time = time.time() - 60
                os.utime(cache._get_entry_path("existing"), (old_time, old_time))
            finally:
                release.set()
                write.result(timeout=5)

        self.assertEqual(cache.get("pending"), "pending")
        with self.assertRaises(KeyError):
            cache.get("existing")
        self.assertEqual(len(os.listdir(self.cache_dir)), 1)

    def test_retries_windows_sharing_violations(self):
        self.cache.set("key", {"version": 1})
        replace = os.replace
        sharing_violation = PermissionError("The file is in use")
        sharing_violation.winerror = 32
        calls = 0

        def replace_after_reader_closes(source, destination):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise sharing_violation
            return replace(source, destination)

        with patch.object(disk_cache.os, "replace", side_effect=replace_after_reader_closes):
            self.cache.set("key", {"version": 2})

        self.assertEqual(self.cache.get("key"), {"version": 2})
        self.assertEqual(calls, 2)
        self.assertEqual(len(os.listdir(self.cache_dir)), 1)

    def test_permanent_replace_error_preserves_previous_entry(self):
        self.cache.set("key", {"version": 1})
        with patch.object(disk_cache.os, "replace", side_effect=PermissionError("Permission denied")) as replace:
            self.cache.set("key", {"version": 2})

        self.assertEqual(self.cache.get("key"), {"version": 1})
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(len(os.listdir(self.cache_dir)), 1)

    @unittest.skipUnless(os.name == "nt", "Windows file sharing behavior")
    def test_replacement_overlaps_an_open_windows_reader(self):
        started = threading.Event()
        release = threading.Event()

        def deserialize(value):
            if value == "old":
                started.set()
                assert release.wait(timeout=5)
            return value

        cache = disk_cache.DiskCache[str](self.cache_dir, deserializer=deserialize, log_warnings=False)
        cache.set("key", "old")
        replace = os.replace

        def replace_and_release_reader(source, destination):
            try:
                return replace(source, destination)
            except PermissionError:
                release.set()
                raise

        with ThreadPoolExecutor(max_workers=1) as executor:
            read = executor.submit(cache.get, "key")
            try:
                self.assertTrue(started.wait(timeout=5))
                with patch.object(disk_cache.os, "replace", side_effect=replace_and_release_reader):
                    cache.set("key", "new")
            finally:
                release.set()
                self.assertEqual(read.result(timeout=5), "old")

        self.assertEqual(cache.get("key"), "new")

    def test_persistent_windows_sharing_violation_is_bounded(self):
        self.cache.set("key", {"version": 1})
        sharing_violation = PermissionError("The file is in use")
        sharing_violation.winerror = 32
        with patch.object(disk_cache.os, "replace", side_effect=sharing_violation) as replace:
            self.cache.set("key", {"version": 2})

        self.assertEqual(self.cache.get("key"), {"version": 1})
        self.assertEqual(replace.call_count, 5)
        self.assertEqual(len(os.listdir(self.cache_dir)), 1)

    def test_reaps_old_staging_files_left_by_an_interrupted_process(self):
        # Exit in the serializer, after the temporary gzip file has been opened.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sys; from braintrust.prompt_cache.disk_cache import DiskCache; "
                "DiskCache(sys.argv[1], serializer=lambda value: os._exit(0)).set('key', 'value')",
                self.cache_dir,
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        staging_files = os.listdir(self.cache_dir)
        self.assertEqual(len(staging_files), 1)
        orphan = os.path.join(self.cache_dir, staging_files[0])
        old_time = time.time() - 2 * 24 * 60 * 60
        os.utime(orphan, (old_time, old_time))
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=".braintrust-cache-", delete=False) as active:
            active_path = active.name

        # Cleanup also applies to unbounded caches and leaves recent writers alone.
        cache = disk_cache.DiskCache[str](self.cache_dir)
        cache.set("new", "value")

        self.assertEqual(cache.get("new"), "value")
        self.assertFalse(os.path.exists(orphan))
        self.assertTrue(os.path.exists(active_path))
        self.assertEqual(len(os.listdir(self.cache_dir)), 2)

    def test_keys_with_invalid_paths(self):
        data = {"1": "2"}
        weird_keys = [
            ".",
            "..",
            "a/b/c",
            "my\0file.txt",
            "file*.txT",
            "what?.txt",
            " asdf ",
            "invalid/name",
            "my<file>.txt",
            "a\nb",
        ]
        for k in weird_keys:
            time.sleep(0.01)  # make sure the mtimes are different
            self.cache.set(k, data)
            result = self.cache.get(k)
            assert data == result

    def test_store_and_retrieve_values(self):
        test_data = {"foo": "bar"}
        self.cache.set("test-key", test_data)
        result = self.cache.get("test-key")
        self.assertEqual(result, test_data)

    def test_raise_keyerror_for_missing_keys(self):
        """Test raising KeyError for missing keys."""
        with self.assertRaises(KeyError) as cm:
            self.cache.get("missing-key")
        self.assertEqual(str(cm.exception), "'Cache key not found: missing-key'")

    def test_raise_keyerror_after_eviction(self):
        """Test that accessing evicted entries raises KeyError."""
        # Fill cache beyond max size (3).
        for i in range(3):
            self.cache.set(f"key{i}", {"value": i})
            time.sleep(0.01)  # wait to ensure different mtimes

        # Add one more to trigger eviction.
        self.cache.set("key3", {"value": 3})

        # The oldest entry should raise KeyError.
        with self.assertRaises(KeyError) as cm:
            self.cache.get("key0")
        self.assertEqual(str(cm.exception), "'Cache key not found: key0'")

    def test_evict_oldest_entries_when_cache_is_full(self):
        # Fill cache beyond max size (3).
        for i in range(3):
            self.cache.set(f"key{i}", {"value": i})
            time.sleep(0.01)  # wait to ensure different mtimes

        # Add one more to trigger eviction.
        self.cache.set("key3", {"value": 3})

        # The oldest entry should be evicted.
        with self.assertRaises(KeyError) as cm:
            self.cache.get("key0")
        self.assertEqual(str(cm.exception), "'Cache key not found: key0'")

        # Newer entries should still exist.
        newer = self.cache.get("key2")
        self.assertEqual(newer, {"value": 2})

    def test_dont_throw_when_write_fails(self):
        # Make cache directory read-only.
        non_existent_dir = os.path.join(self.cache_dir, "non-existent-dir")
        # Don't throw when writing fails.
        broken_cache = disk_cache.DiskCache[dict](
            cache_dir=non_existent_dir,
            max_size=3,
            log_warnings=False,
            mkdirs=False,
        )
        broken_cache.set("test", {"foo": "bar"})

        try:
            broken_cache.get("test")
        except KeyError:
            pass

    def test_dont_throw_when_read_fails(self):
        self.cache.set("test-key", {"foo": "bar"})
        assert self.cache.get("test-key") == {"foo": "bar"}

        # Make cache directory unreadable.
        shutil.rmtree(self.cache_dir)

        try:
            self.cache.get("test-key")
        except KeyError:
            pass
        else:
            assert False, "shouldn't fail"

    def test_throw_on_corrupted_data(self):
        self.cache.set("test-key", {"foo": "bar"})
        assert self.cache.get("test-key") == {"foo": "bar"}

        # Corrupt the file.
        file_path = self.cache._get_entry_path("test-key")
        with open(file_path, "w") as f:
            f.write("invalid data")

        # if the data is corrupted, pretend like its not cached
        try:
            self.cache.get("test-key")
        except KeyError:
            pass
        else:
            assert 0

        # we should be able to write and read again.
        self.cache.set("test-key", {"foo": "bar"})
        assert self.cache.get("test-key") == {"foo": "bar"}

    def test_store_and_retrieve_with_serialization(self):
        """Test storing and retrieving objects using custom serialization."""
        cache = disk_cache.DiskCache[prompt.PromptSchema](
            cache_dir=self.cache_dir,
            max_size=3,
            serializer=lambda x: x.as_dict(),
            deserializer=prompt.PromptSchema.from_dict_deep,
            log_warnings=False,
        )

        # Create a test prompt.
        test_prompt = prompt.PromptSchema(
            id="456",
            project_id="123",
            _xact_id="789",
            name="test-prompt",
            slug="test-prompt",
            description=None,
            prompt_data=prompt.PromptData(),
            tags=None,
        )

        # Store and retrieve.
        cache.set("test-key", test_prompt)
        result = cache.get("test-key")

        # Should get back a PromptSchema instance.
        self.assertIsInstance(result, prompt.PromptSchema)
        self.assertEqual(result.as_dict(), test_prompt.as_dict())

    def test_serializer_handles_complex_objects(self):
        """Test that serializer is used for complex nested objects."""
        cache = disk_cache.DiskCache[prompt.PromptSchema](
            cache_dir=self.cache_dir,
            serializer=lambda x: x.as_dict(),
            deserializer=prompt.PromptSchema.from_dict_deep,
            log_warnings=False,
        )

        # Create a prompt with nested data.
        test_prompt = prompt.PromptSchema(
            id="456",
            project_id="123",
            _xact_id="789",
            name="test-prompt",
            slug="test-prompt",
            description="test description",
            prompt_data=prompt.PromptData(
                prompt=prompt.PromptCompletionBlock(
                    content="test",
                )
            ),
            tags=["tag1", "tag2"],
        )

        # Store and retrieve.
        cache.set("test-key", test_prompt)
        result = cache.get("test-key")

        self.assertEqual(result.as_dict(), test_prompt.as_dict())

    def test_throw_on_deserializer_error(self):
        """Test that deserializer errors are propagated."""

        def bad_deserializer(data: Any) -> prompt.PromptSchema:
            raise ValueError("Deserialization failed")

        cache = disk_cache.DiskCache[prompt.PromptSchema](
            cache_dir=self.cache_dir, serializer=lambda x: x.as_dict(), deserializer=bad_deserializer
        )

        # Store a prompt.
        test_prompt = prompt.PromptSchema(
            id="456",
            project_id="123",
            _xact_id="789",
            name="test-prompt",
            slug="test-prompt",
            description=None,
            prompt_data=prompt.PromptData(),
            tags=None,
        )
        cache.set("test-key", test_prompt)

        # assert a serde error is treated like a cache miss.
        try:
            cache.get("test-key")
        except KeyError:
            pass
        else:
            assert False, "should fail"
