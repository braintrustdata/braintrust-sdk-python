"""Caching for saved Braintrust parameters."""

from ..parameters import RemoteEvalParameters
from . import disk_cache, lru_cache, prompt_cache


class ParametersCache:
    def __init__(
        self,
        memory_cache: lru_cache.LRUCache[str, RemoteEvalParameters],
        disk_cache: disk_cache.DiskCache[RemoteEvalParameters] | None = None,
    ):
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache

    def get(
        self,
        slug: str | None = None,
        version: str = "latest",
        project_id: str | None = None,
        project_name: str | None = None,
        id: str | None = None,
    ) -> RemoteEvalParameters:
        cache_key = prompt_cache._create_cache_key(project_id, project_name, slug, version, id)

        try:
            return self.memory_cache.get(cache_key)
        except KeyError:
            pass

        if self.disk_cache:
            parameters = self.disk_cache.get(cache_key)
            if parameters is None:
                raise KeyError(f"Parameters not found in cache: {cache_key}")
            self.memory_cache.set(cache_key, parameters)
            return parameters

        raise KeyError(f"Parameters not found in cache: {cache_key}")

    def set(
        self,
        value: RemoteEvalParameters,
        slug: str | None = None,
        version: str = "latest",
        project_id: str | None = None,
        project_name: str | None = None,
        id: str | None = None,
    ) -> None:
        cache_key = prompt_cache._create_cache_key(project_id, project_name, slug, version, id)
        self.memory_cache.set(cache_key, value)
        if self.disk_cache:
            self.disk_cache.set(cache_key, value)
