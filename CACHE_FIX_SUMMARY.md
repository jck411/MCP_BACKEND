# Token Cache Memory Fix Summary

## Problem
The token counting system in `src/history/token_counter.py` had an unbounded global cache (`_token_cache`) that grew indefinitely as the system processed diverse user inputs. This could lead to significant memory consumption over time.

## Solution
Replaced the unbounded global dictionary cache with Python's built-in `functools.lru_cache` decorator with a maximum size of 1024 entries.

## Changes Made

### Before
```python
# Global unbounded cache
_token_cache: dict[str, int] = {}

# Manual cache management in count_tokens method
def count_tokens(self, text: str) -> int:
    content_hash = self.compute_content_hash(text)
    if content_hash in _token_cache:
        return _token_cache[content_hash]
    
    tokens = len(self._encoding.encode(text))
    _token_cache[content_hash] = tokens  # Unbounded growth
    return tokens
```

### After
```python
# LRU cache with automatic eviction
@lru_cache(maxsize=1024)
def _count_tokens_cached(_content_hash: str, encoding_name: str, text: str) -> int:
    encoding = _get_encoding(encoding_name)
    return len(encoding.encode(text))

# Simplified count_tokens method
def count_tokens(self, text: str) -> int:
    content_hash = self.compute_content_hash(text)
    return _count_tokens_cached(content_hash, self.encoding_name, text)
```

## Benefits

1. **Memory Bounded**: Cache is limited to 1024 entries maximum
2. **Automatic Eviction**: LRU algorithm automatically removes oldest entries
3. **Performance Preserved**: Recent/frequent entries remain cached
4. **Better Observability**: Enhanced cache statistics with hits/misses
5. **No Breaking Changes**: All existing APIs remain unchanged

## Cache Statistics

The updated cache statistics now provide more detailed information:
- `cache_size`: Current number of cached entries
- `cache_hits`: Number of cache hits
- `cache_misses`: Number of cache misses  
- `max_cache_size`: Maximum cache size (1024)

## Testing

Comprehensive testing confirmed:
- ✅ Cache properly bounds memory at 1024 entries
- ✅ LRU eviction works correctly
- ✅ Cache hits/misses are tracked accurately
- ✅ All existing functionality preserved
- ✅ Performance benefits maintained for recent entries

## Performance Impact

- **Positive**: Memory usage is now bounded and predictable
- **Neutral**: Token counting performance remains equivalent
- **Positive**: Better cache observability for monitoring

The fix ensures the system can handle diverse workloads without unbounded memory growth while maintaining the performance benefits of caching.
