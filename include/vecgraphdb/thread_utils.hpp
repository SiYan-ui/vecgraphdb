#pragma once

// MinGW 8.1.0 has a known bug with <mutex> (win32-seh thread model)
// Thread safety is disabled for now. See README.md for details.
// To enable thread safety, upgrade to MinGW 9+ or use MSVC/Clang.

namespace vecgraphdb {

// Dummy lock type - no actual locking (single-threaded only)
struct ReaderWriterLock {
    void lock() {}
    void unlock() {}
};

// Dummy lock guard that does nothing
struct LockGuard {
    explicit LockGuard(ReaderWriterLock&) {}
    void lock() {}
    void unlock() {}
};

// Aliases for API compatibility
using ReadLock = LockGuard;
using WriteLock = LockGuard;

} // namespace vecgraphdb
