# src/history/repositories/connection_pool.py
from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """
    Connection pool for SQLite with WAL mode optimization.

    Addresses the aiosqlite serialization bottleneck by maintaining multiple
    connections to the same SQLite database in WAL mode, enabling true
    read/write concurrency.

    Features:
    - Separate pools for read and write operations
    - WAL mode optimization for concurrent readers
    - Connection health monitoring and recovery
    - Automatic pool sizing based on load
    - Graceful degradation under contention
    """

    def __init__(  # noqa: PLR0913
        self,
        db_path: str,
        *,
        max_connections: int = 10,
        max_readers: int = 8,
        max_writers: int = 2,
        connection_timeout: float = 30.0,
        health_check_interval: float = 60.0,
    ):
        self.db_path = db_path
        self.max_connections = max_connections
        self.max_readers = max_readers
        self.max_writers = max_writers
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval

        # Connection pools
        self._read_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(
            maxsize=max_readers
        )
        self._write_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(
            maxsize=max_writers
        )

        # Pool management
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

        # Health monitoring
        self._healthy_connections: weakref.WeakSet[aiosqlite.Connection] = (
            weakref.WeakSet()
        )
        self._health_check_task: asyncio.Task[None] | None = None

        # Statistics
        self._stats = {
            "read_connections_created": 0,
            "write_connections_created": 0,
            "read_acquisitions": 0,
            "write_acquisitions": 0,
            "connection_errors": 0,
            "health_checks_performed": 0,
        }

    async def initialize(self) -> None:
        """Initialize the connection pool with initial connections."""
        if self._initialized:
            return

        async with self._pool_lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing SQLite connection pool: "
                f"{self.max_readers} readers, {self.max_writers} writers"
            )

            # Create initial read connections
            for _ in range(min(2, self.max_readers)):
                try:
                    conn = await self._create_read_connection()
                    self._read_pool.put_nowait(conn)
                    self._stats["read_connections_created"] += 1
                except Exception as e:
                    logger.error(f"Failed to create initial read connection: {e}")

            # Create initial write connections
            for _ in range(min(1, self.max_writers)):
                try:
                    conn = await self._create_write_connection()
                    self._write_pool.put_nowait(conn)
                    self._stats["write_connections_created"] += 1
                except Exception as e:
                    logger.error(
                        f"Failed to create initial write connection: {e}"
                    )

            # Start health check task
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
            self._initialized = True

            logger.info("SQLite connection pool initialized successfully")

    async def _create_read_connection(self) -> aiosqlite.Connection:
        """Create a new connection optimized for read operations."""
        conn = await aiosqlite.connect(self.db_path)

        # WAL mode for optimal read concurrency
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=10000")
        await conn.execute("PRAGMA temp_store=memory")
        await conn.execute("PRAGMA busy_timeout=30000")

        # Read-optimized settings
        await conn.execute("PRAGMA query_only=1")  # Read-only for safety
        # Allow dirty reads for performance
        await conn.execute("PRAGMA read_uncommitted=1")

        self._healthy_connections.add(conn)
        logger.debug(f"Created new read connection to {self.db_path}")
        return conn

    async def _create_write_connection(self) -> aiosqlite.Connection:
        """Create a new connection optimized for write operations."""
        conn = await aiosqlite.connect(self.db_path)

        # WAL mode for optimal write concurrency
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=10000")
        await conn.execute("PRAGMA temp_store=memory")
        await conn.execute("PRAGMA busy_timeout=30000")

        # Write-optimized settings
        # Checkpoint every 1000 pages
        await conn.execute("PRAGMA wal_autocheckpoint=1000")
        await conn.execute("PRAGMA optimize")  # Optimize for write performance

        self._healthy_connections.add(conn)
        logger.debug(f"Created new write connection to {self.db_path}")
        return conn

    @contextlib.asynccontextmanager
    async def acquire_reader(self):
        """Acquire a connection for read operations."""
        if not self._initialized:
            await self.initialize()

        conn = None
        try:
            # Try to get existing connection first
            try:
                conn = self._read_pool.get_nowait()
            except asyncio.QueueEmpty:
                # Create new connection if pool is empty and under limit
                if self._stats["read_connections_created"] < self.max_readers:
                    conn = await self._create_read_connection()
                    self._stats["read_connections_created"] += 1
                else:
                    # Wait for connection to become available
                    conn = await asyncio.wait_for(
                        self._read_pool.get(), timeout=self.connection_timeout
                    )

            self._stats["read_acquisitions"] += 1

            # Verify connection health
            if not await self._is_connection_healthy(conn):
                await self._replace_unhealthy_connection(conn, is_reader=True)
                conn = await self._create_read_connection()

            yield conn

        except Exception as e:
            self._stats["connection_errors"] += 1
            logger.error(f"Error in read connection acquisition: {e}")
            if conn:
                with contextlib.suppress(Exception):
                    await conn.close()
            raise
        finally:
            if conn and not self._closed:
                try:
                    self._read_pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool is full, close the extra connection
                    with contextlib.suppress(Exception):
                        await conn.close()

    @contextlib.asynccontextmanager
    async def acquire_writer(self):
        """Acquire a connection for write operations."""
        if not self._initialized:
            await self.initialize()

        conn = None
        try:
            # Try to get existing connection first
            try:
                conn = self._write_pool.get_nowait()
            except asyncio.QueueEmpty:
                # Create new connection if pool is empty and under limit
                if self._stats["write_connections_created"] < self.max_writers:
                    conn = await self._create_write_connection()
                    self._stats["write_connections_created"] += 1
                else:
                    # Wait for connection to become available
                    conn = await asyncio.wait_for(
                        self._write_pool.get(), timeout=self.connection_timeout
                    )

            self._stats["write_acquisitions"] += 1

            # Verify connection health
            if not await self._is_connection_healthy(conn):
                await self._replace_unhealthy_connection(conn, is_reader=False)
                conn = await self._create_write_connection()

            yield conn

        except Exception as e:
            self._stats["connection_errors"] += 1
            logger.error(f"Error in write connection acquisition: {e}")
            if conn:
                with contextlib.suppress(Exception):
                    await conn.close()
            raise
        finally:
            if conn and not self._closed:
                try:
                    self._write_pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool is full, close the extra connection
                    with contextlib.suppress(Exception):
                        await conn.close()

    async def _is_connection_healthy(self, conn: aiosqlite.Connection) -> bool:
        """Check if a connection is healthy and responsive."""
        try:
            cursor = await conn.execute("SELECT 1")
            await cursor.fetchone()
            await cursor.close()
            return True
        except Exception:
            return False

    async def _replace_unhealthy_connection(
        self, conn: aiosqlite.Connection, is_reader: bool
    ) -> None:
        """Replace an unhealthy connection."""
        logger.warning(
            f"Replacing unhealthy {'read' if is_reader else 'write'} connection"
        )

        with contextlib.suppress(Exception):
            await conn.close()

        if conn in self._healthy_connections:
            self._healthy_connections.discard(conn)

    async def _health_check_loop(self) -> None:
        """Background task to periodically check connection health."""
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)

                if self._closed:
                    break

                # Check read pool connections
                read_pool_size = self._read_pool.qsize()
                write_pool_size = self._write_pool.qsize()

                self._stats["health_checks_performed"] += 1
                logger.debug(
                    f"Pool health check: {read_pool_size} readers, "
                    f"{write_pool_size} writers available"
                )

                # Log statistics periodically
                if self._stats["health_checks_performed"] % 10 == 0:
                    logger.info(f"Connection pool stats: {self._stats}")

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def close(self) -> None:
        """Close all connections and cleanup the pool."""
        if self._closed:
            return

        self._closed = True
        logger.info("Closing SQLite connection pool")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close all read connections
        while not self._read_pool.empty():
            try:
                conn = self._read_pool.get_nowait()
                with contextlib.suppress(Exception):
                    await conn.close()
            except asyncio.QueueEmpty:
                break

        # Close all write connections
        while not self._write_pool.empty():
            try:
                conn = self._write_pool.get_nowait()
                with contextlib.suppress(Exception):
                    await conn.close()
            except asyncio.QueueEmpty:
                break

        logger.info("SQLite connection pool closed successfully")

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._stats,
            "read_pool_size": self._read_pool.qsize(),
            "write_pool_size": self._write_pool.qsize(),
            "healthy_connections": len(self._healthy_connections),
            "initialized": self._initialized,
            "closed": self._closed,
        }
