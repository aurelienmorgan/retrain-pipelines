import json
import logging
import os
import random
import time
from datetime import date, datetime
from functools import lru_cache
from uuid import UUID

import grpc
import requests
from google.protobuf.timestamp_pb2 import Timestamp
from sqlalchemy import QueuePool, Uuid, and_, case, create_engine, desc, event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import aliased, object_session, scoped_session, sessionmaker

from ..grpc_client import GrpcClient
from .grpc import task_trace_pb2
from .model import (
    Base,
    Execution,
    ExecutionExt,
    Task,
    TaskContextAttr,
    TaskExt,
    TaskGroup,
    TaskTrace,
    TaskType,
)

logger = logging.getLogger(__name__)


def _truncate_to_millis(dt: datetime) -> datetime:
    """Truncate datetime to millisecond precision."""
    if dt is None or not isinstance(dt, datetime):
        return dt
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


class DAOBase:
    def __init__(self, db_url, is_async=False):
        if is_async:
            self.engine = create_async_engine(db_url, future=True)
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )
        else:
            from sqlalchemy.pool import NullPool

            if "sqlite" in db_url:
                # For SQLite with multiprocessing,
                # we use NullPool
                self.engine = create_engine(
                    db_url,
                    poolclass=NullPool,
                    connect_args={"timeout": 30, "check_same_thread": False},
                )

                # Enable SQLite optimizations for concurrency
                @event.listens_for(self.engine, "connect")
                def set_sqlite_pragma(dbapi_conn, connection_record):
                    cursor = dbapi_conn.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA busy_timeout=60000")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=-64000")
                    cursor.close()

            else:
                self.engine = create_engine(
                    db_url, poolclass=QueuePool, pool_size=5, max_overflow=10, pool_timeout=30
                )

            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(self.session_factory)

        self.is_async = is_async
        self.base_delay = 0.1  # seconds
        self.max_delay = 1.0

    def dispose(self):
        """Release all pooled connections held by this DAO instance."""
        try:
            if hasattr(self, "Session"):
                self.Session.remove()
        except Exception:
            pass
        try:
            self.engine.dispose()
        except Exception:
            pass

    def _get_session(self):
        return self.session_factory() if self.is_async else self.Session()

    def _add_entity(self, entity_class, **kwargs):
        # truncate any datetime fields before insert
        for k, v in kwargs.items():
            if isinstance(v, datetime):
                kwargs[k] = _truncate_to_millis(v)

        if self.is_async:
            return self._async_add_entity(entity_class, **kwargs)
        else:
            for attempt in range(5):
                try:
                    added_entity = self._sync_add_entity(entity_class, **kwargs)
                    if attempt > 0:
                        logger.warning(
                            f"[blink]_sync_add_entity - {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + "\nSUCCEEDED[/]"
                        )
                    return added_entity
                except Exception as e:
                    # database lock
                    if hasattr(self, "Session"):
                        self.Session.remove()
                    if attempt < 4:
                        # Exponential backoff with jitter
                        base_delay = self.base_delay * (2**attempt)
                        jitter = random.uniform(0, base_delay * 0.3)
                        delay = min(self.max_delay, base_delay + jitter)

                        logger.warning(
                            f"[blink]_sync_add_entity - retry {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + f"\ndelay={delay:.2f}s: {e}[/]"
                        )
                        time.sleep(delay)
                    else:
                        # Re-raise the exception
                        # on the last attempt
                        raise

    def _sync_add_entity(self, entity_class, **kwargs):
        session = self._get_session()
        try:
            # CRITICAL: Acquire immediate write lock for SQLite
            if "sqlite" in str(self.engine.url):
                session.execute(text("BEGIN IMMEDIATE"))

            new_entity = entity_class(**kwargs)
            session.add(new_entity)
            session.flush()
            entity_id = getattr(new_entity, "id", None) or getattr(new_entity, "uuid", None)
            session.commit()
            return entity_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _async_add_entity(self, entity_class, **kwargs):
        async with self._get_session() as session:
            new_entity = entity_class(**kwargs)
            session.add(new_entity)
            await session.flush()
            entity_id = new_entity.id
            await session.commit()
            return entity_id

    def _batch_add_entities(self, entity_class, **kwargs):
        # truncate any datetime fields before insert
        for item in list(kwargs.values())[0]:
            for k, v in item.items():
                if isinstance(v, datetime):
                    item[k] = _truncate_to_millis(v)

        if self.is_async:
            return self._async_batch_add_entities(entity_class, **kwargs)
        else:
            for attempt in range(5):
                try:
                    batch_added_entity = self._sync_batch_add_entities(entity_class, **kwargs)
                    if attempt > 0:
                        logger.warning(
                            "[blink]_sync_batch_add_entities - " + f"retry {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + "\nSUCCEEDED[/]"
                        )
                    return batch_added_entity
                except Exception as e:
                    # database lock
                    if hasattr(self, "Session"):
                        self.Session.remove()
                    if attempt < 4:
                        delay = min(self.max_delay, self.base_delay * (2**attempt))
                        logger.warning(
                            "[blink]_sync_batch_add_entities - " + f"retry {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + f"\ndelay={delay:.2f}s: {e}[/]"
                        )
                        time.sleep(delay)
                    else:
                        # Re-raise the exception
                        # on the last attempt
                        raise

    def _sync_batch_add_entities(self, entity_class, **kwargs):
        session = self._get_session()
        try:
            # CRITICAL: Acquire immediate write lock for SQLite
            if "sqlite" in str(self.engine.url):
                session.execute(text("BEGIN IMMEDIATE"))

            session.add_all([entity_class(**item) for item in list(kwargs.values())[0]])
            session.flush()
            session.expire_all()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _async_batch_add_entities(self, entity_class, **kwargs):
        async with self._get_session() as session:
            session.add_all([entity_class(**item) for item in list(kwargs.values())[0]])
            await session.flush()
            await session.commit()

    def _update_entity(self, entity_class, entity_id, **kwargs):
        # truncate any datetime fields before update
        for k, v in kwargs.items():
            if isinstance(v, datetime):
                kwargs[k] = _truncate_to_millis(v)

        if self.is_async:
            return self._async_update_entity(entity_class, entity_id, **kwargs)
        else:
            for attempt in range(5):
                try:
                    updated_entity = self._sync_update_entity(entity_class, entity_id, **kwargs)
                    if attempt > 0:
                        logger.warning(
                            "[blink]_sync_update_entity - " + f"retry {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + "\nSUCCEEDED[/]"
                        )
                    return updated_entity
                except Exception as e:
                    # database lock
                    if hasattr(self, "Session"):
                        self.Session.remove()
                    if attempt < 4:
                        delay = min(self.max_delay, self.base_delay * (2**attempt))
                        logger.warning(
                            "[blink]_sync_update_entity - retry " + f"retry {attempt + 1} - "
                            f"{entity_class} [{kwargs}]" + f"\ndelay={delay:.2f}s: {e}[/]"
                        )
                        time.sleep(delay)
                    else:
                        # Re-raise the exception
                        # on the last attempt
                        raise

    def _sync_update_entity(self, entity_class, entity_id, **kwargs):
        session = self._get_session()
        try:
            # CRITICAL: Acquire immediate write lock for SQLite
            if "sqlite" in str(self.engine.url):
                session.execute(text("BEGIN IMMEDIATE"))

            entity = session.get(entity_class, entity_id)
            if not entity:
                session.close()
                return None
            for key, value in kwargs.items():
                setattr(entity, key, value)
            session.commit()
            return entity
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _async_update_entity(self, entity_class, entity_id, **kwargs):
        async with self._get_session() as session:
            result = await session.execute(select(entity_class).filter_by(id=entity_id))
            entity = result.scalar_one_or_none()
            if not entity:
                return None
            for key, value in kwargs.items():
                setattr(entity, key, value)
            await session.commit()
            return entity

    def _get_entity(self, entity_class, **filters):
        if self.is_async:
            return self._async_get_entity(entity_class, **filters)
        else:
            return self._sync_get_entity(entity_class, **filters)

    def _sync_get_entity(self, entity_class, **filters):
        session = self._get_session()
        result = session.query(entity_class).filter_by(**filters).first()
        session.close()
        return result

    async def _async_get_entity(self, entity_class, **filters):
        async with self._get_session() as session:
            result = await session.execute(select(entity_class).filter_by(**filters))
            return result.scalar_one_or_none()

    def _get_entities(self, entity_class, **filters):
        if self.is_async:
            return self._async_get_entities(entity_class, **filters)
        else:
            return self._sync_get_entities(entity_class, **filters)

    def _sync_get_entities(self, entity_class, **filters):
        session = self._get_session()
        result = session.query(entity_class).filter_by(**filters).all()
        session.close()
        return result

    async def _async_get_entities(self, entity_class, **filters):
        async with self._get_session() as session:
            result = await session.execute(select(entity_class).filter_by(**filters))
            return result.scalars().all()


class DAO(DAOBase):
    def __init__(self, db_url):
        super().__init__(db_url, is_async=False)

    def add_execution(self, **kwargs) -> int:
        return self._add_entity(Execution, **kwargs)

    def add_tasktype(self, **kwargs) -> Uuid:
        return self._add_entity(TaskType, **kwargs)

    def add_task(self, **kwargs) -> int:
        return self._add_entity(Task, **kwargs)

    def add_taskgroup(self, **kwargs) -> Uuid:
        return self._add_entity(TaskGroup, **kwargs)

    def update_execution(self, id, **kwargs) -> Execution:
        """Update execution row’s fields by its id."""
        return self._update_entity(Execution, entity_id=id, **kwargs)

    def update_task(self, id, **kwargs) -> Task:
        """Update task row’s fields by its id."""
        return self._update_entity(Task, entity_id=id, **kwargs)

    def get_executions(self, exec_name) -> list[Execution]:
        return self._get_entities(Execution, name=exec_name)

    def get_execution(self, id) -> Execution:
        return self._get_entity(Execution, id=id)

    def get_task(self, id) -> Task:
        return self._get_entity(Task, id=id)

    def get_tasks_by_execution(self, exec_id) -> list[Task]:
        return self._get_entities(Task, exec_id=exec_id)

    def add_task_trace(self, **kwargs):
        return self._add_entity(TaskTrace, **kwargs)

    def batch_add_task_traces(self, **kwargs):
        return self._batch_add_entities(TaskTrace, **kwargs)

    def add_task_context_attrs(self, rows: list[dict]) -> None:
        """Bulk-insert task_context_attrs rows for a single task exit.

        Parameters
        ----------
        rows : list[dict]
            Each dict must contain: task_id, attr_name, sha, disk_ref, inline_val.
            One row per surviving or deleted context attribute.
        """
        if rows:
            self._batch_add_entities(TaskContextAttr, items=rows)


class AsyncDAO(DAOBase):
    def __init__(self, db_url):
        super().__init__(db_url, is_async=True)

    async def add_execution(self) -> int:
        return await self._add_entity(Execution)

    async def get_execution(self, id: int) -> Execution:
        return await self._get_entity(Execution, id=id)

    async def get_execution_ext(self, id: int) -> ExecutionExt | None:
        """Return a single ExecutionExt by id, including computed success status.

        Parameters
        ----------
        id : int
            Execution id.

        Returns
        -------
        ExecutionExt, optional
            None if not found.
        """
        failed_subquery = (
            select(func.max(case((Task.failed.is_(True), 1), else_=0)))
            .where(Task.exec_id == Execution.id)
            .scalar_subquery()
        )
        statement = select(Execution, failed_subquery.label("has_failed_task")).where(
            Execution.id == id
        )
        async with self._get_session() as session:
            result = await session.execute(statement)
            row = result.first()
            if row is None:
                return None
            execution, has_failed = row
            execution_ext = ExecutionExt(**execution.__dict__)
            execution_ext.success = not bool(has_failed)
            return execution_ext

    async def get_executions_count(
        self, pipeline_name: str, execs_status: str | None = None
    ) -> int:
        """Count executions filtered by pipeline_name and optionally status.

        Parameters
        ----------
        pipeline_name : str
            the pipeline name to consider.
        execs_status : str
            any None/success/failure.

        Returns
        -------
        int
            count of matching executions.

        Notes
        -----
        no cache, executions count is a living var.
        """
        filters = [Execution.name == pipeline_name]

        if execs_status:
            failed_subquery = (
                select(func.max(case((Task.failed.is_(True), 1), else_=0)))
                .where(Task.exec_id == Execution.id)
                .scalar_subquery()
            )

            if execs_status == "success":
                filters.append(failed_subquery == 0)
            elif execs_status == "failure":
                filters.append(failed_subquery == 1)
            # else: invalid status ignored

        statement = select(func.count(Execution.id)).where(and_(*filters))

        async with self._get_session() as session:
            result = await session.execute(statement)
            return result.scalar()

    async def get_distinct_execution_names(self, sorted=False) -> list[str]:
        statement = select(Execution.name).distinct()
        if sorted:
            statement = statement.order_by(Execution.name)

        async with self._get_session() as session:
            result = await session.execute(statement)
            return [row[0] for row in result.all()]

    async def get_distinct_execution_usernames(self, sorted=False) -> list[str]:
        statement = select(Execution.username).distinct()
        if sorted:
            statement = statement.order_by(Execution.username)

        async with self._get_session() as session:
            result = await session.execute(statement)
            return [row[0] for row in result.all()]

    async def get_executions_ext(
        self,
        pipeline_name: str | None = None,
        username: str | None = None,
        before_datetime: datetime | None = None,
        execs_status: str | None = None,
        n: int | None = None,
        descending: bool | None = False,
    ) -> list[ExecutionExt]:
        """List Execution records from a given start time.

        extended to include 'failed y/n' status.

        Parameters
        ----------
        pipeline_name : str
            the only retraining pipeline to consider
            (if mentioned)
        username : str
            the user having lunched the executions
            to consider (if mentioned)
        before_datetime : datetime
            UTC time from which to start listing
        execs_status : str
            any (None)/success/failure
        n : int
            number of Executions to retrieve
        descending : bool
            sorting order, wheter latest comes first
            or last

        Returns
        -------
        list[ExecutionExt]
        """
        # Subquery to check if execution has any failed tasks
        failed_subquery = (
            select(func.max(case((Task.failed.is_(True), 1), else_=0)))
            .where(Task.exec_id == Execution.id)
            .scalar_subquery()
        )
        statement = select(Execution, failed_subquery.label("has_failed_task"))

        filters = []
        if pipeline_name is not None:
            filters.append(Execution.name == pipeline_name)
        if username is not None:
            filters.append(Execution.username == username)
        if before_datetime is not None:
            filters.append(Execution._start_timestamp <= before_datetime)
        if execs_status is not None:
            if execs_status == "success":
                filters.append(and_(failed_subquery == 0, Execution._end_timestamp.is_not(None)))
            elif execs_status == "failure":
                filters.append(failed_subquery == 1)

        if filters:
            statement = statement.where(and_(*filters))

        if n is not None:
            statement = statement.limit(n)

        if descending:
            statement = statement.order_by(desc(Execution._start_timestamp))
        else:
            statement = statement.order_by(Execution._start_timestamp)

        async with self._get_session() as session:
            result = await session.execute(statement)
            executions_ext = []

            for execution, has_failed in result.all():
                execution_ext = ExecutionExt(**execution.__dict__)
                execution_ext.success = not bool(has_failed)
                executions_ext.append(execution_ext)

            return executions_ext

    async def get_execution_tasks_list(self, execution_id: int) -> list[TaskExt] | None:
        """
        Return all Task records for the given execution_id as list.

        Note: no cache, tasks are living vars.

        Parameters
        ----------
        execution_id : int
            The Execution.id to filter TaskType items on.

        Returns
        -------
        list, optional
            list of tasks if found, else None.
        """
        tasktype_subq = (
            select(
                TaskType.uuid,
                TaskType.order,
                TaskType.name,
                TaskType.ui_css,
                TaskType.is_parallel,
                TaskType.merge_func,
                TaskType.taskgroup_uuid,
            )
            .where(TaskType.exec_id == execution_id)
            .subquery()
        )

        main_stmt = (
            select(
                tasktype_subq.c.name,
                tasktype_subq.c.ui_css,
                tasktype_subq.c.is_parallel,
                tasktype_subq.c.merge_func,
                tasktype_subq.c.taskgroup_uuid,
                Task,
            )
            .where(and_(Task.tasktype_uuid == tasktype_subq.c.uuid, Task.exec_id == execution_id))
            .order_by(Task._start_timestamp, Task.id)
        )

        async with self._get_session() as session:
            result = await session.execute(main_stmt)
            rows = result.all()

            if not rows:
                return None

            # Convert ORM objects to model objects
            out_list = []
            for name, ui_css, is_parallel, merge_func, taskgroup_uuid, task_orm_obj in rows:
                task_ext = TaskExt(**{
                    **task_orm_obj.__dict__,
                    "name": name,
                    "ui_css": ui_css,
                    "is_parallel": is_parallel,
                    "merge_func": merge_func["name"] if merge_func else None,
                    "taskgroup_uuid": taskgroup_uuid,
                })
                out_list.append(task_ext)

            return out_list

    @lru_cache
    async def get_execution_tasktypes_list(self, execution_id: int) -> list[TaskType] | None:
        """
        Return all TaskType rows for the given execution_id as list.

        Parameters
        ----------
        execution_id : int
            The Execution.id to filter TaskType items on.

        Returns
        -------
        list, optional
            list of tasktypes if found, else None.
        """
        statement = (
            select(TaskType).where(TaskType.exec_id == execution_id).order_by(TaskType.order)
        )

        async with self._get_session() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()

            if not rows:
                return None

            # Convert ORM objects to model object
            out_list = []
            for tasktype_orm_obj in rows:
                out_list.append(TaskType(**tasktype_orm_obj.__dict__))

            return out_list

    @lru_cache
    async def get_execution_taskgroups_list(self, execution_id: int) -> list[TaskGroup] | None:
        """
        Return all TaskGroup records for the given execution_id as list.

        Parameters
        ----------
        execution_id : int
            The Execution.id to filter TaskGroup items on.

        Returns
        -------
        list, optional
            list of taskgroups if found, else None.
        """
        statement = (
            select(TaskGroup).where(TaskGroup.exec_id == execution_id).order_by(TaskGroup.order)
        )

        async with self._get_session() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()

            if not rows:
                return None

            # Convert ORM objects to model object
            out_list = []
            for taskgroup_orm_obj in rows:
                out_list.append(TaskGroup(**taskgroup_orm_obj.__dict__))

            return out_list

    async def get_execution_tasks_with_name(
        self, execution_id: int, task_type_name: str
    ) -> list[Task] | None:
        stmt = (
            select(Task)
            .join(
                TaskType,
                and_(Task.exec_id == TaskType.exec_id, Task.tasktype_uuid == TaskType.uuid),
            )
            .where(and_(Task.exec_id == execution_id, TaskType.name == task_type_name))
        )

        async with self._get_session() as session:
            result = await session.execute(stmt)
            tasks_list = result.scalars().all()

            return tasks_list

    @lru_cache
    async def get_execution_info(self, execution_id: int) -> dict | None:
        """
        Return basic info (name and _start_timestamp) for an Execution.

        Parameters
        ----------
        execution_id : int
            the Execution.id to query.

        Returns
        -------
        dict, optional
            keys :
                - 'name' (str)
                - 'username' (str)
                - '_start_timestamp' (str) - ISO
                - 'docstring' (str)
            None if not found.
        """
        statement = select(
            Execution.name, Execution.username, Execution._start_timestamp, Execution.docstring
        ).where(Execution.id == execution_id)
        async with self._get_session() as session:
            result = await session.execute(statement)
            row = result.first()
            if row is None:
                return None
            name, username, _start_timestamp, docstring = row
            return {
                "name": name,
                "username": username,
                "start_timestamp": _start_timestamp.isoformat(),
                "docstring": docstring,
            }

    async def get_execution_number(self, execution_id: int) -> dict | None:
        """
        Return occurrences info for an given Execution name.

        Note: no cache, executions count is a living var.

        Parameters
        ----------
        execution_id : int
            the Execution.id to query for a name.

        Returns
        -------
        dict, optional
            keys :
              - execution name
              - total count of executions with the same name
              - number: count of executions with
                start_ts <= that execution's start_ts
              - completed: count executions with
                non-null _end_timestamp
              - failed: count of completed executions that
                have at least one failed Task
            None if not found.
        """
        exec_subq = (
            select(Execution.name, Execution._start_timestamp.label("start_ts"))
            .where(Execution.id == execution_id)
            .subquery()
        )

        task_alias = aliased(Task)
        # Subquery condition: completed executions have a task failed
        failed_exists = (
            select(1)
            .where(task_alias.exec_id == Execution.id)
            .where(task_alias.failed.is_(True))
            .limit(1)
            .exists()
        )

        main_stmt = (
            select(
                Execution.name,
                func.count(Execution.id).label("total_count"),
                func.count(
                    func.nullif(Execution._start_timestamp > exec_subq.c.start_ts, True)
                ).label("number"),
                func.count(func.nullif(Execution._end_timestamp.is_(None), True)).label(
                    "completed"
                ),
                func.count(
                    func.nullif((Execution._end_timestamp.is_not(None)) & failed_exists, False)
                ).label("failed"),
            )
            .where(Execution.name == exec_subq.c.name)
            .group_by(Execution.name)
        )

        async with self._get_session() as session:
            result = await session.execute(main_stmt)
            row = result.first()
            if row is None or row.total_count == 0:
                return None

            return {
                "name": row.name,
                "number": row.number,
                "count": row.total_count,
                "completed": row.completed,
                "failed": row.failed,
            }

    @lru_cache
    async def get_taskgroups_hierarchy(self, taskgroup_uuid: UUID) -> list[dict] | None:
        """Return a hierarchical list of nested taskgroups.

        Upward, starting from the one passed as argument.
        Deepest first.
        """
        recursive_sql = """
        WITH RECURSIVE parent_chain AS (
            SELECT *
            FROM taskgroups
            WHERE uuid = :start_uuid

            UNION ALL

            SELECT tg.*
            FROM taskgroups tg
            JOIN parent_chain pc ON EXISTS (
                SELECT 1
                FROM json_each(tg.elements)
                WHERE REPLACE(json_each.value, '-', '') = pc.uuid
            )
        )
        SELECT * FROM parent_chain ORDER BY "order";
        """
        async with self._get_session() as session:
            result = await session.execute(text(recursive_sql), {"start_uuid": taskgroup_uuid.hex})
            rows = result.fetchall()
            if not rows:
                logger.warning(f"TaskGroup {taskgroup_uuid} not found.")
                return None

            out = []
            for row in rows:
                mapping = row._mapping
                out.append({
                    "uuid": (
                        # bring back dashes
                        str(UUID(hex=mapping["uuid"]))
                    ),
                    "name": mapping.get("name"),
                    "ui_css": json.loads(mapping.get("ui_css")) if mapping.get("ui_css") else None,
                })

            return out

    @lru_cache
    async def get_tasktype_docstring(self, tasktype_uuid: str) -> str | None:
        statement = select(TaskType.docstring).where(TaskType.uuid == UUID(tasktype_uuid))
        async with self._get_session() as session:
            result = await session.execute(statement)
            row = result.first()
            if row is None:
                return None
            return row[0]

    async def get_task_traces(self, task_id: int) -> list[dict] | None:
        """
        Return all TaskTrace records for the given task_id.

        As list of dictionaries.

        Parameters
        ----------
        task_id : int
            The Task.id to filter TaskTrace items on.

        Returns
        -------
        list, optional
            list of trace dicts if found, else None.
        """
        statement = (
            select(TaskTrace)
            .where(TaskTrace.task_id == task_id)
            .order_by(TaskTrace.timestamp, TaskTrace.microsec, TaskTrace.microsec_idx, TaskTrace.id)
        )

        async with self._get_session() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()

            if not rows:
                return None

            # Convert ORM objects to dicts
            out_list = []
            for trace_orm_obj in rows:
                out_list.append({
                    "id": trace_orm_obj.id,
                    "timestamp": trace_orm_obj.timestamp,
                    "microsec": trace_orm_obj.microsec,
                    "microsec_idx": trace_orm_obj.microsec_idx,
                    "content": trace_orm_obj.content,
                    "is_err": trace_orm_obj.is_err,
                })

            return out_list

    async def get_task_context_attrs(self, task_id: int) -> list[TaskContextAttr]:
        """Return all task_context_attrs rows for the given task_id.

        A single query fully reconstructs the task's exit context in O(1).

        Parameters
        ----------
        task_id : int
            The Task.id to fetch context attrs for.

        Returns
        -------
        list[TaskContextAttr]
            All rows for this task (surviving and deleted attrs).
            Empty list if none found.
        """
        statement = select(TaskContextAttr).where(TaskContextAttr.task_id == task_id)
        async with self._get_session() as session:
            result = await session.execute(statement)
            return list(result.scalars().all())

    async def get_task_ext(self, task_id: int) -> TaskExt | None:
        """Return a TaskExt (Task + TaskType.name) by task id.

        Parameters
        ----------
        task_id : int
            The Task.id to fetch.

        Returns
        -------
        TaskExt, optional
            None if not found.
        """
        stmt = (
            select(TaskType.name, Task)
            .join(
                TaskType,
                and_(Task.exec_id == TaskType.exec_id, Task.tasktype_uuid == TaskType.uuid),
            )
            .where(Task.id == task_id)
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.first()
            if row is None:
                return None
            name, task_orm = row
            return TaskExt(**{**task_orm.__dict__, "name": name})


# ////////////////////////////////////////////////////////////////////////////


new_execution_api_endpoint = f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/new_execution_event"


@event.listens_for(Execution, "after_insert")
def after_insert_execution_listener(mapper, connection, target):
    """DAG-engine notifies WebConsole server.

    This fires when Execution is created.
    Emits an Execution dict.
    """
    data_snapshot = {}
    for col in target.__table__.columns:
        value = getattr(target, col.name)
        if value is None:
            data_snapshot[col.name] = None
        elif isinstance(value, (datetime, date)):
            data_snapshot[col.name] = value.isoformat()
        else:
            data_snapshot[col.name] = value

    try:
        requests.post(new_execution_api_endpoint, json=data_snapshot)
    except requests.exceptions.ConnectionError:
        logger.info("WebConsole apparently not running " + f"({os.environ['RP_WEB_SERVER_URL']})")
    except Exception as ex:
        logger.warning(ex)


execution_ended_api_endpoint = f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/execution_end_event"


@event.listens_for(Execution._end_timestamp, "set", retval=False)
def after_end_timestamp_change(target, newValue, oldvalue, initiator):
    """DAG-engine notifies WebConsole server.

    This fires when Execution's _end_timestamp changes.
    Emits an ExecutionExt dict.
    """
    if newValue != oldvalue and newValue is not None:
        # Check if any Tasks have failed=True
        failure_exists = any(task.failed for task in target.tasks)

        # Construct the ExecutionExt object
        data_snapshot = {}
        for col in target.__table__.columns:
            value = getattr(target, col.name)
            if value is None:
                data_snapshot[col.name] = None
            elif isinstance(value, (datetime, date)):
                data_snapshot[col.name] = value.isoformat()
            else:
                data_snapshot[col.name] = value
        data_snapshot["end_timestamp"] = newValue.isoformat()
        data_snapshot["success"] = not failure_exists
        # print(f"ExecutionExt - data_snapshot: {data_snapshot}")

        try:
            requests.post(execution_ended_api_endpoint, json=data_snapshot)
        except requests.exceptions.ConnectionError:
            logger.info(
                "WebConsole apparently not running " + f"({os.environ['RP_WEB_SERVER_URL']})"
            )
        except Exception as ex:
            logger.warning(ex)


# ////////////////////////////////////////////////////////////////////////////


new_task_api_endpoint = f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/new_task_event"


@event.listens_for(Task, "after_insert")
def after_insert_task_listener(mapper, connection, target):
    """Norify WebConsole server from DAG-engine.

    This fires when Task is created.
    Emits a TaskExt dict.
    """
    # --- retrieve Task fields ---
    data_snapshot = {}
    for col in target.__table__.columns:
        value = getattr(target, col.name)
        if value is None:
            data_snapshot[col.name] = None
        elif isinstance(value, (datetime, date)):
            data_snapshot[col.name] = value.isoformat()
        elif isinstance(value, UUID):
            data_snapshot[col.name] = str(value)
        else:
            data_snapshot[col.name] = value

    # logger.info(f"Task after_insert {data_snapshot}")

    # --- inject TaskType fields ---
    # The session is available via connection.engine (sync)
    try:
        Session = scoped_session(sessionmaker(bind=connection.engine))
        session = Session()
        tasktype = session.get(TaskType, (target.exec_id, target.tasktype_uuid))
        if tasktype is None:
            # TaskType row not yet visible in this session
            # (e.g. parent transaction not yet committed).
            # Emit a warning and leave the TaskExt fields absent
            # rather than crashing the insert listener.
            logger.warning(
                f"TaskType ({target.exec_id}, {target.tasktype_uuid}) "
                "not found; TaskExt fields will be absent from this event."
            )
        else:
            # Add TaskType fields for TaskExt
            data_snapshot["docstring"] = tasktype.docstring
            data_snapshot["name"] = tasktype.name
            data_snapshot["ui_css"] = tasktype.ui_css
            data_snapshot["order"] = tasktype.order
            data_snapshot["is_parallel"] = tasktype.is_parallel
            data_snapshot["merge_func"] = tasktype.merge_func
            data_snapshot["taskgroup_uuid"] = (
                str(getattr(tasktype, "taskgroup_uuid", ""))
                if getattr(tasktype, "taskgroup_uuid", None) is not None
                else None
            )
        session.close()
    except Exception:
        import traceback as _traceback

        # Do NOT use traceback.print_exc() here: it routes through
        # builtins.print which may be monkey-patched by rp_logging in the
        # worker process, causing infinite recursion if the patch was applied
        # more than once. format_exc() returns a plain string that goes
        # through the logging system instead, which is not affected.
        logger.error(_traceback.format_exc())
        logger.exception("Worker thread crashed")
    finally:
        if session is not None:
            session.close()

    # logger.info(f"TaskExt after_insert {data_snapshot}")

    try:
        requests.post(new_task_api_endpoint, json=data_snapshot)
    except requests.exceptions.ConnectionError:
        # logger.info(
        # "WebConsole apparently not running " +
        # f"({os.environ['RP_WEB_SERVER_URL']})"
        # )
        pass
    except Exception as ex:
        logger.warning(ex)


task_ended_api_endpoint = f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/task_end_event"


@event.listens_for(Task, "after_update")
def after_task_update(mapper, connection, target):
    """DAG-engine notifies WebConsole server.

    This fires when Task's _end_timestamp changes.
    Emits a TaskExt dict.
    """
    if target._end_timestamp is not None:
        # Here, both _end_timestamp and failed are up-to-date
        # --- retrieve Task fields ---
        data_snapshot = {}
        for col in target.__table__.columns:
            value = getattr(target, col.name)
            if value is None:
                data_snapshot[col.name] = None
            elif isinstance(value, (datetime, date)):
                data_snapshot[col.name] = value.isoformat()
            elif isinstance(value, UUID):
                data_snapshot[col.name] = str(value)
            else:
                data_snapshot[col.name] = value

        # logger.info(f"Task after_end_timestamp_change {data_snapshot}")

        # --- inject TaskType fields ---
        # The session is available via connection.engine (sync)
        try:
            session = object_session(target)
            if session is None:
                logger.warning("No active session for Task; skipping notification.")
                return
            tasktype = session.get(TaskType, (target.exec_id, target.tasktype_uuid))
            # Add TaskType fields for TaskExt
            data_snapshot["docstring"] = tasktype.docstring
            data_snapshot["name"] = tasktype.name
            data_snapshot["ui_css"] = tasktype.ui_css
            data_snapshot["order"] = tasktype.order
            data_snapshot["is_parallel"] = tasktype.is_parallel
            data_snapshot["merge_func"] = tasktype.merge_func
            data_snapshot["taskgroup_uuid"] = (
                str(getattr(tasktype, "taskgroup_uuid", ""))
                if getattr(tasktype, "taskgroup_uuid", None) is not None
                else None
            )
            # DO NOT close the object session we conveniently take advantage of !!
        except Exception:
            import traceback

            traceback.print_exc()
            logger.exception("Worker thread crashed")

        # logger.info(f"TaskExt after_end_timestamp_change {data_snapshot}")

        try:
            requests.post(task_ended_api_endpoint, json=data_snapshot)
        except requests.exceptions.ConnectionError:
            # logger.info(
            # "WebConsole apparently not running " +
            # f"({os.environ['RP_WEB_SERVER_URL']})"
            # )
            pass
        except Exception as ex:
            logger.warning(ex)


# ------

""" task-traces with gRPC streaming

Each trace individual-streams immediately
to WebConsole via gRPC.
"""


@event.listens_for(TaskTrace, "after_insert")
def after_insert_task_trace_listener(mapper, connection, target):
    """Notify WebConsole server via gRPC from DAG-engine.

    Each trace is sent immediately as it arrives.
    """
    if GrpcClient.initiated():
        # Convert timestamp to protobuf Timestamp
        ts = Timestamp()
        if isinstance(target.timestamp, datetime):
            ts.FromDatetime(target.timestamp)

        # Create protobuf message with proper types
        trace = task_trace_pb2.TaskTrace(
            id=target.id,
            task_id=target.task_id,
            timestamp=ts,
            microsec=target.microsec,
            microsec_idx=target.microsec_idx,
            content=target.content,
            is_err=target.is_err,
        )

        try:
            GrpcClient.stub().SendTrace(trace)
        except grpc.RpcError as e:
            logger.error(f"gRPC error: '{target.content}' - {e.code()} - {e.details()}")
        except Exception as ex:
            logger.error(f"Error sending trace: {ex}")
    else:
        # logger.info(
        # "WebConsole apparently not running " +
        # f"({os.environ['RP_WEB_SERVER_URL']})"
        # )
        pass
