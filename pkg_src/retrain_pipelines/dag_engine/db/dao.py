
import os
import json
import logging
import asyncio
import requests

from uuid import UUID
from functools import lru_cache
from typing import List, Optional
from datetime import datetime, date

from sqlalchemy import create_engine, Uuid, \
    select, and_, desc, case, func, event, \
    text
from sqlalchemy.orm import scoped_session, \
    sessionmaker, aliased, object_session
from sqlalchemy.ext.asyncio import create_async_engine, \
    AsyncSession

from .model import Base, Execution, ExecutionExt, \
    TaskType, Task, TaskExt, TaskGroup

logger = logging.getLogger()


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
            self.engine = create_engine(db_url)
            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(self.session_factory)
        self.is_async = is_async

    def _get_session(self):
        return self.session_factory() \
               if self.is_async else self.Session()


    def _add_entity(self, entity_class, **kwargs):
        if self.is_async:
            return self._async_add_entity(entity_class, **kwargs)
        else:
            for attempt in range(3):
                try:
                    return self._sync_add_entity(
                        entity_class, **kwargs
                    )
                except Exception as e:
                    # database lock
                    if attempt < 2:
                        logger.debug(
                            f"_sync_add_entity - retry {attempt+1})"
                        )
                        continue
                    else:
                        # Re-raise the exception
                        # on the last attempt
                        raise

    def _sync_add_entity(self, entity_class, **kwargs):
        session = self._get_session()
        new_entity = entity_class(**kwargs)
        session.add(new_entity)
        session.flush()
        entity_id = getattr(new_entity, "id", None) \
                    or getattr(new_entity, "uuid", None)
        session.commit()
        session.close()
        return entity_id

    async def _async_add_entity(self, entity_class, **kwargs):
        async with self._get_session() as session:
            new_entity = entity_class(**kwargs)
            session.add(new_entity)
            await session.flush()
            entity_id = new_entity.id
            await session.commit()
            return entity_id


    def _update_entity(self, entity_class, entity_id, **kwargs):
        if self.is_async:
            return self._async_update_entity(
                    entity_class, entity_id, **kwargs
                )
        else:
            for attempt in range(3):
                try:
                    return self._sync_update_entity(
                        entity_class, entity_id, **kwargs
                    )
                except Exception as e:
                    # database lock
                    if attempt < 2:
                        logger.debug(
                            f"_sync_update_entity - retry {attempt+1})"
                        )
                        continue
                    else:
                        # Re-raise the exception
                        # on the last attempt
                        raise

    def _sync_update_entity(
        self, entity_class, entity_id, **kwargs
    ):
        session = self._get_session()
        entity = session.query(entity_class).get(entity_id)
        if not entity:
            session.close()
            return None
        for key, value in kwargs.items():
            setattr(entity, key, value)
        session.commit()
        session.close()
        return entity

    async def _async_update_entity(
        self, entity_class, entity_id, **kwargs
    ):
        async with self._get_session() as session:
            result = await session.execute(
                select(entity_class).filter_by(id=entity_id)
            )
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
        result = session.query(entity_class) \
                    .filter_by(**filters).first()
        session.close()
        return result

    async def _async_get_entity(self, entity_class, **filters):
        async with self._get_session() as session:
            result = await session.execute(
                select(entity_class).filter_by(**filters)
            )
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
            result = await session.execute(
                select(entity_class).filter_by(**filters)
            )
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

    def get_execution(self, id) -> Execution:
        return self._get_entity(Execution, id=id)

    def get_task(self, id) -> Task:
        return self._get_entity(Task, id=id)

    def get_tasks_by_execution(self, exec_id) -> List[Task]:
        return self._get_entities(Task, exec_id=exec_id)


class AsyncDAO(DAOBase):
    def __init__(self, db_url):
        super().__init__(db_url, is_async=True)

    async def add_execution(self) -> int:
        return await self._add_entity(Execution)

    async def get_execution(self, id: int) -> Execution:
        return await self._get_entity(Execution, id=id)

    async def get_distinct_execution_names(
        self, sorted=False
    ) -> List[str]:
        statement = select(Execution.name).distinct()
        if sorted:
            statement = statement.order_by(Execution.name)

        async with self._get_session() as session:
            result = await session.execute(statement)
            return [row[0] for row in result.all()]

    async def get_distinct_execution_usernames(
        self, sorted=False
    ) -> List[str]:
        statement = select(Execution.username).distinct()
        if sorted:
            statement = statement.order_by(
                Execution.username)

        async with self._get_session() as session:
            result = await session.execute(statement)
            return [row[0] for row in result.all()]

    async def get_executions_ext(
        self,
        pipeline_name: Optional[str] = None,
        username: Optional[str] = None,
        before_datetime: Optional[datetime] = None,
        execs_status: Optional[str] = None,
        n: Optional[int] = None,
        descending: Optional[bool] = False
    ) -> List[ExecutionExt]:
        """Lists Execution records from a given start time.

        extended to include 'failed y/n' status.

        Params:
            - pipeline_name (str):
                the only retraining pipeline to consider
                (if mentioned)
            - username (str):
                the user having lunched the executions
                to consider (if mentioned)
            - before_datetime (datetime):
                UTC time from which to start listing
            - execs_status str):
                any (None)/success/failure
            - n (int):
                number of Executions to retrieve
            - descending (bool):
                sorting order, wheter latest comes first
                or last

        Results:
            List[ExecutionExt]
        """
        # Subquery to check if execution has any failed tasks
        failed_subquery = (
            select(func.max(case((Task.failed == True, 1), else_=0)))
            .where(Task.exec_id == Execution.id)
            .scalar_subquery()
        )
        statement = select(
            Execution,
            failed_subquery.label('has_failed_task')
        )

        filters = []
        if pipeline_name is not None:
            filters.append(Execution.name == pipeline_name)
        if username is not None:
            filters.append(Execution.username == username)
        if before_datetime is not None:
            filters.append(Execution._start_timestamp <= before_datetime)
        if execs_status is not None:
            if execs_status == "success":
                filters.append(failed_subquery == 0)
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

    async def get_execution_tasks_list(
        self, execution_id: int
    ) -> Optional[List[TaskExt]]:
        """
        Return all Task records for the given execution_id
        as list.

        Note: no cache, tasks are living vars.

        Params:
            - execution_id (int):
                The Execution.id to filter TaskType items on.

        Results:
            list of tasks if found, else None
        """
        tasktype_subq = (
            select(
                TaskType.uuid,
                TaskType.order,
                TaskType.name,
                TaskType.ui_css,
                TaskType.is_parallel,
                TaskType.merge_func,
                TaskType.taskgroup_uuid
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

                Task
            )
            .where(Task.tasktype_uuid == tasktype_subq.c.uuid)
            .order_by(Task._start_timestamp)
        )

        async with self._get_session() as session:
            result = await session.execute(main_stmt)
            rows = result.all()

            if not rows:
                return None

            # Convert ORM objects to model objects
            out_list = []
            for (
                name, ui_css, is_parallel, merge_func, taskgroup_uuid,
                task_orm_obj
            ) in rows:
                task_ext = TaskExt(**{**task_orm_obj.__dict__,
                                      "name": name, "ui_css": ui_css,
                                      "is_parallel": is_parallel,
                                      "merge_func": merge_func["name"] \
                                                    if merge_func else None,
                                      "taskgroup_uuid": taskgroup_uuid})
                out_list.append(task_ext)

            return out_list

    @lru_cache
    async def get_execution_tasktypes_list(
        self, execution_id: int
    ) -> Optional[List[TaskType]]:
        """
        Return all TaskType rows for the given execution_id
        as list.

        Params:
            - execution_id (int):
                The Execution.id to filter TaskType items on.

        Results:
            list of tasktypes if found, else None
        """
        statement = (
            select(TaskType)
            .where(TaskType.exec_id == execution_id)
            .order_by(TaskType.order)
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
    async def get_execution_taskgroups_list(
        self, execution_id: int
    ) -> Optional[List[TaskGroup]]:
        """
        Return all TaskGroup records for the given execution_id
        as list.

        Params:
            - execution_id (int):
                The Execution.id to filter TaskGroup items on.

        Results:
            list of taskgroups if found, else None
        """
        statement = (
            select(TaskGroup)
            .where(TaskGroup.exec_id == execution_id)
            .order_by(TaskGroup.order)
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

    @lru_cache
    async def get_execution_info(
        self, execution_id: int
    ) -> Optional[dict]:
        """
        Return basic info (name and _start_timestamp) for an Execution.

        Params:
            - execution_id (int):
                the Execution.id to query.

        Results:
            dict with keys 'name' and '_start_timestamp' if found,
            else None
        """
        statement = (
            select(Execution.name, Execution.username,
                   Execution._start_timestamp,
                   Execution.docstring)
            .where(Execution.id == execution_id)
        )
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
                "docstring": docstring
            }

    async def get_execution_number(
        self, execution_id: int
    ) -> Optional[dict]:
        """
        Return occurrences info for an given Execution name.

        Note: no cache, executions count is a living var.

        Params:
            - execution_id (int):
                the Execution.id to query for a name.

        Returns:
            dict if execution found else None.
            Includes:
              - execution name
              - total executions with the same name
              - number: count of executions with
                start_ts <= that execution's start_ts
              - completed: count executions with
                non-null _end_timestamp
              - failed: count of completed executions that
                have at least one failed Task
        """
        exec_subq = (
            select(
                Execution.name,
                Execution._start_timestamp.label("start_ts")
            )
            .where(Execution.id == execution_id)
            .subquery()
        )

        task_alias = aliased(Task)
        # Subquery condition: completed executions have a task failed
        failed_exists = (
            select(1)
            .where(task_alias.exec_id == Execution.id)
            .where(task_alias.failed == True)
            .limit(1)
            .exists()
        )

        main_stmt = (
            select(
                Execution.name,
                func.count(Execution.id).label("total_count"),
                func.count(
                    func.nullif(
                        Execution._start_timestamp > exec_subq.c.start_ts, True
                    )
                ).label("number"),
                func.count(
                    func.nullif(Execution._end_timestamp == None, True)
                ).label("completed"),
                func.count(
                    func.nullif(
                        (Execution._end_timestamp != None) & failed_exists,
                        False
                    )
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
    async def get_taskgroups_hierarchy(
        self, taskgroup_uuid: UUID
    ) -> Optional[List[dict]]:
        """Returns a hierarchical list of nested taskgroups.

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
            result = await session.execute(
                                text(recursive_sql),
                                {"start_uuid": taskgroup_uuid.hex})
            rows = result.fetchall()
            if not rows:
                logger.warn(f"TaskGroup {taskgroup_uuid} not found.")
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
                    "ui_css": json.loads(mapping.get("ui_css")) \
                              if mapping.get("ui_css") else None
                })

            return out


#////////////////////////////////////////////////////////////////////////////


new_execution_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/new_execution_event"

@event.listens_for(Execution, "after_insert")
def after_insert_listener(mapper, connection, target):
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
    except requests.exceptions.ConnectionError as ce:
        logger.info(
            "WebConsole apparently not running " +
            f"({os.environ['RP_WEB_SERVER_URL']})"
        )
    except Exception as ex:
        logger.warn(ex)


execution_ended_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/execution_end_event"

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
        except requests.exceptions.ConnectionError as ce:
            logger.info(
                "WebConsole apparently not running " +
                f"({os.environ['RP_WEB_SERVER_URL']})"
            )
        except Exception as ex:
            logger.warn(ex)


new_task_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/new_task_event"

@event.listens_for(Task, "after_insert")
def after_insert_listener(mapper, connection, target):
    """DAG-engine notifies WebConsole server.

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
    Session = scoped_session(sessionmaker(bind=connection.engine))
    session = Session()
    tasktype = session.get(TaskType, (target.exec_id, target.tasktype_uuid))
    # Add TaskType fields for TaskExt
    data_snapshot["docstring"] = tasktype.docstring
    data_snapshot["name"] = tasktype.name
    data_snapshot["ui_css"] = tasktype.ui_css
    data_snapshot["order"] = tasktype.order
    data_snapshot["is_parallel"] = tasktype.is_parallel
    data_snapshot["merge_func"] = tasktype.merge_func
    data_snapshot["taskgroup_uuid"] = str(getattr(tasktype, "taskgroup_uuid", "")) \
        if getattr(tasktype, "taskgroup_uuid", None) is not None else None
    session.close()

    # logger.info(f"TaskExt after_insert {data_snapshot}")

    try:
        requests.post(new_task_api_endpoint, json=data_snapshot)
    except requests.exceptions.ConnectionError as ce:
        logger.info(
            "WebConsole apparently not running " +
            f"({os.environ['RP_WEB_SERVER_URL']})"
        )
    except Exception as ex:
        logger.warn(ex)


task_ended_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/task_end_event"

@event.listens_for(Task._end_timestamp, "set", retval=False)
def after_end_timestamp_change(target, newValue, oldvalue, initiator):
    """DAG-engine notifies WebConsole server.

    This fires when Task's _end_timestamp changes.
    Emits a TaskExt dict.
    """
    if newValue != oldvalue and newValue is not None:
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
        data_snapshot["end_timestamp"] = newValue.isoformat()

        # logger.info(f"Task after_end_timestamp_change {data_snapshot}")

        # --- inject TaskType fields ---
        # The session is available via connection.engine (sync)
        session = object_session(target)
        if session is None:
            logger.warn("No active session for Task; skipping notification.")
            return
        tasktype = session.get(TaskType, (target.exec_id, target.tasktype_uuid))
        # Add TaskType fields for TaskExt
        data_snapshot["docstring"] = tasktype.docstring
        data_snapshot["name"] = tasktype.name
        data_snapshot["ui_css"] = tasktype.ui_css
        data_snapshot["order"] = tasktype.order
        data_snapshot["is_parallel"] = tasktype.is_parallel
        data_snapshot["merge_func"] = tasktype.merge_func
        data_snapshot["taskgroup_uuid"] = str(getattr(tasktype, "taskgroup_uuid", "")) \
            if getattr(tasktype, "taskgroup_uuid", None) is not None else None
        # DO NOT close the object session we conveniently take advantage of !!

        # logger.info(f"TaskExt after_end_timestamp_change {data_snapshot}")

        try:
            requests.post(task_ended_api_endpoint, json=data_snapshot)
        except requests.exceptions.ConnectionError as ce:
            logger.info(
                "WebConsole apparently not running " +
                f"({os.environ['RP_WEB_SERVER_URL']})"
            )
        except Exception as ex:
            logger.warn(ex)

