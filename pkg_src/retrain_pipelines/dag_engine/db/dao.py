
import os
import logging
import asyncio
import requests

from uuid import UUID
from typing import List, Optional
from datetime import datetime, date

from sqlalchemy import create_engine, Uuid, \
    select, and_, desc, case, func, event
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, \
    AsyncSession

from .model import Base, Execution, ExecutionExt, \
    TaskType, Task

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
                        logging.getLogger().debug(
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

    async def get_execution_tasktypes_list(
        self, execution_id: int,
        serializable: bool = False
    ) -> Optional[List[dict]]:
        """
        Return all TaskType rows for the given execution_id
        as list of dicts.

        Params:
            - execution_id (int):
                The Execution.id to filter TaskTypes on.
            - serializable (bool):
                If True, UUIDs are converted to strings
                for JSON-safe output.
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

            # Convert ORM objects to pure dictionaries
            out_list = []
            for row in rows:
                row_dict = {}
                for col in TaskType.__table__.columns:
                    val = getattr(row, col.name)
                    if serializable and isinstance(val, UUID):
                        val = str(val)
                    row_dict[col.name] = val
                out_list.append(row_dict)

            return out_list


#////////////////////////////////////////////////////////////////////////////


new_connection_api_endpoint = \
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
        requests.post(new_connection_api_endpoint, json=data_snapshot)
    except requests.exceptions.ConnectionError as ce:
        logger.info(
            "WebConsole apparently not running " +
            f"({os.environ['RP_WEB_SERVER_URL']})"
        )
    except Exception as ex:
        logger.warn(ex)


connection_ended_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/execution_end_event"


@event.listens_for(Execution._end_timestamp, "set", retval=False)
def after_end_timestamp_change(target, newValue, oldvalue, initiator):
    """DAG-engine notifies WebConsole server.

    This fires when Execution's _end_timestamp changes.
    Emits an ExecutionEnd dict.
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
        print(f"ExecutionExt - data_snapshot: {data_snapshot}")

        try:
            requests.post(connection_ended_api_endpoint, json=data_snapshot)
        except requests.exceptions.ConnectionError as ce:
            logger.info(
                "WebConsole apparently not running " +
                f"({os.environ['RP_WEB_SERVER_URL']})"
            )
        except Exception as ex:
            logger.warn(ex)

