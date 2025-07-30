
import os
import logging
import asyncio
import requests

from typing import List, Optional
from datetime import datetime, date

from sqlalchemy import create_engine, select, \
    and_, desc, event
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, \
    AsyncSession

from .model import Base, Execution, Task


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
        entity_id = new_entity.id
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

    async def get_executions(
        self,
        pipeline_name: Optional[str] = None,
        username: Optional[str] = None,
        before_datetime: Optional[datetime] = None,
        n: Optional[int] = None,
        descending: Optional[bool] = False
    ) -> List[Execution]:
        """Lists Execution records from a given start time.

        Params:
            - pipeline_name (str):
                the only retraining pipeline to consider
                (if mentioned)
            - username (str):
                the user having lunched the executions
                to consider (if mentioned)
            - before_datetime (datetime):
                UTC time from which to start listing
            - n (int):
                number of Executions to retrieve
            - descending (bool):
                sorting order, wheter latest comes first
                or last

        Results:
            List[Execution]
        """
        statement = select(Execution)

        filters = []
        if pipeline_name is not None:
            filters.append(Execution.name == pipeline_name)
        if username is not None:
            filters.append(Execution.username == username)
        if before_datetime is not None:
            filters.append(Execution._start_timestamp <= before_datetime)
        print(f"before_datetime : {before_datetime}")

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
            return result.scalars().all()

    async def get_task(self, id: int) -> Task:
        return await self._get_entity(Task, id=id)

    async def get_tasks_by_execution(self, exec_id: int) -> List[Task]:
        return await self._get_entities(Task, exec_id=exec_id)


    async def add_task(self, exec_id: int) -> int:
        return await self._add_entity(Task, exec_id=exec_id)


#////////////////////////////////////////////////////////////////////////////


new_connection_api_endpoint = \
    f"{os.environ['RP_WEB_SERVER_URL']}/api/v1/new_execution_event"

@event.listens_for(Execution, "after_insert")
def after_insert_listener(mapper, connection, target):
    """DAG-engine notifies WebConsole server."""
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

