
import asyncio

from typing import List
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, \
    AsyncSession

from .model import Base, Execution, Task


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
        return self.session_factory() if self.is_async else self.Session()

    def _add_entity(self, entity_class, **kwargs):
        if self.is_async:
            return self._async_add_entity(entity_class, **kwargs)
        else:
            return self._sync_add_entity(entity_class, **kwargs)

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

    def add_execution(self) -> int:
        return self._add_entity(Execution)

    def add_task(self, exec_id) -> int:
        return self._add_entity(Task, exec_id=exec_id)

    def get_execution(self, execution_id):
        return self._get_entity(Execution, id=execution_id)

    def get_task(self, task_id):
        return self._get_entity(Task, id=task_id)

    def get_tasks_by_execution(self, exec_id):
        return self._get_entities(Task, exec_id=exec_id)


class AsyncDAO(DAOBase):
    def __init__(self, db_url):
        super().__init__(db_url, is_async=True)

    async def add_execution(self) -> int:
        return await self._add_entity(Execution)

    async def add_task(self, exec_id: int) -> int:
        return await self._add_entity(Task, exec_id=exec_id)

    async def get_execution(self, execution_id: int):
        return await self._get_entity(Execution, id=execution_id)

    async def get_executions_before(
        self,
        before_datetime: datetime,
        n: int
    ) -> List[Execution]:
        """Lists Execution records from a given start time.

        Params:
            - before_datetime (datetime):
                time from which to start listing
            - n (int):
                number of Executions to retrieve

        Results:
            List[Executuion]
        """
        ## TODO ## implement datetime in object model + LIMIT filtering
        return await self._get_entities(Execution)

    async def get_task(self, task_id: int):
        return await self._get_entity(Task, id=task_id)

    async def get_tasks_by_execution(self, exec_id: int):
        return await self._get_entities(Task, exec_id=exec_id)

