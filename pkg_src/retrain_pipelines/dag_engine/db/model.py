
"""Note : We store UTC timestamps, some databases return those
right but without the tzinfo param value
so, python considers they're local time,
unless we set the tzinfo timezone info ourselves."""

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Column, \
    Integer, String, DateTime, Boolean, \
    JSON, CheckConstraint

from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Execution(Base):
    __tablename__ = 'executions'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False)
    _start_timestamp = Column('start_timestamp', DateTime(timezone=True), nullable=False)
    _end_timestamp = Column('end_timestamp', DateTime(timezone=True), nullable=True)

    tasks = relationship(
        "Task",
        back_populates="execution"
    )

    @property
    def start_timestamp(self) -> datetime:
        if self._start_timestamp is None:
            return None
        if self._start_timestamp.tzinfo is None:
            return self._start_timestamp.replace(tzinfo=timezone.utc)
        return self._start_timestamp.astimezone(timezone.utc)

    @start_timestamp.setter
    def start_timestamp(self, value: datetime):
        self._start_timestamp = value

    @property
    def end_timestamp(self) -> datetime:
        if self._end_timestamp is None:
            return None
        if self._end_timestamp.tzinfo is None:
            return self._end_timestamp.replace(tzinfo=timezone.utc)
        return self._end_timestamp.astimezone(timezone.utc)

    @end_timestamp.setter
    def end_timestamp(self, value: datetime):
        self._end_timestamp = value


class Task(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)

    exec_id = Column(Integer, ForeignKey('executions.id'))
    execution = relationship(
        "Execution",
        back_populates="tasks"
    )
    name = Column(String, nullable=False)
    rank = Column(JSON, nullable=True)  # ARRAY(Integer)

    _start_timestamp = Column('start_timestamp', DateTime(timezone=True), nullable=False)
    _end_timestamp = Column('end_timestamp', DateTime(timezone=True), nullable=True)

    failed = Column(Boolean, nullable=True)

    __table_args__ = (
        CheckConstraint(
            '(end_timestamp IS NULL AND failed IS NULL) OR '
            '(end_timestamp IS NOT NULL AND failed IS NOT NULL)',
            name='end_failed_null_constraint'
        ),
    )

    @property
    def start_timestamp(self) -> datetime:
        if self._start_timestamp is None:
            return None
        if self._start_timestamp.tzinfo is None:
            return self._start_timestamp.replace(tzinfo=timezone.utc)
        return self._start_timestamp.astimezone(timezone.utc)

    @start_timestamp.setter
    def start_timestamp(self, value: datetime):
        self._start_timestamp = value

    @property
    def end_timestamp(self) -> datetime:
        if self._end_timestamp is None:
            return None
        if self._end_timestamp.tzinfo is None:
            return self._end_timestamp.replace(tzinfo=timezone.utc)
        return self._end_timestamp.astimezone(timezone.utc)

    @end_timestamp.setter
    def end_timestamp(self, value: datetime):
        self._end_timestamp = value

