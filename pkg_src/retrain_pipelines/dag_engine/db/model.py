
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
    start_timestamp = Column(
        DateTime(timezone=False),
        nullable=False
    )
    end_timestamp = Column(
        DateTime(timezone=False),
        nullable=True
    )

    tasks = relationship(
        "Task",
        back_populates="execution"
    )


class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)

    exec_id = Column(Integer, ForeignKey('executions.id'))
    execution = relationship(
        "Execution",
        back_populates="tasks"
    )
    name = Column(String, nullable=False)
    rank = Column(JSON, nullable=True) # ARRAY(Integer)

    start_timestamp = Column(
        DateTime(timezone=False),
        nullable=False
    )
    end_timestamp = Column(
        DateTime(timezone=False),
        nullable=True
    )
    failed = Column(Boolean, nullable=True)

    __table_args__ = (
        CheckConstraint(
            '(end_timestamp IS NULL AND failed IS NULL) OR '
            '(end_timestamp IS NOT NULL AND failed IS NOT NULL)',
            name='end_failed_null_constraint'
        ),
    )

