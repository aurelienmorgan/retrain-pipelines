
"""Note : We store UTC timestamps, some databases return those
right but without the tzinfo param value
so, python considers they're local time,
unless we set the tzinfo timezone info ourselves."""

from datetime import datetime, timezone, \
    date

from sqlalchemy import ForeignKey, Column, \
    Integer, String, DateTime, Boolean, \
    JSON, CheckConstraint

from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from retrain_pipelines.utils import parse_datetime


Base = declarative_base()


class Execution(Base):
    __tablename__ = 'executions'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False)
    _start_timestamp = Column('start_timestamp',
        DateTime(timezone=True), nullable=False)
    _end_timestamp = Column('end_timestamp',
        DateTime(timezone=True), nullable=True)

    tasks = relationship(
        "Task",
        back_populates="execution"
    )

    ui_css = Column(JSON, nullable=True)

    def __init__(self, *args, **kwargs):
        # Support dict as the ONLY positional argument
        data = None
        if len(args) == 1 and isinstance(args[0], dict):
            data = args[0]
        elif len(args) > 0:
            raise TypeError(
                "Execution accepts a dict or keyword arguments")

        if data:
            kwargs = {**data, **kwargs}
            kwargs["id"] = int(kwargs["id"])
            kwargs["name"] = str(kwargs["name"])
            kwargs["username"] = str(kwargs["username"])
            kwargs["_start_timestamp"] = \
                parse_datetime(kwargs["start_timestamp"])
            kwargs["_end_timestamp"] = (
                parse_datetime(kwargs.get("end_timestamp"))
                if (
                    "end_timestamp" in kwargs and
                    kwargs.get("end_timestamp") is not None
                ) else None
            )

        super().__init__(**kwargs)

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


class ExecutionExt(Execution):
    """Execution class plus failure (computed) attribute."""
    """ NOT AN SQLALCHEMY CLASS """
    __mapper_args__ = {
        'polymorphic_identity': 'execution_ext'
    }

    def __init__(self, **kwargs):
        success = kwargs.pop('success', None)
        # Remove SQLAlchemy internal attributes
        kwargs.pop('_sa_instance_state', None)

        # Add underscore prefix for attributes with names
        # that are properties (getters) in parent class
        parent_class = type(self).__bases__[0]
        for attr_name in list(kwargs.keys()):
            if isinstance(getattr(parent_class, attr_name, None), property):
                value = kwargs.pop(attr_name)
                # Check if corresponding SQLAlchemy column is DateTime
                if hasattr(parent_class, '__table__'):
                    column = parent_class.__table__.columns.get(attr_name)
                    if (
                        column is not None and
                        hasattr(column.type, 'python_type')
                    ):
                        if (
                            column.type.python_type in [datetime, date]
                            and isinstance(value, str)
                        ):
                            value = parse_datetime(value)
                kwargs[f'_{attr_name}'] = value

        super().__init__(**kwargs)
        self.success = success

    def to_dict(self):
        result = {}
        for attr_name in vars(self):
            if not attr_name.startswith('_'):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, (datetime, date)):
                    result[attr_name] = attr_value.isoformat()
                else:
                    result[attr_name] = attr_value

        parent_class = type(self).__bases__[0]
        # Use parent_class table info
        if hasattr(parent_class, '__table__'):
            table = parent_class.__table__
            for attr_name in dir(parent_class):
                if isinstance(getattr(parent_class, attr_name, None), property):
                    private_attr = f'_{attr_name}'
                    if hasattr(self, private_attr):
                        attr_value = getattr(self, private_attr)
                        # Check if this property corresponds to a column in the table
                        column = table.columns.get(attr_name)
                        if column is not None and hasattr(column.type, 'python_type'):
                            python_type = column.type.python_type
                            if (
                                python_type in [datetime, date] and
                                isinstance(attr_value, str)
                            ):
                                attr_value = parse_datetime(attr_value)
                            if isinstance(attr_value, (datetime, date)):
                                result[attr_name] = attr_value.isoformat()
                            elif attr_value is not None:
                                result[attr_name] = attr_value
                            else:
                                result[attr_name] = None
                        else:
                            # If no column info, just assign if not None
                            if attr_value is not None:
                                result[attr_name] = attr_value
                            else:
                                result[attr_name] = None

        return result


class Task(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)

    exec_id = Column(Integer, ForeignKey('executions.id'), nullable=False)
    execution = relationship(
        "Execution",
        back_populates="tasks"
    )
    name = Column(String, nullable=False)
    rank = Column(JSON, nullable=True)  # ARRAY(Integer)

    _start_timestamp = Column('start_timestamp', DateTime(timezone=True), nullable=False)
    _end_timestamp = Column('end_timestamp', DateTime(timezone=True), nullable=True)

    failed = Column(Boolean, nullable=True)

    ui_css = Column(JSON, nullable=True)

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

