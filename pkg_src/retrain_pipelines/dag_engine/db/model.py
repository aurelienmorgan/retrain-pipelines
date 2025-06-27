from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Execution(Base):
    __tablename__ = 'executions'
    id = Column(Integer, primary_key=True)
    tasks = relationship("Task", back_populates="execution")

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    exec_id = Column(Integer, ForeignKey('executions.id'))
    execution = relationship("Execution", back_populates="tasks")
