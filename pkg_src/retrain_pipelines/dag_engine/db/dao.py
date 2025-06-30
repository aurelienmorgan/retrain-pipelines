from .model import Base, Execution, Task
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

class DAO:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    def add_execution(self) -> int:
        session = self.Session()
        new_execution = Execution()
        session.add(new_execution)
        session.flush()  # Ensure the execution is inserted into the database
        execution_id = new_execution.id
        session.commit()
        session.close()  # Not strictly necessary with scoped_session, but safe to keep
        return execution_id

    def add_task(self, exec_id) -> int:
        session = self.Session()
        new_task = Task(exec_id=exec_id)
        session.add(new_task)
        session.flush()  # Ensure the execution is inserted into the database
        task_id = new_task.id
        session.commit()
        session.close()  # Not strictly necessary with scoped_session, but safe to keep
        return task_id

    def get_execution(self, execution_id):
        session = self.Session()
        execution = session.query(Execution).filter_by(id=execution_id).first()
        session.close()
        return execution

    def get_task(self, task_id):
        session = self.Session()
        task = session.query(Task).filter_by(id=task_id).first()
        session.close()
        return task

    def get_tasks_by_execution(self, exec_id):
        session = self.Session()
        tasks = session.query(Task).filter_by(exec_id=exec_id).all()
        session.close()
        return tasks

