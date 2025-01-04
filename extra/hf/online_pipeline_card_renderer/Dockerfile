
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /.cache
RUN chmod 777 /.cache

EXPOSE 7860

CMD ["python", "app.py"]

