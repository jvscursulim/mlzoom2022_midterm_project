FROM python:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY . /app/

RUN pipenv install --system --deploy

EXPOSE 4242

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:4242", "predict:app" ]
