FROM python:3.8
RUN pip install pipenv

COPY ./ /app

WORKDIR app
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python", "manage.py"]
