FROM python
RUN pip install pipenv

COPY ./ /app

WORKDIR app
RUN pipenv install --deploy --system
EXPOSE 8000
ENTRYPOINT ["python", "manage.py"]



