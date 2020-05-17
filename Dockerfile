FROM python:3.8



# Layer for installing python dependencies
RUN pip install pipenv
COPY Pipfile* /
RUN pipenv lock --requirements > requirements.txt
WORKDIR /
RUN pip install -r requirements.txt
#COPY requirements.txt /requirements.txt

#COPY Pipefile /Pipefile
#COPY Pipefile.lock /Pipefile.lock

# Layer for adding the application files
COPY ./ /app
WORKDIR app

# Expose port and start the app
EXPOSE 8000
ENTRYPOINT ["python", "manage.py"]
