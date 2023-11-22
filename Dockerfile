FROM python:3.10.9

WORKDIR .

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./app ./app
COPY ./main.py ./main.py

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.application:create_app"]

CMD ["--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
