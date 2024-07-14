FROM python

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

EXPOSE 8081

COPY ./ /app

CMD ["uvicorn", "app:app","--host", "0.0.0.0", "--reload", "--port", "8081"]